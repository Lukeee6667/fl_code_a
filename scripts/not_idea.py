# ==============================
# 初始化：基于已训练完成的联邦学习模型
# ==============================
def initialize_federated_state(
    global_model: Model,  # 已通过FedAvg训练收敛的全局模型（参数θ*）
    clients: List[Client],  # 联邦系统中所有客户端列表（共n个）
    forget_requests: Dict[Client, ForgetSpec]  # 遗忘请求：key=请求客户端，value=遗忘规格（含D_u类型：客户端级/类级/实例级）
) -> Tuple[Model, List[Client], Set[Client]]:
    """
    功能：初始化联邦遗忘所需的核心状态，筛选目标客户端与保留客户端
    论文对应：第5.1节算法概述、第6.1节实验设置
    """
    # 1. 筛选目标客户端（发起遗忘请求的客户端）和保留客户端（无遗忘请求的客户端）
    target_clients = set(forget_requests.keys())  # 目标客户端集合（需遗忘D_u）
    retain_clients = [c for c in clients if c not in target_clients]  # 保留客户端集合（无D_u）
    
    # 2. 为每个目标客户端标记本地数据划分（D_u：待遗忘数据，D_r：保留数据）
    for target_client in target_clients:
        spec = forget_requests[target_client]
        if spec.type == "client-wise":  # 客户端级遗忘：遗忘该客户端全部数据
            target_client.D_u = target_client.local_data  # D_u = 本地所有数据
            target_client.D_r = empty_dataset()  # D_r = 空集（不参与微调）
        elif spec.type == "class-wise":  # 类级遗忘：遗忘指定类别的数据
            target_client.D_u = filter_data(target_client.local_data, class_id=spec.class_id)
            target_client.D_r = filter_data(target_client.local_data, exclude_class_id=spec.class_id)
        elif spec.type == "instance-wise":  # 实例级遗忘：遗忘指定ID的样本
            target_client.D_u = filter_data(target_client.local_data, instance_ids=spec.instance_ids)
            target_client.D_r = filter_data(target_client.local_data, exclude_instance_ids=spec.instance_ids)
    
    # 3. 保留客户端无需划分数据（全部本地数据为D_r，参与微调）
    for retain_client in retain_clients:
        retain_client.D_r = retain_client.local_data
        retain_client.D_u = empty_dataset()
    
    return global_model, clients, target_clients


# ==============================
# 步骤1：服务器执行层-wise权重取反（核心扰动阶段）
# ==============================
def server_weight_negation(
    global_model: Model,
    negation_layers: List[str] = ["first_conv"]  # 待取反层名称，论文推荐优先取反第一层（如CNN的第一个卷积层、ViT的卷积投影层）
) -> Model:
    """
    功能：对全局模型的指定层参数取反，生成扰动模型θ'，破坏层间协同适配
    论文对应：第5.1节公式(3)、第5.2节扰动有效性证明、第5.3节层选择逻辑
    """
    # 1. 复制全局模型参数（避免修改原模型θ*）
    perturbed_model = copy_model(global_model)
    θ_prime = perturbed_model.parameters  # 扰动模型参数θ'
    
    # 2. 遍历待取反层，执行权重取反（乘以-1）
    for layer_name in negation_layers:
        # 安全检查：确保待取反层存在且为可训练参数层（排除不可训练的BatchNorm均值等）
        if layer_name not in θ_prime or not θ_prime[layer_name].trainable:
            raise ValueError(f"Layer {layer_name} is not trainable or does not exist")
        
        # 核心操作：权重取反（论文公式(3)的实现）
        # θ'_ℓ = -θ*_ℓ（对ℓ∈mathscr{L}_neg）；θ'_ℓ = θ*_ℓ（对ℓ∉mathscr{L}_neg）
        θ_prime[layer_name].values = θ_prime[layer_name].values * (-1)
    
    # 3. 特殊处理：避免连续取反卷积层+归一化层（论文第5.3节禁忌，取反会抵消）
    conv_layers = [l for l in negation_layers if "conv" in l.lower()]
    bn_layers = [l for l in negation_layers if "batchnorm" in l.lower() or "bn" in l.lower()]
    if len(conv_layers) > 0 and len(bn_layers) > 0:
        warnings.warn("Consecutive negation of conv and BN layers may cancel out (see Sec.5.3)")
    
    return perturbed_model


# ==============================
# 步骤2：基于保留数据的联邦微调（恢复与精准遗忘阶段）
# ==============================
def federated_finetuning(
    perturbed_model: Model,  # 步骤1生成的扰动模型（参数θ'）
    clients: List[Client],
    target_clients: Set[Client],
    finetune_rounds: int = 5,  # 微调轮次，论文实验中≤10轮（远少于从头重训）
    batch_size: int = 32,  # 微调批次大小
    lr: float = 1e-4  # 微调学习率，低于初始训练（避免破坏保留知识）
) -> Model:
    """
    功能：通过少量联邦轮次微调，恢复保留数据性能，同时彻底遗忘目标数据
    论文对应：第5.1节微调逻辑、第6.2节实验结果（微调轮次设置）
    """
    current_global_model = copy_model(perturbed_model)  # 初始微调模型=扰动模型θ'
    
    for round in range(finetune_rounds):
        # ------------------------------
        # 客户端本地微调（仅用D_r训练）
        # ------------------------------
        client_updates = []  # 存储客户端的模型更新（参数差值）
        for client in clients:
            # 目标客户端规则：若为客户端级遗忘（D_r为空），不参与微调（避免引入D_u）
            if client in target_clients and client.D_r.is_empty():
                continue
            
            # 1. 客户端加载当前全局模型
            client.load_model(current_global_model)
            
            # 2. 本地微调：仅用D_r训练（损失函数为L_Dr(θ)）
            optimizer = Adam(client.model.parameters(), lr=lr)
            for epoch in range(2):  # 本地训练轮次：论文实验中设为2（避免过拟合）
                for batch in client.D_r.iter_batches(batch_size=batch_size):
                    x, y = batch
                    logits = client.model(x)
                    loss = cross_entropy_loss(logits, y)  # 仅最小化保留数据损失
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
            # 3. 计算客户端更新：本地微调后参数 - 当前全局模型参数
            client_update = compute_parameter_diff(
                client.model.parameters(), current_global_model.parameters()
            )
            client_updates.append((client_update, len(client.D_r)))  # 记录更新权重（按D_r大小加权）
        
        # ------------------------------
        # 服务器聚合更新（FedAvg逻辑）
        # ------------------------------
        if len(client_updates) == 0:
            raise ValueError("No clients participated in finetuning (check target client D_r)")
        
        # 1. 按客户端D_r大小加权平均更新（论文实验采用FedAvg聚合）
        total_data_size = sum(data_size for _, data_size in client_updates)
        aggregated_update = defaultdict(float)
        for update, data_size in client_updates:
            weight = data_size / total_data_size
            for layer_name, param_diff in update.items():
                aggregated_update[layer_name] += param_diff * weight
        
        # 2. 应用聚合更新到当前全局模型
        for layer_name, param in current_global_model.parameters.items():
            param.values += aggregated_update[layer_name]
    
    return current_global_model  # 微调后的最终遗忘模型


# ==============================
# 主函数：NoT算法完整流程
# ==============================
def NoT_federated_unlearning(
    trained_global_model: Model,
    all_clients: List[Client],
    forget_requests: Dict[Client, ForgetSpec],
    negation_layers: List[str] = ["first_conv"],
    finetune_rounds: int = 5
) -> Model:
    """
    功能：整合初始化、权重取反、联邦微调三大步骤，输出最终遗忘模型
    论文对应：第5章完整算法逻辑、第6章实验流程
    """
    # 1. 初始化联邦遗忘状态（划分D_u/D_r，筛选目标客户端）
    global_model, clients, target_clients = initialize_federated_state(
        trained_global_model, all_clients, forget_requests
    )
    print(f"Initialized: {len(target_clients)} target clients, {len(clients)-len(target_clients)} retain clients")
    
    # 2. 服务器执行层-wise权重取反（生成扰动模型θ'）
    perturbed_model = server_weight_negation(global_model, negation_layers)
    print(f"Perturbed model generated (negated layers: {negation_layers})")
    
    # 3. 基于保留数据的联邦微调（恢复性能+精准遗忘）
    final_unlearned_model = federated_finetuning(
        perturbed_model, clients, target_clients, finetune_rounds
    )
    print(f"Federated finetuning completed ({finetune_rounds} rounds)")
    
    # 4. （可选）验证遗忘效果（论文第6.2节评估指标）
    # 验证指标：Retain Accuracy（D_r精度）、Forget Accuracy（D_u精度）、MIA（成员推理攻击成功率）
    verify_unlearning_efficacy(final_unlearned_model, clients, target_clients)
    
    return final_unlearned_model


# ==============================
# 辅助函数（论文实验依赖）
# ==============================
def copy_model(model: Model) -> Model:
    """复制模型参数，避免原模型被修改"""
    return Model(parameters=deepcopy(model.parameters), architecture=model.architecture)

def compute_parameter_diff(param1: Dict[str, Tensor], param2: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """计算两个模型参数的差值（param1 - param2）"""
    diff = {}
    for layer_name in param1.keys():
        diff[layer_name] = param1[layer_name].values - param2[layer_name].values
    return diff

def verify_unlearning_efficacy(model: Model, clients: List[Client], target_clients: Set[Client]):
    """验证遗忘效果：输出Retain/Forget/MIA指标（对应论文第6.2节评估逻辑）"""
    retain_acc = []
    forget_acc = []
    mia_scores = []
    
    for client in clients:
        # 计算Retain Accuracy（D_r上的精度）
        if not client.D_r.is_empty():
            acc = model.evaluate_accuracy(client.D_r)
            retain_acc.append(acc)
        
        # 计算Forget Accuracy（D_u上的精度，目标：接近随机猜测）
        if client in target_clients and not client.D_u.is_empty():
            acc = model.evaluate_accuracy(client.D_u)
            forget_acc.append(acc)
            # 计算MIA（成员推理攻击成功率，目标：≤50%）
            mia_score = mia_attack(model, client.D_u, client.D_r)  # MIA攻击实现参考论文[26]
            mia_scores.append(mia_score)
    
    print(f"Average Retain Accuracy: {sum(retain_acc)/len(retain_acc):.2f}%")
    print(f"Average Forget Accuracy: {sum(forget_acc)/len(forget_acc):.2f}%")
    print(f"Average MIA Score: {sum(mia_scores)/len(mia_scores):.2f}%")