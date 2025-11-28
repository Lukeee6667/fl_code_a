import copy

import torch
from torch.nn.utils import parameters_to_vector
import numpy as np
import logging
from utils import vector_to_model, vector_to_name_param

import sklearn.metrics.pairwise as smp
from geom_median.torch import compute_geometric_median 


class Aggregation():
    def __init__(self, agent_data_sizes, n_params, args):
        self.agent_data_sizes = agent_data_sizes
        self.args = args
        self.server_lr = args.server_lr
        self.n_params = n_params
        
        if self.args.aggr == 'foolsgold':
            self.memory_dict = dict()
            self.wv_history = []
         # 如果使用alignins_plr策略，则预先准备辅助数据加载器
        if self.args.aggr == 'alignins_plr':
            self.auxiliary_data_loader = self.prepare_auxiliary_data()
        else:
            self.auxiliary_data_loader = None
        
    def aggregate_updates(self, global_model, agent_updates_dict, auxiliary_data_loader=None, current_round=None):
        lr_vector = torch.Tensor([self.server_lr]*self.n_params).to(self.args.device)
        if self.args.aggr != "rlr":
            lr_vector = lr_vector
        else:
            lr_vector, _ = self.compute_robustLR(agent_updates_dict)
        # mask = torch.ones_like(agent_updates_dict[0])
        aggregated_updates = 0
        cur_global_params = parameters_to_vector(
            [global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
        if self.args.aggr=='avg' or self.args.aggr == 'rlr' or self.args.aggr == 'lockdown':          
            aggregated_updates = self.agg_avg(agent_updates_dict)

        elif self.args.aggr == 'alignins':
            #agg_alignins_layer_lsd
            #agg_alignins_g_v
            #agg_alignins_plr
            # agg_alignins_3_p_v(self, agent_updates_dict, flat_global_model)
            # agg_alignins_3_metrics(self, agent_updates_dict, flat_global_model)
            # def agg_alignins_3_metrics_nonidd_badnet(self, agent_updates_dict, flat_global_model):
            # def agg_alignins_g_v2_onepic(self, agent_updates_dict, flat_global_model, current_round=None):
            aggregated_updates = self.agg_alignins(agent_updates_dict, cur_global_params)
        
        elif self.args.aggr == 'alignins_fedup_standard':
            # 新的标准AlignIns+FedUP聚合方法
            aggregated_updates = self.agg_alignins_fedup_standard(agent_updates_dict, cur_global_params, global_model, current_round)
        elif self.args.aggr == 'alignins_fedup_correct':
            # 正确的AlignIns+FedUP实现：先过滤聚合，再对模型权重剪枝
            aggregated_updates = self.agg_alignins_fedup_correct(agent_updates_dict, cur_global_params, global_model, current_round)
        elif self.args.aggr == 'not_unlearning':
            aggregated_updates = self.agg_not_unlearning(agent_updates_dict, cur_global_params, global_model, current_round)
        elif self.args.aggr=='alignins_v':
            aggregated_updates = self.agg_alignins_v(agent_updates_dict, cur_global_params, current_round)
        elif self.args.aggr == 'alignins_g_v':
            aggregated_updates = self.agg_alignins_g_v(agent_updates_dict, cur_global_params)
        elif self.args.aggr == 'alignins_g_v2':
            aggregated_updates = self.agg_alignins_g_v2(agent_updates_dict, cur_global_params, current_round)
        elif self.args.aggr == 'alignins_g_v2_onepic':
            aggregated_updates = self.agg_alignins_g_v2_onepic(agent_updates_dict, cur_global_params, current_round)
        elif self.args.aggr == 'alignins_layer_3p':
            aggregated_updates = self.agg_alignins_3_p_v(agent_updates_dict, cur_global_params)
        elif self.args.aggr == 'alignins_3_m':
            aggregated_updates = self.agg_alignins_3_metrics(agent_updates_dict, cur_global_params)
        elif self.args.aggr == 'alignins_3_m_noniid_badnet':
            aggregated_updates = self.agg_alignins_3_metrics_nonidd_badnet(agent_updates_dict, cur_global_params)
        elif self.args.aggr == 'alignins_3_m_noniid_dba':
            aggregated_updates = self.agg_alignins_3_metrics_nonidd_dba(agent_updates_dict, cur_global_params)
        elif self.args.aggr == 'alignins_layer_lsd':
            aggregated_updates = self.agg_alignins_layer_lsd(agent_updates_dict, cur_global_params, global_model)
        # elif self.args.aggr == 'agg_alignins_plr':
        #     aggregated_updates = self.agg_alignins_plr(agent_updates_dict, cur_global_params, global_model, auxiliary_data_loader)
        elif self.args.aggr == 'alignins_plr':
            flat_global_model = parameters_to_vector([global_model.state_dict()[name] for name in global_model.state_dict()])
            # 优先使用传入的数据加载器，如果为None则使用初始化时创建的
            data_loader = auxiliary_data_loader if auxiliary_data_loader is not None else self.auxiliary_data_loader
            if data_loader is not None:
                try:
                    aggregated_updates = self.agg_alignins_plr(agent_updates_dict, flat_global_model, global_model, data_loader)
                except Exception as e:
                    import logging
                    logging.error(f"PLR聚合失败: {str(e)}")
                    logging.warning("回退到平均值聚合")
                    aggregated_updates = self.agg_avg(agent_updates_dict)
            else:
                import logging
                logging.warning("缺少辅助数据加载器，使用平均值聚合")
                aggregated_updates = self.agg_avg(agent_updates_dict) 
        elif self.args.aggr == 'mmetric':
            aggregated_updates = self.agg_mul_metric(agent_updates_dict, global_model, cur_global_params)
        elif self.args.aggr == 'foolsgold':
            aggregated_updates = self.agg_foolsgold(agent_updates_dict)
        elif self.args.aggr == 'signguard':
            aggregated_updates = self.agg_signguard(agent_updates_dict)
        elif self.args.aggr == "mkrum":
            aggregated_updates = self.agg_mkrum(agent_updates_dict)
        elif self.args.aggr == "rfa":
            aggregated_updates = self.agg_rfa(agent_updates_dict)
        elif self.args.aggr == "fedup":
            aggregated_updates = self.agg_fedup(agent_updates_dict, cur_global_params, global_model, current_round)
        elif self.args.aggr == "alignins_fedup_hybrid":
            aggregated_updates = self.agg_alignins_fedup_hybrid(agent_updates_dict, cur_global_params, global_model, current_round)
        neurotoxin_mask = {}
        updates_dict = vector_to_name_param(aggregated_updates, copy.deepcopy(global_model.state_dict()))
        for name in updates_dict:
            updates = updates_dict[name].abs().view(-1)
            gradients_length = torch.numel(updates)
            _, indices = torch.topk(-1 * updates, int(gradients_length * self.args.dense_ratio))
            mask_flat = torch.zeros(gradients_length)
            mask_flat[indices.cpu()] = 1
            neurotoxin_mask[name] = (mask_flat.reshape(updates_dict[name].size()))

        cur_global_params = parameters_to_vector([ global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
        new_global_params =  (cur_global_params + lr_vector*aggregated_updates).float()
        vector_to_model(new_global_params, global_model)
        return updates_dict, neurotoxin_mask

    def agg_rfa(self, agent_updates_dict):
        local_updates = []

        for _id, update in agent_updates_dict.items():
            local_updates.append(update)

        n = len(local_updates)
        temp_updates = torch.stack(local_updates, dim=0)
        weights = torch.ones(n).to(self.args.device)  
        gw = compute_geometric_median(local_updates, weights).median
        for i in range(2):
            weights = torch.mul(weights, torch.exp(-1.0*torch.norm(temp_updates-gw, dim=1)))
            gw = compute_geometric_median(local_updates, weights).median

        aggregated_model = gw
        return aggregated_model
    
    def agg_fedup(self, agent_updates_dict, flat_global_model, global_model, current_round=None):
        """
        FedUP: Efficient Pruning-based Federated Unlearning for Model Poisoning Attacks
        基于剪枝的联邦遗忘聚合方法，实现论文Algorithm 1
        """
        import torch.nn.functional as F
        from torch.nn.utils import parameters_to_vector, vector_to_parameters
        
        # FedUP参数设置（基于论文）
        p_max = getattr(self.args, 'fedup_p_max', 0.15)  # 最大剪枝率
        p_min = getattr(self.args, 'fedup_p_min', 0.01)  # 最小剪枝率
        gamma = getattr(self.args, 'fedup_gamma', 5)     # 曲线陡度参数
        sensitivity_threshold = getattr(self.args, 'fedup_sensitivity_threshold', 0.5)  # 异常检测阈值
        
        logging.info(f"FedUP聚合 - P_max: {p_max}, P_min: {p_min}, Gamma: {gamma}")
        
        # 1. 计算客户端更新的统计信息
        update_magnitudes = {}
        update_directions = {}
        
        for client_id, update in agent_updates_dict.items():
            # 计算更新幅度
            update_magnitudes[client_id] = torch.norm(update).item()
            # 计算更新方向（归一化）
            update_directions[client_id] = F.normalize(update.unsqueeze(0), dim=1).squeeze(0)
        
        # 2. 检测异常客户端（基于更新幅度和方向一致性）
        magnitude_values = list(update_magnitudes.values())
        magnitude_mean = np.mean(magnitude_values)
        magnitude_std = np.std(magnitude_values)
        
        # 计算方向一致性矩阵
        direction_similarities = {}
        client_ids = list(agent_updates_dict.keys())
        
        for i, client_i in enumerate(client_ids):
            similarities = []
            for j, client_j in enumerate(client_ids):
                if i != j:
                    sim = torch.cosine_similarity(
                        update_directions[client_i].unsqueeze(0),
                        update_directions[client_j].unsqueeze(0)
                    ).item()
                    similarities.append(sim)
            direction_similarities[client_i] = np.mean(similarities)
        
        # 3. 识别需要遗忘的客户端
        suspicious_clients = set()
        
        # 基于更新幅度异常检测
        for client_id, magnitude in update_magnitudes.items():
            z_score = abs(magnitude - magnitude_mean) / (magnitude_std + 1e-8)
            if z_score > sensitivity_threshold:
                suspicious_clients.add(client_id)
                logging.info(f"客户端 {client_id} 被标记为异常 (幅度异常: z_score={z_score:.3f})")
        
        # 基于方向一致性检测
        similarity_values = list(direction_similarities.values())
        similarity_threshold = np.percentile(similarity_values, 25)  # 使用25%分位数作为阈值
        
        for client_id, similarity in direction_similarities.items():
            if similarity < similarity_threshold:
                suspicious_clients.add(client_id)
                logging.info(f"客户端 {client_id} 被标记为异常 (方向异常: similarity={similarity:.3f})")
        
        # 4. 计算自适应剪枝比例（基于论文公式5）
        benign_clients = [cid for cid in agent_updates_dict.keys() if cid not in suspicious_clients]
        adaptive_pruning_ratio = self._calculate_adaptive_pruning_ratio(
            agent_updates_dict, benign_clients, p_max, p_min, gamma
        )
        
        # 5. 生成遗忘掩码（基于排名机制）
        unlearn_mask = self._generate_unlearn_mask_ranking(
            agent_updates_dict, suspicious_clients, global_model, adaptive_pruning_ratio
        )
        
        # 6. 应用遗忘掩码进行聚合
        if len(suspicious_clients) > 0:
            logging.info(f"应用FedUP遗忘，影响客户端: {suspicious_clients}")
            logging.info(f"自适应剪枝比例: {adaptive_pruning_ratio:.4f}")
            # 过滤掉可疑客户端或降低其权重
            filtered_updates = {}
            total_weight = 0
            
            for client_id, update in agent_updates_dict.items():
                if client_id in suspicious_clients:
                    # 对可疑客户端应用遗忘掩码
                    masked_update = update * unlearn_mask
                    weight = self.agent_data_sizes[client_id] * 0.1  # 大幅降低权重
                else:
                    masked_update = update
                    weight = self.agent_data_sizes[client_id]
                
                filtered_updates[client_id] = masked_update
                total_weight += weight
            
            # 加权平均聚合
            aggregated_updates = torch.zeros_like(flat_global_model)
            for client_id, update in filtered_updates.items():
                if client_id in suspicious_clients:
                    weight = self.agent_data_sizes[client_id] * 0.1
                else:
                    weight = self.agent_data_sizes[client_id]
                aggregated_updates += (weight / total_weight) * update
        else:
            # 没有检测到异常，使用标准平均聚合
            logging.info("未检测到异常客户端，使用标准聚合")
            aggregated_updates = self.agg_avg(agent_updates_dict)
        
        return aggregated_update

    def agg_alignins_fedup_correct(self, agent_updates_dict, flat_global_model, global_model, current_round=None):
        """
        正确的AlignIns + FedUP实现
        先用AlignIns四指标检测和过滤聚合，再对聚合后的模型进行标准FedUP剪枝
        """
        # 导入正确实现
        from agg_alignins_fedup_correct import agg_alignins_fedup_correct
        
        # 转换输入格式
        inter_model_updates = torch.stack(list(agent_updates_dict.values()))
        
        # 获取恶意客户端ID（如果有的话）
        malicious_id = getattr(self.args, 'malicious_id', None)
        
        # 调用正确实现
        aggregated_update, pruned_model = agg_alignins_fedup_correct(
            inter_model_updates,
            flat_global_model,
            global_model,
            self.args,
            malicious_id,
            current_round
        )
        
        return aggregated_update

    def agg_not_unlearning(self, agent_updates_dict, flat_global_model, global_model, current_round=None):
        from agg_not_unlearning import agg_not_unlearning
        inter_model_updates = torch.stack(list(agent_updates_dict.values()))
        malicious_id = getattr(self.args, 'malicious_id', None)
        aggregated_update, _ = agg_not_unlearning(
            inter_model_updates,
            flat_global_model,
            global_model,
            self.args,
            malicious_id,
            current_round,
        )
        return aggregated_update
    
    def _calculate_adaptive_pruning_ratio(self, agent_updates_dict, benign_clients, p_max, p_min, gamma):
        """
        根据论文公式5计算自适应剪枝比例
        P ≈ (P_max - P_min) * z^γ + P_min
        """
        if len(benign_clients) < 2:
            return p_min
        
        # 计算良性客户端更新间的余弦相似度
        similarities = []
        benign_updates = [agent_updates_dict[cid] for cid in benign_clients]
        
        for i in range(len(benign_updates)):
            for j in range(i + 1, len(benign_updates)):
                cos_sim = torch.nn.functional.cosine_similarity(
                    benign_updates[i].unsqueeze(0), benign_updates[j].unsqueeze(0)
                ).item()
                similarities.append(cos_sim)
        
        # 计算平均相似度
        avg_similarity = sum(similarities) / len(similarities)
        
        # 归一化到[0,1]区间（假设收敛时相似度在[0.5,1]之间）
        z = max(0, min(1, (avg_similarity - 0.5) / 0.5))
        
        # 应用论文公式5
        adaptive_ratio = (p_max - p_min) * (z ** gamma) + p_min
        
        return adaptive_ratio
    
    def _generate_unlearn_mask_ranking(self, agent_updates_dict, suspicious_clients, global_model, pruning_ratio):
        """
        基于排名机制生成遗忘掩码（论文Algorithm 1）
        """
        if len(suspicious_clients) == 0:
            # 如果没有可疑客户端，返回全1掩码
            return torch.ones_like(parameters_to_vector(
                [global_model.state_dict()[name] for name in global_model.state_dict()]
            ))
        
        # 计算良性客户端更新的平均值
        benign_clients = [cid for cid in agent_updates_dict.keys() if cid not in suspicious_clients]
        if len(benign_clients) == 0:
            return torch.ones_like(parameters_to_vector(
                [global_model.state_dict()[name] for name in global_model.state_dict()]
            ))
        
        benign_updates = [agent_updates_dict[cid] for cid in benign_clients]
        avg_benign_update = torch.stack(benign_updates).mean(dim=0)
        
        # 计算可疑客户端更新的平均值
        suspicious_updates = [agent_updates_dict[cid] for cid in suspicious_clients]
        avg_suspicious_update = torch.stack(suspicious_updates).mean(dim=0)
        
        # 获取全局模型参数向量
        global_params = parameters_to_vector(
            [global_model.state_dict()[name] for name in global_model.state_dict()]
        )
        
        # 根据Algorithm 1计算rank（差异的平方乘以全局权重值）
        diff_squared = (avg_suspicious_update - avg_benign_update) ** 2
        rank = diff_squared * torch.abs(global_params)
        
        # 基于排名选择前pruning_ratio%的权重进行剪枝
        num_params = len(rank)
        num_prune = int(num_params * pruning_ratio)
        
        if num_prune > 0:
            # 获取排名最高的参数索引
            _, top_indices = torch.topk(rank, num_prune)
            
            # 创建掩码（对排名高的参数进行剪枝）
            mask = torch.ones_like(rank)
            mask[top_indices] = 0.0  # 剪枝掉这些参数
        else:
            mask = torch.ones_like(rank)
        
        return mask
    
    def _generate_unlearn_mask(self, agent_updates_dict, suspicious_clients, global_model, pruning_ratio):
        """
        生成用于遗忘的剪枝掩码（保留原方法以兼容性）
        """
        return self._generate_unlearn_mask_ranking(agent_updates_dict, suspicious_clients, global_model, pruning_ratio)

    def agg_alignins(self, agent_updates_dict, flat_global_model):
        local_updates = []
        benign_id = []
        malicious_id = []

        for _id, update in agent_updates_dict.items():
            local_updates.append(update)
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        chosen_clients = malicious_id + benign_id
        num_chosen_clients = len(malicious_id + benign_id)
        inter_model_updates = torch.stack(local_updates, dim=0)

        tda_list = []
        mpsa_list = []
        major_sign = torch.sign(torch.sum(torch.sign(inter_model_updates), dim=0))
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        for i in range(len(inter_model_updates)):
            _, init_indices = torch.topk(torch.abs(inter_model_updates[i]), int(len(inter_model_updates[i]) * self.args.sparsity))

            mpsa_list.append((torch.sum(torch.sign(inter_model_updates[i][init_indices]) == major_sign[init_indices]) / torch.numel(inter_model_updates[i][init_indices])).item())
    
            tda_list.append(cos(inter_model_updates[i], flat_global_model).item())


        logging.info('TDA: %s' % [round(i, 4) for i in tda_list])
        logging.info('MPSA: %s' % [round(i, 4) for i in mpsa_list])


        ######## MZ-score calculation ########
        mpsa_std = np.std(mpsa_list)
        mpsa_med = np.median(mpsa_list)

        mzscore_mpsa = []
        for i in range(len(mpsa_list)):
            mzscore_mpsa.append(np.abs(mpsa_list[i] - mpsa_med) / mpsa_std)

        logging.info('MZ-score of MPSA: %s' % [round(i, 4) for i in mzscore_mpsa])
        
        tda_std = np.std(tda_list)
        tda_med = np.median(tda_list)
        mzscore_tda = []
        for i in range(len(tda_list)):
            mzscore_tda.append(np.abs(tda_list[i] - tda_med) / tda_std)

        logging.info('MZ-score of TDA: %s' % [round(i, 4) for i in mzscore_tda])

        ######## Anomaly detection with MZ score ########

        benign_idx1 = set([i for i in range(num_chosen_clients)])
        benign_idx1 = benign_idx1.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_mpsa) < self.args.lambda_s)]))
        benign_idx2 = set([i for i in range(num_chosen_clients)])
        benign_idx2 = benign_idx2.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_tda) < self.args.lambda_c)]))

        benign_set = benign_idx2.intersection(benign_idx1)
        
        benign_idx = list(benign_set)
        if len(benign_idx) == 0:
            return torch.zeros_like(local_updates[0])

        benign_updates = torch.stack([local_updates[i] for i in benign_idx], dim=0)

        ######## Post-filtering model clipping ########
        
        updates_norm = torch.norm(benign_updates, dim=1).reshape((-1, 1))
        norm_clip = updates_norm.median(dim=0)[0].item()
        benign_updates = torch.stack(local_updates, dim=0)
        updates_norm = torch.norm(benign_updates, dim=1).reshape((-1, 1))
        updates_norm_clipped = torch.clamp(updates_norm, 0, norm_clip, out=None)
        # del grad_norm
        
        benign_updates = (benign_updates/updates_norm)*updates_norm_clipped

        correct = 0
        for idx in benign_idx:
            if idx >= len(malicious_id):
                correct += 1

        TPR = correct / len(benign_id)

        if len(malicious_id) == 0:
            FPR = 0
        else:
            wrong = 0
            for idx in benign_idx:
                if idx < len(malicious_id):
                    wrong += 1
            FPR = wrong / len(malicious_id)

        logging.info('benign update index:   %s' % str(benign_id))
        logging.info('selected update index: %s' % str(benign_idx))

        logging.info('FPR:       %.4f'  % FPR)
        logging.info('TPR:       %.4f' % TPR)

        current_dict = {}
        for idx in benign_idx:
            current_dict[chosen_clients[idx]] = benign_updates[idx]

        aggregated_update = self.agg_avg(current_dict)
        return aggregated_update
    
    def agg_alignins_v(self, agent_updates_dict, flat_global_model, current_round=None):
        local_updates = []
        benign_id = []
        malicious_id = []

        for _id, update in agent_updates_dict.items():
            local_updates.append(update)
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        chosen_clients = malicious_id + benign_id
        num_chosen_clients = len(malicious_id + benign_id)
        inter_model_updates = torch.stack(local_updates, dim=0)

        tda_list = []
        mpsa_list = []

        major_sign = torch.sign(torch.sum(torch.sign(inter_model_updates), dim=0))
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

        for i in range(len(inter_model_updates)):
            _, init_indices = torch.topk(torch.abs(inter_model_updates[i]), int(len(inter_model_updates[i]) * self.args.sparsity))

            mpsa_list.append((torch.sum(torch.sign(inter_model_updates[i][init_indices]) == major_sign[init_indices]) / torch.numel(inter_model_updates[i][init_indices])).item())
    
            tda_list.append(cos(inter_model_updates[i], flat_global_model).item())

        logging.info('TDA: %s' % [round(i, 4) for i in tda_list])
        logging.info('MPSA: %s' % [round(i, 4) for i in mpsa_list])

        ######## MZ-score calculation ########
        mpsa_std = np.std(mpsa_list)
        mpsa_med = np.median(mpsa_list)

        mzscore_mpsa = []
        for i in range(len(mpsa_list)):
            mzscore_mpsa.append(np.abs(mpsa_list[i] - mpsa_med) / mpsa_std)

        logging.info('MZ-score of MPSA: %s' % [round(i, 4) for i in mzscore_mpsa])
        
        tda_std = np.std(tda_list)
        tda_med = np.median(tda_list)
        mzscore_tda = []
        for i in range(len(tda_list)):
            mzscore_tda.append(np.abs(tda_list[i] - tda_med) / tda_std)

        logging.info('MZ-score of TDA: %s' % [round(i, 4) for i in mzscore_tda])
        
        ######## 添加可视化功能 ########
        import matplotlib.pyplot as plt
        import os
        import datetime
        
        # 只有在提供了current_round参数且current_round是10的倍数时才保存图表
        if current_round is not None and current_round % 10 == 0:
            # 获取当前时间作为文件夹名称的一部分
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 确保输出目录存在，使用时间戳创建唯一的文件夹
            base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'visualization_results')
            output_dir = os.path.join(base_dir, f'round_{current_round}_{current_time}')
            os.makedirs(output_dir, exist_ok=True)
            
            # 创建客户端类型列表
            client_types = []
            for i in range(num_chosen_clients):
                if i < len(malicious_id):
                    client_types.append('Malicious')
                else:
                    client_types.append('Benign')
            # 获取良性和恶意客户端的索引
            benign_indices = [i for i, t in enumerate(client_types) if t == 'Benign']
            malicious_indices = [i for i, t in enumerate(client_types) if t == 'Malicious']
            
            # 创建图表
            plt.figure(figsize=(15, 10))
            
            # 绘制TDA分布图
            plt.subplot(2, 2, 1)
            plt.scatter([i for i in benign_indices], [tda_list[i] for i in benign_indices], 
                    label='Benign Clients', color='blue', marker='o')
            plt.scatter([i for i in malicious_indices], [tda_list[i] for i in malicious_indices], 
                    label='Malicious Clients', color='red', marker='x')
            plt.axhline(y=tda_med, color='green', linestyle='-', label='Median')
            plt.title('TDA Distribution')
            plt.xlabel('Client Index')
            plt.ylabel('TDA Value')
            plt.legend()
            
            # 绘制MPSA分布图
            plt.subplot(2, 2, 2)
            plt.scatter([i for i in benign_indices], [mpsa_list[i] for i in benign_indices], 
                    label='Benign Clients', color='blue', marker='o')
            plt.scatter([i for i in malicious_indices], [mpsa_list[i] for i in malicious_indices], 
                    label='Malicious Clients', color='red', marker='x')
            plt.axhline(y=mpsa_med, color='green', linestyle='-', label='Median')
            plt.title('MPSA Distribution')
            plt.xlabel('Client Index')
            plt.ylabel('MPSA Value')
            plt.legend()
            
            # 绘制TDA MZ-score分布图
            plt.subplot(2, 2, 3)
            plt.scatter([i for i in benign_indices], [mzscore_tda[i] for i in benign_indices], 
                    label='Benign Clients', color='blue', marker='o')
            plt.scatter([i for i in malicious_indices], [mzscore_tda[i] for i in malicious_indices], 
                    label='Malicious Clients', color='red', marker='x')
            plt.axhline(y=self.args.lambda_c, color='green', linestyle='-', label='Threshold')
            plt.title('TDA MZ-score Distribution')
            plt.xlabel('Client Index')
            plt.ylabel('MZ-score Value')
            plt.legend()

            # 绘制MPSA MZ-score分布图
            plt.subplot(2, 2, 4)
            plt.scatter([i for i in benign_indices], [mzscore_mpsa[i] for i in benign_indices], 
                    label='Benign Clients', color='blue', marker='o')
            plt.scatter([i for i in malicious_indices], [mzscore_mpsa[i] for i in malicious_indices], 
                    label='Malicious Clients', color='red', marker='x')
            plt.axhline(y=self.args.lambda_s, color='green', linestyle='-', label='Threshold')
            plt.title('MPSA MZ-score Distribution')
            plt.xlabel('Client Index')
            plt.ylabel('MZ-score Value')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'metrics_analysis.png'))
            plt.close()
            logging.info(f"保存了第 {current_round} 轮的可视化结果到 {output_dir}")

        ######## Anomaly detection with MZ score ########

        benign_idx1 = set([i for i in range(num_chosen_clients)])
        benign_idx1 = benign_idx1.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_mpsa) < self.args.lambda_s)]))
        benign_idx2 = set([i for i in range(num_chosen_clients)])
        benign_idx2 = benign_idx2.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_tda) < self.args.lambda_c)]))

        benign_set = benign_idx2.intersection(benign_idx1)
        
        benign_idx = list(benign_set)
        if len(benign_idx) == 0:
            return torch.zeros_like(local_updates[0])

        benign_updates = torch.stack([local_updates[i] for i in benign_idx], dim=0)

        ######## Post-filtering model clipping ########
        
        updates_norm = torch.norm(benign_updates, dim=1).reshape((-1, 1))
        norm_clip = updates_norm.median(dim=0)[0].item()
        benign_updates = torch.stack(local_updates, dim=0)
        updates_norm = torch.norm(benign_updates, dim=1).reshape((-1, 1))
        updates_norm_clipped = torch.clamp(updates_norm, 0, norm_clip, out=None)
        
        benign_updates = (benign_updates/updates_norm)*updates_norm_clipped

        correct = 0
        for idx in benign_idx:
            if idx >= len(malicious_id):
                correct += 1

        TPR = correct / len(benign_id)

        if len(malicious_id) == 0:
            FPR = 0
        else:
            wrong = 0
            for idx in benign_idx:
                if idx < len(malicious_id):
                    wrong += 1
            FPR = wrong / len(malicious_id)

        logging.info('benign update index:   %s' % str(benign_id))
        logging.info('selected update index: %s' % str(benign_idx))

        logging.info('FPR:       %.4f'  % FPR)
        logging.info('TPR:       %.4f' % TPR)

        current_dict = {}
        for idx in benign_idx:
            current_dict[chosen_clients[idx]] = benign_updates[idx]

        aggregated_update = self.agg_avg(current_dict)
        return aggregated_update
    
    def agg_alignins_3_p_v(self, agent_updates_dict, flat_global_model):    
        local_updates = []
        benign_id = []
        malicious_id = []

        for _id, update in agent_updates_dict.items():
            local_updates.append(update)
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        chosen_clients = malicious_id + benign_id
        num_chosen_clients = len(malicious_id + benign_id)
        inter_model_updates = torch.stack(local_updates, dim=0)

        tda_list = []
        mpsa_list = []
        grad_norm_list = [] # 存储梯度L2范数

        major_sign = torch.sign(torch.sum(torch.sign(inter_model_updates), dim=0))
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

        for i in range(len(inter_model_updates)):
            _, init_indices = torch.topk(torch.abs(inter_model_updates[i]), int(len(inter_model_updates[i]) * self.args.sparsity))

            mpsa_list.append((torch.sum(torch.sign(inter_model_updates[i][init_indices]) == major_sign[init_indices]) / torch.numel(inter_model_updates[i][init_indices])).item())
    
            tda_list.append(cos(inter_model_updates[i], flat_global_model).item())
            grad_norm_list.append(torch.norm(inter_model_updates[i]).item()) # 计算L2范数

        logging.info('TDA: %s' % [round(i, 4) for i in tda_list])
        logging.info('MPSA: %s' % [round(i, 4) for i in mpsa_list])
        logging.info('Grad Norm: %s' % [round(i, 4) for i in grad_norm_list])

        ######## MZ-score calculation ########
        mpsa_std = np.std(mpsa_list)
        mpsa_med = np.median(mpsa_list)

        mzscore_mpsa = []
        for i in range(len(mpsa_list)):
            mzscore_mpsa.append(np.abs(mpsa_list[i] - mpsa_med) / mpsa_std)

        logging.info('MZ-score of MPSA: %s' % [round(i, 4) for i in mzscore_mpsa])
        
        tda_std = np.std(tda_list)
        tda_med = np.median(tda_list)
        mzscore_tda = []
        for i in range(len(tda_list)):
            mzscore_tda.append(np.abs(tda_list[i] - tda_med) / tda_std)

        logging.info('MZ-score of TDA: %s' % [round(i, 4) for i in mzscore_tda])

        # 梯度范数的MZ-score计算
        grad_norm_std = np.std(grad_norm_list)
        grad_norm_med = np.median(grad_norm_list)
        mzscore_grad_norm = []
        for i in range(len(grad_norm_list)):
            mzscore_grad_norm.append(np.abs(grad_norm_list[i] - grad_norm_med) / grad_norm_std)
        logging.info('MZ-score of Grad Norm: %s' % [round(i, 4) for i in mzscore_grad_norm])
        
        ######## 添加可视化功能 ########
        import matplotlib.pyplot as plt
        import os
        
        # 确保输出目录存在
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'visualization_idd_3m_badnet')
        os.makedirs(output_dir, exist_ok=True)
        # 创建客户端类型列表
        client_types = []
        for i in range(num_chosen_clients):
            if i < len(malicious_id):
                client_types.append('Malicious')
            else:
                client_types.append('Benign')
        # 获取良性和恶意客户端的索引
        benign_indices = [i for i, t in enumerate(client_types) if t == 'Benign']
        malicious_indices = [i for i, t in enumerate(client_types) if t == 'Malicious']
        
        # 创建图表 (3x2布局以包含梯度范数)
        plt.figure(figsize=(15, 15))
        
        # 绘制TDA分布图
        plt.subplot(3, 2, 1)
        plt.scatter([i for i in benign_indices], [tda_list[i] for i in benign_indices], 
                label='Benign Clients', color='blue', marker='o')
        plt.scatter([i for i in malicious_indices], [tda_list[i] for i in malicious_indices], 
                label='Malicious Clients', color='red', marker='x')
        plt.axhline(y=tda_med, color='green', linestyle='-', label='Median')
        plt.title('TDA Distribution')
        plt.xlabel('Client Index')
        plt.ylabel('TDA Value')
        plt.legend()
        
        # 绘制MPSA分布图
        plt.subplot(3, 2, 2)
        plt.scatter([i for i in benign_indices], [mpsa_list[i] for i in benign_indices], 
                label='Benign Clients', color='blue', marker='o')
        plt.scatter([i for i in malicious_indices], [mpsa_list[i] for i in malicious_indices], 
                label='Malicious Clients', color='red', marker='x')
        plt.axhline(y=mpsa_med, color='green', linestyle='-', label='Median')
        plt.title('MPSA Distribution')
        plt.xlabel('Client Index')
        plt.ylabel('MPSA Value')
        plt.legend()
        
        # 绘制梯度范数分布图
        plt.subplot(3, 2, 3)
        plt.scatter([i for i in benign_indices], [grad_norm_list[i] for i in benign_indices], 
                label='Benign Clients', color='blue', marker='o')
        plt.scatter([i for i in malicious_indices], [grad_norm_list[i] for i in malicious_indices], 
                label='Malicious Clients', color='red', marker='x')
        plt.axhline(y=grad_norm_med, color='green', linestyle='-', label='Median')
        plt.title('Grad Norm Distribution')
        plt.xlabel('Client Index')
        plt.ylabel('Grad Norm Value')
        plt.legend()
        
        # 绘制TDA MZ-score分布图
        plt.subplot(3, 2, 4)
        plt.scatter([i for i in benign_indices], [mzscore_tda[i] for i in benign_indices], 
                label='Benign Clients', color='blue', marker='o')
        plt.scatter([i for i in malicious_indices], [mzscore_tda[i] for i in malicious_indices], 
                label='Malicious Clients', color='red', marker='x')
        plt.axhline(y=self.args.lambda_c, color='green', linestyle='-', label='Threshold')
        plt.title('TDA MZ-score Distribution')
        plt.xlabel('Client Index')
        plt.ylabel('MZ-score Value')
        plt.legend()

        # 绘制MPSA MZ-score分布图
        plt.subplot(3, 2, 5)
        plt.scatter([i for i in benign_indices], [mzscore_mpsa[i] for i in benign_indices], 
                label='Benign Clients', color='blue', marker='o')
        plt.scatter([i for i in malicious_indices], [mzscore_mpsa[i] for i in malicious_indices], 
                label='Malicious Clients', color='red', marker='x')
        plt.axhline(y=self.args.lambda_s, color='green', linestyle='-', label='Threshold')
        plt.title('MPSA MZ-score Distribution')
        plt.xlabel('Client Index')
        plt.ylabel('MZ-score Value')
        plt.legend()

        # 绘制梯度范数MZ-score分布图
        plt.subplot(3, 2, 6)
        plt.scatter([i for i in benign_indices], [mzscore_grad_norm[i] for i in benign_indices], 
                label='Benign Clients', color='blue', marker='o')
        plt.scatter([i for i in malicious_indices], [mzscore_grad_norm[i] for i in malicious_indices], 
                label='Malicious Clients', color='red', marker='x')
        # 假设有一个新的阈值参数，例如 self.args.lambda_g
        # 如果没有，可以先用一个默认值或者根据实际情况调整
        if hasattr(self.args, 'lambda_g'):
            plt.axhline(y=self.args.lambda_g, color='green', linestyle='-', label='Threshold')
        else:
            # 临时使用一个默认值，或者根据数据分布动态确定
            plt.axhline(y=1.5, color='green', linestyle='-', label='Default Threshold (1.5)') 
        plt.title('Grad Norm MZ-score Distribution')
        plt.xlabel('Client Index')
        plt.ylabel('MZ-score Value')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metrics_analysis.png'))
        plt.close()

        ######## Anomaly detection with MZ score ########

        benign_idx1 = set([i for i in range(num_chosen_clients)])
        benign_idx1 = benign_idx1.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_mpsa) < self.args.lambda_s)]))
        benign_idx2 = set([i for i in range(num_chosen_clients)])
        benign_idx2 = benign_idx2.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_tda) < self.args.lambda_c)]))

        # 基于梯度范数MZ-score的过滤
        benign_idx3 = set([i for i in range(num_chosen_clients)])
        if hasattr(self.args, 'lambda_g'):
            benign_idx3 = benign_idx3.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_grad_norm) < self.args.lambda_g)]))
        else:
            # 如果没有lambda_g，可以考虑不使用此过滤，或者使用一个默认阈值
            benign_idx3 = benign_idx3.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_grad_norm) < 1.5)])) # 临时默认阈值

        benign_set = benign_idx2.intersection(benign_idx1).intersection(benign_idx3) # 结合三个过滤条件
        
        benign_idx = list(benign_set)
        if len(benign_idx) == 0:
            return torch.zeros_like(local_updates[0])

        benign_updates = torch.stack([local_updates[i] for i in benign_idx], dim=0)

        ######## Post-filtering model clipping ########
        
        updates_norm = torch.norm(benign_updates, dim=1).reshape((-1, 1))
        norm_clip = updates_norm.median(dim=0)[0].item()
        benign_updates = torch.stack(local_updates, dim=0)
        updates_norm = torch.norm(benign_updates, dim=1).reshape((-1, 1))
        updates_norm_clipped = torch.clamp(updates_norm, 0, norm_clip, out=None)
        
        benign_updates = (benign_updates/updates_norm)*updates_norm_clipped

        correct = 0
        for idx in benign_idx:
            if idx >= len(malicious_id):
                correct += 1

        TPR = correct / len(benign_id)

        if len(malicious_id) == 0:
            FPR = 0
        else:
            wrong = 0
            for idx in benign_idx:
                if idx < len(malicious_id):
                    wrong += 1
            FPR = wrong / len(malicious_id)

        logging.info('benign update index:   %s' % str(benign_id))
        logging.info('selected update index: %s' % str(benign_idx))

        logging.info('FPR:       %.4f'  % FPR)
        logging.info('TPR:       %.4f' % TPR)

        current_dict = {}
        for idx in benign_idx:
            current_dict[chosen_clients[idx]] = benign_updates[idx]

        aggregated_update = self.agg_avg(current_dict)
        return aggregated_update

    def agg_alignins_3_metrics_nonidd_badnet(self, agent_updates_dict, flat_global_model):
        """
        使用三个指标进行恶意客户端检测和过滤：
        1. MPSA MZ-score
        2. TDA MZ-score  
        3. 与平均更新的余弦相似度 MZ-score
        """
        local_updates = []
        benign_id = []
        malicious_id = []

        for _id, update in agent_updates_dict.items():
            local_updates.append(update)
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        chosen_clients = malicious_id + benign_id
        num_chosen_clients = len(malicious_id + benign_id)
        inter_model_updates = torch.stack(local_updates, dim=0)

        tda_list = []
        mpsa_list = []
        mean_cos_list = []  # 存储与平均更新的余弦相似度

        major_sign = torch.sign(torch.sum(torch.sign(inter_model_updates), dim=0))
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

        # 计算所有客户端更新的平均值
        mean_update = torch.mean(inter_model_updates, dim=0)

        for i in range(len(inter_model_updates)):
            _, init_indices = torch.topk(torch.abs(inter_model_updates[i]), int(len(inter_model_updates[i]) * self.args.sparsity))

            mpsa_list.append((torch.sum(torch.sign(inter_model_updates[i][init_indices]) == major_sign[init_indices]) / torch.numel(inter_model_updates[i][init_indices])).item())
            
            tda_list.append(cos(inter_model_updates[i], flat_global_model).item())
            mean_cos_list.append(cos(inter_model_updates[i], mean_update).item())  # 计算与平均更新的余弦相似度

        logging.info('TDA: %s' % [round(i, 4) for i in tda_list])
        logging.info('MPSA: %s' % [round(i, 4) for i in mpsa_list])
        logging.info('Mean Cos: %s' % [round(i, 4) for i in mean_cos_list])

        ######## MZ-score calculation ########
        # MPSA MZ-score
        mpsa_std = np.std(mpsa_list)
        mpsa_med = np.median(mpsa_list)
        mzscore_mpsa = []
        for i in range(len(mpsa_list)):
            mzscore_mpsa.append(np.abs(mpsa_list[i] - mpsa_med) / mpsa_std)
        logging.info('MZ-score of MPSA: %s' % [round(i, 4) for i in mzscore_mpsa])
        
        # TDA MZ-score
        tda_std = np.std(tda_list)
        tda_med = np.median(tda_list)
        mzscore_tda = []
        for i in range(len(tda_list)):
            mzscore_tda.append(np.abs(tda_list[i] - tda_med) / tda_std)
        logging.info('MZ-score of TDA: %s' % [round(i, 4) for i in mzscore_tda])

        # 与平均更新余弦相似度的MZ-score
        mean_cos_std = np.std(mean_cos_list)
        mean_cos_med = np.median(mean_cos_list)
        mzscore_mean_cos = []
        for i in range(len(mean_cos_list)):
            mzscore_mean_cos.append(np.abs(mean_cos_list[i] - mean_cos_med) / mean_cos_std)
        logging.info('MZ-score of Mean Cos: %s' % [round(i, 4) for i in mzscore_mean_cos])
        
        ######## 添加可视化功能 ########
        import matplotlib.pyplot as plt
        import os
        
        # 确保输出目录存在
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'visualization_3_metrics_noniid_badnet')
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建客户端类型列表
        client_types = []
        for i in range(num_chosen_clients):
            if i < len(malicious_id):
                client_types.append('Malicious')
            else:
                client_types.append('Benign')
        
        # 获取良性和恶意客户端的索引
        benign_indices = [i for i, t in enumerate(client_types) if t == 'Benign']
        malicious_indices = [i for i, t in enumerate(client_types) if t == 'Malicious']
        
        # 创建图表 (2x3布局)
        plt.figure(figsize=(18, 12))
        
        # 绘制TDA分布图
        plt.subplot(2, 3, 1)
        plt.scatter([i for i in benign_indices], [tda_list[i] for i in benign_indices], 
                label='Benign Clients', color='blue', marker='o')
        plt.scatter([i for i in malicious_indices], [tda_list[i] for i in malicious_indices], 
                label='Malicious Clients', color='red', marker='x')
        plt.axhline(y=tda_med, color='green', linestyle='-', label='Median')
        plt.title('TDA Distribution')
        plt.xlabel('Client Index')
        plt.ylabel('TDA Value')
        plt.legend()
        
        # 绘制MPSA分布图
        plt.subplot(2, 3, 2)
        plt.scatter([i for i in benign_indices], [mpsa_list[i] for i in benign_indices], 
                label='Benign Clients', color='blue', marker='o')
        plt.scatter([i for i in malicious_indices], [mpsa_list[i] for i in malicious_indices], 
                label='Malicious Clients', color='red', marker='x')
        plt.axhline(y=mpsa_med, color='green', linestyle='-', label='Median')
        plt.title('MPSA Distribution')
        plt.xlabel('Client Index')
        plt.ylabel('MPSA Value')
        plt.legend()
        
        # 绘制与平均更新余弦相似度分布图
        plt.subplot(2, 3, 3)
        plt.scatter([i for i in benign_indices], [mean_cos_list[i] for i in benign_indices], 
                label='Benign Clients', color='blue', marker='o')
        plt.scatter([i for i in malicious_indices], [mean_cos_list[i] for i in malicious_indices], 
                label='Malicious Clients', color='red', marker='x')
        plt.axhline(y=mean_cos_med, color='green', linestyle='-', label='Median')
        plt.title('Mean Cosine Similarity Distribution')
        plt.xlabel('Client Index')
        plt.ylabel('Mean Cosine Similarity')
        plt.legend()
        
        # 绘制TDA MZ-score分布图
        plt.subplot(2, 3, 4)
        plt.scatter([i for i in benign_indices], [mzscore_tda[i] for i in benign_indices], 
                label='Benign Clients', color='blue', marker='o')
        plt.scatter([i for i in malicious_indices], [mzscore_tda[i] for i in malicious_indices], 
                label='Malicious Clients', color='red', marker='x')
        plt.axhline(y=self.args.lambda_c, color='green', linestyle='-', label='Threshold')
        plt.title('TDA MZ-score Distribution')
        plt.xlabel('Client Index')
        plt.ylabel('MZ-score Value')
        plt.legend()

        # 绘制MPSA MZ-score分布图
        plt.subplot(2, 3, 5)
        plt.scatter([i for i in benign_indices], [mzscore_mpsa[i] for i in benign_indices], 
                label='Benign Clients', color='blue', marker='o')
        plt.scatter([i for i in malicious_indices], [mzscore_mpsa[i] for i in malicious_indices], 
                label='Malicious Clients', color='red', marker='x')
        plt.axhline(y=self.args.lambda_s, color='green', linestyle='-', label='Threshold')
        plt.title('MPSA MZ-score Distribution')
        plt.xlabel('Client Index')
        plt.ylabel('MZ-score Value')
        plt.legend()

        # 绘制与平均更新余弦相似度MZ-score分布图
        plt.subplot(2, 3, 6)
        plt.scatter([i for i in benign_indices], [mzscore_mean_cos[i] for i in benign_indices], 
                label='Benign Clients', color='blue', marker='o')
        plt.scatter([i for i in malicious_indices], [mzscore_mean_cos[i] for i in malicious_indices], 
                label='Malicious Clients', color='red', marker='x')
        # 使用阈值参数或默认值
        if hasattr(self.args, 'lambda_mean_cos'):
            plt.axhline(y=self.args.lambda_mean_cos, color='green', linestyle='-', label='Threshold')
        else:
            plt.axhline(y=2.0, color='green', linestyle='-', label='Default Threshold (2.0)')
        plt.title('Mean Cos MZ-score Distribution')
        plt.xlabel('Client Index')
        plt.ylabel('MZ-score Value')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'three_metrics_analysis.png'))
        plt.close()

        ######## Anomaly detection with MZ score (3 metrics) ########

        # 基于MPSA MZ-score的过滤
        benign_idx1 = set([i for i in range(num_chosen_clients)])
        benign_idx1 = benign_idx1.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_mpsa) < self.args.lambda_s)]))
        
        # 基于TDA MZ-score的过滤
        benign_idx2 = set([i for i in range(num_chosen_clients)])
        benign_idx2 = benign_idx2.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_tda) < self.args.lambda_c)]))

        # 基于与平均更新余弦相似度MZ-score的过滤
        benign_idx3 = set([i for i in range(num_chosen_clients)])
        if hasattr(self.args, 'lambda_mean_cos'):
            benign_idx3 = benign_idx3.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_mean_cos) < self.args.lambda_mean_cos)]))
        else:
            # 如果没有lambda_mean_cos，使用默认阈值
            benign_idx3 = benign_idx3.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_mean_cos) < 2.0)]))

        # 结合三个过滤条件
        benign_set = benign_idx1.intersection(benign_idx2).intersection(benign_idx3)
        
        benign_idx = list(benign_set)
        if len(benign_idx) == 0:
            return torch.zeros_like(local_updates[0])

        ######## Post-filtering model clipping ########
        
        # 修复：使用过滤后的benign_idx来选择更新
        benign_updates = torch.stack([local_updates[i] for i in benign_idx], dim=0)
        
        updates_norm = torch.norm(benign_updates, dim=1).reshape((-1, 1))
        norm_clip = updates_norm.median(dim=0)[0].item()
        
        # 对良性更新进行裁剪
        updates_norm_clipped = torch.clamp(updates_norm, 0, norm_clip, out=None)
        benign_updates = (benign_updates/updates_norm)*updates_norm_clipped

        # 计算TPR和FPR
        correct = 0
        for idx in benign_idx:
            if idx >= len(malicious_id):
                correct += 1

        TPR = correct / len(benign_id)

        if len(malicious_id) == 0:
            FPR = 0
        else:
            wrong = 0
            for idx in benign_idx:
                if idx < len(malicious_id):
                    wrong += 1
            FPR = wrong / len(malicious_id)

        logging.info('benign update index:   %s' % str(benign_id))
        logging.info('selected update index: %s' % str(benign_idx))
        logging.info('FPR:       %.4f'  % FPR)
        logging.info('TPR:       %.4f' % TPR)

        # 构建最终的聚合字典 - 修复索引问题
        current_dict = {}
        for i, idx in enumerate(benign_idx):
            current_dict[chosen_clients[idx]] = benign_updates[i]  # 使用i而不是idx作为benign_updates的索引

        aggregated_update = self.agg_avg(current_dict)
        return aggregated_update

    def agg_alignins_3_metrics_nonidd_dba(self, agent_updates_dict, flat_global_model):
        """
        使用三个指标进行恶意客户端检测和过滤：
        1. MPSA MZ-score
        2. TDA MZ-score  
        3. 与平均更新的余弦相似度 MZ-score
        """
        local_updates = []
        benign_id = []
        malicious_id = []

        for _id, update in agent_updates_dict.items():
            local_updates.append(update)
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        chosen_clients = malicious_id + benign_id
        num_chosen_clients = len(malicious_id + benign_id)
        inter_model_updates = torch.stack(local_updates, dim=0)

        tda_list = []
        mpsa_list = []
        mean_cos_list = []  # 存储与平均更新的余弦相似度

        major_sign = torch.sign(torch.sum(torch.sign(inter_model_updates), dim=0))
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

        # 计算所有客户端更新的平均值
        mean_update = torch.mean(inter_model_updates, dim=0)

        for i in range(len(inter_model_updates)):
            _, init_indices = torch.topk(torch.abs(inter_model_updates[i]), int(len(inter_model_updates[i]) * self.args.sparsity))

            mpsa_list.append((torch.sum(torch.sign(inter_model_updates[i][init_indices]) == major_sign[init_indices]) / torch.numel(inter_model_updates[i][init_indices])).item())
            
            tda_list.append(cos(inter_model_updates[i], flat_global_model).item())
            mean_cos_list.append(cos(inter_model_updates[i], mean_update).item())  # 计算与平均更新的余弦相似度

        logging.info('TDA: %s' % [round(i, 4) for i in tda_list])
        logging.info('MPSA: %s' % [round(i, 4) for i in mpsa_list])
        logging.info('Mean Cos: %s' % [round(i, 4) for i in mean_cos_list])

        ######## MZ-score calculation ########
        # MPSA MZ-score
        mpsa_std = np.std(mpsa_list)
        mpsa_med = np.median(mpsa_list)
        mzscore_mpsa = []
        for i in range(len(mpsa_list)):
            mzscore_mpsa.append(np.abs(mpsa_list[i] - mpsa_med) / mpsa_std)
        logging.info('MZ-score of MPSA: %s' % [round(i, 4) for i in mzscore_mpsa])
        
        # TDA MZ-score
        tda_std = np.std(tda_list)
        tda_med = np.median(tda_list)
        mzscore_tda = []
        for i in range(len(tda_list)):
            mzscore_tda.append(np.abs(tda_list[i] - tda_med) / tda_std)
        logging.info('MZ-score of TDA: %s' % [round(i, 4) for i in mzscore_tda])

        # 与平均更新余弦相似度的MZ-score
        mean_cos_std = np.std(mean_cos_list)
        mean_cos_med = np.median(mean_cos_list)
        mzscore_mean_cos = []
        for i in range(len(mean_cos_list)):
            mzscore_mean_cos.append(np.abs(mean_cos_list[i] - mean_cos_med) / mean_cos_std)
        logging.info('MZ-score of Mean Cos: %s' % [round(i, 4) for i in mzscore_mean_cos])
        
        ######## 添加可视化功能 ########
        import matplotlib.pyplot as plt
        import os
        
        # 确保输出目录存在
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'visualization_3_metrics_noniid_dba')
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建客户端类型列表
        client_types = []
        for i in range(num_chosen_clients):
            if i < len(malicious_id):
                client_types.append('Malicious')
            else:
                client_types.append('Benign')
        
        # 获取良性和恶意客户端的索引
        benign_indices = [i for i, t in enumerate(client_types) if t == 'Benign']
        malicious_indices = [i for i, t in enumerate(client_types) if t == 'Malicious']
        
        # 创建图表 (2x3布局)
        plt.figure(figsize=(18, 12))
        
        # 绘制TDA分布图
        plt.subplot(2, 3, 1)
        plt.scatter([i for i in benign_indices], [tda_list[i] for i in benign_indices], 
                label='Benign Clients', color='blue', marker='o')
        plt.scatter([i for i in malicious_indices], [tda_list[i] for i in malicious_indices], 
                label='Malicious Clients', color='red', marker='x')
        plt.axhline(y=tda_med, color='green', linestyle='-', label='Median')
        plt.title('TDA Distribution')
        plt.xlabel('Client Index')
        plt.ylabel('TDA Value')
        plt.legend()
        
        # 绘制MPSA分布图
        plt.subplot(2, 3, 2)
        plt.scatter([i for i in benign_indices], [mpsa_list[i] for i in benign_indices], 
                label='Benign Clients', color='blue', marker='o')
        plt.scatter([i for i in malicious_indices], [mpsa_list[i] for i in malicious_indices], 
                label='Malicious Clients', color='red', marker='x')
        plt.axhline(y=mpsa_med, color='green', linestyle='-', label='Median')
        plt.title('MPSA Distribution')
        plt.xlabel('Client Index')
        plt.ylabel('MPSA Value')
        plt.legend()
        
        # 绘制与平均更新余弦相似度分布图
        plt.subplot(2, 3, 3)
        plt.scatter([i for i in benign_indices], [mean_cos_list[i] for i in benign_indices], 
                label='Benign Clients', color='blue', marker='o')
        plt.scatter([i for i in malicious_indices], [mean_cos_list[i] for i in malicious_indices], 
                label='Malicious Clients', color='red', marker='x')
        plt.axhline(y=mean_cos_med, color='green', linestyle='-', label='Median')
        plt.title('Mean Cosine Similarity Distribution')
        plt.xlabel('Client Index')
        plt.ylabel('Mean Cosine Similarity')
        plt.legend()
        
        # 绘制TDA MZ-score分布图
        plt.subplot(2, 3, 4)
        plt.scatter([i for i in benign_indices], [mzscore_tda[i] for i in benign_indices], 
                label='Benign Clients', color='blue', marker='o')
        plt.scatter([i for i in malicious_indices], [mzscore_tda[i] for i in malicious_indices], 
                label='Malicious Clients', color='red', marker='x')
        plt.axhline(y=self.args.lambda_c, color='green', linestyle='-', label='Threshold')
        plt.title('TDA MZ-score Distribution')
        plt.xlabel('Client Index')
        plt.ylabel('MZ-score Value')
        plt.legend()

        # 绘制MPSA MZ-score分布图
        plt.subplot(2, 3, 5)
        plt.scatter([i for i in benign_indices], [mzscore_mpsa[i] for i in benign_indices], 
                label='Benign Clients', color='blue', marker='o')
        plt.scatter([i for i in malicious_indices], [mzscore_mpsa[i] for i in malicious_indices], 
                label='Malicious Clients', color='red', marker='x')
        plt.axhline(y=self.args.lambda_s, color='green', linestyle='-', label='Threshold')
        plt.title('MPSA MZ-score Distribution')
        plt.xlabel('Client Index')
        plt.ylabel('MZ-score Value')
        plt.legend()

        # 绘制与平均更新余弦相似度MZ-score分布图
        plt.subplot(2, 3, 6)
        plt.scatter([i for i in benign_indices], [mzscore_mean_cos[i] for i in benign_indices], 
                label='Benign Clients', color='blue', marker='o')
        plt.scatter([i for i in malicious_indices], [mzscore_mean_cos[i] for i in malicious_indices], 
                label='Malicious Clients', color='red', marker='x')
        # 使用阈值参数或默认值
        if hasattr(self.args, 'lambda_mean_cos'):
            plt.axhline(y=self.args.lambda_mean_cos, color='green', linestyle='-', label='Threshold')
        else:
            plt.axhline(y=2.0, color='green', linestyle='-', label='Default Threshold (2.0)')
        plt.title('Mean Cos MZ-score Distribution')
        plt.xlabel('Client Index')
        plt.ylabel('MZ-score Value')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'three_metrics_analysis.png'))
        plt.close()

        ######## Anomaly detection with MZ score (3 metrics) ########

        # 基于MPSA MZ-score的过滤
        benign_idx1 = set([i for i in range(num_chosen_clients)])
        benign_idx1 = benign_idx1.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_mpsa) < self.args.lambda_s)]))
        
        # 基于TDA MZ-score的过滤
        benign_idx2 = set([i for i in range(num_chosen_clients)])
        benign_idx2 = benign_idx2.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_tda) < self.args.lambda_c)]))

        # 基于与平均更新余弦相似度MZ-score的过滤
        benign_idx3 = set([i for i in range(num_chosen_clients)])
        if hasattr(self.args, 'lambda_mean_cos'):
            benign_idx3 = benign_idx3.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_mean_cos) < self.args.lambda_mean_cos)]))
        else:
            # 如果没有lambda_mean_cos，使用默认阈值
            benign_idx3 = benign_idx3.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_mean_cos) < 2.0)]))

        # 结合三个过滤条件
        benign_set = benign_idx1.intersection(benign_idx2).intersection(benign_idx3)
        
        benign_idx = list(benign_set)
        if len(benign_idx) == 0:
            return torch.zeros_like(local_updates[0])

        ######## Post-filtering model clipping ########
        
        # 修复：使用过滤后的benign_idx来选择更新
        benign_updates = torch.stack([local_updates[i] for i in benign_idx], dim=0)
        
        updates_norm = torch.norm(benign_updates, dim=1).reshape((-1, 1))
        norm_clip = updates_norm.median(dim=0)[0].item()
        
        # 对良性更新进行裁剪
        updates_norm_clipped = torch.clamp(updates_norm, 0, norm_clip, out=None)
        benign_updates = (benign_updates/updates_norm)*updates_norm_clipped

        # 计算TPR和FPR
        correct = 0
        for idx in benign_idx:
            if idx >= len(malicious_id):
                correct += 1

        TPR = correct / len(benign_id)

        if len(malicious_id) == 0:
            FPR = 0
        else:
            wrong = 0
            for idx in benign_idx:
                if idx < len(malicious_id):
                    wrong += 1
            FPR = wrong / len(malicious_id)

        logging.info('benign update index:   %s' % str(benign_id))
        logging.info('selected update index: %s' % str(benign_idx))
        logging.info('FPR:       %.4f'  % FPR)
        logging.info('TPR:       %.4f' % TPR)

        # 构建最终的聚合字典 - 修复索引问题
        current_dict = {}
        for i, idx in enumerate(benign_idx):
            current_dict[chosen_clients[idx]] = benign_updates[i]  # 使用i而不是idx作为benign_updates的索引

        aggregated_update = self.agg_avg(current_dict)
        return aggregated_update

    def agg_alignins_3_metrics(self, agent_updates_dict, flat_global_model):
        """
        使用三个指标进行恶意客户端检测和过滤：
        1. MPSA MZ-score
        2. TDA MZ-score  
        3. 与平均更新的余弦相似度 MZ-score
        """
        local_updates = []
        benign_id = []
        malicious_id = []

        for _id, update in agent_updates_dict.items():
            local_updates.append(update)
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        chosen_clients = malicious_id + benign_id
        num_chosen_clients = len(malicious_id + benign_id)
        inter_model_updates = torch.stack(local_updates, dim=0)

        tda_list = []
        mpsa_list = []
        mean_cos_list = []  # 存储与平均更新的余弦相似度

        major_sign = torch.sign(torch.sum(torch.sign(inter_model_updates), dim=0))
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

        # 计算所有客户端更新的平均值
        mean_update = torch.mean(inter_model_updates, dim=0)

        for i in range(len(inter_model_updates)):
            _, init_indices = torch.topk(torch.abs(inter_model_updates[i]), int(len(inter_model_updates[i]) * self.args.sparsity))

            mpsa_list.append((torch.sum(torch.sign(inter_model_updates[i][init_indices]) == major_sign[init_indices]) / torch.numel(inter_model_updates[i][init_indices])).item())
            
            tda_list.append(cos(inter_model_updates[i], flat_global_model).item())
            mean_cos_list.append(cos(inter_model_updates[i], mean_update).item())  # 计算与平均更新的余弦相似度

        logging.info('TDA: %s' % [round(i, 4) for i in tda_list])
        logging.info('MPSA: %s' % [round(i, 4) for i in mpsa_list])
        logging.info('Mean Cos: %s' % [round(i, 4) for i in mean_cos_list])

        ######## MZ-score calculation ########
        # MPSA MZ-score
        mpsa_std = np.std(mpsa_list)
        mpsa_med = np.median(mpsa_list)
        mzscore_mpsa = []
        for i in range(len(mpsa_list)):
            mzscore_mpsa.append(np.abs(mpsa_list[i] - mpsa_med) / mpsa_std)
        logging.info('MZ-score of MPSA: %s' % [round(i, 4) for i in mzscore_mpsa])
        
        # TDA MZ-score
        tda_std = np.std(tda_list)
        tda_med = np.median(tda_list)
        mzscore_tda = []
        for i in range(len(tda_list)):
            mzscore_tda.append(np.abs(tda_list[i] - tda_med) / tda_std)
        logging.info('MZ-score of TDA: %s' % [round(i, 4) for i in mzscore_tda])

        # 与平均更新余弦相似度的MZ-score
        mean_cos_std = np.std(mean_cos_list)
        mean_cos_med = np.median(mean_cos_list)
        mzscore_mean_cos = []
        for i in range(len(mean_cos_list)):
            mzscore_mean_cos.append(np.abs(mean_cos_list[i] - mean_cos_med) / mean_cos_std)
        logging.info('MZ-score of Mean Cos: %s' % [round(i, 4) for i in mzscore_mean_cos])
        
        ######## 添加可视化功能 ########
        import matplotlib.pyplot as plt
        import os
        
        # 确保输出目录存在
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'visualization_3_metrics_non_iid_badnet_beta_0.3')
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建客户端类型列表
        client_types = []
        for i in range(num_chosen_clients):
            if i < len(malicious_id):
                client_types.append('Malicious')
            else:
                client_types.append('Benign')
        
        # 获取良性和恶意客户端的索引
        benign_indices = [i for i, t in enumerate(client_types) if t == 'Benign']
        malicious_indices = [i for i, t in enumerate(client_types) if t == 'Malicious']
        
        # 创建图表 (2x3布局)
        plt.figure(figsize=(18, 12))
        
        # 绘制TDA分布图
        plt.subplot(2, 3, 1)
        plt.scatter([i for i in benign_indices], [tda_list[i] for i in benign_indices], 
                label='Benign Clients', color='blue', marker='o')
        plt.scatter([i for i in malicious_indices], [tda_list[i] for i in malicious_indices], 
                label='Malicious Clients', color='red', marker='x')
        plt.axhline(y=tda_med, color='green', linestyle='-', label='Median')
        plt.title('TDA Distribution')
        plt.xlabel('Client Index')
        plt.ylabel('TDA Value')
        plt.legend()
        
        # 绘制MPSA分布图
        plt.subplot(2, 3, 2)
        plt.scatter([i for i in benign_indices], [mpsa_list[i] for i in benign_indices], 
                label='Benign Clients', color='blue', marker='o')
        plt.scatter([i for i in malicious_indices], [mpsa_list[i] for i in malicious_indices], 
                label='Malicious Clients', color='red', marker='x')
        plt.axhline(y=mpsa_med, color='green', linestyle='-', label='Median')
        plt.title('MPSA Distribution')
        plt.xlabel('Client Index')
        plt.ylabel('MPSA Value')
        plt.legend()
        
        # 绘制与平均更新余弦相似度分布图
        plt.subplot(2, 3, 3)
        plt.scatter([i for i in benign_indices], [mean_cos_list[i] for i in benign_indices], 
                label='Benign Clients', color='blue', marker='o')
        plt.scatter([i for i in malicious_indices], [mean_cos_list[i] for i in malicious_indices], 
                label='Malicious Clients', color='red', marker='x')
        plt.axhline(y=mean_cos_med, color='green', linestyle='-', label='Median')
        plt.title('Mean Cosine Similarity Distribution')
        plt.xlabel('Client Index')
        plt.ylabel('Mean Cosine Similarity')
        plt.legend()
        
        # 绘制TDA MZ-score分布图
        plt.subplot(2, 3, 4)
        plt.scatter([i for i in benign_indices], [mzscore_tda[i] for i in benign_indices], 
                label='Benign Clients', color='blue', marker='o')
        plt.scatter([i for i in malicious_indices], [mzscore_tda[i] for i in malicious_indices], 
                label='Malicious Clients', color='red', marker='x')
        plt.axhline(y=self.args.lambda_c, color='green', linestyle='-', label='Threshold')
        plt.title('TDA MZ-score Distribution')
        plt.xlabel('Client Index')
        plt.ylabel('MZ-score Value')
        plt.legend()

        # 绘制MPSA MZ-score分布图
        plt.subplot(2, 3, 5)
        plt.scatter([i for i in benign_indices], [mzscore_mpsa[i] for i in benign_indices], 
                label='Benign Clients', color='blue', marker='o')
        plt.scatter([i for i in malicious_indices], [mzscore_mpsa[i] for i in malicious_indices], 
                label='Malicious Clients', color='red', marker='x')
        plt.axhline(y=self.args.lambda_s, color='green', linestyle='-', label='Threshold')
        plt.title('MPSA MZ-score Distribution')
        plt.xlabel('Client Index')
        plt.ylabel('MZ-score Value')
        plt.legend()

        # 绘制与平均更新余弦相似度MZ-score分布图
        plt.subplot(2, 3, 6)
        plt.scatter([i for i in benign_indices], [mzscore_mean_cos[i] for i in benign_indices], 
                label='Benign Clients', color='blue', marker='o')
        plt.scatter([i for i in malicious_indices], [mzscore_mean_cos[i] for i in malicious_indices], 
                label='Malicious Clients', color='red', marker='x')
        # 使用阈值参数或默认值
        if hasattr(self.args, 'lambda_mean_cos'):
            plt.axhline(y=self.args.lambda_mean_cos, color='green', linestyle='-', label='Threshold')
        else:
            plt.axhline(y=2.0, color='green', linestyle='-', label='Default Threshold (2.0)')
        plt.title('Mean Cos MZ-score Distribution')
        plt.xlabel('Client Index')
        plt.ylabel('MZ-score Value')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'three_metrics_analysis.png'))
        plt.close()

        ######## Anomaly detection with MZ score (3 metrics) ########

        # 基于MPSA MZ-score的过滤
        benign_idx1 = set([i for i in range(num_chosen_clients)])
        benign_idx1 = benign_idx1.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_mpsa) < self.args.lambda_s)]))
        
        # 基于TDA MZ-score的过滤
        benign_idx2 = set([i for i in range(num_chosen_clients)])
        benign_idx2 = benign_idx2.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_tda) < self.args.lambda_c)]))

        # 基于与平均更新余弦相似度MZ-score的过滤
        benign_idx3 = set([i for i in range(num_chosen_clients)])
        if hasattr(self.args, 'lambda_mean_cos'):
            benign_idx3 = benign_idx3.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_mean_cos) < self.args.lambda_mean_cos)]))
        else:
            # 如果没有lambda_mean_cos，使用默认阈值
            benign_idx3 = benign_idx3.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_mean_cos) < 2.0)]))

        # 结合三个过滤条件
        benign_set = benign_idx1.intersection(benign_idx2).intersection(benign_idx3)
        
        benign_idx = list(benign_set)
        if len(benign_idx) == 0:
            return torch.zeros_like(local_updates[0])

        ######## Post-filtering model clipping ########
        
        # 修复：使用过滤后的benign_idx来选择更新
        benign_updates = torch.stack([local_updates[i] for i in benign_idx], dim=0)
        
        updates_norm = torch.norm(benign_updates, dim=1).reshape((-1, 1))
        norm_clip = updates_norm.median(dim=0)[0].item()
        
        # 对良性更新进行裁剪
        updates_norm_clipped = torch.clamp(updates_norm, 0, norm_clip, out=None)
        benign_updates = (benign_updates/updates_norm)*updates_norm_clipped

        # 计算TPR和FPR
        correct = 0
        for idx in benign_idx:
            if idx >= len(malicious_id):
                correct += 1

        TPR = correct / len(benign_id)

        if len(malicious_id) == 0:
            FPR = 0
        else:
            wrong = 0
            for idx in benign_idx:
                if idx < len(malicious_id):
                    wrong += 1
            FPR = wrong / len(malicious_id)

        logging.info('benign update index:   %s' % str(benign_id))
        logging.info('selected update index: %s' % str(benign_idx))
        logging.info('FPR:       %.4f'  % FPR)
        logging.info('TPR:       %.4f' % TPR)

        # 构建最终的聚合字典 - 修复索引问题
        current_dict = {}
        for i, idx in enumerate(benign_idx):
            current_dict[chosen_clients[idx]] = benign_updates[i]  # 使用i而不是idx作为benign_updates的索引

        aggregated_update = self.agg_avg(current_dict)
        return aggregated_update

    def calculate_layer_sensitivity_difference(self, agent_updates_dict, flat_global_model, global_model):

        """计算客户端底层和顶层参数更新的相似度差异"""
        # 准备数据
        local_updates = []
        benign_id = []
        malicious_id = []

        for _id, update in agent_updates_dict.items():
            local_updates.append(update)
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        chosen_clients = malicious_id + benign_id
        inter_model_updates = torch.stack(local_updates, dim=0)
        
        # 获取模型结构信息
        model_dict = vector_to_name_param(flat_global_model, copy.deepcopy(global_model.state_dict()))
        
        # 分类参数到底层和顶层
        bottom_layers = []
        top_layers = []
        
        # 使用现有的分类逻辑
        for name in model_dict:
            if 'conv' in name.lower() and '0' in name.split('.')[-1]:
                bottom_layers.append(name)
            elif 'fc' in name.lower() or 'classifier' in name.lower():
                top_layers.append(name)
        
        # 如果分类不均匀，根据总参数数量重新分配
        if len(bottom_layers) == 0 or len(top_layers) == 0:
            all_layers = list(model_dict.keys())
            total_layers = len(all_layers)
            bottom_layers = all_layers[:total_layers//3]
            top_layers = all_layers[2*total_layers//3:]
        
        # 提取底层和顶层参数索引
        bottom_indices = []
        top_indices = []
        
        # 提取底层参数索引
        for name in bottom_layers:
            param_size = model_dict[name].numel()
            start_idx = 0
            for n in model_dict:
                if n == name:
                    break
                start_idx += model_dict[n].numel()
            end_idx = start_idx + param_size
            bottom_indices.extend(list(range(start_idx, end_idx)))
        
        # 提取顶层参数索引
        for name in top_layers:
            param_size = model_dict[name].numel()
            start_idx = 0
            for n in model_dict:
                if n == name:
                    break
                start_idx += model_dict[n].numel()
            end_idx = start_idx + param_size
            top_indices.extend(list(range(start_idx, end_idx)))
        
        bottom_indices = torch.tensor(bottom_indices)
        top_indices = torch.tensor(top_indices)
        
        # 计算每个客户端底层和顶层参数更新的相似度
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        lsd_scores = []
        client_types = []
        
        for i, update in enumerate(inter_model_updates):
            # 计算底层参数相似度
            bottom_similarity = cos(update[bottom_indices], flat_global_model[bottom_indices]).item()
            
            # 计算顶层参数相似度
            top_similarity = cos(update[top_indices], flat_global_model[top_indices]).item()
            
            # 计算层级敏感度差异
            lsd = bottom_similarity - top_similarity
            lsd_scores.append(lsd)
            
            # 记录客户端类型
            if i < len(malicious_id):
                client_types.append('Malicious')
            else:
                client_types.append('Benign')
        
        # 计算LSD的MZ-score
        lsd_std = np.std(lsd_scores)
        lsd_med = np.median(lsd_scores)
        mzscore_lsd = [np.abs(val - lsd_med) / lsd_std for val in lsd_scores]
        
        # 记录分析结果
        logging.info('LSD: %s' % [round(i, 4) for i in lsd_scores])
        logging.info('MZ-score of LSD: %s' % [round(i, 4) for i in mzscore_lsd])
        
        # 绘制LSD分布图
        plt.figure(figsize=(10, 6))
        benign_indices = [i for i, t in enumerate(client_types) if t == 'Benign']
        malicious_indices = [i for i, t in enumerate(client_types) if t == 'Malicious']
        
        plt.scatter([i for i in benign_indices], [lsd_scores[i] for i in benign_indices], 
                label='良性客户端', color='blue', marker='o')
        plt.scatter([i for i in malicious_indices], [lsd_scores[i] for i in malicious_indices], 
                label='恶意客户端', color='red', marker='x')
        plt.axhline(y=lsd_med, color='green', linestyle='-', label='中位数')
        plt.title('层级敏感度差异(LSD)分布')
        plt.xlabel('客户端索引')
        plt.ylabel('LSD值')
        plt.legend()
        
        # 保存图表
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'agg_alignins_layer_lsd_v1')
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'lsd_analysis.png'))
        plt.close()
        
        return lsd_scores, mzscore_lsd
    
    def agg_alignins_g_v(self, agent_updates_dict, flat_global_model):
        local_updates = []
        benign_id = []
        malicious_id = []

        for _id, update in agent_updates_dict.items():
            local_updates.append(update)
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        chosen_clients = malicious_id + benign_id
        num_chosen_clients = len(malicious_id + benign_id)
        inter_model_updates = torch.stack(local_updates, dim=0)

        tda_list = []
        mpsa_list = []
        grad_norm_list = [] # 新增：存储梯度L2范数

        major_sign = torch.sign(torch.sum(torch.sign(inter_model_updates), dim=0))
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        for i in range(len(inter_model_updates)):
            _, init_indices = torch.topk(torch.abs(inter_model_updates[i]), int(len(inter_model_updates[i]) * self.args.sparsity))

            mpsa_list.append((torch.sum(torch.sign(inter_model_updates[i][init_indices]) == major_sign[init_indices]) / torch.numel(inter_model_updates[i][init_indices])).item())
    
            tda_list.append(cos(inter_model_updates[i], flat_global_model).item())
            grad_norm_list.append(torch.norm(inter_model_updates[i]).item()) # 计算L2范数


        logging.info('TDA: %s' % [round(i, 4) for i in tda_list])
        logging.info('MPSA: %s' % [round(i, 4) for i in mpsa_list])
        logging.info('Grad Norm: %s' % [round(i, 4) for i in grad_norm_list]) # 新增日志


        ######## MZ-score calculation ########
        mpsa_std = np.std(mpsa_list)
        mpsa_med = np.median(mpsa_list)

        mzscore_mpsa = []
        for i in range(len(mpsa_list)):
            mzscore_mpsa.append(np.abs(mpsa_list[i] - mpsa_med) / mpsa_std)

        logging.info('MZ-score of MPSA: %s' % [round(i, 4) for i in mzscore_mpsa])
        
        tda_std = np.std(tda_list)
        tda_med = np.median(tda_list)
        mzscore_tda = []
        for i in range(len(tda_list)):
            mzscore_tda.append(np.abs(tda_list[i] - tda_med) / tda_std)

        logging.info('MZ-score of TDA: %s' % [round(i, 4) for i in mzscore_tda])

        # 新增：梯度范数的MZ-score计算
        grad_norm_std = np.std(grad_norm_list)
        grad_norm_med = np.median(grad_norm_list)
        mzscore_grad_norm = []
        for i in range(len(grad_norm_list)):
            mzscore_grad_norm.append(np.abs(grad_norm_list[i] - grad_norm_med) / grad_norm_std)
        logging.info('MZ-score of Grad Norm: %s' % [round(i, 4) for i in mzscore_grad_norm])
        
        ######## 添加可视化功能 ########
        import matplotlib.pyplot as plt
        import os
        
        # 确保输出目录存在
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'visualization_non_idd_g_v_badnet_beta_0.3')
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建客户端类型列表
        client_types = []
        for i in range(num_chosen_clients):
            if i < len(malicious_id):
                client_types.append('Malicious')
            else:
                client_types.append('Benign')
        # 获取良性和恶意客户端的索引
        benign_indices = [i for i, t in enumerate(client_types) if t == 'Benign']
        malicious_indices = [i for i, t in enumerate(client_types) if t == 'Malicious']
        
        # 创建图表
        plt.figure(figsize=(15, 10))
        
        # 绘制TDA分布图
        plt.subplot(2, 2, 1)
        plt.scatter([i for i in benign_indices], [tda_list[i] for i in benign_indices], 
                label='Benign Clients', color='blue', marker='o')
        plt.scatter([i for i in malicious_indices], [tda_list[i] for i in malicious_indices], 
                label='Malicious Clients', color='red', marker='x')
        plt.axhline(y=tda_med, color='green', linestyle='-', label='Median')
        plt.title('TDA Distribution')
        plt.xlabel('Client Index')
        plt.ylabel('TDA Value')
        plt.legend()
        
        # 绘制MPSA分布图
        plt.subplot(2, 2, 2)
        plt.scatter([i for i in benign_indices], [mpsa_list[i] for i in benign_indices], 
                label='Benign Clients', color='blue', marker='o')
        plt.scatter([i for i in malicious_indices], [mpsa_list[i] for i in malicious_indices], 
                label='Malicious Clients', color='red', marker='x')
        plt.axhline(y=mpsa_med, color='green', linestyle='-', label='Median')
        plt.title('MPSA Distribution')
        plt.xlabel('Client Index')
        plt.ylabel('MPSA Value')
        plt.legend()
        
        # 绘制TDA MZ-score分布图
        plt.subplot(2, 2, 3)
        plt.scatter([i for i in benign_indices], [mzscore_tda[i] for i in benign_indices], 
                label='Benign Clients', color='blue', marker='o')
        plt.scatter([i for i in malicious_indices], [mzscore_tda[i] for i in malicious_indices], 
                label='Malicious Clients', color='red', marker='x')
        plt.axhline(y=self.args.lambda_c, color='green', linestyle='-', label='Threshold')
        plt.title('TDA MZ-score Distribution')
        plt.xlabel('Client Index')
        plt.ylabel('MZ-score Value')
        plt.legend()
    
        # 绘制MPSA MZ-score分布图
        plt.subplot(2, 2, 4)
        plt.scatter([i for i in benign_indices], [mzscore_mpsa[i] for i in benign_indices], 
                label='Benign Clients', color='blue', marker='o')
        plt.scatter([i for i in malicious_indices], [mzscore_mpsa[i] for i in malicious_indices], 
                label='Malicious Clients', color='red', marker='x')
        plt.axhline(y=self.args.lambda_s, color='green', linestyle='-', label='Threshold')
        plt.title('MPSA MZ-score Distribution')
        plt.xlabel('Client Index')
        plt.ylabel('MZ-score Value')
        plt.legend()

        # 新增：绘制梯度范数MZ-score分布图
        plt.figure(figsize=(7.5, 5)) # 单独绘制，避免子图过多
        plt.scatter([i for i in benign_indices], [mzscore_grad_norm[i] for i in benign_indices], 
                label='Benign Clients', color='blue', marker='o')
        plt.scatter([i for i in malicious_indices], [mzscore_grad_norm[i] for i in malicious_indices], 
                label='Malicious Clients', color='red', marker='x')
        # 假设有一个新的阈值参数，例如 self.args.lambda_g
        # 如果没有，可以先用一个默认值或者根据实际情况调整
        if hasattr(self.args, 'lambda_g'):
            plt.axhline(y=self.args.lambda_g, color='green', linestyle='-', label='Threshold')
        else:
            # 临时使用一个默认值，或者根据数据分布动态确定
            plt.axhline(y=2.0, color='green', linestyle='-', label='Default Threshold (2.0)') 
        plt.title('Grad Norm MZ-score Distribution')
        plt.xlabel('Client Index')
        plt.ylabel('MZ-score Value')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'grad_norm_mzscore_analysis.png'))
        plt.close()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metrics_analysis.png'))
        plt.close()

        ######## Anomaly detection with MZ score ########

        benign_idx1 = set([i for i in range(num_chosen_clients)])
        benign_idx1 = benign_idx1.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_mpsa) < self.args.lambda_s)]))
        benign_idx2 = set([i for i in range(num_chosen_clients)])
        benign_idx2 = benign_idx2.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_tda) < self.args.lambda_c)]))

        # 新增：基于梯度范数MZ-score的过滤
        benign_idx3 = set([i for i in range(num_chosen_clients)])
        if hasattr(self.args, 'lambda_g'):
            benign_idx3 = benign_idx3.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_grad_norm) < self.args.lambda_g)]))
        else:
            # 如果没有lambda_g，可以考虑不使用此过滤，或者使用一个默认阈值
            benign_idx3 = benign_idx3.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_grad_norm) < 2.0)])) # 临时默认阈值

        benign_set = benign_idx2.intersection(benign_idx1).intersection(benign_idx3) # 结合所有过滤条件
        
        benign_idx = list(benign_set)
        if len(benign_idx) == 0:
            return torch.zeros_like(local_updates[0])

        benign_updates = torch.stack([local_updates[i] for i in benign_idx], dim=0)

        ######## Post-filtering model clipping ########
        
        updates_norm = torch.norm(benign_updates, dim=1).reshape((-1, 1))
        norm_clip = updates_norm.median(dim=0)[0].item()
        benign_updates = torch.stack(local_updates, dim=0)
        updates_norm = torch.norm(benign_updates, dim=1).reshape((-1, 1))
        updates_norm_clipped = torch.clamp(updates_norm, 0, norm_clip, out=None)
        # del grad_norm
        
        benign_updates = (benign_updates/updates_norm)*updates_norm_clipped

        correct = 0
        for idx in benign_idx:
            if idx >= len(malicious_id):
                correct += 1

        TPR = correct / len(benign_id)

        if len(malicious_id) == 0:
            FPR = 0
        else:
            wrong = 0
            for idx in benign_idx:
                if idx < len(malicious_id):
                    wrong += 1
            FPR = wrong / len(malicious_id)

        logging.info('benign update index:   %s' % str(benign_id))
        logging.info('selected update index: %s' % str(benign_idx))

        logging.info('FPR:       %.4f'  % FPR)
        logging.info('TPR:       %.4f' % TPR)

        current_dict = {}
        for idx in benign_idx:
            current_dict[chosen_clients[idx]] = benign_updates[idx]

        aggregated_update = self.agg_avg(current_dict)
        return aggregated_update



    def agg_alignins_g_v2(self, agent_updates_dict, flat_global_model, current_round=None):       
        local_updates = []
        benign_id = []
        malicious_id = []

        for _id, update in agent_updates_dict.items():
            local_updates.append(update)
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        chosen_clients = malicious_id + benign_id
        num_chosen_clients = len(malicious_id + benign_id)
        inter_model_updates = torch.stack(local_updates, dim=0)

        tda_list = []
        mpsa_list = []
        grad_norm_list = [] # 存储梯度L2范数
        mean_cos_list = [] # 新增：存储与平均更新的余弦相似度

        major_sign = torch.sign(torch.sum(torch.sign(inter_model_updates), dim=0))
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

        # 计算所有客户端更新的平均值
        mean_update = torch.mean(inter_model_updates, dim=0)

        for i in range(len(inter_model_updates)):
            _, init_indices = torch.topk(torch.abs(inter_model_updates[i]), int(len(inter_model_updates[i]) * self.args.sparsity))

            mpsa_list.append((torch.sum(torch.sign(inter_model_updates[i][init_indices]) == major_sign[init_indices]) / torch.numel(inter_model_updates[i][init_indices])).item())
    
            tda_list.append(cos(inter_model_updates[i], flat_global_model).item())
            grad_norm_list.append(torch.norm(inter_model_updates[i]).item()) # 计算L2范数
            mean_cos_list.append(cos(inter_model_updates[i], mean_update).item()) # 计算与平均更新的余弦相似度


        logging.info('TDA: %s' % [round(i, 4) for i in tda_list])
        logging.info('MPSA: %s' % [round(i, 4) for i in mpsa_list])
        logging.info('Grad Norm: %s' % [round(i, 4) for i in grad_norm_list])
        logging.info('Mean Cos: %s' % [round(i, 4) for i in mean_cos_list]) # 新增日志


        ######## MZ-score calculation ########
        mpsa_std = np.std(mpsa_list)
        mpsa_med = np.median(mpsa_list)

        mzscore_mpsa = []
        for i in range(len(mpsa_list)):
            mzscore_mpsa.append(np.abs(mpsa_list[i] - mpsa_med) / mpsa_std)

        logging.info('MZ-score of MPSA: %s' % [round(i, 4) for i in mzscore_mpsa])
        
        tda_std = np.std(tda_list)
        tda_med = np.median(tda_list)
        mzscore_tda = []
        for i in range(len(tda_list)):
            mzscore_tda.append(np.abs(tda_list[i] - tda_med) / tda_std)

        logging.info('MZ-score of TDA: %s' % [round(i, 4) for i in mzscore_tda])

        # 梯度范数的MZ-score计算
        grad_norm_std = np.std(grad_norm_list)
        grad_norm_med = np.median(grad_norm_list)
        mzscore_grad_norm = []
        for i in range(len(grad_norm_list)):
            mzscore_grad_norm.append(np.abs(grad_norm_list[i] - grad_norm_med) / grad_norm_std)
        logging.info('MZ-score of Grad Norm: %s' % [round(i, 4) for i in mzscore_grad_norm])

        # 新增：与平均更新余弦相似度的MZ-score计算
        mean_cos_std = np.std(mean_cos_list)
        mean_cos_med = np.median(mean_cos_list)
        mzscore_mean_cos = []
        for i in range(len(mean_cos_list)):
            # 注意：余弦相似度越小（越接近-1），可能越异常，所以这里计算的是与中位数的绝对差
            mzscore_mean_cos.append(np.abs(mean_cos_list[i] - mean_cos_med) / mean_cos_std)
        logging.info('MZ-score of Mean Cos: %s' % [round(i, 4) for i in mzscore_mean_cos]) # 新增日志
        
        ######## 添加可视化功能 ########
        import matplotlib.pyplot as plt
        import os
        import datetime
        
        # 只有在提供了current_round参数且current_round是10的倍数时才保存图表
        if current_round is not None and current_round % 10 == 0:
            # 获取当前时间作为文件夹名称的一部分
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 确保输出目录存在，使用时间戳创建唯一的文件夹
            base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'visualization_results_4p')
            output_dir = os.path.join(base_dir, f'round_{current_round}_{current_time}')
            os.makedirs(output_dir, exist_ok=True)
        if current_round is not None and current_round % 10 == 0:
            # 创建客户端类型列表
            client_types = []
            for i in range(num_chosen_clients):
                if i < len(malicious_id):
                    client_types.append('Malicious')
                else:
                    client_types.append('Benign')
            # 获取良性和恶意客户端的索引
            benign_indices = [i for i, t in enumerate(client_types) if t == 'Benign']
            malicious_indices = [i for i, t in enumerate(client_types) if t == 'Malicious']
            
            # 创建图表
            plt.figure(figsize=(15, 10))
            
            # 绘制TDA分布图
            plt.subplot(2, 2, 1)
            plt.scatter([i for i in benign_indices], [tda_list[i] for i in benign_indices], 
                    label='Benign Clients', color='blue', marker='o')
            plt.scatter([i for i in malicious_indices], [tda_list[i] for i in malicious_indices], 
                    label='Malicious Clients', color='red', marker='x')
            plt.axhline(y=tda_med, color='green', linestyle='-', label='Median')
            plt.title('TDA Distribution')
            plt.xlabel('Client Index')
            plt.ylabel('TDA Value')
            plt.legend()
            
            # 绘制MPSA分布图
            plt.subplot(2, 2, 2)
            plt.scatter([i for i in benign_indices], [mpsa_list[i] for i in benign_indices], 
                    label='Benign Clients', color='blue', marker='o')
            plt.scatter([i for i in malicious_indices], [mpsa_list[i] for i in malicious_indices], 
                    label='Malicious Clients', color='red', marker='x')
            plt.axhline(y=mpsa_med, color='green', linestyle='-', label='Median')
            plt.title('MPSA Distribution')
            plt.xlabel('Client Index')
            plt.ylabel('MPSA Value')
            plt.legend()
            
            # 绘制TDA MZ-score分布图
            plt.subplot(2, 2, 3)
            plt.scatter([i for i in benign_indices], [mzscore_tda[i] for i in benign_indices], 
                    label='Benign Clients', color='blue', marker='o')
            plt.scatter([i for i in malicious_indices], [mzscore_tda[i] for i in malicious_indices], 
                    label='Malicious Clients', color='red', marker='x')
            plt.axhline(y=self.args.lambda_c, color='green', linestyle='-', label='Threshold')
            plt.title('TDA MZ-score Distribution')
            plt.xlabel('Client Index')
            plt.ylabel('MZ-score Value')
            plt.legend()

            # 绘制MPSA MZ-score分布图
            plt.subplot(2, 2, 4)
            plt.scatter([i for i in benign_indices], [mzscore_mpsa[i] for i in benign_indices], 
                    label='Benign Clients', color='blue', marker='o')
            plt.scatter([i for i in malicious_indices], [mzscore_mpsa[i] for i in malicious_indices], 
                    label='Malicious Clients', color='red', marker='x')
            plt.axhline(y=self.args.lambda_s, color='green', linestyle='-', label='Threshold')
            plt.title('MPSA MZ-score Distribution')
            plt.xlabel('Client Index')
            plt.ylabel('MZ-score Value')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'metrics_analysis.png'))
            plt.close()

            # 绘制梯度范数MZ-score分布图
            plt.figure(figsize=(7.5, 5)) # 单独绘制，避免子图过多
            plt.scatter([i for i in benign_indices], [mzscore_grad_norm[i] for i in benign_indices], 
                    label='Benign Clients', color='blue', marker='o')
            plt.scatter([i for i in malicious_indices], [mzscore_grad_norm[i] for i in malicious_indices], 
                    label='Malicious Clients', color='red', marker='x')
            # 假设有一个新的阈值参数，例如 self.args.lambda_g
            # 如果没有，可以先用一个默认值或者根据实际情况调整
            if hasattr(self.args, 'lambda_g'):
                plt.axhline(y=self.args.lambda_g, color='green', linestyle='-', label='Threshold')
            else:
                # 临时使用一个默认值，或者根据数据分布动态确定
                plt.axhline(y=2.0, color='green', linestyle='-', label='Default Threshold (2.0)') 
            plt.title('Grad Norm MZ-score Distribution')
            plt.xlabel('Client Index')
            plt.ylabel('MZ-score Value')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'grad_norm_mzscore_analysis.png'))
            plt.close()

            # 新增：绘制与平均更新余弦相似度MZ-score分布图
            plt.figure(figsize=(7.5, 5)) # 单独绘制
            plt.scatter([i for i in benign_indices], [mzscore_mean_cos[i] for i in benign_indices], 
                    label='Benign Clients', color='blue', marker='o')
            plt.scatter([i for i in malicious_indices], [mzscore_mean_cos[i] for i in malicious_indices], 
                    label='Malicious Clients', color='red', marker='x')
            # 假设有一个新的阈值参数，例如 self.args.lambda_mean_cos
            if hasattr(self.args, 'lambda_mean_cos'):
                plt.axhline(y=self.args.lambda_mean_cos, color='green', linestyle='-', label='Threshold')
            else:
                # 临时使用一个默认值
                plt.axhline(y=2.0, color='green', linestyle='-', label='Default Threshold (2.0)') 
            plt.title('Mean Cos MZ-score Distribution')
            plt.xlabel('Client Index')
            plt.ylabel('MZ-score Value')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'mean_cos_mzscore_analysis.png'))
            plt.close()
            
            # 绘制Grad Norm分布图
            plt.figure(figsize=(7.5, 5))
            plt.scatter([i for i in benign_indices], [grad_norm_list[i] for i in benign_indices], 
                    label='Benign Clients', color='blue', marker='o')
            plt.scatter([i for i in malicious_indices], [grad_norm_list[i] for i in malicious_indices], 
                    label='Malicious Clients', color='red', marker='x')
            plt.axhline(y=grad_norm_med, color='green', linestyle='-', label='Median')
            plt.title('Grad Norm Distribution')
            plt.xlabel('Client Index')
            plt.ylabel('Grad Norm Value')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'grad_norm_analysis.png'))
            plt.close()
            
            # 绘制Mean Cos分布图
            plt.figure(figsize=(7.5, 5))
            plt.scatter([i for i in benign_indices], [mean_cos_list[i] for i in benign_indices], 
                    label='Benign Clients', color='blue', marker='o')
            plt.scatter([i for i in malicious_indices], [mean_cos_list[i] for i in malicious_indices], 
                    label='Malicious Clients', color='red', marker='x')
            plt.axhline(y=mean_cos_med, color='green', linestyle='-', label='Median')
            plt.title('Mean Cos Distribution')
            plt.xlabel('Client Index')
            plt.ylabel('Mean Cos Value')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'mean_cos_analysis.png'))
            plt.close()
            
            logging.info(f"保存了第 {current_round} 轮的可视化结果到 {output_dir}")

        ######## Anomaly detection with MZ score ########

        benign_idx1 = set([i for i in range(num_chosen_clients)])
        benign_idx1 = benign_idx1.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_mpsa) < self.args.lambda_s)]))
        benign_idx2 = set([i for i in range(num_chosen_clients)])
        benign_idx2 = benign_idx2.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_tda) < self.args.lambda_c)]))

        # 基于梯度范数MZ-score的过滤
        benign_idx3 = set([i for i in range(num_chosen_clients)])
        if hasattr(self.args, 'lambda_g'):
            benign_idx3 = benign_idx3.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_grad_norm) < self.args.lambda_g)]))
        else:
            # 如果没有lambda_g，可以考虑不使用此过滤，或者使用一个默认阈值
            benign_idx3 = benign_idx3.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_grad_norm) < 1.5)])) # 临时默认阈值

        # 新增：基于与平均更新余弦相似度MZ-score的过滤
        benign_idx4 = set([i for i in range(num_chosen_clients)])
        if hasattr(self.args, 'lambda_mean_cos'):
            # 注意：这里判断的是MZ-score，MZ-score越大表示越偏离中位数，所以仍然是小于阈值
            benign_idx4 = benign_idx4.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_mean_cos) < self.args.lambda_mean_cos)]))
        else:
            # 如果没有lambda_mean_cos，可以考虑不使用此过滤，或者使用一个默认阈值
            benign_idx4 = benign_idx4.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_mean_cos) < 1.5)])) # 临时默认阈值

        benign_set = benign_idx2.intersection(benign_idx1).intersection(benign_idx3).intersection(benign_idx4) # 结合所有过滤条件
        
        benign_idx = list(benign_set)
        if len(benign_idx) == 0:
            return torch.zeros_like(local_updates[0])

        benign_updates = torch.stack([local_updates[i] for i in benign_idx], dim=0)

        ######## Post-filtering model clipping ########
        
        updates_norm = torch.norm(benign_updates, dim=1).reshape((-1, 1))
        norm_clip = updates_norm.median(dim=0)[0].item()
        benign_updates = torch.stack(local_updates, dim=0)
        updates_norm = torch.norm(benign_updates, dim=1).reshape((-1, 1))
        updates_norm_clipped = torch.clamp(updates_norm, 0, norm_clip, out=None)
        # del grad_norm
        
        benign_updates = (benign_updates/updates_norm)*updates_norm_clipped

        correct = 0
        for idx in benign_idx:
            if idx >= len(malicious_id):
                correct += 1

        TPR = correct / len(benign_id)

        if len(malicious_id) == 0:
            FPR = 0
        else:
            wrong = 0
            for idx in benign_idx:
                if idx < len(malicious_id):
                    wrong += 1
            FPR = wrong / len(malicious_id)

        logging.info('benign update index:   %s' % str(benign_id))
        logging.info('selected update index: %s' % str(benign_idx))

        logging.info('FPR:       %.4f'  % FPR)
        logging.info('TPR:       %.4f' % TPR)

        current_dict = {}
        for idx in benign_idx:
            current_dict[chosen_clients[idx]] = benign_updates[idx]

        aggregated_update = self.agg_avg(current_dict)
        return aggregated_update

    def agg_alignins_g_v2_onepic(self, agent_updates_dict, flat_global_model, current_round=None):
        local_updates = []
        benign_id = []
        malicious_id = []

        for _id, update in agent_updates_dict.items():
            local_updates.append(update)
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        chosen_clients = malicious_id + benign_id
        num_chosen_clients = len(malicious_id + benign_id)
        inter_model_updates = torch.stack(local_updates, dim=0)

        tda_list = []
        mpsa_list = []
        grad_norm_list = [] # 存储梯度L2范数
        mean_cos_list = [] # 新增：存储与平均更新的余弦相似度

        major_sign = torch.sign(torch.sum(torch.sign(inter_model_updates), dim=0))
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

        # 计算所有客户端更新的平均值
        mean_update = torch.mean(inter_model_updates, dim=0)

        for i in range(len(inter_model_updates)):
            _, init_indices = torch.topk(torch.abs(inter_model_updates[i]), int(len(inter_model_updates[i]) * self.args.sparsity))

            mpsa_list.append((torch.sum(torch.sign(inter_model_updates[i][init_indices]) == major_sign[init_indices]) / torch.numel(inter_model_updates[i][init_indices])).item())
    
            tda_list.append(cos(inter_model_updates[i], flat_global_model).item())
            grad_norm_list.append(torch.norm(inter_model_updates[i]).item()) # 计算L2范数
            mean_cos_list.append(cos(inter_model_updates[i], mean_update).item()) # 计算与平均更新的余弦相似度


        logging.info('TDA: %s' % [round(i, 4) for i in tda_list])
        logging.info('MPSA: %s' % [round(i, 4) for i in mpsa_list])
        logging.info('Grad Norm: %s' % [round(i, 4) for i in grad_norm_list])
        logging.info('Mean Cos: %s' % [round(i, 4) for i in mean_cos_list]) # 新增日志


        ######## MZ-score calculation ########
        mpsa_std = np.std(mpsa_list)
        mpsa_med = np.median(mpsa_list)

        mzscore_mpsa = []
        for i in range(len(mpsa_list)):
            mzscore_mpsa.append(np.abs(mpsa_list[i] - mpsa_med) / mpsa_std)

        logging.info('MZ-score of MPSA: %s' % [round(i, 4) for i in mzscore_mpsa])
        
        tda_std = np.std(tda_list)
        tda_med = np.median(tda_list)
        mzscore_tda = []
        for i in range(len(tda_list)):
            mzscore_tda.append(np.abs(tda_list[i] - tda_med) / tda_std)

        logging.info('MZ-score of TDA: %s' % [round(i, 4) for i in mzscore_tda])

        # 梯度范数的MZ-score计算
        grad_norm_std = np.std(grad_norm_list)
        grad_norm_med = np.median(grad_norm_list)
        mzscore_grad_norm = []
        for i in range(len(grad_norm_list)):
            mzscore_grad_norm.append(np.abs(grad_norm_list[i] - grad_norm_med) / grad_norm_std)
        logging.info('MZ-score of Grad Norm: %s' % [round(i, 4) for i in mzscore_grad_norm])

        # 新增：与平均更新余弦相似度的MZ-score计算
        mean_cos_std = np.std(mean_cos_list)
        mean_cos_med = np.median(mean_cos_list)
        mzscore_mean_cos = []
        for i in range(len(mean_cos_list)):
            # 注意：余弦相似度越小（越接近-1），可能越异常，所以这里计算的是与中位数的绝对差
            mzscore_mean_cos.append(np.abs(mean_cos_list[i] - mean_cos_med) / mean_cos_std)
        logging.info('MZ-score of Mean Cos: %s' % [round(i, 4) for i in mzscore_mean_cos]) # 新增日志
        
        ######## 添加可视化功能 ########
        import matplotlib.pyplot as plt
        import os
        import datetime
        
        # 只有在提供了current_round参数且current_round是10的倍数时才保存图表
        if current_round is not None and current_round % 10 == 0:
            # 获取当前时间作为文件夹名称的一部分
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 确保输出目录存在，使用时间戳创建唯一的文件夹
            base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'visualization_results_onepic_iid')
            output_dir = os.path.join(base_dir, f'round_{current_round}_{current_time}')
            os.makedirs(output_dir, exist_ok=True)
        if current_round is not None and current_round % 10 == 0:
            # 创建客户端类型列表
            client_types = []
            for i in range(num_chosen_clients):
                if i < len(malicious_id):
                    client_types.append('Malicious')
                else:
                    client_types.append('Benign')
            # 获取良性和恶意客户端的索引
            benign_indices = [i for i, t in enumerate(client_types) if t == 'Benign']
            malicious_indices = [i for i, t in enumerate(client_types) if t == 'Malicious']
            
            # 计算所有客户端之间的相似度矩阵
            similarity_matrix = np.zeros((num_chosen_clients, num_chosen_clients))
            for i in range(num_chosen_clients):
                for j in range(num_chosen_clients):
                    if i == j:
                        similarity_matrix[i][j] = 1.0
                    else:
                        sim = cos(inter_model_updates[i], inter_model_updates[j]).item()
                        similarity_matrix[i][j] = sim
            
            # 创建一个大图表，包含所有指标和相似度热力图
            plt.figure(figsize=(24, 20))  # 进一步增大图表尺寸以容纳更多子图
            
            # 第一行：原始指标分布
            # 绘制TDA分布图
            plt.subplot(4, 2, 1)
            plt.scatter([i for i in benign_indices], [tda_list[i] for i in benign_indices], 
                    label='Benign', color='blue', marker='o', s=30)
            plt.scatter([i for i in malicious_indices], [tda_list[i] for i in malicious_indices], 
                    label='Malicious', color='red', marker='x', s=40)
            plt.axhline(y=tda_med, color='green', linestyle='-', label='Median')
            plt.title('TDA Distribution')
            plt.xlabel('Client Index')
            plt.ylabel('TDA Value')
            plt.legend()
            
            # 绘制MPSA分布图
            plt.subplot(4, 2, 2)
            plt.scatter([i for i in benign_indices], [mpsa_list[i] for i in benign_indices], 
                    label='Benign', color='blue', marker='o', s=30)
            plt.scatter([i for i in malicious_indices], [mpsa_list[i] for i in malicious_indices], 
                    label='Malicious', color='red', marker='x', s=40)
            plt.axhline(y=mpsa_med, color='green', linestyle='-', label='Median')
            plt.title('MPSA Distribution')
            plt.xlabel('Client Index')
            plt.ylabel('MPSA Value')
            plt.legend()
            
            # 绘制Grad Norm分布图
            plt.subplot(4, 2, 3)
            plt.scatter([i for i in benign_indices], [grad_norm_list[i] for i in benign_indices], 
                    label='Benign', color='blue', marker='o', s=30)
            plt.scatter([i for i in malicious_indices], [grad_norm_list[i] for i in malicious_indices], 
                    label='Malicious', color='red', marker='x', s=40)
            plt.axhline(y=grad_norm_med, color='green', linestyle='-', label='Median')
            plt.title('Grad Norm Distribution')
            plt.xlabel('Client Index')
            plt.ylabel('Grad Norm Value')
            plt.legend()
            
            # 绘制Mean Cos分布图
            plt.subplot(4, 2, 4)
            plt.scatter([i for i in benign_indices], [mean_cos_list[i] for i in benign_indices], 
                    label='Benign', color='blue', marker='o', s=30)
            plt.scatter([i for i in malicious_indices], [mean_cos_list[i] for i in malicious_indices], 
                    label='Malicious', color='red', marker='x', s=40)
            plt.axhline(y=mean_cos_med, color='green', linestyle='-', label='Median')
            plt.title('Mean Cos Distribution')
            plt.xlabel('Client Index')
            plt.ylabel('Mean Cos Value')
            plt.legend()
            
            # 第二行：MZ-score分布
            # 绘制TDA MZ-score分布图
            plt.subplot(4, 2, 5)
            plt.scatter([i for i in benign_indices], [mzscore_tda[i] for i in benign_indices], 
                    label='Benign', color='blue', marker='o', s=30)
            plt.scatter([i for i in malicious_indices], [mzscore_tda[i] for i in malicious_indices], 
                    label='Malicious', color='red', marker='x', s=40)
            plt.axhline(y=self.args.lambda_c, color='green', linestyle='-', label='Threshold')
            plt.title('TDA MZ-score Distribution')
            plt.xlabel('Client Index')
            plt.ylabel('MZ-score Value')
            plt.legend()

            # 绘制MPSA MZ-score分布图
            plt.subplot(4, 2, 6)
            plt.scatter([i for i in benign_indices], [mzscore_mpsa[i] for i in benign_indices], 
                    label='Benign', color='blue', marker='o', s=30)
            plt.scatter([i for i in malicious_indices], [mzscore_mpsa[i] for i in malicious_indices], 
                    label='Malicious', color='red', marker='x', s=40)
            plt.axhline(y=self.args.lambda_s, color='green', linestyle='-', label='Threshold')
            plt.title('MPSA MZ-score Distribution')
            plt.xlabel('Client Index')
            plt.ylabel('MZ-score Value')
            plt.legend()
            
            # 绘制梯度范数MZ-score分布图
            plt.subplot(4, 2, 7)
            plt.scatter([i for i in benign_indices], [mzscore_grad_norm[i] for i in benign_indices], 
                    label='Benign', color='blue', marker='o', s=30)
            plt.scatter([i for i in malicious_indices], [mzscore_grad_norm[i] for i in malicious_indices], 
                    label='Malicious', color='red', marker='x', s=40)
            # 假设有一个新的阈值参数，例如 self.args.lambda_g
            if hasattr(self.args, 'lambda_g'):
                plt.axhline(y=self.args.lambda_g, color='green', linestyle='-', label='Threshold')
            else:
                # 临时使用一个默认值
                plt.axhline(y=2.0, color='green', linestyle='-', label='Default Threshold (2.0)') 
            plt.title('Grad Norm MZ-score Distribution')
            plt.xlabel('Client Index')
            plt.ylabel('MZ-score Value')
            plt.legend()

            # 绘制与平均更新余弦相似度MZ-score分布图
            plt.subplot(4, 2, 8)
            plt.scatter([i for i in benign_indices], [mzscore_mean_cos[i] for i in benign_indices], 
                    label='Benign', color='blue', marker='o', s=30)
            plt.scatter([i for i in malicious_indices], [mzscore_mean_cos[i] for i in malicious_indices], 
                    label='Malicious', color='red', marker='x', s=40)
            # 假设有一个新的阈值参数，例如 self.args.lambda_mean_cos
            if hasattr(self.args, 'lambda_mean_cos'):
                plt.axhline(y=self.args.lambda_mean_cos, color='green', linestyle='-', label='Threshold')
            else:
                # 临时使用一个默认值
                plt.axhline(y=2.0, color='green', linestyle='-', label='Default Threshold (2.0)') 
            plt.title('Mean Cos MZ-score Distribution')
            plt.xlabel('Client Index')
            plt.ylabel('MZ-score Value')
            plt.legend()
            
            # 添加总标题
            plt.suptitle(f'Round {current_round} - Client Metrics Analysis', fontsize=16)
            
            # 调整布局并保存
            plt.tight_layout(rect=[0, 0, 1, 0.97])  # 为总标题留出空间
            plt.savefig(os.path.join(output_dir, 'all_metrics_analysis.png'), dpi=300)
            plt.close()
            
            logging.info(f"保存了第 {current_round} 轮的可视化结果到 {output_dir}")

        ######## Anomaly detection with MZ score ########

        benign_idx1 = set([i for i in range(num_chosen_clients)])
        benign_idx1 = benign_idx1.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_mpsa) < self.args.lambda_s)]))
        benign_idx2 = set([i for i in range(num_chosen_clients)])
        benign_idx2 = benign_idx2.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_tda) < self.args.lambda_c)]))

        # 基于梯度范数MZ-score的过滤
        benign_idx3 = set([i for i in range(num_chosen_clients)])
        if hasattr(self.args, 'lambda_g'):
            benign_idx3 = benign_idx3.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_grad_norm) < self.args.lambda_g)]))
        else:
            # 如果没有lambda_g，可以考虑不使用此过滤，或者使用一个默认阈值
            benign_idx3 = benign_idx3.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_grad_norm) < 1.5)])) # 临时默认阈值

        # 新增：基于与平均更新余弦相似度MZ-score的过滤
        benign_idx4 = set([i for i in range(num_chosen_clients)])
        if hasattr(self.args, 'lambda_mean_cos'):
             # 注意：这里判断的是MZ-score，MZ-score越大表示越偏离中位数，所以仍然是小于阈值
            benign_idx4 = benign_idx4.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_mean_cos) < self.args.lambda_mean_cos)]))
        else:
            # 如果没有lambda_mean_cos，可以考虑不使用此过滤，或者使用一个默认阈值
            benign_idx4 = benign_idx4.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_mean_cos) < 1.5)])) # 临时默认阈值

        benign_set = benign_idx2.intersection(benign_idx1).intersection(benign_idx3).intersection(benign_idx4) # 结合所有过滤条件
        
        benign_idx = list(benign_set)
        if len(benign_idx) == 0:
            return torch.zeros_like(local_updates[0])

        benign_updates = torch.stack([local_updates[i] for i in benign_idx], dim=0)

        ######## Post-filtering model clipping ########
        
        updates_norm = torch.norm(benign_updates, dim=1).reshape((-1, 1))
        norm_clip = updates_norm.median(dim=0)[0].item()
        benign_updates = torch.stack(local_updates, dim=0)
        updates_norm = torch.norm(benign_updates, dim=1).reshape((-1, 1))
        updates_norm_clipped = torch.clamp(updates_norm, 0, norm_clip, out=None)
        # del grad_norm
        
        benign_updates = (benign_updates/updates_norm)*updates_norm_clipped

        correct = 0
        for idx in benign_idx:
            if idx >= len(malicious_id):
                correct += 1

        TPR = correct / len(benign_id)

        if len(malicious_id) == 0:
            FPR = 0
        else:
            wrong = 0
            for idx in benign_idx:
                if idx < len(malicious_id):
                    wrong += 1
            FPR = wrong / len(malicious_id)

        logging.info('benign update index:   %s' % str(benign_id))
        logging.info('selected update index: %s' % str(benign_idx))

        logging.info('FPR:       %.4f'  % FPR)
        logging.info('TPR:       %.4f' % TPR)

        current_dict = {}
        for idx in benign_idx:
            current_dict[chosen_clients[idx]] = benign_updates[idx]

        aggregated_update = self.agg_avg(current_dict)
        return aggregated_update

    def agg_alignins_fedup_hybrid(self, agent_updates_dict, flat_global_model, global_model, current_round=None):
        """
        改进的混合方法：结合AlignIns的多指标异常检测和FedUP的自适应剪枝策略
        
        改进策略：
        1. 使用更宽松的AlignIns检测阈值，将客户端分为三类：良性、可疑、恶意
        2. 对可疑客户端应用FedUP自适应剪枝，对恶意客户端直接排除
        3. 结合良性客户端和剪枝后的可疑客户端进行聚合
        4. 动态调整检测阈值以平衡clean accuracy和attack success ratio
        """
        local_updates = []
        benign_id = []
        malicious_id = []

        for _id, update in agent_updates_dict.items():
            local_updates.append(update)
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        chosen_clients = malicious_id + benign_id
        num_chosen_clients = len(malicious_id + benign_id)
        inter_model_updates = torch.stack(local_updates, dim=0)

        # ========== 第一阶段：AlignIns多指标异常检测 ==========
        tda_list = []
        mpsa_list = []
        grad_norm_list = []
        mean_cos_list = []

        major_sign = torch.sign(torch.sum(torch.sign(inter_model_updates), dim=0))
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        mean_update = torch.mean(inter_model_updates, dim=0)

        for i in range(len(inter_model_updates)):
            _, init_indices = torch.topk(torch.abs(inter_model_updates[i]), int(len(inter_model_updates[i]) * self.args.sparsity))
            mpsa_list.append((torch.sum(torch.sign(inter_model_updates[i][init_indices]) == major_sign[init_indices]) / torch.numel(inter_model_updates[i][init_indices])).item())
            tda_list.append(cos(inter_model_updates[i], flat_global_model).item())
            grad_norm_list.append(torch.norm(inter_model_updates[i]).item())
            mean_cos_list.append(cos(inter_model_updates[i], mean_update).item())

        logging.info('Hybrid Method - TDA: %s' % [round(i, 4) for i in tda_list])
        logging.info('Hybrid Method - MPSA: %s' % [round(i, 4) for i in mpsa_list])
        logging.info('Hybrid Method - Grad Norm: %s' % [round(i, 4) for i in grad_norm_list])
        logging.info('Hybrid Method - Mean Cos: %s' % [round(i, 4) for i in mean_cos_list])

        # MZ-score计算
        def calculate_mz_scores(values):
            std_val = np.std(values)
            med_val = np.median(values)
            return [np.abs(val - med_val) / std_val for val in values]

        mzscore_mpsa = calculate_mz_scores(mpsa_list)
        mzscore_tda = calculate_mz_scores(tda_list)
        mzscore_grad_norm = calculate_mz_scores(grad_norm_list)
        mzscore_mean_cos = calculate_mz_scores(mean_cos_list)

        logging.info('Hybrid Method - MZ-score MPSA: %s' % [round(i, 4) for i in mzscore_mpsa])
        logging.info('Hybrid Method - MZ-score TDA: %s' % [round(i, 4) for i in mzscore_tda])
        logging.info('Hybrid Method - MZ-score Grad Norm: %s' % [round(i, 4) for i in mzscore_grad_norm])
        logging.info('Hybrid Method - MZ-score Mean Cos: %s' % [round(i, 4) for i in mzscore_mean_cos])

        # 改进的三层检测策略
        # 第一层：严格阈值检测明确的良性客户端
        strict_lambda_s = self.args.lambda_s * 0.8  # 更严格的阈值
        strict_lambda_c = self.args.lambda_c * 0.8
        strict_lambda_g = getattr(self.args, 'lambda_g', 1.5) * 0.8
        strict_lambda_mean_cos = getattr(self.args, 'lambda_mean_cos', 1.5) * 0.8
        
        benign_idx1_strict = set([i for i in range(num_chosen_clients) if mzscore_mpsa[i] < strict_lambda_s])
        benign_idx2_strict = set([i for i in range(num_chosen_clients) if mzscore_tda[i] < strict_lambda_c])
        benign_idx3_strict = set([i for i in range(num_chosen_clients) if mzscore_grad_norm[i] < strict_lambda_g])
        benign_idx4_strict = set([i for i in range(num_chosen_clients) if mzscore_mean_cos[i] < strict_lambda_mean_cos])
        
        # 明确良性客户端（通过所有严格检测）
        clear_benign_set = benign_idx1_strict.intersection(benign_idx2_strict).intersection(benign_idx3_strict).intersection(benign_idx4_strict)
        
        # 第二层：标准阈值检测明确的恶意客户端
        loose_lambda_s = self.args.lambda_s * 1.0  # 使用标准阈值
        loose_lambda_c = self.args.lambda_c * 1.0
        loose_lambda_g = getattr(self.args, 'lambda_g', 1.5) * 1.0
        loose_lambda_mean_cos = getattr(self.args, 'lambda_mean_cos', 1.5) * 1.0
        
        # 恶意客户端：在任意一个指标上超过宽松阈值
        malicious_idx1 = set([i for i in range(num_chosen_clients) if mzscore_mpsa[i] > loose_lambda_s])
        malicious_idx2 = set([i for i in range(num_chosen_clients) if mzscore_tda[i] > loose_lambda_c])
        malicious_idx3 = set([i for i in range(num_chosen_clients) if mzscore_grad_norm[i] > loose_lambda_g])
        malicious_idx4 = set([i for i in range(num_chosen_clients) if mzscore_mean_cos[i] > loose_lambda_mean_cos])
        
        clear_malicious_set = malicious_idx1.union(malicious_idx2).union(malicious_idx3).union(malicious_idx4)
        
        # 第三层：可疑客户端（介于良性和恶意之间）
        suspicious_set = set(range(num_chosen_clients)) - clear_benign_set - clear_malicious_set
        
        logging.info('Hybrid Method - Clear benign clients: %s' % list(clear_benign_set))
        logging.info('Hybrid Method - Suspicious clients (for FedUP): %s' % list(suspicious_set))
        logging.info('Hybrid Method - Clear malicious clients (excluded): %s' % list(clear_malicious_set))

        # 计算客户端相似度矩阵（用于可视化）
        similarity_matrix = np.zeros((num_chosen_clients, num_chosen_clients))
        for i in range(num_chosen_clients):
            for j in range(num_chosen_clients):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    similarity_matrix[i, j] = cos(inter_model_updates[i], inter_model_updates[j]).item()

        ######## 添加可视化功能 ########
        import matplotlib.pyplot as plt
        import os
        import datetime
        
        # 只有在提供了current_round参数且current_round是10的倍数时才保存图表
        if current_round is not None and current_round % 10 == 0:
            # 获取当前时间作为文件夹名称的一部分
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 确保输出目录存在，使用时间戳创建唯一的文件夹
            base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'visualization_results_hybrid')
            output_dir = os.path.join(base_dir, f'round_{current_round}_{current_time}')
            os.makedirs(output_dir, exist_ok=True)
            
            # 创建客户端类型列表
            client_types = []
            for i in range(num_chosen_clients):
                if i < len(malicious_id):
                    client_types.append('Malicious')
                else:
                    client_types.append('Benign')
            
            # 获取良性和恶意客户端的索引
            benign_indices = [i for i, t in enumerate(client_types) if t == 'Benign']
            malicious_indices = [i for i, t in enumerate(client_types) if t == 'Malicious']
            
            # 创建一个大图表，包含所有8个指标
            plt.figure(figsize=(20, 16))  # 增大图表尺寸以容纳所有子图
            
            # 第一行：原始指标分布
            # 绘制TDA分布图
            plt.subplot(3, 3, 1)
            plt.scatter([i for i in benign_indices], [tda_list[i] for i in benign_indices], 
                    label='Benign', color='blue', marker='o', s=30)
            plt.scatter([i for i in malicious_indices], [tda_list[i] for i in malicious_indices], 
                    label='Malicious', color='red', marker='x', s=40)
            tda_med = np.median(tda_list)
            plt.axhline(y=tda_med, color='green', linestyle='-', label='Median')
            plt.title('TDA Distribution')
            plt.xlabel('Client Index')
            plt.ylabel('TDA Value')
            plt.legend()
            
            # 绘制MPSA分布图
            plt.subplot(3, 3, 2)
            plt.scatter([i for i in benign_indices], [mpsa_list[i] for i in benign_indices], 
                    label='Benign', color='blue', marker='o', s=30)
            plt.scatter([i for i in malicious_indices], [mpsa_list[i] for i in malicious_indices], 
                    label='Malicious', color='red', marker='x', s=40)
            mpsa_med = np.median(mpsa_list)
            plt.axhline(y=mpsa_med, color='green', linestyle='-', label='Median')
            plt.title('MPSA Distribution')
            plt.xlabel('Client Index')
            plt.ylabel('MPSA Value')
            plt.legend()
            
            # 绘制Grad Norm分布图
            plt.subplot(3, 3, 3)
            plt.scatter([i for i in benign_indices], [grad_norm_list[i] for i in benign_indices], 
                    label='Benign', color='blue', marker='o', s=30)
            plt.scatter([i for i in malicious_indices], [grad_norm_list[i] for i in malicious_indices], 
                    label='Malicious', color='red', marker='x', s=40)
            grad_norm_med = np.median(grad_norm_list)
            plt.axhline(y=grad_norm_med, color='green', linestyle='-', label='Median')
            plt.title('Grad Norm Distribution')
            plt.xlabel('Client Index')
            plt.ylabel('Grad Norm Value')
            plt.legend()
            
            # 第二行：MZ-score分布（添加严格阈值虚线）
            # 绘制TDA MZ-score分布图
            plt.subplot(3, 3, 4)
            plt.scatter([i for i in benign_indices], [mzscore_tda[i] for i in benign_indices], 
                    label='Benign', color='blue', marker='o', s=30)
            plt.scatter([i for i in malicious_indices], [mzscore_tda[i] for i in malicious_indices], 
                    label='Malicious', color='red', marker='x', s=40)
            plt.axhline(y=self.args.lambda_c, color='green', linestyle='-', label='Standard Threshold')
            plt.axhline(y=strict_lambda_c, color='orange', linestyle='--', label='Strict Threshold (Benign/Suspicious)')
            plt.title('TDA MZ-score Distribution')
            plt.xlabel('Client Index')
            plt.ylabel('MZ-score Value')
            plt.legend()

            # 绘制MPSA MZ-score分布图
            plt.subplot(3, 3, 5)
            plt.scatter([i for i in benign_indices], [mzscore_mpsa[i] for i in benign_indices], 
                    label='Benign', color='blue', marker='o', s=30)
            plt.scatter([i for i in malicious_indices], [mzscore_mpsa[i] for i in malicious_indices], 
                    label='Malicious', color='red', marker='x', s=40)
            plt.axhline(y=self.args.lambda_s, color='green', linestyle='-', label='Standard Threshold')
            plt.axhline(y=strict_lambda_s, color='orange', linestyle='--', label='Strict Threshold (Benign/Suspicious)')
            plt.title('MPSA MZ-score Distribution')
            plt.xlabel('Client Index')
            plt.ylabel('MZ-score Value')
            plt.legend()
            
            # 绘制梯度范数MZ-score分布图
            plt.subplot(3, 3, 6)
            plt.scatter([i for i in benign_indices], [mzscore_grad_norm[i] for i in benign_indices], 
                    label='Benign', color='blue', marker='o', s=30)
            plt.scatter([i for i in malicious_indices], [mzscore_grad_norm[i] for i in malicious_indices], 
                    label='Malicious', color='red', marker='x', s=40)
            plt.axhline(y=loose_lambda_g, color='green', linestyle='-', label='Standard Threshold')
            plt.axhline(y=strict_lambda_g, color='orange', linestyle='--', label='Strict Threshold (Benign/Suspicious)')
            plt.title('Grad Norm MZ-score Distribution')
            plt.xlabel('Client Index')
            plt.ylabel('MZ-score Value')
            plt.legend()

            # 第三行：相似度分析和综合信息
            # 绘制Mean Cos分布图
            plt.subplot(3, 3, 7)
            plt.scatter([i for i in benign_indices], [mean_cos_list[i] for i in benign_indices], 
                    label='Benign', color='blue', marker='o', s=30)
            plt.scatter([i for i in malicious_indices], [mean_cos_list[i] for i in malicious_indices], 
                    label='Malicious', color='red', marker='x', s=40)
            mean_cos_med = np.median(mean_cos_list)
            plt.axhline(y=mean_cos_med, color='green', linestyle='-', label='Median')
            plt.title('Mean Cos Distribution')
            plt.xlabel('Client Index')
            plt.ylabel('Mean Cos Value')
            plt.legend()

            # 绘制与平均更新余弦相似度MZ-score分布图
            plt.subplot(3, 3, 8)
            plt.scatter([i for i in benign_indices], [mzscore_mean_cos[i] for i in benign_indices], 
                    label='Benign', color='blue', marker='o', s=30)
            plt.scatter([i for i in malicious_indices], [mzscore_mean_cos[i] for i in malicious_indices], 
                    label='Malicious', color='red', marker='x', s=40)
            plt.axhline(y=loose_lambda_mean_cos, color='green', linestyle='-', label='Standard Threshold')
            plt.axhline(y=strict_lambda_mean_cos, color='orange', linestyle='--', label='Strict Threshold (Benign/Suspicious)')
            plt.title('Mean Cos MZ-score Distribution')
            plt.xlabel('Client Index')
            plt.ylabel('MZ-score Value')
            plt.legend()
            
            # 绘制客户端相似度热力图
            plt.subplot(3, 3, 9)
            im = plt.imshow(similarity_matrix, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
            plt.colorbar(im, label='Cosine Similarity')
            plt.title('Client Similarity Heatmap')
            plt.xlabel('Client Index')
            plt.ylabel('Client Index')
            
            # 添加客户端类型标注
            for i in range(num_chosen_clients):
                for j in range(num_chosen_clients):
                    if i != j:  # 不在对角线上显示数值
                        text_color = 'white' if abs(similarity_matrix[i, j]) > 0.5 else 'black'
                        plt.text(j, i, f'{similarity_matrix[i, j]:.2f}', 
                                ha='center', va='center', color=text_color, fontsize=8)
            
            # 在热力图边缘标注客户端类型
            client_labels = ['M' if i < len(malicious_id) else 'B' for i in range(num_chosen_clients)]
            plt.xticks(range(num_chosen_clients), [f'{i}({client_labels[i]})' for i in range(num_chosen_clients)], rotation=45)
            plt.yticks(range(num_chosen_clients), [f'{i}({client_labels[i]})' for i in range(num_chosen_clients)])
            
            # 添加总标题
            plt.suptitle(f'Round {current_round} - Hybrid Defense Method Analysis', fontsize=16)
            
            # 调整布局并保存
            plt.tight_layout(rect=[0, 0, 1, 0.97])  # 为总标题留出空间
            plt.savefig(os.path.join(output_dir, 'hybrid_defense_analysis.png'), dpi=300)
            plt.close()
            
            logging.info(f"Hybrid Method - 保存了第 {current_round} 轮的可视化结果到 {output_dir}")

        # ========== 第二阶段：对可疑客户端应用FedUP自适应剪枝 ==========
        pruned_suspicious_updates = []
        
        if len(suspicious_set) > 0:
            # 计算良性客户端间的相似度（用于FedUP自适应剪枝）
            if len(clear_benign_set) > 1:
                benign_updates = [inter_model_updates[i] for i in clear_benign_set]
                benign_similarities = []
                for i in range(len(benign_updates)):
                    for j in range(i+1, len(benign_updates)):
                        sim = cos(benign_updates[i], benign_updates[j]).item()
                        benign_similarities.append(sim)
                
                # 计算归一化相似度z
                if len(benign_similarities) > 0:
                    avg_similarity = np.mean(benign_similarities)
                    z = max(0, min(1, (avg_similarity + 1) / 2))  # 将[-1,1]映射到[0,1]
                else:
                    z = 0.5
            else:
                z = 0.3  # 当良性客户端不足时，使用更保守的剪枝
            
            # 改进的FedUP自适应剪枝比例计算
            p_max = getattr(self.args, 'fedup_p_max', 0.8)  # 使用更大的剪枝范围
            p_min = getattr(self.args, 'fedup_p_min', 0.1)
            gamma = getattr(self.args, 'fedup_gamma', 3)  # 降低gamma值，使剪枝更敏感
            
            # 基于可疑程度动态调整剪枝比例
            suspicious_updates = [inter_model_updates[i] for i in suspicious_set]
            
            for idx in suspicious_set:
                # 为每个可疑客户端计算个性化的剪枝比例
                client_suspicion_score = 0
                client_suspicion_score += max(0, mzscore_mpsa[idx] - strict_lambda_s) / (loose_lambda_s - strict_lambda_s)
                client_suspicion_score += max(0, mzscore_tda[idx] - strict_lambda_c) / (loose_lambda_c - strict_lambda_c)
                client_suspicion_score += max(0, mzscore_grad_norm[idx] - strict_lambda_g) / (loose_lambda_g - strict_lambda_g)
                client_suspicion_score += max(0, mzscore_mean_cos[idx] - strict_lambda_mean_cos) / (loose_lambda_mean_cos - strict_lambda_mean_cos)
                client_suspicion_score = min(1.0, client_suspicion_score / 4)  # 归一化到[0,1]
                
                # 结合良性相似度和可疑程度计算剪枝比例
                combined_factor = (1 - z) * 0.3 + client_suspicion_score * 0.7  # 可疑程度权重更大
                adaptive_pruning_ratio = p_min + (p_max - p_min) * (combined_factor ** gamma)
                
                logging.info('Hybrid Method - Client %d suspicion score: %.4f, pruning ratio: %.4f' % 
                           (idx, client_suspicion_score, adaptive_pruning_ratio))
                
                # 计算该客户端的权重重要性
                client_update = inter_model_updates[idx]
                weight_diff = client_update - flat_global_model
                weight_importance = torch.abs(weight_diff) * torch.abs(flat_global_model)
                
                # 应用个性化剪枝
                num_params_to_prune = int(len(weight_importance) * adaptive_pruning_ratio)
                if num_params_to_prune > 0:
                    _, top_indices = torch.topk(weight_importance, num_params_to_prune)
                    unlearn_mask = torch.ones_like(flat_global_model)
                    unlearn_mask[top_indices] = 0
                    
                    # 应用剪枝掩码
                    pruned_update = client_update * unlearn_mask
                    pruned_suspicious_updates.append(pruned_update)
                    
                    logging.info('Hybrid Method - Pruned %d/%d parameters for suspicious client %d' % 
                               (num_params_to_prune, len(weight_importance), idx))
                else:
                    pruned_suspicious_updates.append(client_update)
                    logging.info('Hybrid Method - No pruning applied for client %d (ratio too small)' % idx)
        
        logging.info('Hybrid Method - Benign similarity z: %.4f' % z)
        logging.info('Hybrid Method - Processed %d suspicious clients with FedUP pruning' % len(suspicious_set))
        
        # ========== 第三阶段：智能加权聚合 ==========
        # 准备最终聚合的更新和权重
        final_updates = []
        final_weights = []
        final_client_ids = []
        
        # 添加明确良性客户端（高权重）
        for idx in clear_benign_set:
            final_updates.append(inter_model_updates[idx])
            final_weights.append(1.0)  # 良性客户端权重为1
            final_client_ids.append(idx)
        
        # 添加剪枝后的可疑客户端（降低权重）
        for i, idx in enumerate(suspicious_set):
            if i < len(pruned_suspicious_updates):
                final_updates.append(pruned_suspicious_updates[i])
                # 基于可疑程度动态调整权重
                client_suspicion_score = 0
                client_suspicion_score += max(0, mzscore_mpsa[idx] - strict_lambda_s) / (loose_lambda_s - strict_lambda_s)
                client_suspicion_score += max(0, mzscore_tda[idx] - strict_lambda_c) / (loose_lambda_c - strict_lambda_c)
                client_suspicion_score += max(0, mzscore_grad_norm[idx] - strict_lambda_g) / (loose_lambda_g - strict_lambda_g)
                client_suspicion_score += max(0, mzscore_mean_cos[idx] - strict_lambda_mean_cos) / (loose_lambda_mean_cos - strict_lambda_mean_cos)
                client_suspicion_score = min(1.0, client_suspicion_score / 4)
                
                # 可疑客户端权重：0.1到0.6之间
                weight = 0.6 - 0.5 * client_suspicion_score
                final_weights.append(weight)
                final_client_ids.append(idx)
        
        # 如果没有可用客户端，返回零更新
        if len(final_updates) == 0:
            logging.warning('Hybrid Method - No available clients found, returning zero update')
            return torch.zeros_like(local_updates[0])
        
        # 转换为张量（确保设备一致性）
        final_updates_tensor = torch.stack(final_updates, dim=0)
        # 确保权重张量与更新张量在同一设备上
        device = final_updates_tensor.device
        final_weights_tensor = torch.tensor(final_weights, dtype=torch.float32, device=device).reshape(-1, 1)
        
        # 自适应范数裁剪
        updates_norm = torch.norm(final_updates_tensor, dim=1).reshape((-1, 1))
        # 使用加权中位数而不是简单中位数
        sorted_norms, sorted_indices = torch.sort(updates_norm.flatten())
        sorted_weights = final_weights_tensor.flatten()[sorted_indices]
        cumsum_weights = torch.cumsum(sorted_weights, dim=0)
        total_weight = cumsum_weights[-1]
        median_idx = torch.searchsorted(cumsum_weights, total_weight / 2)
        norm_clip = sorted_norms[median_idx].item()
        
        updates_norm_clipped = torch.clamp(updates_norm, 0, norm_clip, out=None)
        final_updates_tensor = (final_updates_tensor / updates_norm) * updates_norm_clipped
        
        # 加权聚合（确保所有张量在同一设备）
        weighted_updates = final_updates_tensor * final_weights_tensor
        aggregated_update = torch.sum(weighted_updates, dim=0) / torch.sum(final_weights_tensor)
        
        # 计算性能指标
        benign_selected = sum(1 for idx in final_client_ids if idx >= len(malicious_id))
        malicious_selected = sum(1 for idx in final_client_ids if idx < len(malicious_id))
        
        TPR = benign_selected / len(benign_id) if len(benign_id) > 0 else 0
        FPR = malicious_selected / len(malicious_id) if len(malicious_id) > 0 else 0
        
        # 计算加权指标（考虑权重影响）
        weighted_benign = sum(final_weights[i] for i, idx in enumerate(final_client_ids) if idx >= len(malicious_id))
        weighted_malicious = sum(final_weights[i] for i, idx in enumerate(final_client_ids) if idx < len(malicious_id))
        total_weighted = weighted_benign + weighted_malicious
        
        weighted_TPR = weighted_benign / len(benign_id) if len(benign_id) > 0 else 0
        weighted_FPR = weighted_malicious / len(malicious_id) if len(malicious_id) > 0 else 0
        
        logging.info('Hybrid Method - Final clients: %d benign + %d suspicious (pruned), excluded %d malicious' % 
                   (len(clear_benign_set), len(suspicious_set), len(clear_malicious_set)))
        logging.info('Hybrid Method - TPR: %.4f, FPR: %.4f' % (TPR, FPR))
        logging.info('Hybrid Method - Weighted TPR: %.4f, Weighted FPR: %.4f' % (weighted_TPR, weighted_FPR))
        logging.info('Hybrid Method - Client weights: %s' % [round(w, 3) for w in final_weights])
        
        return aggregated_update

    def agg_alignins_layer(self, agent_updates_dict, flat_global_model,global_model):
        """ classic fed avg """
        sm_updates, total_data = 0, 0
        for _id, update in agent_updates_dict.items():
            n_agent_data = self.agent_data_sizes[_id]
            sm_updates +=  n_agent_data * update
            total_data += n_agent_data
        aggregated_update = sm_updates / total_data
        """
        Analyze differences between different layers (bottom, middle, top) when under attack,
        and generate visualization charts
        """
        import matplotlib.pyplot as plt
        import os
        
        local_updates = []
        benign_id = []
        malicious_id = []
        iid_clients = []
        non_iid_clients = []

        for _id, update in agent_updates_dict.items():
            local_updates.append(update)
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)
            
            # 根据客户端ID判断是否为non-IID客户端
            # 假设non-IID客户端的ID是连续的，从0开始
            # 这里需要根据您的实际数据分布情况进行调整
            if hasattr(self.args, 'non_iid') and self.args.non_iid:
                # 在non-IID设置下，所有客户端都是non-IID的
                non_iid_clients.append(_id)
            else:
                # 在IID设置下，所有客户端都是IID的
                iid_clients.append(_id)

        chosen_clients = malicious_id + benign_id
        num_chosen_clients = len(malicious_id + benign_id)
        inter_model_updates = torch.stack(local_updates, dim=0)

        # Get model structure information, divide parameters into bottom, middle and top layers
        model_dict = vector_to_name_param(flat_global_model, copy.deepcopy(global_model.state_dict()))
        
        # Classify parameters into three layers based on names
        bottom_layers = []
        middle_layers = []
        top_layers = []
        
        # Classify by layer name
        for name in model_dict:
            if 'conv' in name.lower() and '0' in name.split('.')[-1]:
                bottom_layers.append(name)
            elif 'conv' in name.lower() and '1' in name.split('.')[-1]:
                middle_layers.append(name)
            elif 'fc' in name.lower() or 'classifier' in name.lower():
                top_layers.append(name)
        
        # If classification is uneven, redistribute based on total parameter count
        if len(bottom_layers) == 0 or len(middle_layers) == 0 or len(top_layers) == 0:
            all_layers = list(model_dict.keys())
            total_layers = len(all_layers)
            bottom_layers = all_layers[:total_layers//3]
            middle_layers = all_layers[total_layers//3:2*total_layers//3]
            top_layers = all_layers[2*total_layers//3:]
        
        logging.info(f'Bottom Layer Parameters: {bottom_layers}')
        logging.info(f'Middle Layer Parameters: {middle_layers}')
        logging.info(f'Top Layer Parameters: {top_layers}')
        
        # Calculate TDA and MPSA metrics for each layer
        layer_groups = {
            'Bottom Layer': bottom_layers,
            'Middle Layer': middle_layers,
            'Top Layer': top_layers
        }
        
        # Ensure output directory exists
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'layer_analysis_nonidd_lsd')
        os.makedirs(output_dir, exist_ok=True)
        
        # Analyze each layer group
        for layer_name, layer_list in layer_groups.items():
            tda_list = []
            mpsa_list = []
            client_types = []  # Record client type (benign or malicious)
            client_dist_types = []  # Record distribution type (IID or non-IID)
            
            # Extract parameters for this layer group
            layer_indices = []
            for name in layer_list:
                param_shape = model_dict[name].shape
                param_size = model_dict[name].numel()
                start_idx = 0
                for n in model_dict:
                    if n == name:
                        break
                    start_idx += model_dict[n].numel()
                end_idx = start_idx + param_size
                layer_indices.extend(list(range(start_idx, end_idx)))
            
            layer_indices = torch.tensor(layer_indices)
            
            # Calculate major sign for this layer
            layer_updates = inter_model_updates[:, layer_indices]
            major_sign = torch.sign(torch.sum(torch.sign(layer_updates), dim=0))
            cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
            
            # Calculate TDA and MPSA for each client in this layer
            for i in range(len(inter_model_updates)):
                client_update = inter_model_updates[i][layer_indices]
                _, init_indices = torch.topk(torch.abs(client_update), int(len(client_update) * self.args.sparsity))
                
                mpsa_value = (torch.sum(torch.sign(client_update[init_indices]) == major_sign[init_indices]) / 
                             torch.numel(client_update[init_indices])).item()
                mpsa_list.append(mpsa_value)
                
                tda_value = cos(client_update, flat_global_model[layer_indices]).item()
                tda_list.append(tda_value)
                
                # Record client type
                if i < len(malicious_id):
                    client_types.append('Malicious')
                else:
                    client_types.append('Benign')
                
                # Record distribution type
                if chosen_clients[i] in non_iid_clients:
                    client_dist_types.append('Non-IID')
                else:
                    client_dist_types.append('IID')
            
            # Calculate MZ-score
            mpsa_std = np.std(mpsa_list)
            mpsa_med = np.median(mpsa_list)
            mzscore_mpsa = [np.abs(val - mpsa_med) / mpsa_std for val in mpsa_list]
            
            tda_std = np.std(tda_list)
            tda_med = np.median(tda_list)
            mzscore_tda = [np.abs(val - tda_med) / tda_std for val in tda_list]
            
            # Log analysis results
            logging.info(f'{layer_name} TDA: %s' % [round(i, 4) for i in tda_list])
            logging.info(f'{layer_name} MPSA: %s' % [round(i, 4) for i in mpsa_list])
            logging.info(f'{layer_name} MZ-score of MPSA: %s' % [round(i, 4) for i in mzscore_mpsa])
            logging.info(f'{layer_name} MZ-score of TDA: %s' % [round(i, 4) for i in mzscore_tda])
            
            # 创建两组图表：一组按恶意/良性分类，一组按IID/non-IID分类
            
            # 图1：按恶意/良性分类
            plt.figure(figsize=(15, 10))
            
            # Draw TDA chart
            plt.subplot(2, 2, 1)
            benign_indices = [i for i, t in enumerate(client_types) if t == 'Benign']
            malicious_indices = [i for i, t in enumerate(client_types) if t == 'Malicious']
            
            plt.scatter([i for i in benign_indices], [tda_list[i] for i in benign_indices], 
                       label='Benign Clients', color='blue', marker='o')
            plt.scatter([i for i in malicious_indices], [tda_list[i] for i in malicious_indices], 
                       label='Malicious Clients', color='red', marker='x')
            plt.axhline(y=tda_med, color='green', linestyle='-', label='Median')
            plt.title(f'{layer_name} - TDA Distribution')
            plt.xlabel('Client Index')
            plt.ylabel('TDA Value')
            plt.legend()
            
            # Draw MPSA chart
            plt.subplot(2, 2, 2)
            plt.scatter([i for i in benign_indices], [mpsa_list[i] for i in benign_indices], 
                       label='Benign Clients', color='blue', marker='o')
            plt.scatter([i for i in malicious_indices], [mpsa_list[i] for i in malicious_indices], 
                       label='Malicious Clients', color='red', marker='x')
            plt.axhline(y=mpsa_med, color='green', linestyle='-', label='Median')
            plt.title(f'{layer_name} - MPSA Distribution')
            plt.xlabel('Client Index')
            plt.ylabel('MPSA Value')
            plt.legend()
            
            # Draw TDA MZ-score chart
            plt.subplot(2, 2, 3)
            plt.scatter([i for i in benign_indices], [mzscore_tda[i] for i in benign_indices], 
                       label='Benign Clients', color='blue', marker='o')
            plt.scatter([i for i in malicious_indices], [mzscore_tda[i] for i in malicious_indices], 
                       label='Malicious Clients', color='red', marker='x')
            plt.axhline(y=self.args.lambda_c, color='green', linestyle='-', label='Threshold')
            plt.title(f'{layer_name} - TDA MZ-score Distribution')
            plt.xlabel('Client Index')
            plt.ylabel('MZ-score Value')
            plt.legend()
            
            # Draw MPSA MZ-score chart
            plt.subplot(2, 2, 4)
            plt.scatter([i for i in benign_indices], [mzscore_mpsa[i] for i in benign_indices], 
                       label='Benign Clients', color='blue', marker='o')
            plt.scatter([i for i in malicious_indices], [mzscore_mpsa[i] for i in malicious_indices], 
                       label='Malicious Clients', color='red', marker='x')
            plt.axhline(y=self.args.lambda_s, color='green', linestyle='-', label='Threshold')
            plt.title(f'{layer_name} - MPSA MZ-score Distribution')
            plt.xlabel('Client Index')
            plt.ylabel('MZ-score Value')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{layer_name}_analysis_by_type.png'))
            plt.close()
            
            # 图2：按IID/non-IID分类
            plt.figure(figsize=(15, 10))
            
            # Draw TDA chart by distribution type
            plt.subplot(2, 2, 1)
            iid_indices = [i for i, t in enumerate(client_dist_types) if t == 'IID']
            non_iid_indices = [i for i, t in enumerate(client_dist_types) if t == 'Non-IID']
            
            plt.scatter([i for i in iid_indices], [tda_list[i] for i in iid_indices], 
                       label='IID Clients', color='green', marker='o')
            plt.scatter([i for i in non_iid_indices], [tda_list[i] for i in non_iid_indices], 
                       label='Non-IID Clients', color='purple', marker='x')
            plt.axhline(y=tda_med, color='black', linestyle='-', label='Median')
            plt.title(f'{layer_name} - TDA Distribution by Data Distribution')
            plt.xlabel('Client Index')
            plt.ylabel('TDA Value')
            plt.legend()
            
            # Draw MPSA chart by distribution type
            plt.subplot(2, 2, 2)
            plt.scatter([i for i in iid_indices], [mpsa_list[i] for i in iid_indices], 
                       label='IID Clients', color='green', marker='o')
            plt.scatter([i for i in non_iid_indices], [mpsa_list[i] for i in non_iid_indices], 
                       label='Non-IID Clients', color='purple', marker='x')
            plt.axhline(y=mpsa_med, color='black', linestyle='-', label='Median')
            plt.title(f'{layer_name} - MPSA Distribution by Data Distribution')
            plt.xlabel('Client Index')
            plt.ylabel('MPSA Value')
            plt.legend()
            
            # Draw TDA MZ-score chart by distribution type
            plt.subplot(2, 2, 3)
            plt.scatter([i for i in iid_indices], [mzscore_tda[i] for i in iid_indices], 
                       label='IID Clients', color='green', marker='o')
            plt.scatter([i for i in non_iid_indices], [mzscore_tda[i] for i in non_iid_indices], 
                       label='Non-IID Clients', color='purple', marker='x')
            plt.axhline(y=self.args.lambda_c, color='black', linestyle='-', label='Threshold')
            plt.title(f'{layer_name} - TDA MZ-score by Data Distribution')
            plt.xlabel('Client Index')
            plt.ylabel('MZ-score Value')
            plt.legend()
            
            # Draw MPSA MZ-score chart by distribution type
            plt.subplot(2, 2, 4)
            plt.scatter([i for i in iid_indices], [mzscore_mpsa[i] for i in iid_indices], 
                       label='IID Clients', color='green', marker='o')
            plt.scatter([i for i in non_iid_indices], [mzscore_mpsa[i] for i in non_iid_indices], 
                       label='Non-IID Clients', color='purple', marker='x')
            plt.axhline(y=self.args.lambda_s, color='black', linestyle='-', label='Threshold')
            plt.title(f'{layer_name} - MPSA MZ-score by Data Distribution')
            plt.xlabel('Client Index')
            plt.ylabel('MZ-score Value')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{layer_name}_analysis_by_distribution.png'))
            plt.close()
            
            # 图3：综合分析 - 同时显示恶意/良性和IID/non-IID
            plt.figure(figsize=(15, 10))
            
            # 创建四种类型的客户端索引
            benign_iid_indices = [i for i, (t, d) in enumerate(zip(client_types, client_dist_types)) 
                                if t == 'Benign' and d == 'IID']
            benign_non_iid_indices = [i for i, (t, d) in enumerate(zip(client_types, client_dist_types)) 
                                    if t == 'Benign' and d == 'Non-IID']
            malicious_iid_indices = [i for i, (t, d) in enumerate(zip(client_types, client_dist_types)) 
                                   if t == 'Malicious' and d == 'IID']
            malicious_non_iid_indices = [i for i, (t, d) in enumerate(zip(client_types, client_dist_types)) 
                                       if t == 'Malicious' and d == 'Non-IID']
            
            # Draw TDA chart with combined classification
            plt.subplot(2, 2, 1)
            plt.scatter([i for i in benign_iid_indices], [tda_list[i] for i in benign_iid_indices], 
                       label='Benign IID', color='blue', marker='o')
            plt.scatter([i for i in benign_non_iid_indices], [tda_list[i] for i in benign_non_iid_indices], 
                       label='Benign Non-IID', color='cyan', marker='s')
            plt.scatter([i for i in malicious_iid_indices], [tda_list[i] for i in malicious_iid_indices], 
                       label='Malicious IID', color='red', marker='x')
            plt.scatter([i for i in malicious_non_iid_indices], [tda_list[i] for i in malicious_non_iid_indices], 
                       label='Malicious Non-IID', color='magenta', marker='+')
            plt.axhline(y=tda_med, color='green', linestyle='-', label='Median')
            plt.title(f'{layer_name} - TDA Distribution (Combined)')
            plt.xlabel('Client Index')
            plt.ylabel('TDA Value')
            plt.legend()
            
            # Draw MPSA chart with combined classification
            plt.subplot(2, 2, 2)
            plt.scatter([i for i in benign_iid_indices], [mpsa_list[i] for i in benign_iid_indices], 
                       label='Benign IID', color='blue', marker='o')
            plt.scatter([i for i in benign_non_iid_indices], [mpsa_list[i] for i in benign_non_iid_indices], 
                       label='Benign Non-IID', color='cyan', marker='s')
            plt.scatter([i for i in malicious_iid_indices], [mpsa_list[i] for i in malicious_iid_indices], 
                       label='Malicious IID', color='red', marker='x')
            plt.scatter([i for i in malicious_non_iid_indices], [mpsa_list[i] for i in malicious_non_iid_indices], 
                       label='Malicious Non-IID', color='magenta', marker='+')
            plt.axhline(y=mpsa_med, color='green', linestyle='-', label='Median')
            plt.title(f'{layer_name} - MPSA Distribution (Combined)')
            plt.xlabel('Client Index')
            plt.ylabel('MPSA Value')
            plt.legend()
            
            # Draw TDA MZ-score chart with combined classification
            plt.subplot(2, 2, 3)
            plt.scatter([i for i in benign_iid_indices], [mzscore_tda[i] for i in benign_iid_indices], 
                       label='Benign IID', color='blue', marker='o')
            plt.scatter([i for i in benign_non_iid_indices], [mzscore_tda[i] for i in benign_non_iid_indices], 
                       label='Benign Non-IID', color='cyan', marker='s')
            plt.scatter([i for i in malicious_iid_indices], [mzscore_tda[i] for i in malicious_iid_indices], 
                       label='Malicious IID', color='red', marker='x')
            plt.scatter([i for i in malicious_non_iid_indices], [mzscore_tda[i] for i in malicious_non_iid_indices], 
                       label='Malicious Non-IID', color='magenta', marker='+')
            plt.axhline(y=self.args.lambda_c, color='green', linestyle='-', label='Threshold')
            plt.title(f'{layer_name} - TDA MZ-score (Combined)')
            plt.xlabel('Client Index')
            plt.ylabel('MZ-score Value')
            plt.legend()
            
            # Draw MPSA MZ-score chart with combined classification
            plt.subplot(2, 2, 4)
            plt.scatter([i for i in benign_iid_indices], [mzscore_mpsa[i] for i in benign_iid_indices], 
                       label='Benign IID', color='blue', marker='o')
            plt.scatter([i for i in benign_non_iid_indices], [mzscore_mpsa[i] for i in benign_non_iid_indices], 
                       label='Benign Non-IID', color='cyan', marker='s')
            plt.scatter([i for i in malicious_iid_indices], [mzscore_mpsa[i] for i in malicious_iid_indices], 
                       label='Malicious IID', color='red', marker='x')
            plt.scatter([i for i in malicious_non_iid_indices], [mzscore_mpsa[i] for i in malicious_non_iid_indices], 
                       label='Malicious Non-IID', color='magenta', marker='+')
            plt.axhline(y=self.args.lambda_s, color='green', linestyle='-', label='Threshold')
            plt.title(f'{layer_name} - MPSA MZ-score (Combined)')
            plt.xlabel('Client Index')
            plt.ylabel('MZ-score Value')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{layer_name}_analysis_combined.png'))
            plt.close()
            
        # 执行与原始agg_alignins相同的聚合逻辑
        # ... existing code ...
            
        # Execute the same aggregation logic as the original agg_alignins
        tda_list = []
        mpsa_list = []
        major_sign = torch.sign(torch.sum(torch.sign(inter_model_updates), dim=0))
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        for i in range(len(inter_model_updates)):
            _, init_indices = torch.topk(torch.abs(inter_model_updates[i]), int(len(inter_model_updates[i]) * self.args.sparsity))

            mpsa_list.append((torch.sum(torch.sign(inter_model_updates[i][init_indices]) == major_sign[init_indices]) / torch.numel(inter_model_updates[i][init_indices])).item())
    
            tda_list.append(cos(inter_model_updates[i], flat_global_model).item())
            
        mpsa_std = np.std(mpsa_list)
        mpsa_med = np.median(mpsa_list)

        mzscore_mpsa = []
        for i in range(len(mpsa_list)):
            mzscore_mpsa.append(np.abs(mpsa_list[i] - mpsa_med) / mpsa_std)
        
        tda_std = np.std(tda_list)
        tda_med = np.median(tda_list)
        mzscore_tda = []
        for i in range(len(tda_list)):
            mzscore_tda.append(np.abs(tda_list[i] - tda_med) / tda_std)
            
        benign_idx1 = set([i for i in range(num_chosen_clients)])
        benign_idx1 = benign_idx1.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_mpsa) < self.args.lambda_s)]))
        benign_idx2 = set([i for i in range(num_chosen_clients)])
        benign_idx2 = benign_idx2.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_tda) < self.args.lambda_c)]))

        benign_set = benign_idx2.intersection(benign_idx1)
        
        benign_idx = list(benign_set)
        if len(benign_idx) == 0:
            return torch.zeros_like(local_updates[0])

        benign_updates = torch.stack([local_updates[i] for i in benign_idx], dim=0)

        # Post-filtering model clipping
        updates_norm = torch.norm(benign_updates, dim=1).reshape((-1, 1))
        norm_clip = updates_norm.median(dim=0)[0].item()
        benign_updates = torch.stack(local_updates, dim=0)
        updates_norm = torch.norm(benign_updates, dim=1).reshape((-1, 1))
        updates_norm_clipped = torch.clamp(updates_norm, 0, norm_clip, out=None)
        
        benign_updates = (benign_updates/updates_norm)*updates_norm_clipped

        correct = 0
        for idx in benign_idx:
            if idx >= len(malicious_id):
                correct += 1

        TPR = correct / len(benign_id)

        if len(malicious_id) == 0:
            FPR = 0
        else:
            wrong = 0
            for idx in benign_idx:
                if idx < len(malicious_id):
                    wrong += 1
            FPR = wrong / len(malicious_id)

        logging.info('benign update index:   %s' % str(benign_id))
        logging.info('selected update index: %s' % str(benign_idx))

        logging.info('FPR:       %.4f'  % FPR)
        logging.info('TPR:       %.4f' % TPR)

        current_dict = {}
        for idx in benign_idx:
            current_dict[chosen_clients[idx]] = benign_updates[idx]

        aggregated_update = self.agg_avg(current_dict)
        return aggregated_update

    def agg_alignins_layer_lsd(self, agent_updates_dict, flat_global_model, global_model):
        """ Classic fed avg with layer sensitivity difference analysis """
        import matplotlib.pyplot as plt
        import os
        
        local_updates = []
        benign_id = []
        malicious_id = []

        for _id, update in agent_updates_dict.items():
            local_updates.append(update)
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        chosen_clients = malicious_id + benign_id
        num_chosen_clients = len(malicious_id + benign_id)
        inter_model_updates = torch.stack(local_updates, dim=0)

        # Get model structure information, divide parameters into bottom, middle and top layers
        model_dict = vector_to_name_param(flat_global_model, copy.deepcopy(global_model.state_dict()))
        
        # Classify parameters into three layers based on names
        bottom_layers = []
        middle_layers = []
        top_layers = []
        
        # Classify by layer name
        for name in model_dict:
            if 'conv' in name.lower() and '0' in name.split('.')[-1]:
                bottom_layers.append(name)
            elif 'conv' in name.lower() and '1' in name.split('.')[-1]:
                middle_layers.append(name)
            elif 'fc' in name.lower() or 'classifier' in name.lower():
                top_layers.append(name)
        
        # If classification is uneven, redistribute based on total parameter count
        if len(bottom_layers) == 0 or len(middle_layers) == 0 or len(top_layers) == 0:
            all_layers = list(model_dict.keys())
            total_layers = len(all_layers)
            bottom_layers = all_layers[:total_layers//3]
            middle_layers = all_layers[total_layers//3:2*total_layers//3]
            top_layers = all_layers[2*total_layers//3:]
        
        logging.info(f'Bottom Layer Parameters: {bottom_layers}')
        logging.info(f'Middle Layer Parameters: {middle_layers}')
        logging.info(f'Top Layer Parameters: {top_layers}')
        
        # Ensure output directory exists
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'layer_analysis_lsd_client_10_v1')
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate TDA and MPSA metrics for each layer
        layer_groups = {
            'Bottom Layer': bottom_layers,
            'Middle Layer': middle_layers,
            'Top Layer': top_layers
        }
        
        # Store metrics for each client in different layers
        client_layer_metrics = {}
        for i in range(len(inter_model_updates)):
            client_layer_metrics[i] = {'tda': {}, 'mpsa': {}, 'lsd': {}}
        
        # Analyze each layer group
        for layer_name, layer_list in layer_groups.items():
            tda_list = []
            mpsa_list = []
            client_types = []  # Record client type (benign or malicious)
            
            # Extract parameters for this layer group
            layer_indices = []
            for name in layer_list:
                param_shape = model_dict[name].shape
                param_size = model_dict[name].numel()
                start_idx = 0
                for n in model_dict:
                    if n == name:
                        break
                    start_idx += model_dict[n].numel()
                end_idx = start_idx + param_size
                layer_indices.extend(list(range(start_idx, end_idx)))
            
            layer_indices = torch.tensor(layer_indices)
            
            # Calculate major sign for this layer
            layer_updates = inter_model_updates[:, layer_indices]
            major_sign = torch.sign(torch.sum(torch.sign(layer_updates), dim=0))
            cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
            
            # Calculate TDA and MPSA for each client in this layer
            for i in range(len(inter_model_updates)):
                client_update = inter_model_updates[i][layer_indices]
                _, init_indices = torch.topk(torch.abs(client_update), int(len(client_update) * self.args.sparsity))
                
                mpsa_value = (torch.sum(torch.sign(client_update[init_indices]) == major_sign[init_indices]) / 
                             torch.numel(client_update[init_indices])).item()
                mpsa_list.append(mpsa_value)
                client_layer_metrics[i]['mpsa'][layer_name] = mpsa_value
                
                tda_value = cos(client_update, flat_global_model[layer_indices]).item()
                tda_list.append(tda_value)
                client_layer_metrics[i]['tda'][layer_name] = tda_value
                
                # Record client type
                if i < len(malicious_id):
                    client_types.append('Malicious')
                else:
                    client_types.append('Benign')
            
            # Calculate MZ-score
            mpsa_std = np.std(mpsa_list)
            mpsa_med = np.median(mpsa_list)
            mzscore_mpsa = [np.abs(val - mpsa_med) / mpsa_std for val in mpsa_list]
            
            tda_std = np.std(tda_list)
            tda_med = np.median(tda_list)
            mzscore_tda = [np.abs(val - tda_med) / tda_std for val in tda_list]
            
            # Log analysis results
            logging.info(f'{layer_name} TDA: %s' % [round(i, 4) for i in tda_list])
            logging.info(f'{layer_name} MPSA: %s' % [round(i, 4) for i in mpsa_list])
            logging.info(f'{layer_name} MZ-score of MPSA: %s' % [round(i, 4) for i in mzscore_mpsa])
            logging.info(f'{layer_name} MZ-score of TDA: %s' % [round(i, 4) for i in mzscore_tda])
            
            # Draw charts
            plt.figure(figsize=(15, 10))
            
            # Draw TDA chart
            plt.subplot(2, 2, 1)
            benign_indices = [i for i, t in enumerate(client_types) if t == 'Benign']
            malicious_indices = [i for i, t in enumerate(client_types) if t == 'Malicious']
            
            plt.scatter([i for i in benign_indices], [tda_list[i] for i in benign_indices], 
                       label='Benign Clients', color='blue', marker='o')
            plt.scatter([i for i in malicious_indices], [tda_list[i] for i in malicious_indices], 
                       label='Malicious Clients', color='red', marker='x')
            plt.axhline(y=tda_med, color='green', linestyle='-', label='Median')
            plt.title(f'{layer_name} - TDA Distribution')
            plt.xlabel('Client Index')
            plt.ylabel('TDA Value')
            plt.legend()
            
            # Draw MPSA chart
            plt.subplot(2, 2, 2)
            plt.scatter([i for i in benign_indices], [mpsa_list[i] for i in benign_indices], 
                       label='Benign Clients', color='blue', marker='o')
            plt.scatter([i for i in malicious_indices], [mpsa_list[i] for i in malicious_indices], 
                       label='Malicious Clients', color='red', marker='x')
            plt.axhline(y=mpsa_med, color='green', linestyle='-', label='Median')
            plt.title(f'{layer_name} - MPSA Distribution')
            plt.xlabel('Client Index')
            plt.ylabel('MPSA Value')
            plt.legend()
            
            # Draw TDA MZ-score chart
            plt.subplot(2, 2, 3)
            plt.scatter([i for i in benign_indices], [mzscore_tda[i] for i in benign_indices], 
                       label='Benign Clients', color='blue', marker='o')
            plt.scatter([i for i in malicious_indices], [mzscore_tda[i] for i in malicious_indices], 
                       label='Malicious Clients', color='red', marker='x')
            plt.axhline(y=self.args.lambda_c, color='green', linestyle='-', label='Threshold')
            plt.title(f'{layer_name} - TDA MZ-score Distribution')
            plt.xlabel('Client Index')
            plt.ylabel('MZ-score Value')
            plt.legend()
            
            # Draw MPSA MZ-score chart
            plt.subplot(2, 2, 4)
            plt.scatter([i for i in benign_indices], [mzscore_mpsa[i] for i in benign_indices], 
                       label='Benign Clients', color='blue', marker='o')
            plt.scatter([i for i in malicious_indices], [mzscore_mpsa[i] for i in malicious_indices], 
                       label='Malicious Clients', color='red', marker='x')
            plt.axhline(y=self.args.lambda_s, color='green', linestyle='-', label='Threshold')
            plt.title(f'{layer_name} - MPSA MZ-score Distribution')
            plt.xlabel('Client Index')
            plt.ylabel('MZ-score Value')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{layer_name}_analysis.png'))
            plt.close()
        
        # Calculate Layer Sensitivity Difference (LSD)
        lsd_list = []
        for i in range(len(inter_model_updates)):
            # Calculate TDA difference between bottom and top layers
            tda_bottom = client_layer_metrics[i]['tda']['Bottom Layer']
            tda_top = client_layer_metrics[i]['tda']['Top Layer']
            tda_diff = abs(tda_bottom - tda_top)
            
            # Calculate MPSA difference between bottom and top layers
            mpsa_bottom = client_layer_metrics[i]['mpsa']['Bottom Layer']
            mpsa_top = client_layer_metrics[i]['mpsa']['Top Layer']
            mpsa_diff = abs(mpsa_bottom - mpsa_top)
            
            # Combined LSD metric (can adjust weights as needed)
            lsd_value = 0.5 * tda_diff + 0.5 * mpsa_diff
            lsd_list.append(lsd_value)
            client_layer_metrics[i]['lsd'] = lsd_value
        
        logging.info('Layer Sensitivity Difference (LSD): %s' % [round(i, 4) for i in lsd_list])
        
        # Calculate LSD MZ-score
        lsd_std = np.std(lsd_list)
        lsd_med = np.median(lsd_list)
        mzscore_lsd = [np.abs(val - lsd_med) / lsd_std for val in lsd_list]
        
        logging.info('MZ-score of LSD: %s' % [round(i, 4) for i in mzscore_lsd])
        
        # Draw LSD analysis charts
        plt.figure(figsize=(15, 10))
        
        # Draw LSD distribution chart
        plt.subplot(2, 1, 1)
        plt.scatter([i for i in benign_indices], [lsd_list[i] for i in benign_indices], 
                   label='Benign Clients', color='blue', marker='o')
        plt.scatter([i for i in malicious_indices], [lsd_list[i] for i in malicious_indices], 
                   label='Malicious Clients', color='red', marker='x')
        plt.axhline(y=lsd_med, color='green', linestyle='-', label='Median')
        plt.title('Layer Sensitivity Difference (LSD) Distribution')
        plt.xlabel('Client Index')
        plt.ylabel('LSD Value')
        plt.legend()
        
        # Draw LSD MZ-score distribution chart
        plt.subplot(2, 1, 2)
        plt.scatter([i for i in benign_indices], [mzscore_lsd[i] for i in benign_indices], 
                   label='Benign Clients', color='blue', marker='o')
        plt.scatter([i for i in malicious_indices], [mzscore_lsd[i] for i in malicious_indices], 
                   label='Malicious Clients', color='red', marker='x')
        # Use same threshold as other metrics, or can set a new threshold parameter
        lsd_threshold = self.args.lambda_c  # Can adjust as needed
        plt.axhline(y=lsd_threshold, color='green', linestyle='-', label='Threshold')
        plt.title('LSD MZ-score Distribution')
        plt.xlabel('Client Index')
        plt.ylabel('MZ-score Value')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'LSD_analysis.png'))
        plt.close()
        
        # Execute the same aggregation logic as the original agg_alignins, but add LSD metric
        tda_list = []
        mpsa_list = []
        major_sign = torch.sign(torch.sum(torch.sign(inter_model_updates), dim=0))
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        for i in range(len(inter_model_updates)):
            _, init_indices = torch.topk(torch.abs(inter_model_updates[i]), int(len(inter_model_updates[i]) * self.args.sparsity))

            mpsa_list.append((torch.sum(torch.sign(inter_model_updates[i][init_indices]) == major_sign[init_indices]) / torch.numel(inter_model_updates[i][init_indices])).item())
    
            tda_list.append(cos(inter_model_updates[i], flat_global_model).item())
            
        mpsa_std = np.std(mpsa_list)
        mpsa_med = np.median(mpsa_list)

        mzscore_mpsa = []
        for i in range(len(mpsa_list)):
            mzscore_mpsa.append(np.abs(mpsa_list[i] - mpsa_med) / mpsa_std)
        
        tda_std = np.std(tda_list)
        tda_med = np.median(tda_list)
        mzscore_tda = []
        for i in range(len(tda_list)):
            mzscore_tda.append(np.abs(tda_list[i] - tda_med) / tda_std)
        
        # Use LSD metric for anomaly detection
        benign_idx1 = set([i for i in range(num_chosen_clients)])
        benign_idx1 = benign_idx1.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_mpsa) < self.args.lambda_s)]))
        benign_idx2 = set([i for i in range(num_chosen_clients)])
        benign_idx2 = benign_idx2.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_tda) < self.args.lambda_c)]))
        benign_idx3 = set([i for i in range(num_chosen_clients)])
        benign_idx3 = benign_idx3.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_lsd) < lsd_threshold)]))

        # Combine all three metrics
        benign_set = benign_idx2.intersection(benign_idx1).intersection(benign_idx3)
        
        benign_idx = list(benign_set)
        if len(benign_idx) == 0:
            return torch.zeros_like(local_updates[0])

        benign_updates = torch.stack([local_updates[i] for i in benign_idx], dim=0)

        # Post-filtering model clipping
        updates_norm = torch.norm(benign_updates, dim=1).reshape((-1, 1))
        norm_clip = updates_norm.median(dim=0)[0].item()
        benign_updates = torch.stack(local_updates, dim=0)
        updates_norm = torch.norm(benign_updates, dim=1).reshape((-1, 1))
        updates_norm_clipped = torch.clamp(updates_norm, 0, norm_clip, out=None)
        
        benign_updates = (benign_updates/updates_norm)*updates_norm_clipped

        correct = 0
        for idx in benign_idx:
            if idx >= len(malicious_id):
                correct += 1

        TPR = correct / len(benign_id)

        if len(malicious_id) == 0:
            FPR = 0
        else:
            wrong = 0
            for idx in benign_idx:
                if idx < len(malicious_id):
                    wrong += 1
            FPR = wrong / len(malicious_id)

        logging.info('Benign update index:   %s' % str(benign_id))
        logging.info('Selected update index: %s' % str(benign_idx))

        logging.info('FPR:       %.4f'  % FPR)
        logging.info('TPR:       %.4f' % TPR)

        current_dict = {}
        for idx in benign_idx:
            current_dict[chosen_clients[idx]] = benign_updates[idx]

        aggregated_update = self.agg_avg(current_dict)
        return aggregated_update
    
    def agg_alignins_plr(self, agent_updates_dict, flat_global_model, global_model, auxiliary_data_loader):
        """
        使用表示层分析方法来检测恶意客户端，并生成可视化图表
        """
        import matplotlib.pyplot as plt
        import os
        import numpy as np
        import torch
        from sklearn.manifold import TSNE
        import copy
        import logging
        
        # 创建输出目录
        logging.info("开始PLR聚合...")
        
        # 收集客户端信息
        local_updates = []
        benign_id = []
        malicious_id = []
        
        for _id, update in agent_updates_dict.items():
            local_updates.append(update)
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)
        
        chosen_clients = malicious_id + benign_id
        num_chosen_clients = len(chosen_clients)
        
        # 提取每个客户端模型的倒数第二层表示(PLR)
        plr_features = []
        client_types = []  # 记录客户端类型(良性/恶意)
        
        logging.info("提取客户端模型的倒数第二层表示...")
        
        # 检查辅助数据加载器
        if auxiliary_data_loader is None:
            logging.error("辅助数据加载器为None，无法进行PLR分析")
            return self.agg_avg(agent_updates_dict)
        
        # 检查一个批次样本
        try:
            test_batch = next(iter(auxiliary_data_loader))
            logging.info(f"辅助数据加载器批次类型: {type(test_batch)}")
            logging.info(f"辅助数据加载器批次长度: {len(test_batch)}")
            if len(test_batch) >= 2:
                inputs, labels = test_batch
                logging.info(f"输入形状: {inputs.shape}")
                logging.info(f"标签形状: {labels.shape}")
        except Exception as e:
            logging.error(f"检查辅助数据加载器时出错: {str(e)}")
        
        # 为每个客户端创建临时模型并提取PLR
        for i, update in enumerate(local_updates):
            try:
                client_id = chosen_clients[i]
                logging.info(f"处理客户端 {client_id} (索引 {i})...")
                
                # 应用更新到临时模型
                temp_model = copy.deepcopy(global_model)
                temp_params = flat_global_model + update
                vector_to_model(temp_params, temp_model)
                
                # 检查模型是否有必要的属性
                if not hasattr(temp_model, 'res2') or not hasattr(temp_model, 'classifier'):
                    logging.error(f"客户端 {client_id} 的模型缺少必要的属性")
                    logging.info(f"模型属性: {[attr for attr in dir(temp_model) if not attr.startswith('_')]}")
                    return self.agg_avg(agent_updates_dict)
                
                # 尝试对一个小批次运行模型
                try:
                    test_batch = next(iter(auxiliary_data_loader))
                    if len(test_batch) >= 2:
                        inputs, _ = test_batch
                        inputs = inputs.to(self.args.device)
                        with torch.no_grad():
                            test_output = temp_model(inputs)
                        logging.info(f"模型测试输出形状: {test_output.shape}")
                except Exception as e:
                    logging.error(f"模型前向传播测试失败: {str(e)}")
                
                # 提取PLR序列
                logging.info(f"开始为客户端 {client_id} 提取PLR...")
                client_features = self.extract_plr_sequence(temp_model, auxiliary_data_loader)
                
                if client_features is None:
                    logging.error(f"客户端 {client_id} 的PLR提取返回None")
                    return self.agg_avg(agent_updates_dict)
                
                logging.info(f"客户端 {client_id} 的PLR特征形状: {client_features.shape}")
                plr_features.append(client_features)
                
                # 记录客户端类型
                if i < len(malicious_id):
                    client_types.append('Malicious')
                else:
                    client_types.append('Benign')
                    
            except Exception as e:
                logging.error(f"处理客户端 {chosen_clients[i]} 时出错: {str(e)}")
                import traceback
                logging.error(traceback.format_exc())
                return self.agg_avg(agent_updates_dict)
        
        # 确保所有特征都被提取
        if len(plr_features) != num_chosen_clients:
            logging.error(f"只提取了 {len(plr_features)}/{num_chosen_clients} 个客户端的PLR特征")
            return self.agg_avg(agent_updates_dict)
    
    # 继续后续处理...

    def extract_plr_sequence(self, model, data_loader):
        """
        提取模型倒数第二层的表示序列，针对ResNet9模型优化
        
        参数:
        - model: 神经网络模型
        - data_loader: 辅助数据加载器
        
        返回:
        - features: 倒数第二层表示序列
        """
        import torch
        import torch.nn as nn
        import logging
        
        features = []
        model.eval()
        
        # 获取倒数第二层的输出
        def get_penultimate_layer(model, x):
            # 对于ResNet9模型
            if hasattr(model, 'res2') and hasattr(model, 'classifier'):
                # ResNet9模型的前向传播，但不执行最后的分类器
                out = model.conv1(x)
                out = model.conv2(out)
                out = model.res1(out) + out
                out = model.conv3(out)
                out = model.conv4(out)
                out = model.res2(out) + out
                
                # 在这里，我们需要应用classifier的前两个操作（MaxPool2d和Flatten）
                # 但不执行最后的Linear层
                if isinstance(model.classifier, nn.Sequential) and len(model.classifier) >= 3:
                    # 应用MaxPool2d
                    out = model.classifier[0](out)
                    # 应用Flatten
                    out = model.classifier[1](out)
                    return out
                else:
                    # 如果classifier的结构不是预期的，使用手动方法
                    out = nn.MaxPool2d(4)(out)
                    out = out.flatten(1)
                    return out
            else:
                logging.error("模型不是预期的ResNet9结构")
                return None
        
        try:
            if data_loader is None:
                logging.error("数据加载器为None，无法提取特征")
                return None
            
            with torch.no_grad():
                for batch in data_loader:
                    if len(batch) < 2:
                        logging.warning(f"跳过格式不正确的批次")
                        continue
                        
                    inputs, _ = batch
                    
                    if inputs is None or inputs.numel() == 0:
                        logging.warning("跳过空输入批次")
                        continue
                        
                    inputs = inputs.to(self.args.device)
                    
                    # 获取倒数第二层输出
                    feature = get_penultimate_layer(model, inputs)
                    
                    if feature is not None and feature.numel() > 0:
                        features.append(feature)
                    else:
                        logging.warning("获取到空特征，跳过此批次")
                
            # 检查是否提取了任何特征
            if not features:
                logging.error("未能提取任何有效特征")
                return None
            
            # 连接所有批次的特征
            return torch.cat(features, dim=0)
            
        except Exception as e:
            logging.error(f"特征提取失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def compute_mmd_matrix_optimized(self, plr_sequences):
        """
        优化的MMD距离矩阵计算
        
        参数:
        - plr_sequences: 客户端PLR序列列表
        
        返回:
        - mmd_matrix: MMD距离矩阵
        """
        import torch
        import numpy as np
        
        n_clients = len(plr_sequences)
        mmd_matrix = torch.zeros((n_clients, n_clients))
    
    # 定义批量化的高斯核函数
    def batch_gaussian_kernel(x, y, sigma=1.0):
        # 计算 ||x-y||^2 的批量版本
        x_norm = torch.sum(x**2, dim=1).view(-1, 1)
        y_norm = torch.sum(y**2, dim=1).view(1, -1)
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y.transpose(0, 1))
        return torch.exp(-dist / (2 * sigma**2))
    
        for i in range(n_clients):
            for j in range(i, n_clients):  # 只计算上三角矩阵
                if i == j:
                    mmd_matrix[i, j] = 0
                else:
                    x, y = plr_sequences[i], plr_sequences[j]
                    # 批量计算核矩阵
                    xx = batch_gaussian_kernel(x, x)
                    yy = batch_gaussian_kernel(y, y)
                    xy = batch_gaussian_kernel(x, y)
                    
                    # 计算MMD
                    mmd = torch.mean(xx) + torch.mean(yy) - 2 * torch.mean(xy)
                    mmd_matrix[i, j] = mmd.sqrt()
                    mmd_matrix[j, i] = mmd_matrix[i, j]  # 对称性
        
        return mmd_matrix.cpu().numpy()

    def compute_trust_scores_improved(self, mmd_distances, percentile=50, kappa=1.0):
        """
        改进的信任分数计算方法
        
        参数:
        - mmd_distances: MMD距离矩阵
        - percentile: 确定邻居的百分比阈值，默认50%
        - kappa: 温度参数，调整分数敏感度
        
        返回:
        - trust_scores: 改进的客户端信任分数
        """
        import numpy as np
        
        n_clients = mmd_distances.shape[0]
        similarity_scores = np.zeros(n_clients)
        
        # 距离转换为相似度 (距离越小相似度越高)
        max_distance = np.max(mmd_distances) if np.max(mmd_distances) > 0 else 1.0
        similarity_matrix = max_distance - mmd_distances
        
        # 对每个客户端计算加权相似度分数
        for i in range(n_clients):
            # 获取当前客户端与其他客户端的相似度
            similarities = similarity_matrix[i]
            # 排除自身
            other_similarities = np.concatenate([similarities[:i], similarities[i+1:]])
            # 找出相似度最高的一部分邻居
            threshold = np.percentile(other_similarities, 100 - percentile)
            
            # 计算加权相似度分数
            for j in range(n_clients):
                if i != j and similarity_matrix[j, i] >= threshold:
                    # 使用相似度作为权重
                    similarity_scores[i] += similarity_matrix[j, i]
        
        # 归一化并应用softmax
        if np.sum(similarity_scores) > 0:
            similarity_scores = similarity_scores / np.sum(similarity_scores)
        
        exp_scores = np.exp(similarity_scores / kappa)
        trust_scores = exp_scores / np.sum(exp_scores)
        
        return trust_scores

    def prepare_auxiliary_data(self, args=None, dataset_name='cifar10', num_samples=500, seed=42):
        """
        准备用于PLR分析的辅助数据加载器
        
        参数:
        - args: 程序参数，如果为None则使用self.args
        - dataset_name: 数据集名称
        - num_samples: 辅助数据样本数量
        - seed: 随机种子，确保可复现性
        
        返回:
        - auxiliary_data_loader: 辅助数据加载器
        """
        import torch
        import torchvision
        import torchvision.transforms as transforms
        from torch.utils.data import DataLoader, Subset
        import numpy as np
        
        if args is None:
            args = self.args
        
        # 设置随机种子
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # 设置数据转换
        if dataset_name.lower() == 'cifar10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            # 加载完整数据集
            full_dataset = torchvision.datasets.CIFAR10(
                root='./data', train=True, download=True, transform=transform)
        elif dataset_name.lower() == 'mnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            full_dataset = torchvision.datasets.MNIST(
                root='./data', train=True, download=True, transform=transform)
        else:
            raise ValueError(f"不支持的数据集: {dataset_name}")
        
        # 随机选择样本
        dataset_size = len(full_dataset)
        indices = np.random.choice(dataset_size, min(num_samples, dataset_size), replace=False)
        auxiliary_dataset = Subset(full_dataset, indices)
        
        # 创建数据加载器
        batch_size = args.batch_size if hasattr(args, 'batch_size') else 64
        auxiliary_data_loader = DataLoader(
            auxiliary_dataset, 
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
        
        print(f"已准备辅助数据: {len(auxiliary_dataset)}个样本")
        return auxiliary_data_loader

    def agg_avg(self, agent_updates_dict):
        """ classic fed avg """

        sm_updates, total_data = 0, 0
        for _id, update in agent_updates_dict.items():
            n_agent_data = self.agent_data_sizes[_id]
            sm_updates +=  n_agent_data * update
            total_data += n_agent_data
        return  sm_updates / total_data

    
    def agg_mkrum(self, agent_updates_dict):
        krum_param_m = 10
        def _compute_krum_score( vec_grad_list, byzantine_client_num):
            krum_scores = []
            num_client = len(vec_grad_list)
            for i in range(0, num_client):
                dists = []
                for j in range(0, num_client):
                    if i != j:
                        dists.append(
                            torch.norm(vec_grad_list[i]- vec_grad_list[j])
                            .item() ** 2
                        )
                dists.sort()  # ascending
                score = dists[0: num_client - byzantine_client_num - 2]
                krum_scores.append(sum(score))
            return krum_scores

        benign_id = []
        malicious_id = []

        for _id, update in agent_updates_dict.items():
            # local_updates.append(update)
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        # Compute list of scores
        __nbworkers = len(agent_updates_dict)
        krum_scores = _compute_krum_score(agent_updates_dict, self.args.num_corrupt)
        score_index = torch.argsort(
            torch.Tensor(krum_scores)
        ).tolist()  # indices; ascending
        score_index = score_index[0: krum_param_m]

        print('%d clients are selected' % len(score_index))
        return_updates = [agent_updates_dict[i] for i in score_index]


        return sum(return_updates)/len(return_updates)

    def compute_robustLR(self, agent_updates_dict):

        agent_updates_sign = [torch.sign(update) for update in agent_updates_dict.values()]  
        sm_of_signs = torch.abs(sum(agent_updates_sign))
        mask=torch.zeros_like(sm_of_signs)
        mask[sm_of_signs < self.args.theta] = 0
        mask[sm_of_signs >= self.args.theta] = 1
        sm_of_signs[sm_of_signs < self.args.theta] = -self.server_lr
        sm_of_signs[sm_of_signs >= self.args.theta] = self.server_lr
        return sm_of_signs.to(self.args.device), mask

    def agg_mul_metric(self, agent_updates_dict, global_model, flat_global_model):
        local_updates = []
        benign_id = []
        malicious_id = []

        for _id, update in agent_updates_dict.items():
            local_updates.append(update)
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        chosen_clients = malicious_id + benign_id
        num_chosen_clients = len(malicious_id + benign_id)

        vectorize_nets = [update.detach().cpu().numpy() for update in agent_updates_dict.values()]

        cos_dis = [0.0] * len(vectorize_nets)
        length_dis = [0.0] * len(vectorize_nets)
        manhattan_dis = [0.0] * len(vectorize_nets)
        for i, g_i in enumerate(vectorize_nets):
            for j in range(len(vectorize_nets)):
                if i != j:
                    g_j = vectorize_nets[j]

                    cosine_distance = float(
                        (1 - np.dot(g_i, g_j) / (np.linalg.norm(g_i) * np.linalg.norm(g_j))) ** 2)   #Compute the different value of cosine distance
                    manhattan_distance = float(np.linalg.norm(g_i - g_j, ord=1))    #Compute the different value of Manhattan distance
                    length_distance = np.abs(float(np.linalg.norm(g_i) - np.linalg.norm(g_j)))    #Compute the different value of Euclidean distance

                    cos_dis[i] += cosine_distance
                    length_dis[i] += length_distance
                    manhattan_dis[i] += manhattan_distance

        tri_distance = np.vstack([cos_dis, manhattan_dis, length_dis]).T

        cov_matrix = np.cov(tri_distance.T)
        inv_matrix = np.linalg.inv(cov_matrix)

        ma_distances = []
        for i, g_i in enumerate(vectorize_nets):
            t = tri_distance[i]
            ma_dis = np.dot(np.dot(t, inv_matrix), t.T)
            ma_distances.append(ma_dis)

        scores = ma_distances
        print(scores)

        p = 0.3
        p_num = p*len(scores)
        topk_ind = np.argpartition(scores, int(p_num))[:int(p_num)]   #sort

        print(topk_ind)
        current_dict = {}

        for idx in topk_ind:
            current_dict[chosen_clients[idx]] = agent_updates_dict[chosen_clients[idx]]

        update = self.agg_avg(current_dict)

        return update
   
    def agg_foolsgold(self, agent_updates_dict):
        def foolsgold(updates):
            """
            :param updates:
            :return: compute similatiry and return weightings
            """
            n_clients = updates.shape[0]
            cs = smp.cosine_similarity(updates) - np.eye(n_clients)

            maxcs = np.max(cs, axis=1)
            # pardoning
            for i in range(n_clients):
                for j in range(n_clients):
                    if i == j:
                        continue
                    if maxcs[i] < maxcs[j]:
                        cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
            wv = 1 - (np.max(cs, axis=1))

            wv[wv > 1] = 1
            wv[wv < 0] = 0

            alpha = np.max(cs, axis=1)

            # Rescale so that max value is wv
            wv = wv / np.max(wv)
            wv[(wv == 1)] = .99

            # Logit function
            wv = (np.log(wv / (1 - wv)) + 0.5)
            wv[(np.isinf(wv) + wv > 1)] = 1
            wv[(wv < 0)] = 0

            # wv is the weight
            return wv, alpha

        local_updates = []
        benign_id = []
        malicious_id = []

        for _id, update in agent_updates_dict.items():
            local_updates.append(update)
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        names = malicious_id + benign_id
        num_chosen_clients = len(malicious_id + benign_id)

        client_updates = [update.detach().cpu().numpy() for update in agent_updates_dict.values()]
        update_len = np.array(client_updates[0].shape).prod()
        # print("client_updates size", client_models[0].parameters())
        # update_len = len(client_updates)
        # if self.memory is None:
        #     self.memory = np.zeros((self.num_clients, update_len))
        if len(names) < len(client_updates):
            names = np.append([-1], names)  # put in adv

        num_clients = num_chosen_clients
        memory = np.zeros((num_clients, update_len))
        updates = np.zeros((num_clients, update_len))

        for i in range(len(client_updates)):
            # updates[i] = np.reshape(client_updates[i][-2].cpu().data.numpy(), (update_len))
            updates[i] = np.reshape(client_updates[i], (update_len))
            if names[i] in self.memory_dict.keys():
                self.memory_dict[names[i]] += updates[i]
            else:
                self.memory_dict[names[i]] = copy.deepcopy(updates[i])
            memory[i] = self.memory_dict[names[i]]
        # self.memory += updates
        use_memory = False

        if use_memory:
            wv, alpha = foolsgold(None)  # Use FG
        else:
            wv, alpha = foolsgold(updates)  # Use FG
        # logger.info(f'[foolsgold agg] wv: {wv}')
        self.wv_history.append(wv)

        print(len(client_updates), len(wv))


        weighted_updates = [update * wv[i] for update, i in zip(agent_updates_dict.values(), range(len(wv)))]

        aggregated_model = torch.mean(torch.stack(weighted_updates, dim=0), dim=0)
        
        return aggregated_model
    
    def agg_alignins_fedup_standard(self, agent_updates_dict, flat_global_model, global_model, current_round=None):
        """
        AlignIns + FedUP Standard 聚合方法
        结合AlignIns多指标异常检测和标准FedUP剪枝
        """
        # 导入标准实现
        from agg_alignins_fedup_standard import agg_alignins_fedup_standard
        
        # 转换输入格式
        inter_model_updates = torch.stack(list(agent_updates_dict.values()))
        
        # 获取恶意客户端ID（如果有的话）
        malicious_id = getattr(self.args, 'malicious_id', None)
        
        # 调用标准实现
        aggregated_update = agg_alignins_fedup_standard(
            self.args,
            inter_model_updates,
            flat_global_model,
            malicious_id,
            current_round
        )
        
        return aggregated_update

        print(aggregated_model.shape)

        return aggregated_model
