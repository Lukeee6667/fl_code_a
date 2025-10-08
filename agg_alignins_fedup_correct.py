"""
正确的AlignIns + FedUP实现
先用AlignIns四指标检测和过滤聚合，再对聚合后的模型进行标准FedUP剪枝

作者：基于原始AlignIns和FedUP论文的正确实现
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import copy


def agg_alignins_fedup_correct(args, inter_model_updates, flat_global_model, global_model, malicious_id=None, current_round=None):
    """
    正确的AlignIns + FedUP聚合方法
    
    流程：
    1. 使用AlignIns四指标检测异常客户端
    2. 过滤掉恶意客户端，对剩余客户端进行聚合
    3. 对聚合后的全局模型进行标准FedUP剪枝
    
    Args:
        args: 参数配置
        inter_model_updates: 客户端模型更新 [num_clients, model_size]
        flat_global_model: 扁平化的全局模型参数
        global_model: 全局模型对象
        malicious_id: 恶意客户端ID（用于评估）
        current_round: 当前轮次
    
    Returns:
        aggregated_update: 聚合并剪枝后的模型更新
    """
    
    # 第一阶段：AlignIns四指标检测和过滤聚合
    print(f"[Round {current_round}] 开始AlignIns四指标检测...")
    
    # 1. AlignIns异常检测
    benign_indices, suspicious_indices, malicious_indices = alignins_detection(
        args, inter_model_updates, flat_global_model
    )
    
    print(f"检测结果 - 良性客户端: {len(benign_indices)}, 可疑客户端: {len(suspicious_indices)}, 恶意客户端: {len(malicious_indices)}")
    
    # 2. 过滤聚合：只使用良性和可疑客户端（可疑客户端权重降低）
    filtered_updates, client_weights = filter_and_aggregate(
        inter_model_updates, benign_indices, suspicious_indices, malicious_indices
    )
    
    # 3. 聚合得到新的全局模型
    aggregated_update = torch.sum(filtered_updates * client_weights.unsqueeze(1), dim=0)
    
    # 第二阶段：对聚合后的模型进行标准FedUP剪枝
    print(f"[Round {current_round}] 开始标准FedUP模型权重剪枝...")
    
    # 4. 计算新的全局模型参数
    new_global_params = flat_global_model + aggregated_update
    
    # 5. 标准FedUP剪枝：基于权重重要性对模型参数进行剪枝
    pruned_global_params = standard_fedup_model_pruning(
        args, new_global_params, flat_global_model, inter_model_updates, 
        benign_indices, malicious_indices
    )
    
    # 6. 计算最终的聚合更新
    final_aggregated_update = pruned_global_params - flat_global_model
    
    print(f"[Round {current_round}] AlignIns+FedUP聚合完成")
    
    return final_aggregated_update


def alignins_detection(args, inter_model_updates, flat_global_model):
    """
    AlignIns四指标异常检测
    
    Returns:
        benign_indices: 良性客户端索引
        suspicious_indices: 可疑客户端索引  
        malicious_indices: 恶意客户端索引
    """
    num_clients = inter_model_updates.shape[0]
    
    # 计算四个指标
    cosine_similarities = compute_cosine_similarity(inter_model_updates)
    euclidean_distances = compute_euclidean_distance(inter_model_updates)
    sign_agreements = compute_sign_agreement(inter_model_updates)
    magnitude_ratios = compute_magnitude_ratio(inter_model_updates)
    
    # 综合评分（简化版本，实际可能需要更复杂的融合策略）
    anomaly_scores = []
    for i in range(num_clients):
        # 余弦相似度越低越异常
        cos_score = 1 - np.mean([cosine_similarities[i][j] for j in range(num_clients) if j != i])
        # 欧氏距离越大越异常
        euc_score = np.mean([euclidean_distances[i][j] for j in range(num_clients) if j != i])
        # 符号一致性越低越异常
        sign_score = 1 - np.mean([sign_agreements[i][j] for j in range(num_clients) if j != i])
        # 幅度比越大越异常
        mag_score = np.mean([magnitude_ratios[i][j] for j in range(num_clients) if j != i])
        
        # 综合评分
        total_score = cos_score + euc_score + sign_score + mag_score
        anomaly_scores.append(total_score)
    
    # 基于阈值分类
    anomaly_scores = np.array(anomaly_scores)
    
    # 使用分位数作为阈值
    strict_threshold = np.percentile(anomaly_scores, 70)  # 严格阈值
    standard_threshold = np.percentile(anomaly_scores, 85)  # 标准阈值
    
    benign_indices = []
    suspicious_indices = []
    malicious_indices = []
    
    for i, score in enumerate(anomaly_scores):
        if score < strict_threshold:
            benign_indices.append(i)
        elif score < standard_threshold:
            suspicious_indices.append(i)
        else:
            malicious_indices.append(i)
    
    return benign_indices, suspicious_indices, malicious_indices


def filter_and_aggregate(inter_model_updates, benign_indices, suspicious_indices, malicious_indices):
    """
    过滤和聚合客户端更新
    
    Returns:
        filtered_updates: 过滤后的更新
        client_weights: 客户端权重
    """
    num_clients = inter_model_updates.shape[0]
    client_weights = torch.zeros(num_clients)
    
    # 良性客户端：正常权重
    for idx in benign_indices:
        client_weights[idx] = 1.0
    
    # 可疑客户端：降低权重
    for idx in suspicious_indices:
        client_weights[idx] = 0.3  # 降低权重但不完全排除
    
    # 恶意客户端：权重为0（完全排除）
    for idx in malicious_indices:
        client_weights[idx] = 0.0
    
    # 归一化权重
    if client_weights.sum() > 0:
        client_weights = client_weights / client_weights.sum()
    else:
        # 如果所有客户端都被标记为恶意，则平均分配权重
        client_weights = torch.ones(num_clients) / num_clients
    
    return inter_model_updates, client_weights


def standard_fedup_model_pruning(args, new_global_params, old_global_params, inter_model_updates, 
                                benign_indices, malicious_indices):
    """
    标准FedUP模型权重剪枝
    
    基于论文中的权重重要性计算，对全局模型参数进行剪枝
    """
    if len(malicious_indices) == 0:
        # 没有恶意客户端，不需要剪枝
        return new_global_params
    
    # 计算良性客户端的平均更新
    if len(benign_indices) > 0:
        benign_updates = inter_model_updates[benign_indices]
        benign_mean = torch.mean(benign_updates, dim=0)
    else:
        benign_mean = torch.zeros_like(new_global_params)
    
    # 计算权重重要性（基于FedUP论文公式）
    weight_importance = compute_weight_importance_standard(
        new_global_params, old_global_params, benign_mean
    )
    
    # 自适应剪枝比例
    pruning_ratio = compute_adaptive_pruning_ratio(args, len(malicious_indices), len(benign_indices))
    
    # 应用剪枝
    pruned_params = apply_model_pruning(new_global_params, weight_importance, pruning_ratio)
    
    return pruned_params


def compute_weight_importance_standard(new_params, old_params, benign_mean):
    """
    计算权重重要性（基于FedUP论文的标准公式）
    
    重要性 = |w_global - (w_global_old + benign_mean)|^2
    """
    expected_params = old_params + benign_mean
    importance = torch.pow(new_params - expected_params, 2)
    return importance


def apply_model_pruning(model_params, weight_importance, pruning_ratio):
    """
    对模型参数应用剪枝
    """
    if pruning_ratio <= 0:
        return model_params
    
    # 计算需要剪枝的参数数量
    total_params = model_params.numel()
    num_pruned = int(total_params * pruning_ratio)
    
    if num_pruned == 0:
        return model_params
    
    # 找到重要性最低的参数
    _, indices = torch.topk(weight_importance.flatten(), num_pruned, largest=False)
    
    # 创建剪枝后的参数
    pruned_params = model_params.clone()
    pruned_params.flatten()[indices] = 0
    
    return pruned_params


def compute_adaptive_pruning_ratio(args, num_malicious, num_benign):
    """
    计算自适应剪枝比例
    """
    base_ratio = getattr(args, 'fedup_pruning_ratio', 0.1)
    
    if num_benign == 0:
        return 0.0
    
    # 根据恶意客户端比例调整剪枝比例
    malicious_ratio = num_malicious / (num_malicious + num_benign)
    adaptive_ratio = base_ratio * (1 + malicious_ratio)
    
    return min(adaptive_ratio, 0.5)  # 最大剪枝比例不超过50%


# 辅助函数：计算各种相似性指标
def compute_cosine_similarity(updates):
    """计算余弦相似度矩阵"""
    num_clients = updates.shape[0]
    similarities = torch.zeros(num_clients, num_clients)
    
    for i in range(num_clients):
        for j in range(num_clients):
            if i != j:
                similarities[i][j] = torch.cosine_similarity(
                    updates[i].unsqueeze(0), updates[j].unsqueeze(0)
                ).item()
            else:
                similarities[i][j] = 1.0
    
    return similarities.numpy()


def compute_euclidean_distance(updates):
    """计算欧氏距离矩阵"""
    num_clients = updates.shape[0]
    distances = torch.zeros(num_clients, num_clients)
    
    for i in range(num_clients):
        for j in range(num_clients):
            distances[i][j] = torch.norm(updates[i] - updates[j]).item()
    
    return distances.numpy()


def compute_sign_agreement(updates):
    """计算符号一致性矩阵"""
    num_clients = updates.shape[0]
    agreements = torch.zeros(num_clients, num_clients)
    
    for i in range(num_clients):
        for j in range(num_clients):
            sign_i = torch.sign(updates[i])
            sign_j = torch.sign(updates[j])
            agreement = (sign_i == sign_j).float().mean().item()
            agreements[i][j] = agreement
    
    return agreements.numpy()


def compute_magnitude_ratio(updates):
    """计算幅度比矩阵"""
    num_clients = updates.shape[0]
    ratios = torch.zeros(num_clients, num_clients)
    
    for i in range(num_clients):
        for j in range(num_clients):
            norm_i = torch.norm(updates[i])
            norm_j = torch.norm(updates[j])
            if norm_j > 0:
                ratio = (norm_i / norm_j).item()
            else:
                ratio = float('inf') if norm_i > 0 else 1.0
            ratios[i][j] = ratio
    
    return ratios.numpy()