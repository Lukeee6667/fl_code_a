"""
AlignIns Detection Module
AlignIns 异常检测核心逻辑独立模块，用于复用。

包含：
1. AlignInsDetector 类：封装四指标检测逻辑 (Phase 1)
2. weighted_filter_aggregation 函数：封装基于检测结果的加权聚合逻辑 (Phase 2)
"""

import torch
import numpy as np
import logging
from typing import List, Dict, Tuple, Set

class AlignInsDetector:
    """
    AlignIns 异常检测器
    封装了TDA, MPSA, Grad Norm, Mean Cos四指标检测逻辑
    """
    def __init__(self, args):
        self.args = args
        # 检测阈值参数
        self.lambda_s = getattr(args, 'lambda_s', 1.5)  # MPSA阈值
        self.lambda_c = getattr(args, 'lambda_c', 1.5)  # TDA阈值
        self.lambda_g = getattr(args, 'lambda_g', 1.5)  # Grad Norm阈值
        self.lambda_mean_cos = getattr(args, 'lambda_mean_cos', 1.5)  # Mean Cos阈值
        self.sparsity = getattr(args, 'sparsity', 0.1)  # MPSA稀疏度
        
    def detect(self, inter_model_updates: torch.Tensor) -> Dict[str, Set[int]]:
        """
        执行AlignIns四指标异常检测
        
        Args:
            inter_model_updates: 客户端模型更新 [num_clients, model_dim]
            
        Returns:
            Dict: 包含 'benign', 'suspicious', 'malicious' 三个集合
        """
        num_clients = len(inter_model_updates)
        
        # 计算四个指标
        tda_scores = self._calculate_tda(inter_model_updates)
        mpsa_scores = self._calculate_mpsa(inter_model_updates)
        grad_norm_scores = self._calculate_grad_norm(inter_model_updates)
        mean_cos_scores = self._calculate_mean_cos(inter_model_updates)
        
        # MZ-score标准化
        mz_tda = self._mz_score(tda_scores)
        mz_mpsa = self._mz_score(mpsa_scores)
        mz_grad_norm = self._mz_score(grad_norm_scores)
        mz_mean_cos = self._mz_score(mean_cos_scores)
        
        # 三层检测策略
        return self._three_layer_detection(
            mz_tda, mz_mpsa, mz_grad_norm, mz_mean_cos, num_clients
        )

    def _calculate_tda(self, inter_model_updates: torch.Tensor) -> np.ndarray:
        """计算TDA指标"""
        num_clients = len(inter_model_updates)
        tda_scores = np.zeros(num_clients)
        
        for i in range(num_clients):
            others = torch.cat([inter_model_updates[j:j+1] for j in range(num_clients) if j != i])
            if len(others) > 0:
                mean_others = torch.mean(others, dim=0)
                tda_scores[i] = torch.norm(inter_model_updates[i] - mean_others).item()
        
        return tda_scores
    
    def _calculate_mpsa(self, inter_model_updates: torch.Tensor) -> np.ndarray:
        """计算MPSA指标"""
        num_clients = len(inter_model_updates)
        mpsa_scores = np.zeros(num_clients)
        
        for i in range(num_clients):
            update = inter_model_updates[i]
            # 计算稀疏性
            k = int(len(update) * self.sparsity)
            if k > 0:
                _, top_indices = torch.topk(torch.abs(update), k)
                sparse_update = torch.zeros_like(update)
                sparse_update[top_indices] = update[top_indices]
                mpsa_scores[i] = torch.norm(update - sparse_update).item()
        
        return mpsa_scores
    
    def _calculate_grad_norm(self, inter_model_updates: torch.Tensor) -> np.ndarray:
        """计算梯度范数指标"""
        return torch.norm(inter_model_updates, dim=1).cpu().numpy()
    
    def _calculate_mean_cos(self, inter_model_updates: torch.Tensor) -> np.ndarray:
        """计算平均余弦相似度指标"""
        num_clients = len(inter_model_updates)
        mean_cos_scores = np.zeros(num_clients)
        
        for i in range(num_clients):
            similarities = []
            for j in range(num_clients):
                if i != j:
                    cos_sim = torch.nn.functional.cosine_similarity(
                        inter_model_updates[i].unsqueeze(0),
                        inter_model_updates[j].unsqueeze(0)
                    ).item()
                    similarities.append(cos_sim)
            
            mean_cos_scores[i] = np.mean(similarities) if similarities else 0
        
        return mean_cos_scores
    
    def _mz_score(self, scores: np.ndarray) -> np.ndarray:
        """MZ-score标准化"""
        median = np.median(scores)
        mad = np.median(np.abs(scores - median))
        if mad == 0:
            return np.zeros_like(scores)
        return 0.6745 * (scores - median) / mad
    
    def _three_layer_detection(self, 
                              mz_tda: np.ndarray,
                              mz_mpsa: np.ndarray, 
                              mz_grad_norm: np.ndarray,
                              mz_mean_cos: np.ndarray,
                              num_clients: int) -> Dict[str, Set[int]]:
        """三层检测策略"""
        # 第一层：严格阈值检测明确良性客户端
        strict_factor = 0.8
        strict_lambda_s = self.lambda_s * strict_factor
        strict_lambda_c = self.lambda_c * strict_factor
        strict_lambda_g = self.lambda_g * strict_factor
        strict_lambda_mean_cos = self.lambda_mean_cos * strict_factor
        
        benign_tda = {i for i in range(num_clients) if mz_tda[i] < strict_lambda_c}
        benign_mpsa = {i for i in range(num_clients) if mz_mpsa[i] < strict_lambda_s}
        benign_grad = {i for i in range(num_clients) if mz_grad_norm[i] < strict_lambda_g}
        benign_cos = {i for i in range(num_clients) if mz_mean_cos[i] < strict_lambda_mean_cos}
        
        # 明确良性：通过所有严格检测
        clear_benign = benign_tda & benign_mpsa & benign_grad & benign_cos
        
        # 第二层：标准阈值检测明确恶意客户端
        malicious_tda = {i for i in range(num_clients) if mz_tda[i] > self.lambda_c}
        malicious_mpsa = {i for i in range(num_clients) if mz_mpsa[i] > self.lambda_s}
        malicious_grad = {i for i in range(num_clients) if mz_grad_norm[i] > self.lambda_g}
        malicious_cos = {i for i in range(num_clients) if mz_mean_cos[i] > self.lambda_mean_cos}
        
        # 明确恶意：任一指标超过标准阈值
        clear_malicious = malicious_tda | malicious_mpsa | malicious_grad | malicious_cos
        
        # 第三层：可疑客户端（介于良性和恶意之间）
        all_clients = set(range(num_clients))
        suspicious = all_clients - clear_benign - clear_malicious
        
        return {
            'benign': clear_benign,
            'suspicious': suspicious,
            'malicious': clear_malicious
        }

def weighted_filter_aggregation(inter_model_updates: torch.Tensor,
                              detection_results: Dict[str, Set[int]],
                              benign_weight: float = 1.0,
                              suspicious_weight: float = 0.5) -> torch.Tensor:
    """
    基于检测结果的加权聚合逻辑 (Phase 2)
    
    Args:
        inter_model_updates: 模型更新张量
        detection_results: 检测结果字典
        benign_weight: 良性客户端权重
        suspicious_weight: 可疑客户端权重
        
    Returns:
        aggregated_update: 聚合后的更新
    """
    benign_clients = detection_results['benign']
    suspicious_clients = detection_results['suspicious']
    malicious_clients = detection_results['malicious']
    flat_global_model = torch.zeros_like(inter_model_updates[0])
    
    # 只使用良性和可疑客户端，完全排除恶意客户端
    valid_clients = benign_clients | suspicious_clients
    
    if len(valid_clients) == 0:
        logging.warning("没有有效客户端，返回零更新")
        return flat_global_model
    
    # 计算权重
    client_weights = {}
    total_weight = 0
    
    for client_id in valid_clients:
        if client_id in benign_clients:
            weight = benign_weight
        else:  # suspicious clients
            weight = suspicious_weight
        
        client_weights[client_id] = weight
        total_weight += weight
    
    # 加权聚合
    aggregated_update = torch.zeros_like(flat_global_model)
    for client_id in valid_clients:
        weight = client_weights[client_id] / total_weight
        aggregated_update += weight * inter_model_updates[client_id]
    
    logging.info(f"Aggregated with {len(valid_clients)} clients (excluded {len(malicious_clients)} malicious)")
    return aggregated_update
