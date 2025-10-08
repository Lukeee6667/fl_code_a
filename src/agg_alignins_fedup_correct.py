"""
AlignIns + FedUP Correct Implementation
正确的AlignIns检测 + FedUP模型权重剪枝实现

流程：
1. AlignIns四指标异常检测
2. 过滤恶意客户端并聚合
3. 对聚合后的模型进行标准FedUP权重剪枝

作者：AI Assistant
日期：2024
"""

import torch
import numpy as np
import logging
from typing import List, Dict, Tuple, Set
import copy
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class AlignInsFedUPCorrectAggregator:
    """
    正确的AlignIns + FedUP实现
    """
    
    def __init__(self, args):
        self.args = args
        # AlignIns参数
        self.lambda_s = getattr(args, 'lambda_s', 1.5)  # MPSA阈值
        self.lambda_c = getattr(args, 'lambda_c', 1.5)  # TDA阈值
        self.lambda_g = getattr(args, 'lambda_g', 1.5)  # Grad Norm阈值
        self.lambda_mean_cos = getattr(args, 'lambda_mean_cos', 1.5)  # Mean Cos阈值
        self.sparsity = getattr(args, 'sparsity', 0.1)  # MPSA稀疏度
        
        # FedUP参数
        self.fedup_p_max = getattr(args, 'fedup_p_max', 0.15)  # 最大剪枝率
        self.fedup_p_min = getattr(args, 'fedup_p_min', 0.01)  # 最小剪枝率
        self.fedup_gamma = getattr(args, 'fedup_gamma', 5)     # 曲线陡度参数
        
        # 聚合参数
        self.norm_clip = getattr(args, 'norm_clip', 10.0)  # 范数裁剪
        
    def aggregate(self, 
                  inter_model_updates: torch.Tensor,
                  flat_global_model: torch.Tensor,
                  global_model: torch.nn.Module,
                  malicious_id: List[int] = None,
                  current_round: int = None) -> Tuple[torch.Tensor, torch.nn.Module]:
        """
        主聚合方法
        
        Args:
            inter_model_updates: 客户端模型更新 [num_clients, model_dim]
            flat_global_model: 扁平化的全局模型参数
            global_model: 全局模型（用于FedUP剪枝）
            malicious_id: 已知恶意客户端ID（用于评估）
            current_round: 当前轮次
            
        Returns:
            aggregated_update: 聚合后的模型更新
            pruned_model: 经过FedUP剪枝的模型
        """
        num_clients = len(inter_model_updates)
        
        logging.info(f"=== AlignIns+FedUP Correct Implementation (Round {current_round}) ===")
        logging.info(f"Total clients: {num_clients}")
        
        # ========== 第一阶段：AlignIns四指标异常检测 ==========
        detection_results = self._alignins_detection(inter_model_updates, flat_global_model)
        
        benign_clients = detection_results['benign']
        suspicious_clients = detection_results['suspicious'] 
        malicious_clients = detection_results['malicious']
        
        logging.info(f"Detection results - Benign: {list(benign_clients)}")
        logging.info(f"Detection results - Suspicious: {list(suspicious_clients)}")
        logging.info(f"Detection results - Malicious: {list(malicious_clients)}")
        
        # ========== 第二阶段：过滤聚合 ==========
        # 只使用良性和可疑客户端，完全排除恶意客户端
        valid_clients = benign_clients | suspicious_clients
        
        if len(valid_clients) == 0:
            logging.warning("没有有效客户端，返回零更新")
            return torch.zeros_like(flat_global_model), copy.deepcopy(global_model)
        
        # 计算权重（良性客户端权重更高）
        client_weights = {}
        total_weight = 0
        
        for client_id in valid_clients:
            if client_id in benign_clients:
                weight = 1.0  # 良性客户端全权重
            else:  # suspicious clients
                weight = 0.5  # 可疑客户端降权
            
            client_weights[client_id] = weight
            total_weight += weight
        
        # 加权聚合
        aggregated_update = torch.zeros_like(flat_global_model)
        for client_id in valid_clients:
            weight = client_weights[client_id] / total_weight
            aggregated_update += weight * inter_model_updates[client_id]
        
        logging.info(f"Aggregated with {len(valid_clients)} clients (excluded {len(malicious_clients)} malicious)")
        
        # ========== 第三阶段：标准FedUP模型权重剪枝 ==========
        if len(malicious_clients) > 0:
            # 只有检测到恶意客户端时才进行FedUP剪枝
            pruned_model = self._standard_fedup_model_pruning(
                global_model,
                inter_model_updates,
                malicious_clients,
                benign_clients
            )
            logging.info("Applied FedUP model weight pruning")
        else:
            # 没有恶意客户端，不需要剪枝
            pruned_model = copy.deepcopy(global_model)
            logging.info("No malicious clients detected, skipping FedUP pruning")
        
        # 性能评估
        if malicious_id is not None:
            self._evaluate_performance(detection_results, malicious_id, num_clients)
            
        return aggregated_update, pruned_model
    
    def _alignins_detection(self, 
                           inter_model_updates: torch.Tensor,
                           flat_global_model: torch.Tensor) -> Dict[str, Set[int]]:
        """
        AlignIns四指标异常检测
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
    
    def _standard_fedup_model_pruning(self,
                                     global_model: torch.nn.Module,
                                     inter_model_updates: torch.Tensor,
                                     malicious_clients: Set[int],
                                     benign_clients: Set[int]) -> torch.nn.Module:
        """
        标准FedUP模型权重剪枝（这才是真正的FedUP剪枝）
        """
        if len(malicious_clients) == 0 or len(benign_clients) == 0:
            return copy.deepcopy(global_model)
        
        # 计算恶意和良性客户端的平均更新
        malicious_updates = torch.stack([inter_model_updates[i] for i in malicious_clients])
        benign_updates = torch.stack([inter_model_updates[i] for i in benign_clients])
        
        avg_malicious_update = torch.mean(malicious_updates, dim=0)
        avg_benign_update = torch.mean(benign_updates, dim=0)
        
        # 计算自适应剪枝比例
        adaptive_pruning_ratio = self._calculate_fedup_adaptive_ratio(benign_updates)
        
        # 计算权重重要性（论文标准公式）
        # rank = (w_malicious - w_benign)^2 * |w_global|
        diff_squared = torch.pow(avg_malicious_update - avg_benign_update, 2)
        weight_importance = diff_squared * torch.abs(parameters_to_vector(global_model.parameters()))
        
        # 选择需要剪枝的权重
        total_params = len(weight_importance)
        num_pruned = int(total_params * adaptive_pruning_ratio)
        
        if num_pruned > 0:
            # 选择重要性最高的权重进行剪枝
            _, top_indices = torch.topk(weight_importance, num_pruned, largest=True)
            
            # 创建剪枝掩码
            pruning_mask = torch.ones_like(weight_importance)
            pruning_mask[top_indices] = 0.0
            
            # 应用剪枝到模型权重
            pruned_model = copy.deepcopy(global_model)
            pruned_params = pruning_mask * parameters_to_vector(pruned_model.parameters())
            vector_to_parameters(pruned_params, pruned_model.parameters())
            
            logging.info(f"FedUP pruning: {num_pruned}/{total_params} parameters ({adaptive_pruning_ratio:.4f})")
            
            return pruned_model
        else:
            return copy.deepcopy(global_model)
    
    def _calculate_fedup_adaptive_ratio(self, benign_updates: torch.Tensor) -> float:
        """
        计算FedUP自适应剪枝比例
        """
        if len(benign_updates) < 2:
            return self.fedup_p_min
        
        # 计算良性客户端间的余弦相似度
        similarities = []
        for i in range(len(benign_updates)):
            for j in range(i + 1, len(benign_updates)):
                cos_sim = torch.nn.functional.cosine_similarity(
                    benign_updates[i].unsqueeze(0), 
                    benign_updates[j].unsqueeze(0)
                ).item()
                similarities.append(cos_sim)
        
        # 计算平均相似度作为收敛指标
        avg_similarity = np.mean(similarities)
        
        # 归一化到[0,1]区间
        z = max(0, min(1, (avg_similarity + 1) / 2))  # cosine similarity范围[-1,1]
        
        # 应用FedUP公式：P = (P_max - P_min) * z^γ + P_min
        adaptive_ratio = (self.fedup_p_max - self.fedup_p_min) * (z ** self.fedup_gamma) + self.fedup_p_min
        
        return adaptive_ratio
    
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
    
    def _evaluate_performance(self, detection_results: Dict[str, Set[int]], 
                            true_malicious: List[int], num_clients: int):
        """评估检测性能"""
        detected_malicious = detection_results['malicious']
        true_malicious_set = set(true_malicious)
        
        tp = len(detected_malicious & true_malicious_set)
        fp = len(detected_malicious - true_malicious_set)
        fn = len(true_malicious_set - detected_malicious)
        tn = num_clients - tp - fp - fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        logging.info(f"Detection Performance - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")


def agg_alignins_fedup_correct(inter_model_updates: torch.Tensor,
                              flat_global_model: torch.Tensor,
                              global_model: torch.nn.Module,
                              args,
                              malicious_id: List[int] = None,
                              current_round: int = None) -> Tuple[torch.Tensor, torch.nn.Module]:
    """
    正确的AlignIns + FedUP聚合方法
    """
    aggregator = AlignInsFedUPCorrectAggregator(args)
    return aggregator.aggregate(
        inter_model_updates, 
        flat_global_model, 
        global_model,
        malicious_id, 
        current_round
    )