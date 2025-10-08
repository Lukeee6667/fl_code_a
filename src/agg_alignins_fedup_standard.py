"""
AlignIns + FedUP Standard Implementation
结合AlignIns多指标异常检测和标准FedUP剪枝的改进聚合方法

基于论文：
- AlignIns: Improving Adversarial Robustness by Aligning Gradients
- FedUP: Efficient Pruning-based Federated Unlearning for Model Poisoning Attacks

作者：AI Assistant
日期：2024
"""

import torch
import numpy as np
import logging
from typing import List, Dict, Tuple, Set
import copy


class AlignInsFedUPStandardAggregator:
    """
    结合AlignIns多指标异常检测和标准FedUP剪枝的聚合器
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
        self.alpha = getattr(args, 'fedup_alpha', 0.1)  # 基础剪枝比例
        self.beta = getattr(args, 'fedup_beta', 0.5)   # 自适应调整因子
        self.gamma = getattr(args, 'fedup_gamma', 0.8)  # 良性相似度阈值
        
        # 聚合参数
        self.norm_clip = getattr(args, 'norm_clip', 10.0)  # 范数裁剪
        
    def aggregate(self, 
                  inter_model_updates: torch.Tensor,
                  flat_global_model: torch.Tensor,
                  malicious_id: List[int] = None,
                  current_round: int = None) -> torch.Tensor:
        """
        主聚合方法
        
        Args:
            inter_model_updates: 客户端模型更新 [num_clients, model_dim]
            flat_global_model: 扁平化的全局模型参数
            malicious_id: 已知恶意客户端ID（用于评估）
            current_round: 当前轮次
            
        Returns:
            聚合后的模型更新
        """
        num_clients = len(inter_model_updates)
        
        logging.info(f"=== AlignIns+FedUP Standard Aggregation (Round {current_round}) ===")
        logging.info(f"Total clients: {num_clients}")
        
        # ========== 第一阶段：AlignIns多指标异常检测 ==========
        detection_results = self._alignins_detection(inter_model_updates, flat_global_model)
        
        benign_clients = detection_results['benign']
        suspicious_clients = detection_results['suspicious'] 
        malicious_clients = detection_results['malicious']
        
        logging.info(f"Detection results - Benign: {list(benign_clients)}")
        logging.info(f"Detection results - Suspicious: {list(suspicious_clients)}")
        logging.info(f"Detection results - Malicious: {list(malicious_clients)}")
        
        # ========== 第二阶段：标准FedUP剪枝处理 ==========
        processed_updates = {}
        
        # 1. 良性客户端直接使用原始更新
        for client_id in benign_clients:
            processed_updates[client_id] = {
                'update': inter_model_updates[client_id].clone(),
                'weight': 1.0,
                'type': 'benign'
            }
            
        # 2. 可疑客户端应用标准FedUP剪枝
        if len(suspicious_clients) > 0 and len(benign_clients) > 0:
            pruned_updates = self._standard_fedup_pruning(
                inter_model_updates, 
                suspicious_clients, 
                benign_clients
            )
            
            for client_id in suspicious_clients:
                # 根据与良性客户端的相似度计算权重
                similarity_weight = self._calculate_similarity_weight(
                    pruned_updates[client_id], 
                    inter_model_updates, 
                    benign_clients
                )
                
                processed_updates[client_id] = {
                    'update': pruned_updates[client_id],
                    'weight': similarity_weight,
                    'type': 'suspicious_pruned'
                }
        
        # 3. 恶意客户端完全排除
        logging.info(f"Excluded malicious clients: {list(malicious_clients)}")
        
        # ========== 第三阶段：加权聚合 ==========
        aggregated_update = self._weighted_aggregation(processed_updates)
        
        # 性能评估
        if malicious_id is not None:
            self._evaluate_performance(detection_results, malicious_id, num_clients)
            
        return aggregated_update
    
    def _alignins_detection(self, 
                           inter_model_updates: torch.Tensor, 
                           flat_global_model: torch.Tensor) -> Dict[str, Set[int]]:
        """
        AlignIns多指标异常检测
        """
        num_clients = len(inter_model_updates)
        
        # 计算四个核心指标
        tda_list = []
        mpsa_list = []
        grad_norm_list = []
        mean_cos_list = []
        
        major_sign = torch.sign(torch.sum(torch.sign(inter_model_updates), dim=0))
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        mean_update = torch.mean(inter_model_updates, dim=0)
        
        for i in range(num_clients):
            # TDA: 与全局模型的余弦相似度
            tda_list.append(cos(inter_model_updates[i], flat_global_model).item())
            
            # MPSA: 主要符号一致性
            _, init_indices = torch.topk(
                torch.abs(inter_model_updates[i]), 
                int(len(inter_model_updates[i]) * self.sparsity)
            )
            mpsa_score = (torch.sum(
                torch.sign(inter_model_updates[i][init_indices]) == major_sign[init_indices]
            ) / torch.numel(inter_model_updates[i][init_indices])).item()
            mpsa_list.append(mpsa_score)
            
            # Grad Norm: 梯度范数
            grad_norm_list.append(torch.norm(inter_model_updates[i]).item())
            
            # Mean Cos: 与平均更新的余弦相似度
            mean_cos_list.append(cos(inter_model_updates[i], mean_update).item())
        
        logging.info(f'TDA scores: {[round(x, 4) for x in tda_list]}')
        logging.info(f'MPSA scores: {[round(x, 4) for x in mpsa_list]}')
        logging.info(f'Grad Norm scores: {[round(x, 4) for x in grad_norm_list]}')
        logging.info(f'Mean Cos scores: {[round(x, 4) for x in mean_cos_list]}')
        
        # MZ-score标准化
        mz_tda = self._calculate_mz_scores(tda_list)
        mz_mpsa = self._calculate_mz_scores(mpsa_list)
        mz_grad_norm = self._calculate_mz_scores(grad_norm_list)
        mz_mean_cos = self._calculate_mz_scores(mean_cos_list)
        
        logging.info(f'MZ-TDA: {[round(x, 4) for x in mz_tda]}')
        logging.info(f'MZ-MPSA: {[round(x, 4) for x in mz_mpsa]}')
        logging.info(f'MZ-Grad Norm: {[round(x, 4) for x in mz_grad_norm]}')
        logging.info(f'MZ-Mean Cos: {[round(x, 4) for x in mz_mean_cos]}')
        
        # 三层检测策略
        return self._three_tier_detection(
            mz_tda, mz_mpsa, mz_grad_norm, mz_mean_cos, num_clients
        )
    
    def _calculate_mz_scores(self, values: List[float]) -> List[float]:
        """计算MZ-score（基于中位数的标准化分数）"""
        if len(values) <= 1:
            return [0.0] * len(values)
            
        median_val = np.median(values)
        std_val = np.std(values)
        
        if std_val == 0:
            return [0.0] * len(values)
            
        return [abs(val - median_val) / std_val for val in values]
    
    def _three_tier_detection(self, 
                             mz_tda: List[float], 
                             mz_mpsa: List[float], 
                             mz_grad_norm: List[float], 
                             mz_mean_cos: List[float], 
                             num_clients: int) -> Dict[str, Set[int]]:
        """
        三层检测策略
        """
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
    
    def _standard_fedup_pruning(self, 
                               inter_model_updates: torch.Tensor,
                               suspicious_clients: Set[int],
                               benign_clients: Set[int]) -> Dict[int, torch.Tensor]:
        """
        标准FedUP剪枝算法（符合论文规范）
        """
        pruned_updates = {}
        
        # 计算良性客户端的平均更新作为参考
        benign_updates = torch.stack([inter_model_updates[i] for i in benign_clients])
        benign_mean = torch.mean(benign_updates, dim=0)
        
        # 计算良性客户端间的平均相似度（收敛指标）
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        benign_similarities = []
        benign_list = list(benign_clients)
        
        for i in range(len(benign_list)):
            for j in range(i + 1, len(benign_list)):
                sim = cos(inter_model_updates[benign_list[i]], 
                         inter_model_updates[benign_list[j]]).item()
                benign_similarities.append(sim)
        
        # 良性客户端收敛状态 z
        z = np.mean(benign_similarities) if benign_similarities else 0.5
        
        logging.info(f"Benign convergence state z: {z:.4f}")
        
        # 对每个可疑客户端应用标准FedUP剪枝
        for client_id in suspicious_clients:
            client_update = inter_model_updates[client_id]
            
            # 1. 计算权重重要性（论文标准公式）
            weight_importance = self._calculate_weight_importance_standard(
                client_update, benign_mean
            )
            
            # 2. 计算自适应剪枝比例
            adaptive_ratio = self._calculate_adaptive_pruning_ratio(z)
            
            # 3. 应用剪枝
            pruned_update = self._apply_pruning(client_update, weight_importance, adaptive_ratio)
            
            pruned_updates[client_id] = pruned_update
            
            logging.info(f"Client {client_id}: adaptive_ratio={adaptive_ratio:.4f}")
        
        return pruned_updates
    
    def _calculate_weight_importance_standard(self, 
                                            client_update: torch.Tensor, 
                                            benign_mean: torch.Tensor) -> torch.Tensor:
        """
        计算权重重要性（符合论文标准）
        使用差异的平方作为重要性度量
        """
        # 论文公式：importance = (w_i - w_benign)^2
        importance = torch.pow(client_update - benign_mean, 2)
        return importance
    
    def _calculate_adaptive_pruning_ratio(self, z: float) -> float:
        """
        计算自适应剪枝比例（符合论文标准）
        """
        # 论文公式：p = α + β * (1 - z)
        # z 接近1表示良性客户端高度收敛，需要更少剪枝
        # z 接近0表示良性客户端分散，需要更多剪枝
        adaptive_ratio = self.alpha + self.beta * (1 - z)
        
        # 确保剪枝比例在合理范围内
        adaptive_ratio = max(0.05, min(0.8, adaptive_ratio))
        
        return adaptive_ratio
    
    def _apply_pruning(self, 
                      client_update: torch.Tensor, 
                      weight_importance: torch.Tensor, 
                      pruning_ratio: float) -> torch.Tensor:
        """
        应用剪枝操作
        """
        # 计算需要剪枝的参数数量
        total_params = len(client_update)
        num_pruned = int(total_params * pruning_ratio)
        
        if num_pruned == 0:
            return client_update.clone()
        
        # 找到重要性最低的参数进行剪枝
        _, pruned_indices = torch.topk(weight_importance, num_pruned, largest=False)
        
        # 创建剪枝后的更新
        pruned_update = client_update.clone()
        pruned_update[pruned_indices] = 0.0
        
        return pruned_update
    
    def _calculate_similarity_weight(self, 
                                   pruned_update: torch.Tensor,
                                   all_updates: torch.Tensor,
                                   benign_clients: Set[int]) -> float:
        """
        计算剪枝后更新与良性客户端的相似度权重
        """
        if len(benign_clients) == 0:
            return 0.5
        
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        similarities = []
        
        for benign_id in benign_clients:
            sim = cos(pruned_update, all_updates[benign_id]).item()
            similarities.append(max(0, sim))  # 只考虑正相似度
        
        avg_similarity = np.mean(similarities)
        
        # 将相似度转换为权重（相似度越高，权重越大）
        weight = min(1.0, max(0.1, avg_similarity))
        
        return weight
    
    def _weighted_aggregation(self, processed_updates: Dict[int, Dict]) -> torch.Tensor:
        """
        加权聚合处理后的更新
        """
        if not processed_updates:
            raise ValueError("No valid updates for aggregation")
        
        # 收集所有更新和权重
        updates = []
        weights = []
        
        for client_id, update_info in processed_updates.items():
            updates.append(update_info['update'])
            weights.append(update_info['weight'])
        
        # 转换为张量
        updates_tensor = torch.stack(updates)
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        
        # 归一化权重
        weights_tensor = weights_tensor / torch.sum(weights_tensor)
        
        # 加权平均
        aggregated = torch.sum(updates_tensor * weights_tensor.unsqueeze(1), dim=0)
        
        # 应用范数裁剪
        norm = torch.norm(aggregated)
        if norm > self.norm_clip:
            aggregated = aggregated * (self.norm_clip / norm)
            logging.info(f"Applied norm clipping: {norm:.4f} -> {self.norm_clip}")
        
        logging.info(f"Aggregation weights: {[round(w, 4) for w in weights]}")
        logging.info(f"Final aggregated norm: {torch.norm(aggregated).item():.4f}")
        
        return aggregated
    
    def _evaluate_performance(self, 
                            detection_results: Dict[str, Set[int]], 
                            true_malicious: List[int], 
                            total_clients: int):
        """
        评估检测性能
        """
        true_malicious_set = set(true_malicious)
        true_benign_set = set(range(total_clients)) - true_malicious_set
        
        detected_malicious = detection_results['malicious']
        detected_benign = detection_results['benign']
        detected_suspicious = detection_results['suspicious']
        
        # 计算性能指标
        tp = len(detected_malicious & true_malicious_set)  # 正确检测的恶意
        fp = len(detected_malicious & true_benign_set)     # 误检的良性
        tn = len(detected_benign & true_benign_set)        # 正确检测的良性
        fn = len(detected_benign & true_malicious_set)     # 漏检的恶意
        
        # 可疑客户端的处理
        suspicious_malicious = len(detected_suspicious & true_malicious_set)
        suspicious_benign = len(detected_suspicious & true_benign_set)
        
        # 计算TPR和FPR
        tpr = tp / len(true_malicious_set) if len(true_malicious_set) > 0 else 0
        fpr = fp / len(true_benign_set) if len(true_benign_set) > 0 else 0
        
        logging.info("=== Detection Performance ===")
        logging.info(f"True Positive (Malicious detected as Malicious): {tp}")
        logging.info(f"False Positive (Benign detected as Malicious): {fp}")
        logging.info(f"True Negative (Benign detected as Benign): {tn}")
        logging.info(f"False Negative (Malicious detected as Benign): {fn}")
        logging.info(f"Suspicious Malicious: {suspicious_malicious}")
        logging.info(f"Suspicious Benign: {suspicious_benign}")
        logging.info(f"TPR (True Positive Rate): {tpr:.4f}")
        logging.info(f"FPR (False Positive Rate): {fpr:.4f}")
        logging.info("=" * 30)


def agg_alignins_fedup_standard(args, 
                               inter_model_updates: torch.Tensor,
                               flat_global_model: torch.Tensor,
                               malicious_id: List[int] = None,
                               current_round: int = None) -> torch.Tensor:
    """
    AlignIns + FedUP Standard 聚合方法的入口函数
    
    Args:
        args: 参数配置
        inter_model_updates: 客户端模型更新
        flat_global_model: 扁平化全局模型
        malicious_id: 已知恶意客户端ID
        current_round: 当前轮次
        
    Returns:
        聚合后的模型更新
    """
    aggregator = AlignInsFedUPStandardAggregator(args)
    return aggregator.aggregate(
        inter_model_updates, 
        flat_global_model, 
        malicious_id, 
        current_round
    )