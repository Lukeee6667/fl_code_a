#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FedUP: Efficient Pruning-based Federated Unlearning for Model Poisoning Attacks

本脚本实现了FedUP论文中提出的基于剪枝的联邦遗忘方法。
该方法通过以下步骤实现对恶意客户端的遗忘：
1. 生成遗忘掩码：识别恶意和良性客户端模型差异最大的权重
2. 应用剪枝：将识别出的权重置零
3. 性能恢复：通过额外训练轮次恢复良性知识

主要特点：
- 高效性：通过剪枝避免完全重训练
- 针对性：专门针对模型中毒攻击
- 实用性：考虑了DoS攻击防护机制
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
import copy

class FedUPUnlearning:
    """
    FedUP联邦遗忘算法实现类
    
    该类实现了论文中描述的两个核心算法：
    1. Algorithm 1: 生成遗忘掩码
    2. Algorithm 2: 应用遗忘掩码进行剪枝
    """
    
    def __init__(self, p_max: float = 0.15, p_min: float = 0.01, gamma: float = 5, rate_limit_threshold: int = 5):
        """
        初始化FedUP遗忘器
        
        Args:
            p_max: 最大剪枝率，默认0.15 (15%)
            p_min: 最小剪枝率，默认0.01 (1%)
            gamma: 控制曲线陡度的指数，默认5
            rate_limit_threshold: 速率限制阈值T，防止DoS攻击
        """
        self.p_max = p_max
        self.p_min = p_min
        self.gamma = gamma
        self.rate_limit_threshold = rate_limit_threshold
        self.unlearning_count = 0  # 记录遗忘操作次数
        
    def calculate_adaptive_pruning_ratio(self, benign_models: List[torch.nn.Module]) -> float:
        """
        根据论文公式5计算自适应剪枝比例
        
        Args:
            benign_models: 良性客户端模型列表
            
        Returns:
            float: 自适应剪枝比例
        """
        if len(benign_models) < 2:
            return self.p_min
        
        # 计算客户端模型间的余弦相似度（使用最后一层）
        similarities = []
        for i in range(len(benign_models)):
            for j in range(i + 1, len(benign_models)):
                # 获取最后一层参数
                params_i = list(benign_models[i].parameters())[-1].flatten()
                params_j = list(benign_models[j].parameters())[-1].flatten()
                
                # 计算余弦相似度
                cos_sim = torch.nn.functional.cosine_similarity(
                    params_i.unsqueeze(0), params_j.unsqueeze(0)
                ).item()
                similarities.append(cos_sim)
        
        # 计算平均相似度
        avg_similarity = sum(similarities) / len(similarities)
        
        # 归一化到[0,1]区间（假设收敛时相似度在[0.5,1]之间）
        z = max(0, min(1, (avg_similarity - 0.5) / 0.5))
        
        # 应用论文公式5
        pruning_ratio = (self.p_max - self.p_min) * (z ** self.gamma) + self.p_min
        
        return pruning_ratio
    
    def generate_unlearning_mask(self, 
                                local_models: Dict[int, torch.nn.Module],
                                global_model: torch.nn.Module,
                                malicious_client_ids: List[int],
                                benign_client_ids: List[int]) -> Dict[str, torch.Tensor]:
        """
        Algorithm 1: 生成遗忘掩码，基于论文的排名机制
        
        Args:
            local_models: 所有客户端的本地模型 {client_id: model}
            global_model: 全局模型
            malicious_client_ids: 恶意客户端ID列表
            benign_client_ids: 良性客户端ID列表
            
        Returns:
            mask: 二进制掩码字典 {layer_name: mask_tensor}
        """
        logging.info("开始生成遗忘掩码...")
        
        # Step 1: 收集恶意客户端模型
        malicious_models = [local_models[i] for i in malicious_client_ids]
        
        # Step 2: 计算恶意模型平均
        avg_malicious_model = self._average_models(malicious_models)
        
        # Step 3: 收集良性客户端模型
        benign_models = [local_models[i] for i in benign_client_ids]
        
        # Step 4: 计算良性模型平均
        avg_benign_model = self._average_models(benign_models)
        
        # Step 5: 计算自适应剪枝比例
        adaptive_pruning_ratio = self.calculate_adaptive_pruning_ratio(benign_models)
        
        # Step 6: 计算差异的平方
        difference_squared = self._compute_difference_squared(avg_malicious_model, avg_benign_model)
        
        # Step 7: 用全局模型权重缩放差异
        rank = self._scale_by_global_model(difference_squared, global_model)
        
        # Step 8-12: 为每层生成掩码
        mask = self._generate_layer_masks(rank, adaptive_pruning_ratio)
        
        logging.info(f"遗忘掩码生成完成，自适应剪枝比例: {adaptive_pruning_ratio:.4f}")
        return mask
    
    def apply_unlearning_mask(self, 
                            avg_benign_model: torch.nn.Module,
                            mask: Dict[str, torch.Tensor]) -> torch.nn.Module:
        """
        Algorithm 2: 应用遗忘掩码进行剪枝
        
        Args:
            avg_benign_model: 良性客户端的平均模型
            mask: 遗忘掩码
            
        Returns:
            unlearned_model: 经过剪枝的遗忘后模型
        """
        logging.info("开始应用遗忘掩码进行剪枝...")
        
        # Step 1: 从良性模型平均开始
        unlearned_model = copy.deepcopy(avg_benign_model)
        
        # Step 2-6: 对每层应用掩码
        pruned_params = 0
        total_params = 0
        
        for name, param in unlearned_model.named_parameters():
            if self._is_prunable_layer(name) and name in mask:
                # 应用掩码：将指定权重置零
                original_shape = param.data.shape
                param.data[mask[name] == 1] = 0
                
                # 统计剪枝参数数量
                pruned_params += torch.sum(mask[name]).item()
                total_params += param.numel()
                
                logging.info(f"层 {name}: 剪枝了 {torch.sum(mask[name]).item()} / {param.numel()} 个参数")
        
        actual_pruning_ratio = pruned_params / total_params if total_params > 0 else 0
        logging.info(f"剪枝完成，实际剪枝比例: {actual_pruning_ratio:.4f}")
        
        return unlearned_model
    
    def federated_unlearning_with_recovery(self,
                                         local_models: Dict[int, torch.nn.Module],
                                         global_model: torch.nn.Module,
                                         malicious_client_ids: List[int],
                                         benign_client_ids: List[int],
                                         recovery_rounds: int = 3) -> Tuple[torch.nn.Module, Dict]:
        """
        完整的FedUP联邦遗忘流程，包含性能恢复
        
        Args:
            local_models: 所有客户端的本地模型
            global_model: 全局模型
            malicious_client_ids: 恶意客户端ID列表
            benign_client_ids: 良性客户端ID列表
            recovery_rounds: 恢复训练轮数
            
        Returns:
            final_model: 最终的遗忘后模型
            stats: 统计信息
        """
        # 速率限制检查
        if not self._check_rate_limit():
            logging.warning(f"遗忘操作过于频繁，已达到速率限制阈值 {self.rate_limit_threshold}")
            return global_model, {"status": "rate_limited"}
        
        # 生成遗忘掩码
        mask = self.generate_unlearning_mask(
            local_models, global_model, malicious_client_ids, benign_client_ids
        )
        
        # 计算良性模型平均
        benign_models = [local_models[i] for i in benign_client_ids]
        avg_benign_model = self._average_models(benign_models)
        
        # 应用遗忘掩码
        unlearned_model = self.apply_unlearning_mask(avg_benign_model, mask)
        
        # 性能恢复（这里只是示意，实际需要与联邦学习框架集成）
        logging.info(f"建议进行 {recovery_rounds} 轮恢复训练以恢复良性知识性能")
        
        # 更新遗忘计数
        self.unlearning_count += 1
        
        stats = {
            "status": "success",
            "malicious_clients": len(malicious_client_ids),
            "benign_clients": len(benign_client_ids),
            "pruning_ratio": self.pruning_ratio,
            "unlearning_count": self.unlearning_count,
            "recovery_rounds_suggested": recovery_rounds
        }
        
        return unlearned_model, stats
    
    def _average_models(self, models: List[torch.nn.Module]) -> torch.nn.Module:
        """计算模型列表的平均值"""
        if not models:
            raise ValueError("模型列表不能为空")
        
        avg_model = copy.deepcopy(models[0])
        
        # 初始化为零
        for param in avg_model.parameters():
            param.data.zero_()
        
        # 累加所有模型参数
        for model in models:
            for avg_param, model_param in zip(avg_model.parameters(), model.parameters()):
                avg_param.data += model_param.data
        
        # 计算平均值
        for param in avg_model.parameters():
            param.data /= len(models)
        
        return avg_model
    
    def _compute_difference_squared(self, 
                                  malicious_model: torch.nn.Module,
                                  benign_model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """计算恶意和良性模型差异的平方"""
        difference_squared = {}
        
        malicious_dict = dict(malicious_model.named_parameters())
        benign_dict = dict(benign_model.named_parameters())
        
        for name in malicious_dict:
            if name in benign_dict:
                diff = malicious_dict[name].data - benign_dict[name].data
                difference_squared[name] = diff ** 2
        
        return difference_squared
    
    def _scale_by_global_model(self, 
                             difference_squared: Dict[str, torch.Tensor],
                             global_model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """用全局模型权重缩放差异"""
        rank = {}
        global_dict = dict(global_model.named_parameters())
        
        for name, diff_sq in difference_squared.items():
            if name in global_dict:
                # 使用全局模型权重的绝对值进行缩放
                rank[name] = diff_sq * torch.abs(global_dict[name].data)
        
        return rank
    
    def _generate_layer_masks(self, rank: Dict[str, torch.Tensor], pruning_ratio: float) -> Dict[str, torch.Tensor]:
        """为每层生成剪枝掩码"""
        mask = {}
        
        for layer_name, layer_rank in rank.items():
            if self._is_prunable_layer(layer_name):
                # 将权重展平并排序
                flat_rank = layer_rank.flatten()
                sorted_indices = torch.argsort(flat_rank, descending=True)
                
                # 选择前pruning_ratio%的权重进行剪枝
                num_to_prune = int(len(flat_rank) * pruning_ratio)
                prune_indices = sorted_indices[:num_to_prune]
                
                # 生成二进制掩码
                flat_mask = torch.zeros_like(flat_rank)
                flat_mask[prune_indices] = 1
                
                # 恢复原始形状
                mask[layer_name] = flat_mask.reshape(layer_rank.shape)
        
        return mask
    
    def _is_prunable_layer(self, layer_name: str) -> bool:
        """判断是否为可剪枝层（全连接层或卷积层）"""
        prunable_keywords = ['weight', 'conv', 'linear', 'fc']
        layer_name_lower = layer_name.lower()
        
        # 排除批归一化和偏置参数
        exclude_keywords = ['bias', 'bn', 'batchnorm', 'running_mean', 'running_var', 'num_batches_tracked']
        
        for exclude in exclude_keywords:
            if exclude in layer_name_lower:
                return False
        
        for keyword in prunable_keywords:
            if keyword in layer_name_lower:
                return True
        
        return False
    
    def _check_rate_limit(self) -> bool:
        """检查是否超过速率限制"""
        return self.unlearning_count < self.rate_limit_threshold
    
    def reset_rate_limit(self):
        """重置速率限制计数器（可在新的时间窗口调用）"""
        self.unlearning_count = 0
        logging.info("速率限制计数器已重置")


def compare_with_alignins(fedup_unlearning: FedUPUnlearning):
    """
    FedUP方法与AlignIns方法的对比分析
    
    Args:
        fedup_unlearning: FedUP遗忘器实例
    """
    print("\n=== FedUP vs AlignIns 方法对比 ===")
    print("\n1. 核心思想差异:")
    print("   - AlignIns: 通过方向对齐检测后门攻击，使用TDA和MPSA指标")
    print("   - FedUP: 通过剪枝实现联邦遗忘，专门针对模型中毒攻击")
    
    print("\n2. 技术方法差异:")
    print("   - AlignIns: 基于统计检测（MZ-score），过滤异常客户端")
    print("   - FedUP: 基于权重重要性分析，选择性剪枝恶意影响")
    
    print("\n3. 应用场景差异:")
    print("   - AlignIns: 实时检测和防御，适用于训练过程中的防护")
    print("   - FedUP: 事后遗忘和修复，适用于发现攻击后的模型恢复")
    
    print("\n4. 性能特点:")
    print("   - AlignIns: 低延迟，实时性好，但可能有误检")
    print("   - FedUP: 高效遗忘，避免重训练，但需要明确知道恶意客户端")
    
    print(f"\n5. FedUP配置参数:")
    print(f"   - 最大剪枝率: {fedup_unlearning.p_max}")
    print(f"   - 最小剪枝率: {fedup_unlearning.p_min}")
    print(f"   - 曲线陡度参数: {fedup_unlearning.gamma}")
    print(f"   - 速率限制阈值: {fedup_unlearning.rate_limit_threshold}")


if __name__ == "__main__":
    # 示例使用
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("FedUP: Efficient Pruning-based Federated Unlearning")
    print("="*50)
    
    # 创建FedUP遗忘器
    fedup = FedUPUnlearning(p_max=0.15, p_min=0.01, gamma=5, rate_limit_threshold=5)
    
    # 方法对比分析
    compare_with_alignins(fedup)
    
    print("\n=== 使用说明 ===")
    print("1. 实例化FedUPUnlearning类")
    print("2. 调用federated_unlearning_with_recovery方法")
    print("3. 提供本地模型、全局模型和客户端分类信息")
    print("4. 获得遗忘后的模型和统计信息")
    print("\n注意: 本实现需要与具体的联邦学习框架集成使用")