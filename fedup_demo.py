#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FedUP方法演示脚本

本脚本展示了如何使用FedUP联邦遗忘方法，并与AlignIns方法进行对比分析。
包含完整的使用示例和方法特点说明。
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, List
import sys
import os

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.fedup_unlearning import FedUPUnlearning

class SimpleModel(nn.Module):
    """简单的演示模型"""
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def create_demo_models(num_clients=10, num_malicious=2):
    """
    创建演示用的客户端模型
    
    Args:
        num_clients: 总客户端数量
        num_malicious: 恶意客户端数量
    
    Returns:
        local_models: 本地模型字典
        global_model: 全局模型
        malicious_ids: 恶意客户端ID列表
        benign_ids: 良性客户端ID列表
    """
    print(f"\n创建演示模型: {num_clients}个客户端 ({num_malicious}个恶意)")
    
    # 创建全局模型
    global_model = SimpleModel()
    
    # 创建本地模型
    local_models = {}
    
    for i in range(num_clients):
        model = SimpleModel()
        
        # 为恶意客户端添加一些"恶意"扰动
        if i < num_malicious:
            for param in model.parameters():
                # 添加较大的随机扰动模拟恶意行为
                param.data += torch.randn_like(param.data) * 0.5
        else:
            # 良性客户端只添加小的随机扰动
            for param in model.parameters():
                param.data += torch.randn_like(param.data) * 0.1
        
        local_models[i] = model
    
    malicious_ids = list(range(num_malicious))
    benign_ids = list(range(num_malicious, num_clients))
    
    print(f"恶意客户端ID: {malicious_ids}")
    print(f"良性客户端ID: {benign_ids}")
    
    return local_models, global_model, malicious_ids, benign_ids

def analyze_model_differences(local_models, malicious_ids, benign_ids):
    """
    分析恶意和良性客户端模型的差异
    """
    print("\n=== 模型差异分析 ===")
    
    # 计算恶意模型平均
    malicious_models = [local_models[i] for i in malicious_ids]
    malicious_params = []
    for model in malicious_models:
        params = torch.cat([p.data.flatten() for p in model.parameters()])
        malicious_params.append(params)
    avg_malicious_params = torch.stack(malicious_params).mean(dim=0)
    
    # 计算良性模型平均
    benign_models = [local_models[i] for i in benign_ids]
    benign_params = []
    for model in benign_models:
        params = torch.cat([p.data.flatten() for p in model.parameters()])
        benign_params.append(params)
    avg_benign_params = torch.stack(benign_params).mean(dim=0)
    
    # 计算差异统计
    diff = avg_malicious_params - avg_benign_params
    diff_norm = torch.norm(diff).item()
    diff_mean = torch.mean(torch.abs(diff)).item()
    diff_std = torch.std(diff).item()
    
    print(f"恶意vs良性模型差异:")
    print(f"  - L2范数: {diff_norm:.6f}")
    print(f"  - 平均绝对差异: {diff_mean:.6f}")
    print(f"  - 标准差: {diff_std:.6f}")
    
    return diff_norm, diff_mean, diff_std

def demonstrate_fedup_process():
    """
    演示完整的FedUP联邦遗忘过程
    """
    print("\n" + "="*60)
    print("FedUP联邦遗忘方法演示")
    print("="*60)
    
    # 1. 创建演示模型
    local_models, global_model, malicious_ids, benign_ids = create_demo_models(
        num_clients=10, num_malicious=3
    )
    
    # 2. 分析模型差异
    analyze_model_differences(local_models, malicious_ids, benign_ids)
    
    # 3. 创建FedUP遗忘器
    print("\n=== 初始化FedUP遗忘器 ===")
    fedup = FedUPUnlearning(pruning_ratio=0.15, rate_limit_threshold=5)
    print(f"剪枝比例: {fedup.pruning_ratio}")
    print(f"速率限制阈值: {fedup.rate_limit_threshold}")
    
    # 4. 执行联邦遗忘
    print("\n=== 执行联邦遗忘过程 ===")
    unlearned_model, stats = fedup.federated_unlearning_with_recovery(
        local_models=local_models,
        global_model=global_model,
        malicious_client_ids=malicious_ids,
        benign_client_ids=benign_ids,
        recovery_rounds=3
    )
    
    # 5. 显示结果统计
    print("\n=== 遗忘结果统计 ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # 6. 分析遗忘效果
    print("\n=== 遗忘效果分析 ===")
    
    # 计算原始良性模型平均
    benign_models = [local_models[i] for i in benign_ids]
    original_benign_avg = fedup._average_models(benign_models)
    
    # 比较遗忘前后的模型差异
    original_params = torch.cat([p.data.flatten() for p in original_benign_avg.parameters()])
    unlearned_params = torch.cat([p.data.flatten() for p in unlearned_model.parameters()])
    
    model_change_norm = torch.norm(original_params - unlearned_params).item()
    print(f"模型变化L2范数: {model_change_norm:.6f}")
    
    # 统计被剪枝的参数数量
    total_params = 0
    zero_params = 0
    for param in unlearned_model.parameters():
        total_params += param.numel()
        zero_params += (param.data == 0).sum().item()
    
    actual_pruning_ratio = zero_params / total_params
    print(f"实际剪枝比例: {actual_pruning_ratio:.4f}")
    print(f"剪枝参数数量: {zero_params} / {total_params}")
    
    return unlearned_model, stats

def compare_methods_detailed():
    """
    详细对比FedUP和AlignIns方法
    """
    print("\n" + "="*60)
    print("FedUP vs AlignIns 详细方法对比")
    print("="*60)
    
    comparison_data = {
        "维度": [
            "核心目标",
            "技术原理", 
            "检测机制",
            "处理方式",
            "时间复杂度",
            "空间复杂度",
            "适用场景",
            "优势",
            "局限性",
            "实时性",
            "准确性要求"
        ],
        "AlignIns方法": [
            "实时检测后门攻击",
            "方向对齐检测 + 统计分析",
            "TDA(方向对齐) + MPSA(符号一致性)",
            "过滤异常客户端更新",
            "O(n*d) - n个客户端，d维参数",
            "O(d) - 存储全局模型和统计量",
            "训练过程中的实时防护",
            "低延迟、实时性好、无需先验知识",
            "可能误检、依赖阈值设置",
            "高 - 每轮都能检测",
            "中等 - 基于统计假设"
        ],
        "FedUP方法": [
            "事后遗忘恶意影响",
            "权重重要性分析 + 选择性剪枝",
            "恶意vs良性模型差异分析",
            "剪枝恶意相关权重",
            "O(n*d*log(d)) - 需要排序操作",
            "O(d) - 存储掩码和模型副本",
            "发现攻击后的模型修复",
            "高效遗忘、避免重训练、针对性强",
            "需要明确恶意客户端、后验处理",
            "低 - 需要完整攻击信息",
            "高 - 基于明确的恶意标识"
        ]
    }
    
    # 打印对比表格
    print(f"{'维度':<15} {'AlignIns方法':<35} {'FedUP方法':<35}")
    print("-" * 85)
    
    for i, dimension in enumerate(comparison_data["维度"]):
        alignins_desc = comparison_data["AlignIns方法"][i]
        fedup_desc = comparison_data["FedUP方法"][i]
        
        print(f"{dimension:<15} {alignins_desc:<35} {fedup_desc:<35}")
    
    print("\n=== 方法互补性分析 ===")
    print("1. 时间维度互补:")
    print("   - AlignIns: 训练时实时防护")
    print("   - FedUP: 训练后遗忘修复")
    
    print("\n2. 技术维度互补:")
    print("   - AlignIns: 基于统计检测，适合未知攻击")
    print("   - FedUP: 基于明确标识，适合已知攻击")
    
    print("\n3. 应用场景互补:")
    print("   - 可以先用AlignIns检测，再用FedUP遗忘")
    print("   - 形成完整的攻击防护-检测-修复链条")

def demonstrate_integration_potential():
    """
    演示两种方法的集成潜力
    """
    print("\n" + "="*60)
    print("AlignIns + FedUP 集成方案演示")
    print("="*60)
    
    print("\n=== 集成工作流程 ===")
    print("1. 训练阶段: 使用AlignIns实时检测")
    print("   - 每轮聚合时计算TDA和MPSA")
    print("   - 基于MZ-score过滤异常客户端")
    print("   - 记录可疑客户端历史")
    
    print("\n2. 检测阶段: 分析历史数据")
    print("   - 统计客户端被过滤频率")
    print("   - 识别持续异常的客户端")
    print("   - 确定恶意客户端列表")
    
    print("\n3. 遗忘阶段: 使用FedUP修复模型")
    print("   - 基于确定的恶意客户端列表")
    print("   - 生成针对性的遗忘掩码")
    print("   - 执行选择性剪枝")
    print("   - 进行性能恢复训练")
    
    print("\n=== 集成优势 ===")
    print("✓ 全生命周期防护: 训练中防护 + 训练后修复")
    print("✓ 互补检测能力: 统计检测 + 明确标识")
    print("✓ 渐进式处理: 实时过滤 → 历史分析 → 精确遗忘")
    print("✓ 鲁棒性增强: 多层防护机制")
    
    # 模拟集成使用场景
    print("\n=== 模拟集成场景 ===")
    
    # 假设AlignIns检测到的可疑客户端历史
    suspicious_history = {
        0: [True, True, False, True, True],   # 客户端0: 5轮中4轮被标记
        1: [True, False, True, True, False],  # 客户端1: 5轮中3轮被标记
        2: [False, False, True, False, False], # 客户端2: 5轮中1轮被标记
        3: [False, False, False, False, False] # 客户端3: 从未被标记
    }
    
    print("AlignIns检测历史 (True=被标记为可疑):")
    for client_id, history in suspicious_history.items():
        suspicious_rate = sum(history) / len(history)
        print(f"  客户端 {client_id}: {history} (可疑率: {suspicious_rate:.2f})")
    
    # 基于历史确定恶意客户端
    threshold = 0.6  # 可疑率阈值
    confirmed_malicious = []
    for client_id, history in suspicious_history.items():
        if sum(history) / len(history) >= threshold:
            confirmed_malicious.append(client_id)
    
    print(f"\n基于阈值 {threshold} 确定的恶意客户端: {confirmed_malicious}")
    print("→ 这些客户端将被FedUP方法处理")

def main():
    """
    主函数：运行完整的演示
    """
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('fedup_demo.log')
        ]
    )
    
    print("FedUP: Efficient Pruning-based Federated Unlearning")
    print("基于剪枝的高效联邦遗忘方法演示")
    print("="*60)
    
    try:
        # 1. 演示FedUP过程
        unlearned_model, stats = demonstrate_fedup_process()
        
        # 2. 详细方法对比
        compare_methods_detailed()
        
        # 3. 集成方案演示
        demonstrate_integration_potential()
        
        print("\n" + "="*60)
        print("演示完成！")
        print("\n关键要点总结:")
        print("1. FedUP通过选择性剪枝实现高效的联邦遗忘")
        print("2. 与AlignIns方法在时间和技术维度上形成互补")
        print("3. 两种方法可以集成形成完整的防护体系")
        print("4. 实际应用需要根据具体场景调整参数")
        
    except Exception as e:
        logging.error(f"演示过程中出现错误: {e}")
        raise

if __name__ == "__main__":
    main()