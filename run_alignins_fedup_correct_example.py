#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
正确的AlignIns-FedUP方法示例脚本
先用AlignIns四指标检测和过滤聚合，再对聚合后的模型进行标准FedUP剪枝
"""

import argparse
import os
import sys
import torch
import numpy as np

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from federated import Federated

def main():
    parser = argparse.ArgumentParser(description='正确的AlignIns-FedUP方法联邦学习实验')
    
    # 基础参数
    parser.add_argument('--dataset', type=str, default='cifar10', 
                       choices=['cifar10', 'cifar100', 'mnist', 'fmnist'],
                       help='数据集选择')
    parser.add_argument('--model', type=str, default='resnet18',
                       choices=['resnet18', 'resnet34', 'vgg16', 'lenet'],
                       help='模型架构')
    parser.add_argument('--num_agents', type=int, default=40,
                       help='客户端总数')
    parser.add_argument('--num_corrupt', type=int, default=10,
                       help='恶意客户端数量')
    parser.add_argument('--attack_type', type=str, default='badnet',
                       choices=['badnet', 'dba', 'neurotoxin', 'semantic'],
                       help='攻击类型')
    
    # 联邦学习参数
    parser.add_argument('--rounds', type=int, default=200,
                       help='联邦学习轮数')
    parser.add_argument('--local_epochs', type=int, default=5,
                       help='本地训练轮数')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='学习率')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批次大小')
    
    # 聚合方法参数 - 使用正确的实现
    parser.add_argument('--aggr', type=str, default='alignins_fedup_correct',
                       help='聚合方法：正确的AlignIns+FedUP实现')
    
    # AlignIns参数
    parser.add_argument('--alignins_strict_threshold', type=float, default=0.7,
                       help='AlignIns严格阈值（分位数）')
    parser.add_argument('--alignins_standard_threshold', type=float, default=0.85,
                       help='AlignIns标准阈值（分位数）')
    parser.add_argument('--suspicious_weight', type=float, default=0.3,
                       help='可疑客户端权重')
    
    # FedUP参数
    parser.add_argument('--fedup_pruning_ratio', type=float, default=0.1,
                       help='FedUP基础剪枝比例')
    parser.add_argument('--fedup_adaptive', type=bool, default=True,
                       help='是否使用自适应剪枝比例')
    
    # 实验参数
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='计算设备')
    parser.add_argument('--save_results', type=bool, default=True,
                       help='是否保存实验结果')
    parser.add_argument('--results_dir', type=str, default='./results',
                       help='结果保存目录')
    
    # 日志参数
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别')
    parser.add_argument('--log_interval', type=int, default=10,
                       help='日志输出间隔（轮数）')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 创建结果保存目录
    if args.save_results:
        os.makedirs(args.results_dir, exist_ok=True)
    
    print("=" * 80)
    print("正确的AlignIns-FedUP方法联邦学习实验")
    print("=" * 80)
    print(f"数据集: {args.dataset}")
    print(f"模型: {args.model}")
    print(f"客户端总数: {args.num_agents}")
    print(f"恶意客户端数量: {args.num_corrupt}")
    print(f"攻击类型: {args.attack_type}")
    print(f"聚合方法: {args.aggr}")
    print(f"联邦学习轮数: {args.rounds}")
    print(f"本地训练轮数: {args.local_epochs}")
    print(f"学习率: {args.lr}")
    print(f"批次大小: {args.batch_size}")
    print(f"设备: {args.device}")
    print("-" * 80)
    print("AlignIns参数:")
    print(f"  严格阈值: {args.alignins_strict_threshold}")
    print(f"  标准阈值: {args.alignins_standard_threshold}")
    print(f"  可疑客户端权重: {args.suspicious_weight}")
    print("-" * 80)
    print("FedUP参数:")
    print(f"  基础剪枝比例: {args.fedup_pruning_ratio}")
    print(f"  自适应剪枝: {args.fedup_adaptive}")
    print("=" * 80)
    
    try:
        # 创建联邦学习实例
        federated = Federated(args)
        
        # 开始训练
        print("开始联邦学习训练...")
        results = federated.train()
        
        # 保存结果
        if args.save_results:
            results_file = os.path.join(args.results_dir, 
                                      f"alignins_fedup_correct_{args.dataset}_{args.model}_{args.attack_type}.pt")
            torch.save(results, results_file)
            print(f"实验结果已保存到: {results_file}")
        
        print("=" * 80)
        print("实验完成！")
        print("=" * 80)
        
        # 输出最终结果
        if 'test_accuracy' in results:
            print(f"最终测试准确率: {results['test_accuracy'][-1]:.4f}")
        if 'attack_success_rate' in results:
            print(f"最终攻击成功率: {results['attack_success_rate'][-1]:.4f}")
        
    except Exception as e:
        print(f"实验过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)