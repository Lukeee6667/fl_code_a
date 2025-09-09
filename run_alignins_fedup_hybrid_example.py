#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AlignIns-FedUP混合方法示例脚本
结合AlignIns多指标异常检测和FedUP自适应剪枝策略
"""

import argparse
import os
import sys

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    parser = argparse.ArgumentParser(description='AlignIns-FedUP混合方法联邦学习实验')
    
    # 基础参数
    parser.add_argument('--dataset', type=str, default='cifar10', 
                       choices=['cifar10', 'cifar100', 'mnist', 'fmnist'],
                       help='数据集选择')
    parser.add_argument('--model', type=str, default='resnet18',
                       choices=['resnet18', 'resnet34', 'vgg16', 'lenet'],
                       help='模型架构')
    parser.add_argument('--num_agents', type=int, default=100,
                       help='客户端总数')
    parser.add_argument('--num_corrupt', type=int, default=20,
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
    
    # 聚合方法参数
    parser.add_argument('--aggr', type=str, default='alignins_fedup_hybrid',
                       help='聚合方法')
    
    # AlignIns参数
    parser.add_argument('--sensitivity_threshold', type=float, default=0.1,
                       help='异常检测敏感度阈值')
    
    # FedUP自适应剪枝参数
    parser.add_argument('--p_max', type=float, default=0.8,
                       help='最大剪枝率')
    parser.add_argument('--p_min', type=float, default=0.1,
                       help='最小剪枝率')
    parser.add_argument('--gamma', type=float, default=2.0,
                       help='曲线陡度参数')
    
    # 混合方法特定参数
    parser.add_argument('--alignins_weight', type=float, default=0.6,
                       help='AlignIns检测权重')
    parser.add_argument('--fedup_weight', type=float, default=0.4,
                       help='FedUP检测权重')
    
    # 数据分布参数
    parser.add_argument('--iid', type=int, default=0,
                       help='是否IID分布 (0: Non-IID, 1: IID)')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Dirichlet分布参数')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cuda',
                       help='计算设备')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--save_model', type=int, default=1,
                       help='是否保存模型')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别')
    
    args = parser.parse_args()
    
    # 构建命令
    cmd_parts = [
        'python', 'src/federated.py',
        f'--dataset={args.dataset}',
        f'--model={args.model}',
        f'--num_agents={args.num_agents}',
        f'--num_corrupt={args.num_corrupt}',
        f'--attack_type={args.attack_type}',
        f'--rounds={args.rounds}',
        f'--local_epochs={args.local_epochs}',
        f'--lr={args.lr}',
        f'--batch_size={args.batch_size}',
        f'--aggr={args.aggr}',
        f'--sensitivity_threshold={args.sensitivity_threshold}',
        f'--p_max={args.p_max}',
        f'--p_min={args.p_min}',
        f'--gamma={args.gamma}',
        f'--iid={args.iid}',
        f'--alpha={args.alpha}',
        f'--device={args.device}',
        f'--seed={args.seed}',
        f'--save_model={args.save_model}'
    ]
    
    cmd = ' '.join(cmd_parts)
    
    print("="*80)
    print("AlignIns-FedUP混合方法联邦学习实验")
    print("="*80)
    print(f"数据集: {args.dataset}")
    print(f"模型: {args.model}")
    print(f"客户端总数: {args.num_agents}")
    print(f"恶意客户端数: {args.num_corrupt}")
    print(f"攻击类型: {args.attack_type}")
    print(f"聚合方法: {args.aggr}")
    print(f"敏感度阈值: {args.sensitivity_threshold}")
    print(f"自适应剪枝参数: p_max={args.p_max}, p_min={args.p_min}, gamma={args.gamma}")
    print(f"数据分布: {'IID' if args.iid else 'Non-IID'}")
    print("="*80)
    print(f"执行命令: {cmd}")
    print("="*80)
    
    # 执行命令
    os.system(cmd)

if __name__ == '__main__':
    main()