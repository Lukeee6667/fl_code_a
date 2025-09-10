#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的AlignIns-FedUP混合防御方法测试脚本

主要改进：
1. 三层检测策略：明确良性、可疑、明确恶意
2. 个性化剪枝：基于可疑程度动态调整剪枝比例
3. 智能加权聚合：根据客户端可信度分配权重
4. 自适应阈值：平衡clean accuracy和attack success ratio
"""

import subprocess
import sys
import os

def run_experiment(config):
    """
    运行单个实验配置
    """
    cmd = [
        sys.executable, "federated.py",
        "--poison_frac", str(config["poison_frac"]),
        "--num_corrupt", str(config["num_corrupt"]),
        "--num_agents", str(config["num_agents"]),
        "--aggr", "alignins_fedup_hybrid",
        "--data", config["dataset"],
        "--attack", config["attack"],
        "--epochs", str(config["epochs"]),
        "--lr", str(config["lr"]),
        "--batch_size", str(config["batch_size"]),
        "--lambda_s", str(config["lambda_s"]),
        "--lambda_c", str(config["lambda_c"]),
        "--lambda_g", str(config["lambda_g"]),
        "--lambda_mean_cos", str(config["lambda_mean_cos"]),
        "--fedup_p_max", str(config["fedup_p_max"]),
        "--fedup_p_min", str(config["fedup_p_min"]),
        "--fedup_gamma", str(config["fedup_gamma"])
    ]
    
    if config.get("non_iid", False):
        cmd.extend(["--non_iid", "--beta", str(config["beta"])])
    
    print(f"\n{'='*80}")
    print(f"运行实验: {config['name']}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*80}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        print("实验完成!")
        if result.returncode != 0:
            print(f"错误: {result.stderr}")
        return result
    except subprocess.TimeoutExpired:
        print("实验超时!")
        return None
    except Exception as e:
        print(f"实验失败: {e}")
        return None

def main():
    """
    运行改进混合方法的对比实验
    """
    
    # 实验配置
    base_config = {
        "epochs": 200,
        "lr": 0.01,
        "batch_size": 64,
        "dataset": "cifar10",
        "num_agents": 40,
    }
    
    # 不同场景的实验配置
    experiments = [
        {
            **base_config,
            "name": "IID场景 - BadNet攻击 (轻度)",
            "poison_frac": 0.2,
            "num_corrupt": 8,
            "attack": "badnet",
            "non_iid": False,
            "beta": 1.0,
            # 较宽松的检测阈值，适合IID场景
            "lambda_s": 2.5,
            "lambda_c": 2.5,
            "lambda_g": 2.0,
            "lambda_mean_cos": 2.0,
            # 中等剪枝强度
            "fedup_p_max": 0.6,
            "fedup_p_min": 0.05,
            "fedup_gamma": 3
        },
        {
            **base_config,
            "name": "Non-IID场景 - BadNet攻击 (中度)",
            "poison_frac": 0.3,
            "num_corrupt": 12,
            "attack": "badnet",
            "non_iid": True,
            "beta": 0.5,
            # 平衡的检测阈值
            "lambda_s": 2.0,
            "lambda_c": 2.0,
            "lambda_g": 1.8,
            "lambda_mean_cos": 1.8,
            # 较强剪枝强度
            "fedup_p_max": 0.8,
            "fedup_p_min": 0.1,
            "fedup_gamma": 3
        },
        {
            **base_config,
            "name": "强攻击场景 - DBA攻击 (重度)",
            "poison_frac": 0.4,
            "num_corrupt": 16,
            "attack": "dba",
            "non_iid": True,
            "beta": 0.3,
            # 较严格的检测阈值
            "lambda_s": 1.5,
            "lambda_c": 1.5,
            "lambda_g": 1.5,
            "lambda_mean_cos": 1.5,
            # 最强剪枝强度
            "fedup_p_max": 1.0,
            "fedup_p_min": 0.15,
            "fedup_gamma": 2
        },
        {
            **base_config,
            "name": "极端场景 - BadNet攻击 (极重度)",
            "poison_frac": 0.5,
            "num_corrupt": 20,
            "attack": "badnet",
            "non_iid": True,
            "beta": 0.1,
            # 最严格的检测阈值
            "lambda_s": 1.2,
            "lambda_c": 1.2,
            "lambda_g": 1.2,
            "lambda_mean_cos": 1.2,
            # 极强剪枝强度
            "fedup_p_max": 1.2,
            "fedup_p_min": 0.2,
            "fedup_gamma": 2
        }
    ]
    
    print("开始运行改进的AlignIns-FedUP混合防御方法实验")
    print(f"总共 {len(experiments)} 个实验配置")
    
    results = []
    for i, config in enumerate(experiments, 1):
        print(f"\n进度: {i}/{len(experiments)}")
        result = run_experiment(config)
        results.append((config["name"], result))
    
    # 输出实验总结
    print(f"\n{'='*80}")
    print("实验总结")
    print(f"{'='*80}")
    
    for name, result in results:
        if result and result.returncode == 0:
            print(f"✓ {name}: 成功")
        else:
            print(f"✗ {name}: 失败")
    
    print("\n改进要点:")
    print("1. 三层检测策略 - 将客户端分为明确良性、可疑、明确恶意三类")
    print("2. 个性化剪枝 - 基于每个客户端的可疑程度动态调整剪枝比例")
    print("3. 智能加权聚合 - 根据客户端可信度分配不同权重")
    print("4. 自适应阈值 - 在不同场景下使用不同的检测阈值")
    print("\n预期效果:")
    print("- 提高Clean Accuracy: 通过保留更多良性客户端并降低其权重损失")
    print("- 降低Attack Success Ratio: 通过个性化剪枝和智能排除恶意客户端")
    print("- 更好的鲁棒性: 在不同攻击强度下都能保持良好性能")

if __name__ == "__main__":
    main()