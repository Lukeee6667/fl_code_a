#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FedUP联邦遗忘方法运行示例
基于AlignIns项目集成FedUP: Efficient Pruning-based Federated Unlearning for Model Poisoning Attacks

使用方法:
1. 基础FedUP运行:
   python run_fedup_example.py --basic

2. 自定义参数运行:
   python run_fedup_example.py --custom

3. 对比实验:
   python run_fedup_example.py --compare
"""

import subprocess
import sys
import os
import argparse

def run_basic_fedup():
    """
    运行基础FedUP实验
    """
    print("=" * 60)
    print("运行基础FedUP联邦遗忘实验")
    print("=" * 60)
    
    cmd = [
        sys.executable, "src/federated.py",
        "--data", "cifar10",
        "--num_agents", "20",
        "--num_corrupt", "4",
        "--rounds", "50",
        "--local_ep", "2",
        "--bs", "64",
        "--client_lr", "0.1",
        "--server_lr", "1.0",
        "--aggr", "fedup",
        "--attack", "badnet",
        "--fedup_p_max", "0.15",
        "--fedup_p_min", "0.01",
        "--fedup_gamma", "5",
        "--fedup_sensitivity_threshold", "0.5",
        "--exp_name_extra", "fedup_basic"
    ]
    
    print("执行命令:")
    print(" ".join(cmd))
    print()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("实验完成!")
        print("输出:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"实验失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False
    
    return True

def run_custom_fedup():
    """
    运行自定义参数的FedUP实验
    """
    print("=" * 60)
    print("运行自定义FedUP联邦遗忘实验")
    print("=" * 60)
    
    # 更激进的剪枝设置
    cmd = [
        sys.executable, "src/federated.py",
        "--data", "cifar10",
        "--num_agents", "20",
        "--num_corrupt", "6",  # 更多恶意客户端
        "--rounds", "80",
        "--local_ep", "3",
        "--bs", "32",
        "--client_lr", "0.05",
        "--server_lr", "1.0",
        "--aggr", "fedup",
        "--attack", "DBA",  # 使用DBA攻击
        "--fedup_p_max", "0.25",  # 更高的剪枝比例
        "--fedup_p_min", "0.02",
        "--fedup_gamma", "8",
        "--fedup_sensitivity_threshold", "0.3",  # 更敏感的检测
        "--exp_name_extra", "fedup_aggressive"
    ]
    
    print("执行命令:")
    print(" ".join(cmd))
    print()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("实验完成!")
        print("输出:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"实验失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False
    
    return True

def run_comparison_experiments():
    """
    运行对比实验：FedUP vs AlignIns vs FedAvg
    """
    print("=" * 60)
    print("运行对比实验: FedUP vs AlignIns vs FedAvg")
    print("=" * 60)
    
    experiments = [
        {
            "name": "FedAvg (基线)",
            "aggr": "avg",
            "exp_name": "fedavg_baseline"
        },
        {
            "name": "AlignIns",
            "aggr": "alignins",
            "exp_name": "alignins_comparison"
        },
        {
            "name": "FedUP",
            "aggr": "fedup",
            "exp_name": "fedup_comparison"
        }
    ]
    
    base_cmd = [
        sys.executable, "src/federated.py",
        "--data", "cifar10",
        "--num_agents", "20",
        "--num_corrupt", "4",
        "--rounds", "30",  # 较短的轮数用于快速对比
        "--local_ep", "2",
        "--bs", "64",
        "--client_lr", "0.1",
        "--server_lr", "1.0",
        "--attack", "badnet"
    ]
    
    results = {}
    
    for exp in experiments:
        print(f"\n运行 {exp['name']} 实验...")
        
        cmd = base_cmd + [
            "--aggr", exp["aggr"],
            "--exp_name_extra", exp["exp_name"]
        ]
        
        # 为FedUP添加特定参数
        if exp["aggr"] == "fedup":
            cmd.extend([
                "--fedup_p_max", "0.15",
                "--fedup_p_min", "0.01",
                "--fedup_gamma", "5",
                "--fedup_sensitivity_threshold", "0.4"
            ])
        
        print("执行命令:")
        print(" ".join(cmd))
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            results[exp["name"]] = "成功"
            print(f"{exp['name']} 实验完成!")
        except subprocess.CalledProcessError as e:
            results[exp["name"]] = f"失败: {e}"
            print(f"{exp['name']} 实验失败: {e}")
    
    print("\n" + "=" * 60)
    print("对比实验结果总结:")
    print("=" * 60)
    for name, result in results.items():
        print(f"{name}: {result}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="FedUP联邦遗忘方法运行示例")
    parser.add_argument("--basic", action="store_true", help="运行基础FedUP实验")
    parser.add_argument("--custom", action="store_true", help="运行自定义参数FedUP实验")
    parser.add_argument("--compare", action="store_true", help="运行对比实验")
    
    args = parser.parse_args()
    
    # 检查是否在正确的目录
    if not os.path.exists("src/federated.py"):
        print("错误: 请在AlignIns项目根目录下运行此脚本")
        print("当前目录:", os.getcwd())
        return
    
    if args.basic:
        run_basic_fedup()
    elif args.custom:
        run_custom_fedup()
    elif args.compare:
        run_comparison_experiments()
    else:
        print("请选择运行模式:")
        print("  --basic   : 运行基础FedUP实验")
        print("  --custom  : 运行自定义参数FedUP实验")
        print("  --compare : 运行对比实验")
        print("\n示例:")
        print("  python run_fedup_example.py --basic")
        print("  python run_fedup_example.py --custom")
        print("  python run_fedup_example.py --compare")

if __name__ == "__main__":
    main()