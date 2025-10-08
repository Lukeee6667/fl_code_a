"""
测试脚本：AlignIns + FedUP Standard 聚合方法
验证新实现的标准方法是否正常工作

使用方法：
python test_alignins_fedup_standard.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import numpy as np
import argparse
import logging
from agg_alignins_fedup_standard import agg_alignins_fedup_standard

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_test_args():
    """创建测试参数"""
    args = argparse.Namespace()
    
    # AlignIns参数
    args.lambda_s = 1.5      # MPSA阈值
    args.lambda_c = 1.5      # TDA阈值
    args.lambda_g = 1.5      # Grad Norm阈值
    args.lambda_mean_cos = 1.5  # Mean Cos阈值
    args.sparsity = 0.1      # MPSA稀疏度
    
    # FedUP参数
    args.fedup_alpha = 0.1   # 基础剪枝比例
    args.fedup_beta = 0.5    # 自适应调整因子
    args.fedup_gamma = 0.8   # 良性相似度阈值
    
    # 聚合参数
    args.norm_clip = 10.0    # 范数裁剪
    args.device = 'cpu'      # 设备
    
    return args

def generate_test_data(num_clients=10, model_dim=1000, num_malicious=2):
    """
    生成测试数据
    
    Args:
        num_clients: 客户端数量
        model_dim: 模型维度
        num_malicious: 恶意客户端数量
        
    Returns:
        inter_model_updates: 客户端更新
        flat_global_model: 全局模型
        malicious_id: 恶意客户端ID
    """
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 生成全局模型
    flat_global_model = torch.randn(model_dim)
    
    # 生成客户端更新
    inter_model_updates = []
    
    # 良性客户端：更新相对较小且相似
    for i in range(num_clients - num_malicious):
        # 良性更新：小幅度的正常更新
        update = torch.randn(model_dim) * 0.1 + torch.randn(model_dim) * 0.05
        inter_model_updates.append(update)
    
    # 恶意客户端：更新较大且异常
    malicious_id = list(range(num_clients - num_malicious, num_clients))
    for i in range(num_malicious):
        # 恶意更新：大幅度的异常更新
        if i == 0:
            # 第一个恶意客户端：梯度范数异常大
            update = torch.randn(model_dim) * 2.0
        else:
            # 第二个恶意客户端：方向异常
            update = -torch.randn(model_dim) * 1.5
        inter_model_updates.append(update)
    
    inter_model_updates = torch.stack(inter_model_updates)
    
    logging.info(f"Generated test data:")
    logging.info(f"  - Total clients: {num_clients}")
    logging.info(f"  - Benign clients: {num_clients - num_malicious}")
    logging.info(f"  - Malicious clients: {num_malicious} (IDs: {malicious_id})")
    logging.info(f"  - Model dimension: {model_dim}")
    
    return inter_model_updates, flat_global_model, malicious_id

def test_basic_functionality():
    """测试基本功能"""
    print("=" * 60)
    print("测试1: 基本功能测试")
    print("=" * 60)
    
    # 创建测试参数和数据
    args = create_test_args()
    inter_model_updates, flat_global_model, malicious_id = generate_test_data()
    
    try:
        # 调用聚合方法
        aggregated_update = agg_alignins_fedup_standard(
            args,
            inter_model_updates,
            flat_global_model,
            malicious_id,
            current_round=1
        )
        
        # 验证输出
        assert aggregated_update is not None, "聚合结果不能为None"
        assert aggregated_update.shape == flat_global_model.shape, "聚合结果维度不匹配"
        assert torch.isfinite(aggregated_update).all(), "聚合结果包含无效值"
        
        print("✓ 基本功能测试通过")
        print(f"  - 聚合结果维度: {aggregated_update.shape}")
        print(f"  - 聚合结果范数: {torch.norm(aggregated_update).item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ 基本功能测试失败: {e}")
        return False

def test_different_scenarios():
    """测试不同场景"""
    print("\n" + "=" * 60)
    print("测试2: 不同场景测试")
    print("=" * 60)
    
    args = create_test_args()
    scenarios = [
        {"num_clients": 5, "num_malicious": 1, "name": "小规模场景"},
        {"num_clients": 20, "num_malicious": 4, "name": "大规模场景"},
        {"num_clients": 8, "num_malicious": 0, "name": "无恶意客户端"},
        {"num_clients": 6, "num_malicious": 3, "name": "高恶意比例"},
    ]
    
    success_count = 0
    
    for scenario in scenarios:
        try:
            print(f"\n测试场景: {scenario['name']}")
            
            inter_model_updates, flat_global_model, malicious_id = generate_test_data(
                num_clients=scenario['num_clients'],
                num_malicious=scenario['num_malicious']
            )
            
            aggregated_update = agg_alignins_fedup_standard(
                args,
                inter_model_updates,
                flat_global_model,
                malicious_id if scenario['num_malicious'] > 0 else None,
                current_round=10
            )
            
            # 验证结果
            assert aggregated_update is not None
            assert torch.isfinite(aggregated_update).all()
            
            print(f"  ✓ {scenario['name']} 测试通过")
            print(f"    - 聚合结果范数: {torch.norm(aggregated_update).item():.4f}")
            success_count += 1
            
        except Exception as e:
            print(f"  ✗ {scenario['name']} 测试失败: {e}")
    
    print(f"\n场景测试结果: {success_count}/{len(scenarios)} 通过")
    return success_count == len(scenarios)

def test_parameter_sensitivity():
    """测试参数敏感性"""
    print("\n" + "=" * 60)
    print("测试3: 参数敏感性测试")
    print("=" * 60)
    
    base_args = create_test_args()
    inter_model_updates, flat_global_model, malicious_id = generate_test_data()
    
    # 测试不同参数设置
    param_tests = [
        {"name": "严格阈值", "lambda_s": 1.0, "lambda_c": 1.0},
        {"name": "宽松阈值", "lambda_s": 2.0, "lambda_c": 2.0},
        {"name": "高剪枝比例", "fedup_alpha": 0.3, "fedup_beta": 0.7},
        {"name": "低剪枝比例", "fedup_alpha": 0.05, "fedup_beta": 0.3},
    ]
    
    success_count = 0
    results = []
    
    for test in param_tests:
        try:
            # 创建测试参数
            test_args = create_test_args()
            for key, value in test.items():
                if key != "name":
                    setattr(test_args, key, value)
            
            # 运行测试
            aggregated_update = agg_alignins_fedup_standard(
                test_args,
                inter_model_updates,
                flat_global_model,
                malicious_id,
                current_round=5
            )
            
            norm = torch.norm(aggregated_update).item()
            results.append({"name": test["name"], "norm": norm})
            
            print(f"  ✓ {test['name']}: 范数 = {norm:.4f}")
            success_count += 1
            
        except Exception as e:
            print(f"  ✗ {test['name']}: 失败 - {e}")
    
    print(f"\n参数敏感性测试结果: {success_count}/{len(param_tests)} 通过")
    
    # 显示结果对比
    if len(results) > 1:
        print("\n结果对比:")
        for result in results:
            print(f"  - {result['name']}: {result['norm']:.4f}")
    
    return success_count == len(param_tests)

def test_edge_cases():
    """测试边界情况"""
    print("\n" + "=" * 60)
    print("测试4: 边界情况测试")
    print("=" * 60)
    
    args = create_test_args()
    success_count = 0
    total_tests = 0
    
    # 测试1: 单个客户端
    try:
        total_tests += 1
        print("测试: 单个客户端")
        
        inter_model_updates = torch.randn(1, 1000)
        flat_global_model = torch.randn(1000)
        
        result = agg_alignins_fedup_standard(
            args, inter_model_updates, flat_global_model, None, 1
        )
        
        assert result is not None
        print("  ✓ 单个客户端测试通过")
        success_count += 1
        
    except Exception as e:
        print(f"  ✗ 单个客户端测试失败: {e}")
    
    # 测试2: 所有客户端都是恶意的
    try:
        total_tests += 1
        print("测试: 所有客户端都是恶意的")
        
        inter_model_updates, flat_global_model, _ = generate_test_data(
            num_clients=5, num_malicious=5
        )
        malicious_id = list(range(5))
        
        result = agg_alignins_fedup_standard(
            args, inter_model_updates, flat_global_model, malicious_id, 1
        )
        
        assert result is not None
        print("  ✓ 全恶意客户端测试通过")
        success_count += 1
        
    except Exception as e:
        print(f"  ✗ 全恶意客户端测试失败: {e}")
    
    # 测试3: 极小模型维度
    try:
        total_tests += 1
        print("测试: 极小模型维度")
        
        inter_model_updates, flat_global_model, malicious_id = generate_test_data(
            num_clients=5, model_dim=10, num_malicious=1
        )
        
        result = agg_alignins_fedup_standard(
            args, inter_model_updates, flat_global_model, malicious_id, 1
        )
        
        assert result is not None
        print("  ✓ 极小模型维度测试通过")
        success_count += 1
        
    except Exception as e:
        print(f"  ✗ 极小模型维度测试失败: {e}")
    
    print(f"\n边界情况测试结果: {success_count}/{total_tests} 通过")
    return success_count == total_tests

def main():
    """主测试函数"""
    print("开始测试 AlignIns + FedUP Standard 聚合方法")
    print("=" * 80)
    
    # 运行所有测试
    test_results = []
    
    test_results.append(("基本功能", test_basic_functionality()))
    test_results.append(("不同场景", test_different_scenarios()))
    test_results.append(("参数敏感性", test_parameter_sensitivity()))
    test_results.append(("边界情况", test_edge_cases()))
    
    # 汇总结果
    print("\n" + "=" * 80)
    print("测试结果汇总")
    print("=" * 80)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name:15} : {status}")
        if result:
            passed += 1
    
    print("-" * 40)
    print(f"总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试都通过了！新的AlignIns+FedUP标准方法实现正确。")
        return True
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败，需要检查实现。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)