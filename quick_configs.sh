#!/bin/bash

# =============================================================================
# 快速配置脚本 - 提供多种预设实验配置
# 使用方法: source quick_configs.sh 然后选择配置函数
# =============================================================================

# 快速测试配置 - 用于验证代码是否正常工作
config_quick_test() {
    echo "=== 快速测试配置 ==="
    python run_alignins_fedup_correct_example.py \
        --dataset mnist \
        --model lenet \
        --num_agents 20 \
        --num_corrupt 4 \
        --attack_type badnet \
        --rounds 20 \
        --local_epochs 2 \
        --lr 0.01 \
        --batch_size 32 \
        --alignins_strict_threshold 0.7 \
        --alignins_standard_threshold 0.85 \
        --suspicious_weight 0.3 \
        --fedup_pruning_ratio 0.1 \
        --seed 42
}

# 标准CIFAR-10实验配置
config_cifar10_standard() {
    echo "=== CIFAR-10标准配置 ==="
    python run_alignins_fedup_correct_example.py \
        --dataset cifar10 \
        --model resnet18 \
        --num_agents 50 \
        --num_corrupt 10 \
        --attack_type badnet \
        --rounds 100 \
        --local_epochs 5 \
        --lr 0.01 \
        --batch_size 64 \
        --alignins_strict_threshold 0.7 \
        --alignins_standard_threshold 0.85 \
        --suspicious_weight 0.3 \
        --fedup_pruning_ratio 0.1 \
        --seed 42
}

# 大规模实验配置
config_large_scale() {
    echo "=== 大规模实验配置 ==="
    python run_alignins_fedup_correct_example.py \
        --dataset cifar10 \
        --model resnet18 \
        --num_agents 100 \
        --num_corrupt 20 \
        --attack_type badnet \
        --rounds 200 \
        --local_epochs 5 \
        --lr 0.01 \
        --batch_size 64 \
        --alignins_strict_threshold 0.7 \
        --alignins_standard_threshold 0.85 \
        --suspicious_weight 0.3 \
        --fedup_pruning_ratio 0.1 \
        --seed 42
}

# BadNet攻击实验
config_badnet_attack() {
    echo "=== BadNet攻击实验 ==="
    python run_alignins_fedup_correct_example.py \
        --dataset cifar10 \
        --model resnet18 \
        --num_agents 100 \
        --num_corrupt 20 \
        --attack_type badnet \
        --rounds 200 \
        --local_epochs 5 \
        --lr 0.01 \
        --batch_size 64 \
        --alignins_strict_threshold 0.7 \
        --alignins_standard_threshold 0.85 \
        --suspicious_weight 0.3 \
        --fedup_pruning_ratio 0.1 \
        --seed 42
}

# DBA攻击实验
config_dba_attack() {
    echo "=== DBA攻击实验 ==="
    python run_alignins_fedup_correct_example.py \
        --dataset cifar10 \
        --model resnet18 \
        --num_agents 100 \
        --num_corrupt 20 \
        --attack_type dba \
        --rounds 200 \
        --local_epochs 5 \
        --lr 0.01 \
        --batch_size 64 \
        --alignins_strict_threshold 0.6 \
        --alignins_standard_threshold 0.8 \
        --suspicious_weight 0.2 \
        --fedup_pruning_ratio 0.15 \
        --seed 42
}

# Neurotoxin攻击实验
config_neurotoxin_attack() {
    echo "=== Neurotoxin攻击实验 ==="
    python run_alignins_fedup_correct_example.py \
        --dataset cifar10 \
        --model resnet18 \
        --num_agents 100 \
        --num_corrupt 20 \
        --attack_type neurotoxin \
        --rounds 200 \
        --local_epochs 5 \
        --lr 0.01 \
        --batch_size 64 \
        --alignins_strict_threshold 0.6 \
        --alignins_standard_threshold 0.8 \
        --suspicious_weight 0.2 \
        --fedup_pruning_ratio 0.2 \
        --seed 42
}

# 高剪枝比例实验
config_high_pruning() {
    echo "=== 高剪枝比例实验 ==="
    python run_alignins_fedup_correct_example.py \
        --dataset cifar10 \
        --model resnet18 \
        --num_agents 100 \
        --num_corrupt 20 \
        --attack_type badnet \
        --rounds 200 \
        --local_epochs 5 \
        --lr 0.01 \
        --batch_size 64 \
        --alignins_strict_threshold 0.7 \
        --alignins_standard_threshold 0.85 \
        --suspicious_weight 0.3 \
        --fedup_pruning_ratio 0.3 \
        --seed 42
}

# 严格检测实验
config_strict_detection() {
    echo "=== 严格检测实验 ==="
    python run_alignins_fedup_correct_example.py \
        --dataset cifar10 \
        --model resnet18 \
        --num_agents 100 \
        --num_corrupt 20 \
        --attack_type badnet \
        --rounds 200 \
        --local_epochs 5 \
        --lr 0.01 \
        --batch_size 64 \
        --alignins_strict_threshold 0.5 \
        --alignins_standard_threshold 0.7 \
        --suspicious_weight 0.1 \
        --fedup_pruning_ratio 0.1 \
        --seed 42
}

# CIFAR-100实验
config_cifar100() {
    echo "=== CIFAR-100实验 ==="
    python run_alignins_fedup_correct_example.py \
        --dataset cifar100 \
        --model resnet18 \
        --num_agents 100 \
        --num_corrupt 20 \
        --attack_type badnet \
        --rounds 300 \
        --local_epochs 5 \
        --lr 0.01 \
        --batch_size 64 \
        --alignins_strict_threshold 0.7 \
        --alignins_standard_threshold 0.85 \
        --suspicious_weight 0.3 \
        --fedup_pruning_ratio 0.1 \
        --seed 42
}

# VGG模型实验
config_vgg_model() {
    echo "=== VGG模型实验 ==="
    python run_alignins_fedup_correct_example.py \
        --dataset cifar10 \
        --model vgg16 \
        --num_agents 100 \
        --num_corrupt 20 \
        --attack_type badnet \
        --rounds 200 \
        --local_epochs 5 \
        --lr 0.001 \
        --batch_size 32 \
        --alignins_strict_threshold 0.7 \
        --alignins_standard_threshold 0.85 \
        --suspicious_weight 0.3 \
        --fedup_pruning_ratio 0.1 \
        --seed 42
}

# 显示所有可用配置
show_configs() {
    echo "=========================================="
    echo "可用的预设配置:"
    echo "=========================================="
    echo "1. config_quick_test        - 快速测试配置 (MNIST, 20轮)"
    echo "2. config_cifar10_standard  - CIFAR-10标准配置 (100轮)"
    echo "3. config_large_scale       - 大规模实验配置 (200轮)"
    echo "4. config_badnet_attack     - BadNet攻击实验"
    echo "5. config_dba_attack        - DBA攻击实验"
    echo "6. config_neurotoxin_attack - Neurotoxin攻击实验"
    echo "7. config_high_pruning      - 高剪枝比例实验"
    echo "8. config_strict_detection  - 严格检测实验"
    echo "9. config_cifar100          - CIFAR-100实验"
    echo "10. config_vgg_model        - VGG模型实验"
    echo "=========================================="
    echo "使用方法:"
    echo "  source quick_configs.sh"
    echo "  config_quick_test"
    echo "或者:"
    echo "  bash -c 'source quick_configs.sh && config_quick_test'"
    echo "=========================================="
}

# 交互式配置选择
interactive_config() {
    show_configs
    echo
    read -p "请选择配置 (1-10): " choice
    
    case $choice in
        1) config_quick_test ;;
        2) config_cifar10_standard ;;
        3) config_large_scale ;;
        4) config_badnet_attack ;;
        5) config_dba_attack ;;
        6) config_neurotoxin_attack ;;
        7) config_high_pruning ;;
        8) config_strict_detection ;;
        9) config_cifar100 ;;
        10) config_vgg_model ;;
        *) echo "无效选择，请输入1-10之间的数字" ;;
    esac
}

# 如果直接运行脚本，显示配置选项
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    interactive_config
fi