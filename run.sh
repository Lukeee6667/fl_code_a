#!/bin/bash

# =============================================================================
# 正确的AlignIns+FedUP方法运行脚本
# 使用方法: bash run.sh
# =============================================================================

echo "=========================================="
echo "正确的AlignIns+FedUP方法联邦学习实验"
echo "=========================================="

# =============================================================================
# 实验配置参数 - 可以根据需要修改这些参数
# =============================================================================

# 基础实验参数
DATASET="cifar10"           # 数据集: cifar10, cifar100, mnist, fmnist
MODEL="resnet18"            # 模型: resnet18, resnet34, vgg16, lenet
NUM_AGENTS=100              # 客户端总数
NUM_CORRUPT=20              # 恶意客户端数量
ATTACK_TYPE="badnet"        # 攻击类型: badnet, dba, neurotoxin, semantic

# 联邦学习参数
ROUNDS=200                  # 联邦学习轮数
LOCAL_EPOCHS=5              # 本地训练轮数
LEARNING_RATE=0.01          # 学习率
BATCH_SIZE=64               # 批次大小

# AlignIns检测参数
STRICT_THRESHOLD=0.7        # 严格阈值(分位数): 低于此值为良性客户端
STANDARD_THRESHOLD=0.85     # 标准阈值(分位数): 高于此值为恶意客户端
SUSPICIOUS_WEIGHT=0.3       # 可疑客户端权重: 0-1之间

# FedUP剪枝参数
PRUNING_RATIO=0.1           # 基础剪枝比例: 0-1之间
ADAPTIVE_PRUNING=true       # 是否启用自适应剪枝

# 系统参数
SEED=42                     # 随机种子
DEVICE="auto"               # 计算设备: auto, cpu, cuda
SAVE_RESULTS=true           # 是否保存结果
RESULTS_DIR="./results"     # 结果保存目录
LOG_LEVEL="INFO"            # 日志级别: DEBUG, INFO, WARNING, ERROR
LOG_INTERVAL=10             # 日志输出间隔(轮数)

# =============================================================================
# 预设实验配置 - 取消注释来使用预设配置
# =============================================================================

# 快速测试配置 (小规模, 快速验证)
# DATASET="mnist"
# MODEL="lenet"
# NUM_AGENTS=20
# NUM_CORRUPT=4
# ROUNDS=50
# LOCAL_EPOCHS=3

# 标准实验配置 (中等规模)
# DATASET="cifar10"
# MODEL="resnet18"
# NUM_AGENTS=50
# NUM_CORRUPT=10
# ROUNDS=100
# LOCAL_EPOCHS=5

# 大规模实验配置 (完整规模)
# DATASET="cifar10"
# MODEL="resnet18"
# NUM_AGENTS=100
# NUM_CORRUPT=20
# ROUNDS=200
# LOCAL_EPOCHS=5

# =============================================================================
# 不同攻击类型的实验配置
# =============================================================================

# BadNet攻击实验
# ATTACK_TYPE="badnet"
# PRUNING_RATIO=0.1

# DBA攻击实验
# ATTACK_TYPE="dba"
# PRUNING_RATIO=0.15

# Neurotoxin攻击实验
# ATTACK_TYPE="neurotoxin"
# PRUNING_RATIO=0.2

# =============================================================================
# 显示当前配置
# =============================================================================

echo "当前实验配置:"
echo "------------------------------------------"
echo "数据集: $DATASET"
echo "模型: $MODEL"
echo "客户端总数: $NUM_AGENTS"
echo "恶意客户端数量: $NUM_CORRUPT"
echo "攻击类型: $ATTACK_TYPE"
echo "联邦学习轮数: $ROUNDS"
echo "本地训练轮数: $LOCAL_EPOCHS"
echo "学习率: $LEARNING_RATE"
echo "批次大小: $BATCH_SIZE"
echo "------------------------------------------"
echo "AlignIns参数:"
echo "  严格阈值: $STRICT_THRESHOLD"
echo "  标准阈值: $STANDARD_THRESHOLD"
echo "  可疑客户端权重: $SUSPICIOUS_WEIGHT"
echo "------------------------------------------"
echo "FedUP参数:"
echo "  基础剪枝比例: $PRUNING_RATIO"
echo "  自适应剪枝: $ADAPTIVE_PRUNING"
echo "------------------------------------------"
echo "系统参数:"
echo "  随机种子: $SEED"
echo "  计算设备: $DEVICE"
echo "  保存结果: $SAVE_RESULTS"
echo "  结果目录: $RESULTS_DIR"
echo "=========================================="

# 确认是否继续
read -p "是否使用以上配置开始实验? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "实验已取消"
    exit 1
fi

# =============================================================================
# 检查环境和依赖
# =============================================================================

echo "检查Python环境..."
if ! command -v python &> /dev/null; then
    echo "错误: 未找到Python"
    exit 1
fi

echo "检查必要文件..."
if [ ! -f "run_alignins_fedup_correct_example.py" ]; then
    echo "错误: 未找到运行脚本 run_alignins_fedup_correct_example.py"
    exit 1
fi

if [ ! -d "src" ]; then
    echo "错误: 未找到src目录"
    exit 1
fi

# 创建结果目录
if [ "$SAVE_RESULTS" = true ]; then
    mkdir -p "$RESULTS_DIR"
    echo "结果将保存到: $RESULTS_DIR"
fi

# =============================================================================
# 构建运行命令
# =============================================================================

echo "构建运行命令..."

PYTHON_CMD="python run_alignins_fedup_correct_example.py"
PYTHON_CMD="$PYTHON_CMD --dataset $DATASET"
PYTHON_CMD="$PYTHON_CMD --model $MODEL"
PYTHON_CMD="$PYTHON_CMD --num_agents $NUM_AGENTS"
PYTHON_CMD="$PYTHON_CMD --num_corrupt $NUM_CORRUPT"
PYTHON_CMD="$PYTHON_CMD --attack_type $ATTACK_TYPE"
PYTHON_CMD="$PYTHON_CMD --rounds $ROUNDS"
PYTHON_CMD="$PYTHON_CMD --local_epochs $LOCAL_EPOCHS"
PYTHON_CMD="$PYTHON_CMD --lr $LEARNING_RATE"
PYTHON_CMD="$PYTHON_CMD --batch_size $BATCH_SIZE"
PYTHON_CMD="$PYTHON_CMD --alignins_strict_threshold $STRICT_THRESHOLD"
PYTHON_CMD="$PYTHON_CMD --alignins_standard_threshold $STANDARD_THRESHOLD"
PYTHON_CMD="$PYTHON_CMD --suspicious_weight $SUSPICIOUS_WEIGHT"
PYTHON_CMD="$PYTHON_CMD --fedup_pruning_ratio $PRUNING_RATIO"
PYTHON_CMD="$PYTHON_CMD --fedup_adaptive $ADAPTIVE_PRUNING"
PYTHON_CMD="$PYTHON_CMD --seed $SEED"
PYTHON_CMD="$PYTHON_CMD --save_results $SAVE_RESULTS"
PYTHON_CMD="$PYTHON_CMD --results_dir $RESULTS_DIR"
PYTHON_CMD="$PYTHON_CMD --log_level $LOG_LEVEL"
PYTHON_CMD="$PYTHON_CMD --log_interval $LOG_INTERVAL"

if [ "$DEVICE" != "auto" ]; then
    PYTHON_CMD="$PYTHON_CMD --device $DEVICE"
fi

# =============================================================================
# 开始实验
# =============================================================================

echo "=========================================="
echo "开始运行实验..."
echo "命令: $PYTHON_CMD"
echo "=========================================="

# 记录开始时间
START_TIME=$(date)
echo "实验开始时间: $START_TIME"

# 运行实验
eval $PYTHON_CMD

# 记录结束时间
END_TIME=$(date)
echo "=========================================="
echo "实验完成!"
echo "开始时间: $START_TIME"
echo "结束时间: $END_TIME"

# 显示结果文件
if [ "$SAVE_RESULTS" = true ] && [ -d "$RESULTS_DIR" ]; then
    echo "=========================================="
    echo "实验结果文件:"
    ls -la "$RESULTS_DIR"/*.pt 2>/dev/null || echo "未找到结果文件"
fi

echo "=========================================="