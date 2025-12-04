#!/bin/bash

# =============================================================================
# 用户自定义配置运行脚本
# 基于用户常用的配置：两张GPU，特定的联邦学习参数
# =============================================================================

echo "=========================================="
echo "用户自定义配置 - AlignIns+FedUP实验"
echo "=========================================="

# =============================================================================
# 用户常用配置参数
# =============================================================================

# GPU配置
export CUDA_VISIBLE_DEVICES=0,1

# 基础实验参数 (基于用户提供的配置)
POISON_FRAC=0.3             # 投毒比例
NUM_CORRUPT=10              # 恶意客户端数量
NUM_AGENTS=40               # 客户端总数
DATA="cifar10"              # 数据集
ATTACK="badnet"             # 攻击类型
NON_IID="--non_iid"         # 非独立同分布
BETA=0.5                    # Dirichlet分布参数

# 聚合方法选择
AGGR_METHOD="alignins_fedup_correct"  # 使用正确的实现

# 其他参数 (可根据需要调整)
LOCAL_EP=2                  # 本地训练轮数
BS=64                       # 批次大小
CLIENT_LR=0.1               # 客户端学习率
SERVER_LR=1                 # 服务器学习率
ROUNDS=100                  # 通信轮数 (根据数据集自动调整)
NOT_FINETUNE_ROUNDS=5       # 微调轮数 (改为5轮)
NOT_FINETUNE_LOCAL_EP=1
NOT_FINETUNE_LR=0.0001

# AlignIns参数
ALIGNINS_STRICT_THRESHOLD=0.7
ALIGNINS_STANDARD_THRESHOLD=0.85
SUSPICIOUS_WEIGHT=0.3

# FedUP参数
FEDUP_PRUNING_RATIO=0.1
FEDUP_P_MAX=0.15
FEDUP_P_MIN=0.01
FEDUP_GAMMA=5
FEDUP_SENSITIVITY_THRESHOLD=0.5

# =============================================================================
# 预设配置选项
# =============================================================================

# 配置1：用户原始配置 + 正确实现
config_user_original() {
    echo "=== 用户原始配置 + 正确AlignIns+FedUP实现 ==="
    
    CUDA_VISIBLE_DEVICES=0,1 python src/federated.py \
        --poison_frac $POISON_FRAC \
        --num_corrupt $NUM_CORRUPT \
        --num_agents $NUM_AGENTS \
        --aggr $AGGR_METHOD \
        --data $DATA \
        --attack $ATTACK \
        $NON_IID \
        --beta $BETA \
        --local_ep $LOCAL_EP \
        --bs $BS \
        --client_lr $CLIENT_LR \
        --server_lr $SERVER_LR
}

# 配置2：增强版配置 (更多轮数)
config_enhanced() {
    echo "=== 增强版配置 (更多轮数) ==="
    
    CUDA_VISIBLE_DEVICES=0,1 python src/federated.py \
        --poison_frac $POISON_FRAC \
        --num_corrupt $NUM_CORRUPT \
        --num_agents $NUM_AGENTS \
        --aggr $AGGR_METHOD \
        --data $DATA \
        --attack $ATTACK \
        $NON_IID \
        --beta $BETA \
        --local_ep 5 \
        --bs $BS \
        --client_lr $CLIENT_LR \
        --server_lr $SERVER_LR \
        --rounds 200
}

# 配置3：高攻击强度配置
config_high_attack() {
    echo "=== 高攻击强度配置 ==="
    
    CUDA_VISIBLE_DEVICES=0,1 python src/federated.py \
        --poison_frac 0.5 \
        --num_corrupt 15 \
        --num_agents $NUM_AGENTS \
        --aggr $AGGR_METHOD \
        --data $DATA \
        --attack $ATTACK \
        $NON_IID \
        --beta $BETA \
        --local_ep $LOCAL_EP \
        --bs $BS \
        --client_lr $CLIENT_LR \
        --server_lr $SERVER_LR
}

# 配置4：DBA攻击配置
config_dba_attack() {
    echo "=== DBA攻击配置 ==="
    
    CUDA_VISIBLE_DEVICES=0,1 python src/federated.py \
        --poison_frac $POISON_FRAC \
        --num_corrupt $NUM_CORRUPT \
        --num_agents $NUM_AGENTS \
        --aggr $AGGR_METHOD \
        --data $DATA \
        --attack "DBA" \
        $NON_IID \
        --beta $BETA \
        --local_ep $LOCAL_EP \
        --bs $BS \
        --client_lr $CLIENT_LR \
        --server_lr $SERVER_LR
}

# 配置5：CIFAR-100配置
config_cifar100() {
    echo "=== CIFAR-100配置 ==="
    
    CUDA_VISIBLE_DEVICES=0,1 python src/federated.py \
        --poison_frac $POISON_FRAC \
        --num_corrupt $NUM_CORRUPT \
        --num_agents $NUM_AGENTS \
        --aggr $AGGR_METHOD \
        --data "cifar100" \
        --attack $ATTACK \
        $NON_IID \
        --beta $BETA \
        --local_ep $LOCAL_EP \
        --bs $BS \
        --client_lr $CLIENT_LR \
        --server_lr $SERVER_LR
}

# 配置6：对比实验 - 使用混合方法
config_compare_hybrid() {
    echo "=== 对比实验 - 混合方法 ==="
    
    CUDA_VISIBLE_DEVICES=0,1 python src/federated.py \
        --poison_frac $POISON_FRAC \
        --num_corrupt $NUM_CORRUPT \
        --num_agents $NUM_AGENTS \
        --aggr "alignins_fedup_hybrid" \
        --data $DATA \
        --attack $ATTACK \
        $NON_IID \
        --beta $BETA \
        --local_ep $LOCAL_EP \
        --bs $BS \
        --client_lr $CLIENT_LR \
        --server_lr $SERVER_LR
}

config_not_unlearning() {
    echo "=== NoT联邦遗忘方法 ==="
    
    CUDA_VISIBLE_DEVICES=0,1 python src/federated.py \
        --poison_frac $POISON_FRAC \
        --num_corrupt $NUM_CORRUPT \
        --num_agents $NUM_AGENTS \
        --aggr "not_unlearning" \
        --data $DATA \
        --attack $ATTACK \
        $NON_IID \
        --beta $BETA \
        --local_ep $LOCAL_EP \
        --bs $BS \
        --client_lr $CLIENT_LR \
        --server_lr $SERVER_LR \
        --not_finetune_rounds $NOT_FINETUNE_ROUNDS \
        --not_finetune_local_ep $NOT_FINETUNE_LOCAL_EP \
        --not_finetune_lr $NOT_FINETUNE_LR
}

# 配置9：IMS防御配置
config_ims() {
    echo "=== IMS防御配置 ==="
    
    CUDA_VISIBLE_DEVICES=0,1 python src/federated.py \
        --poison_frac $POISON_FRAC \
        --num_corrupt $NUM_CORRUPT \
        --num_agents $NUM_AGENTS \
        --aggr "ims" \
        --data $DATA \
        --attack $ATTACK \
        $NON_IID \
        --beta $BETA \
        --local_ep $LOCAL_EP \
        --bs $BS \
        --client_lr $CLIENT_LR \
        --server_lr $SERVER_LR \
        --ims_r1 20 \
        --ims_r2 15 \
        --ims_r3 5 \
        --ims_k 20 \
        --ims_epsilon 1.0
}

# 配置7：自定义参数配置
config_custom() {
    echo "=== 自定义参数配置 ==="
    echo "请根据需要修改脚本中的参数"
    
    # 在这里可以自定义任何参数
    CUSTOM_POISON_FRAC=${1:-$POISON_FRAC}
    CUSTOM_NUM_CORRUPT=${2:-$NUM_CORRUPT}
    CUSTOM_NUM_AGENTS=${3:-$NUM_AGENTS}
    CUSTOM_ATTACK=${4:-$ATTACK}
    
    CUDA_VISIBLE_DEVICES=0,1 python src/federated.py \
        --poison_frac $CUSTOM_POISON_FRAC \
        --num_corrupt $CUSTOM_NUM_CORRUPT \
        --num_agents $CUSTOM_NUM_AGENTS \
        --aggr $AGGR_METHOD \
        --data $DATA \
        --attack $CUSTOM_ATTACK \
        $NON_IID \
        --beta $BETA \
        --local_ep $LOCAL_EP \
        --bs $BS \
        --client_lr $CLIENT_LR \
        --server_lr $SERVER_LR
}

# =============================================================================
# 显示配置选项
# =============================================================================

show_configs() {
    echo "=========================================="
    echo "可用的配置选项:"
    echo "=========================================="
    echo "1. config_user_original   - 用户原始配置 + 正确实现"
    echo "2. config_enhanced        - 增强版配置 (更多轮数)"
    echo "3. config_high_attack     - 高攻击强度配置"
    echo "4. config_dba_attack      - DBA攻击配置"
    echo "5. config_cifar100        - CIFAR-100配置"
    echo "6. config_compare_hybrid  - 对比实验 (混合方法)"
    echo "7. config_custom          - 自定义参数配置"
    echo "8. config_not_unlearning  - NoT联邦遗忘方法"
    echo "9. config_ims             - IMS防御配置"
    echo "=========================================="
    echo "当前GPU配置: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    echo "当前聚合方法: $AGGR_METHOD"
    echo "=========================================="
    echo "使用方法:"
    echo "  bash run_user_config.sh                    # 交互式选择"
    echo "  bash run_user_config.sh config_user_original  # 直接运行配置1"
    echo "  bash run_user_config.sh config_custom 0.4 12 50 DBA  # 自定义参数"
    echo "=========================================="
}

# =============================================================================
# 交互式配置选择
# =============================================================================

interactive_config() {
    show_configs
    echo
    read -p "请选择配置 (1-9): " choice
    
    case $choice in
        1) config_user_original ;;
        2) config_enhanced ;;
        3) config_high_attack ;;
        4) config_dba_attack ;;
        5) config_cifar100 ;;
        6) config_compare_hybrid ;;
        7) 
            echo "自定义配置使用方法:"
            echo "bash run_user_config.sh config_custom [poison_frac] [num_corrupt] [num_agents] [attack]"
            echo "例如: bash run_user_config.sh config_custom 0.4 12 50 DBA"
            ;;
        8) config_not_unlearning ;;
        9) config_ims ;;
        *) echo "无效选择，请输入1-9之间的数字" ;;
    esac
}

# =============================================================================
# 主程序
# =============================================================================

# 检查GPU可用性
echo "检查GPU状态..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits | head -2
else
    echo "警告: 未找到nvidia-smi命令，无法检查GPU状态"
fi

echo "当前CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo

# 根据参数决定运行方式
if [ $# -eq 0 ]; then
    # 无参数，交互式选择
    interactive_config
elif [ "$1" = "config_custom" ]; then
    # 自定义配置，传递额外参数
    config_custom "$2" "$3" "$4" "$5"
else
    # 直接运行指定配置
    if declare -f "$1" > /dev/null; then
        $1
    else
        echo "错误: 未找到配置函数 $1"
        show_configs
        exit 1
    fi
fi
