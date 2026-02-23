#!/bin/bash

# =============================================================================
# 用户自定义配置运行脚本
# 基于用户常用的配置：两张GPU，特定的联邦学习参数
# =============================================================================

echo "=========================================="
echo "用户自定义配置 - AlignIns+FedUP实验"
echo "=========================================="

# =============================================================================
# GPU 选择逻辑
# =============================================================================

setup_gpu() {
    echo "=========================================="
    echo "正在检测可用GPU..."
    if command -v nvidia-smi &> /dev/null; then
        echo "检测到以下GPU设备:"
        # 显示GPU索引、名称、显存使用情况
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
    else
        echo "警告: 未找到 nvidia-smi 命令，无法列出GPU详情。"
    fi
    echo "=========================================="
    
    # 默认值
    DEFAULT_GPU="0,1"
    
    read -p "请输入要使用的GPU编号 (例如 0 或 0,1) [默认: $DEFAULT_GPU]: " GPU_INPUT
    
    if [ -z "$GPU_INPUT" ]; then
        GPU_IDS="$DEFAULT_GPU"
    else
        GPU_IDS="$GPU_INPUT"
    fi
    
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
    echo "已设置 CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    echo "=========================================="
}

# 在脚本开始时调用 GPU 设置
setup_gpu

# =============================================================================
# 用户常用配置参数
# =============================================================================

# 基础实验参数 (基于用户提供的配置)
POISON_FRAC=0.3             # 投毒比例
NUM_CORRUPT=10              # 恶意客户端数量
NUM_AGENTS=40               # 客户端总数
DATA="cifar10"              # 数据集
ATTACK="neurotoxin"             # 攻击类型
NON_IID="--non_iid"         # 非独立同分布
BETA=0.5                    # Dirichlet分布参数

# 聚合方法选择
AGGR_METHOD="alignins"  # 使用正确的实现

# 其他参数 (可根据需要调整)
LOCAL_EP=2                  # 本地训练轮数
BS=64                       # 批次大小
CLIENT_LR=0.1               # 客户端学习率
SERVER_LR=1                 # 服务器学习率
ROUNDS=100                  # 通信轮数 (根据数据集自动调整)
NOT_FINETUNE_ROUNDS=1       # 微调轮数 (改为5轮)
NOT_FINETUNE_LOCAL_EP=10
NOT_FINETUNE_LR=0.0001

# AlignIns参数
ALIGNINS_STRICT_THRESHOLD=0.7
ALIGNINS_STANDARD_THRESHOLD=1.5
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
    
    python src/federated.py \
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
    
    python src/federated.py \
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
    
    python src/federated.py \
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
    
    python src/federated.py \
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
    
    python src/federated.py \
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
    
    python src/federated.py \
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
    
    python src/federated.py \
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
    
    python src/federated.py \
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

# 配置10：AlignIns + NoT Unlearning
config_alignins_not_unlearning() {
    echo "=== AlignIns + NoT Unlearning ==="
    
    python src/federated.py \
        --poison_frac $POISON_FRAC \
        --num_corrupt $NUM_CORRUPT \
        --num_agents $NUM_AGENTS \
        --aggr "alignins_not_unlearning" \
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

# 配置11：AlignIns + IMS
config_alignins_ims() {
    echo "=== AlignIns + IMS ==="
    
    python src/federated.py \
        --poison_frac $POISON_FRAC \
        --num_corrupt $NUM_CORRUPT \
        --num_agents $NUM_AGENTS \
        --aggr "alignins_ims" \
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
        --ims_lambda_final 0.5 \
        --ims_epsilon 1.0 \
        --suspicious_weight $SUSPICIOUS_WEIGHT \
        --strict_factor $ALIGNINS_STRICT_THRESHOLD \
        --lambda_s $ALIGNINS_STANDARD_THRESHOLD \
        --lambda_c $ALIGNINS_STANDARD_THRESHOLD \
        --lambda_g $ALIGNINS_STANDARD_THRESHOLD \
        --lambda_mean_cos $ALIGNINS_STANDARD_THRESHOLD
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
    
    python src/federated.py \
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

# 配置12：A4FL 防御配置
config_a4fl() {
    echo "=== A4FL 防御配置 ==="
    
    python src/federated.py \
        --poison_frac $POISON_FRAC \
        --num_corrupt $NUM_CORRUPT \
        --num_agents $NUM_AGENTS \
        --aggr a4fl \
        --data $DATA \
        --attack $ATTACK \
        $NON_IID \
        --beta $BETA \
        --local_ep $LOCAL_EP \
        --bs $BS \
        --client_lr $CLIENT_LR \
        --server_lr $SERVER_LR
}

# 配置13：A4FL + AlignIns 混合防御配置
config_a4fl_alignins() {
    echo "=== A4FL + AlignIns 混合防御配置 ==="
    echo "本地防御：A4FL (Unlearning + UAP + Pruning)"
    echo "全局聚合：AlignIns + FedUP (Statistical Filtering + Pruning)"
    
    python src/federated.py \
        --poison_frac $POISON_FRAC \
        --num_corrupt $NUM_CORRUPT \
        --num_agents $NUM_AGENTS \
        --aggr a4fl_alignins \
        --data $DATA \
        --attack $ATTACK \
        $NON_IID \
        --beta $BETA \
        --local_ep $LOCAL_EP \
        --bs $BS \
        --client_lr $CLIENT_LR \
        --server_lr $SERVER_LR
}

# 配置14：AlignIns + IMS (Mask Recovery)
config_alignins_ims_recover() {
    echo "=== AlignIns + IMS (Mask Recovery) ==="
    echo "引入掩码恢复机制的 IMS 改进版"
    
    python src/federated.py \
        --poison_frac $POISON_FRAC \
        --num_corrupt $NUM_CORRUPT \
        --num_agents $NUM_AGENTS \
        --aggr "alignins_ims_recover" \
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
        --ims_epsilon 1.0 \
        --suspicious_weight $SUSPICIOUS_WEIGHT \
        --strict_factor $ALIGNINS_STRICT_THRESHOLD \
        --lambda_s $ALIGNINS_STANDARD_THRESHOLD \
        --lambda_c $ALIGNINS_STANDARD_THRESHOLD \
        --lambda_g $ALIGNINS_STANDARD_THRESHOLD \
        --lambda_mean_cos $ALIGNINS_STANDARD_THRESHOLD
}

# 配置15：IMS Fast (Optimized)
config_ims_fast() {
    echo "=== IMS Fast (Optimized) ==="
    echo "IMS优化版：In-place掩码应用 + Clean Output缓存"
    
    python src/federated.py \
        --poison_frac $POISON_FRAC \
        --num_corrupt $NUM_CORRUPT \
        --num_agents $NUM_AGENTS \
        --aggr "ims_fast" \
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

# 配置16：AlignIns + IMS Fast Plus
config_alignins_ims_fast_plus() {
    echo "=== AlignIns + IMS Fast Plus ==="
    echo "对齐过滤 + IMS快速版 + 额外稳健损失"
    
    python src/federated.py \
        --poison_frac $POISON_FRAC \
        --num_corrupt $NUM_CORRUPT \
        --num_agents $NUM_AGENTS \
        --aggr "alignins_ims_fast_plus" \
        --data $DATA \
        --attack $ATTACK \
        $NON_IID \
        --beta $BETA \
        --local_ep $LOCAL_EP \
        --bs $BS \
        --client_lr $CLIENT_LR \
        --server_lr $SERVER_LR \
        --ims_start_round 100 \
        --ims_r1 20 \
        --ims_r2 15 \
        --ims_r3 5 \
        --ims_k 20 \
        --ims_lambda_final 10.0 \
        --ims_epsilon 1.0 \
        --ims_clean_agree_weight 1.0 \
        --ims_backdoor_recover_weight 1.0 \
        --ims_poison_entropy_weight 0.0 \
        --suspicious_weight 0.3 \
        --strict_factor 0.8 \
        --lambda_s $ALIGNINS_STANDARD_THRESHOLD \
        --lambda_c $ALIGNINS_STANDARD_THRESHOLD \
        --lambda_g $ALIGNINS_STANDARD_THRESHOLD \
        --lambda_mean_cos $ALIGNINS_STANDARD_THRESHOLD
}

# 配置17：AlignIns + IMS Standard (Custom Request)
config_alignins_ims_standard() {
    echo "=== AlignIns + IMS Standard (Config 17) ==="
    echo "AlignIns (Strict=0.8) + 普通 IMS (Config 9)"
    
    python src/federated.py \
        --poison_frac $POISON_FRAC \
        --num_corrupt $NUM_CORRUPT \
        --num_agents $NUM_AGENTS \
        --aggr "alignins_ims" \
        --data $DATA \
        --attack $ATTACK \
        $NON_IID \
        --beta $BETA \
        --local_ep $LOCAL_EP \
        --bs $BS \
        --client_lr $CLIENT_LR \
        --server_lr $SERVER_LR \
        --ims_start_round 100 \
        --ims_r1 20 \
        --ims_r2 15 \
        --ims_r3 5 \
        --ims_k 20 \
        --ims_epsilon 1.0 \
        --suspicious_weight 0.3 \
        --strict_factor 0.8 \
        --lambda_s $ALIGNINS_STANDARD_THRESHOLD \
        --lambda_c $ALIGNINS_STANDARD_THRESHOLD \
        --lambda_g $ALIGNINS_STANDARD_THRESHOLD \
        --lambda_mean_cos $ALIGNINS_STANDARD_THRESHOLD
}

# 配置18：IMS Prune + Finetune
config_ims_prune_finetune() {
    echo "=== IMS Prune + Finetune (Config 18) ==="
    echo "IMS结构化剪枝 + 轻量化微调 + 交互约束"
    
    python src/federated.py \
        --poison_frac $POISON_FRAC \
        --num_corrupt $NUM_CORRUPT \
        --num_agents $NUM_AGENTS \
        --aggr "ims_prune_finetune" \
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
        --ims_epsilon 1.0 \
        --not_finetune_rounds 5 \
        --not_finetune_local_ep 2 \
        --not_finetune_lr 0.0001
}

# 配置18-Stage1：运行至99轮并保存
config_ims_prune_finetune_stage1() {
    echo "=== IMS Prune + Finetune (Stage 1: 0-99 Rounds) ==="
    echo "运行前99轮，生成checkpoint用于后续调试"
    
    python src/federated.py \
        --poison_frac $POISON_FRAC \
        --num_corrupt $NUM_CORRUPT \
        --num_agents $NUM_AGENTS \
        --aggr "ims_prune_finetune" \
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
        --ims_epsilon 1.0 \
        --not_finetune_rounds 5 \
        --not_finetune_local_ep 2 \
        --not_finetune_lr 0.0001 \
        --rounds 99
}

# 配置18-Debug：从99轮恢复并调试
config_ims_prune_finetune_debug() {
    CHECKPOINT_PATH=$1
    if [ -z "$CHECKPOINT_PATH" ]; then
        echo "错误: 请提供Checkpoint路径"
        echo "用法: bash run_user_config.sh config_ims_prune_finetune_debug <checkpoint_path>"
        return
    fi

    echo "=== IMS Prune + Finetune (Debug: Resume from 99) ==="
    echo "从Checkpoint恢复，执行第100轮(剪枝+微调)及后续"
    
    python src/federated.py \
        --poison_frac $POISON_FRAC \
        --num_corrupt $NUM_CORRUPT \
        --num_agents $NUM_AGENTS \
        --aggr "ims_prune_finetune" \
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
        --ims_epsilon 1.0 \
        --not_finetune_rounds 5 \
        --not_finetune_local_ep 2 \
        --not_finetune_lr 0.0001 \
        --rounds 100 \
        --resume \
        --checkpoint_path "$CHECKPOINT_PATH"
}

# 配置19：原始AlignIns配置
config_origin_alignins() {
    echo "=== 原始AlignIns配置 (Config 19) ==="
    echo "使用origin_alignins聚合方法"
    
    python src/federated.py \
        --poison_frac $POISON_FRAC \
        --num_corrupt $NUM_CORRUPT \
        --num_agents $NUM_AGENTS \
        --aggr "origin_alignins" \
        --data $DATA \
        --attack $ATTACK \
        $NON_IID \
        --beta $BETA \
        --local_ep $LOCAL_EP \
        --bs $BS \
        --client_lr $CLIENT_LR \
        --server_lr $SERVER_LR
}

# 配置20：原始AlignIns配置 + 4指标
config_origin_alignins_4metrics() {
    echo "=== 原始AlignIns配置 + 4指标 (Config 20) ==="
    echo "使用origin_alignins_4metrics聚合方法"
    
    python src/federated.py \
        --poison_frac $POISON_FRAC \
        --num_corrupt $NUM_CORRUPT \
        --num_agents $NUM_AGENTS \
        --aggr "origin_alignins_4metrics" \
        --data $DATA \
        --attack $ATTACK \
        $NON_IID \
        --beta $BETA \
        --local_ep $LOCAL_EP \
        --bs $BS \
        --client_lr $CLIENT_LR \
        --server_lr $SERVER_LR \
        --lambda_s 1.0 \
        --lambda_c 1.0 \
        --lambda_g 1.5 \
        --lambda_mean_cos 1.5
}

# 配置22：Clustering + IMS Prune + Finetune
config_origin_alignins_clustering_prune_finetune() {
    echo "=== Clustering + IMS Prune + Finetune (Config 22) ==="
    echo "使用origin_alignins_clustering_prune_finetune聚合方法"
    echo "结合了聚类检测和IMS结构化剪枝+微调"
    
    python src/federated.py \
        --poison_frac $POISON_FRAC \
        --num_corrupt $NUM_CORRUPT \
        --num_agents $NUM_AGENTS \
        --aggr "origin_alignins_clustering_prune_finetune" \
        --data $DATA \
        --attack $ATTACK \
        $NON_IID \
        --beta $BETA \
        --local_ep $LOCAL_EP \
        --bs $BS \
        --client_lr $CLIENT_LR \
        --server_lr $SERVER_LR \
        --rounds 105 \
        --lambda_s 1.0 \
        --lambda_c 1.0 \
        --lambda_g 1.5 \
        --lambda_mean_cos 1.5 \
        --ims_start_round 100 \
        --ims_r1 20 \
        --ims_r2 15 \
        --ims_r3 5 \
        --ims_k 20 \
        --ims_epsilon 1.0 \
        --not_finetune_rounds 1 \
        --not_finetune_local_ep 10 \
        --not_finetune_lr 0.0001
}

# 配置21：原始AlignIns配置 + 聚类
config_origin_alignins_clustering() {
    echo "=== 原始AlignIns配置 + 聚类 (Config 21) ==="
    echo "使用origin_alignins_clustering聚合方法"
    
    python src/federated.py \
        --poison_frac $POISON_FRAC \
        --num_corrupt $NUM_CORRUPT \
        --num_agents $NUM_AGENTS \
        --aggr "origin_alignins_clustering" \
        --data $DATA \
        --attack $ATTACK \
        $NON_IID \
        --beta $BETA \
        --local_ep $LOCAL_EP \
        --bs $BS \
        --client_lr $CLIENT_LR \
        --server_lr $SERVER_LR \
        --lambda_s 1.0 \
        --lambda_c 1.0 \
        --lambda_g 1.5 \
        --lambda_mean_cos 1.5
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
    echo "10. config_alignins_not_unlearning - AlignIns + NoT Unlearning"
    echo "11. config_alignins_ims   - AlignIns + IMS"
    echo "12. config_a4fl           - A4FL 防御配置"
    echo "13. config_a4fl_alignins  - A4FL + AlignIns 混合防御"
    echo "14. config_alignins_ims_recover - AlignIns + IMS (Mask Recovery)"
    echo "15. config_ims_fast         - IMS Fast (Optimized)"
    echo "16. config_alignins_ims_fast_plus - AlignIns + IMS Fast Plus"
    echo "17. config_alignins_ims_standard - AlignIns + IMS Standard"
    echo "18. config_ims_prune_finetune - IMS Prune + Finetune (Config 18)"
    echo "    config_ims_prune_finetune_stage1 - Config 18 Stage 1 (Run to 99)"
    echo "    config_ims_prune_finetune_debug  - Config 18 Debug (Resume from checkpoint)"
    echo "19. config_origin_alignins - 原始AlignIns配置 (Config 19)"
    echo "20. config_origin_alignins_4metrics - 原始AlignIns配置 + 4指标 (Config 20)"
    echo "21. config_origin_alignins_clustering - 原始AlignIns配置 + 聚类 (Config 21)"
    echo "22. config_origin_alignins_clustering_prune_finetune - Clustering + IMS Prune + Finetune (Config 22)"
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
    read -p "请选择配置 (1-22): " choice
    
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
        10) config_alignins_not_unlearning ;;
        11) config_alignins_ims ;;
        12) config_a4fl ;;
        13) config_a4fl_alignins ;;
        14) config_alignins_ims_recover ;;
        15) config_ims_fast ;;
        16) config_alignins_ims_fast_plus ;;
        17) config_alignins_ims_standard ;;
        18) config_ims_prune_finetune ;;
        19) config_origin_alignins ;;
        20) config_origin_alignins_4metrics ;;
        21) config_origin_alignins_clustering ;;
        22) config_origin_alignins_clustering_prune_finetune ;;
        *) echo "无效选择，请输入1-22之间的数字" ;;
    esac
}

# =============================================================================
# Main Program
# =============================================================================

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
        "$@"
    else
        echo "错误: 未找到配置函数 $1"
        show_configs
        exit 1
    fi
fi
