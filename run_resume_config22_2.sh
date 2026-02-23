#!/bin/bash

# =============================================================================
# 恢复训练脚本 (Config 22)
# 从指定的 Checkpoint 恢复并继续训练
# =============================================================================

# Checkpoint 路径 (请确保路径正确)
CKPT_PATH="/home/dell/fl_code/fl_code_260114/fl_code_a/logs/cifar10/attack_pgd_ar_0.25/defense_origin_alignins_clustering/2026-02-09-23-07_noniid(0.5)_pr(0.3)/checkpoint_100.pt"

if [ ! -f "$CKPT_PATH" ]; then
    echo "错误: Checkpoint 文件不存在: $CKPT_PATH"
    exit 1
fi

echo "从 Checkpoint 恢复: $CKPT_PATH"

# 基础参数 (必须与原始训练一致)
POISON_FRAC=0.3
NUM_CORRUPT=10
NUM_AGENTS=40
DATA="cifar10"
ATTACK="pgd"
NON_IID="--non_iid"
BETA=0.5
LOCAL_EP=2
BS=64
CLIENT_LR=0.1
SERVER_LR=1

# GPU 设置
export CUDA_VISIBLE_DEVICES="1"

echo "=== Resuming Clustering + IMS Prune + Finetune (Config 22) ==="

python src/federated.py \
    --resume \
    --checkpoint_path "$CKPT_PATH" \
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
    --rounds 101 \
    --lambda_s 1.0 \
    --lambda_c 1.0 \
    --lambda_g 1.5 \
    --lambda_mean_cos 1.5 \
    --ims_start_round 101 \
    --ims_r1 20 \
    --ims_r2 15 \
    --ims_r3 5 \
    --ims_k 20 \
    --ims_epsilon 1.0 \
    --aux_num_samples 10000 \
    --not_finetune_rounds 1 \
    --not_finetune_local_ep 10 \
    --not_finetune_lr 0.0001
