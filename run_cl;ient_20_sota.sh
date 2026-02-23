#!/bin/bash
POISON_FRAC=0.5
NUM_CORRUPT=4
NUM_AGENTS=20
DATA="cifar10"
ATTACKS=("badnet" "DBA" "neurotoxin" "pgd")
NON_IID="--non_iid"
BETA=0.5
LOCAL_EP=2
BS=64
CLIENT_LR=0.1
SERVER_LR=1
ROUNDS=150
NOT_FINETUNE_LOCAL_EP=10
NOT_FINETUNE_LR=0.0001
export CUDA_VISIBLE_DEVICES="0"

for ATTACK in "${ATTACKS[@]}"; do
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
        --rounds $ROUNDS \
        --lambda_s 1.0 \
        --lambda_c 1.0 \
        --lambda_g 1.5 \
        --lambda_mean_cos 1.5 \
        --ims_start_round 150 \
        --ims_r1 20 \
        --ims_r2 15 \
        --ims_r3 5 \
        --ims_k 20 \
        --ims_epsilon 1.0 \
        --not_finetune_local_ep $NOT_FINETUNE_LOCAL_EP \
        --not_finetune_lr $NOT_FINETUNE_LR

done