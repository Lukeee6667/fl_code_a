#!/bin/bash
POISON_FRAC=0.3
NUM_CORRUPT=10
NUM_AGENTS=40
DATA="cifar10"
ATTACKS=("badnet" "DBA" "neurotoxin" "pgd")
NON_IID="--non_iid"
BETA_VALUES=(0.5 0.3)
LOCAL_EP=2
BS=64
CLIENT_LR=0.1
SERVER_LR=1
AGGR_METHOD="alignins"
export CUDA_VISIBLE_DEVICES="1"
for BETA in "${BETA_VALUES[@]}"; do
    for ATTACK in "${ATTACKS[@]}"; do
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
    done
done
