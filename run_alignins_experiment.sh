#!/bin/bash

# 使用第二个GPU (ID: 1)
export CUDA_VISIBLE_DEVICES=1

python src/federated.py \
     --poison_frac 0.3 \
     --num_corrupt 10 \
     --num_agents 40 \
     --aggr alignins \
     --data cifar10 \
     --attack badnet \
     --non_iid \
     --beta 0.5 \
     --local_ep 2 \
     --bs 64 \
     --client_lr 0.1 \
     --server_lr 1
