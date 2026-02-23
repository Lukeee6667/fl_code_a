
import sys
import os
import argparse
import torch
import torch.nn as nn
import logging
import copy
import numpy as np
from torch.utils.data import DataLoader, Subset
import random

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

import utils
import models
from agg_ims_prune import IMSPruneAggregator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

class Args:
    def __init__(self):
        # Hyperparameters from log
        self.data = 'cifar10'
        self.num_agents = 40
        self.agent_frac = 1.0
        self.num_corrupt = 10
        self.rounds = 100
        self.local_ep = 2
        self.bs = 64
        self.client_lr = 0.1
        self.server_lr = 1.0
        self.target_class = 7
        self.poison_frac = 0.3
        self.pattern_type = 'plus'
        self.theta = 8
        self.theta_ld = 10
        self.snap = 1
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.num_workers = 0
        self.dense_ratio = 0.25
        self.anneal_factor = 0.0001
        self.se_threshold = 0.0001
        self.non_iid = True
        self.debug = False
        self.beta = 0.5
        self.attack = 'badnet'
        self.aggr = 'ims_prune_finetune'
        self.not_finetune_rounds = 5
        self.not_finetune_local_ep = 2
        self.not_finetune_lr = 0.0001
        self.lr_decay = 0.99
        self.momentum = 0.0
        self.mask_init = 'ERK'
        self.wd = 0.0001
        self.ims_start_round = 100
        self.ims_margin = 0.5
        self.same_mask = 1
        self.cease_poison = 100000
        self.super_power = False
        self.clean = False
        self.sparsity = 0.3
        self.lambda_s = 1.0
        self.lambda_c = 1.0
        self.lambda_g = 1.5
        self.lambda_mean_cos = 1.5
        self.suspicious_weight = 0.5
        self.strict_factor = 0.8
        self.fedup_pruning_ratio = 0.1
        self.fedup_p_max = 0.15
        self.fedup_p_min = 0.01
        self.fedup_gamma = 5
        self.fedup_sensitivity_threshold = 0.5
        self.ims_lr = 0.01
        self.ims_r1 = 20
        self.ims_r2 = 15
        self.ims_r3 = 5
        self.ims_k = 20.0
        self.ims_lambda_init = 0.0
        self.ims_lambda_final = 10.0
        self.ims_epsilon = 1.0
        self.ims_clean_agree_weight = 1.0
        self.ims_backdoor_recover_weight = 1.0
        self.resume = False
        self.num_target = 10
        
        # Paths
        self.log_dir = "logs/cifar10/attack_badnet_ar_0.25/defense_ims_prune_finetune/2026-02-01-16-57_noniid(0.5)_pr(0.3)/"
        self.checkpoint_path = os.path.join(self.log_dir, "checkpoint_99.pt")

def test_model(model, test_dataset, args, prefix="Test"):
    criterion = nn.CrossEntropyLoss().to(args.device)
    test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers)
    
    # Clean Accuracy
    clean_acc = utils.get_loss_n_accuracy(model, criterion, test_loader, args, 0)
    
    # ASR (Backdoor Accuracy)
    # Create poisoned dataset
    poison_dataset = copy.deepcopy(test_dataset)
    utils.poison_dataset(poison_dataset, args, poison_all=True)
    poison_loader = DataLoader(poison_dataset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers)
    
    asr = utils.get_loss_n_accuracy(model, criterion, poison_loader, args, 0)
    
    # Backdoor Accuracy (on target class, excluding target class samples if needed, but usually just ASR)
    # Here using the log's definition: ASR is usually success on poisoned images.
    # The log also has "Backdoor ACC", which might be accuracy on poisoned images for non-target classes?
    # Or maybe it's the same.
    # Let's stick to Clean ACC and ASR.
    
    logging.info(f"{prefix} - Clean ACC: {clean_acc:.4f}, ASR: {asr:.4f}")
    return clean_acc, asr

def main():
    args = Args()
    
    # Load Datasets
    logging.info("Loading datasets...")
    train_dataset, test_dataset = utils.get_datasets(args.data)
    # utils.get_datasets(data, args) ? No, get_datasets(data, args) signature check:
    # def get_datasets(data, data_dir):
    # wait, utils.get_datasets(args.data, './data') usually.
    # In utils.py: def get_datasets(data, data_dir=None): ?
    # Let's re-read utils.py signature carefully.
    
    # Load Model
    logging.info("Loading model...")
    global_model = models.get_model(args.data, args)
    global_model.to(args.device)
    
    # Load Checkpoint
    if os.path.isfile(args.checkpoint_path):
        logging.info(f"Loading checkpoint '{args.checkpoint_path}'")
        checkpoint = torch.load(args.checkpoint_path)
        global_model.load_state_dict(checkpoint['state_dict'])
        logging.info(f"Loaded checkpoint round {checkpoint['round']}")
    else:
        logging.error(f"Checkpoint not found at {args.checkpoint_path}")
        return

    # Create Auxiliary Data Loader (100 samples from test set)
    # IMS uses a small clean dataset.
    aux_indices = list(range(100))
    aux_dataset = Subset(test_dataset, aux_indices)
    aux_loader = DataLoader(aux_dataset, batch_size=args.bs, shuffle=True)
    
    # Baseline Test
    logging.info("--- Baseline (Round 99 Checkpoint) ---")
    test_model(global_model, test_dataset, args, prefix="Baseline")
    
    # IMS Pruning
    logging.info("--- Starting IMS Pruning ---")
    aggregator = IMSPruneAggregator(args, args.device)
    
    # Get prunable layers
    prunable_layers = aggregator._get_prunable_layers(global_model)
    
    # Step A: Mask Initialization
    logging.info("Identifying structural masks...")
    A_init, S_init = aggregator.mask_initialization(aux_loader, global_model, prunable_layers)
    
    # Step B: Inner Subproblem
    delta_dict = aggregator.inner_subproblem(aux_loader, global_model, A_init, S_init, prunable_layers)
    
    # Step C: Outer Subproblem
    A_final, S_final = aggregator.outer_subproblem(aux_loader, global_model, delta_dict, A_init, S_init, prunable_layers)
    
    # Step 4: Apply Defense (Structured Pruning)
    # We create a COPY for the pruned model to test it before fine-tuning
    pruned_model = aggregator.deploy_defense(global_model, A_final, S_final, prunable_layers)
    
    # Test Pruned Model (Before Fine-tuning)
    logging.info("--- Pruned Model (No Fine-tuning) ---")
    test_model(pruned_model, test_dataset, args, prefix="Pruned")
    
    # Step 5: Fine-tuning
    logging.info("--- Starting Fine-tuning ---")
    # Note: fine_tune_model modifies model in-place
    aggregator.fine_tune_model(pruned_model, aux_loader, A_final, S_final, prunable_layers)
    
    # Test Fine-tuned Model
    logging.info("--- Fine-tuned Model ---")
    test_model(pruned_model, test_dataset, args, prefix="Fine-tuned")

if __name__ == "__main__":
    # Fix for utils.get_datasets arguments
    # I need to check utils.get_datasets signature again to be sure.
    main()
