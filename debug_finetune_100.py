
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import copy
from torch.utils.data import DataLoader, Subset

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

import utils
import models

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

class Args:
    def __init__(self):
        # Hyperparameters
        self.data = 'cifar10'
        self.bs = 64
        self.num_workers = 0
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Attack/Defense params needed for utils/models
        self.num_agents = 40
        self.num_corrupt = 10
        self.target_class = 7
        self.poison_frac = 0.3
        self.pattern_type = 'plus'
        self.attack = 'badnet'
        
        # Fine-tuning params
        self.lr = 0.0001 # Reduced LR
        self.epochs = 10
        self.num_aux_samples = 500
        
        # Checkpoint
        self.log_dir = "logs/cifar10/attack_badnet_ar_0.25/defense_ims_prune_finetune/2026-02-01-16-57_noniid(0.5)_pr(0.3)/"
        self.checkpoint_path = os.path.join(self.log_dir, "checkpoint_100.pt")

def test_model(model, test_dataset, args, prefix="Test"):
    criterion = nn.CrossEntropyLoss().to(args.device)
    
    # 1. Clean ACC
    test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers)
    clean_acc = utils.get_loss_n_accuracy(model, criterion, test_loader, args, 0)
    
    # 2. ASR (Attack Success Ratio) - modify_label=True
    # Need to filter out target class samples for ASR typically? 
    # In federated.py: idxs = (val_dataset.targets != args.target_class).nonzero().flatten().tolist()
    # Let's replicate that.
    
    # Filter indices where target is NOT the target class
    non_target_idxs = (test_dataset.targets != args.target_class).nonzero().flatten().tolist()
    
    # Create dataset for ASR
    asr_dataset = copy.deepcopy(test_dataset)
    # We use utils.DatasetSplit to wrap it if we want to poison specific indices, 
    # but utils.poison_dataset takes the dataset directly.
    # However, utils.poison_dataset modifies in place.
    # Let's look at federated.py again.
    # poisoned_val_set = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs)
    # utils.poison_dataset(poisoned_val_set.dataset, args, idxs, poison_all=True)
    # Wait, DatasetSplit has .dataset attribute? 
    # Let's just pass the subset or handle it manually.
    
    # Simpler approach:
    # utils.poison_dataset modifies dataset.data/targets at given indices.
    # If we want ASR on non-target classes:
    asr_dataset_subset = torch.utils.data.Subset(asr_dataset, non_target_idxs)
    # But poison_dataset expects a full dataset structure typically or we pass indices.
    # Let's assume we can just pass indices to poison_dataset.
    
    utils.poison_dataset(asr_dataset, args, data_idxs=non_target_idxs, poison_all=True, modify_label=True)
    # Now create loader for these indices
    asr_loader = DataLoader(torch.utils.data.Subset(asr_dataset, non_target_idxs), batch_size=args.bs, shuffle=False, num_workers=args.num_workers)
    asr = utils.get_loss_n_accuracy(model, criterion, asr_loader, args, 0)
    
    # 3. Backdoor ACC (BA) - modify_label=False
    ba_dataset = copy.deepcopy(test_dataset)
    utils.poison_dataset(ba_dataset, args, data_idxs=non_target_idxs, poison_all=True, modify_label=False)
    ba_loader = DataLoader(torch.utils.data.Subset(ba_dataset, non_target_idxs), batch_size=args.bs, shuffle=False, num_workers=args.num_workers)
    ba_acc = utils.get_loss_n_accuracy(model, criterion, ba_loader, args, 0)
    
    logging.info("------------------------------")
    logging.info(f"{prefix} Results:")
    logging.info("Clean ACC:              %.4f" % clean_acc)
    logging.info("Attack Success Ratio:   %.4f" % asr)
    logging.info("Backdoor ACC:           %.4f" % ba_acc)
    logging.info("------------------------------")
    
    return clean_acc, asr, ba_acc

def main():
    args = Args()
    
    # Load Datasets
    logging.info("Loading datasets...")
    train_dataset, test_dataset = utils.get_datasets(args.data)
    
    # Load Model
    logging.info("Loading model...")
    global_model = models.get_model(args.data, args)
    global_model.to(args.device)
    
    # Load Checkpoint 100
    if os.path.isfile(args.checkpoint_path):
        logging.info(f"Loading checkpoint '{args.checkpoint_path}'")
        checkpoint = torch.load(args.checkpoint_path)
        global_model.load_state_dict(checkpoint['state_dict'])
        logging.info(f"Loaded checkpoint round {checkpoint['round']}")
    else:
        logging.error(f"Checkpoint not found at {args.checkpoint_path}")
        return

    # Evaluate Baseline
    logging.info("--- Baseline (Round 100 Checkpoint) ---")
    test_model(global_model, test_dataset, args, prefix="Baseline")
    
    # Prepare Fine-tuning
    logging.info(f"--- Preparing Fine-tuning (Samples: {args.num_aux_samples}, LR: {args.lr}) ---")
    
    # Auxiliary Data
    # Use a random subset of 500 samples from test set (or train set? usually test/val set is used for defense assuming server has some clean data)
    # federated.py uses val_dataset (which is test_dataset in this script)
    aux_indices = list(range(args.num_aux_samples))
    aux_dataset = Subset(test_dataset, aux_indices)
    aux_loader = DataLoader(aux_dataset, batch_size=args.bs, shuffle=True)
    
    # Infer Masks from Weights (Interaction Constraint)
    masks = {}
    pruned_count = 0
    total_count = 0
    for name, param in global_model.named_parameters():
        if param.requires_grad:
            # Create mask: 1 if weight != 0, else 0
            # Use a small threshold for float comparison
            mask = (param.abs() > 1e-6).float().to(args.device)
            masks[name] = mask
            
            p_count = (mask == 0).sum().item()
            t_count = mask.numel()
            pruned_count += p_count
            total_count += t_count
            
    logging.info(f"Inferred Masks: {pruned_count}/{total_count} parameters pruned ({pruned_count/total_count:.2%})")
    
    # Optimizer
    optimizer = optim.SGD(global_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    global_model.eval() # Keep BN stats frozen
    for param in global_model.parameters():
        param.requires_grad = True
        
    # Fine-tuning Loop
    for epoch in range(args.epochs):
        total_loss = 0
        batches = 0
        
        for x, y in aux_loader:
            x, y = x.to(args.device), y.to(args.device)
            optimizer.zero_grad()
            output = global_model(x)
            loss = criterion(output, y)
            loss.backward()
            
            # Apply Interaction Constraint (Gradient Masking)
            with torch.no_grad():
                for name, param in global_model.named_parameters():
                    if name in masks and param.grad is not None:
                        param.grad.mul_(masks[name])
            
            optimizer.step()
            total_loss += loss.item()
            batches += 1
            
        avg_loss = total_loss / batches
        logging.info(f"Fine-tuning Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}")
        
        # Evaluate
        test_model(global_model, test_dataset, args, prefix=f"Epoch {epoch+1}")

if __name__ == "__main__":
    main()
