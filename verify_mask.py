import torch
import os
import sys
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def verify_checkpoint_mask(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        logging.error(f"Checkpoint not found: {checkpoint_path}")
        return

    logging.info(f"Loading checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        logging.error(f"Failed to load checkpoint: {e}")
        return

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        logging.info("Loaded from 'model_state_dict' key.")
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        logging.info("Loaded from 'state_dict' key.")
    elif isinstance(checkpoint, dict):
        # Assume the dict itself is the state dict if no specific keys found
        # But check if it has metadata like 'epoch', 'round', etc.
        if any(k in checkpoint for k in ['epoch', 'round', 'args']):
             # Try to find the model part
             keys = list(checkpoint.keys())
             logging.info(f"Checkpoint keys: {keys}")
             # If strictly these keys, it might not contain model if not obvious
             # Usually 'model' or just the dict is weights
             if 'model' in checkpoint:
                 state_dict = checkpoint['model']
                 logging.info("Loaded from 'model' key.")
             else:
                 # Check if values are tensors
                 if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                     state_dict = checkpoint
                     logging.info("Loaded dictionary as state_dict (all values are tensors).")
                 else:
                     logging.error("Could not identify state_dict in checkpoint.")
                     return
    else:
        # Assume direct state_dict
        state_dict = checkpoint
        logging.info("Loaded object directly as state_dict.")

    total_params = 0
    zero_params = 0
    
    logging.info("Analyzing parameter sparsity...")
    
    layer_stats = []

    for name, param in state_dict.items():
        # Only check weight parameters for pruning, usually bias is not pruned or counts less
        # But IMS might prune everything. Let's check everything.
        if param.dim() > 0: # Skip scalars if any
            n_params = param.numel()
            n_zeros = torch.sum(param == 0).item()
            
            sparsity = n_zeros / n_params
            total_params += n_params
            zero_params += n_zeros
            
            layer_stats.append({
                'name': name,
                'total': n_params,
                'zeros': n_zeros,
                'sparsity': sparsity
            })
            
            # logging.info(f"Layer: {name} | Size: {param.size()} | Sparsity: {sparsity:.4f} ({n_zeros}/{n_params})")

    global_sparsity = zero_params / total_params if total_params > 0 else 0
    
    logging.info("="*50)
    logging.info(f"Global Statistics for {os.path.basename(checkpoint_path)}")
    logging.info("="*50)
    logging.info(f"Total Parameters: {total_params}")
    logging.info(f"Zero Parameters:  {zero_params}")
    logging.info(f"Global Sparsity:  {global_sparsity:.4%}")
    logging.info("="*50)
    
    # Print layers with high sparsity (likely pruned)
    logging.info("Pruned Layers (Sparsity > 0):")
    for stat in layer_stats:
        if stat['zeros'] > 0:
             logging.info(f"  {stat['name']:<40} | Sparsity: {stat['sparsity']:.4%} ({stat['zeros']}/{stat['total']})")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        ckpt_path = sys.argv[1]
    else:
        # Default to the one provided by user
        ckpt_path = "/home/dell/fl_code/fl_code_260114/fl_code_a/logs/cifar10/attack_badnet_ar_0.25/defense_ims_prune_finetune/2026-02-01-16-57_noniid(0.5)_pr(0.3)/checkpoint_100.pt"
    
    verify_checkpoint_mask(ckpt_path)
