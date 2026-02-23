"""
IMS Prune + Finetune Aggregator
Extends IMS Fast to include structured pruning and lightweight fine-tuning with interaction constraints.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import copy
import logging
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from agg_ims_fast import IMSAggregator
import utils
from torch.utils.data import DataLoader

class IMSPruneAggregator(IMSAggregator):
    def __init__(self, args, device):
        super().__init__(args, device)
        self.saved_mask_A = None
        self.saved_mask_S = None
        self.saved_prunable_layers = None
        self.has_pruned = False

    def aggregate(self, agent_updates_dict, flat_global_model, global_model, auxiliary_data_loader, initial_update=None, current_round=None, auxiliary_data_loader_finetune=None, val_loader=None, poisoned_val_loader=None, poisoned_val_only_x_loader=None):
        start_round = getattr(self.args, 'ims_start_round', 100)
        
        if current_round is not None and current_round >= start_round:
            logging.info("=== IMS Prune + Finetune Aggregation Started ===")
        
        # 1. Standard Aggregation (Reuse logic from IMSAggregator)
        num_clients = len(agent_updates_dict)
        if num_clients == 0:
            return torch.zeros_like(flat_global_model)
            
        if initial_update is not None:
             avg_update = initial_update
        else:
             avg_update = torch.stack(list(agent_updates_dict.values())).mean(dim=0)
        
        # Default start round
        
        # Logic for Pruning Phase
        # 1. Before start_round: Standard Aggregation
        if current_round is not None and current_round < start_round:
            return avg_update
        
        if self.has_pruned or (current_round is not None and current_round > start_round):
            logging.info(f"IMS Prune: Round {current_round} > {start_round}. Skipping IMS prune + finetune.")
            return avg_update

        logging.info(f"IMS Prune: Round {current_round} (Start: {start_round}). Executing one-shot IMS prune + finetune.")
        
        # 2. Prepare Candidate Model (Apply average update)
        candidate_model = copy.deepcopy(global_model)
        cur_params = parameters_to_vector(candidate_model.parameters())
        server_lr = self.args.server_lr
        candidate_params = (cur_params + server_lr * avg_update).detach()
        vector_to_parameters(candidate_params, candidate_model.parameters())
        candidate_model.to(self.device)
        candidate_model.eval()
        for param in candidate_model.parameters():
            param.requires_grad = False
        
        if auxiliary_data_loader is None:
            logging.warning("IMS Prune: No auxiliary data, skipping defense.")
            return avg_update

        # Identify prunable layers (Once)
        if self.saved_prunable_layers is None:
            self.saved_prunable_layers = self._get_prunable_layers(candidate_model)
        prunable_layers = self.saved_prunable_layers
        
        # 3. Run IMS Pipeline (Mask Search)
        if self.saved_mask_A is None or self.saved_mask_S is None:
            logging.info("IMS Prune: Calculating Pruning Mask (One-shot)...")
            
            # Step A: Mask Initialization
            A_init, S_init = self.mask_initialization(auxiliary_data_loader, candidate_model, prunable_layers)
            
            # Step B: Inner Subproblem (Perturbation)
            delta_dict = self.inner_subproblem(auxiliary_data_loader, candidate_model, A_init, S_init, prunable_layers)
            
            # Step C: Outer Subproblem (Mask Optimization)
            A_final, S_final = self.outer_subproblem(auxiliary_data_loader, candidate_model, delta_dict, A_init, S_init, prunable_layers)
            
            # Save the mask
            self.saved_mask_A = A_final
            self.saved_mask_S = S_final
        else:
            logging.info("IMS Prune: Using SAVED Pruning Mask...")
            A_final = self.saved_mask_A
            S_final = self.saved_mask_S
        
        # 4. Apply Defense (Structured Pruning)
        # This returns a model where weights are multiplied by the mask
        logging.info("---------Test Before IMS Prune (Candidate Model) ------------")
        criterion = nn.CrossEntropyLoss()
        
        if val_loader is not None:
            clean_acc = utils.get_loss_n_accuracy(candidate_model, criterion, val_loader, self.args, 0)
            logging.info("Pre-Prune Clean ACC:            %.4f" % clean_acc)
        
        if poisoned_val_loader is not None:
            asr = utils.get_loss_n_accuracy(candidate_model, criterion, poisoned_val_loader, self.args, 0, num_classes=self.args.num_target)
            logging.info("Pre-Prune Attack Success Ratio: %.4f" % asr)

        if poisoned_val_only_x_loader is not None:
            ba = utils.get_loss_n_accuracy(candidate_model, criterion, poisoned_val_only_x_loader, self.args, 0, num_classes=self.args.num_target)
            logging.info("Pre-Prune Backdoor ACC:         %.4f" % ba)

        pruned_model = self.deploy_defense(candidate_model, A_final, S_final, prunable_layers)
        
        logging.info("---------Test After IMS Prune (Before Finetune) ------------")
        
        criterion = nn.CrossEntropyLoss()
        
        if val_loader is not None:
            clean_acc = utils.get_loss_n_accuracy(pruned_model, criterion, val_loader, self.args, 0)
            logging.info("Post-Prune Clean ACC:            %.4f" % clean_acc)
        
        if poisoned_val_loader is not None:
            asr = utils.get_loss_n_accuracy(pruned_model, criterion, poisoned_val_loader, self.args, 0, num_classes=self.args.num_target)
            logging.info("Post-Prune Attack Success Ratio: %.4f" % asr)

        if poisoned_val_only_x_loader is not None:
            ba = utils.get_loss_n_accuracy(pruned_model, criterion, poisoned_val_only_x_loader, self.args, 0, num_classes=self.args.num_target)
            logging.info("Post-Prune Backdoor ACC:         %.4f" % ba)
        
        # 5. Lightweight Fine-tuning with Interaction Constraint
        # Use separate loader if provided, else fallback to the one used for pruning
        ft_loader = auxiliary_data_loader_finetune if auxiliary_data_loader_finetune is not None else auxiliary_data_loader
        self.fine_tune_model(pruned_model, ft_loader, A_final, S_final, prunable_layers, val_loader=val_loader, poisoned_val_loader=poisoned_val_loader, poisoned_val_only_x_loader=poisoned_val_only_x_loader)
        
        # Test After Finetune
        # logging.info("---------Test After IMS Prune + Finetune ------------")
        # ...
        
        # 6. Return Update (Calculate update based on final model)
        pruned_params = parameters_to_vector(pruned_model.parameters())
        old_params = parameters_to_vector(global_model.parameters())
        effective_update = (pruned_params - old_params) / server_lr
        
        self.has_pruned = True
        return effective_update

    def fine_tune_model(self, model, loader, A_final, S_final, prunable_layers, val_loader=None, poisoned_val_loader=None, poisoned_val_only_x_loader=None):
        logging.info("IMS Prune: Starting lightweight fine-tuning...")
        
        # Hyperparameters for fine-tuning (reuse NoT config or default)
        epochs = getattr(self.args, 'not_finetune_local_ep', 2)
        lr = getattr(self.args, 'not_finetune_lr', 0.001)
        
        # Enable gradients for fine-tuning
        # Use eval mode to freeze BatchNorm stats and disable Dropout, as we have limited data
        model.eval()
        for param in model.parameters():
            param.requires_grad = True
            
        # Compute masks for constraint (Interaction Constraint)
        # We use the computed masks to zero out gradients of pruned components
        a_prime_list, _ = self.compute_mask_and_inverse(A_final, S_final, self.k)
        masks = [m.detach() for m in a_prime_list]
        
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        masked_modules = dict(model.named_modules())
        
        for ep in range(epochs):
            total_loss = 0
            idx = 0
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output, y)
                loss.backward()
                
                # === Interaction Constraint ===
                # Forbid activation of pruned backdoor-related components.
                # We enforce this by multiplying the gradient by the mask.
                # If mask is close to 0 (pruned), gradient becomes 0, keeping the weight at 0.
                with torch.no_grad():
                    for i, layer_info in enumerate(prunable_layers):
                        name = layer_info['name']
                        module = masked_modules[name]
                        mask = masks[i]
                        reshaped_mask = mask.view(layer_info['shape'])
                        
                        if module.weight.grad is not None:
                             module.weight.grad.data.mul_(reshaped_mask)
                
                optimizer.step()
                total_loss += loss.item()
                idx += 1
            
            logging.info(f"IMS Prune FT Epoch {ep+1}/{epochs}, Loss: {total_loss/idx:.4f}")
            
            # Test after each fine-tuning epoch if loaders are provided
            if val_loader is not None and poisoned_val_loader is not None:
                logging.info(f"--------- Test After FT Epoch {ep+1} ------------")
                clean_acc = utils.get_loss_n_accuracy(model, criterion, val_loader, self.args, 0)
                asr = utils.get_loss_n_accuracy(model, criterion, poisoned_val_loader, self.args, 0, num_classes=self.args.num_target)
                logging.info(f"FT Epoch {ep+1} Clean ACC: {clean_acc:.4f}")
                logging.info(f"FT Epoch {ep+1} Attack Success Ratio: {asr:.4f}")
                
                if poisoned_val_only_x_loader is not None:
                    ba = utils.get_loss_n_accuracy(model, criterion, poisoned_val_only_x_loader, self.args, 0, num_classes=self.args.num_target)
                    logging.info(f"FT Epoch {ep+1} Backdoor ACC: {ba:.4f}")

def agg_ims_prune_finetune(agent_updates_dict, flat_global_model, global_model, args, auxiliary_data_loader, current_round=None, initial_update=None, auxiliary_data_loader_finetune=None):
    aggregator = IMSPruneAggregator(args, args.device)
    return aggregator.aggregate(agent_updates_dict, flat_global_model, global_model, auxiliary_data_loader, initial_update, current_round, auxiliary_data_loader_finetune)
