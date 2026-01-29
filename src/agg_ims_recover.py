# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import logging
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from typing import List, Tuple, Set

class IMSRecoverAggregator:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        
        # Hyperparameters (from ims_idea.txt or args)
        self.lr = getattr(args, 'ims_lr', 1e-2)  # Learning rate for optimization
        self.r1 = getattr(args, 'ims_r1', 20)    # Mask initialization rounds
        self.r2 = getattr(args, 'ims_r2', 15)    # Outer loop rounds
        self.r3 = getattr(args, 'ims_r3', 5)     # Inner loop rounds
        self.k = getattr(args, 'ims_k', 20)      # Scaling factor
        self.lambda_init = getattr(args, 'ims_lambda_init', 0.0)
        self.lambda_final = getattr(args, 'ims_lambda_final', 0.5)
        self.epsilon = getattr(args, 'ims_epsilon', 1.0) # Perturbation constraint
        self.margin = getattr(args, 'ims_margin', 0.5)
        
        # New hyperparameters for Mask Recovery
        self.recovery_threshold = getattr(args, 'ims_recovery_threshold', 0.1) # Threshold for recovery
        self.recovery_momentum = getattr(args, 'ims_recovery_momentum', 0.9) # Momentum for gradient tracking
        self.recovery_rate = getattr(args, 'ims_recovery_rate', 0.01) # How much to recover per step

    def aggregate(self, agent_updates_dict, flat_global_model, global_model, auxiliary_data_loader, initial_update=None, current_round=None):
        """
        IMS Aggregation Logic with Mask Recovery Mechanism
        """
        logging.info("=== IMS (With Recovery) Aggregation Started ===")
        
        num_clients = len(agent_updates_dict)
        if num_clients == 0:
            return torch.zeros_like(flat_global_model)
            
        # 1. Standard Aggregation / Initial Update
        if initial_update is not None:
             avg_update = initial_update
             logging.info("IMS-Recover: Using provided initial_update.")
        else:
             avg_update = torch.stack(list(agent_updates_dict.values())).mean(dim=0)
             logging.info("IMS-Recover: Using standard FedAvg as initial update.")
        
        # Default start round to 100 if not specified
        start_round = getattr(self.args, 'ims_start_round', 100)
        
        # Standard FedAvg (or initial_update) for the first 'start_round' rounds
        if current_round is not None and current_round < start_round:
            # logging.info(f"IMS-Recover: Round {current_round} < {start_round}, performing standard update.")
            return avg_update

        logging.info(f"IMS-Recover: Round {current_round} >= {start_round}, executing IMS defense.")
        
        # Create candidate model
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
            logging.warning("IMS-Recover: No auxiliary data loader! Skipping defense.")
            return avg_update

        # 2. IMS Defense Process
        prunable_layers = self._get_prunable_layers(candidate_model)
        logging.info(f"IMS-Recover: Found {len(prunable_layers)} prunable layers.")
        
        # Step 2: Mask Initialization
        logging.info("IMS-Recover: Step 2 - Mask Initialization")
        A_init, S_init = self.mask_initialization(auxiliary_data_loader, candidate_model, prunable_layers)
        
        # Step 3: Inner Subproblem (Perturbation Synthesis)
        logging.info("IMS-Recover: Step 3 - Inner Subproblem")
        delta_dict = self.inner_subproblem(auxiliary_data_loader, candidate_model, A_init, S_init, prunable_layers)
        
        # Step 4: Outer Subproblem (Mask Optimization with Recovery)
        logging.info("IMS-Recover: Step 4 - Outer Subproblem (Mask Optimization + Recovery)")
        A_final, S_final = self.outer_subproblem_with_recovery(auxiliary_data_loader, candidate_model, delta_dict, A_init, S_init, prunable_layers)
        
        # Step 5: Deploy Defense
        logging.info("IMS-Recover: Step 5 - Deploy Defense")
        defended_model = self.deploy_defense(candidate_model, A_final, S_final, prunable_layers)
        
        defended_params = parameters_to_vector(defended_model.parameters())
        old_params = parameters_to_vector(global_model.parameters())
        
        effective_update = (defended_params - old_params) / server_lr
        
        return effective_update

    def _get_prunable_layers(self, model):
        layers = []
        num_classes = getattr(self.args, 'num_target', None)
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                layers.append({'name': name, 'module': module, 'type': 'conv', 'shape': (module.out_channels, 1, 1, 1), 'size': module.out_channels})
            elif isinstance(module, nn.Linear):
                if num_classes is not None and module.out_features == num_classes:
                    continue
                layers.append({'name': name, 'module': module, 'type': 'linear', 'shape': (module.out_features, 1), 'size': module.out_features})
        return layers

    def compute_mask_and_inverse(self, A_list, S_list, k):
        a_prime_list = []
        a_bar_prime_list = []
        if S_list is None:
            S_list = [torch.zeros_like(a) for a in A_list]
        for a, s in zip(A_list, S_list):
            term_a = torch.sigmoid(k * (a - 0.5))
            a_prime = term_a * s
            term_a_bar = torch.sigmoid(k * ((1 - a) - 0.5))
            a_bar_prime = term_a_bar * (1 - s)
            a_prime_list.append(a_prime)
            a_bar_prime_list.append(a_bar_prime)
        return a_prime_list, a_bar_prime_list

    def apply_mask(self, model, mask_list, prunable_layers):
        masked_model = copy.deepcopy(model)
        masked_modules = dict(masked_model.named_modules())
        for idx, layer_info in enumerate(prunable_layers):
            name = layer_info['name']
            mask = mask_list[idx]
            mask_shape = layer_info['shape']
            module = masked_modules[name]
            reshaped_mask = mask.view(mask_shape)
            original_weight = module.weight
            del module.weight
            module.weight = original_weight * reshaped_mask
        return masked_model

    def _kl_div(self, p, q):
        eps = 1e-8
        return torch.mean(torch.sum(p * (torch.log(p + eps) - torch.log(q + eps)), dim=1))

    def compute_agree_loss(self, q_hat, q):
        return self._kl_div(q, q_hat)

    def compute_disagree_loss(self, q_hat, q):
        kl = self._kl_div(q, q_hat)
        return torch.clamp(self.margin - kl, min=0.0)

    def mask_initialization(self, loader, model, prunable_layers):
        A_init = []
        S_init = []
        for layer in prunable_layers:
            size = layer['size']
            a = torch.rand(size, device=self.device, requires_grad=True)
            s = torch.rand(size, device=self.device, requires_grad=True)
            A_init.append(a)
            S_init.append(s)
        
        optimizer = optim.AdamW(A_init + S_init, lr=self.lr, weight_decay=1e-4)
        
        for epoch in range(self.r1):
            for x, _ in loader:
                x = x.to(self.device)
                a_prime, a_bar_prime = self.compute_mask_and_inverse(A_init, S_init, self.k)
                masked_model = self.apply_mask(model, a_prime, prunable_layers)
                inverse_masked_model = self.apply_mask(model, a_bar_prime, prunable_layers)
                
                with torch.no_grad():
                    p = torch.softmax(model(x), dim=1)
                
                p_A_prime = torch.softmax(masked_model(x), dim=1)
                p_A_bar_prime = torch.softmax(inverse_masked_model(x), dim=1)
                
                agree_loss = self.compute_agree_loss(p_A_prime, p)
                disagree_loss = self.compute_disagree_loss(p_A_bar_prime, p)
                
                reg_loss = 0
                total_params = 0
                for s in S_init:
                    reg_loss += torch.sum(torch.abs(s))
                    total_params += s.numel()
                reg_loss = self.lambda_init * (reg_loss / total_params)
                
                batch_loss = agree_loss + disagree_loss + reg_loss
                
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                
                with torch.no_grad():
                    for a in A_init: a.clamp_(0, 1)
                    for s in S_init: s.clamp_(0, 1)
                    
        return [a.detach() for a in A_init], [s.detach() for s in S_init]

    def inner_subproblem(self, loader, model, A_init, S_init, prunable_layers):
        delta_list = []
        temp_model = copy.deepcopy(model)
        for param in temp_model.parameters(): param.requires_grad = False
        _, a_bar_prime_init = self.compute_mask_and_inverse(A_init, S_init, self.k)
        a_bar_prime_init = [m.detach() for m in a_bar_prime_init]
        inverse_masked_model = self.apply_mask(temp_model, a_bar_prime_init, prunable_layers)
        inverse_masked_model.eval()
        
        for x, _ in loader:
            x = x.to(self.device)
            delta = torch.zeros_like(x, requires_grad=True)
            delta_optimizer = optim.AdamW([delta], lr=self.lr)
            
            for _ in range(self.r3):
                x_hat = torch.clamp(x + delta, 0, 1)
                with torch.no_grad():
                    p = torch.softmax(model(x), dim=1)
                p_hat = torch.softmax(model(x_hat), dim=1)
                p_hat_A_bar = torch.softmax(inverse_masked_model(x_hat), dim=1)
                
                disagree_loss = self.compute_disagree_loss(p_hat, p)
                agree_loss = self.compute_agree_loss(p_hat, p_hat_A_bar)
                delta_loss = disagree_loss + agree_loss
                
                delta_optimizer.zero_grad()
                delta_loss.backward()
                delta_optimizer.step()
                with torch.no_grad():
                    delta.clamp_(-self.epsilon, self.epsilon)
            
            delta_list.append(delta.detach().cpu())
        return delta_list

    def outer_subproblem_with_recovery(self, loader, model, delta_list, A_init, S_init, prunable_layers):
        """
        Modified Outer Subproblem with Mask Recovery (Regrowth)
        """
        A_final = [a.clone().detach().to(self.device).requires_grad_(True) for a in A_init]
        S_final = [s.clone().detach().to(self.device).requires_grad_(True) for s in S_init]
        
        optimizer = optim.AdamW(A_final + S_final, lr=self.lr, weight_decay=1e-4)
        
        # Initialize Gradient Momentum storage for Recovery
        grad_momentum = [torch.zeros_like(a) for a in A_final]
        
        lambda_step = (self.lambda_final - self.lambda_init) / self.r2
        current_lambda = self.lambda_init
        
        for epoch in range(self.r2):
            total_loss = 0.0
            batch_idx = 0
            
            for x, _ in loader:
                x = x.to(self.device)
                delta = delta_list[batch_idx].to(self.device)
                x_hat = torch.clamp(x + delta, 0, 1)
                
                a_prime, _ = self.compute_mask_and_inverse(A_final, S_final, self.k)
                masked_model = self.apply_mask(model, a_prime, prunable_layers)
                
                with torch.no_grad():
                    p = torch.softmax(model(x), dim=1)
                    p_hat = torch.softmax(model(x_hat), dim=1)
                
                p_A_prime = torch.softmax(masked_model(x), dim=1)
                p_hat_A_prime = torch.softmax(masked_model(x_hat), dim=1)
                
                clean_agree_loss = self.compute_agree_loss(p_A_prime, p)
                backdoor_recover_loss = self.compute_agree_loss(p_hat_A_prime, p)
                backdoor_valid_loss = self.compute_disagree_loss(p_hat, p)
                
                reg_loss = 0
                total_params = 0
                for s in S_final:
                    reg_loss += torch.sum(torch.abs(s))
                    total_params += s.numel()
                reg_loss = current_lambda * (reg_loss / total_params)
                
                batch_loss = clean_agree_loss + backdoor_recover_loss + backdoor_valid_loss + reg_loss
                
                optimizer.zero_grad()
                batch_loss.backward()
                
                # --- Mask Recovery Logic Start ---
                # Before step, check gradients for pruned weights
                with torch.no_grad():
                    for i, a in enumerate(A_final):
                        if a.grad is None: continue
                        
                        # Update momentum: m = beta * m + (1-beta) * grad
                        # Note: We want gradients that suggest INCREASE in mask value (negative gradient for Loss)
                        # But Adam minimizes loss, so we look at negative grad. 
                        # Actually, if we want to increase 'a', we look for negative grad (descent direction).
                        # Let's track magnitude of gradients.
                        
                        grad_momentum[i] = self.recovery_momentum * grad_momentum[i] + (1 - self.recovery_momentum) * a.grad
                        
                        # Identification: If mask value is low (pruned) BUT gradient suggests it should increase greatly
                        # (i.e., strong negative gradient), we boost it.
                        
                        # Define "Pruned" as a < 0.1 (soft threshold)
                        is_pruned = a < 0.1
                        
                        # Strong negative gradient means "increasing 'a' will reduce loss significantly"
                        # We use a threshold for the gradient magnitude
                        # For simplicity, we use the top-k gradients or a fixed threshold?
                        # Use Quantile-based threshold for robustness against Non-IID noise
                        abs_grad = torch.abs(grad_momentum[i])
                        # Calculate top 10% threshold (90th percentile) to avoid outliers affecting mean
                        if abs_grad.numel() > 0:
                            top_k_threshold = torch.quantile(abs_grad, 0.90)
                        else:
                            top_k_threshold = 0.0
                            
                        # Strong negative gradient means we should increase mask value to reduce loss
                        # We only recover if the gradient signal is very strong (top 10%)
                        strong_signal = grad_momentum[i] < -top_k_threshold 
                        
                        recover_mask = is_pruned & strong_signal
                        num_recovered = recover_mask.sum().item()
                        
                        if num_recovered > 0:
                            # Boost the mask value carefully
                            # Reset to a low but active state (0.2) to allow gradients to flow again
                            a[recover_mask] = torch.max(a[recover_mask], torch.tensor(0.2, device=self.device))
                            if batch_idx % 10 == 0 and i == 0: # Log occasionally for first layer
                                logging.debug(f"Layer {i}: Recovered {num_recovered} neurons. Threshold: {top_k_threshold:.4f}")
                            
                # --- Mask Recovery Logic End ---
                
                optimizer.step()
                
                with torch.no_grad():
                    for a in A_final: a.clamp_(0, 1)
                    for s in S_final: s.clamp_(0, 1)
                
                total_loss += batch_loss.item()
                batch_idx += 1
            
            current_lambda = min(current_lambda + lambda_step, self.lambda_final)
            if (epoch + 1) % 5 == 0:
                logging.info(f"IMS-Recover Outer Epoch {epoch+1}/{self.r2}, Loss: {total_loss/batch_idx:.4f}")
                
        return [a.detach() for a in A_final], [s.detach() for s in S_final]

    def deploy_defense(self, model, A_final, S_final, prunable_layers):
        a_prime_list, _ = self.compute_mask_and_inverse(A_final, S_final, self.k)
        defended_model = copy.deepcopy(model)
        masked_modules = dict(defended_model.named_modules())
        
        for idx, layer_info in enumerate(prunable_layers):
            name = layer_info['name']
            mask = a_prime_list[idx].detach()
            mask_shape = layer_info['shape']
            module = masked_modules[name]
            reshaped_mask = mask.view(mask_shape)
            module.weight.data = module.weight.data * reshaped_mask
            
        logging.info(f"IMS-Recover: Defense deployed. Pruned {len(prunable_layers)} layers.")
        return defended_model

def agg_ims_recover(agent_updates_dict, flat_global_model, global_model, args, auxiliary_data_loader, current_round=None, initial_update=None):
    aggregator = IMSRecoverAggregator(args, device=args.device if hasattr(args, 'device') else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return aggregator.aggregate(agent_updates_dict, flat_global_model, global_model, auxiliary_data_loader, initial_update=initial_update, current_round=current_round)
