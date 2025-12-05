"""
IMS: Intelligent Mask Selection for Backdoor Defense
基于ims_idea.txt实现的聚合防御方法

流程：
1. 基础聚合（FedAvg）得到全局模型
2. 利用少量干净数据（auxiliary_data）进行IMS防御
   - 掩码初始化
   - 内层子问题（扰动合成）
   - 外层子问题（掩码优化）
3. 应用最终掩码剪枝模型
4. 返回导致防御后模型的更新向量

作者：AI Assistant
日期：2024
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import logging
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from typing import List, Tuple, Set

class IMSAggregator:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        
        # Hyperparameters (from ims_idea.txt or args)
        self.lr = getattr(args, 'ims_lr', 1e-2)  # Learning rate for optimization
        self.r1 = getattr(args, 'ims_r1', 20)    # Mask initialization rounds (reduced from 50 for speed)
        self.r2 = getattr(args, 'ims_r2', 15)    # Outer loop rounds (reduced from 30)
        self.r3 = getattr(args, 'ims_r3', 5)     # Inner loop rounds (reduced from 10)
        self.k = getattr(args, 'ims_k', 20)      # Scaling factor
        self.lambda_init = getattr(args, 'ims_lambda_init', 0.0)
        self.lambda_final = getattr(args, 'ims_lambda_final', 10.0)
        self.epsilon = getattr(args, 'ims_epsilon', 1.0) # Perturbation constraint
        self.margin = getattr(args, 'ims_margin', 0.5)
        
    def aggregate(self, agent_updates_dict, flat_global_model, global_model, auxiliary_data_loader):
        """
        IMS Aggregation Logic
        """
        logging.info("=== IMS Aggregation Started ===")
        
        # 1. Standard Aggregation (FedAvg) to get the starting point
        # Note: We assume agent_updates_dict contains updates (delta), not full models.
        # global_model currently holds the OLD parameters.
        # We want to apply the updates to get the candidate global model.
        
        num_clients = len(agent_updates_dict)
        if num_clients == 0:
            return torch.zeros_like(flat_global_model)
            
        # Average updates
        avg_update = torch.stack(list(agent_updates_dict.values())).mean(dim=0)
        
        # Create a temporary model with the aggregated parameters
        # new_params = old_params + server_lr * avg_update
        # But typically aggregation returns the update.
        # Here we want to operate on the MODEL that would result from this update.
        # To avoid modifying global_model in place yet, we clone it.
        
        candidate_model = copy.deepcopy(global_model)
        cur_params = parameters_to_vector(candidate_model.parameters())
        # Assume server_lr is applied outside or we need to apply it here?
        # In aggregation.py: new_global_params = (cur_global_params + lr_vector*aggregated_updates)
        # So we should emulate this.
        
        server_lr = self.args.server_lr
        candidate_params = (cur_params + server_lr * avg_update).detach()
        vector_to_parameters(candidate_params, candidate_model.parameters())
        candidate_model.to(self.device)
        candidate_model.eval() # Initially eval
        # Disable gradients for candidate_model parameters as we only optimize masks/deltas
        for param in candidate_model.parameters():
            param.requires_grad = False
        
        if auxiliary_data_loader is None:
            logging.warning("IMS: No auxiliary data loader provided! Skipping defense.")
            return avg_update

        # 2. IMS Defense Process
        
        # Identify prunable layers
        prunable_layers = self._get_prunable_layers(candidate_model)
        logging.info(f"IMS: Found {len(prunable_layers)} prunable layers.")
        
        # Step 2: Mask Initialization
        logging.info("IMS: Step 2 - Mask Initialization")
        A_init, S_init = self.mask_initialization(auxiliary_data_loader, candidate_model, prunable_layers)
        
        # Step 3: Inner Subproblem (Perturbation Synthesis)
        logging.info("IMS: Step 3 - Inner Subproblem (Perturbation Synthesis)")
        delta_dict = self.inner_subproblem(auxiliary_data_loader, candidate_model, A_init, S_init, prunable_layers)
        
        # Step 4: Outer Subproblem (Mask Optimization)
        logging.info("IMS: Step 4 - Outer Subproblem (Mask Optimization)")
        A_final, S_final = self.outer_subproblem(auxiliary_data_loader, candidate_model, delta_dict, A_init, S_init, prunable_layers)
        
        # Step 5: Deploy Defense (Apply Mask)
        logging.info("IMS: Step 5 - Deploy Defense")
        defended_model = self.deploy_defense(candidate_model, A_final, S_final, prunable_layers)
        
        # 3. Calculate the effective update
        # defended_model = old_params + server_lr * effective_update
        # effective_update = (defended_model - old_params) / server_lr
        
        defended_params = parameters_to_vector(defended_model.parameters())
        old_params = parameters_to_vector(global_model.parameters()) # The original one passed in
        
        effective_update = (defended_params - old_params) / server_lr
        
        return effective_update

    def _get_prunable_layers(self, model):
        """
        Identify layers to prune (Conv2d and Linear).
        Returns list of (name, module, shape_of_mask)
        """
        layers = []
        num_classes = getattr(self.args, 'num_target', None)
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                # Prune output channels
                # Mask shape: (out_channels, 1, 1, 1)
                layers.append({'name': name, 'module': module, 'type': 'conv', 'shape': (module.out_channels, 1, 1, 1), 'size': module.out_channels})
            elif isinstance(module, nn.Linear):
                if num_classes is not None and module.out_features == num_classes:
                    continue
                layers.append({'name': name, 'module': module, 'type': 'linear', 'shape': (module.out_features, 1), 'size': module.out_features})
        return layers

    def compute_mask_and_inverse(self, A_list, S_list, k):
        """
        Compute a_prime and a_bar_prime for each layer.
        """
        a_prime_list = []
        a_bar_prime_list = []
        
        # If S_list is None (e.g. not used), treat as zeros
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
        """
        Apply mask to the model. Returns a NEW model (copy).
        """
        masked_model = copy.deepcopy(model)
        # We need to map mask_list to the modules in masked_model
        # prunable_layers contains names, we can find them in masked_model
        
        masked_modules = dict(masked_model.named_modules())
        
        for idx, layer_info in enumerate(prunable_layers):
            name = layer_info['name']
            mask = mask_list[idx]
            mask_shape = layer_info['shape']
            
            # Find the module in masked_model
            module = masked_modules[name]
            
            # Reshape mask
            reshaped_mask = mask.view(mask_shape)
            
            # Apply mask to weight
            # We delete the parameter and assign the tensor property.
            # This allows autograd to flow through 'module.weight' which is now just a tensor attribute.
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
        # Initialize A and S
        A_init = []
        S_init = []
        for layer in prunable_layers:
            size = layer['size']
            # random uniform [0, 1]
            a = torch.rand(size, device=self.device, requires_grad=True)
            s = torch.rand(size, device=self.device, requires_grad=True)
            A_init.append(a)
            S_init.append(s)
            
        optimizer = optim.AdamW(A_init + S_init, lr=self.lr, weight_decay=1e-4)
        
        for epoch in range(self.r1):
            total_loss = 0.0
            num_batches = 0
            for x, _ in loader:
                x = x.to(self.device)
                
                # Compute masks
                a_prime, a_bar_prime = self.compute_mask_and_inverse(A_init, S_init, self.k)
                
                # Apply masks
                # Note: apply_mask creates a COPY of the model and modifies weights.
                # This is expensive inside a loop.
                # Optimization: Only do forward pass with masks applied?
                # Since we can't easily change forward pass without hooks, we do the copy method.
                # It might be slow.
                
                # To speed up: maybe reuse the same model copy and just update weights?
                # But weights depend on A/S which change.
                
                masked_model = self.apply_mask(model, a_prime, prunable_layers)
                inverse_masked_model = self.apply_mask(model, a_bar_prime, prunable_layers)
                
                # Forward
                # model is fixed, so no grad needed for model outputs
                with torch.no_grad():
                    p = torch.softmax(model(x), dim=1)
                
                p_A_prime = torch.softmax(masked_model(x), dim=1)
                p_A_bar_prime = torch.softmax(inverse_masked_model(x), dim=1)
                
                agree_loss = self.compute_agree_loss(p_A_prime, p)
                disagree_loss = self.compute_disagree_loss(p_A_bar_prime, p)
                
                # Regularization
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
                
                # Clamp
                with torch.no_grad():
                    for a in A_init:
                        a.clamp_(0, 1)
                    for s in S_init:
                        s.clamp_(0, 1)
                
                total_loss += batch_loss.item()
                num_batches += 1
            
            if (epoch + 1) % 5 == 0:
                logging.info(f"IMS Init Epoch {epoch+1}/{self.r1}, Loss: {total_loss/num_batches:.4f}")
                
        return [a.detach() for a in A_init], [s.detach() for s in S_init]

    def inner_subproblem(self, loader, model, A_init, S_init, prunable_layers):
        # Generate perturbation delta for each sample
        # Since loader shuffles, we need to be careful about mapping delta to samples.
        # The pseudocode uses a dictionary delta_dict[sample_idx].
        # We need to ensure we can retrieve the same delta for the same sample in the outer loop.
        # However, standard DataLoaders don't give sample indices easily unless the dataset returns them.
        # For simplicity, and since we just need "virtual backdoor data", maybe we can just generate them on the fly 
        # OR generate a fixed set of deltas for the dataset once.
        
        # The pseudocode iterates over clean_loader in both inner and outer loops.
        # If shuffling is True, the order changes.
        # We need to store deltas. 
        # Option: Iterate loader ONCE to generate all deltas and store them in a list/dict alongside inputs.
        # But memory might be an issue.
        # Wait, the pseudocode says: "FOR each batch... save current batch's optimal delta to dict".
        # Then in outer loop: "FOR each batch... get delta from dict".
        # This implies deterministic order or index tracking.
        # We can turn off shuffle for auxiliary_data_loader during IMS?
        # auxiliary_data_loader in federated.py has shuffle=False (lines 257-263 in federated.py, wait, check).
        # In federated.py: `auxiliary_data_loader = DataLoader(..., shuffle=False, ...)`
        # Yes, shuffle is False! So we can rely on order.
        
        delta_list = [] # Store deltas in order
        
        # Fixed inverse mask model
        # Create a detached copy of the model for inverse mask application
        # This ensures no graph connections remain between inverse_masked_model and the main model
        temp_model = copy.deepcopy(model)
        for param in temp_model.parameters():
            param.requires_grad = False
            
        _, a_bar_prime_init = self.compute_mask_and_inverse(A_init, S_init, self.k)
        # Detach masks to avoid graph retention issues in inner loop
        a_bar_prime_init = [m.detach() for m in a_bar_prime_init]
        inverse_masked_model = self.apply_mask(temp_model, a_bar_prime_init, prunable_layers)
        inverse_masked_model.eval()
        
        for x, y in loader:
            x = x.to(self.device)
            batch_size = x.shape[0]
            
            delta = torch.zeros_like(x, requires_grad=True)
            delta_optimizer = optim.AdamW([delta], lr=self.lr)
            
            for _ in range(self.r3):
                x_hat = torch.clamp(x + delta, 0, 1)
                
                with torch.no_grad():
                    p = torch.softmax(model(x), dim=1)
                
                # Note: inverse_masked_model weights are fixed tensors, but x_hat has grad.
                # So we can backprop through the model to delta.
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
            
            delta_list.append(delta.detach().cpu()) # Save to CPU to save GPU memory
            
        logging.info(f"IMS Inner: Generated perturbations for {len(delta_list)} batches.")
        return delta_list

    def outer_subproblem(self, loader, model, delta_list, A_init, S_init, prunable_layers):
        A_final = [a.clone().detach().to(self.device).requires_grad_(True) for a in A_init]
        S_final = [s.clone().detach().to(self.device).requires_grad_(True) for s in S_init]
        
        optimizer = optim.AdamW(A_final + S_final, lr=self.lr, weight_decay=1e-4)
        
        lambda_step = (self.lambda_final - self.lambda_init) / self.r2
        current_lambda = self.lambda_init
        
        for epoch in range(self.r2):
            total_loss = 0.0
            batch_idx = 0
            
            for x, y in loader:
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
                optimizer.step()
                
                with torch.no_grad():
                    for a in A_final:
                        a.clamp_(0, 1)
                    for s in S_final:
                        s.clamp_(0, 1)
                
                total_loss += batch_loss.item()
                batch_idx += 1
            
            current_lambda = min(current_lambda + lambda_step, self.lambda_final)
            if (epoch + 1) % 5 == 0:
                logging.info(f"IMS Outer Epoch {epoch+1}/{self.r2}, Loss: {total_loss/batch_idx:.4f}, Lambda: {current_lambda:.2f}")
                
        return [a.detach() for a in A_final], [s.detach() for s in S_final]

    def deploy_defense(self, model, A_final, S_final, prunable_layers):
        # Compute the final mask
        a_prime_list, _ = self.compute_mask_and_inverse(A_final, S_final, self.k)
        
        # Apply mask permanently
        # For deployment, we don't need gradients. We want to modify the weights.
        defended_model = copy.deepcopy(model)
        masked_modules = dict(defended_model.named_modules())
        
        for idx, layer_info in enumerate(prunable_layers):
            name = layer_info['name']
            mask = a_prime_list[idx].detach() # Detach!
            mask_shape = layer_info['shape']
            
            module = masked_modules[name]
            reshaped_mask = mask.view(mask_shape)
            
            # Permanent pruning
            module.weight.data = module.weight.data * reshaped_mask
            
        logging.info(f"IMS: Defense deployed. Pruned {len(prunable_layers)} layers.")
        return defended_model

def agg_ims(agent_updates_dict, flat_global_model, global_model, args, auxiliary_data_loader):
    aggregator = IMSAggregator(args, args.device)
    return aggregator.aggregate(agent_updates_dict, flat_global_model, global_model, auxiliary_data_loader)

class IMSOncePipeline:
    def __init__(self, args):
        self.args = args

    def run(self,
            inter_model_updates: torch.Tensor,
            flat_global_model: torch.Tensor,
            global_model: torch.nn.Module,
            auxiliary_data_loader,
            current_round: int = None) -> Tuple[torch.Tensor, torch.nn.Module, Set[int]]:
        final_update = torch.mean(inter_model_updates, dim=0)
        poisoned_model = copy.deepcopy(global_model)
        final_params = flat_global_model + final_update
        vector_to_parameters(final_params, poisoned_model.parameters())
        ims = IMSAggregator(self.args, self.args.device)
        defended_model = self._defend_once(ims, poisoned_model, auxiliary_data_loader)
        defended_params = parameters_to_vector(defended_model.parameters())
        base_params = flat_global_model
        effective_update = defended_params - base_params
        target_clients: Set[int] = set()
        return effective_update, defended_model, target_clients

    def _defend_once(self,
                     ims: IMSAggregator,
                     candidate_model: torch.nn.Module,
                     auxiliary_data_loader) -> torch.nn.Module:
        prunable_layers = ims._get_prunable_layers(candidate_model)
        A_init, S_init = ims.mask_initialization(auxiliary_data_loader, candidate_model, prunable_layers)
        delta_dict = ims.inner_subproblem(auxiliary_data_loader, candidate_model, A_init, S_init, prunable_layers)
        A_final, S_final = ims.outer_subproblem(auxiliary_data_loader, candidate_model, delta_dict, A_init, S_init, prunable_layers)
        defended_model = ims.deploy_defense(candidate_model, A_final, S_final, prunable_layers)
        return defended_model

def agg_ims_once(inter_model_updates: torch.Tensor,
                 flat_global_model: torch.Tensor,
                 global_model: torch.nn.Module,
                 args,
                 auxiliary_data_loader,
                 malicious_id: List[int] = None,
                 current_round: int = None) -> Tuple[torch.Tensor, torch.nn.Module, Set[int]]:
    pipeline = IMSOncePipeline(args)
    return pipeline.run(
        inter_model_updates,
        flat_global_model,
        global_model,
        auxiliary_data_loader,
        current_round,
    )
