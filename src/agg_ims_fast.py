"""
IMS Fast: Optimized Intelligent Mask Selection for Backdoor Defense
Optimizations:
1. MaskContext: Apply masks in-place without deep copying the model.
2. Clean Output Caching: Cache clean model outputs to avoid redundant forward passes.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import copy
import logging
from torch.nn.utils import parameters_to_vector, vector_to_parameters

class MaskContext:
    """
    Context manager to apply masks to a model in-place and restore them on exit.
    This avoids expensive deepcopy operations.
    """
    def __init__(self, model, mask_list, prunable_layers):
        self.model = model
        self.mask_list = mask_list
        self.prunable_layers = prunable_layers
        self.original_weights = {}
        self.masked_modules = dict(model.named_modules())

    def __enter__(self):
        for idx, layer_info in enumerate(self.prunable_layers):
            name = layer_info['name']
            mask = self.mask_list[idx]
            mask_shape = layer_info['shape']
            
            module = self.masked_modules[name]
            reshaped_mask = mask.view(mask_shape)
            
            # Save original weight
            self.original_weights[name] = module.weight
            
            # Apply mask (use original_weight * mask)
            # We temporarily replace the parameter with the masked tensor
            # Note: This makes module.weight a Tensor, not a Parameter, but that's fine for forward pass
            # If we needed gradients W.R.T weights, this might be tricky, but here we fix weights and optimize masks.
            # However, if we optimized weights, we would need to be careful.
            # IMS only optimizes masks (A, S) or deltas, not model weights during these phases.
            del module.weight
            module.weight = self.original_weights[name] * reshaped_mask
            
        return self.model

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original weights
        for name, original_weight in self.original_weights.items():
            module = self.masked_modules[name]
            # If we deleted the parameter, we need to re-assign it
            # But module.weight might be a tensor now.
            if hasattr(module, 'weight'):
                del module.weight
            module.weight = original_weight
        
        self.original_weights.clear()

class IMSAggregator:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        
        # Hyperparameters
        self.lr = getattr(args, 'ims_lr', 1e-2)
        self.r1 = getattr(args, 'ims_r1', 20)
        self.r2 = getattr(args, 'ims_r2', 15)
        self.r3 = getattr(args, 'ims_r3', 5)
        self.k = getattr(args, 'ims_k', 20)
        self.lambda_init = getattr(args, 'ims_lambda_init', 0.0)
        self.lambda_final = getattr(args, 'ims_lambda_final', 10.0)
        self.epsilon = getattr(args, 'ims_epsilon', 1.0)
        self.margin = getattr(args, 'ims_margin', 0.5)
        self.clean_agree_weight = getattr(args, 'ims_clean_agree_weight', 1.0)
        self.backdoor_recover_weight = getattr(args, 'ims_backdoor_recover_weight', 1.0)
        self.poison_entropy_weight = getattr(args, 'ims_poison_entropy_weight', 0.0)
        self.adaptive = bool(getattr(args, 'ims_adaptive', False))
        self.adapt_r1_min_epochs = int(getattr(args, 'ims_adapt_r1_min_epochs', 5))
        self.adapt_r1_tol = float(getattr(args, 'ims_adapt_r1_tol', 1e-4))
        self.adapt_r1_patience = int(getattr(args, 'ims_adapt_r1_patience', 3))
        self.adapt_r1_mask_tol = float(getattr(args, 'ims_adapt_r1_mask_tol', 1e-4))
        self.adapt_r1_mask_patience = int(getattr(args, 'ims_adapt_r1_mask_patience', 2))

        self.adapt_r2_min_epochs = int(getattr(args, 'ims_adapt_r2_min_epochs', 5))
        self.adapt_r2_tol = float(getattr(args, 'ims_adapt_r2_tol', 1e-4))
        self.adapt_r2_patience = int(getattr(args, 'ims_adapt_r2_patience', 3))
        self.adapt_r2_mask_tol = float(getattr(args, 'ims_adapt_r2_mask_tol', 1e-4))
        self.adapt_r2_mask_patience = int(getattr(args, 'ims_adapt_r2_mask_patience', 2))

        self.adapt_r3_min_steps = int(getattr(args, 'ims_adapt_r3_min_steps', 1))
        self.adapt_r3_tol = float(getattr(args, 'ims_adapt_r3_tol', 1e-4))
        self.adapt_r3_patience = int(getattr(args, 'ims_adapt_r3_patience', 2))
        self.adapt_r3_saturation_ratio = float(getattr(args, 'ims_adapt_r3_saturation_ratio', 0.98))
        
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
        
        # Optimization: Cache clean outputs
        clean_outputs_cache = {} 
        
        best_epoch_loss = None
        loss_plateau_count = 0
        prev_a_prime = None
        mask_stable_count = 0

        for epoch in range(self.r1):
            total_loss = 0.0
            num_batches = 0
            
            # Note: Assuming loader is deterministic or we iterate it sequentially
            # If loader shuffles, caching by index is hard.
            # But federated.py sets shuffle=False for auxiliary_data_loader.
            
            for batch_idx, (x, _) in enumerate(loader):
                x = x.to(self.device)
                
                # Get or compute clean output
                if batch_idx not in clean_outputs_cache:
                    with torch.no_grad():
                        clean_outputs_cache[batch_idx] = torch.softmax(model(x), dim=1)
                p = clean_outputs_cache[batch_idx]
                
                a_prime, a_bar_prime = self.compute_mask_and_inverse(A_init, S_init, self.k)
                
                # Use MaskContext instead of deepcopy
                with MaskContext(model, a_prime, prunable_layers):
                    p_A_prime = torch.softmax(model(x), dim=1)
                    
                with MaskContext(model, a_bar_prime, prunable_layers):
                    p_A_bar_prime = torch.softmax(model(x), dim=1)
                
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
                    for a in A_init:
                        a.clamp_(0, 1)
                    for s in S_init:
                        s.clamp_(0, 1)
                
                total_loss += batch_loss.item()
                num_batches += 1
            
            epoch_loss = total_loss / max(num_batches, 1)
            if (epoch + 1) % 5 == 0:
                logging.info(f"IMS Fast Init Epoch {epoch+1}/{self.r1}, Loss: {total_loss/num_batches:.4f}")

            if self.adaptive:
                if best_epoch_loss is None:
                    best_epoch_loss = epoch_loss
                else:
                    improvement = best_epoch_loss - epoch_loss
                    if improvement > self.adapt_r1_tol:
                        best_epoch_loss = epoch_loss
                        loss_plateau_count = 0
                    else:
                        loss_plateau_count += 1

                with torch.no_grad():
                    a_prime, _ = self.compute_mask_and_inverse(A_init, S_init, self.k)
                    if prev_a_prime is not None:
                        total_diff = 0.0
                        total_elems = 0
                        for cur, prev in zip(a_prime, prev_a_prime):
                            total_diff += torch.mean(torch.abs(cur.detach() - prev.detach())).item()
                            total_elems += 1
                        avg_diff = total_diff / max(total_elems, 1)
                        if avg_diff < self.adapt_r1_mask_tol:
                            mask_stable_count += 1
                        else:
                            mask_stable_count = 0
                    prev_a_prime = [m.detach().clone() for m in a_prime]

                if (epoch + 1) >= self.adapt_r1_min_epochs:
                    if loss_plateau_count >= self.adapt_r1_patience or mask_stable_count >= self.adapt_r1_mask_patience:
                        logging.info(
                            f"IMS Adaptive: early-stopped r1 at epoch {epoch+1}/{self.r1} "
                            f"(loss_plateau={loss_plateau_count}, mask_stable={mask_stable_count})"
                        )
                        break
                
        return [a.detach() for a in A_init], [s.detach() for s in S_init]

    def inner_subproblem(self, loader, model, A_init, S_init, prunable_layers):
        delta_list = []
        
        # Optimization: Pre-compute inverse masked model for this phase
        # Since A and S are fixed here, we can compute the inverse masked model ONCE
        # and reuse it, instead of applying mask every time or inside loop.
        # However, to be safe with gradients if we were optimizing A/S, we'd need care.
        # But here we optimize delta.
        
        temp_model = copy.deepcopy(model)
        for param in temp_model.parameters():
            param.requires_grad = False
            
        _, a_bar_prime_init = self.compute_mask_and_inverse(A_init, S_init, self.k)
        a_bar_prime_init = [m.detach() for m in a_bar_prime_init]
        
        # We can use apply_mask (copy) here once, or use MaskContext but we need it to persist
        # across the inner loop. Since we iterate batches, maybe just copy once is better/easier
        # for the inverse model as it is static during inner loop.
        
        # Helper to apply mask permanently to a copy
        def create_masked_copy(base_model, masks):
            m_model = copy.deepcopy(base_model)
            m_modules = dict(m_model.named_modules())
            for idx, layer_info in enumerate(prunable_layers):
                name = layer_info['name']
                mask = masks[idx]
                mask_shape = layer_info['shape']
                module = m_modules[name]
                reshaped_mask = mask.view(mask_shape)
                module.weight.data = module.weight.data * reshaped_mask
            return m_model

        inverse_masked_model = create_masked_copy(temp_model, a_bar_prime_init)
        inverse_masked_model.eval()
        inverse_masked_model.to(self.device)
        
        for x, y in loader:
            x = x.to(self.device)
            
            delta = torch.zeros_like(x, requires_grad=True)
            delta_optimizer = optim.AdamW([delta], lr=self.lr)
            
            # Cache clean output for this batch
            with torch.no_grad():
                p = torch.softmax(model(x), dim=1)
            
            best_delta_loss = None
            plateau_count = 0
            for step_idx in range(self.r3):
                x_hat = torch.clamp(x + delta, 0, 1)
                
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

                if self.adaptive:
                    cur_loss = float(delta_loss.detach().item())
                    if best_delta_loss is None:
                        best_delta_loss = cur_loss
                    else:
                        improvement = best_delta_loss - cur_loss
                        if improvement > self.adapt_r3_tol:
                            best_delta_loss = cur_loss
                            plateau_count = 0
                        else:
                            plateau_count += 1

                    with torch.no_grad():
                        saturation = (delta.detach().abs() >= (self.epsilon * 0.999)).float().mean().item()
                    if (step_idx + 1) >= self.adapt_r3_min_steps:
                        if plateau_count >= self.adapt_r3_patience or saturation >= self.adapt_r3_saturation_ratio:
                            break
            
            delta_list.append(delta.detach().cpu())
            
        logging.info(f"IMS Fast Inner: Generated perturbations for {len(delta_list)} batches.")
        return delta_list

    def outer_subproblem(self, loader, model, delta_list, A_init, S_init, prunable_layers):
        A_final = [a.clone().detach().to(self.device).requires_grad_(True) for a in A_init]
        S_final = [s.clone().detach().to(self.device).requires_grad_(True) for s in S_init]
        
        optimizer = optim.AdamW(A_final + S_final, lr=self.lr, weight_decay=1e-4)
        
        lambda_step = (self.lambda_final - self.lambda_init) / self.r2
        current_lambda = self.lambda_init
        
        clean_outputs_cache = {}
        best_epoch_loss = None
        loss_plateau_count = 0
        prev_a_prime = None
        mask_stable_count = 0
        
        for epoch in range(self.r2):
            total_loss = 0.0
            batch_idx = 0
            
            for x, y in loader:
                x = x.to(self.device)
                delta = delta_list[batch_idx].to(self.device)
                x_hat = torch.clamp(x + delta, 0, 1)
                
                if batch_idx not in clean_outputs_cache:
                    with torch.no_grad():
                        clean_outputs_cache[batch_idx] = torch.softmax(model(x), dim=1)
                p = clean_outputs_cache[batch_idx]
                
                # Compute p_hat (poisoned input, clean model) - this changes if model changes?
                # Model is fixed in IMS (candidate_model), only masks change.
                # So p_hat is also constant for a given batch + delta!
                # Wait, delta is fixed from inner loop. Model is fixed.
                # So p_hat can ALSO be cached!
                
                # We can cache p_hat too.
                p_hat_key = f"p_hat_{batch_idx}"
                if p_hat_key not in clean_outputs_cache:
                    with torch.no_grad():
                        clean_outputs_cache[p_hat_key] = torch.softmax(model(x_hat), dim=1)
                p_hat = clean_outputs_cache[p_hat_key]
                
                a_prime, _ = self.compute_mask_and_inverse(A_final, S_final, self.k)
                
                # Use MaskContext
                with MaskContext(model, a_prime, prunable_layers):
                    p_A_prime = torch.softmax(model(x), dim=1)
                    p_hat_A_prime = torch.softmax(model(x_hat), dim=1)
                
                clean_agree_loss = self.compute_agree_loss(p_A_prime, p)
                backdoor_recover_loss = self.compute_agree_loss(p_hat_A_prime, p)
                backdoor_valid_loss = self.compute_disagree_loss(p_hat, p) # This is constant w.r.t A/S?
                # Wait, backdoor_valid_loss depends on p_hat and p. Both are fixed here.
                # So this term is constant for the optimization of A and S!
                # It doesn't affect gradients of A/S. We can skip computing it for optimization, 
                # but maybe keep it for loss reporting.
                entropy = -torch.mean(torch.sum(p_hat_A_prime * torch.log(p_hat_A_prime + 1e-8), dim=1))
                poison_entropy_loss = -entropy
                
                reg_loss = 0
                total_params = 0
                for s in S_final:
                    reg_loss += torch.sum(torch.abs(s))
                    total_params += s.numel()
                reg_loss = current_lambda * (reg_loss / total_params)
                
                batch_loss = (
                    self.clean_agree_weight * clean_agree_loss
                    + self.backdoor_recover_weight * backdoor_recover_loss
                    + self.poison_entropy_weight * poison_entropy_loss
                    + backdoor_valid_loss
                    + reg_loss
                )
                
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
                logging.info(f"IMS Fast Outer Epoch {epoch+1}/{self.r2}, Loss: {total_loss/batch_idx:.4f}, Lambda: {current_lambda:.2f}")

            if self.adaptive:
                epoch_loss = total_loss / max(batch_idx, 1)
                if best_epoch_loss is None:
                    best_epoch_loss = epoch_loss
                else:
                    improvement = best_epoch_loss - epoch_loss
                    if improvement > self.adapt_r2_tol:
                        best_epoch_loss = epoch_loss
                        loss_plateau_count = 0
                    else:
                        loss_plateau_count += 1

                with torch.no_grad():
                    a_prime, _ = self.compute_mask_and_inverse(A_final, S_final, self.k)
                    if prev_a_prime is not None:
                        total_diff = 0.0
                        total_elems = 0
                        for cur, prev in zip(a_prime, prev_a_prime):
                            total_diff += torch.mean(torch.abs(cur.detach() - prev.detach())).item()
                            total_elems += 1
                        avg_diff = total_diff / max(total_elems, 1)
                        if avg_diff < self.adapt_r2_mask_tol:
                            mask_stable_count += 1
                        else:
                            mask_stable_count = 0
                    prev_a_prime = [m.detach().clone() for m in a_prime]

                if (epoch + 1) >= self.adapt_r2_min_epochs:
                    if loss_plateau_count >= self.adapt_r2_patience or mask_stable_count >= self.adapt_r2_mask_patience:
                        logging.info(
                            f"IMS Adaptive: early-stopped r2 at epoch {epoch+1}/{self.r2} "
                            f"(loss_plateau={loss_plateau_count}, mask_stable={mask_stable_count})"
                        )
                        break
                
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
            
        logging.info(f"IMS Fast: Defense deployed. Pruned {len(prunable_layers)} layers.")
        return defended_model
    
    def aggregate(self, agent_updates_dict, flat_global_model, global_model, auxiliary_data_loader, initial_update=None, current_round=None):
        logging.info("=== IMS Fast Aggregation Started ===")
        
        num_clients = len(agent_updates_dict)
        if num_clients == 0:
            return torch.zeros_like(flat_global_model)
            
        if initial_update is not None:
             avg_update = initial_update
        else:
             avg_update = torch.stack(list(agent_updates_dict.values())).mean(dim=0)
        
        # Default start round to 100 if not specified
        start_round = getattr(self.args, 'ims_start_round', 100)
        
        # Standard FedAvg (or initial_update) for the first 'start_round' rounds
        if current_round is not None and current_round < start_round:
            # logging.info(f"IMS Fast: Round {current_round} < {start_round}, performing standard FedAvg.")
            return avg_update

        logging.info(f"IMS Fast: Round {current_round} >= {start_round}, executing IMS defense.")
        
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
            return avg_update

        prunable_layers = self._get_prunable_layers(candidate_model)
        
        # Fast Pipeline
        A_init, S_init = self.mask_initialization(auxiliary_data_loader, candidate_model, prunable_layers)
        delta_dict = self.inner_subproblem(auxiliary_data_loader, candidate_model, A_init, S_init, prunable_layers)
        A_final, S_final = self.outer_subproblem(auxiliary_data_loader, candidate_model, delta_dict, A_init, S_init, prunable_layers)
        defended_model = self.deploy_defense(candidate_model, A_final, S_final, prunable_layers)
        
        defended_params = parameters_to_vector(defended_model.parameters())
        old_params = parameters_to_vector(global_model.parameters())
        
        effective_update = (defended_params - old_params) / server_lr
        
        return effective_update

def agg_ims_fast(agent_updates_dict, flat_global_model, global_model, args, auxiliary_data_loader, current_round=None, initial_update=None):
    aggregator = IMSAggregator(args, args.device)
    return aggregator.aggregate(agent_updates_dict, flat_global_model, global_model, auxiliary_data_loader, initial_update, current_round)
