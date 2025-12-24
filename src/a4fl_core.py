import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import utils
import logging
import math

class A4FL_Core:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.loss_fn = nn.CrossEntropyLoss()

    def add_trigger(self, image, pattern_type='plus', similar=False):
        # Helper to add trigger manually, supporting 'similar' variation
        # Image shape: (C, H, W)
        x = np.array(image.cpu().numpy().squeeze())
        # If squeeze removed channel dim (e.g. grayscale), handle it. 
        # But here assuming CIFAR (3, 32, 32).
        
        # Check shape format. Usually it's (C, H, W) for PyTorch tensors.
        # But numpy indexing x[i, j][d] suggests (H, W, C)?
        # The error "IndexError: index 5 is out of bounds for axis 0 with size 3"
        # suggests x is (3, 32, 32) (C, H, W) but code accesses x[i, start_idx] where i is > 3.
        # So x is indeed (C, H, W).
        # We need to access x[d, i, start_idx] instead of x[i, start_idx][d]
        
        if pattern_type == 'plus':
            start_idx = 5
            size = 6
            if similar:
                # Shift pattern by 2 pixels for "similar" trigger
                start_idx += 2 
            
            # Simple plus pattern
            for d in range(0, 3):
                # Vertical
                for i in range(start_idx, min(start_idx + size + 1, 32)):
                    if d == 2:
                        # x[channel, row, col]
                        x[d, i, start_idx] = 0
                    else:
                        x[d, i, start_idx] = 255
                # Horizontal
                for i in range(start_idx - size // 2, min(start_idx + size // 2 + 1, 32)):
                    if d == 2:
                        x[d, start_idx + size // 2, i] = 0
                    else:
                        x[d, start_idx + size // 2, i] = 255
                        
        return torch.tensor(x).unsqueeze(0)

    def create_training_samples(self, train_loader, global_model, m=500):
        # Collect some clean samples
        clean_samples = []
        clean_labels = []
        
        # Iterate to get enough samples
        for inputs, labels in train_loader:
            clean_samples.append(inputs)
            clean_labels.append(labels)
            if len(clean_samples) * inputs.shape[0] >= 2000: # Limit memory usage
                break
                
        if not clean_samples:
            return train_loader # Fallback
            
        clean_inputs = torch.cat(clean_samples)
        clean_targets = torch.cat(clean_labels)
        
        # 1. Clean Samples (subset)
        X_clean = clean_inputs
        y_clean = clean_targets
        
        # 2. Messy Samples (Clean + Trigger -> Target Label)
        # Randomly select m samples
        indices = np.random.choice(len(X_clean), min(m, len(X_clean)), replace=False)
        X_messy = X_clean[indices].clone()
        # Add trigger
        for i in range(len(X_messy)):
            X_messy[i] = self.add_trigger(X_messy[i], self.args.pattern_type, similar=False)
        
        y_messy = torch.ones_like(y_clean[indices]) * self.args.target_class
        
        # 3. Wrap Samples (Clean + Similar Trigger -> Original Label)
        indices_wrap = np.random.choice(len(X_clean), min(m, len(X_clean)), replace=False)
        X_wrap = X_clean[indices_wrap].clone()
        for i in range(len(X_wrap)):
            X_wrap[i] = self.add_trigger(X_wrap[i], self.args.pattern_type, similar=True)
        y_wrap = y_clean[indices_wrap]
        
        # Combine
        X_combined = torch.cat([X_clean, X_messy, X_wrap])
        y_combined = torch.cat([y_clean, y_messy, y_wrap])
        
        # Standardization is handled by the model/transform usually, but here tensors are already normalized if they came from loader
        
        return DataLoader(TensorDataset(X_combined, y_combined), batch_size=self.args.bs, shuffle=True)

    def generate_UAP(self, model, clean_loader, epsilon=0.05, steps=10):
        model.eval()
        UAP = torch.zeros((3, 32, 32), device=self.device, requires_grad=True)
        optimizer = optim.SGD([UAP], lr=0.01)
        
        # Use a few batches for UAP generation
        for _ in range(steps):
            for inputs, labels in clean_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Perturb
                perturbed = inputs + UAP
                outputs = model(perturbed)
                
                # Maximize loss (make model misclassify)
                loss = -self.loss_fn(outputs, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Clip UAP
                UAP.data = torch.clamp(UAP.data, -epsilon, epsilon)
                break # Just one batch per step for speed
                
        return UAP.detach()

    def adversarial_training(self, model, train_loader, UAP, epochs=5):
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=self.args.client_lr, momentum=self.args.momentum)
        
        for epoch in range(epochs):
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Clean forward
                optimizer.zero_grad()
                outputs = model(inputs)
                cls_loss = self.loss_fn(outputs, labels)
                
                # Adversarial forward
                adv_inputs = inputs + UAP
                adv_outputs = model(adv_inputs)
                adv_loss = self.loss_fn(adv_outputs, labels)
                
                # Composite loss
                loss = cls_loss + 0.5 * adv_loss
                loss.backward()
                optimizer.step()
                
        return model

    def calculate_neuron_importance(self, model, loader):
        # Calculate importance based on weight * gradient
        model.eval()
        importance = {}
        
        # Get gradients
        optimizer = optim.SGD(model.parameters(), lr=0.0) # Dummy
        optimizer.zero_grad()
        
        # Accumulate gradients over a few batches
        count = 0
        for inputs, labels in loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = model(inputs)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            count += 1
            if count >= 2: break
            
        for name, param in model.named_parameters():
            if param.grad is not None:
                # Importance = |weight * grad|
                imp = torch.abs(param.data * param.grad)
                importance[name] = imp
            else:
                importance[name] = torch.zeros_like(param.data)
                
        return importance

    def prune_and_finetune(self, model, loader, threshold=0.1, epochs=2):
        # 1. Calculate importance
        importance = self.calculate_neuron_importance(model, loader)
        
        # 2. Prune (Apply mask)
        mask = {}
        for name, imp in importance.items():
            # Determine threshold value for this layer
            if imp.numel() > 0:
                # Find the value at the 'threshold' percentile
                k = int(imp.numel() * threshold)
                if k > 0:
                    topk, _ = torch.topk(imp.view(-1), k, largest=False)
                    thresh_val = topk[-1]
                    mask[name] = (imp >= thresh_val).float()
                else:
                    mask[name] = torch.ones_like(imp)
            else:
                mask[name] = torch.ones_like(imp)
                
        # Apply mask
        for name, param in model.named_parameters():
            if name in mask:
                param.data *= mask[name]
                
        # 3. Fine-tune
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=self.args.client_lr * 0.1, momentum=self.args.momentum)
        
        for epoch in range(epochs):
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                
                # Apply mask to gradients to keep pruned weights at 0
                for name, param in model.named_parameters():
                    if name in mask and param.grad is not None:
                        param.grad *= mask[name]
                        
                optimizer.step()
                
        return model

    def statistical_test(self, local_model, global_model, clean_test_loader):
        # Metrics: Min, Max, Var, Cos Sim, Eucl Dist, Loss, Acc
        local_model.eval()
        global_model.eval()
        
        local_preds = []
        global_preds = []
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in clean_test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                local_logits = local_model(inputs)
                global_logits = global_model(inputs)
                
                local_out = torch.softmax(local_logits, dim=1)
                global_out = torch.softmax(global_logits, dim=1)
                
                local_preds.append(local_out)
                global_preds.append(global_out)
                
                # Loss and Acc for local model
                loss = self.loss_fn(local_logits, labels)
                total_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(local_logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Use limited batches for efficiency if dataset is large
                if total >= 200: 
                    break 
        
        local_preds = torch.cat(local_preds)
        global_preds = torch.cat(global_preds)
        
        # Calculate metrics
        var = torch.var(local_preds).item()
        
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        cos_sim = torch.mean(cos(local_preds, global_preds)).item()
        
        eucl_dist = torch.mean(torch.norm(local_preds - global_preds, dim=1)).item()
        
        avg_loss = total_loss / total if total > 0 else 100.0
        acc = correct / total if total > 0 else 0.0
        
        metrics = {
            "var": var,
            "cos_sim": cos_sim,
            "eucl_dist": eucl_dist,
            "loss": avg_loss,
            "acc": acc
        }
        
        return metrics
