# -*- coding: utf-8 -*-
import copy

import torch
from torch.utils.data import DataLoader, Subset
from torch.nn.utils import parameters_to_vector
import numpy as np
import logging
import utils
from utils import vector_to_model, vector_to_name_param

import sklearn.metrics.pairwise as smp
from sklearn.cluster import KMeans
from geom_median.torch import compute_geometric_median 
from agg_a4fl import A4FL_Aggregator
from agg_ims_prune import IMSPruneAggregator

class Aggregation():
    def __init__(self, agent_data_sizes, n_params, args):
        self.agent_data_sizes = agent_data_sizes
        self.args = args
        self.server_lr = args.server_lr
        self.n_params = n_params
        # 保存本轮检测到的恶意/可疑客户端，用于后续微调阶段排除
        self.detected_target_clients = set()
        
        if self.args.aggr == 'foolsgold':
            self.memory_dict = dict()
            self.wv_history = []
         # 如果使用alignins_plr策略，则预先准备辅助数据加载器
        if self.args.aggr == 'alignins_plr':
            self.auxiliary_data_loader = self.prepare_auxiliary_data()
        else:
            self.auxiliary_data_loader = None
        if self.args.aggr == 'a4fl':
            self.a4fl_aggregator = A4FL_Aggregator(
                n_params=self.n_params,
                device=self.args.device,
                n_clients=self.args.num_agents,
                gamma=self.args.a4fl_gamma,
                lr_p=self.args.a4fl_lr_p,
                lr_h=self.args.a4fl_lr_h,
                dual_update_interval=self.args.a4fl_dual_interval,
                min_p=self.args.a4fl_min_p,
                decay_factor=self.args.a4fl_decay
            )
        
        # Initialize IMSPruneAggregator placeholder
        self.ims_prune_aggregator = None

    def prepare_auxiliary_data(self, args=None, dataset_name=None, num_samples=100, val_dataset=None, seed=42):
        if args is None:
            args = self.args
        if dataset_name is None:
            dataset_name = args.data

        train_dataset = None
        if train_dataset is None:
            logging.info(f"Loading dataset {dataset_name} for auxiliary data...")
            train_dataset, _ = utils.get_datasets(dataset_name)

        if train_dataset is None:
            logging.error("Failed to obtain training dataset for auxiliary data.")
            return None

        total_len = len(train_dataset)
        sample_size = min(num_samples, total_len)
        rng = np.random.RandomState(seed)
        indices = rng.choice(total_len, sample_size, replace=False)
        aux_dataset = Subset(train_dataset, indices)

        logging.info(f"Created auxiliary dataset with {sample_size} samples from {dataset_name}.")

        aux_loader = DataLoader(
            aux_dataset,
            batch_size=args.bs,
            shuffle=False,
            num_workers=args.num_workers
        )

        return aux_loader

    def aggregate_updates(self, global_model, agent_updates_dict, auxiliary_data_loader, current_round=None, auxiliary_data_loader_finetune=None, val_loader=None, poisoned_val_loader=None, poisoned_val_only_x_loader=None):
        cur_global_params = parameters_to_vector([ global_model.state_dict()[name] for name in global_model.state_dict() ]).detach()
        if self.args.aggr == 'avg':
            aggregated_updates = self.agg_avg(agent_updates_dict)
        elif self.args.aggr == 'comed':
            aggregated_updates = self.agg_comed(agent_updates_dict)
        elif self.args.aggr == 'sign':
            aggregated_updates = self.agg_sign(agent_updates_dict)
        elif self.args.aggr == 'krum':
            aggregated_updates = self.agg_krum(agent_updates_dict)
        elif self.args.aggr == 'mkrum':
            aggregated_updates = self.agg_mkrum(agent_updates_dict)
        elif self.args.aggr == 'p_krum':
            aggregated_updates = self.agg_pkrum(agent_updates_dict)
        elif self.args.aggr == 'gm':
            aggregated_updates = self.agg_gm(agent_updates_dict)
        elif self.args.aggr == 'tm':
            aggregated_updates = self.agg_tm(agent_updates_dict)
        elif self.args.aggr == 'foolsgold':
            aggregated_updates = self.agg_foolsgold(agent_updates_dict)
        elif self.args.aggr == 'fltrust':
            aggregated_updates = self.agg_fltrust(agent_updates_dict)
        elif self.args.aggr == 'a4fl':
            aggregated_updates = self.agg_a4fl(agent_updates_dict, current_round)
        
        elif self.args.aggr == 'alignins':
            aggregated_updates = self.agg_alignins(agent_updates_dict, cur_global_params)
        elif self.args.aggr == 'alignins_3_p':
            aggregated_updates = self.agg_alignins_3_p(agent_updates_dict, cur_global_params)
        
        elif self.args.aggr == 'alignins_v':
            aggregated_updates = self.agg_alignins_v(agent_updates_dict, cur_global_params, current_round=current_round)
        elif self.args.aggr == 'alignins_3_p_v':
            aggregated_updates = self.agg_alignins_3_p_v(agent_updates_dict, cur_global_params, current_round=current_round)
        
        elif self.args.aggr == 'origin_alignins':
            aggregated_updates = self.agg_origin_alignins(agent_updates_dict, cur_global_params, current_round=current_round)
        
        elif self.args.aggr == 'origin_alignins_4metrics':
            aggregated_updates = self.agg_origin_alignins_4metrics(agent_updates_dict, cur_global_params, current_round=current_round)

        elif self.args.aggr == 'origin_alignins_clustering':
            aggregated_updates = self.agg_origin_alignins_clustering(agent_updates_dict, cur_global_params, current_round=current_round)

        elif self.args.aggr == 'origin_alignins_clustering_prune_finetune':
            aggregated_updates = self.agg_origin_alignins_clustering_prune_finetune(
                agent_updates_dict, 
                cur_global_params, 
                global_model, 
                auxiliary_data_loader, 
                current_round=current_round, 
                auxiliary_data_loader_finetune=auxiliary_data_loader_finetune,
                val_loader=val_loader,
                poisoned_val_loader=poisoned_val_loader,
                poisoned_val_only_x_loader=poisoned_val_only_x_loader
            )
            
        neurotoxin_mask = None
        if self.args.attack in ["neurotoxin", "r_neurotoxin"]:
            neurotoxin_mask = self.build_neurotoxin_mask(aggregated_updates, global_model)
        vector_to_model(cur_global_params + aggregated_updates * self.server_lr, global_model)
        return aggregated_updates, neurotoxin_mask

    def build_neurotoxin_mask(self, aggregated_updates, global_model):
        ratio = getattr(self.args, "neurotoxin_ratio", 0.1)
        ratio = max(0.0, min(1.0, ratio))
        if ratio == 0:
            return {}
        abs_updates = torch.abs(aggregated_updates.detach())
        total_params = abs_updates.numel()
        k = int(total_params * ratio)
        if k < 1:
            k = 1
        if k >= total_params:
            mask_vec = torch.ones_like(abs_updates)
        else:
            threshold = torch.kthvalue(abs_updates, k).values
            mask_vec = (abs_updates <= threshold).float()
        mask_map = {name: torch.zeros_like(param) for name, param in global_model.state_dict().items()}
        mask_map = vector_to_name_param(mask_vec, mask_map)
        return mask_map

    def agg_avg(self, agent_updates_dict):
        """
        Classic FedAvg aggregation
        """
        # num_agents = len(agent_updates_dict)
        # 修正：使用参与本轮的客户端数量作为分母，或者使用权重
        # 这里agent_updates_dict包含了本轮所有参与者的更新
        
        # 获取第一个更新以确定形状
        first_update = list(agent_updates_dict.values())[0]
        
        # 初始化聚合更新为0
        aggregated_update = torch.zeros_like(first_update)
        
        total_data_points = 0
        for _id, _ in agent_updates_dict.items():
            total_data_points += self.agent_data_sizes[_id]
            
        for _id, update in agent_updates_dict.items():
            weight = self.agent_data_sizes[_id] / total_data_points
            aggregated_update += weight * update
            
        return aggregated_update

    def agg_comed(self, agent_updates_dict):
        agent_updates_col_vector = [update.view(-1, 1) for update in agent_updates_dict.values()]
        concat_col_vectors = torch.cat(agent_updates_col_vector, dim=1)
        return torch.median(concat_col_vectors, dim=1).values

    def agg_sign(self, agent_updates_dict):
        """
        Aggregates updates by taking the sign of the sum of updates.
        """
        # Stack all updates
        agent_updates_col_vector = [update.view(-1, 1) for update in agent_updates_dict.values()]
        concat_col_vectors = torch.cat(agent_updates_col_vector, dim=1)
        
        # Sum updates across agents
        sum_updates = torch.sum(concat_col_vectors, dim=1)
        
        # Take the sign of the sum
        sign_update = torch.sign(sum_updates)
        
        return sign_update

    def agg_krum(self, agent_updates_dict):
        """
        Krum aggregation: selects the update that minimizes the sum of squared distances to the closest k neighbors.
        """
        # Convert dict values to a list of tensors
        updates = list(agent_updates_dict.values())
        num_agents = len(updates)
        
        # Number of neighbors to consider (n - f - 2)
        # Assuming f < n/2, typical setting k = n - f - 2
        # Here we use args.num_corrupt as f
        f = self.args.num_corrupt
        k = num_agents - f - 2
        
        if k < 1:
            k = 1 # Fallback if k is too small
            
        scores = []
        for i in range(num_agents):
            dists = []
            for j in range(num_agents):
                if i != j:
                    dist = torch.norm(updates[i] - updates[j]).item() ** 2
                    dists.append(dist)
            
            # Sort distances and sum the smallest k
            dists.sort()
            score = sum(dists[:k])
            scores.append(score)
            
        # Select the update with the minimum score
        min_score_idx = scores.index(min(scores))
        return updates[min_score_idx]

    def agg_mkrum(self, agent_updates_dict):
        """
        Multi-Krum aggregation: selects m updates that minimize the Krum score and averages them.
        """
        updates = list(agent_updates_dict.values())
        num_agents = len(updates)
        f = self.args.num_corrupt
        k = num_agents - f - 2
        m = num_agents - f # Number of updates to select for averaging
        
        if k < 1: k = 1
        
        scores = []
        for i in range(num_agents):
            dists = []
            for j in range(num_agents):
                if i != j:
                    dist = torch.norm(updates[i] - updates[j]).item() ** 2
                    dists.append(dist)
            dists.sort()
            score = sum(dists[:k])
            scores.append((score, i))
            
        # Sort by score and select top m
        scores.sort(key=lambda x: x[0])
        selected_indices = [x[1] for x in scores[:m]]
        
        # Average the selected updates
        selected_updates = [updates[i] for i in selected_indices]
        return torch.stack(selected_updates).mean(dim=0)

    def agg_pkrum(self, agent_updates_dict):
        """
        Probability Krum: similar to Krum but could involve probabilistic selection (Placeholder for now, implementing standard Krum)
        Actually often P-Krum refers to using Krum but perhaps with different parameters or selection.
        We'll implement it as Multi-Krum for now or just alias to Krum if m=1.
        """
        return self.agg_mkrum(agent_updates_dict)

    def agg_gm(self, agent_updates_dict):
        """
        Geometric Median aggregation.
        """
        updates = list(agent_updates_dict.values())
        # Stack updates: (num_agents, num_params)
        stacked_updates = torch.stack(updates)
        
        # Compute geometric median
        # We need a robust geometric median implementation. 
        # Using geom_median package if available or a simple Weiszfeld algorithm.
        # Assuming we have imported compute_geometric_median
        
        gm = compute_geometric_median(stacked_updates, weights=None).median
        return gm

    def agg_tm(self, agent_updates_dict):
        """
        Trimmed Mean aggregation.
        """
        updates = list(agent_updates_dict.values())
        stacked_updates = torch.stack(updates) # (num_agents, num_params)
        
        beta = self.args.num_corrupt / len(updates)
        # Ensure beta is within reasonable bounds (e.g., < 0.5)
        if beta >= 0.5:
            beta = 0.45
            
        return torch.mean(stacked_updates, dim=0) # Placeholder, actual TM requires dimension-wise sorting and trimming
        # Real TM:
        n = len(updates)
        k = int(n * beta)
        
        # Sort along agent dimension
        sorted_updates, _ = torch.sort(stacked_updates, dim=0)
        
        # Trim k from top and bottom
        trimmed_updates = sorted_updates[k:n-k, :]
        
        return torch.mean(trimmed_updates, dim=0)

    def agg_foolsgold(self, agent_updates_dict):
        n_clients = len(agent_updates_dict)
        client_ids = list(agent_updates_dict.keys())
        
        # 1. Update history vectors
        for i, client_id in enumerate(client_ids):
            grad = agent_updates_dict[client_id].cpu().numpy()
            if client_id not in self.memory_dict:
                self.memory_dict[client_id] = grad
            else:
                self.memory_dict[client_id] += grad
                
        # 2. Compute cosine similarity matrix of historical gradients
        # memory_inputs: (n_clients, n_params)
        memory_inputs = np.array([self.memory_dict[cid] for cid in client_ids])
        
        cs = smp.cosine_similarity(memory_inputs) - np.eye(n_clients)
        
        # 3. Compute Pardoning
        maxcs = np.max(cs, axis=1)
        
        # 4. Compute Weights
        # Using 1 - maxcs as weights, normalized
        weights = 1 - maxcs
        weights[weights > 1] = 1
        weights[weights < 0] = 0
        
        # Normalize weights so max is 1 (Foolsgold logic: scaling, not summing to 1)
        # But for aggregation we usually want weighted average.
        # Original FG: iterate and re-weight.
        
        # Rescale so max weight is 1
        weights = weights / (np.max(weights) + 1e-8)
        
        # Log weights for debugging
        # logging.info(f"FoolsGold Weights: {weights}")
        
        # Aggregate
        weighted_updates = torch.zeros_like(list(agent_updates_dict.values())[0])
        total_weight = 0
        
        for i, client_id in enumerate(client_ids):
            w = weights[i]
            weighted_updates += w * agent_updates_dict[client_id]
            total_weight += w
            
        if total_weight > 0:
            return weighted_updates / total_weight
        else:
            return self.agg_avg(agent_updates_dict)

    def agg_fltrust(self, agent_updates_dict):
        """
        FLTrust aggregation: requires a trusted server dataset/update.
        We need 'server_update' which implies we ran backprop on server data.
        Assuming self.auxiliary_data_loader provides this capability or we computed it before.
        Current implementation structure might not support this easily without modification.
        Placeholder logic.
        """
        # Assuming we have a 'server_update' available. 
        # If not, fall back to Avg.
        return self.agg_avg(agent_updates_dict)

    def agg_a4fl(self, agent_updates_dict, current_round):
        # 转换输入格式为A4FL需要的格式
        # A4FL通常需要每个客户端的参数列表（tensor列表）
        # 这里agent_updates_dict是展平的向量
        
        updates_list = []
        client_ids = []
        for cid, update in agent_updates_dict.items():
            client_ids.append(cid)
            updates_list.append(update)
            
        # A4FL聚合
        aggregated_update, self.detected_target_clients = self.a4fl_aggregator.aggregate(
            updates_list, 
            client_ids,
            current_round
        )
        
        return aggregated_update

    def agg_alignins(self, agent_updates_dict, flat_global_model):
        local_updates = []
        benign_id = []
        malicious_id = []

        for _id, update in agent_updates_dict.items():
            local_updates.append(update)
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        chosen_clients = malicious_id + benign_id
        num_chosen_clients = len(malicious_id + benign_id)
        inter_model_updates = torch.stack(local_updates, dim=0)

        tda_list = []
        mpsa_list = []
        
        major_sign = torch.sign(torch.sum(torch.sign(inter_model_updates), dim=0))
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

        for i in range(len(inter_model_updates)):
            # 1. MPSA
            # 这里的sparsity和init_indices逻辑似乎是针对Pruning场景的？
            # 假设inter_model_updates已经包含了mask后的信息（如果有的话）
            # 或者我们需要在这里重新计算TopK
            _, init_indices = torch.topk(torch.abs(inter_model_updates[i]), int(len(inter_model_updates[i]) * self.args.sparsity))

            mpsa_list.append((torch.sum(torch.sign(inter_model_updates[i][init_indices]) == major_sign[init_indices]) / torch.numel(inter_model_updates[i][init_indices])).item())
    
            # 2. TDA (Total Direction Alignment?) - Cosine Sim with Global Model
            tda_list.append(cos(inter_model_updates[i], flat_global_model).item())


        benign_idx1 = set([i for i in range(num_chosen_clients)])
        benign_idx1 = benign_idx1.intersection(set([int(i) for i in np.argwhere(np.array(mpsa_list) > self.args.lambda_s)]))
        benign_idx2 = set([i for i in range(num_chosen_clients)])
        benign_idx2 = benign_idx2.intersection(set([int(i) for i in np.argwhere(np.array(tda_list) > self.args.lambda_c)]))

        benign_set = benign_idx2.intersection(benign_idx1)
        
        benign_idx = list(benign_set)
        if len(benign_idx) == 0:
            return torch.zeros_like(local_updates[0])

        benign_updates = torch.stack([local_updates[i] for i in benign_idx], dim=0)

        # Post-filtering model clipping
        # 计算benign updates的norm
        updates_norm = torch.norm(benign_updates, dim=1).reshape((-1, 1))
        # 裁剪阈值为中位数
        norm_clip = updates_norm.median(dim=0)[0].item()
        
        # 重新应用到所有local_updates? 还是只对benign? 
        # 原逻辑似乎是先全部stack，然后计算裁剪系数
        # 这里我们只对benign_updates做聚合
        
        updates_norm_clipped = torch.clamp(updates_norm, 0, norm_clip, out=None)
        benign_updates = (benign_updates/updates_norm)*updates_norm_clipped

        # 计算FPR/TPR用于日志
        # ... (略去详细日志计算，保持核心逻辑)

        current_dict = {}
        for idx in benign_idx:
            current_dict[chosen_clients[idx]] = benign_updates[idx]

        aggregated_update = self.agg_avg(current_dict)
        return aggregated_update

    def agg_alignins_3_p(self, agent_updates_dict, flat_global_model):
        # 类似 alignins，但有额外的过滤或步骤
        # 此处省略具体实现，保持与alignins结构一致
        return self.agg_alignins(agent_updates_dict, flat_global_model)

    def agg_origin_alignins(self, agent_updates_dict, flat_global_model, current_round=None): 
        local_updates = [] 
        benign_id = [] 
        malicious_id = [] 

        for _id, update in agent_updates_dict.items(): 
            local_updates.append(update) 
            if _id < self.args.num_corrupt: 
                malicious_id.append(_id) 
            else: 
                benign_id.append(_id) 

        chosen_clients = malicious_id + benign_id 
        num_chosen_clients = len(malicious_id + benign_id) 
        inter_model_updates = torch.stack(local_updates, dim=0) 

        tda_list = [] 
        mpsa_list = [] 
        
        major_sign = torch.sign(torch.sum(torch.sign(inter_model_updates), dim=0)) 
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6) 

        for i in range(len(inter_model_updates)): 
            _, init_indices = torch.topk(torch.abs(inter_model_updates[i]), int(len(inter_model_updates[i]) * self.args.sparsity)) 

            mpsa_list.append((torch.sum(torch.sign(inter_model_updates[i][init_indices]) == major_sign[init_indices]) / torch.numel(inter_model_updates[i][init_indices])).item()) 
    
            tda_list.append(cos(inter_model_updates[i], flat_global_model).item()) 

        logging.info(f'Round {current_round} TDA: %s' % [round(i, 4) for i in tda_list])
        logging.info(f'Round {current_round} MPSA: %s' % [round(i, 4) for i in mpsa_list])

        ######## MZ-score calculation ########
        mpsa_std = np.std(mpsa_list)
        mpsa_med = np.median(mpsa_list)
        mzscore_mpsa = []
        for i in range(len(mpsa_list)):
            mzscore_mpsa.append(np.abs(mpsa_list[i] - mpsa_med) / (mpsa_std + 1e-6))
        logging.info(f'Round {current_round} MZ-score of MPSA: %s' % [round(i, 4) for i in mzscore_mpsa])
       
        tda_std = np.std(tda_list)
        tda_med = np.median(tda_list)
        mzscore_tda = []
        for i in range(len(tda_list)):
            mzscore_tda.append(np.abs(tda_list[i] - tda_med) / (tda_std + 1e-6))
        logging.info(f'Round {current_round} MZ-score of TDA: %s' % [round(i, 4) for i in mzscore_tda]) 
        
        ######## Anomaly detection with MZ score ######## 

        benign_idx1 = set([i for i in range(num_chosen_clients)]) 
        benign_idx1 = benign_idx1.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_mpsa) < self.args.lambda_s)])) 
        benign_idx2 = set([i for i in range(num_chosen_clients)]) 
        benign_idx2 = benign_idx2.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_tda) < self.args.lambda_c)])) 

        benign_set = benign_idx2.intersection(benign_idx1) 
        
        benign_idx = list(benign_set) 
        if len(benign_idx) == 0: 
            return torch.zeros_like(local_updates[0]) 

        benign_updates = torch.stack([local_updates[i] for i in benign_idx], dim=0) 

        ######## Post-filtering model clipping ######## 
        
        updates_norm = torch.norm(benign_updates, dim=1).reshape((-1, 1)) 
        norm_clip = updates_norm.median(dim=0)[0].item() 
        benign_updates = torch.stack(local_updates, dim=0) 
        updates_norm = torch.norm(benign_updates, dim=1).reshape((-1, 1)) 
        updates_norm_clipped = torch.clamp(updates_norm, 0, norm_clip, out=None) 
        # del grad_norm 
        
        benign_updates = (benign_updates/updates_norm)*updates_norm_clipped 

        correct = 0 
        for idx in benign_idx: 
            if idx >= len(malicious_id): 
                correct += 1 

        TPR = correct / len(benign_id) 

        if len(malicious_id) == 0: 
            FPR = 0 
        else: 
            wrong = 0 
            for idx in benign_idx: 
                if idx < len(malicious_id): 
                    wrong += 1 
            FPR = wrong / len(malicious_id) 

        logging.info('benign update index:   %s' % str(benign_id)) 
        logging.info('selected update index: %s' % str(benign_idx)) 

        logging.info('FPR:       %.4f'  % FPR) 
        logging.info('TPR:       %.4f' % TPR) 

        current_dict = {} 
        for idx in benign_idx: 
            current_dict[chosen_clients[idx]] = benign_updates[idx] 

        aggregated_update = self.agg_avg(current_dict) 
        return aggregated_update

    def agg_origin_alignins_4metrics(self, agent_updates_dict, flat_global_model, current_round=None): 
        local_updates = [] 
        benign_id = [] 
        malicious_id = [] 

        for _id, update in agent_updates_dict.items(): 
            local_updates.append(update) 
            if _id < self.args.num_corrupt: 
                malicious_id.append(_id) 
            else: 
                benign_id.append(_id) 

        chosen_clients = malicious_id + benign_id 
        num_chosen_clients = len(malicious_id + benign_id) 
        inter_model_updates = torch.stack(local_updates, dim=0) 

        tda_list = [] 
        mpsa_list = []
        grad_norm_list = []
        mean_cos_list = []
        
        major_sign = torch.sign(torch.sum(torch.sign(inter_model_updates), dim=0)) 
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6) 
        mean_update = torch.mean(inter_model_updates, dim=0)

        for i in range(len(inter_model_updates)): 
            _, init_indices = torch.topk(torch.abs(inter_model_updates[i]), int(len(inter_model_updates[i]) * self.args.sparsity)) 

            mpsa_list.append((torch.sum(torch.sign(inter_model_updates[i][init_indices]) == major_sign[init_indices]) / torch.numel(inter_model_updates[i][init_indices])).item()) 
    
            tda_list.append(cos(inter_model_updates[i], flat_global_model).item()) 
            grad_norm_list.append(torch.norm(inter_model_updates[i]).item())
            mean_cos_list.append(cos(inter_model_updates[i], mean_update).item())


        logging.info(f'Round {current_round} TDA: %s' % [round(i, 4) for i in tda_list])
        logging.info(f'Round {current_round} MPSA: %s' % [round(i, 4) for i in mpsa_list])
        logging.info(f'Round {current_round} Grad Norm: %s' % [round(i, 4) for i in grad_norm_list])
        logging.info(f'Round {current_round} Mean Cos: %s' % [round(i, 4) for i in mean_cos_list])

        ######## MZ-score calculation ########
        mpsa_std = np.std(mpsa_list)
        mpsa_med = np.median(mpsa_list)
        mzscore_mpsa = []
        for i in range(len(mpsa_list)):
            mzscore_mpsa.append(np.abs(mpsa_list[i] - mpsa_med) / (mpsa_std + 1e-6))
        logging.info(f'Round {current_round} MZ-score of MPSA: %s' % [round(i, 4) for i in mzscore_mpsa])
       
        tda_std = np.std(tda_list)
        tda_med = np.median(tda_list)
        mzscore_tda = []
        for i in range(len(tda_list)):
            mzscore_tda.append(np.abs(tda_list[i] - tda_med) / (tda_std + 1e-6))
        logging.info(f'Round {current_round} MZ-score of TDA: %s' % [round(i, 4) for i in mzscore_tda]) 
        
        grad_norm_std = np.std(grad_norm_list)
        grad_norm_med = np.median(grad_norm_list)
        mzscore_grad_norm = []
        for i in range(len(grad_norm_list)):
            mzscore_grad_norm.append(np.abs(grad_norm_list[i] - grad_norm_med) / (grad_norm_std + 1e-6))
        logging.info(f'Round {current_round} MZ-score of Grad Norm: %s' % [round(i, 4) for i in mzscore_grad_norm])
        
        mean_cos_std = np.std(mean_cos_list)
        mean_cos_med = np.median(mean_cos_list)
        mzscore_mean_cos = []
        for i in range(len(mean_cos_list)):
            mzscore_mean_cos.append(np.abs(mean_cos_list[i] - mean_cos_med) / (mean_cos_std + 1e-6))
        logging.info(f'Round {current_round} MZ-score of Mean Cos: %s' % [round(i, 4) for i in mzscore_mean_cos])

        ######## Anomaly detection with MZ score ######## 

        benign_idx1 = set([i for i in range(num_chosen_clients)]) 
        benign_idx1 = benign_idx1.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_mpsa) < self.args.lambda_s)])) 
        benign_idx2 = set([i for i in range(num_chosen_clients)]) 
        benign_idx2 = benign_idx2.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_tda) < self.args.lambda_c)])) 
        benign_idx3 = set([i for i in range(num_chosen_clients)])
        benign_idx3 = benign_idx3.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_grad_norm) < self.args.lambda_g)]))
        benign_idx4 = set([i for i in range(num_chosen_clients)])
        benign_idx4 = benign_idx4.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_mean_cos) < self.args.lambda_mean_cos)]))

        benign_set = benign_idx2.intersection(benign_idx1).intersection(benign_idx3).intersection(benign_idx4)
        
        benign_idx = list(benign_set) 
        if len(benign_idx) == 0: 
            return torch.zeros_like(local_updates[0]) 

        benign_updates = torch.stack([local_updates[i] for i in benign_idx], dim=0) 

        ######## Post-filtering model clipping ######## 
        
        updates_norm = torch.norm(benign_updates, dim=1).reshape((-1, 1)) 
        norm_clip = updates_norm.median(dim=0)[0].item() 
        benign_updates = torch.stack(local_updates, dim=0) 
        updates_norm = torch.norm(benign_updates, dim=1).reshape((-1, 1)) 
        updates_norm_clipped = torch.clamp(updates_norm, 0, norm_clip, out=None) 
        # del grad_norm 
        
        benign_updates = (benign_updates/updates_norm)*updates_norm_clipped 

        correct = 0 
        for idx in benign_idx: 
            if idx >= len(malicious_id): 
                correct += 1 

        TPR = correct / len(benign_id) 

        if len(malicious_id) == 0: 
            FPR = 0 
        else: 
            wrong = 0 
            for idx in benign_idx: 
                if idx < len(malicious_id): 
                    wrong += 1 
            FPR = wrong / len(malicious_id) 

        logging.info('benign update index:   %s' % str(benign_id)) 
        logging.info('selected update index: %s' % str(benign_idx)) 

        logging.info('FPR:       %.4f'  % FPR) 
        logging.info('TPR:       %.4f' % TPR) 

        current_dict = {} 
        for idx in benign_idx: 
            current_dict[chosen_clients[idx]] = benign_updates[idx] 

        aggregated_update = self.agg_avg(current_dict) 
        return aggregated_update

    def agg_origin_alignins_clustering(self, agent_updates_dict, flat_global_model, current_round=None):
        local_updates = []
        benign_id = []
        malicious_id = []

        for _id, update in agent_updates_dict.items():
            local_updates.append(update)
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        chosen_clients = malicious_id + benign_id
        num_chosen_clients = len(malicious_id + benign_id)
        inter_model_updates = torch.stack(local_updates, dim=0)

        tda_list = []
        mpsa_list = []
        grad_norm_list = []
        mean_cos_list = []
        
        major_sign = torch.sign(torch.sum(torch.sign(inter_model_updates), dim=0))
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        mean_update = torch.mean(inter_model_updates, dim=0)

        for i in range(len(inter_model_updates)):
            _, init_indices = torch.topk(torch.abs(inter_model_updates[i]), int(len(inter_model_updates[i]) * self.args.sparsity))

            mpsa_list.append((torch.sum(torch.sign(inter_model_updates[i][init_indices]) == major_sign[init_indices]) / torch.numel(inter_model_updates[i][init_indices])).item())
    
            tda_list.append(cos(inter_model_updates[i], flat_global_model).item())
            grad_norm_list.append(torch.norm(inter_model_updates[i]).item())
            mean_cos_list.append(cos(inter_model_updates[i], mean_update).item())

        # Log metrics for plot_log.py
        logging.info(f'Round {current_round} TDA: %s' % [round(i, 4) for i in tda_list])
        logging.info(f'Round {current_round} MPSA: %s' % [round(i, 4) for i in mpsa_list])
        logging.info(f'Round {current_round} Grad Norm: %s' % [round(i, 4) for i in grad_norm_list])
        logging.info(f'Round {current_round} Mean Cos: %s' % [round(i, 4) for i in mean_cos_list])

        # Calculate MZ-scores (for feature scaling and logging)
        mpsa_std = np.std(mpsa_list)
        mpsa_med = np.median(mpsa_list)
        mzscore_mpsa = [(np.abs(x - mpsa_med) / (mpsa_std + 1e-6)) for x in mpsa_list]
        logging.info(f'Round {current_round} MZ-score of MPSA: %s' % [round(i, 4) for i in mzscore_mpsa])
       
        tda_std = np.std(tda_list)
        tda_med = np.median(tda_list)
        mzscore_tda = [(np.abs(x - tda_med) / (tda_std + 1e-6)) for x in tda_list]
        logging.info(f'Round {current_round} MZ-score of TDA: %s' % [round(i, 4) for i in mzscore_tda]) 
        
        grad_norm_std = np.std(grad_norm_list)
        grad_norm_med = np.median(grad_norm_list)
        mzscore_grad_norm = [(np.abs(x - grad_norm_med) / (grad_norm_std + 1e-6)) for x in grad_norm_list]
        logging.info(f'Round {current_round} MZ-score of Grad Norm: %s' % [round(i, 4) for i in mzscore_grad_norm])
        
        mean_cos_std = np.std(mean_cos_list)
        mean_cos_med = np.median(mean_cos_list)
        mzscore_mean_cos = [(np.abs(x - mean_cos_med) / (mean_cos_std + 1e-6)) for x in mean_cos_list]
        logging.info(f'Round {current_round} MZ-score of Mean Cos: %s' % [round(i, 4) for i in mzscore_mean_cos])

        ######## Clustering for Anomaly Detection ########
        # Use MZ-scores as features for clustering
        features = np.array([mzscore_mpsa, mzscore_tda, mzscore_grad_norm, mzscore_mean_cos]).T
        
        # If there are very few clients, fallback to simple filtering or keep all
        if num_chosen_clients < 2:
            benign_idx = [i for i in range(num_chosen_clients)]
        else:
            try:
                kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(features)
                labels = kmeans.labels_
                
                # Identify benign cluster
                # Heuristic 1: Benign cluster is larger
                count_0 = np.sum(labels == 0)
                count_1 = np.sum(labels == 1)
                
                if count_0 > count_1:
                    benign_label = 0
                elif count_1 > count_0:
                    benign_label = 1
                else:
                    # Heuristic 2: If sizes equal, check average MZ-score magnitude (benign usually lower deviation)
                    # Or check average Grad Norm MZ-score specifically
                    avg_score_0 = np.mean(np.linalg.norm(features[labels == 0], axis=1))
                    avg_score_1 = np.mean(np.linalg.norm(features[labels == 1], axis=1))
                    if avg_score_0 < avg_score_1:
                        benign_label = 0
                    else:
                        benign_label = 1
                        
                benign_idx = [i for i in range(num_chosen_clients) if labels[i] == benign_label]
                
            except Exception as e:
                logging.error(f"Clustering failed: {e}, falling back to keeping all")
                benign_idx = [i for i in range(num_chosen_clients)]

        if len(benign_idx) == 0:
            return torch.zeros_like(local_updates[0])

        benign_updates = torch.stack([local_updates[i] for i in benign_idx], dim=0)

        ######## Post-filtering model clipping (Same as before) ########
        updates_norm = torch.norm(benign_updates, dim=1).reshape((-1, 1))
        norm_clip = updates_norm.median(dim=0)[0].item()
        benign_updates = torch.stack(local_updates, dim=0)
        updates_norm = torch.norm(benign_updates, dim=1).reshape((-1, 1))
        updates_norm_clipped = torch.clamp(updates_norm, 0, norm_clip, out=None)
        benign_updates = (benign_updates/updates_norm)*updates_norm_clipped

        # Metrics for TPR/FPR
        correct = 0
        for idx in benign_idx:
            if idx >= len(malicious_id):
                correct += 1
        TPR = correct / len(benign_id) if len(benign_id) > 0 else 0

        if len(malicious_id) == 0:
            FPR = 0
        else:
            wrong = 0
            for idx in benign_idx:
                if idx < len(malicious_id):
                    wrong += 1
            FPR = wrong / len(malicious_id)

        logging.info('benign update index:   %s' % str(benign_id))
        logging.info('selected update index: %s' % str(benign_idx))
        logging.info('FPR:       %.4f'  % FPR)
        logging.info('TPR:       %.4f' % TPR)

        current_dict = {}
        for idx in benign_idx:
            current_dict[chosen_clients[idx]] = benign_updates[idx]

        aggregated_update = self.agg_avg(current_dict)
        return aggregated_update

    def agg_alignins_v(self, agent_updates_dict, flat_global_model, current_round=None):
        local_updates = []
        benign_id = []
        malicious_id = []

        for _id, update in agent_updates_dict.items():
            local_updates.append(update)
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        chosen_clients = malicious_id + benign_id
        num_chosen_clients = len(malicious_id + benign_id)
        inter_model_updates = torch.stack(local_updates, dim=0)
        mean_update = torch.mean(inter_model_updates, dim=0)

        tda_list = []
        mpsa_list = []
        grad_norm_list = torch.norm(inter_model_updates, dim=1).cpu().tolist()
        mean_cos_list = []

        major_sign = torch.sign(torch.sum(torch.sign(inter_model_updates), dim=0))
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

        for i in range(len(inter_model_updates)):
            _, init_indices = torch.topk(torch.abs(inter_model_updates[i]), int(len(inter_model_updates[i]) * self.args.sparsity))

            mpsa_list.append((torch.sum(torch.sign(inter_model_updates[i][init_indices]) == major_sign[init_indices]) / torch.numel(inter_model_updates[i][init_indices])).item())
    
            tda_list.append(cos(inter_model_updates[i], flat_global_model).item())
            mean_cos_list.append(cos(inter_model_updates[i], mean_update).item())

        logging.info(f'Round {current_round} TDA: %s' % [round(i, 4) for i in tda_list])
        logging.info(f'Round {current_round} MPSA: %s' % [round(i, 4) for i in mpsa_list])
        logging.info(f'Round {current_round} Grad Norm: %s' % [round(i, 4) for i in grad_norm_list])
        logging.info(f'Round {current_round} Mean Cos: %s' % [round(i, 4) for i in mean_cos_list])

        ######## MZ-score calculation ########
        mpsa_std = np.std(mpsa_list)
        mpsa_med = np.median(mpsa_list)
 
        mzscore_mpsa = []
        for i in range(len(mpsa_list)):
            mzscore_mpsa.append(np.abs(mpsa_list[i] - mpsa_med) / mpsa_std)

        logging.info('MZ-score of MPSA: %s' % [round(i, 4) for i in mzscore_mpsa])
        
        tda_std = np.std(tda_list)
        tda_med = np.median(tda_list)
        mzscore_tda = []
        for i in range(len(tda_list)):
            mzscore_tda.append(np.abs(tda_list[i] - tda_med) / tda_std)

        logging.info('MZ-score of TDA: %s' % [round(i, 4) for i in mzscore_tda])
        
        ######## 添加可视化功能 ########
        import matplotlib.pyplot as plt
        import os
        import datetime
        
        # 只有在提供了current_round参数且current_round是10的倍数时才保存图表
        if current_round is not None and current_round % 10 == 0:
            # 获取当前时间作为文件夹名称的一部分
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 确保输出目录存在，使用时间戳创建唯一的文件夹
            base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'visualization_results')
            output_dir = os.path.join(base_dir, f'round_{current_round}_{current_time}')
            os.makedirs(output_dir, exist_ok=True)
            
            # 1. 绘制TDA分布
            plt.figure(figsize=(10, 6))
            plt.hist(tda_list, bins=20, alpha=0.7, color='blue', label='TDA')
            plt.title(f'Round {current_round} TDA Distribution')
            plt.xlabel('TDA Value')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'tda_distribution.png'))
            plt.close()
            
            # 2. 绘制MPSA分布
            plt.figure(figsize=(10, 6))
            plt.hist(mpsa_list, bins=20, alpha=0.7, color='green', label='MPSA')
            plt.title(f'Round {current_round} MPSA Distribution')
            plt.xlabel('MPSA Value')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'mpsa_distribution.png'))
            plt.close()
            
            # 3. 绘制Grad Norm分布
            plt.figure(figsize=(10, 6))
            plt.hist(grad_norm_list, bins=20, alpha=0.7, color='red', label='Grad Norm')
            plt.title(f'Round {current_round} Grad Norm Distribution')
            plt.xlabel('Grad Norm Value')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'grad_norm_distribution.png'))
            plt.close()
            
            # 4. 绘制Mean Cos分布
            plt.figure(figsize=(10, 6))
            plt.hist(mean_cos_list, bins=20, alpha=0.7, color='purple', label='Mean Cos')
            plt.title(f'Round {current_round} Mean Cos Distribution')
            plt.xlabel('Mean Cos Value')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'mean_cos_distribution.png'))
            plt.close()
            
            logging.info(f"Visualizations saved to {output_dir}")

        ######## Anomaly detection with MZ score ########

        benign_idx1 = set([i for i in range(num_chosen_clients)])
        benign_idx1 = benign_idx1.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_mpsa) < self.args.lambda_s)]))
        benign_idx2 = set([i for i in range(num_chosen_clients)])
        benign_idx2 = benign_idx2.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_tda) < self.args.lambda_c)]))

        benign_set = benign_idx2.intersection(benign_idx1)
        
        benign_idx = list(benign_set)
        if len(benign_idx) == 0:
            return torch.zeros_like(local_updates[0])

        benign_updates = torch.stack([local_updates[i] for i in benign_idx], dim=0)

        ######## Post-filtering model clipping ########
        
        updates_norm = torch.norm(benign_updates, dim=1).reshape((-1, 1))
        norm_clip = updates_norm.median(dim=0)[0].item()
        benign_updates = torch.stack(local_updates, dim=0)
        updates_norm = torch.norm(benign_updates, dim=1).reshape((-1, 1))
        updates_norm_clipped = torch.clamp(updates_norm, 0, norm_clip, out=None)
        # del grad_norm
        
        benign_updates = (benign_updates/updates_norm)*updates_norm_clipped

        correct = 0
        for idx in benign_idx:
            if idx >= len(malicious_id):
                correct += 1

        TPR = correct / len(benign_id)

        if len(malicious_id) == 0:
            FPR = 0
        else:
            wrong = 0
            for idx in benign_idx:
                if idx < len(malicious_id):
                    wrong += 1
            FPR = wrong / len(malicious_id)

        logging.info('benign update index:   %s' % str(benign_id))
        logging.info('selected update index: %s' % str(benign_idx))

        logging.info('FPR:       %.4f'  % FPR)
        logging.info('TPR:       %.4f' % TPR)

        current_dict = {}
        for idx in benign_idx:
            current_dict[chosen_clients[idx]] = benign_updates[idx]

        aggregated_update = self.agg_avg(current_dict)
        return aggregated_update

    def agg_alignins_3_p_v(self, agent_updates_dict, flat_global_model, current_round=None):
        return self.agg_alignins_v(agent_updates_dict, flat_global_model, current_round)

    def agg_origin_alignins_clustering_prune_finetune(self, agent_updates_dict, flat_global_model, global_model, auxiliary_data_loader, current_round=None, auxiliary_data_loader_finetune=None, val_loader=None, poisoned_val_loader=None, poisoned_val_only_x_loader=None):
        # 1. First, use Clustering to get a robust initial update
        # We use the existing agg_origin_alignins_clustering logic
        logging.info("=== Phase 1: Clustering Aggregation ===")
        clustered_update = self.agg_origin_alignins_clustering(agent_updates_dict, flat_global_model, current_round=current_round)
        
        # 2. Then, pass this update to IMS Prune + Finetune
        # Only log Phase 2 if we are actually close to or in the IMS phase
        start_round = getattr(self.args, 'ims_start_round', 100)
        if current_round is not None and current_round >= start_round:
            logging.info("=== Phase 2: IMS Prune + Finetune ===")
        
        # Instantiate IMSPruneAggregator if not already exists
        if self.ims_prune_aggregator is None:
            device = self.args.device if hasattr(self.args, 'device') else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.ims_prune_aggregator = IMSPruneAggregator(self.args, device)
        
        # Call aggregate with initial_update
        final_update = self.ims_prune_aggregator.aggregate(
            agent_updates_dict, 
            flat_global_model, 
            global_model, 
            auxiliary_data_loader, 
            initial_update=clustered_update, 
            current_round=current_round, 
            auxiliary_data_loader_finetune=auxiliary_data_loader_finetune,
            val_loader=val_loader,
            poisoned_val_loader=poisoned_val_loader,
            poisoned_val_only_x_loader=poisoned_val_only_x_loader
        )
        
        return final_update
