import torch
import copy
import logging
import numpy as np
from a4fl_core import A4FL_Core
from torch.nn.utils import parameters_to_vector
import utils

class A4FL_Aggregator:
    def __init__(self, args):
        self.args = args
        self.device = args.device if hasattr(args, 'device') else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.a4fl_core = A4FL_Core(args, self.device)

    def aggregate(self, agent_updates_dict, global_model, auxiliary_loader, agent_data_sizes):
        """
        Implementation of A4FL Aggregation Logic.
        
        Args:
            agent_updates_dict: Dictionary {agent_id: update_vector}
            global_model: The current global model
            auxiliary_loader: DataLoader containing clean samples on server
            agent_data_sizes: Dictionary {agent_id: n_samples}
        
        Returns:
            aggregated_update: The weighted average of legitimate updates
        """
        logging.info("A4FL: Starting aggregation process (Statistical Filtering)...")
        
        if auxiliary_loader is None:
            logging.error("A4FL: Auxiliary loader is None! Cannot perform statistical test.")
            return self._simple_avg(agent_updates_dict, agent_data_sizes)

        # 1. Collect Metrics
        client_metrics = {}
        initial_params = parameters_to_vector([global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
        temp_model = copy.deepcopy(global_model)
        
        for agent_id, update in agent_updates_dict.items():
            # Reconstruct local model parameters
            local_params = initial_params + update
            
            # Load into temp model
            utils.vector_to_model_wo_load(local_params, temp_model)
            
            # Perform Statistical Test
            metrics = self.a4fl_core.statistical_test(temp_model, global_model, auxiliary_loader)
            client_metrics[agent_id] = metrics
            
            logging.info(f"Client {agent_id}: Loss={metrics['loss']:.4f}, Acc={metrics['acc']:.4f}, "
                         f"CosSim={metrics['cos_sim']:.4f}, Eucl={metrics['eucl_dist']:.4f}, Var={metrics['var']:.4f}")

        # 2. Statistical Filtering (Outlier Detection)
        # Use Loss on Auxiliary Data as the primary metric for filtering
        losses = [m['loss'] for m in client_metrics.values()]
        if len(losses) > 0:
            median_loss = np.median(losses)
            mad_loss = np.median([abs(l - median_loss) for l in losses])
            # Threshold: Median + 2 * MAD (Standard robust outlier detection)
            # If MAD is 0 (all same), we might need a fallback or use mean/std
            if mad_loss < 1e-6:
                 mad_loss = np.std(losses)
            
            threshold = median_loss + 2.0 * mad_loss
            
            # Also consider Cosine Similarity
            cos_sims = [m['cos_sim'] for m in client_metrics.values()]
            median_cos = np.median(cos_sims)
            mad_cos = np.median([abs(c - median_cos) for c in cos_sims])
            if mad_cos < 1e-6: mad_cos = np.std(cos_sims)
            threshold_cos = median_cos - 2.0 * mad_cos # Lower is bad
        else:
            threshold = float('inf')
            threshold_cos = -1.0

        logging.info(f"A4FL Thresholds: Loss > {threshold:.4f}, CosSim < {threshold_cos:.4f}")

        legitimate_updates = []
        malicious_count = 0
        total_samples = 0
        
        for agent_id, metrics in client_metrics.items():
            is_legit = True
            
            # Filter logic
            if metrics['loss'] > threshold:
                is_legit = False
            
            # Optional: Double check with Cosine Similarity
            if metrics['cos_sim'] < threshold_cos:
                is_legit = False
                
            # If we know the number of corrupt clients, we can also use Top-K filtering
            # But let's stick to statistical outlier detection first.
            
            if is_legit:
                legitimate_updates.append((agent_data_sizes[agent_id], agent_updates_dict[agent_id]))
                total_samples += agent_data_sizes[agent_id]
            else:
                malicious_count += 1
                
        logging.info(f"A4FL: Filtered {malicious_count} malicious clients. {len(legitimate_updates)} legitimate clients remain.")
        
        # Global Aggregation
        if len(legitimate_updates) == 0:
            logging.warning("A4FL: No legitimate updates found! Returning zero update.")
            return torch.zeros_like(initial_params), None
            
        accumulated_update = torch.zeros_like(initial_params)
        for n_samples, update in legitimate_updates:
            accumulated_update += update * n_samples
        
        aggregated_update = accumulated_update / total_samples
        
        return aggregated_update, None

    def _simple_avg(self, agent_updates_dict, agent_data_sizes):
        total_n = sum(agent_data_sizes.values())
        accumulated_update = None
        for agent_id, update in agent_updates_dict.items():
            n_samples = agent_data_sizes[agent_id]
            if accumulated_update is None:
                accumulated_update = update * n_samples
            else:
                accumulated_update += update * n_samples
        return accumulated_update / total_n, None
