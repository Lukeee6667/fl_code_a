import torch
import copy
import logging
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

        # Statistical Filtering
        legitimate_updates = []
        malicious_count = 0
        total_samples = 0
        
        initial_params = parameters_to_vector([global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
        
        # We use a temp model to load client parameters for testing
        temp_model = copy.deepcopy(global_model)
        
        for agent_id, update in agent_updates_dict.items():
            n_samples = agent_data_sizes[agent_id]
            
            # Reconstruct local model parameters
            local_params = initial_params + update
            
            # Load into temp model
            utils.vector_to_model_wo_load(local_params, temp_model)
            
            # Perform Statistical Test
            # Compares temp_model (local) vs global_model (baseline) using auxiliary data
            is_legit = self.a4fl_core.statistical_test(temp_model, global_model, auxiliary_loader)
            
            if is_legit:
                legitimate_updates.append((n_samples, update))
                total_samples += n_samples
            else:
                malicious_count += 1
                
        logging.info(f"A4FL: Filtered {malicious_count} malicious clients. {len(legitimate_updates)} legitimate clients remain.")
        
        # Global Aggregation
        if len(legitimate_updates) == 0:
            logging.warning("A4FL: No legitimate updates found! Returning zero update.")
            return torch.zeros_like(initial_params)
            
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
