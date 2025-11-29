import torch
import copy
from typing import List, Tuple, Set
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import logging

class NoTUnlearningAggregator:
    def __init__(self, args):
        self.args = args

    def aggregate(self,
                  inter_model_updates: torch.Tensor,
                  flat_global_model: torch.Tensor,
                  global_model: torch.nn.Module,
                  malicious_id: List[int] = None,
                  current_round: int = None) -> Tuple[torch.Tensor, torch.nn.Module, Set[int]]:
        
        num_clients = inter_model_updates.shape[0]
        
        # ==========================================
        # Standard FedAvg Aggregation
        # ==========================================
        logging.info(f"NoT Unlearning: Executing standard FedAvg aggregation (Round {current_round})...")
        
        # Simply average all updates
        final_update = torch.mean(inter_model_updates, dim=0)
        
        final_model = copy.deepcopy(global_model)
        final_params = flat_global_model + final_update
        vector_to_parameters(final_params, final_model.parameters())
        
        # Return empty set for target clients as we are not detecting anymore
        target_clients = set()
        
        return final_update, final_model, target_clients


def agg_not_unlearning(inter_model_updates: torch.Tensor,
                       flat_global_model: torch.Tensor,
                       global_model: torch.nn.Module,
                       args,
                       malicious_id: List[int] = None,
                       current_round: int = None) -> Tuple[torch.Tensor, torch.nn.Module, Set[int]]:
    aggregator = NoTUnlearningAggregator(args)
    return aggregator.aggregate(
        inter_model_updates,
        flat_global_model,
        global_model,
        malicious_id,
        current_round,
    )
