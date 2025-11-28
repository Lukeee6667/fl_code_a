import torch
import copy
from typing import List, Tuple
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class NoTUnlearningAggregator:
    def __init__(self, args):
        self.args = args

    def aggregate(self,
                  inter_model_updates: torch.Tensor,
                  flat_global_model: torch.Tensor,
                  global_model: torch.nn.Module,
                  malicious_id: List[int] = None,
                  current_round: int = None) -> Tuple[torch.Tensor, torch.nn.Module]:
        num_clients = inter_model_updates.shape[0]
        if malicious_id is None:
            target_clients = set(range(min(getattr(self.args, 'num_corrupt', 0), num_clients)))
        else:
            target_clients = set([i for i in malicious_id if i < num_clients])
        retain_clients = set(range(num_clients)) - target_clients

        perturbed_model = copy.deepcopy(global_model)
        target_name = None
        for name, _ in perturbed_model.named_parameters():
            if ('conv' in name.lower()) and ('weight' in name.lower()):
                target_name = name
                break
        if target_name is None:
            for name, _ in perturbed_model.named_parameters():
                target_name = name
                break
        for name, param in perturbed_model.named_parameters():
            if name == target_name:
                param.data = -param.data
                break

        perturbed_params = parameters_to_vector([perturbed_model.state_dict()[n] for n in perturbed_model.state_dict()])
        negation_update = perturbed_params - flat_global_model

        if len(retain_clients) > 0:
            retain_updates = torch.stack([inter_model_updates[i] for i in sorted(list(retain_clients))])
            retain_avg = torch.mean(retain_updates, dim=0)
        else:
            retain_avg = torch.zeros_like(flat_global_model)

        final_update = negation_update + retain_avg
        final_model = copy.deepcopy(global_model)
        final_params = flat_global_model + final_update
        vector_to_parameters(final_params, final_model.parameters())
        return final_update, final_model


def agg_not_unlearning(inter_model_updates: torch.Tensor,
                       flat_global_model: torch.Tensor,
                       global_model: torch.nn.Module,
                       args,
                       malicious_id: List[int] = None,
                       current_round: int = None) -> Tuple[torch.Tensor, torch.nn.Module]:
    aggregator = NoTUnlearningAggregator(args)
    return aggregator.aggregate(
        inter_model_updates,
        flat_global_model,
        global_model,
        malicious_id,
        current_round,
    )
