# -*- coding: utf-8 -*-
import utils
import models
import math
import copy
import numpy as np
from agent import Agent
from agent_sparse import Agent as Agent_s
from aggregation import Aggregation
import torch
import random
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import logging
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True

    def run_not_unlearning(global_model, args, criterion, val_loader, poisoned_val_loader, poisoned_val_only_x_loader, neurotoxin_mask=None):
        logging.info("NoT Unlearning: Starting one-shot unlearning...")

        target_name = None
        for name, _ in global_model.named_parameters():
            if ("conv" in name.lower()) and ("weight" in name.lower()):
                target_name = name
                break
        if target_name is None:
            for name, _ in global_model.named_parameters():
                target_name = name
                break

        if target_name:
            logging.info("NoT Unlearning: Negating layer '%s'" % target_name)
            for name, param in global_model.named_parameters():
                if name == target_name:
                    param.data = -param.data
                    break
        else:
            logging.warning("NoT Unlearning: No suitable layer found for negation!")

        if args.not_finetune_rounds <= 0:
            args.not_finetune_rounds = 5

        logging.info("NoT Unlearning: Starting fine-tuning for %d rounds..." % args.not_finetune_rounds)

        orig_lr = args.client_lr
        orig_local_ep = args.local_ep

        args.client_lr = args.not_finetune_lr
        args.local_ep = args.not_finetune_local_ep

        logging.info("NoT Unlearning: Reloading clean dataset for fine-tuning...")
        clean_train_dataset, _ = utils.get_datasets(args.data)

        full_train_loader = DataLoader(
            clean_train_dataset,
            batch_size=args.bs,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=False,
            drop_last=True,
        )
        logging.info("NoT Unlearning: Fine-tuning using full training dataset.")

        dummy_agent = Agent(
            id=args.num_agents,
            args=args,
            train_dataset=clean_train_dataset,
            data_idxs=list(range(len(clean_train_dataset))),
            backdoor_train_dataset=None,
        )
        dummy_agent.train_loader = full_train_loader
        dummy_agent.n_data = len(clean_train_dataset)
        dummy_agent.is_malicious = 0

        for ft_rnd in range(1, args.not_finetune_rounds + 1):
            logging.info("NoT Unlearning: Starting fine-tuning round %d..." % ft_rnd)
            update = dummy_agent.local_train(
                global_model, criterion, ft_rnd, neurotoxin_mask=neurotoxin_mask or {}
            )

            cur_params = parameters_to_vector(
                [global_model.state_dict()[name] for name in global_model.state_dict()]
            )
            lr_vec = torch.tensor([args.server_lr] * len(cur_params), device=args.device)
            new_params = (cur_params + lr_vec * update).float()
            utils.vector_to_model(new_params, global_model)
            logging.info("NoT Unlearning: Fine-tuning round %d completed" % ft_rnd)

        args.client_lr = orig_lr
        args.local_ep = orig_local_ep

        logging.info("---------Test After NoT Unlearning ------------")
        val_acc = utils.get_loss_n_accuracy(
            global_model, criterion, val_loader, args, args.rounds + args.not_finetune_rounds, args.num_target
        )
        asr = utils.get_loss_n_accuracy(
            global_model,
            criterion,
            poisoned_val_loader,
            args,
            args.rounds + args.not_finetune_rounds,
            num_classes=args.num_target,
        )
        poison_acc = utils.get_loss_n_accuracy(
            global_model,
            criterion,
            poisoned_val_only_x_loader,
            args,
            args.rounds + args.not_finetune_rounds,
            args.num_target,
        )
        logging.info("Post-Unlearning Clean ACC:            %.4f" % val_acc)
        logging.info("Post-Unlearning Attack Success Ratio: %.4f" % asr)
        logging.info("Post-Unlearning Backdoor ACC:         %.4f" % poison_acc)

        return val_acc, asr, poison_acc

    parser = argparse.ArgumentParser(description="pass in a parameter")

    parser.add_argument(
        "--data", type=str, default="cifar10", help="dataset we want to train on"
    )
    parser.add_argument("--num_agents", type=int, default=20, help="number of agents:K")
    parser.add_argument(
        "--agent_frac", type=float, default=1.0, help="fraction of agents per round:C"
    )
    parser.add_argument(
        "--num_corrupt", type=int, default=2, help="number of corrupt agents"
    )
    parser.add_argument(
        "--rounds", type=int, default=150, help="number of communication rounds:R"
    )
    parser.add_argument(
        "--local_ep", type=int, default=2, help="number of local epochs:E"
    )
    parser.add_argument("--bs", type=int, default=64, help="local batch size: B")
    parser.add_argument(
        "--client_lr", type=float, default=0.1, help="clients learning rate"
    )
    parser.add_argument(
        "--server_lr", type=float, default=1, help="servers learning rate"
    )
    parser.add_argument(
        "--target_class", type=int, default=7, help="target class for backdoor attack"
    )
    parser.add_argument(
        "--poison_frac",
        type=float,
        default=0.5,
        help="fraction of dataset to corrupt for backdoor attack",
    )
    parser.add_argument(
        "--pattern_type", type=str, default="plus", help="shape of bd pattern"
    )
    parser.add_argument(
        "--theta", type=int, default=8, help="break ties when votes sum to 0"
    )
    parser.add_argument(
        "--theta_ld", type=int, default=10, help="break ties when votes sum to 0"
    )
    parser.add_argument(
        "--snap", type=int, default=1, help="do inference in every num of snap rounds"
    )
    parser.add_argument(
        "--device",
        default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        help="To use cuda, set to a specific GPU ID.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="num of workers for multithreading"
    )
    parser.add_argument(
        "--dense_ratio",
        type=float,
        default=0.25,
        help="num of workers for multithreading",
    )
    parser.add_argument(
        "--anneal_factor",
        type=float,
        default=0.0001,
        help="num of workers for multithreading",
    )
    parser.add_argument(
        "--se_threshold",
        type=float,
        default=1e-4,
        help="num of workers for multithreading",
    )
    parser.add_argument("--non_iid", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument(
        "--attack",
        type=str,
        default="badnet",
        choices=["badnet", "DBA", "neurotoxin", "pgd"],
    )
    parser.add_argument(
        "--aggr",
        type=str,
        default="avg",
        choices=[
            "avg",
            "alignins",
            "origin_alignins",
            "origin_alignins_4metrics",
            "origin_alignins_4metrics_voting",
            "alignins_v",
            "alignins_g_v",
            "alignins_g_v2",
            "alignins_g_v2_onepic",
            "alignins_layer",
            "alignins_layer_lsd",
            "alignins_layer_3p",
            "alignins_3_m",
            "alignins_3_m_noniid_badnet",
            "alignins_3_m_noniid_dba",
            "alignins_3_m_noniid_dba",
            "alignins_plr",
            "rlr",
            "mkrum",
            "mmetric",
            "lockdown",
            "foolsgold",
            "rfa",
            "fedup",
            "fedup_avg",
            "alignins_fedup_hybrid",
            "alignins_fedup_correct",
            "not_unlearning",
            "not_unlearning_from_ckpt",
            "alignins_not_unlearning",
            "alignins_ims",
            "ims",
            "a4fl",
            "a4fl_alignins",
            "alignins_ims_recover",
            "ims_fast",
            "alignins_ims_fast_plus",
            "alignins_ims_standard",
            "ims_prune_finetune",
            "origin_alignins_clustering",
            "origin_alignins_clustering_weighted3",
            "origin_alignins_clustering_weighted3_dynamic",
            "origin_alignins_clustering_weighted3_dynamic_softmax_gated",
            "origin_alignins_clustering_weighted3_softmax_gated",
            "origin_alignins_clustering_prune_finetune",
            "origin_alignins_clustering_prune_finetune_adaptive",
        ],
        help="aggregation function to aggregate agents' local weights",
    )
    parser.add_argument("--not_finetune_rounds", type=int, default=0)
    parser.add_argument("--not_finetune_local_ep", type=int, default=2, help='Fine-tuning epochs (default: 2)')
    parser.add_argument("--not_finetune_lr", type=float, default=0.001, help='Fine-tuning learning rate (default: 0.001)')
    parser.add_argument("--not_finetune_patience", type=int, default=2, help="Early-stopping patience for fine-tuning (<=0 disables)")
    parser.add_argument("--lr_decay", type=float, default=0.99)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--mask_init", type=str, default="ERK")
    parser.add_argument("--wd", type=float, default=1e-4)
    
    # IMS Arguments
    parser.add_argument("--ims_start_round", type=int, default=100, help="Start round for IMS defense")
    parser.add_argument("--ims_margin", type=float, default=0.5, help="Margin for IMS KL loss")
    parser.add_argument("--same_mask", type=int, default=1)
    parser.add_argument("--cease_poison", type=float, default=100000)
    parser.add_argument("--exp_name_extra", type=str, help="defence name", default="")
    parser.add_argument("--super_power", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--sparsity", type=float, default=0.3)
    parser.add_argument("--lambda_s", type=float, default=1.0)
    parser.add_argument("--lambda_c", type=float, default=1.0)
    parser.add_argument("--lambda_g", type=float, default=1.5)
    parser.add_argument("--lambda_mean_cos", type=float, default=1.5)
    parser.add_argument("--cluster_imbalance_ratio", type=float, default=0.9)
    parser.add_argument(
        "--cluster_mz_metrics",
        type=str,
        default="mpsa,tda,grad_norm,mean_cos",
        help="comma-separated MZ-score metrics for KMeans (mpsa,tda,grad_norm,mean_cos)",
    )
    parser.add_argument("--suspicious_weight", type=float, default=0.5, help="Weight for suspicious clients in AlignIns")
    parser.add_argument("--benign_weight", type=float, default=1.0, help="Weight for benign clients in 3-way aggregation")
    parser.add_argument("--gated2_bd_delta_max", type=float, default=0.02, help="Max allowed backdoor ACC drop for choosing B+S in gated2")
    parser.add_argument("--strict_factor", type=float, default=0.8, help="Factor for strict threshold in AlignIns (default: 0.8)")
    
    # FedUP相关参数（基于论文）
    parser.add_argument('--fedup_pruning_ratio', type=float, default=0.1,
                        help='FedUP基础剪枝比例 (default: 0.1)')
    parser.add_argument('--fedup_p_max', type=float, default=0.15,
                        help='FedUP最大剪枝率 (default: 0.15)')
    parser.add_argument('--fedup_p_min', type=float, default=0.01,
                        help='FedUP最小剪枝率 (default: 0.01)')
    parser.add_argument('--fedup_gamma', type=float, default=5,
                        help='FedUP曲线陡度参数 (default: 5)')
    parser.add_argument('--fedup_sensitivity_threshold', type=float, default=0.5,
                        help='FedUP异常检测敏感度阈值 (default: 0.5)')

    # IMS参数
    parser.add_argument('--ims_lr', type=float, default=1e-2, help='IMS learning rate')
    parser.add_argument('--ims_r1', type=int, default=20, help='IMS mask initialization rounds')
    parser.add_argument('--ims_r2', type=int, default=15, help='IMS outer loop rounds')
    parser.add_argument('--ims_r3', type=int, default=5, help='IMS inner loop rounds')
    parser.add_argument('--ims_k', type=float, default=20, help='IMS scaling factor')
    parser.add_argument('--ims_lambda_init', type=float, default=0.0, help='IMS lambda init')
    parser.add_argument('--ims_lambda_final', type=float, default=10.0, help='IMS lambda final')
    parser.add_argument('--ims_epsilon', type=float, default=1.0, help='IMS perturbation constraint')
    parser.add_argument('--ims_clean_agree_weight', type=float, default=1.0, help='IMS clean agree loss weight')
    parser.add_argument('--ims_backdoor_recover_weight', type=float, default=1.0, help='IMS backdoor recover loss weight')
    parser.add_argument('--ims_adaptive', action='store_true', help='enable adaptive early-stop for r1/r2/r3')
    parser.add_argument('--ims_adapt_r1_min_epochs', type=int, default=5)
    parser.add_argument('--ims_adapt_r1_tol', type=float, default=1e-4)
    parser.add_argument('--ims_adapt_r1_patience', type=int, default=3)
    parser.add_argument('--ims_adapt_r1_mask_tol', type=float, default=1e-4)
    parser.add_argument('--ims_adapt_r1_mask_patience', type=int, default=2)
    parser.add_argument('--ims_adapt_r2_min_epochs', type=int, default=5)
    parser.add_argument('--ims_adapt_r2_tol', type=float, default=1e-4)
    parser.add_argument('--ims_adapt_r2_patience', type=int, default=3)
    parser.add_argument('--ims_adapt_r2_mask_tol', type=float, default=1e-4)
    parser.add_argument('--ims_adapt_r2_mask_patience', type=int, default=2)
    parser.add_argument('--ims_adapt_r3_min_steps', type=int, default=1)
    parser.add_argument('--ims_adapt_r3_tol', type=float, default=1e-4)
    parser.add_argument('--ims_adapt_r3_patience', type=int, default=2)
    parser.add_argument('--ims_adapt_r3_saturation_ratio', type=float, default=0.98)
    parser.add_argument('--aux_num_samples', type=int, default=1000, help='Auxiliary data sample size (default: 1000)')
    parser.add_argument("--resume", action="store_true", help="resume from checkpoint")
    parser.add_argument("--checkpoint_path", type=str, default="", help="path to checkpoint")
    parser.add_argument(
        "--prune_finetune_only",
        action="store_true",
        help="run IMS prune+finetune once from --checkpoint_path, then save and exit",
    )
    parser.add_argument(
        "--output_checkpoint_path",
        type=str,
        default="",
        help="where to save the pruned+finetuned checkpoint (default: derive from checkpoint_path)",
    )
    parser.add_argument("--save_freq", type=int, default=10, help="save checkpoint frequency")

    args = parser.parse_args()

    if args.aggr == "origin_alignins_clustering_prune_finetune_adaptive":
        args.ims_adaptive = True

    if args.clean:
        args.num_corrupt = 0
        args.exp_name_extra = "clean"

    if args.super_power:
        args.exp_name_extra = "sp"

    per_data_dict = {
        "num_target": {"fmnist": 10, "cifar10": 10, "cifar100": 100, "tinyimagenet": 200,},
    }

    args.num_target = per_data_dict["num_target"][args.data]

    args.log_dir = utils.setup_logging(args)
    
    # 记录所有超参数
    logging.info("================ Hyperparameters ================")
    for arg in vars(args):
        logging.info(f"{arg}: {getattr(args, arg)}")
    logging.info("=================================================")

    train_dataset, val_dataset = utils.get_datasets(args.data)
    backdoor_train_dataset = None

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    if args.non_iid:
        user_groups = utils.distribute_data_dirichlet(train_dataset, args)
    else:
        user_groups = utils.distribute_data(
            train_dataset, args, n_classes=args.num_target
        )

    idxs = (val_dataset.targets != args.target_class).nonzero().flatten().tolist()

    if args.data != "tinyimagenet":
        poisoned_val_set = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs)
        utils.poison_dataset(poisoned_val_set.dataset, args, idxs, poison_all=True)
    else:
        poisoned_val_set = utils.DatasetSplit(
            copy.deepcopy(val_dataset), idxs, runtime_poison=True, args=args
        )

    poisoned_val_loader = DataLoader(
        poisoned_val_set,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    if args.data != "tinyimagenet":
        idxs = (val_dataset.targets != args.target_class).nonzero().flatten().tolist()
        poisoned_val_set_only_x = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs)
        utils.poison_dataset(
            poisoned_val_set_only_x.dataset,
            args,
            idxs,
            poison_all=True,
            modify_label=False,
        )
    else:
        poisoned_val_set_only_x = utils.DatasetSplit(
            copy.deepcopy(val_dataset),
            idxs,
            runtime_poison=True,
            args=args,
            modify_label=False,
        )

    poisoned_val_only_x_loader = DataLoader(
        poisoned_val_set_only_x,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    # auxiliary_data_size = min(500, len(val_dataset))
    # auxiliary_indices = np.random.choice(len(val_dataset), auxiliary_data_size, replace=False)
    # auxiliary_dataset = torch.utils.data.Subset(val_dataset, auxiliary_indices)
    
    # auxiliary_data_loader = DataLoader(
    #     auxiliary_dataset,
    #     batch_size=args.bs,
    #     shuffle=False,
    #     num_workers=args.num_workers,
    #     pin_memory=False
    # )
    auxiliary_data_loader = None
    auxiliary_data_loader_finetune = None
    if args.prune_finetune_only or args.aggr in {
        'alignins_plr',
        'ims',
        'ims_fast',
        'alignins_ims',
        'alignins_ims_fast_plus',
        'alignins_ims_standard',
        'alignins_ims_recover',
        'ims_prune_finetune',
        'origin_alignins_clustering_weighted3_dynamic',
        'origin_alignins_clustering_weighted3_dynamic_softmax_gated',
        'origin_alignins_clustering_weighted3_softmax_gated',
        'origin_alignins_clustering_prune_finetune',
        'origin_alignins_clustering_prune_finetune_adaptive',
        'origin_alignins_clustering_gated2_auxclean',
        'a4fl',
        'a4fl_alignins',
    }:
        from aggregation import Aggregation
        
        temp_aggregator = Aggregation({}, 0, args)
        
        logging.info("为PLR分析准备辅助数据...")
        auxiliary_data_loader = temp_aggregator.prepare_auxiliary_data(
            args=args,
            dataset_name=args.data,
            num_samples=min(args.aux_num_samples, len(val_dataset)),
            val_dataset=val_dataset
        )
        
        if auxiliary_data_loader is None:
            logging.warning("无法创建辅助数据加载器，PLR分析可能无法正常工作")
        else:
            logging.info("成功创建辅助数据加载器")
            
        if args.aggr in {
            'ims_prune_finetune',
            'origin_alignins_clustering_prune_finetune',
            'origin_alignins_clustering_prune_finetune_adaptive',
            'origin_alignins_clustering_gated2_auxclean',
        }:
            logging.info("Generating separate auxiliary data for fine-tuning...")
            auxiliary_data_loader_finetune = temp_aggregator.prepare_auxiliary_data(
                args=args,
                dataset_name=args.data,
                num_samples=min(args.aux_num_samples, len(val_dataset)),
                val_dataset=val_dataset
            )
            if auxiliary_data_loader_finetune is None:
                logging.warning("Failed to create auxiliary data loader for fine-tuning")
            else:
                logging.info("Successfully created auxiliary data loader for fine-tuning")
            
        del temp_aggregator

    # initialize a model, and the agents
    global_model = models.get_model(args.data, args).to(args.device)

    if args.prune_finetune_only:
        if not args.checkpoint_path:
            raise ValueError("--checkpoint_path is required when --prune_finetune_only is set")

        checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        global_model.load_state_dict(state_dict)

        args.ims_start_round = 0

        from agg_ims_prune import IMSPruneAggregator

        pruner = IMSPruneAggregator(args, args.device)
        flat = parameters_to_vector(global_model.parameters()).detach()
        zero = torch.zeros_like(flat)

        effective_update = pruner.aggregate(
            agent_updates_dict={0: zero},
            flat_global_model=flat,
            global_model=global_model,
            auxiliary_data_loader=auxiliary_data_loader,
            initial_update=zero,
            current_round=0,
            auxiliary_data_loader_finetune=auxiliary_data_loader_finetune,
            val_loader=val_loader,
            poisoned_val_loader=poisoned_val_loader,
            poisoned_val_only_x_loader=poisoned_val_only_x_loader,
        )

        old_params = parameters_to_vector(global_model.parameters()).detach()
        new_params = (old_params + args.server_lr * effective_update).detach()
        vector_to_parameters(new_params, global_model.parameters())

        criterion = nn.CrossEntropyLoss().to(args.device)
        logging.info("---------Test After IMS Prune + Finetune (Standalone) ------------")
        val_acc = utils.get_loss_n_accuracy(global_model, criterion, val_loader, args, 0, args.num_target)
        asr = utils.get_loss_n_accuracy(global_model, criterion, poisoned_val_loader, args, 0, num_classes=args.num_target)
        ba = utils.get_loss_n_accuracy(global_model, criterion, poisoned_val_only_x_loader, args, 0, args.num_target)
        logging.info("Clean ACC:              %.4f" % val_acc)
        logging.info("Attack Success Ratio:   %.4f" % asr)
        logging.info("Backdoor ACC:           %.4f" % ba)

        if args.output_checkpoint_path:
            out_path = args.output_checkpoint_path
        else:
            root, ext = os.path.splitext(args.checkpoint_path)
            ext = ext if ext else ".pt"
            out_path = f"{root}_imsprune_ft_ep{getattr(args, 'not_finetune_local_ep', 0)}{ext}"

        save_dict = {
            "round": 0,
            "state_dict": global_model.state_dict(),
            "source_checkpoint": args.checkpoint_path,
            "mode": "prune_finetune_only",
        }
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        torch.save(save_dict, out_path)
        logging.info(f"Saved pruned+finetuned checkpoint to {out_path}")
        sys.exit(0)

    if args.aggr == "not_unlearning_from_ckpt":
        if not args.checkpoint_path:
            raise ValueError("--checkpoint_path is required when --aggr not_unlearning_from_ckpt")

        checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        global_model.load_state_dict(state_dict)

        criterion = nn.CrossEntropyLoss().to(args.device)
        run_not_unlearning(
            global_model,
            args,
            criterion,
            val_loader,
            poisoned_val_loader,
            poisoned_val_only_x_loader,
            neurotoxin_mask={},
        )

        if args.output_checkpoint_path:
            out_path = args.output_checkpoint_path
        else:
            root, ext = os.path.splitext(args.checkpoint_path)
            ext = ext if ext else ".pt"
            out_path = f"{root}_notunlearning_ft_ep{getattr(args, 'not_finetune_local_ep', 0)}{ext}"

        save_dict = {
            "round": 0,
            "state_dict": global_model.state_dict(),
            "source_checkpoint": args.checkpoint_path,
            "mode": "not_unlearning_from_ckpt",
        }
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        torch.save(save_dict, out_path)
        logging.info(f"Saved unlearned checkpoint to {out_path}")
        sys.exit(0)

    global_mask = {}
    neurotoxin_mask = {}
    updates_dict = {}
    n_model_params = len(
        parameters_to_vector(
            [global_model.state_dict()[name] for name in global_model.state_dict()]
        )
    )
    params = {
        name: copy.deepcopy(global_model.state_dict()[name])
        for name in global_model.state_dict()
    }

    if args.aggr == "lockdown":
        sparsity = utils.calculate_sparsities(args, params, distribution=args.mask_init)
        mask = utils.init_masks(params, sparsity)

    agents, agent_data_sizes = [], {}
    for _id in range(0, args.num_agents):
        if args.aggr == "lockdown":
            if args.same_mask == 0:
                agent = Agent_s(
                    _id,
                    args,
                    train_dataset,
                    user_groups[_id],
                    mask=utils.init_masks(params, sparsity),
                    backdoor_train_dataset=backdoor_train_dataset,
                )
            else:
                agent = Agent_s(
                    _id,
                    args,
                    train_dataset,
                    user_groups[_id],
                    mask=mask,
                    backdoor_train_dataset=backdoor_train_dataset,
                )
        else:
            agent = Agent(
                _id,
                args,
                train_dataset,
                user_groups[_id],
                backdoor_train_dataset=backdoor_train_dataset,
            )
        agent.is_malicious = 1 if _id < args.num_corrupt else 0
        agent_data_sizes[_id] = agent.n_data
        agents.append(agent)

        logging.info(
            "build client:{} mal:{} data_num:{}".format(
                _id, agent.is_malicious, agent.n_data
            )
        )

    aggregator = Aggregation(agent_data_sizes, n_model_params, args)

    criterion = nn.CrossEntropyLoss().to(args.device)
    agent_updates_dict = {}

    best_acc = -1
    best_asr = -1
    best_bcdr_acc = -1
    start_round = 1
    fedup_last_global_params = None
    fedup_last_updates = None
    fedup_last_client_ids = None

    if args.resume and args.checkpoint_path:
        if os.path.isfile(args.checkpoint_path):
            logging.info(f"=> loading checkpoint '{args.checkpoint_path}'")
            checkpoint = torch.load(args.checkpoint_path)
            start_round = checkpoint['round'] + 1
            global_model.load_state_dict(checkpoint['state_dict'])
            best_acc = checkpoint['best_acc']
            best_asr = checkpoint['best_asr']
            best_bcdr_acc = checkpoint['best_bcdr_acc']
            
            # Load IMS masks if available (for Config 22)
            if args.aggr == 'origin_alignins_clustering_prune_finetune':
                if 'ims_mask_A' in checkpoint and 'ims_mask_S' in checkpoint:
                    logging.info("=> Found IMS masks in checkpoint, loading...")
                    # Initialize IMS aggregator early to load masks
                    if aggregator.ims_prune_aggregator is None:
                        from agg_ims_prune import IMSPruneAggregator
                        aggregator.ims_prune_aggregator = IMSPruneAggregator(args, args.device)
                    
                    aggregator.ims_prune_aggregator.saved_mask_A = checkpoint['ims_mask_A']
                    aggregator.ims_prune_aggregator.saved_mask_S = checkpoint['ims_mask_S']
                    logging.info("=> Loaded IMS masks successfully.")
                else:
                    logging.info("=> No IMS masks found in checkpoint.")

            logging.info(f"=> loaded checkpoint '{args.checkpoint_path}' (round {checkpoint['round']})")
            
            # Evaluate after resuming
            logging.info(f"---------Test After Resume (Round {checkpoint['round']}) ------------")
            if args.aggr != "lockdown":
                val_acc = utils.get_loss_n_accuracy(
                    global_model, criterion, val_loader, args, checkpoint['round'], args.num_target
                )
                asr = utils.get_loss_n_accuracy(
                    global_model,
                    criterion,
                    poisoned_val_loader,
                    args,
                    checkpoint['round'],
                    num_classes=args.num_target,
                )
                poison_acc = utils.get_loss_n_accuracy(
                    global_model,
                    criterion,
                    poisoned_val_only_x_loader,
                    args,
                    checkpoint['round'],
                    args.num_target,
                )
                logging.info("Clean ACC:              %.4f" % val_acc)
                logging.info("Attack Success Ratio:   %.4f" % asr)
                logging.info("Backdoor ACC:           %.4f" % poison_acc)
        else:
            logging.info(f"=> no checkpoint found at '{args.checkpoint_path}'")

    for rnd in range(start_round, args.rounds + 1):
        logging.info("--------round {} ------------".format(rnd))
        rnd_global_params = parameters_to_vector(
            [
                copy.deepcopy(global_model.state_dict()[name])
                for name in global_model.state_dict()
            ]
        )
        agent_updates_dict = {}
        chosen = np.random.choice(
            args.num_agents,
            math.floor(args.num_agents * args.agent_frac),
            replace=False,
        )
        chosen = sorted(chosen)
        if args.aggr == "lockdown":
            old_mask = [copy.deepcopy(agent.mask) for agent in agents]

        for agent_id in chosen:
            if agents[agent_id].is_malicious and args.super_power:
                continue
            global_model = global_model.to(args.device)

            if args.aggr == "lockdown":
                update = agents[agent_id].local_train(
                    global_model,
                    criterion,
                    rnd,
                    global_mask=global_mask,
                    neurotoxin_mask=neurotoxin_mask,
                    updates_dict=updates_dict,
                )
            else:
                update = agents[agent_id].local_train(
                    global_model, criterion, rnd, neurotoxin_mask=neurotoxin_mask
                )
            agent_updates_dict[agent_id] = update
            utils.vector_to_model(copy.deepcopy(rnd_global_params), global_model)

        if args.aggr == "fedup_avg" and rnd == args.rounds:
            fedup_last_global_params = rnd_global_params.detach().cpu()
            fedup_last_updates = {cid: up.detach().cpu() for cid, up in agent_updates_dict.items()}
            fedup_last_client_ids = list(agent_updates_dict.keys())

        # aggregate params obtained by agents and update the global params
        updates_dict, neurotoxin_mask = aggregator.aggregate_updates(
            global_model, 
            agent_updates_dict, 
            auxiliary_data_loader, 
            rnd, 
            auxiliary_data_loader_finetune=auxiliary_data_loader_finetune,
            val_loader=val_loader,
            poisoned_val_loader=poisoned_val_loader,
            poisoned_val_only_x_loader=poisoned_val_only_x_loader
        )

        # inference in every args.snap rounds
        logging.info("---------Test {} ------------".format(rnd))
        if rnd % args.snap == 0:
            if args.aggr != "lockdown":
                val_acc = utils.get_loss_n_accuracy(
                    global_model, criterion, val_loader, args, rnd, args.num_target
                )
                asr = utils.get_loss_n_accuracy(
                    global_model,
                    criterion,
                    poisoned_val_loader,
                    args,
                    rnd,
                    num_classes=args.num_target,
                )
                poison_acc = utils.get_loss_n_accuracy(
                    global_model,
                    criterion,
                    poisoned_val_only_x_loader,
                    args,
                    rnd,
                    args.num_target,
                )
            else:
                test_model = copy.deepcopy(global_model)

                # CF
                for name, param in test_model.named_parameters():
                    mask = 0
                    for id, agent in enumerate(agents):
                        mask += old_mask[id][name].to(args.device)
                    param.data = torch.where(
                        mask.to(args.device) >= args.theta_ld,
                        param,
                        torch.zeros_like(param),
                    )
                val_acc = utils.get_loss_n_accuracy(
                    test_model, criterion, val_loader, args, rnd, args.num_target
                )
                asr = utils.get_loss_n_accuracy(
                    test_model,
                    criterion,
                    poisoned_val_loader,
                    args,
                    rnd,
                    args.num_target,
                )
                poison_acc = utils.get_loss_n_accuracy(
                    test_model,
                    criterion,
                    poisoned_val_only_x_loader,
                    args,
                    rnd,
                    args.num_target,
                )
                del test_model

            logging.info("Clean ACC:              %.4f" % val_acc)
            logging.info("Attack Success Ratio:   %.4f" % asr)
            logging.info("Backdoor ACC:           %.4f" % poison_acc)

            if val_acc > best_acc:
                best_acc = val_acc
                best_asr = asr
                best_bcdr_acc = poison_acc

        logging.info("------------------------------".format(rnd))
        
        # Save Checkpoint
        latest_path = os.path.join(args.log_dir, "checkpoint_latest.pt")
        
        # Always save latest
        save_dict = {
            'round': rnd,
            'state_dict': global_model.state_dict(),
            'best_acc': best_acc,
            'best_asr': best_asr,
            'best_bcdr_acc': best_bcdr_acc,
        }
        
        # Save IMS masks if available (for Config 22)
        if args.aggr == 'origin_alignins_clustering_prune_finetune':
            if hasattr(aggregator, 'ims_prune_aggregator') and aggregator.ims_prune_aggregator is not None:
                ims_agg = aggregator.ims_prune_aggregator
                if ims_agg.saved_mask_A is not None and ims_agg.saved_mask_S is not None:
                    save_dict['ims_mask_A'] = ims_agg.saved_mask_A
                    save_dict['ims_mask_S'] = ims_agg.saved_mask_S
                    logging.info(f"Saved IMS masks to checkpoint at round {rnd}")

        torch.save(save_dict, latest_path)

        # Save historical checkpoint based on frequency
        if rnd % args.save_freq == 0 or rnd == args.rounds:
            checkpoint_path = os.path.join(args.log_dir, f"checkpoint_{rnd}.pt")
            torch.save(save_dict, checkpoint_path)

        if args.aggr == "origin_alignins_clustering_prune_finetune":
            if (
                aggregator.ims_prune_aggregator is not None
                and aggregator.ims_prune_aggregator.has_pruned
                and rnd >= args.ims_start_round
            ):
                logging.info("IMS Prune completed at round %d. Stopping further aggregation.", rnd)
                break

    if args.aggr == "fedup_avg":
        logging.info("FedUP(avg): Starting post-training unlearning...")
        if fedup_last_global_params is None or fedup_last_updates is None:
            logging.warning("FedUP(avg): Missing cached last-round updates; skipping unlearning.")
        else:
            from fedup_unlearning import FedUPUnlearning

            base_model = copy.deepcopy(global_model).to(args.device)
            utils.vector_to_model(fedup_last_global_params.to(args.device), base_model)

            local_models = {}
            for cid, update in fedup_last_updates.items():
                m = copy.deepcopy(base_model).to(args.device)
                utils.vector_to_model((fedup_last_global_params + update).to(args.device), m)
                local_models[cid] = m

            malicious_ids = [cid for cid in local_models.keys() if cid < args.num_corrupt]
            benign_ids = [cid for cid in local_models.keys() if cid >= args.num_corrupt]
            if len(malicious_ids) == 0 or len(benign_ids) == 0:
                logging.warning(f"FedUP(avg): Need both malicious and benign models, got mal={len(malicious_ids)} benign={len(benign_ids)}; skipping.")
            else:
                fedup = FedUPUnlearning(
                    p_max=getattr(args, "fedup_p_max", 0.15),
                    p_min=getattr(args, "fedup_p_min", 0.01),
                    gamma=getattr(args, "fedup_gamma", 5),
                    rate_limit_threshold=5,
                )

                mask = fedup.generate_unlearning_mask(
                    local_models=local_models,
                    global_model=base_model,
                    malicious_client_ids=malicious_ids,
                    benign_client_ids=benign_ids,
                )

                benign_models = [local_models[cid] for cid in benign_ids]
                avg_benign_model = fedup._average_models(benign_models)

                unlearned_model = fedup.apply_unlearning_mask(avg_benign_model, mask)
                global_model.load_state_dict(unlearned_model.state_dict())

                logging.info("---------Test After FedUP(avg) Unlearning ------------")
                val_acc = utils.get_loss_n_accuracy(global_model, criterion, val_loader, args, args.rounds, args.num_target)
                asr = utils.get_loss_n_accuracy(global_model, criterion, poisoned_val_loader, args, args.rounds, num_classes=args.num_target)
                poison_acc = utils.get_loss_n_accuracy(global_model, criterion, poisoned_val_only_x_loader, args, args.rounds, args.num_target)
                logging.info("Post-FedUP Clean ACC:              %.4f" % val_acc)
                logging.info("Post-FedUP Attack Success Ratio:   %.4f" % asr)
                logging.info("Post-FedUP Backdoor ACC:           %.4f" % poison_acc)

                fedup_ckpt_path = os.path.join(args.log_dir, f"checkpoint_fedup_avg_unlearned_r{args.rounds}.pt")
                torch.save(
                    {
                        "round": args.rounds,
                        "state_dict": global_model.state_dict(),
                        "best_acc": best_acc,
                        "best_asr": best_asr,
                        "best_bcdr_acc": best_bcdr_acc,
                        "mode": "fedup_avg_post_unlearning",
                    },
                    fedup_ckpt_path,
                )
                logging.info(f"Saved FedUP(avg) unlearned checkpoint to {fedup_ckpt_path}")

    # ==========================================
    # NoT Unlearning Logic (Post-Training)
    # ==========================================
    if args.aggr in {"not_unlearning", "alignins_not_unlearning"}:
        logging.info("NoT Unlearning: Starting one-shot unlearning after training...")
        
        # 1. Negate the first conv layer
        target_name = None
        # Prioritize finding the first conv layer weights
        for name, _ in global_model.named_parameters():
            if ('conv' in name.lower()) and ('weight' in name.lower()):
                target_name = name
                break
        # Fallback to first parameter if no conv weight found
        if target_name is None:
            for name, _ in global_model.named_parameters():
                target_name = name
                break
                
        if target_name:
            logging.info("NoT Unlearning: Negating layer '%s'" % target_name)
            for name, param in global_model.named_parameters():
                if name == target_name:
                    param.data = -param.data
                    break
        else:
            logging.warning("NoT Unlearning: No suitable layer found for negation!")
        
        # 2. Fine-tuning with clean data
        # Use benign clients (ground truth) for fine-tuning
        if args.not_finetune_rounds <= 0:
             args.not_finetune_rounds = 5
             
        logging.info("NoT Unlearning: Starting fine-tuning for %d rounds..." % args.not_finetune_rounds)
        
        # Backup args
        orig_lr = args.client_lr
        orig_local_ep = args.local_ep
        
        args.client_lr = args.not_finetune_lr
        args.local_ep = args.not_finetune_local_ep
        
        # RELOAD a clean dataset to ensure no poisoning from previous agents affects fine-tuning
        logging.info("NoT Unlearning: Reloading clean dataset for fine-tuning...")
        clean_train_dataset, _ = utils.get_datasets(args.data)

        # Use full CIFAR-10 training data for fine-tuning
        full_train_loader = DataLoader(
            clean_train_dataset,
            batch_size=args.bs,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=False,
            drop_last=True
        )
        logging.info("NoT Unlearning: Fine-tuning using full CIFAR-10 training dataset.")
        
        # Create a dummy agent for fine-tuning with the full dataset
        # This agent will not have a specific user_group, but will use the full_train_loader
        dummy_agent = Agent(
            id=args.num_agents, # Use a safe ID > num_corrupt to behave as benign
            args=args,
            train_dataset=clean_train_dataset, # Pass the clean dataset
            data_idxs=list(range(len(clean_train_dataset))), # Dummy user_group for full dataset
            backdoor_train_dataset=None # No backdoor data for fine-tuning
        )
        dummy_agent.train_loader = full_train_loader # Assign the full train loader
        dummy_agent.n_data = len(clean_train_dataset) # Update data size
        dummy_agent.is_malicious = 0 # Not malicious
        
        for ft_rnd in range(1, args.not_finetune_rounds + 1):
            logging.info("NoT Unlearning: Starting fine-tuning round %d..." % ft_rnd)
            
            # Perform local training on the dummy agent with the full dataset
            update = dummy_agent.local_train(
                global_model, criterion, ft_rnd, neurotoxin_mask=neurotoxin_mask
            )
            
            # Apply the update to the global model
            cur_params = parameters_to_vector(
                [global_model.state_dict()[name] for name in global_model.state_dict()]
            )
            lr_vec = torch.tensor([args.server_lr] * len(cur_params), device=args.device)
            new_params = (cur_params + lr_vec * update).float()
            utils.vector_to_model(new_params, global_model)
            
            logging.info("NoT Unlearning: Fine-tuning round %d completed" % ft_rnd)

        # Restore args
        args.client_lr = orig_lr
        args.local_ep = orig_local_ep
        
        # Evaluate after unlearning & fine-tuning
        logging.info("---------Test After NoT Unlearning ------------")
        val_acc = utils.get_loss_n_accuracy(
            global_model, criterion, val_loader, args, args.rounds + args.not_finetune_rounds, args.num_target
        )
        asr = utils.get_loss_n_accuracy(
            global_model,
            criterion,
            poisoned_val_loader,
            args,
            args.rounds + args.not_finetune_rounds,
            num_classes=args.num_target,
        )
        poison_acc = utils.get_loss_n_accuracy(
            global_model,
            criterion,
            poisoned_val_only_x_loader,
            args,
            args.rounds + args.not_finetune_rounds,
            args.num_target,
        )
        logging.info("Post-Unlearning Clean ACC:            %.4f" % val_acc)
        logging.info("Post-Unlearning Attack Success Ratio: %.4f" % asr)
        logging.info("Post-Unlearning Backdoor ACC:         %.4f" % poison_acc)
        
        # Update best results if improved (optional, but unlearning usually drops performance initially)
        if val_acc > best_acc:
             best_acc = val_acc
             best_asr = asr
             best_bcdr_acc = poison_acc

    # ==========================================
    # IMS Prune + Finetune Logic (Moved to Aggregation)
    # ==========================================
    # The logic has been integrated into IMSPruneAggregator in agg_ims_prune.py
    # to avoid double execution and ensure consistency within the FL loop.

    logging.info("Best results:")
    logging.info("Clean ACC:              %.4f" % best_acc)
    logging.info("Attack Success Ratio:   %.4f" % best_asr)
    logging.info("Backdoor ACC:           %.4f" % best_bcdr_acc)
    logging.info("Training has finished!")
