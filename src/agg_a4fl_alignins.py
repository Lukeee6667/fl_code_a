import torch
import logging
from typing import Dict, List
from alignins_detector import AlignInsDetector
from agg_a4fl import A4FL_Aggregator

def agg_a4fl_alignins(agent_updates_dict, flat_global_model, global_model, args, auxiliary_loader, agent_data_sizes):
    """
    A4FL + AlignIns 混合聚合策略
    
    步骤:
    1. AlignIns 检测 (Phase 1): 使用四指标检测过滤掉明显的恶意客户端。
    2. A4FL 校验 (Phase 2): 对剩余客户端使用辅助数据进行 Loss/Acc 校验。
    3. 最终聚合: 聚合通过两层筛选的更新。
    """
    logging.info("=== A4FL + AlignIns Hybrid Aggregation ===")
    
    # -------------------------------------------------------------------------
    # 步骤 1: AlignIns 初步筛选 (无监督)
    # -------------------------------------------------------------------------
    logging.info("[Phase 1] AlignIns Detection...")
    
    # 准备 AlignIns 所需的输入 (Stack updates to tensor)
    # 注意：需要保持 client_id 的映射关系
    client_ids = list(agent_updates_dict.keys())
    inter_model_updates = torch.stack([agent_updates_dict[cid] for cid in client_ids])
    
    # 初始化检测器
    detector = AlignInsDetector(args)
    
    # 执行检测
    detection_results = detector.detect(inter_model_updates)
    
    # 获取 AlignIns 认为 "恶意" 的索引
    malicious_indices = detection_results['malicious']
    malicious_clients_alignins = {client_ids[i] for i in malicious_indices}
    
    # 获取 AlignIns 认为 "良性" 和 "可疑" 的索引 (保留进入下一阶段)
    # 注意：这里我们比较严格，如果AlignIns认为是恶意，直接剔除
    # 如果是 "suspicious"，我们保留给 A4FL 进一步判断
    kept_indices = detection_results['benign'] | detection_results['suspicious']
    kept_clients_alignins = {client_ids[i] for i in kept_indices}
    
    logging.info(f"AlignIns Filtered: {len(malicious_clients_alignins)} malicious clients.")
    logging.info(f"Clients passing Phase 1: {len(kept_clients_alignins)} (Benign + Suspicious)")
    
    if len(kept_clients_alignins) == 0:
        logging.warning("AlignIns filtered ALL clients! Returning zero update.")
        return torch.zeros_like(flat_global_model)

    # -------------------------------------------------------------------------
    # 步骤 2: A4FL 辅助数据校验 (有监督)
    # -------------------------------------------------------------------------
    logging.info("[Phase 2] A4FL Statistical Validation...")
    
    # 构造仅包含 Phase 1 通过者的 updates dict
    filtered_updates_dict = {cid: agent_updates_dict[cid] for cid in kept_clients_alignins}
    
    # 初始化 A4FL 聚合器
    a4fl_aggregator = A4FL_Aggregator(args)
    
    if auxiliary_loader is None:
        logging.warning("No auxiliary_loader provided for A4FL! Falling back to average of AlignIns survivors.")
        # Fallback: simple average of survivors
        total_weight = 0
        accumulated_update = torch.zeros_like(flat_global_model)
        for cid, update in filtered_updates_dict.items():
            w = agent_data_sizes[cid]
            accumulated_update += update * w
            total_weight += w
        return accumulated_update / total_weight
        
    # 调用 A4FL 的 aggregate 方法
    # A4FL 内部会进行辅助数据测试和统计过滤
    # 注意：A4FL.aggregate 返回 (aggregated_update, extra_info)
    aggregated_update, _ = a4fl_aggregator.aggregate(
        filtered_updates_dict, 
        global_model, 
        auxiliary_loader, 
        agent_data_sizes
    )
    
    logging.info("Hybrid Aggregation Completed.")
    
    return aggregated_update
