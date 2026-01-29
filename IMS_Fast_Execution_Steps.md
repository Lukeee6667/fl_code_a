# IMS Fast Execution Steps (Configuration 15)

本文档总结了使用 `config_ims_fast`（配置15）运行优化版 IMS 防御算法的完整执行流程和对应代码位置。该版本主要针对内存占用和计算速度进行了优化。

## 1. 初始化配置
**文件**: [run_user_config.sh](run_user_config.sh)

在 `run_user_config.sh` 中，用户选择配置15调用 `config_ims_fast` 函数。该函数设置了 `--aggr "ims_fast"` 参数。

- **位置**: `config_ims_fast` 函数 (L400-L422)
- **关键参数**:
  - `--aggr "ims_fast"`: 指定聚合方法为优化版 IMS
  - `--ims_start_round`: 默认为 100 (在 `agg_ims_fast.py` 中设定)

## 2. 联邦学习主循环与聚合分发
**文件**: [src/federated.py](src/federated.py) & [src/aggregation.py](src/aggregation.py)

主循环逻辑与原始 IMS 相同。在 `Aggregation` 类中，根据参数分发到 `agg_ims_fast` 方法。

- **位置**: `Aggregation.aggregate_updates` (L32)
- **IMS Fast 分支**:
  ```python
  elif self.args.aggr == 'ims_fast':
      # IMS Fast: Optimized Intelligent Mask Selection
      data_loader = auxiliary_data_loader if auxiliary_data_loader is not None else self.auxiliary_data_loader
      aggregated_updates = self.agg_ims_fast(agent_updates_dict, cur_global_params, global_model, data_loader, current_round=current_round)
  ```
  (L89-92)
- **调用实现**: `agg_ims_fast` 方法 (L378-398)，进而调用 `src/agg_ims_fast.py`。

## 3. IMS Fast 核心逻辑与优化
**文件**: [src/agg_ims_fast.py](src/agg_ims_fast.py)

优化版的核心在于引入了 `MaskContext` 和输出缓存机制，避免了昂贵的 `deepcopy` 和重复的前向传播。

### 3.1 管道入口与触发逻辑
- **位置**: `agg_ims_fast` 函数 (L409) -> `IMSAggregator.aggregate` (L359)
- **触发条件**: 同样检查 `current_round >= start_round` (L371-379)，未触发时返回标准 FedAvg 更新。

### 3.2 关键优化组件

#### A. 上下文管理器 (`MaskContext`)
- **目标**: 消除 `copy.deepcopy(model)` 的内存和时间开销。
- **原理**: 
    - `__enter__`: 保存原始权重，将掩码应用后的权重直接赋值给模型（In-place 修改）。
    - `__exit__`: 恢复原始权重，确保模型状态无副作用。
- **代码位置**: `MaskContext` 类 (L15-60)

#### B. 清洁输出缓存 (`Clean Output Caching`)
- **目标**: 消除内层/外层循环中重复计算清洁模型输出的时间开销。
- **原理**: 
    - 利用辅助数据集加载器的确定性（`shuffle=False`），以 `batch_idx` 为键缓存模型输出。
    - 如果缓存中存在，直接使用；否则计算并存入缓存。
- **代码位置**: 在 `mask_initialization` (L145-149) 和 `outer_subproblem` (L279-282) 中实现。

### 3.3 防御执行流程

1.  **掩码初始化 (`mask_initialization`)**:
    - 使用 `MaskContext` 应用掩码，避免模型复制。
    - 缓存清洁样本的输出 `p`。
    - **代码位置**: L118-187
2.  **内层子问题 (`inner_subproblem`)**:
    - **目标**: 合成触发器 $\delta$。
    - **优化**: 
        - 使用 `temp_model` (浅拷贝+固定权重) 配合 `apply_mask` 生成反向掩码模型，减少计算图开销。
        - 同样利用确定性 DataLoader 顺序生成和存储 $\delta$。
    - **代码位置**: L250-L318
3.  **外层子问题 (`outer_subproblem`)**:
    - **目标**: 优化掩码 $A, S$。
    - **优化**: 
        - 缓存清洁输出 `p` 和对抗样本的清洁模型输出 `p_hat` (因为模型参数固定，仅掩码变化)。
        - 使用 `MaskContext` 动态应用当前优化的掩码 $A'$，无需每次迭代都复制模型。
    - **代码位置**: L320-378
4.  **部署防御 (`deploy_defense`)**:
    - 最终计算掩码并永久修改模型权重（剪枝）。
    - **代码位置**: L380-398

最终计算并返回归一化的更新向量 `effective_update`。
