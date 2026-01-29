# IMS Execution Steps (Configuration 9)

本文档总结了使用 `config_ims`（配置9）运行 IMS 防御算法的完整执行流程和对应代码位置。

## 1. 初始化配置
**文件**: [run_user_config.sh](run_user_config.sh)

在 `run_user_config.sh` 中，用户选择配置9调用 `config_ims` 函数。该函数设置了 `--aggr "ims"` 参数以及 IMS 相关的超参数。

- **位置**: `config_ims` 函数 (L226-L247)
- **关键参数**:
  - `--aggr "ims"`: 指定聚合方法为 IMS
  - `--ims_start_round`: 默认为 100 (在 `agg_ims.py` 中设定)
  - 其他参数: `ims_r1`, `ims_r2`, `ims_r3` (IMS 内部循环轮数)

## 2. 联邦学习主循环
**文件**: [src/federated.py](src/federated.py)

主程序启动后进入联邦学习的训练循环。每一轮（round）服务器分发模型给客户端，客户端本地训练后返回更新，服务器进行聚合。

- **位置**: 主循环 `for rnd in range(1, args.rounds + 1):` (L383)
- **聚合调用**: `aggregator.aggregate_updates(...)` (L423)
  - 这里调用聚合器处理收集到的 `agent_updates_dict`。

## 3. 聚合方法分发
**文件**: [src/aggregation.py](src/aggregation.py)

`Aggregation` 类根据 `args.aggr` 参数决定使用哪种聚合策略。

- **位置**: `aggregate_updates` 方法 (L32)
- **IMS 分支**:
  ```python
  elif self.args.aggr == 'ims':
      # IMS: Intelligent Mask Selection
      # Ensure auxiliary_data_loader is available
      data_loader = auxiliary_data_loader if auxiliary_data_loader is not None else self.auxiliary_data_loader
      aggregated_updates = self.agg_ims(agent_updates_dict, cur_global_params, global_model, data_loader, current_round=current_round)
  ```
  (L84-88)
- **调用实现**: `agg_ims` 方法 (L356-376)，进而导入并调用 `src/agg_ims.py` 中的 `agg_ims` 函数。

## 4. IMS 核心逻辑
**文件**: [src/agg_ims.py](src/agg_ims.py)

IMS 的核心逻辑由 `IMSOncePipeline` 类控制，该类实现了“一次性防御”策略（即在指定轮次触发防御）。

### 4.1 管道入口
- **位置**: `agg_ims` 函数 (L435) -> `agg_ims_once` (L506) -> `IMSOncePipeline.run` (L455)

### 4.2 触发条件判断
在 `IMSOncePipeline.run` 方法中，判断当前轮次是否达到触发防御的轮次（默认为第 100 轮）。

- **位置**: L463-481
- **逻辑**:
  - 如果 `current_round < start_round`: 执行标准 FedAvg (L473-479)。
  - 如果 `current_round >= start_round`: 执行 IMS 防御 (L481-493)。

### 4.3 防御执行 (`_defend_once`)
当触发防御时，调用 `_defend_once` 方法 (L495-504)，按顺序执行以下步骤：

1.  **掩码初始化 (`mask_initialization`)**:
    - 初始化掩码变量 A 和 S。
    - L500: `A_init, S_init = ims.mask_initialization(...)`
2.  **内层子问题 (`inner_subproblem`)**:
    - **目标**: 合成触发器（对抗性扰动 $\delta$），使得反向掩码模型在扰动输入下的输出尽可能接近原始模型在清洁输入下的输出（模拟后门行为）。
    - **过程**:
        - 遍历辅助数据集的每个批次。
        - 对每个样本初始化一个可学习的扰动 $\delta$。
        - 使用当前掩码 $A, S$ 计算反向掩码 $\bar{A}'$ 并构建反向掩码模型 `inverse_masked_model`。
        - 优化 $\delta$ 以最小化两个损失：
            - `disagree_loss`: 最大化扰动输入 $x+\delta$ 在原始模型上的预测差异（攻击有效性）。
            - `agree_loss`: 最小化扰动输入 $x+\delta$ 在反向掩码模型上的预测与原始模型清洁输出的一致性（隐藏后门）。
        - 保存每个批次的最佳 $\delta$ 用于外层循环。
    - **代码位置**: L280-350

3.  **外层子问题 (`outer_subproblem`)**:
    - **目标**: 优化掩码 $A, S$ 以消除后门，同时保持清洁准确率。
    - **过程**:
        - 初始化掩码优化器（AdamW）。
        - 遍历辅助数据集和对应的扰动 $\delta$。
        - 计算当前掩码 $A'$ 并构建掩码模型 `masked_model`。
        - 计算四个损失项：
            - `clean_agree_loss`: 掩码模型在清洁数据上的输出应接近原始模型（保持 ACC）。
            - `backdoor_recover_loss`: 掩码模型在扰动数据上的输出应接近原始模型清洁输出（消除后门）。
            - `backdoor_valid_loss`: 原始模型在扰动数据上的输出与清洁输出的差异（验证后门存在性，作为参考）。
            - `reg_loss`: 掩码稀疏性正则化（L1 范数）。
        - 更新掩码 $A, S$。
        - 逐渐增加正则化系数 `lambda` 以鼓励稀疏性。
    - **代码位置**: L352-410
4.  **部署防御 (`deploy_defense`)**:
    - 将最终掩码应用到模型权重上（剪枝）。
    - L503: `defended_model = ims.deploy_defense(...)`

最终返回防御后的模型更新向量 `effective_update`。
