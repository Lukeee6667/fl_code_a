# 联邦学习鲁棒防御框架：AlignIns 与对抗式掩码优化
## (AlignIns + Adversarial Mask Optimization)

本技术文档详细描述了结合 **AlignIns 异常检测** 与 **对抗式掩码优化 (Adversarial Mask Optimization, AMO)** 的联邦学习防御框架。该框架旨在非独立同分布 (Non-IID) 数据和高比例投毒攻击场景下，保障模型的安全性与可用性。

---

### 1. 整体框架 (Overall Framework)

本框架采用“检测-防御-恢复”的三阶段策略：

1.  **检测与预聚合 (Detection & Pre-aggregation)**: 使用 AlignIns 算法对客户端上传的梯度更新进行多维统计分析，识别并剔除恶意更新，生成初步的全局模型。
2.  **对抗式掩码优化 (Adversarial Mask Optimization, AMO)**: 在初步聚合的模型上，通过双层优化问题 (Bi-level Optimization) 寻找最优神经元掩码。内层生成对抗扰动，外层优化掩码以过滤后门路径。
3.  **梯度动量恢复 (Gradient Momentum Recovery)**: 针对 Non-IID 场景导致的梯度噪声问题，引入基于分位数的梯度动量追踪机制，动态“复活”被误剪枝的关键良性神经元。

### 2. 核心创新点 (Core Innovations)

*   **多维统计异常检测 (AlignIns)**:
    *   结合 KL 散度、余弦相似度及图结构特征，构建高维特征空间，有效区分 Non-IID 造成的良性差异与投毒攻击造成的恶意差异。
*   **对抗式掩码优化 (Adversarial Mask Optimization)**:
    *   替代传统的静态剪枝或启发式剪枝，采用 Min-Max 博弈思想。通过模拟最强攻击（内层循环）来训练最强防御掩码（外层循环），从结构层面物理切断后门神经元通路。
*   **梯度动量恢复机制 (Mask Recovery)**:
    *   **创新点**: 解决了在 Non-IID 数据上进行模型剪枝容易导致良性性能崩塌的难题。
    *   **机制**: 利用动量平滑梯度噪声，并使用基于层级统计量的动态分位数阈值（Top-k Quantile），精准识别并恢复对良性任务至关重要的神经元，即使它们在初始阶段被错误标记为冗余。

### 3. 模块位置与代码对应 (Module Locations)

*   **框架入口与配置**:
    *   文件: [`run_user_config.sh`](file:///c:/Users/王昭懿/Desktop/fl_code/origin_code/AlignIns/run_user_config.sh#L419)
    *   配置项: `config_alignins_ims_recover` (Config 14)
*   **AlignIns 检测模块**:
    *   文件: [`src/alignins_detector.py`](file:///c:/Users/王昭懿/Desktop/fl_code/origin_code/AlignIns/src/alignins_detector.py)
    *   功能: 计算异常分数，执行加权聚合。
*   **对抗式掩码优化核心 (AMO)**:
    *   文件: [`src/agg_ims_recover.py`](file:///c:/Users/王昭懿/Desktop/fl_code/origin_code/AlignIns/src/agg_ims_recover.py)
    *   **内层循环 (Inner Loop)**: [`inner_subproblem`](file:///c:/Users/王昭懿/Desktop/fl_code/origin_code/AlignIns/src/agg_ims_recover.py#L190) - 生成对抗扰动。
    *   **外层循环与恢复 (Outer Loop + Recovery)**: [`outer_subproblem_with_recovery`](file:///c:/Users/王昭懿/Desktop/fl_code/origin_code/AlignIns/src/agg_ims_recover.py#L224)
        *   **核心恢复逻辑**: 位于 L274-310，实现了基于动量和分位数的神经元复活。

### 4. 过程详解 (Process Description)

1.  **AlignIns 过滤**: 服务器接收 $N$ 个客户端更新。AlignIns 计算每对更新的一致性矩阵，识别出 $M$ 个潜在恶意客户端并降低其聚合权重。
    *   代码位置: `src/federated.py` 调用 `AlignIns` 类。
2.  **掩码初始化**: 对聚合后的候选模型，识别所有卷积层和全连接层，初始化掩码参数 $A$ (Mask values) 和 $S$ (Scaling factors)。
    *   代码位置: `agg_ims_recover.py` -> `mask_initialization`
3.  **内层对抗 (Inner Loop)**:
    *   固定掩码，利用辅助数据生成扰动 $\delta$。
    *   目标: 最大化模型预测与真实标签的差异 (模拟触发器攻击)。
    *   代码位置: `agg_ims_recover.py` -> `inner_subproblem`
4.  **外层优化与恢复 (Outer Loop with Recovery)**:
    *   固定 $\delta$，更新掩码 $A, S$。
    *   目标: 最小化 `Clean Loss` + `Recovery Loss` (在扰动下恢复正确标签) + `Sparsity Penalty` (稀疏正则化)。
    *   **恢复操作**:
        *   计算掩码梯度 $g_A$。
        *   更新梯度动量 $m_t = \beta m_{t-1} + (1-\beta) g_A$。
        *   计算当前层动量幅值的 90% 分位数阈值 $T_{90}$。
        *   若某被剪枝神经元 ($a < 0.1$) 的动量 $m < -T_{90}$ (强负梯度，指示需增加权重)，则强制将其值重置为 0.2。
    *   代码位置: `agg_ims_recover.py` -> `outer_subproblem_with_recovery`

### 5. 问题分析：为什么直接在污染模型上运行 IMS 会失败？

直接在严重污染的模型上（即跳过或AlignIns失效）运行 IMS 导致失败的原因主要有三点：

1.  **初始锚点偏移 (Initialization Bias)**: IMS 的优化目标包含保持模型在清洁数据上的性能。如果模型本身已被严重污染，其“正常功能”可能已经依赖于部分后门神经元。IMS 为了维持准确率，可能会错误地锁定并保护这些后门神经元。
2.  **梯度信号混淆 (Gradient Confusion)**: 在 Non-IID 场景下，良性梯度的方差极大，而后门梯度的方向通常非常一致且强烈（强信号）。简单的梯度下降或剪枝容易保留强信号（后门），而剪掉弱信号（良性长尾特征），导致“劣币驱逐良性币”。
3.  **对抗扰动失效**: 内层循环生成的扰动 $\delta$ 旨在模拟最坏情况。如果模型已经被植入特定触发器，生成的 $\delta$ 可能收敛到该触发器模式。此时，外层优化面临一个悖论：为了在 $\delta$ 存在时输出正确标签，模型需要“适应”这个触发器，这反而可能加固了后门逻辑，而不是消除它。

---
**总结**: 本框架通过 AlignIns 的前置过滤减轻初始污染，并通过改进的 IMS (含动量恢复) 在高噪声环境下精确剔除残留后门，形成了闭环防御体系。
