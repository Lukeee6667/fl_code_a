# AlignIns-FedUP混合防御方法

## 概述

AlignIns-FedUP混合方法结合了AlignIns的多指标异常检测能力和FedUP的自适应剪枝策略，提供了更强大和灵活的联邦学习防御机制。

## 核心特性

### 1. 双重异常检测机制
- **AlignIns多指标检测**: 基于梯度相似性、更新幅度、层级敏感度等多个指标
- **FedUP自适应检测**: 基于可疑客户端更新的统计特征
- **融合策略**: 综合两种检测结果，提高检测准确性

### 2. 自适应剪枝策略
- **动态剪枝比例**: 根据检测到的异常程度自适应调整
- **基于排名的权重选择**: 优先保护重要参数
- **渐进式防御**: 随着训练进行逐步增强防御强度

### 3. 鲁棒性聚合
- **选择性参数更新**: 只对可疑参数进行剪枝
- **权重平衡**: 在防御效果和模型性能间找到平衡
- **多轮次适应**: 根据历史信息调整防御策略

## 算法流程

### 第一阶段：AlignIns异常检测
1. **梯度相似性分析**
   - 计算客户端更新间的余弦相似度
   - 识别异常相似或异常不同的更新

2. **更新幅度检测**
   - 分析更新向量的L2范数
   - 检测异常大或异常小的更新

3. **层级敏感度分析**
   - 计算不同层的敏感度差异
   - 识别针对特定层的攻击

### 第二阶段：FedUP自适应剪枝
1. **可疑客户端识别**
   - 基于统计特征检测异常更新
   - 计算更新重要性分数

2. **自适应剪枝比例计算**
   ```
   P(r) = P_min + (P_max - P_min) * (1 - exp(-γ * r / R))
   ```
   其中：
   - P(r): 第r轮的剪枝比例
   - P_max, P_min: 最大和最小剪枝率
   - γ: 曲线陡度参数
   - R: 总训练轮数

3. **基于排名的权重选择**
   - 对参数重要性进行排名
   - 选择性剪枝低重要性参数

### 第三阶段：融合与聚合
1. **检测结果融合**
   ```
   Score_final = α * Score_AlignIns + β * Score_FedUP
   ```

2. **自适应权重调整**
   - 根据检测置信度动态调整权重
   - 考虑历史检测准确性

3. **鲁棒性聚合**
   - 应用混合掩码进行选择性聚合
   - 保持模型收敛性

## 参数配置

### 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `sensitivity_threshold` | 0.1 | AlignIns异常检测敏感度阈值 |
| `p_max` | 0.8 | 最大剪枝率 |
| `p_min` | 0.1 | 最小剪枝率 |
| `gamma` | 2.0 | 自适应曲线陡度参数 |
| `alignins_weight` | 0.6 | AlignIns检测权重 |
| `fedup_weight` | 0.4 | FedUP检测权重 |

### 场景化配置建议

#### IID数据分布
```bash
--sensitivity_threshold=0.05 \
--p_max=0.6 \
--p_min=0.05 \
--gamma=1.5 \
--alignins_weight=0.7 \
--fedup_weight=0.3
```

#### Non-IID数据分布
```bash
--sensitivity_threshold=0.15 \
--p_max=0.8 \
--p_min=0.1 \
--gamma=2.0 \
--alignins_weight=0.6 \
--fedup_weight=0.4
```

#### 强攻击场景
```bash
--sensitivity_threshold=0.2 \
--p_max=0.9 \
--p_min=0.2 \
--gamma=3.0 \
--alignins_weight=0.5 \
--fedup_weight=0.5
```

## 使用方法

### 基本使用

```bash
python run_alignins_fedup_hybrid_example.py \
    --dataset=cifar10 \
    --model=resnet18 \
    --num_agents=100 \
    --num_corrupt=20 \
    --attack_type=badnet \
    --rounds=200
```

### 高级配置

```bash
python run_alignins_fedup_hybrid_example.py \
    --dataset=cifar100 \
    --model=resnet34 \
    --num_agents=200 \
    --num_corrupt=40 \
    --attack_type=dba \
    --rounds=300 \
    --sensitivity_threshold=0.12 \
    --p_max=0.85 \
    --p_min=0.15 \
    --gamma=2.5 \
    --alignins_weight=0.65 \
    --fedup_weight=0.35 \
    --iid=0 \
    --alpha=0.3
```

## 实验配置示例

### 轻量级防御（适用于良性环境）
```bash
python src/federated.py \
    --dataset=cifar10 \
    --model=resnet18 \
    --num_agents=50 \
    --num_corrupt=5 \
    --attack_type=badnet \
    --rounds=100 \
    --aggr=alignins_fedup_hybrid \
    --sensitivity_threshold=0.05 \
    --p_max=0.5 \
    --p_min=0.05 \
    --gamma=1.0 \
    --alignins_weight=0.8 \
    --fedup_weight=0.2
```

### 标准防御（平衡性能与安全）
```bash
python src/federated.py \
    --dataset=cifar10 \
    --model=resnet18 \
    --num_agents=100 \
    --num_corrupt=20 \
    --attack_type=badnet \
    --rounds=200 \
    --aggr=alignins_fedup_hybrid \
    --sensitivity_threshold=0.1 \
    --p_max=0.8 \
    --p_min=0.1 \
    --gamma=2.0 \
    --alignins_weight=0.6 \
    --fedup_weight=0.4
```

### 强化防御（高威胁环境）
```bash
python src/federated.py \
    --dataset=cifar100 \
    --model=resnet34 \
    --num_agents=200 \
    --num_corrupt=50 \
    --attack_type=dba \
    --rounds=300 \
    --aggr=alignins_fedup_hybrid \
    --sensitivity_threshold=0.2 \
    --p_max=0.95 \
    --p_min=0.2 \
    --gamma=3.0 \
    --alignins_weight=0.5 \
    --fedup_weight=0.5
```

## 性能特点

### 优势
1. **高检测准确性**: 双重检测机制减少误报和漏报
2. **自适应性强**: 根据攻击强度动态调整防御策略
3. **鲁棒性好**: 对多种攻击类型都有良好的防御效果
4. **收敛性保证**: 在防御的同时保持模型收敛

### 适用场景
1. **混合攻击环境**: 面临多种类型攻击的场景
2. **动态威胁**: 攻击强度随时间变化的环境
3. **高安全要求**: 对防御效果要求较高的应用
4. **Non-IID数据**: 数据分布不均匀的联邦学习场景

### 计算开销
- **检测阶段**: O(n²) 相似度计算 + O(n) 统计分析
- **剪枝阶段**: O(p log p) 参数排序 + O(p) 掩码生成
- **聚合阶段**: O(p) 加权聚合

其中 n 为客户端数量，p 为模型参数数量。

## 调优建议

### 参数调优策略

1. **敏感度阈值调优**
   - 从0.05开始，根据误报率调整
   - IID环境可适当降低，Non-IID环境需提高

2. **剪枝参数调优**
   - p_max: 根据攻击强度调整，强攻击需要更高值
   - p_min: 保证基本防御能力，通常不低于0.05
   - gamma: 控制自适应速度，快速收敛场景可增大

3. **权重平衡调优**
   - AlignIns适合检测复杂攻击模式
   - FedUP适合检测统计异常
   - 根据主要威胁类型调整权重比例

### 性能监控指标

1. **防御效果**
   - 攻击成功率 (ASR)
   - 后门准确率 (BA)
   - 主任务准确率 (MA)

2. **检测性能**
   - 真正率 (TPR)
   - 假正率 (FPR)
   - F1分数

3. **收敛性能**
   - 收敛轮数
   - 最终准确率
   - 训练稳定性

## 故障排除

### 常见问题

1. **检测率过低**
   - 降低sensitivity_threshold
   - 增加AlignIns权重
   - 检查数据分布是否过于不均匀

2. **误报率过高**
   - 提高sensitivity_threshold
   - 降低p_max值
   - 调整权重平衡

3. **收敛速度慢**
   - 降低gamma值
   - 减小p_min值
   - 检查学习率设置

4. **内存占用过高**
   - 减少相似度计算频率
   - 使用梯度检查点
   - 调整批次大小

### 调试模式

启用详细日志：
```bash
--log_level=DEBUG
```

这将输出详细的检测和剪枝信息，帮助分析算法行为。

## 扩展性

### 自定义检测指标

可以在`agg_alignins_fedup_hybrid`方法中添加新的检测指标：

```python
# 添加自定义检测逻辑
custom_scores = self.custom_detection_method(agent_updates_dict)
final_scores = alpha * alignins_scores + beta * fedup_scores + gamma * custom_scores
```

### 动态权重调整

可以实现基于历史性能的动态权重调整：

```python
# 根据历史检测准确性调整权重
if historical_alignins_accuracy > historical_fedup_accuracy:
    alignins_weight *= 1.1
    fedup_weight *= 0.9
```

## 引用

如果您在研究中使用了此混合方法，请引用相关论文：

```bibtex
@article{alignins2024,
  title={AlignIns: Robust Federated Learning against Backdoor Attacks},
  author={...},
  journal={...},
  year={2024}
}

@article{fedup2024,
  title={FedUP: Federated Unlearning with Adaptive Pruning},
  author={...},
  journal={...},
  year={2024}
}
```