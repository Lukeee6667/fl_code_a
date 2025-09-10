# 改进的AlignIns-FedUP混合防御方法

## 问题分析

### 原始混合方法的问题

1. **检测过于严格**: AlignIns使用严格的交集操作，导致大部分客户端被标记为可疑
2. **FedUP剪枝很少执行**: 由于AlignIns已经过滤掉大部分客户端，FedUP的剪枝逻辑很少被触发
3. **二元分类局限**: 只区分"良性"和"可疑"，缺乏细粒度的处理策略
4. **权重分配不合理**: 所有选中的客户端权重相同，没有考虑可信度差异

### 实验结果相同的根本原因

`alignins_fedup_hybrid` 和 `alignins_g_v2_onepic` 产生相同结果是因为：
- 两者使用完全相同的四个检测指标（TDA、MPSA、梯度范数、余弦相似度）
- 相同的MZ-score计算和阈值判断逻辑
- 相同的交集操作选择良性客户端
- FedUP剪枝在AlignIns完全过滤后无法执行
- 最终都使用相同的平均聚合方式

## 改进方案

### 核心思想

**从提高Clean Accuracy和降低Attack Success Ratio的目标出发，设计更合理的混合策略：**

1. **保留更多良性客户端** → 提高Clean Accuracy
2. **精准识别和处理恶意客户端** → 降低Attack Success Ratio
3. **对不确定客户端进行剪枝而非排除** → 平衡两个目标

### 改进策略

#### 1. 三层检测策略

```
严格阈值 ←→ 原始阈值 ←→ 宽松阈值
    ↓           ↓           ↓
明确良性    可疑客户端    明确恶意
(权重=1.0)  (剪枝+降权)   (完全排除)
```

**第一层 - 明确良性客户端**:
- 使用 `0.8 × 原始阈值` 的严格标准
- 通过所有四个指标检测的客户端
- 权重 = 1.0，直接参与聚合

**第二层 - 明确恶意客户端**:
- 使用 `1.5 × 原始阈值` 的宽松标准
- 在任意一个指标上超过宽松阈值
- 完全排除，不参与聚合

**第三层 - 可疑客户端**:
- 介于严格和宽松阈值之间
- 应用FedUP个性化剪枝
- 权重 = 0.1~0.6，基于可疑程度动态调整

#### 2. 个性化剪枝策略

**可疑程度计算**:
```python
suspicion_score = Σ(max(0, mzscore[i] - strict_threshold[i]) / (loose_threshold[i] - strict_threshold[i])) / 4
```

**个性化剪枝比例**:
```python
combined_factor = (1 - benign_similarity) × 0.3 + suspicion_score × 0.7
adaptive_pruning_ratio = p_min + (p_max - p_min) × (combined_factor ^ gamma)
```

**权重重要性计算**:
```python
weight_importance = |client_update - global_model| × |global_model|
```

#### 3. 智能加权聚合

**权重分配策略**:
- 明确良性客户端: `weight = 1.0`
- 可疑客户端: `weight = 0.6 - 0.5 × suspicion_score`
- 明确恶意客户端: `weight = 0` (排除)

**加权聚合公式**:
```python
aggregated_update = Σ(weight[i] × update[i]) / Σ(weight[i])
```

**自适应范数裁剪**:
- 使用加权中位数而非简单中位数
- 考虑客户端权重的影响

## 参数配置指南

### 场景化参数设置

#### IID场景 (数据分布均匀)
```bash
# 较宽松的检测阈值
--lambda_s 2.5 --lambda_c 2.5 --lambda_g 2.0 --lambda_mean_cos 2.0
# 中等剪枝强度
--fedup_p_max 0.6 --fedup_p_min 0.05 --fedup_gamma 3
```

#### Non-IID场景 (数据分布不均)
```bash
# 平衡的检测阈值
--lambda_s 2.0 --lambda_c 2.0 --lambda_g 1.8 --lambda_mean_cos 1.8
# 较强剪枝强度
--fedup_p_max 0.8 --fedup_p_min 0.1 --fedup_gamma 3
```

#### 强攻击场景 (高攻击比例)
```bash
# 较严格的检测阈值
--lambda_s 1.5 --lambda_c 1.5 --lambda_g 1.5 --lambda_mean_cos 1.5
# 最强剪枝强度
--fedup_p_max 1.0 --fedup_p_min 0.15 --fedup_gamma 2
```

### 参数调优策略

#### 提高Clean Accuracy
1. **增大检测阈值** (`lambda_*`): 保留更多客户端
2. **降低剪枝强度** (`p_max`, `p_min`): 减少对可疑客户端的损害
3. **增大gamma值**: 使剪枝更保守

#### 降低Attack Success Ratio
1. **减小检测阈值** (`lambda_*`): 更严格地筛选客户端
2. **增大剪枝强度** (`p_max`, `p_min`): 更激进地剪枝可疑客户端
3. **减小gamma值**: 使剪枝更敏感

#### 平衡策略
- **动态阈值**: 根据检测到的攻击强度自动调整
- **自适应权重**: 基于历史表现调整客户端权重
- **多轮验证**: 结合多轮的检测结果

## 使用方法

### 基本使用

```bash
python federated.py \
  --poison_frac 0.3 \
  --num_corrupt 12 \
  --num_agents 40 \
  --aggr alignins_fedup_hybrid \
  --data cifar10 \
  --attack badnet \
  --non_iid --beta 0.5 \
  --lambda_s 2.0 --lambda_c 2.0 \
  --lambda_g 1.8 --lambda_mean_cos 1.8 \
  --fedup_p_max 0.8 --fedup_p_min 0.1 --fedup_gamma 3
```

### 批量实验

```bash
python run_improved_hybrid_example.py
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lambda_s` | 2.0 | MPSA检测阈值 |
| `lambda_c` | 2.0 | TDA检测阈值 |
| `lambda_g` | 1.8 | 梯度范数检测阈值 |
| `lambda_mean_cos` | 1.8 | 平均余弦相似度检测阈值 |
| `fedup_p_max` | 0.8 | 最大剪枝比例 |
| `fedup_p_min` | 0.1 | 最小剪枝比例 |
| `fedup_gamma` | 3 | 剪枝敏感度参数 |

## 预期效果

### 性能提升

1. **Clean Accuracy提升**:
   - 保留更多良性客户端参与训练
   - 减少对良性客户端的误伤
   - 智能权重分配优化聚合质量

2. **Attack Success Ratio降低**:
   - 精准识别和排除明确恶意客户端
   - 对可疑客户端进行有效剪枝
   - 动态调整防御强度

3. **鲁棒性增强**:
   - 适应不同攻击强度和数据分布
   - 自动平衡防御效果和模型性能
   - 减少超参数敏感性

### 理论优势

1. **更细粒度的客户端分类**: 三层检测比二元分类更精确
2. **个性化处理策略**: 根据可疑程度采用不同处理方式
3. **智能权重机制**: 考虑客户端可信度的差异
4. **自适应防御强度**: 根据攻击情况动态调整

## 实验验证

### 对比基线

- **AlignIns**: 原始多指标检测方法
- **FedUP**: 原始自适应剪枝方法
- **原始混合方法**: 简单结合AlignIns和FedUP
- **改进混合方法**: 本文提出的三层检测+个性化剪枝+智能聚合

### 评估指标

- **Clean Accuracy**: 在干净测试集上的准确率
- **Attack Success Ratio**: 后门攻击的成功率
- **True Positive Rate (TPR)**: 正确识别良性客户端的比例
- **False Positive Rate (FPR)**: 错误识别良性客户端为恶意的比例
- **Weighted TPR/FPR**: 考虑权重影响的加权指标

### 实验场景

1. **不同攻击类型**: BadNet, DBA, Semantic攻击
2. **不同攻击强度**: 10%-50%的恶意客户端比例
3. **不同数据分布**: IID和Non-IID (β=0.1, 0.3, 0.5, 1.0)
4. **不同数据集**: CIFAR-10, CIFAR-100, MNIST

## 总结

改进的混合方法通过**三层检测策略**、**个性化剪枝**和**智能加权聚合**，有效解决了原始方法的局限性。从提高Clean Accuracy和降低Attack Success Ratio的目标出发，该方法能够：

1. **更好地保护良性客户端**: 避免过度排除，保持模型训练质量
2. **更精准地处理恶意客户端**: 结合检测和剪枝，提高防御效果
3. **更灵活地适应不同场景**: 通过参数调整适应各种攻击和数据分布

这种设计理念更符合联邦学习的实际需求，在安全性和可用性之间取得了更好的平衡。