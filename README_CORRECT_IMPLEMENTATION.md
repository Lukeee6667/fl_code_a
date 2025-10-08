# 正确的AlignIns + FedUP实现

## 概述

这个实现修正了之前版本中的概念性错误，提供了真正符合原论文的AlignIns + FedUP方法：

1. **第一阶段：AlignIns四指标检测和过滤聚合**
   - 使用四种指标（余弦相似度、欧氏距离、符号一致性、幅度比）检测异常客户端
   - 将客户端分为良性、可疑、恶意三类
   - 过滤掉恶意客户端，降低可疑客户端权重，然后进行聚合

2. **第二阶段：标准FedUP模型权重剪枝**
   - 对聚合后的全局模型参数进行剪枝
   - 基于权重重要性计算，移除被恶意客户端污染的权重
   - 使用自适应剪枝比例

## 主要文件

### 核心实现
- `agg_alignins_fedup_correct.py`: 正确的AlignIns+FedUP实现
- `src/agg_alignins_fedup_correct.py`: 源码目录中的实现文件
- `src/aggregation.py`: 已更新，包含新的聚合方法调用

### 运行脚本
- `run_alignins_fedup_correct_example.py`: 运行示例脚本

## 使用方法

### 1. 基本运行

```bash
python run_alignins_fedup_correct_example.py
```

### 2. 自定义参数运行

```bash
python run_alignins_fedup_correct_example.py \
    --dataset cifar10 \
    --model resnet18 \
    --num_agents 100 \
    --num_corrupt 20 \
    --attack_type badnet \
    --rounds 200 \
    --local_epochs 5 \
    --lr 0.01 \
    --fedup_pruning_ratio 0.1 \
    --alignins_strict_threshold 0.7 \
    --alignins_standard_threshold 0.85
```

### 3. 参数说明

#### 基础参数
- `--dataset`: 数据集选择 (cifar10, cifar100, mnist, fmnist)
- `--model`: 模型架构 (resnet18, resnet34, vgg16, lenet)
- `--num_agents`: 客户端总数
- `--num_corrupt`: 恶意客户端数量
- `--attack_type`: 攻击类型 (badnet, dba, neurotoxin, semantic)

#### 联邦学习参数
- `--rounds`: 联邦学习轮数
- `--local_epochs`: 本地训练轮数
- `--lr`: 学习率
- `--batch_size`: 批次大小

#### AlignIns参数
- `--alignins_strict_threshold`: 严格阈值（分位数，默认0.7）
- `--alignins_standard_threshold`: 标准阈值（分位数，默认0.85）
- `--suspicious_weight`: 可疑客户端权重（默认0.3）

#### FedUP参数
- `--fedup_pruning_ratio`: 基础剪枝比例（默认0.1）
- `--fedup_adaptive`: 是否使用自适应剪枝比例（默认True）

## 方法流程

### 1. AlignIns四指标检测
```python
# 计算四个指标
cosine_similarities = compute_cosine_similarity(inter_model_updates)
euclidean_distances = compute_euclidean_distance(inter_model_updates)
sign_agreements = compute_sign_agreement(inter_model_updates)
magnitude_ratios = compute_magnitude_ratio(inter_model_updates)

# 综合评分和分类
benign_indices, suspicious_indices, malicious_indices = alignins_detection(...)
```

### 2. 过滤聚合
```python
# 设置客户端权重
# 良性客户端：权重1.0
# 可疑客户端：权重0.3
# 恶意客户端：权重0.0

aggregated_update = torch.sum(filtered_updates * client_weights.unsqueeze(1), dim=0)
```

### 3. 标准FedUP模型权重剪枝
```python
# 计算新的全局模型参数
new_global_params = flat_global_model + aggregated_update

# 计算权重重要性
weight_importance = compute_weight_importance_standard(
    new_global_params, old_global_params, benign_mean
)

# 应用剪枝
pruned_params = apply_model_pruning(new_global_params, weight_importance, pruning_ratio)
```

## 与之前实现的区别

### 错误的实现（之前）
- 对客户端更新进行"剪枝"（实际是过滤）
- 概念混淆：将更新过滤称为剪枝
- 不符合FedUP论文的原始定义

### 正确的实现（现在）
- 先用AlignIns检测和过滤聚合
- 再对聚合后的全局模型权重进行真正的剪枝
- 符合两个论文的原始定义和设计思路

## 实验结果

运行实验后，结果将保存在`./results`目录中，包含：
- 测试准确率变化
- 攻击成功率变化
- 客户端检测统计
- 剪枝统计信息

## 注意事项

1. **设备要求**: 建议使用GPU进行训练，CPU训练会比较慢
2. **内存要求**: 大模型和大数据集需要足够的内存
3. **参数调优**: 阈值参数可能需要根据具体场景调整
4. **攻击类型**: 不同攻击类型可能需要不同的防御参数

## 扩展和自定义

如果需要自定义实现，可以修改以下部分：

1. **异常检测指标**: 在`alignins_detection`函数中添加新的指标
2. **权重重要性计算**: 修改`compute_weight_importance_standard`函数
3. **剪枝策略**: 调整`apply_model_pruning`函数中的剪枝逻辑
4. **自适应策略**: 修改`compute_adaptive_pruning_ratio`函数

## 参考文献

1. AlignIns论文：多指标异常检测方法
2. FedUP论文：联邦遗忘和权重剪枝方法