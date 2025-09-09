# FedUP集成说明

## 概述

本项目已成功集成了**FedUP: Efficient Pruning-based Federated Unlearning for Model Poisoning Attacks**方法，这是一种基于剪枝的高效联邦遗忘技术，专门用于防御联邦学习中的模型中毒攻击。

## FedUP方法特点

### 🎯 核心优势
- **高效遗忘**: 通过选择性剪枝快速移除恶意客户端的影响
- **实时检测**: 基于更新幅度和方向一致性检测异常客户端
- **自适应剪枝**: 根据攻击强度动态调整剪枝策略
- **轻量级**: 相比完全重训练，计算开销显著降低

### 🔧 算法原理

基于论文《FedUP: Efficient Pruning-based Federated Unlearning for Model Poisoning Attacks》，FedUP通过以下步骤实现联邦遗忘：

#### 核心算法流程（Algorithm 1）

1. **异常客户端检测**：
   - 计算客户端更新的幅度和方向一致性
   - 识别偏离正常模式的可疑更新

2. **自适应剪枝比例计算**（公式5）：
   ```
   P ≈ (P_max - P_min) × z^γ + P_min
   ```
   - z：良性客户端模型相似度（归一化到[0,1]）
   - 根据模型收敛状态动态调整剪枝强度

3. **基于排名的权重选择**：
   - 计算权重重要性：rank = (w_diff)² × |w_global|
   - 选择排名前P%的权重进行剪枝

4. **选择性剪枝与聚合**：
   - 对选中权重应用遗忘掩码
   - 聚合剪枝后的客户端更新

#### 核心特点

- **自适应性**：根据模型收敛状态自动调整剪枝强度
- **高效性**：相比重训练，显著减少计算开销（约1/10的时间）
- **精确性**：基于权重重要性的精准剪枝
- **鲁棒性**：在IID和Non-IID场景下均有效
- **隐私保护**：无需访问原始训练数据

## 使用方法

### 1. 基础使用

```bash
# 使用FedUP方法进行联邦学习
python src/federated.py \
    --data cifar10 \
    --num_agents 20 \
    --num_corrupt 4 \
    --rounds 100 \
    --aggr fedup \
    --attack badnet
```

### 2. 自定义参数

```bash
# 自定义FedUP参数
python src/federated.py \
    --data cifar10 \
    --num_agents 20 \
    --num_corrupt 6 \
    --rounds 100 \
    --aggr fedup \
    --attack DBA \
    --fedup_pruning_ratio 0.2 \
    --fedup_sensitivity_threshold 0.3 \
    --fedup_unlearn_threshold 0.9
```

### 3. 使用运行示例脚本

```bash
# 基础实验
python run_fedup_example.py --basic

# 自定义参数实验
python run_fedup_example.py --custom

# 对比实验 (FedUP vs AlignIns vs FedAvg)
python run_fedup_example.py --compare
```

## 参数说明

### FedUP专用参数

基于论文《FedUP: Efficient Pruning-based Federated Unlearning for Model Poisoning Attacks》的实现：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|---------|
| `--fedup_p_max` | float | 0.15 | 最大剪枝率P_max（15%） |
| `--fedup_p_min` | float | 0.01 | 最小剪枝率P_min（1%） |
| `--fedup_gamma` | float | 5 | 曲线陡度参数γ，控制自适应剪枝曲线的陡度 |
| `--fedup_sensitivity_threshold` | float | 0.5 | 异常检测敏感度阈值 |

**自适应剪枝比例计算**：根据论文公式5，实际剪枝比例P会根据良性客户端模型间的相似度自动调整：
```
P ≈ (P_max - P_min) × z^γ + P_min
```
其中z是归一化后的客户端相似度（0到1之间）。

### 参数调优建议

1. **自适应剪枝参数调优**：
   - **IID场景**（客户端数据分布相似）：
     - `--fedup_p_max 0.15 --fedup_p_min 0.01 --fedup_gamma 5`
     - 预期剪枝比例：约10%（z≈0.98时）
   
   - **Non-IID场景**（客户端数据分布差异大）：
     - `--fedup_p_max 0.12 --fedup_p_min 0.02 --fedup_gamma 3`
     - 预期剪枝比例：约5%（z≈0.78时）
   
   - **强攻击场景**：
     - `--fedup_p_max 0.25 --fedup_p_min 0.05 --fedup_gamma 8`
     - 更激进的剪枝策略

2. **异常检测敏感度调优**：
   - 高精度检测：`--fedup_sensitivity_threshold 0.3`
   - 平衡检测：`--fedup_sensitivity_threshold 0.5`（默认）
   - 快速检测：`--fedup_sensitivity_threshold 0.7`

3. **曲线陡度参数γ的影响**：
   - γ=1：线性变化
   - γ=5：标准设置，适度非线性
   - γ=10：陡峭变化，对相似度敏感

## 实验配置示例

### 1. IID设置下的FedUP实验
```bash
python src/federated.py \
    --dataset cifar10 \
    --model resnet18 \
    --num_agents 10 \
    --num_corrupted 2 \
    --num_rounds 100 \
    --aggr fedup \
    --attack badnet \
    --fedup_p_max 0.15 \
    --fedup_p_min 0.01 \
    --fedup_gamma 5 \
    --fedup_sensitivity_threshold 0.5
```

### 2. Non-IID设置下的FedUP实验
```bash
python src/federated.py \
    --dataset cifar10 \
    --model resnet18 \
    --num_agents 20 \
    --num_corrupted 4 \
    --num_rounds 150 \
    --aggr fedup \
    --attack DBA \
    --non_iid \
    --fedup_p_max 0.12 \
    --fedup_p_min 0.02 \
    --fedup_gamma 3 \
    --fedup_sensitivity_threshold 0.3
```

### 3. 强攻击场景下的FedUP实验
```bash
python src/federated.py \
    --dataset cifar10 \
    --model resnet18 \
    --num_agents 15 \
    --num_corrupted 5 \
    --num_rounds 200 \
    --aggr fedup \
    --attack semantic \
    --fedup_p_max 0.25 \
    --fedup_p_min 0.05 \
    --fedup_gamma 8 \
    --fedup_sensitivity_threshold 0.3
```

## 与其他方法对比

### FedUP vs AlignIns

| 特性 | FedUP | AlignIns |
|------|-------|----------|
| **检测方式** | 统计异常检测 | 方向对齐检查 |
| **防御策略** | 选择性剪枝遗忘 | 权重过滤 |
| **计算复杂度** | 中等 | 较高 |
| **适用攻击** | 模型中毒 | 后门攻击 |
| **遗忘能力** | ✅ 支持 | ❌ 不支持 |

### FedUP vs FedAvg

| 特性 | FedUP | FedAvg |
|------|-------|--------|
| **鲁棒性** | 高 | 低 |
| **攻击防御** | ✅ 主动防御 | ❌ 无防御 |
| **计算开销** | 轻微增加 | 基线 |
| **准确性** | 保持 | 基线 |

## 日志输出示例

```
INFO - FedUP聚合 - 剪枝比例: 0.1, 敏感度阈值: 0.5
INFO - 客户端 2 被标记为异常 (幅度异常: z_score=1.234)
INFO - 客户端 5 被标记为异常 (方向异常: similarity=0.123)
INFO - 应用FedUP遗忘，影响客户端: {2, 5}
```

## 性能评估指标

### 主要指标
- **Clean ACC**: 干净数据准确率
- **Attack Success Ratio (ASR)**: 攻击成功率
- **Backdoor ACC**: 后门准确率

### FedUP期望效果
- ✅ 保持较高的Clean ACC
- ✅ 显著降低ASR
- ✅ 降低Backdoor ACC
- ✅ 快速收敛

## 故障排除

### 常见问题

1. **导入错误**
   ```
   ModuleNotFoundError: No module named 'torch'
   ```
   **解决**: 确保安装了PyTorch
   ```bash
   pip install torch torchvision
   ```

2. **参数错误**
   ```
   error: argument --aggr: invalid choice: 'fedup'
   ```
   **解决**: 确保使用修改后的federated.py文件

3. **内存不足**
   ```
   RuntimeError: CUDA out of memory
   ```
   **解决**: 减少batch size或使用CPU
   ```bash
   --bs 32 --device cpu
   ```

### 调试技巧

1. **启用详细日志**
   ```bash
   --debug
   ```

2. **减少实验规模**
   ```bash
   --rounds 10 --num_agents 10
   ```

3. **使用较小数据集**
   ```bash
   --data fmnist
   ```

## 文件结构

```
AlignIns/
├── src/
│   ├── federated.py          # 主训练脚本 (已修改)
│   ├── aggregation.py        # 聚合方法 (已添加FedUP)
│   └── ...
├── run_fedup_example.py      # FedUP运行示例 (新增)
├── FEDUP_README.md           # FedUP说明文档 (新增)
└── ...
```

## 引用

如果您在研究中使用了FedUP方法，请引用原论文：

```bibtex
@article{fedup2024,
  title={FedUP: Efficient Pruning-based Federated Unlearning for Model Poisoning Attacks},
  author={Authors},
  journal={arXiv preprint arXiv:2508.13853},
  year={2024}
}
```

## 联系方式

如有问题或建议，请通过以下方式联系：
- 📧 Email: [your-email@example.com]
- 🐛 Issues: [GitHub Issues]
- 📖 Documentation: [项目文档]

---

**注意**: 本集成基于AlignIns项目框架，确保在使用前已正确安装所有依赖项。