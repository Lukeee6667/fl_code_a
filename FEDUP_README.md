# FedUP集成说明

## 概述

本项目已成功集成了**FedUP: Efficient Pruning-based Federated Unlearning for Model Poisoning Attacks**方法，这是一种基于剪枝的高效联邦遗忘技术，专门用于防御联邦学习中的模型中毒攻击。

## FedUP方法特点

### 🎯 核心优势
- **高效遗忘**: 通过选择性剪枝快速移除恶意客户端的影响
- **实时检测**: 基于更新幅度和方向一致性检测异常客户端
- **自适应剪枝**: 根据攻击强度动态调整剪枝策略
- **轻量级**: 相比完全重训练，计算开销显著降低

### 🔧 技术原理
1. **异常检测**: 分析客户端更新的统计特征
2. **遗忘掩码生成**: 基于可疑更新创建剪枝掩码
3. **选择性聚合**: 对异常客户端应用遗忘策略
4. **权重调整**: 降低可疑客户端的聚合权重

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

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--fedup_pruning_ratio` | float | 0.1 | 剪枝比例，控制遗忘强度 |
| `--fedup_sensitivity_threshold` | float | 0.5 | 异常检测敏感度阈值 |
| `--fedup_unlearn_threshold` | float | 0.8 | 遗忘决策阈值 |

### 参数调优建议

#### 剪枝比例 (`fedup_pruning_ratio`)
- **0.05-0.1**: 轻度剪枝，适用于轻微攻击
- **0.1-0.2**: 中度剪枝，适用于一般攻击
- **0.2-0.3**: 重度剪枝，适用于强烈攻击

#### 敏感度阈值 (`fedup_sensitivity_threshold`)
- **0.3-0.4**: 高敏感度，检测更多异常
- **0.5-0.6**: 中等敏感度，平衡检测精度
- **0.7-0.8**: 低敏感度，减少误报

## 实验配置示例

### 1. 轻量级防御配置
```bash
python src/federated.py \
    --aggr fedup \
    --fedup_pruning_ratio 0.05 \
    --fedup_sensitivity_threshold 0.6 \
    --fedup_unlearn_threshold 0.7
```

### 2. 标准防御配置
```bash
python src/federated.py \
    --aggr fedup \
    --fedup_pruning_ratio 0.1 \
    --fedup_sensitivity_threshold 0.5 \
    --fedup_unlearn_threshold 0.8
```

### 3. 强化防御配置
```bash
python src/federated.py \
    --aggr fedup \
    --fedup_pruning_ratio 0.2 \
    --fedup_sensitivity_threshold 0.3 \
    --fedup_unlearn_threshold 0.9
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