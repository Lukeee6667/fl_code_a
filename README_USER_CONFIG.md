# 用户自定义配置运行指南

## 概述

基于您提供的常用配置，我们为您创建了专门的运行脚本，支持新的正确 AlignIns + FedUP 实现。

## 您的原始配置

```bash
CUDA_VISIBLE_DEVICES=0,1 python federated.py \
    --poison_frac 0.3 \
    --num_corrupt 10 \
    --num_agents 40 \
    --aggr alignins_fedup_hybrid \
    --data cifar10 \
    --attack badnet \
    --non_iid \
    --beta 0.5
```

## 新的正确实现配置

```bash
CUDA_VISIBLE_DEVICES=0,1 python federated.py \
    --poison_frac 0.3 \
    --num_corrupt 10 \
    --num_agents 40 \
    --aggr alignins_fedup_correct \  # 使用正确的实现
    --data cifar10 \
    --attack badnet \
    --non_iid \
    --beta 0.5
```

## 快速使用

### Linux/Mac 用户

```bash
# 交互式选择配置
bash run_user_config.sh

# 直接运行您的原始配置 + 正确实现
bash run_user_config.sh config_user_original

# 运行增强版配置（更多轮数）
bash run_user_config.sh config_enhanced

# 运行高攻击强度配置
bash run_user_config.sh config_high_attack
```

### Windows 用户

```cmd
# 交互式选择配置
run_user_config.bat

# 直接运行配置1（您的原始配置 + 正确实现）
run_user_config.bat 1

# 运行配置2（增强版配置）
run_user_config.bat 2
```

## 可用配置选项

| 配置 | 描述 | 主要特点 |
|------|------|----------|
| 1. config_user_original | 您的原始配置 + 正确实现 | 使用 `alignins_fedup_correct` |
| 2. config_enhanced | 增强版配置 | 更多本地训练轮数(5)和通信轮数(200) |
| 3. config_high_attack | 高攻击强度配置 | 投毒比例0.5，恶意客户端15个 |
| 4. config_dba_attack | DBA攻击配置 | 使用DBA攻击替代BadNet |
| 5. config_cifar100 | CIFAR-100配置 | 使用CIFAR-100数据集 |
| 6. config_compare_hybrid | 对比实验 | 使用原来的混合方法进行对比 |
| 7. config_custom | 自定义配置 | 支持自定义参数 |

## 配置参数说明

### 基础参数（基于您的配置）
- **GPU配置**: `CUDA_VISIBLE_DEVICES=0,1` (使用两张GPU)
- **投毒比例**: `poison_frac=0.3` (30%的数据被投毒)
- **恶意客户端**: `num_corrupt=10` (10个恶意客户端)
- **总客户端数**: `num_agents=40` (40个客户端)
- **数据集**: `cifar10`
- **攻击类型**: `badnet`
- **数据分布**: `--non_iid --beta 0.5` (非独立同分布)

### 新增的关键参数
- **聚合方法**: `alignins_fedup_correct` (正确的两阶段实现)
- **本地训练轮数**: `local_ep=2`
- **批次大小**: `bs=64`
- **学习率**: `client_lr=0.1, server_lr=1`

## 正确实现 vs 原始实现

### 原始实现 (`alignins_fedup_hybrid`)
- 对客户端更新进行"剪枝"（概念错误）
- 混合了两种方法但流程不清晰

### 正确实现 (`alignins_fedup_correct`)
- **阶段1**: AlignIns 四指标检测和过滤聚合
- **阶段2**: 对聚合后的全局模型进行标准 FedUP 权重剪枝
- 符合 FedUP 论文的原始设计

## 实验结果

运行后会在以下位置生成结果：
- **日志文件**: `logs/` 目录
- **模型文件**: `saved_models/` 目录
- **结果文件**: `results/` 目录

## 自定义配置示例

### Linux 自定义参数
```bash
# 自定义投毒比例、恶意客户端数、总客户端数、攻击类型
bash run_user_config.sh config_custom 0.4 12 50 DBA
```

### 修改脚本参数
您可以直接编辑 `run_user_config.sh` 或 `run_user_config.bat` 文件中的参数：

```bash
# 修改这些变量
POISON_FRAC=0.3             # 投毒比例
NUM_CORRUPT=10              # 恶意客户端数量
NUM_AGENTS=40               # 客户端总数
DATA="cifar10"              # 数据集
ATTACK="badnet"             # 攻击类型
AGGR_METHOD="alignins_fedup_correct"  # 聚合方法
```

## 性能对比建议

为了验证正确实现的效果，建议运行以下对比实验：

1. **原始配置 + 正确实现**:
   ```bash
   bash run_user_config.sh config_user_original
   ```

2. **原始配置 + 混合方法**（对比基线）:
   ```bash
   bash run_user_config.sh config_compare_hybrid
   ```

3. **高攻击强度测试**:
   ```bash
   bash run_user_config.sh config_high_attack
   ```

## 注意事项

1. **GPU内存**: 确保两张GPU有足够内存
2. **数据集**: 首次运行会自动下载数据集
3. **结果保存**: 每次实验结果会自动保存，避免覆盖
4. **参数调优**: 可根据实验结果调整 AlignIns 和 FedUP 的敏感度参数

## 故障排除

### 常见问题
1. **CUDA内存不足**: 减少批次大小 `bs`
2. **GPU不可用**: 检查 `CUDA_VISIBLE_DEVICES` 设置
3. **依赖缺失**: 确保安装了所有必要的Python包

### 联系支持
如果遇到问题，请检查：
1. GPU状态: `nvidia-smi`
2. Python环境: `python --version`
3. 依赖包: `pip list`

---

**快速开始**: 直接运行 `bash run_user_config.sh` 或 `run_user_config.bat` 开始您的实验！