# 联邦学习中基于方向对齐检测的后门攻击检测框架

本项目是论文“Detecting Backdoor Attacks in Federated Learning via Direction Alignment Inspection”的官方实现，框架在 AlignIns 方法的基础上进行了改进与扩展。核心第三章方法对应实现为 `origin_alignins_clustering_weighted3_dynamic_softmax_gated`。

论文地址：[arXiv:2503.07978](https://arxiv.org/abs/2503.07978)

## 硬件要求

- 复现大论文中的关键实验必须使用 **NVIDIA RTX 2080Ti**（单卡）运行。若使用其他显卡，由于不同 GPU 的数值精度实现存在差异，结果会出现细微浮动，无法严格对齐论文中的精确数值。
- 如本地没有 2080Ti，可在 AutoDL 等平台租用 2080Ti 机器进行实验。

## 环境配置

项目已提供完整依赖环境文件 `fl_env.yml`。使用以下命令创建并激活环境：

```bash
conda env create -f fl_env.yml
conda activate fl
```

环境要点：
- Python 3.9.5
- PyTorch 1.8.0（CUDA 11.1.1）
- 其余依赖已包含于 `fl_env.yml`

## 数据集

- CIFAR-10 / CIFAR-100：通过 `torchvision` 自动下载
- Tiny-ImageNet：可从 Kaggle 下载

## 核心方法与配置
- 原alignins方法：`origin_alignins`
- 第三章核心方法：`origin_alignins_clustering_weighted3_dynamic_softmax_gated`

示例关键参数（在 `src/federated.py` 中可查阅更详细说明）：
- `cluster_imbalance_ratio`：聚类不平衡比例
- `suspicious_weight` / `benign_weight`：可疑/良性客户端权重
- `gated2_bd_delta_max`：门控机制的最大后门 ACC 变化阈值
- `strict_factor`：严格阈值因子
- `cluster_mz_metrics`：聚类评估指标


## 实验启动脚本 Demo

提供了完整的实验启动脚本示例，路径如下：
- [run_client_30_8_0.3.sh](fl_code_a/run_client_30_8_0.3.sh)

使用方法：
```bash
chmod +x run_client_30_8_0.3.sh
./run_client_30_8_0.3.sh
```

## 致谢

本框架参考并构建自 https://github.com/JiiahaoXU/AlignIns ，感谢其出色的开源贡献。
