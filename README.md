# MixFormer: Co-Scaling Up Dense and Sequence in Industrial Recommenders

A PyTorch implementation of **MixFormer**, a unified Transformer-style architecture for jointly modeling sequence behavior and feature interactions in recommender systems.

> **Paper**: [MixFormer: Co-Scaling Up Dense and Sequence in Industrial Recommenders](https://arxiv.org/abs/2602.14110)  
> **Authors**: Xu Huang, Hao Zhang, Zhifang Fan, et al. (ByteDance)

## Overview

MixFormer addresses the **co-scaling challenge** in industrial recommender systems: how to efficiently scale both dense feature interactions and user sequence modeling within a single architecture under limited computational budgets.

### Key Features

- **Unified Architecture**: Joint modeling of dense features and user behavior sequences in a single Transformer backbone
- **MixFormer Block**: Three core modules — Query Mixer, Cross Attention, Output Fusion
- **TorchRec Integration**: 使用 Meta 开源的 [TorchRec](https://github.com/pytorch/torchrec) 管理稀疏特征嵌入表
- **Alibaba UserBehavior Dataset**: 支持 DIN 论文中的阿里巴巴用户行为数据集 (天池)
- **Training Pipeline**: Complete training with BCE loss, Adam optimizer, AUC/LogLoss metrics, checkpointing, and mixed precision support
- **Inference Pipeline**: Standard inference with throughput benchmarking

## Architecture

```
Input → Feature Encoding → [MixFormer Block × L] → Task Head → CTR Prediction

MixFormer Block:
  ├── Query Mixer (QM): HeadMixing + Per-Head SwiGLU FFN
  ├── Cross Attention (CA): Sequence preprocessing + Scaled Dot-Product Attention
  └── Output Fusion (OF): Per-Head SwiGLU FFN
```

### Data Flow

```
目标物品 (item_id + cate_id)
    ↓ TorchRec EmbeddingBag → 拼接 → MLP
    ↓ → Query (batch, N, D)

历史行为序列 (item_ids + cate_ids)
    ↓ nn.Embedding → 拼接 → Linear + Position Encoding
    ↓ → Sequence (batch, T, N*D)

Query + Sequence → L × MixFormerBlock → TaskHead → CTR
```

### Model Configurations

| Config   | N (heads) | L (layers) | D (dim) | Parameters |
|----------|-----------|------------|---------|------------|
| Default  | 8         | 3          | 64      | ~7M        |
| Medium   | 8         | 4          | 128     | ~15M       |

## Installation

```bash
# Clone the repository
cd MixFormer

# Install dependencies
pip install -r requirements.txt

# TorchRec (可选, 有 nn.EmbeddingBag 自动 fallback)
pip install torchrec
```

### Requirements

- Python >= 3.10
- PyTorch >= 2.1.0
- NumPy >= 1.24.0
- tqdm >= 4.65.0
- scikit-learn >= 1.3.0 (optional, for AUC computation)
- torchrec >= 0.6.0 (optional, for TorchRec embedding management)
- pandas >= 2.0.0 (for data preprocessing)

## Quick Start

### 1. 数据准备

```bash
# 方式 A: 生成模拟数据 (快速验证)
python scripts/download_alibaba.py --generate_mock --output_dir data/alibaba \
    --mock_users 10000 --mock_items 50000

# 方式 B: 使用真实数据 (从天池下载 UserBehavior.csv)
# 下载地址: https://tianchi.aliyun.com/dataset/649
python scripts/download_alibaba.py \
    --raw_data_path /path/to/UserBehavior.csv \
    --output_dir data/alibaba

# 方式 C: 使用真实数据子集 (前 500 万条)
python scripts/download_alibaba.py \
    --raw_data_path /path/to/UserBehavior.csv \
    --output_dir data/alibaba \
    --max_rows 5000000
```

### 2. 训练

```bash
# 默认配置训练
python train.py --data_dir data/alibaba --epochs 5 --batch_size 4096 --lr 0.001

# Medium 配置训练
python train.py --config medium --data_dir data/alibaba --epochs 5

# 混合精度训练
python train.py --data_dir data/alibaba --epochs 5 --batch_size 4096 --amp
```

### 3. 推理

```bash
# 从 checkpoint 推理
python inference.py --checkpoint checkpoints/best_model.pt --data_dir data/alibaba

# 随机权重测试推理流程
python inference.py --data_dir data/alibaba --no_checkpoint
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test class
python -m pytest tests/test_model.py::TestMixFormer -v

# Run with coverage
python -m pytest tests/ -v --tb=short
```

## Alibaba UserBehavior Dataset

本项目使用 DIN 论文（[Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1706.06978)）中提到的阿里巴巴用户行为数据集。

### 数据集信息

- **来源**: [天池](https://tianchi.aliyun.com/dataset/649)
- **规模**: ~1 亿条用户行为记录
- **用户数**: 987,994
- **商品数**: 4,162,024
- **类目数**: 9,439
- **时间范围**: 2017-11-25 至 2017-12-03

### 数据格式

```
user_id, item_id, category_id, behavior_type, timestamp
```

| 字段 | 说明 |
|------|------|
| user_id | 用户 ID |
| item_id | 商品 ID |
| category_id | 商品类目 ID |
| behavior_type | 行为类型: pv(点击), buy(购买), cart(加购), fav(收藏) |
| timestamp | 行为时间戳 |

### 数据处理策略

按 DIN 论文方式构建训练/测试样本：
1. 按用户分组，按时间排序行为序列
2. 只保留 `pv` (点击) 行为用于 CTR 预测
3. 对每个用户: 前 T-1 条行为 → 历史序列, 第 T 条行为 → 目标物品 (正样本)
4. 负采样: 随机采样用户未交互的物品 (标签 0)
5. 最后一条行为留作测试，其余用于训练

### TorchRec 集成

使用 TorchRec 的 `EmbeddingBagCollection` 管理 `item_id` 和 `category_id` 的嵌入表。如果 TorchRec (fbgemm-gpu) 不可用，自动 fallback 到 `nn.EmbeddingBag`。

```python
# 自动检测 TorchRec 可用性
from mixformer.model import create_embedding_collection

emb = create_embedding_collection(
    feature_names=["item_id", "category_id"],
    vocab_sizes=[4200000, 10000],
    embed_dim=64,
    use_torchrec=True,  # 自动 fallback
)
```

## Project Structure

```
MixFormer/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── mixformer/
│   ├── __init__.py              # Package exports
│   ├── config.py                # MixFormerConfig (default/medium presets)
│   ├── modules.py               # SwiGLUFFN, HeadMixing, PerHeadSwiGLUFFN
│   ├── layers.py                # QueryMixer, CrossAttention, OutputFusion, MixFormerBlock
│   ├── model.py                 # FeatureEncoder, TaskHead, MixFormer
│   └── data.py                  # AlibabaDataset, DataLoader factory
├── scripts/
│   └── download_alibaba.py      # 数据集下载与预处理脚本
├── train.py                     # Training script
├── inference.py                 # Inference script
└── tests/
    └── test_model.py            # Unit tests
```

## Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `data/alibaba` | Data directory |
| `--config` | `default` | Model preset (default/medium) |
| `--epochs` | `10` | Number of training epochs |
| `--batch_size` | `256` | Batch size |
| `--lr` | `0.001` | Learning rate |
| `--weight_decay` | `1e-5` | Weight decay |
| `--amp` | `false` | Enable mixed precision training |
| `--device` | `auto` | Compute device (cpu/cuda/mps/auto) |
| `--save_dir` | `checkpoints` | Checkpoint save directory |

## Citation

```bibtex
@article{huang2026mixformer,
  title={MixFormer: Co-Scaling Up Dense and Sequence in Industrial Recommenders},
  author={Huang, Xu and Zhang, Hao and Fan, Zhifang and Huang, Yunwen and Wei, Zhuoxing and Chai, Zheng and Ni, Jinan and Zheng, Yuchao and Chen, Qiwei},
  journal={arXiv preprint arXiv:2602.14110},
  year={2026}
}

@article{zhou2018deep,
  title={Deep Interest Network for Click-Through Rate Prediction},
  author={Zhou, Guorui and Zhu, Xiaoqiang and Song, Chengru and Fan, Ying and Zhu, Han and Ma, Xiao and Yan, Yanghui and Jin, Junqi and Li, Han and Gai, Kun},
  journal={Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  year={2018}
}
```

## License

This implementation is for research and educational purposes.
