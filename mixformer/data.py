"""
MixFormer 数据处理模块。

实现合成推荐数据集生成和 DataLoader 工厂函数。
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import MixFormerConfig


class SyntheticRecDataset(Dataset):
    """合成推荐数据集。

    生成包含类别特征和用户行为序列的合成数据，用于验证模型训练和推理流程。

    数据包含:
    - 非序列特征: 随机生成的类别特征 (user_id, item_id, feature_0, ...)
    - 序列特征: 随机生成的用户历史行为序列 (物品 ID 序列)
    - 标签: 随机生成的二分类标签 (0/1)

    Args:
        config: MixFormer 模型配置
        num_samples: 数据集样本数量
        seed: 随机种子
    """

    def __init__(
        self,
        config: MixFormerConfig,
        num_samples: int = 10000,
        seed: int = 42,
    ):
        super().__init__()
        self.config = config
        self.num_samples = num_samples

        rng = np.random.RandomState(seed)

        # 生成非序列特征
        self.non_seq_features: Dict[str, np.ndarray] = {}
        for feat_name, vocab_size in sorted(config.vocab_sizes.items()):
            self.non_seq_features[feat_name] = rng.randint(
                0, vocab_size, size=(num_samples,)
            )

        # 生成序列特征 (用户历史物品 ID)
        self.seq_features = rng.randint(
            0, config.num_items, size=(num_samples, config.seq_length)
        )

        # 生成序列 mask (模拟变长序列)
        # 每个样本的有效序列长度为 [1, seq_length]
        seq_lengths = rng.randint(1, config.seq_length + 1, size=(num_samples,))
        self.seq_mask = np.zeros((num_samples, config.seq_length), dtype=bool)
        for i in range(num_samples):
            self.seq_mask[i, : seq_lengths[i]] = True

        # 生成标签 (二分类: 0 或 1)
        # 使用轻微偏置以模拟真实 CTR 数据 (正样本比例约 5%-20%)
        self.labels = (rng.random(num_samples) < 0.1).astype(np.float32)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        """获取单个样本。

        Returns:
            dict 包含:
                - non_seq_features: {feature_name: int}
                - seq_features: (T,) int array
                - seq_mask: (T,) bool array
                - label: float
        """
        sample = {
            "non_seq_features": {
                feat_name: torch.tensor(feat_values[idx], dtype=torch.long)
                for feat_name, feat_values in self.non_seq_features.items()
            },
            "seq_features": torch.tensor(self.seq_features[idx], dtype=torch.long),
            "seq_mask": torch.tensor(self.seq_mask[idx], dtype=torch.bool),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
        }
        return sample


def collate_fn(batch: list[dict]) -> dict:
    """自定义 collate 函数，处理嵌套字典结构。

    Args:
        batch: 单个样本字典的列表

    Returns:
        批量字典
    """
    # 收集非序列特征
    non_seq_features = {}
    feature_names = batch[0]["non_seq_features"].keys()
    for feat_name in feature_names:
        non_seq_features[feat_name] = torch.stack(
            [sample["non_seq_features"][feat_name] for sample in batch]
        )

    return {
        "non_seq_features": non_seq_features,
        "seq_features": torch.stack([sample["seq_features"] for sample in batch]),
        "seq_mask": torch.stack([sample["seq_mask"] for sample in batch]),
        "label": torch.stack([sample["label"] for sample in batch]),
    }


def create_dataloader(
    config: MixFormerConfig,
    num_samples: int = 10000,
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 0,
    seed: int = 42,
) -> DataLoader:
    """创建 DataLoader 的工厂函数。

    Args:
        config: MixFormer 模型配置
        num_samples: 数据集样本数量
        batch_size: 批量大小
        shuffle: 是否打乱数据
        num_workers: 数据加载工作线程数
        seed: 随机种子

    Returns:
        DataLoader 实例
    """
    dataset = SyntheticRecDataset(config, num_samples=num_samples, seed=seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
