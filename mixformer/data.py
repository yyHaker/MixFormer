"""
MixFormer 数据处理模块。

实现 Alibaba UserBehavior 数据集 (DIN 论文) 的加载和 DataLoader 工厂函数。
数据来源: 天池 https://tianchi.aliyun.com/dataset/649
字段: user_id, item_id, category_id, behavior_type, timestamp
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import MixFormerConfig

logger = logging.getLogger("MixFormer.Data")


# ============================================================================
# 1. Alibaba UserBehavior 数据集
# ============================================================================


class AlibabaDataset(Dataset):
    """Alibaba UserBehavior 数据集。

    来源: 天池 https://tianchi.aliyun.com/dataset/649
    引用论文: Deep Interest Network for Click-Through Rate Prediction (1706.06978)

    数据集包含约 1 亿条用户行为记录:
    - user_id: 用户ID
    - item_id: 商品ID
    - category_id: 商品类目ID
    - behavior_type: 行为类型 (pv/buy/cart/fav)
    - timestamp: 行为时间戳

    数据处理策略 (参考 DIN 论文):
    1. 按用户分组，按时间排序用户的历史行为
    2. 对每个用户的历史行为序列做 sliding window，生成训练样本:
       - 历史行为序列: 前 t-1 个行为中的 (item_id, category_id)
       - 目标物品: 第 t 个行为的 (item_id, category_id)
       - 标签: 1 (正样本)
    3. 负采样: 随机采样用户未交互的物品作为负样本 (标签: 0)

    Args:
        data_dir: 预处理后的数据目录 (包含 dataset.pkl)
        config: MixFormer 模型配置
        split: 数据集拆分 ("train" | "test")
        max_seq_length: 最大序列长度
    """

    def __init__(
        self,
        data_dir: str,
        config: MixFormerConfig,
        split: str = "train",
        max_seq_length: int = 50,
    ):
        super().__init__()
        self.config = config
        self.split = split
        self.max_seq_length = max_seq_length

        # 加载预处理后的数据
        data_path = os.path.join(data_dir, f"{split}_data.pkl")
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"预处理后的数据文件不存在: {data_path}\n"
                f"请先运行 scripts/download_alibaba.py 进行数据预处理。"
            )

        logger.info(f"Loading {split} data from {data_path}...")
        with open(data_path, "rb") as f:
            data = pickle.load(f)

        self.samples = data["samples"]  # List of dicts
        self.item_to_cate = data["item_to_cate"]  # item_id -> category_id 映射

        logger.info(f"Loaded {len(self.samples)} {split} samples.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """获取单个样本。

        每个样本包含:
        - target_item_id: 目标物品 ID
        - target_cate_id: 目标物品类目 ID
        - hist_item_ids: 历史行为物品 ID 序列 (已填充到 max_seq_length)
        - hist_cate_ids: 历史行为类目 ID 序列 (已填充到 max_seq_length)
        - seq_mask: 序列有效位置掩码
        - label: 0/1 标签
        """
        sample = self.samples[idx]

        target_item_id = sample["target_item"]
        target_cate_id = sample["target_cate"]
        hist_items = sample["hist_items"]
        hist_cates = sample["hist_cates"]
        label = sample["label"]

        # 截断到最大序列长度
        seq_len = min(len(hist_items), self.max_seq_length)
        hist_items = hist_items[-seq_len:]  # 取最近的
        hist_cates = hist_cates[-seq_len:]

        # Padding 到固定长度
        padded_items = np.zeros(self.max_seq_length, dtype=np.int64)
        padded_cates = np.zeros(self.max_seq_length, dtype=np.int64)
        seq_mask = np.zeros(self.max_seq_length, dtype=bool)

        padded_items[:seq_len] = hist_items
        padded_cates[:seq_len] = hist_cates
        seq_mask[:seq_len] = True

        return {
            "target_item_id": torch.tensor(target_item_id, dtype=torch.long),
            "target_cate_id": torch.tensor(target_cate_id, dtype=torch.long),
            "hist_item_ids": torch.tensor(padded_items, dtype=torch.long),
            "hist_cate_ids": torch.tensor(padded_cates, dtype=torch.long),
            "seq_mask": torch.tensor(seq_mask, dtype=torch.bool),
            "label": torch.tensor(label, dtype=torch.float32),
        }


# ============================================================================
# 2. Collate Function
# ============================================================================


def collate_fn(batch: list[dict]) -> dict:
    """Alibaba 数据集的 collate 函数。

    Args:
        batch: 单个样本字典的列表

    Returns:
        批量字典，包含:
        - target_item_id: (batch_size,) 目标物品 ID
        - target_cate_id: (batch_size,) 目标物品类目 ID
        - hist_item_ids: (batch_size, T) 历史物品 ID 序列
        - hist_cate_ids: (batch_size, T) 历史类目 ID 序列
        - seq_mask: (batch_size, T) 序列有效位置掩码
        - label: (batch_size,) 标签
    """
    return {
        "target_item_id": torch.stack([s["target_item_id"] for s in batch]),
        "target_cate_id": torch.stack([s["target_cate_id"] for s in batch]),
        "hist_item_ids": torch.stack([s["hist_item_ids"] for s in batch]),
        "hist_cate_ids": torch.stack([s["hist_cate_ids"] for s in batch]),
        "seq_mask": torch.stack([s["seq_mask"] for s in batch]),
        "label": torch.stack([s["label"] for s in batch]),
    }


# ============================================================================
# 3. DataLoader 工厂函数
# ============================================================================


def create_dataloader(
    data_dir: str,
    config: MixFormerConfig,
    split: str = "train",
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """创建 Alibaba UserBehavior 数据集 DataLoader。

    Args:
        data_dir: 预处理后的数据目录
        config: MixFormer 模型配置
        split: 数据集拆分 ("train" | "test")
        batch_size: 批量大小
        shuffle: 是否打乱数据
        num_workers: 数据加载工作线程数

    Returns:
        DataLoader 实例
    """
    dataset = AlibabaDataset(
        data_dir=data_dir,
        config=config,
        split=split,
        max_seq_length=config.seq_length,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        drop_last=(split == "train"),
    )
