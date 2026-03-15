#!/usr/bin/env python3
"""
Alibaba UserBehavior 数据集下载和预处理脚本。

数据集来源:
  - 天池: https://tianchi.aliyun.com/dataset/649
  - 论文: Deep Interest Network for Click-Through Rate Prediction (1706.06978)

数据集格式 (CSV, 无表头):
  user_id, item_id, category_id, behavior_type, timestamp

行为类型:
  - pv: 浏览 (点击)
  - buy: 购买
  - cart: 加购物车
  - fav: 收藏

预处理流程:
  1. 读取原始 CSV 数据
  2. 过滤只保留 'pv' (点击) 行为用于 CTR 预测
  3. 按用户分组，按时间排序
  4. 对 user_id / item_id / category_id 做 ID 重映射 (从 0 开始)
  5. 按 DIN 论文方式构建训练/测试样本:
     - 对每个用户的行为序列，取倒数第 2 条之前的行为构建训练样本
     - 取最后一条行为作为测试样本
     - 负采样: 随机采样用户未交互的物品
  6. 保存为 pickle 格式

Usage:
  # 方式 1: 已有原始数据
  python scripts/download_alibaba.py --raw_data_path /path/to/UserBehavior.csv --output_dir data/alibaba

  # 方式 2: 使用子集 (前 N 行)
  python scripts/download_alibaba.py --raw_data_path /path/to/UserBehavior.csv --output_dir data/alibaba --max_rows 5000000

  # 方式 3: 生成模拟小数据集 (用于快速验证流程)
  python scripts/download_alibaba.py --output_dir data/alibaba --generate_mock --mock_users 10000 --mock_items 50000
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import random
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("AlibabaPreprocess")


def load_raw_data(
    raw_data_path: str,
    max_rows: Optional[int] = None,
    behavior_filter: str = "pv",
) -> List[Tuple[int, int, int, str, int]]:
    """加载原始 UserBehavior.csv 数据。

    Args:
        raw_data_path: CSV 文件路径
        max_rows: 最大读取行数 (None 表示全部)
        behavior_filter: 过滤的行为类型 (None 表示不过滤)

    Returns:
        列表，每个元素为 (user_id, item_id, cate_id, behavior, timestamp)
    """
    logger.info(f"Loading raw data from {raw_data_path}...")
    records = []
    count = 0
    skipped = 0

    with open(raw_data_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 5:
                skipped += 1
                continue

            try:
                user_id = int(parts[0])
                item_id = int(parts[1])
                cate_id = int(parts[2])
                behavior = parts[3].strip()
                timestamp = int(parts[4])
            except (ValueError, IndexError):
                skipped += 1
                continue

            # 过滤行为类型
            if behavior_filter and behavior != behavior_filter:
                continue

            records.append((user_id, item_id, cate_id, behavior, timestamp))
            count += 1

            if max_rows and count >= max_rows:
                break

            if count % 10000000 == 0:
                logger.info(f"  Loaded {count:,} records...")

    logger.info(f"Loaded {count:,} records (skipped {skipped:,} invalid lines)")
    return records


def remap_ids(
    records: List[Tuple[int, int, int, str, int]],
) -> Tuple[List[Tuple[int, int, int, int]], Dict, Dict, Dict]:
    """对 user_id, item_id, cate_id 做 ID 重映射。

    Returns:
        (remapped_records, user_map, item_map, cate_map)
    """
    logger.info("Remapping IDs...")
    user_set = set()
    item_set = set()
    cate_set = set()

    for user_id, item_id, cate_id, _, _ in records:
        user_set.add(user_id)
        item_set.add(item_id)
        cate_set.add(cate_id)

    # ID 重映射: 0 保留给 padding
    user_map = {uid: idx + 1 for idx, uid in enumerate(sorted(user_set))}
    item_map = {iid: idx + 1 for idx, iid in enumerate(sorted(item_set))}
    cate_map = {cid: idx + 1 for idx, cid in enumerate(sorted(cate_set))}

    remapped = []
    for user_id, item_id, cate_id, _, timestamp in records:
        remapped.append((
            user_map[user_id],
            item_map[item_id],
            cate_map[cate_id],
            timestamp,
        ))

    logger.info(
        f"ID Remapping: {len(user_map):,} users, "
        f"{len(item_map):,} items, {len(cate_map):,} categories"
    )
    return remapped, user_map, item_map, cate_map


def build_user_sequences(
    records: List[Tuple[int, int, int, int]],
) -> Dict[int, List[Tuple[int, int, int]]]:
    """按用户分组并按时间排序。

    Returns:
        {user_id: [(item_id, cate_id, timestamp), ...]}，按时间升序排列
    """
    logger.info("Building user sequences...")
    user_seqs = defaultdict(list)
    for user_id, item_id, cate_id, timestamp in records:
        user_seqs[user_id].append((item_id, cate_id, timestamp))

    # 按时间排序
    for uid in user_seqs:
        user_seqs[uid].sort(key=lambda x: x[2])

    logger.info(f"Built sequences for {len(user_seqs):,} users")
    return dict(user_seqs)


def build_item_to_cate(
    records: List[Tuple[int, int, int, int]],
) -> Dict[int, int]:
    """构建 item_id -> category_id 的映射。"""
    item_to_cate = {}
    for _, item_id, cate_id, _ in records:
        item_to_cate[item_id] = cate_id
    return item_to_cate


def build_dataset(
    user_seqs: Dict[int, List[Tuple[int, int, int]]],
    item_to_cate: Dict[int, int],
    num_items: int,
    neg_ratio: int = 1,
    min_hist_len: int = 2,
    seed: int = 42,
) -> Tuple[List[dict], List[dict]]:
    """按 DIN 论文方式构建训练/测试样本。

    对每个用户的行为序列:
    - 训练: 序列中第 2 ~ 倒数第 2 个位置作为目标, 之前的作为历史
    - 测试: 序列中最后一个位置作为目标, 之前的所有作为历史
    - 每个正样本配一个负样本 (随机未交互物品)

    Args:
        user_seqs: 用户行为序列
        item_to_cate: item_id -> category_id 映射
        num_items: 物品总数 (含 padding, 实际 ID 范围 [1, num_items])
        neg_ratio: 负样本数比例 (每个正样本配几个负样本)
        min_hist_len: 用户最少历史行为数 (少于此数的用户跳过)
        seed: 随机种子

    Returns:
        (train_samples, test_samples)
    """
    logger.info("Building train/test samples...")
    rng = random.Random(seed)

    all_item_ids = list(range(1, num_items + 1))  # [1, num_items]

    train_samples = []
    test_samples = []
    skipped_users = 0

    for user_id, seq in user_seqs.items():
        if len(seq) < min_hist_len:
            skipped_users += 1
            continue

        # 该用户的正向交互物品集合
        user_items = set(item_id for item_id, _, _ in seq)

        for i in range(1, len(seq)):
            # 历史行为: seq[0:i]
            hist_items = np.array([s[0] for s in seq[:i]], dtype=np.int64)
            hist_cates = np.array([s[1] for s in seq[:i]], dtype=np.int64)

            # 目标物品 (正样本)
            target_item = seq[i][0]
            target_cate = seq[i][1]

            pos_sample = {
                "target_item": target_item,
                "target_cate": target_cate,
                "hist_items": hist_items,
                "hist_cates": hist_cates,
                "label": 1.0,
            }

            # 负采样: 随机选择一个用户未交互的物品
            neg_samples = []
            for _ in range(neg_ratio):
                neg_item = rng.choice(all_item_ids)
                while neg_item in user_items:
                    neg_item = rng.choice(all_item_ids)
                neg_cate = item_to_cate.get(neg_item, 0)

                neg_samples.append({
                    "target_item": neg_item,
                    "target_cate": neg_cate,
                    "hist_items": hist_items,
                    "hist_cates": hist_cates,
                    "label": 0.0,
                })

            if i == len(seq) - 1:
                # 最后一个行为 → 测试集
                test_samples.append(pos_sample)
                test_samples.extend(neg_samples)
            else:
                # 中间行为 → 训练集
                train_samples.append(pos_sample)
                train_samples.extend(neg_samples)

    # 打乱
    rng.shuffle(train_samples)
    rng.shuffle(test_samples)

    logger.info(
        f"Built {len(train_samples):,} train samples, "
        f"{len(test_samples):,} test samples "
        f"(skipped {skipped_users:,} users with < {min_hist_len} behaviors)"
    )
    return train_samples, test_samples


def generate_mock_data(
    num_users: int = 10000,
    num_items: int = 50000,
    num_categories: int = 500,
    avg_behaviors_per_user: int = 20,
    seed: int = 42,
) -> Tuple[List[dict], List[dict], Dict[int, int], int, int, int]:
    """生成模拟的 Alibaba UserBehavior 数据。

    用于在没有真实数据时快速验证整个训练/推理流程。

    Returns:
        (train_samples, test_samples, item_to_cate, num_users, num_items, num_categories)
    """
    logger.info(
        f"Generating mock Alibaba data: "
        f"{num_users:,} users, {num_items:,} items, {num_categories:,} categories"
    )
    rng = np.random.RandomState(seed)
    random.seed(seed)

    # 随机分配每个物品到一个类目
    item_to_cate = {}
    for item_id in range(1, num_items + 1):
        item_to_cate[item_id] = rng.randint(1, num_categories + 1)

    # 为每个用户生成行为序列
    user_seqs = {}
    for user_id in range(1, num_users + 1):
        # 每个用户的行为数量
        num_behaviors = max(3, rng.poisson(avg_behaviors_per_user))
        num_behaviors = min(num_behaviors, 200)  # 上限

        # 随机生成用户交互的物品
        items = rng.randint(1, num_items + 1, size=num_behaviors)
        # 生成递增时间戳
        timestamps = np.sort(rng.randint(1511539200, 1512316800, size=num_behaviors))

        seq = [
            (int(items[j]), item_to_cate[int(items[j])], int(timestamps[j]))
            for j in range(num_behaviors)
        ]
        user_seqs[user_id] = seq

    # 构建训练/测试样本
    train_samples, test_samples = build_dataset(
        user_seqs=user_seqs,
        item_to_cate=item_to_cate,
        num_items=num_items,
        neg_ratio=1,
        min_hist_len=3,
        seed=seed,
    )

    return (
        train_samples,
        test_samples,
        item_to_cate,
        num_users,
        num_items,
        num_categories,
    )


def save_processed_data(
    output_dir: str,
    train_samples: List[dict],
    test_samples: List[dict],
    item_to_cate: Dict[int, int],
    num_users: int,
    num_items: int,
    num_categories: int,
):
    """保存预处理后的数据。"""
    os.makedirs(output_dir, exist_ok=True)

    # 保存训练数据
    train_path = os.path.join(output_dir, "train_data.pkl")
    with open(train_path, "wb") as f:
        pickle.dump({
            "samples": train_samples,
            "item_to_cate": item_to_cate,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Train data saved: {train_path} ({len(train_samples):,} samples)")

    # 保存测试数据
    test_path = os.path.join(output_dir, "test_data.pkl")
    with open(test_path, "wb") as f:
        pickle.dump({
            "samples": test_samples,
            "item_to_cate": item_to_cate,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Test data saved: {test_path} ({len(test_samples):,} samples)")

    # 保存元数据
    meta_path = os.path.join(output_dir, "meta.pkl")
    meta = {
        "num_users": num_users,
        "num_items": num_items,
        "num_categories": num_categories,
        "num_train_samples": len(train_samples),
        "num_test_samples": len(test_samples),
    }
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Metadata saved: {meta_path}")
    logger.info(f"  num_users: {num_users:,}")
    logger.info(f"  num_items: {num_items:,}")
    logger.info(f"  num_categories: {num_categories:,}")
    logger.info(f"  num_train_samples: {len(train_samples):,}")
    logger.info(f"  num_test_samples: {len(test_samples):,}")


def process_real_data(args):
    """处理真实 UserBehavior.csv 数据。"""
    start_time = time.time()

    # 1. 加载原始数据
    records = load_raw_data(
        raw_data_path=args.raw_data_path,
        max_rows=args.max_rows,
        behavior_filter=args.behavior_filter,
    )

    # 2. ID 重映射
    remapped_records, user_map, item_map, cate_map = remap_ids(records)
    num_users = len(user_map)
    num_items = len(item_map)
    num_categories = len(cate_map)

    # 3. 构建用户序列
    user_seqs = build_user_sequences(remapped_records)

    # 4. 构建 item -> cate 映射
    item_to_cate = build_item_to_cate(remapped_records)

    # 5. 构建训练/测试样本
    train_samples, test_samples = build_dataset(
        user_seqs=user_seqs,
        item_to_cate=item_to_cate,
        num_items=num_items,
        neg_ratio=args.neg_ratio,
        min_hist_len=args.min_hist_len,
        seed=args.seed,
    )

    # 6. 保存
    save_processed_data(
        output_dir=args.output_dir,
        train_samples=train_samples,
        test_samples=test_samples,
        item_to_cate=item_to_cate,
        num_users=num_users,
        num_items=num_items,
        num_categories=num_categories,
    )

    elapsed = time.time() - start_time
    logger.info(f"Total preprocessing time: {elapsed:.1f}s")


def process_mock_data(args):
    """生成并保存模拟数据。"""
    start_time = time.time()

    train_samples, test_samples, item_to_cate, num_users, num_items, num_categories = (
        generate_mock_data(
            num_users=args.mock_users,
            num_items=args.mock_items,
            num_categories=args.mock_categories,
            avg_behaviors_per_user=args.mock_avg_behaviors,
            seed=args.seed,
        )
    )

    save_processed_data(
        output_dir=args.output_dir,
        train_samples=train_samples,
        test_samples=test_samples,
        item_to_cate=item_to_cate,
        num_users=num_users,
        num_items=num_items,
        num_categories=num_categories,
    )

    elapsed = time.time() - start_time
    logger.info(f"Total mock data generation time: {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Alibaba UserBehavior Dataset Preprocessing"
    )

    # 数据路径
    parser.add_argument(
        "--raw_data_path",
        type=str,
        default=None,
        help="Path to UserBehavior.csv (原始数据路径)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/alibaba",
        help="Output directory for processed data",
    )

    # 过滤和采样参数
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Maximum rows to read (None = all)",
    )
    parser.add_argument(
        "--behavior_filter",
        type=str,
        default="pv",
        help="Behavior type filter (pv/buy/cart/fav, None = all)",
    )
    parser.add_argument(
        "--neg_ratio",
        type=int,
        default=1,
        help="Number of negative samples per positive sample",
    )
    parser.add_argument(
        "--min_hist_len",
        type=int,
        default=3,
        help="Minimum history length to include a user",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # 模拟数据参数
    parser.add_argument(
        "--generate_mock",
        action="store_true",
        help="Generate mock data instead of processing real data",
    )
    parser.add_argument(
        "--mock_users",
        type=int,
        default=10000,
        help="Number of mock users",
    )
    parser.add_argument(
        "--mock_items",
        type=int,
        default=50000,
        help="Number of mock items",
    )
    parser.add_argument(
        "--mock_categories",
        type=int,
        default=500,
        help="Number of mock categories",
    )
    parser.add_argument(
        "--mock_avg_behaviors",
        type=int,
        default=20,
        help="Average behaviors per mock user",
    )

    args = parser.parse_args()

    if args.generate_mock:
        logger.info("=" * 60)
        logger.info("Generating Mock Alibaba UserBehavior Data")
        logger.info("=" * 60)
        process_mock_data(args)
    elif args.raw_data_path:
        if not os.path.exists(args.raw_data_path):
            logger.error(f"Raw data file not found: {args.raw_data_path}")
            logger.error(
                "请从天池下载 UserBehavior.csv:\n"
                "  https://tianchi.aliyun.com/dataset/649\n"
                "或使用 --generate_mock 生成模拟数据"
            )
            sys.exit(1)

        logger.info("=" * 60)
        logger.info("Processing Real Alibaba UserBehavior Data")
        logger.info("=" * 60)
        process_real_data(args)
    else:
        logger.error(
            "请指定数据源:\n"
            "  --raw_data_path /path/to/UserBehavior.csv  (真实数据)\n"
            "  --generate_mock                            (模拟数据)\n"
            "\n真实数据下载地址: https://tianchi.aliyun.com/dataset/649"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
