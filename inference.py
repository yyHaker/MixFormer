"""
MixFormer 推理入口脚本。

支持标准推理和 UI-MixFormer 解耦推理（用户侧编码缓存 + 物品侧快速打分）。

Usage:
    # 标准推理
    python inference.py --checkpoint checkpoints/best_model.pt --mode standard

    # 解耦推理 (UI-MixFormer)
    python inference.py --checkpoint checkpoints/best_model.pt --mode decoupled

    # 不加载 checkpoint，直接用随机权重测试推理流程
    python inference.py --config small --mode standard --no_checkpoint
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from mixformer.config import MixFormerConfig
from mixformer.data import SyntheticRecDataset, collate_fn, create_dataloader
from mixformer.model import MixFormer, UIMixFormer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("MixFormer.Inference")


class Predictor:
    """MixFormer 推理器。

    支持标准推理和解耦推理两种模式。

    Args:
        model: MixFormer 或 UIMixFormer 模型
        device: 计算设备
    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    def _move_batch_to_device(self, batch: dict) -> dict:
        """将 batch 数据移动到指定设备。"""
        result = {}
        result["non_seq_features"] = {
            k: v.to(self.device) for k, v in batch["non_seq_features"].items()
        }
        result["seq_features"] = batch["seq_features"].to(self.device)
        result["seq_mask"] = batch["seq_mask"].to(self.device)
        if "label" in batch:
            result["label"] = batch["label"].to(self.device)
        return result

    @torch.no_grad()
    def predict(self, batch: dict) -> torch.Tensor:
        """标准推理: 输入一个 batch，返回 CTR 预测。

        Args:
            batch: 数据批量字典

        Returns:
            predictions: (batch_size,) CTR 预测概率
        """
        batch = self._move_batch_to_device(batch)
        pred = self.model(
            non_seq_features=batch["non_seq_features"],
            seq_features=batch["seq_features"],
            seq_mask=batch["seq_mask"],
        )
        return pred.squeeze(-1).cpu()

    @torch.no_grad()
    def predict_decoupled(
        self,
        user_batch: dict,
        item_batches: list[dict],
    ) -> list[torch.Tensor]:
        """解耦推理: 先编码用户，再对多个候选物品集快速打分。

        仅适用于 UIMixFormer。

        Args:
            user_batch: 包含用户特征和序列的 batch
            item_batches: 多个候选物品 batch 的列表

        Returns:
            predictions_list: 每个物品 batch 的 CTR 预测列表
        """
        assert isinstance(
            self.model, UIMixFormer
        ), "Decoupled inference requires UIMixFormer"

        user_batch = self._move_batch_to_device(user_batch)

        # 1. 编码用户侧（可缓存）
        user_repr = self.model.encode_user(
            non_seq_features=user_batch["non_seq_features"],
            seq_features=user_batch["seq_features"],
            seq_mask=user_batch["seq_mask"],
        )  # (batch, N_U, D)

        # 2. 对每个候选物品集快速打分
        predictions_list = []
        for item_batch in item_batches:
            item_batch = self._move_batch_to_device(item_batch)
            pred = self.model.predict_with_user_cache(
                user_repr=user_repr,
                item_non_seq_features=item_batch["non_seq_features"],
                seq_features=item_batch["seq_features"],
                seq_mask=item_batch["seq_mask"],
            )
            predictions_list.append(pred.squeeze(-1).cpu())

        return predictions_list

    @torch.no_grad()
    def batch_predict(
        self,
        dataloader,
        return_labels: bool = False,
    ) -> dict:
        """批量推理整个数据集。

        Args:
            dataloader: DataLoader 实例
            return_labels: 是否返回真实标签

        Returns:
            dict 包含 predictions 和可选的 labels
        """
        all_preds = []
        all_labels = []

        start_time = time.time()

        for batch in dataloader:
            pred = self.predict(batch)
            all_preds.append(pred.numpy())
            if return_labels and "label" in batch:
                all_labels.append(batch["label"].numpy())

        elapsed = time.time() - start_time
        all_preds = np.concatenate(all_preds)

        result = {
            "predictions": all_preds,
            "num_samples": len(all_preds),
            "inference_time": elapsed,
            "samples_per_second": len(all_preds) / elapsed,
        }

        if return_labels and all_labels:
            result["labels"] = np.concatenate(all_labels)

        return result


def load_model_from_checkpoint(
    checkpoint_path: str, device: str = "cpu"
) -> tuple[nn.Module, MixFormerConfig]:
    """从 checkpoint 加载模型。

    Args:
        checkpoint_path: checkpoint 文件路径
        device: 计算设备

    Returns:
        (model, config) 元组
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 重建配置
    config_dict = checkpoint["config"]
    config = MixFormerConfig(**config_dict)

    # 判断模型类型（通过检查是否有解耦掩码）
    state_dict = checkpoint["model_state_dict"]
    if "decouple_mask" in state_dict:
        model = UIMixFormer(config)
    else:
        model = MixFormer(config)

    model.load_state_dict(state_dict)
    logger.info(f"Model loaded from {checkpoint_path}")
    logger.info(f"  Epoch: {checkpoint['epoch'] + 1}")
    logger.info(f"  Metrics: {checkpoint.get('metrics', {})}")

    return model, config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MixFormer Inference Script")

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="small",
        choices=["small", "medium"],
        help="Model config preset (used when no checkpoint)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="mixformer",
        choices=["mixformer", "ui_mixformer"],
        help="Model type (used when no checkpoint)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="standard",
        choices=["standard", "decoupled"],
        help="Inference mode",
    )
    parser.add_argument(
        "--no_checkpoint",
        action="store_true",
        help="Run without checkpoint (random weights)",
    )
    parser.add_argument(
        "--num_samples", type=int, default=1000, help="Number of test samples"
    )
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--device", type=str, default="auto", help="Device")

    return parser.parse_args()


def get_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    args = parse_args()

    device = get_device(args.device)
    logger.info(f"Using device: {device}")

    # 加载模型
    if args.checkpoint and not args.no_checkpoint:
        model, config = load_model_from_checkpoint(args.checkpoint, device)
    else:
        logger.info("Running with random weights (no checkpoint)")
        if args.config == "small":
            config = MixFormerConfig.small()
        else:
            config = MixFormerConfig.medium()

        if args.mode == "decoupled" or args.model_type == "ui_mixformer":
            model = UIMixFormer(config)
        else:
            model = MixFormer(config)

    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Parameters: {model.get_num_params():,}")

    # 创建数据加载器
    test_loader = create_dataloader(
        config,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        shuffle=False,
        seed=123,
    )

    # 创建推理器
    predictor = Predictor(model, device)

    if args.mode == "standard":
        # 标准推理
        logger.info("Running standard inference...")
        result = predictor.batch_predict(test_loader, return_labels=True)

        logger.info(f"Inference Results:")
        logger.info(f"  Samples: {result['num_samples']}")
        logger.info(f"  Time: {result['inference_time']:.3f}s")
        logger.info(f"  Throughput: {result['samples_per_second']:.0f} samples/s")
        logger.info(
            f"  Prediction range: [{result['predictions'].min():.4f}, "
            f"{result['predictions'].max():.4f}]"
        )
        logger.info(f"  Prediction mean: {result['predictions'].mean():.4f}")

        if "labels" in result:
            from train import compute_auc, compute_logloss

            auc = compute_auc(result["labels"], result["predictions"])
            logloss = compute_logloss(result["labels"], result["predictions"])
            logger.info(f"  AUC: {auc:.4f}")
            logger.info(f"  LogLoss: {logloss:.6f}")

    elif args.mode == "decoupled":
        # 解耦推理
        assert isinstance(
            model, UIMixFormer
        ), "Decoupled mode requires UIMixFormer (use --model_type ui_mixformer)"

        logger.info("Running decoupled inference...")

        # 模拟: 每个用户对 3 组候选物品打分
        dataset = SyntheticRecDataset(config, num_samples=args.batch_size, seed=123)
        user_batch = collate_fn([dataset[i] for i in range(min(32, len(dataset)))])

        # 生成多组候选物品
        item_datasets = [
            SyntheticRecDataset(config, num_samples=32, seed=200 + i) for i in range(3)
        ]
        item_batches = [
            collate_fn([ds[j] for j in range(len(ds))]) for ds in item_datasets
        ]

        start_time = time.time()
        predictions_list = predictor.predict_decoupled(user_batch, item_batches)
        elapsed = time.time() - start_time

        logger.info(f"Decoupled Inference Results:")
        logger.info(f"  Users: {user_batch['seq_features'].size(0)}")
        logger.info(f"  Candidate item groups: {len(item_batches)}")
        logger.info(f"  Total time: {elapsed:.3f}s")
        for i, preds in enumerate(predictions_list):
            logger.info(
                f"  Item group {i}: mean={preds.mean():.4f}, "
                f"range=[{preds.min():.4f}, {preds.max():.4f}]"
            )


if __name__ == "__main__":
    main()
