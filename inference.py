"""
MixFormer 推理入口脚本。

使用 Alibaba UserBehavior 数据集进行推理评估。

Usage:
    # 从 checkpoint 推理
    python inference.py --checkpoint checkpoints/best_model.pt --data_dir data/alibaba

    # 不加载 checkpoint，直接用随机权重测试推理流程
    python inference.py --data_dir data/alibaba --no_checkpoint
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn

from mixformer.config import MixFormerConfig
from mixformer.data import create_dataloader
from mixformer.model import MixFormer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("MixFormer.Inference")


class Predictor:
    """MixFormer 推理器。

    Args:
        model: MixFormer 模型
        device: 计算设备
    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    def _move_batch_to_device(self, batch: dict) -> dict:
        """将 batch 数据移动到指定设备。"""
        result = {
            "target_item_id": batch["target_item_id"].to(self.device),
            "target_cate_id": batch["target_cate_id"].to(self.device),
            "hist_item_ids": batch["hist_item_ids"].to(self.device),
            "hist_cate_ids": batch["hist_cate_ids"].to(self.device),
            "seq_mask": batch["seq_mask"].to(self.device),
        }
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
            target_item_id=batch["target_item_id"],
            target_cate_id=batch["target_cate_id"],
            hist_item_ids=batch["hist_item_ids"],
            hist_cate_ids=batch["hist_cate_ids"],
            seq_mask=batch["seq_mask"],
        )
        return pred.squeeze(-1).cpu()

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

    config_dict = checkpoint["config"]

    # 过滤掉已移除的旧字段 (兼容旧版 checkpoint)
    removed_keys = {"dataset_type", "vocab_sizes"}
    config_dict = {k: v for k, v in config_dict.items() if k not in removed_keys}

    # 重建配置
    config = MixFormerConfig(**config_dict)

    # 创建模型并加载权重
    model = MixFormer(config)
    model.load_state_dict(checkpoint["model_state_dict"])

    logger.info(f"Model loaded from {checkpoint_path}")
    logger.info(f"  Model type: {model.__class__.__name__}")
    logger.info(f"  Epoch: {checkpoint['epoch'] + 1}")
    logger.info(f"  Metrics: {checkpoint.get('metrics', {})}")

    return model, config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MixFormer Inference Script")

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/alibaba",
        help="Data directory (contains test_data.pkl, meta.pkl)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        choices=["default", "medium"],
        help="Model config preset (used when no checkpoint)",
    )
    parser.add_argument(
        "--no_checkpoint",
        action="store_true",
        help="Run without checkpoint (random weights)",
    )
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--device", type=str, default="auto", help="Device")

    return parser.parse_args()


def get_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Detected Apple MPS device, using MPS for acceleration.")
        return "mps"
    return "cpu"


def main():
    args = parse_args()

    device = get_device(args.device)
    logger.info(f"Using device: {device}")

    if args.checkpoint and not args.no_checkpoint:
        # 从 checkpoint 加载模型
        if not os.path.exists(args.checkpoint):
            logger.error(f"Checkpoint not found: {args.checkpoint}")
            # 列出可用的 checkpoint
            for d in ["checkpoints", "checkpoints_alibaba"]:
                if os.path.isdir(d):
                    for f in os.listdir(d):
                        if f.endswith(".pt"):
                            logger.error(f"  Available: {os.path.join(d, f)}")
            return

        model, config = load_model_from_checkpoint(args.checkpoint, device)
    else:
        # 随机权重测试
        logger.info("Running with random weights (no checkpoint)")

        meta_path = os.path.join(args.data_dir, "meta.pkl")
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            if args.config == "medium":
                config = MixFormerConfig.medium(
                    num_items=meta["num_items"] + 1,
                    num_categories=meta["num_categories"] + 1,
                    num_users=meta["num_users"] + 1,
                )
            else:
                config = MixFormerConfig.default(
                    num_items=meta["num_items"] + 1,
                    num_categories=meta["num_categories"] + 1,
                    num_users=meta["num_users"] + 1,
                )
        else:
            if args.config == "medium":
                config = MixFormerConfig.medium()
            else:
                config = MixFormerConfig.default()

        model = MixFormer(config)

    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Parameters: {model.get_num_params():,}")

    # 创建数据加载器
    test_loader = create_dataloader(
        data_dir=args.data_dir,
        config=config,
        split="test",
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # 创建推理器
    predictor = Predictor(model, device)

    # 标准推理
    logger.info("Running inference...")
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


if __name__ == "__main__":
    main()
