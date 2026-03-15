"""
MixFormer 训练入口脚本。

实现 Trainer 类（训练循环、验证循环、指标计算、checkpoint 保存/加载、
混合精度训练支持），使用 Alibaba UserBehavior 数据集。

Usage:
    # 默认配置训练
    python train.py --data_dir data/alibaba --epochs 5 --batch_size 4096

    # Medium 配置训练
    python train.py --config medium --data_dir data/alibaba --epochs 5

    # 混合精度训练
    python train.py --data_dir data/alibaba --epochs 5 --batch_size 4096 --amp
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mixformer.config import MixFormerConfig
from mixformer.data import create_dataloader
from mixformer.model import MixFormer

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("MixFormer.Train")


def compute_auc(labels: np.ndarray, predictions: np.ndarray) -> float:
    """计算 AUC (Area Under ROC Curve)。"""
    try:
        from sklearn.metrics import roc_auc_score
        import warnings

        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return 0.5

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return roc_auc_score(labels, predictions)
    except ImportError:
        pos_indices = np.where(labels == 1)[0]
        neg_indices = np.where(labels == 0)[0]

        if len(pos_indices) == 0 or len(neg_indices) == 0:
            return 0.5

        pos_preds = predictions[pos_indices]
        neg_preds = predictions[neg_indices]

        count = 0
        for p in pos_preds:
            count += np.sum(neg_preds < p) + 0.5 * np.sum(neg_preds == p)

        auc = count / (len(pos_preds) * len(neg_preds))
        return auc


def compute_logloss(labels: np.ndarray, predictions: np.ndarray) -> float:
    """计算 Log Loss (Binary Cross-Entropy)。"""
    eps = 1e-7
    predictions = np.clip(predictions, eps, 1 - eps)
    logloss = -np.mean(
        labels * np.log(predictions) + (1 - labels) * np.log(1 - predictions)
    )
    return logloss


class Trainer:
    """MixFormer 训练器。

    封装训练循环、验证循环、指标计算、学习率调度、
    混合精度训练支持、模型 checkpoint 保存/加载。

    Args:
        model: MixFormer 模型
        config: 模型配置
        train_loader: 训练数据 DataLoader
        val_loader: 验证数据 DataLoader
        lr: 学习率
        weight_decay: 权重衰减
        epochs: 训练轮数
        save_dir: checkpoint 保存目录
        use_amp: 是否使用混合精度训练
        device: 计算设备
        log_interval: 日志打印间隔（步数）
    """

    def __init__(
        self,
        model: nn.Module,
        config: MixFormerConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        lr: float = 0.01,
        weight_decay: float = 1e-5,
        epochs: int = 10,
        save_dir: str = "checkpoints",
        use_amp: bool = False,
        device: str = "cpu",
        log_interval: int = 50,
    ):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.use_amp = use_amp
        self.device = device
        self.log_interval = log_interval

        # 损失函数: BCE Loss
        self.criterion = nn.BCELoss()

        # 优化器: Adam (工业级推荐系统常用)
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # 学习率调度: 带 warmup 的余弦退火
        total_steps = len(train_loader) * epochs
        warmup_steps = min(total_steps // 10, 500)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1 + np.cos(np.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lr_lambda
        )

        # 混合精度
        self.scaler = torch.amp.GradScaler(enabled=use_amp)

        # 训练状态
        self.global_step = 0
        self.best_auc = 0.0

    def _move_batch_to_device(self, batch: dict) -> dict:
        """将 batch 数据移动到指定设备。"""
        return {
            "target_item_id": batch["target_item_id"].to(self.device),
            "target_cate_id": batch["target_cate_id"].to(self.device),
            "hist_item_ids": batch["hist_item_ids"].to(self.device),
            "hist_cate_ids": batch["hist_cate_ids"].to(self.device),
            "seq_mask": batch["seq_mask"].to(self.device),
            "label": batch["label"].to(self.device),
        }

    def _forward_batch(self, batch: dict) -> torch.Tensor:
        """执行模型前向传播。"""
        return self.model(
            target_item_id=batch["target_item_id"],
            target_cate_id=batch["target_cate_id"],
            hist_item_ids=batch["hist_item_ids"],
            hist_cate_ids=batch["hist_cate_ids"],
            seq_mask=batch["seq_mask"],
        )

    def train_epoch(self, epoch: int) -> dict:
        """训练一个 epoch。"""
        self.model.train()
        total_loss = 0.0
        all_labels = []
        all_preds = []
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            batch = self._move_batch_to_device(batch)

            self.optimizer.zero_grad()

            # 前向传播
            device_type = self.device.split(":")[0] if ":" in self.device else self.device
            with torch.amp.autocast(
                device_type=device_type,
                enabled=self.use_amp,
            ):
                pred = self._forward_batch(batch)  # (batch, 1)
                loss = self.criterion(pred.squeeze(-1), batch["label"])

            # 反向传播
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            # 记录
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            all_labels.append(batch["label"].detach().cpu().numpy())
            all_preds.append(pred.squeeze(-1).detach().cpu().numpy())

            # 日志
            if (batch_idx + 1) % self.log_interval == 0:
                avg_loss = total_loss / num_batches
                lr = self.scheduler.get_last_lr()[0]
                logger.info(
                    f"Epoch {epoch+1} | Step {batch_idx+1}/{len(self.train_loader)} | "
                    f"Loss: {avg_loss:.6f} | LR: {lr:.6f}"
                )

        # Epoch 级别指标
        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_preds)
        epoch_loss = total_loss / num_batches
        epoch_auc = compute_auc(all_labels, all_preds)
        epoch_logloss = compute_logloss(all_labels, all_preds)

        return {
            "loss": epoch_loss,
            "auc": epoch_auc,
            "logloss": epoch_logloss,
        }

    @torch.no_grad()
    def validate(self) -> dict:
        """验证模型。"""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        all_labels = []
        all_preds = []
        num_batches = 0

        for batch in self.val_loader:
            batch = self._move_batch_to_device(batch)

            device_type = self.device.split(":")[0] if ":" in self.device else self.device
            with torch.amp.autocast(
                device_type=device_type,
                enabled=self.use_amp,
            ):
                pred = self._forward_batch(batch)
                loss = self.criterion(pred.squeeze(-1), batch["label"])

            total_loss += loss.item()
            num_batches += 1
            all_labels.append(batch["label"].cpu().numpy())
            all_preds.append(pred.squeeze(-1).cpu().numpy())

        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_preds)
        val_loss = total_loss / num_batches
        val_auc = compute_auc(all_labels, all_preds)
        val_logloss = compute_logloss(all_labels, all_preds)

        return {
            "val_loss": val_loss,
            "val_auc": val_auc,
            "val_logloss": val_logloss,
        }

    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """保存模型 checkpoint。"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "global_step": self.global_step,
            "metrics": metrics,
            "config": {
                "num_heads": self.config.num_heads,
                "num_layers": self.config.num_layers,
                "hidden_dim": self.config.hidden_dim,
                "seq_length": self.config.seq_length,
                "num_non_seq_features": self.config.num_non_seq_features,
                "feature_embed_dim": self.config.feature_embed_dim,
                "num_items": self.config.num_items,
                "num_users": self.config.num_users,
                "ffn_multiplier": self.config.ffn_multiplier,
                "dropout": self.config.dropout,
                "user_heads": self.config.user_heads,
                "item_heads": self.config.item_heads,
                "num_categories": self.config.num_categories,
                "sparse_feature_names": self.config.sparse_feature_names,
                "sparse_vocab_sizes": self.config.sparse_vocab_sizes,
                "sparse_embed_dim": self.config.sparse_embed_dim,
                "use_torchrec": self.config.use_torchrec,
                "target_item_mlp_dims": self.config.target_item_mlp_dims,
            },
        }

        # 保存最新 checkpoint
        path = self.save_dir / f"checkpoint_epoch_{epoch+1}.pt"
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")

        # 保存最佳模型
        if is_best:
            best_path = self.save_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved: {best_path}")

    def load_checkpoint(self, path: str):
        """加载模型 checkpoint。"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.global_step = checkpoint["global_step"]
        logger.info(f"Checkpoint loaded from: {path}, epoch: {checkpoint['epoch']+1}")
        return checkpoint

    def train(self):
        """完整训练流程。"""
        logger.info("=" * 60)
        logger.info("MixFormer Training Started")
        logger.info(f"Model: {self.model.__class__.__name__}")
        logger.info(f"Config: {self.config}")
        logger.info(f"Total parameters: {self.model.get_num_params():,}")
        logger.info(f"Trainable parameters: {self.model.get_num_trainable_params():,}")
        logger.info(f"Device: {self.device}")
        logger.info(f"AMP: {self.use_amp}")
        logger.info(f"Epochs: {self.epochs}")
        logger.info(f"Train batches: {len(self.train_loader)}")
        if self.val_loader:
            logger.info(f"Val batches: {len(self.val_loader)}")
        logger.info("=" * 60)

        training_history = []

        for epoch in range(self.epochs):
            epoch_start = time.time()

            # 训练
            train_metrics = self.train_epoch(epoch)
            epoch_time = time.time() - epoch_start

            # 验证
            val_metrics = self.validate()

            # 合并指标
            metrics = {**train_metrics, **val_metrics, "epoch_time": epoch_time}
            training_history.append(metrics)

            # 判断是否最佳
            current_auc = val_metrics.get("val_auc", train_metrics["auc"])
            is_best = current_auc > self.best_auc
            if is_best:
                self.best_auc = current_auc

            # 保存 checkpoint
            self.save_checkpoint(epoch, metrics, is_best=is_best)

            # 打印 Epoch 总结
            logger.info("-" * 60)
            logger.info(f"Epoch {epoch+1}/{self.epochs} | Time: {epoch_time:.1f}s")
            logger.info(
                f"  Train - Loss: {train_metrics['loss']:.6f} | "
                f"AUC: {train_metrics['auc']:.4f} | "
                f"LogLoss: {train_metrics['logloss']:.6f}"
            )
            if val_metrics:
                logger.info(
                    f"  Valid - Loss: {val_metrics['val_loss']:.6f} | "
                    f"AUC: {val_metrics['val_auc']:.4f} | "
                    f"LogLoss: {val_metrics['val_logloss']:.6f}"
                )
            if is_best:
                logger.info(f"  ★ New best AUC: {current_auc:.4f}")
            logger.info("-" * 60)

        # 保存训练历史
        history_path = self.save_dir / "training_history.json"
        serializable_history = []
        for entry in training_history:
            serializable_history.append(
                {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                 for k, v in entry.items()}
            )
        with open(history_path, "w") as f:
            json.dump(serializable_history, f, indent=2)
        logger.info(f"Training history saved: {history_path}")

        logger.info("=" * 60)
        logger.info(f"Training completed! Best AUC: {self.best_auc:.4f}")
        logger.info("=" * 60)

        return training_history


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="MixFormer Training Script")

    # 数据目录
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/alibaba",
        help="Data directory (contains train_data.pkl, test_data.pkl, meta.pkl)",
    )

    # 模型配置
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        choices=["default", "medium"],
        help="Model configuration preset (default: default)",
    )

    # 训练参数
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--weight_decay", type=float, default=1e-5, help="Weight decay"
    )
    parser.add_argument("--amp", action="store_true", help="Use AMP (mixed precision)")

    # 数据参数
    parser.add_argument(
        "--num_workers", type=int, default=0, help="DataLoader num_workers"
    )

    # 其他
    parser.add_argument(
        "--save_dir", type=str, default="checkpoints", help="Checkpoint directory"
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device (cpu/cuda/mps/auto)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--log_interval", type=int, default=50, help="Log interval (steps)"
    )

    return parser.parse_args()


def get_device(device_arg: str) -> str:
    """自动选择最佳可用设备。"""
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

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 设备
    device = get_device(args.device)
    logger.info(f"Using device: {device}")

    # 加载元数据
    logger.info("=" * 60)
    logger.info("MixFormer Training with Alibaba UserBehavior Dataset")
    logger.info("=" * 60)

    meta_path = os.path.join(args.data_dir, "meta.pkl")
    if not os.path.exists(meta_path):
        logger.error(
            f"Metadata not found: {meta_path}\n"
            f"Please run preprocessing first:\n"
            f"  python scripts/download_alibaba.py --generate_mock --output_dir {args.data_dir}\n"
            f"  or\n"
            f"  python scripts/download_alibaba.py --raw_data_path /path/to/UserBehavior.csv --output_dir {args.data_dir}"
        )
        return

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    logger.info(f"Dataset metadata: {meta}")

    # 创建配置 (使用实际数据的元信息)
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

    logger.info(f"Model config: {config}")

    # 创建数据加载器
    train_loader = create_dataloader(
        data_dir=args.data_dir,
        config=config,
        split="train",
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    # 测试集作为验证
    val_loader = create_dataloader(
        data_dir=args.data_dir,
        config=config,
        split="test",
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # 创建模型
    model = MixFormer(config)

    logger.info(f"Model type: {model.__class__.__name__}")
    logger.info(f"Total parameters: {model.get_num_params():,}")

    # 创建训练器
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        save_dir=args.save_dir,
        use_amp=args.amp,
        device=device,
        log_interval=args.log_interval,
    )

    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()
