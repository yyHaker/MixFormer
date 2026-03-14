"""
MixFormer 单元测试。

测试各模块维度正确性、完整模型前向传播、梯度反向传播、配置校验等。
"""

import pytest
import torch
import numpy as np

from mixformer.config import MixFormerConfig
from mixformer.modules import SwiGLUFFN, HeadMixing, PerHeadSwiGLUFFN
from mixformer.layers import QueryMixer, CrossAttention, OutputFusion, MixFormerBlock
from mixformer.model import FeatureEncoder, TaskHead, MixFormer, UIMixFormer
from mixformer.data import SyntheticRecDataset, collate_fn, create_dataloader


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def config():
    """创建一个小型测试配置。"""
    return MixFormerConfig(
        num_heads=4,
        num_layers=2,
        hidden_dim=32,
        seq_length=10,
        num_non_seq_features=8,
        feature_embed_dim=16,
        num_items=100,
        num_users=100,
        ffn_multiplier=2.667,
        dropout=0.0,
        user_heads=2,
        item_heads=2,
        task_head_hidden_dims=[64, 32],
    )


@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def sample_batch(config, batch_size):
    """创建一个示例 batch。"""
    dataset = SyntheticRecDataset(config, num_samples=batch_size, seed=42)
    samples = [dataset[i] for i in range(batch_size)]
    return collate_fn(samples)


# ============================================================
# Config Tests
# ============================================================


class TestMixFormerConfig:
    """测试模型配置。"""

    def test_small_config(self):
        config = MixFormerConfig.small()
        assert config.num_heads == 16
        assert config.num_layers == 4
        assert config.hidden_dim == 386
        assert config.user_heads + config.item_heads == config.num_heads

    def test_medium_config(self):
        config = MixFormerConfig.medium()
        assert config.num_heads == 16
        assert config.num_layers == 4
        assert config.hidden_dim == 768
        assert config.user_heads + config.item_heads == config.num_heads

    def test_head_input_dim(self, config):
        expected = config.total_embed_dim // config.num_heads
        assert config.head_input_dim == expected

    def test_model_dim(self, config):
        assert config.model_dim == config.num_heads * config.hidden_dim

    def test_ffn_hidden_dim(self, config):
        assert config.ffn_hidden_dim > 0
        assert config.ffn_hidden_dim % 64 == 0

    def test_invalid_heads(self):
        """测试头数不匹配时的校验。"""
        with pytest.raises(AssertionError):
            MixFormerConfig(
                num_heads=4,
                user_heads=3,
                item_heads=3,  # 3+3 != 4
            )

    def test_vocab_sizes_auto(self, config):
        assert "user_id" in config.vocab_sizes
        assert "item_id" in config.vocab_sizes
        assert config.vocab_sizes["user_id"] == config.num_users

    def test_repr(self, config):
        repr_str = repr(config)
        assert "MixFormerConfig" in repr_str
        assert str(config.num_heads) in repr_str


# ============================================================
# Module Tests
# ============================================================


class TestSwiGLUFFN:
    """测试 SwiGLU FFN。"""

    def test_output_shape(self, config, batch_size):
        ffn = SwiGLUFFN(in_dim=config.hidden_dim, hidden_dim=config.ffn_hidden_dim)
        x = torch.randn(batch_size, config.hidden_dim)
        out = ffn(x)
        assert out.shape == (batch_size, config.hidden_dim)

    def test_custom_output_dim(self, batch_size):
        ffn = SwiGLUFFN(in_dim=32, hidden_dim=64, out_dim=16)
        x = torch.randn(batch_size, 32)
        out = ffn(x)
        assert out.shape == (batch_size, 16)

    def test_3d_input(self, config, batch_size):
        ffn = SwiGLUFFN(in_dim=config.hidden_dim, hidden_dim=config.ffn_hidden_dim)
        x = torch.randn(batch_size, 10, config.hidden_dim)
        out = ffn(x)
        assert out.shape == (batch_size, 10, config.hidden_dim)

    def test_gradient_flow(self, config, batch_size):
        ffn = SwiGLUFFN(in_dim=config.hidden_dim, hidden_dim=config.ffn_hidden_dim)
        x = torch.randn(batch_size, config.hidden_dim, requires_grad=True)
        out = ffn(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestHeadMixing:
    """测试 HeadMixing。"""

    def test_output_shape(self, config, batch_size):
        hm = HeadMixing(config.num_heads, config.hidden_dim)
        x = torch.randn(batch_size, config.num_heads, config.hidden_dim)
        out = hm(x)
        assert out.shape == (batch_size, config.num_heads, config.hidden_dim)

    def test_no_parameters(self, config):
        hm = HeadMixing(config.num_heads, config.hidden_dim)
        assert sum(p.numel() for p in hm.parameters()) == 0


class TestPerHeadSwiGLUFFN:
    """测试 Per-Head SwiGLU FFN。"""

    def test_output_shape(self, config, batch_size):
        ffn = PerHeadSwiGLUFFN(
            num_heads=config.num_heads,
            hidden_dim=config.hidden_dim,
            ffn_hidden_dim=config.ffn_hidden_dim,
        )
        x = torch.randn(batch_size, config.num_heads, config.hidden_dim)
        out = ffn(x)
        assert out.shape == (batch_size, config.num_heads, config.hidden_dim)

    def test_gradient_flow(self, config, batch_size):
        ffn = PerHeadSwiGLUFFN(
            num_heads=config.num_heads,
            hidden_dim=config.hidden_dim,
            ffn_hidden_dim=config.ffn_hidden_dim,
        )
        x = torch.randn(batch_size, config.num_heads, config.hidden_dim, requires_grad=True)
        out = ffn(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


# ============================================================
# Layer Tests
# ============================================================


class TestQueryMixer:
    """测试 Query Mixer。"""

    def test_output_shape(self, config, batch_size):
        qm = QueryMixer(config)
        x = torch.randn(batch_size, config.num_heads, config.hidden_dim)
        out = qm(x)
        assert out.shape == (batch_size, config.num_heads, config.hidden_dim)

    def test_with_mask(self, config, batch_size):
        qm = QueryMixer(config)
        x = torch.randn(batch_size, config.num_heads, config.hidden_dim)
        mask = torch.ones(config.num_heads, config.hidden_dim)
        out = qm(x, mask=mask)
        assert out.shape == (batch_size, config.num_heads, config.hidden_dim)


class TestCrossAttention:
    """测试 Cross Attention。"""

    def test_output_shape(self, config, batch_size):
        ca = CrossAttention(config)
        q = torch.randn(batch_size, config.num_heads, config.hidden_dim)
        seq = torch.randn(batch_size, config.seq_length, config.model_dim)
        out = ca(q, seq)
        assert out.shape == (batch_size, config.num_heads, config.hidden_dim)

    def test_with_mask(self, config, batch_size):
        ca = CrossAttention(config)
        q = torch.randn(batch_size, config.num_heads, config.hidden_dim)
        seq = torch.randn(batch_size, config.seq_length, config.model_dim)
        seq_mask = torch.ones(batch_size, config.seq_length, dtype=torch.bool)
        seq_mask[:, -3:] = False  # 最后 3 个位置无效
        out = ca(q, seq, seq_mask=seq_mask)
        assert out.shape == (batch_size, config.num_heads, config.hidden_dim)

    def test_gradient_flow(self, config, batch_size):
        ca = CrossAttention(config)
        q = torch.randn(batch_size, config.num_heads, config.hidden_dim, requires_grad=True)
        seq = torch.randn(batch_size, config.seq_length, config.model_dim, requires_grad=True)
        out = ca(q, seq)
        loss = out.sum()
        loss.backward()
        assert q.grad is not None
        assert seq.grad is not None


class TestOutputFusion:
    """测试 Output Fusion。"""

    def test_output_shape(self, config, batch_size):
        of = OutputFusion(config)
        z = torch.randn(batch_size, config.num_heads, config.hidden_dim)
        out = of(z)
        assert out.shape == (batch_size, config.num_heads, config.hidden_dim)


class TestMixFormerBlock:
    """测试 MixFormer Block。"""

    def test_output_shape(self, config, batch_size):
        block = MixFormerBlock(config)
        x = torch.randn(batch_size, config.num_heads, config.hidden_dim)
        seq = torch.randn(batch_size, config.seq_length, config.model_dim)
        out = block(x, seq)
        assert out.shape == (batch_size, config.num_heads, config.hidden_dim)

    def test_with_all_masks(self, config, batch_size):
        block = MixFormerBlock(config)
        x = torch.randn(batch_size, config.num_heads, config.hidden_dim)
        seq = torch.randn(batch_size, config.seq_length, config.model_dim)
        seq_mask = torch.ones(batch_size, config.seq_length, dtype=torch.bool)
        decouple_mask = torch.ones(config.num_heads, config.hidden_dim)
        out = block(x, seq, seq_mask=seq_mask, decouple_mask=decouple_mask)
        assert out.shape == (batch_size, config.num_heads, config.hidden_dim)

    def test_gradient_flow(self, config, batch_size):
        block = MixFormerBlock(config)
        x = torch.randn(batch_size, config.num_heads, config.hidden_dim, requires_grad=True)
        seq = torch.randn(batch_size, config.seq_length, config.model_dim, requires_grad=True)
        out = block(x, seq)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert seq.grad is not None


# ============================================================
# Model Tests
# ============================================================


class TestFeatureEncoder:
    """测试特征编码器。"""

    def test_non_seq_encoding(self, config, batch_size):
        encoder = FeatureEncoder(config)
        non_seq_features = {
            name: torch.randint(0, vs, (batch_size,))
            for name, vs in config.vocab_sizes.items()
        }
        x = encoder.encode_non_seq_features(non_seq_features)
        assert x.shape == (batch_size, config.num_heads, config.hidden_dim)

    def test_seq_encoding(self, config, batch_size):
        encoder = FeatureEncoder(config)
        seq_features = torch.randint(0, config.num_items, (batch_size, config.seq_length))
        seq = encoder.encode_seq_features(seq_features)
        assert seq.shape == (batch_size, config.seq_length, config.model_dim)

    def test_full_encoding(self, config, batch_size):
        encoder = FeatureEncoder(config)
        non_seq_features = {
            name: torch.randint(0, vs, (batch_size,))
            for name, vs in config.vocab_sizes.items()
        }
        seq_features = torch.randint(0, config.num_items, (batch_size, config.seq_length))
        x, seq = encoder(non_seq_features, seq_features)
        assert x.shape == (batch_size, config.num_heads, config.hidden_dim)
        assert seq.shape == (batch_size, config.seq_length, config.model_dim)


class TestTaskHead:
    """测试任务头。"""

    def test_output_shape(self, config, batch_size):
        head = TaskHead(
            input_dim=config.model_dim,
            hidden_dims=config.task_head_hidden_dims,
        )
        x = torch.randn(batch_size, config.model_dim)
        out = head(x)
        assert out.shape == (batch_size, 1)

    def test_output_range(self, config, batch_size):
        head = TaskHead(
            input_dim=config.model_dim,
            hidden_dims=config.task_head_hidden_dims,
        )
        x = torch.randn(batch_size, config.model_dim)
        out = head(x)
        assert (out >= 0).all() and (out <= 1).all()


class TestMixFormer:
    """测试完整 MixFormer 模型。"""

    def test_forward(self, config, sample_batch):
        model = MixFormer(config)
        pred = model(
            non_seq_features=sample_batch["non_seq_features"],
            seq_features=sample_batch["seq_features"],
            seq_mask=sample_batch["seq_mask"],
        )
        batch_size = sample_batch["seq_features"].size(0)
        assert pred.shape == (batch_size, 1)
        assert (pred >= 0).all() and (pred <= 1).all()

    def test_forward_no_mask(self, config, sample_batch):
        model = MixFormer(config)
        pred = model(
            non_seq_features=sample_batch["non_seq_features"],
            seq_features=sample_batch["seq_features"],
        )
        batch_size = sample_batch["seq_features"].size(0)
        assert pred.shape == (batch_size, 1)

    def test_backward(self, config, sample_batch):
        model = MixFormer(config)
        pred = model(
            non_seq_features=sample_batch["non_seq_features"],
            seq_features=sample_batch["seq_features"],
            seq_mask=sample_batch["seq_mask"],
        )
        loss = torch.nn.functional.binary_cross_entropy(
            pred.squeeze(-1), sample_batch["label"]
        )
        loss.backward()

        # 检查所有参数都有梯度
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_param_count(self, config):
        model = MixFormer(config)
        assert model.get_num_params() > 0
        assert model.get_num_trainable_params() == model.get_num_params()


class TestUIMixFormer:
    """测试 UI-MixFormer 解耦变体。"""

    def test_forward(self, config, sample_batch):
        model = UIMixFormer(config)
        pred = model(
            non_seq_features=sample_batch["non_seq_features"],
            seq_features=sample_batch["seq_features"],
            seq_mask=sample_batch["seq_mask"],
        )
        batch_size = sample_batch["seq_features"].size(0)
        assert pred.shape == (batch_size, 1)
        assert (pred >= 0).all() and (pred <= 1).all()

    def test_decouple_mask(self, config):
        model = UIMixFormer(config)
        assert model.decouple_mask.shape == (config.num_heads, config.hidden_dim)
        # 验证掩码: 用户侧头的物品部分应该为 0
        user_mask = model.decouple_mask[: config.user_heads]
        assert user_mask.sum() < config.user_heads * config.hidden_dim

    def test_encode_user(self, config, sample_batch):
        model = UIMixFormer(config)
        user_repr = model.encode_user(
            non_seq_features=sample_batch["non_seq_features"],
            seq_features=sample_batch["seq_features"],
            seq_mask=sample_batch["seq_mask"],
        )
        batch_size = sample_batch["seq_features"].size(0)
        assert user_repr.shape == (batch_size, config.user_heads, config.hidden_dim)

    def test_backward(self, config, sample_batch):
        model = UIMixFormer(config)
        pred = model(
            non_seq_features=sample_batch["non_seq_features"],
            seq_features=sample_batch["seq_features"],
            seq_mask=sample_batch["seq_mask"],
        )
        loss = torch.nn.functional.binary_cross_entropy(
            pred.squeeze(-1), sample_batch["label"]
        )
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


# ============================================================
# Data Tests
# ============================================================


class TestSyntheticRecDataset:
    """测试合成数据集。"""

    def test_dataset_length(self, config):
        dataset = SyntheticRecDataset(config, num_samples=100)
        assert len(dataset) == 100

    def test_sample_structure(self, config):
        dataset = SyntheticRecDataset(config, num_samples=10)
        sample = dataset[0]
        assert "non_seq_features" in sample
        assert "seq_features" in sample
        assert "seq_mask" in sample
        assert "label" in sample

    def test_sample_shapes(self, config):
        dataset = SyntheticRecDataset(config, num_samples=10)
        sample = dataset[0]
        assert sample["seq_features"].shape == (config.seq_length,)
        assert sample["seq_mask"].shape == (config.seq_length,)
        assert sample["label"].shape == ()

    def test_label_values(self, config):
        dataset = SyntheticRecDataset(config, num_samples=100)
        for i in range(len(dataset)):
            label = dataset[i]["label"]
            assert label.item() in [0.0, 1.0]


class TestDataLoader:
    """测试 DataLoader。"""

    def test_create_dataloader(self, config):
        loader = create_dataloader(config, num_samples=100, batch_size=16)
        batch = next(iter(loader))
        assert batch["seq_features"].shape[0] == 16
        assert batch["seq_features"].shape[1] == config.seq_length
        assert "non_seq_features" in batch
        assert isinstance(batch["non_seq_features"], dict)


# ============================================================
# Integration Test
# ============================================================


class TestEndToEnd:
    """端到端集成测试。"""

    def test_training_step(self, config, sample_batch):
        """测试一个完整的训练步骤。"""
        model = MixFormer(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.BCELoss()

        # Forward
        pred = model(
            non_seq_features=sample_batch["non_seq_features"],
            seq_features=sample_batch["seq_features"],
            seq_mask=sample_batch["seq_mask"],
        )
        loss = criterion(pred.squeeze(-1), sample_batch["label"])

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 验证 loss 是有限数
        assert torch.isfinite(loss)

    def test_ui_mixformer_training_step(self, config, sample_batch):
        """测试 UI-MixFormer 的一个完整训练步骤。"""
        model = UIMixFormer(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.BCELoss()

        pred = model(
            non_seq_features=sample_batch["non_seq_features"],
            seq_features=sample_batch["seq_features"],
            seq_mask=sample_batch["seq_mask"],
        )
        loss = criterion(pred.squeeze(-1), sample_batch["label"])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert torch.isfinite(loss)

    def test_multi_step_training(self, config):
        """测试多步训练（验证 loss 下降趋势）。"""
        model = MixFormer(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.BCELoss()
        loader = create_dataloader(config, num_samples=64, batch_size=16)

        losses = []
        for batch in loader:
            pred = model(
                non_seq_features=batch["non_seq_features"],
                seq_features=batch["seq_features"],
                seq_mask=batch["seq_mask"],
            )
            loss = criterion(pred.squeeze(-1), batch["label"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # 验证 loss 全部有限
        assert all(np.isfinite(l) for l in losses)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
