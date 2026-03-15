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
from mixformer.model import FeatureEncoder, TaskHead, MixFormer


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
        num_non_seq_features=4,
        feature_embed_dim=16,
        num_items=100,
        num_users=100,
        num_categories=20,
        ffn_multiplier=2.667,
        dropout=0.0,
        user_heads=2,
        item_heads=2,
        task_head_hidden_dims=[64, 32],
        sparse_feature_names=["item_id", "category_id"],
        sparse_vocab_sizes=[100, 20],
        sparse_embed_dim=32,
        use_torchrec=False,
        target_item_mlp_dims=[64],
    )


@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def sample_batch(config, batch_size):
    """创建一个 Alibaba 格式的示例 batch。"""
    return {
        "target_item_id": torch.randint(1, config.num_items, (batch_size,)),
        "target_cate_id": torch.randint(1, config.num_categories, (batch_size,)),
        "hist_item_ids": torch.randint(0, config.num_items, (batch_size, config.seq_length)),
        "hist_cate_ids": torch.randint(0, config.num_categories, (batch_size, config.seq_length)),
        "seq_mask": torch.ones(batch_size, config.seq_length, dtype=torch.bool),
        "label": torch.randint(0, 2, (batch_size,)).float(),
    }


# ============================================================
# Config Tests
# ============================================================


class TestMixFormerConfig:
    """测试模型配置。"""

    def test_default_config(self):
        config = MixFormerConfig.default()
        assert config.num_heads == 8
        assert config.num_layers == 3
        assert config.hidden_dim == 64
        assert config.user_heads + config.item_heads == config.num_heads

    def test_medium_config(self):
        config = MixFormerConfig.medium()
        assert config.num_heads == 8
        assert config.num_layers == 4
        assert config.hidden_dim == 128
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

    def test_sparse_config_defaults(self):
        config = MixFormerConfig()
        assert config.sparse_feature_names == ["item_id", "category_id"]
        assert config.sparse_vocab_sizes == [config.num_items, config.num_categories]
        assert config.target_item_mlp_dims == [256]

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

    def test_target_encoding(self, config, batch_size):
        encoder = FeatureEncoder(config)
        target_item_id = torch.randint(1, config.num_items, (batch_size,))
        target_cate_id = torch.randint(1, config.num_categories, (batch_size,))
        query = encoder.encode_target(target_item_id, target_cate_id)
        assert query.shape == (batch_size, config.num_heads, config.hidden_dim)

    def test_sequence_encoding(self, config, batch_size):
        encoder = FeatureEncoder(config)
        hist_item_ids = torch.randint(0, config.num_items, (batch_size, config.seq_length))
        hist_cate_ids = torch.randint(0, config.num_categories, (batch_size, config.seq_length))
        seq = encoder.encode_sequence(hist_item_ids, hist_cate_ids)
        assert seq.shape == (batch_size, config.seq_length, config.model_dim)

    def test_full_encoding(self, config, batch_size):
        encoder = FeatureEncoder(config)
        target_item_id = torch.randint(1, config.num_items, (batch_size,))
        target_cate_id = torch.randint(1, config.num_categories, (batch_size,))
        hist_item_ids = torch.randint(0, config.num_items, (batch_size, config.seq_length))
        hist_cate_ids = torch.randint(0, config.num_categories, (batch_size, config.seq_length))
        query, seq = encoder(target_item_id, target_cate_id, hist_item_ids, hist_cate_ids)
        assert query.shape == (batch_size, config.num_heads, config.hidden_dim)
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
            target_item_id=sample_batch["target_item_id"],
            target_cate_id=sample_batch["target_cate_id"],
            hist_item_ids=sample_batch["hist_item_ids"],
            hist_cate_ids=sample_batch["hist_cate_ids"],
            seq_mask=sample_batch["seq_mask"],
        )
        batch_size = sample_batch["target_item_id"].size(0)
        assert pred.shape == (batch_size, 1)
        assert (pred >= 0).all() and (pred <= 1).all()

    def test_forward_no_mask(self, config, sample_batch):
        model = MixFormer(config)
        pred = model(
            target_item_id=sample_batch["target_item_id"],
            target_cate_id=sample_batch["target_cate_id"],
            hist_item_ids=sample_batch["hist_item_ids"],
            hist_cate_ids=sample_batch["hist_cate_ids"],
        )
        batch_size = sample_batch["target_item_id"].size(0)
        assert pred.shape == (batch_size, 1)

    def test_backward(self, config, sample_batch):
        model = MixFormer(config)
        pred = model(
            target_item_id=sample_batch["target_item_id"],
            target_cate_id=sample_batch["target_cate_id"],
            hist_item_ids=sample_batch["hist_item_ids"],
            hist_cate_ids=sample_batch["hist_cate_ids"],
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
            target_item_id=sample_batch["target_item_id"],
            target_cate_id=sample_batch["target_cate_id"],
            hist_item_ids=sample_batch["hist_item_ids"],
            hist_cate_ids=sample_batch["hist_cate_ids"],
            seq_mask=sample_batch["seq_mask"],
        )
        loss = criterion(pred.squeeze(-1), sample_batch["label"])

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 验证 loss 是有限数
        assert torch.isfinite(loss)

    def test_multi_step_training(self, config):
        """测试多步训练（验证 loss 全部有限）。"""
        model = MixFormer(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.BCELoss()

        losses = []
        for _ in range(4):
            batch = {
                "target_item_id": torch.randint(1, config.num_items, (8,)),
                "target_cate_id": torch.randint(1, config.num_categories, (8,)),
                "hist_item_ids": torch.randint(0, config.num_items, (8, config.seq_length)),
                "hist_cate_ids": torch.randint(0, config.num_categories, (8, config.seq_length)),
                "seq_mask": torch.ones(8, config.seq_length, dtype=torch.bool),
                "label": torch.randint(0, 2, (8,)).float(),
            }

            pred = model(
                target_item_id=batch["target_item_id"],
                target_cate_id=batch["target_cate_id"],
                hist_item_ids=batch["hist_item_ids"],
                hist_cate_ids=batch["hist_cate_ids"],
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
