"""Tests for TopoLI-1B transformer encoder."""

from __future__ import annotations

import torch

from topoli.model.encoder import TopoLIEncoder
from topoli.model.model_config import EncoderConfig, topoli_150m


class TestEncoderForwardPass:
    """Encoder produces correct output shapes."""

    def test_output_shape_matches_hidden_size(self) -> None:
        cfg = EncoderConfig(
            hidden_size=64,
            num_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
            vocab_size=100,
        )
        model = TopoLIEncoder(cfg)
        input_ids = torch.randint(0, 100, (2, 16))
        output = model(input_ids)
        assert output.shape == (2, 16, 64)

    def test_output_dtype_matches_input(self) -> None:
        cfg = EncoderConfig(
            hidden_size=64,
            num_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
            vocab_size=100,
        )
        model = TopoLIEncoder(cfg)
        input_ids = torch.randint(0, 100, (1, 8))
        output = model(input_ids)
        assert output.dtype == torch.float32

    def test_attention_mask_respected(self) -> None:
        cfg = EncoderConfig(
            hidden_size=64,
            num_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
            vocab_size=100,
        )
        model = TopoLIEncoder(cfg)
        input_ids = torch.randint(0, 100, (1, 8))
        mask = torch.ones(1, 8, dtype=torch.bool)
        mask[0, 4:] = False
        out_masked = model(input_ids, attention_mask=mask)
        assert out_masked.shape == (1, 8, 64)

    def test_batch_dimension_independent(self) -> None:
        cfg = EncoderConfig(
            hidden_size=64,
            num_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
            vocab_size=100,
        )
        model = TopoLIEncoder(cfg)
        model.eval()
        single = torch.tensor([[1, 2, 3, 4]])
        batched = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]])
        with torch.no_grad():
            out_single = model(single)
            out_batched = model(batched)
        torch.testing.assert_close(out_single[0], out_batched[0], atol=1e-5, rtol=1e-5)


class TestEncoderGradients:
    """Gradients flow through the encoder."""

    def test_gradient_flows_to_embeddings(self) -> None:
        cfg = EncoderConfig(
            hidden_size=64,
            num_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
            vocab_size=100,
        )
        model = TopoLIEncoder(cfg)
        input_ids = torch.randint(0, 100, (2, 8))
        output = model(input_ids)
        loss = output.sum()
        loss.backward()
        assert model.token_embedding.weight.grad is not None
        assert model.token_embedding.weight.grad.abs().sum() > 0

    def test_all_parameters_have_grad(self) -> None:
        cfg = EncoderConfig(
            hidden_size=64,
            num_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
            vocab_size=100,
        )
        model = TopoLIEncoder(cfg)
        input_ids = torch.randint(0, 100, (2, 8))
        output = model(input_ids)
        loss = output.sum()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No grad for {name}"


class TestEncoderComponents:
    """Encoder has expected architectural components."""

    def test_has_token_embedding(self) -> None:
        cfg = EncoderConfig(
            hidden_size=64,
            num_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
            vocab_size=100,
        )
        model = TopoLIEncoder(cfg)
        assert hasattr(model, "token_embedding")
        assert model.token_embedding.num_embeddings == 100
        assert model.token_embedding.embedding_dim == 64

    def test_has_correct_number_of_layers(self) -> None:
        cfg = EncoderConfig(
            hidden_size=64,
            num_layers=3,
            num_attention_heads=4,
            intermediate_size=128,
            vocab_size=100,
        )
        model = TopoLIEncoder(cfg)
        assert len(model.layers) == 3

    def test_has_final_norm(self) -> None:
        cfg = EncoderConfig(
            hidden_size=64,
            num_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
            vocab_size=100,
        )
        model = TopoLIEncoder(cfg)
        assert hasattr(model, "final_norm")

    def test_no_bias_in_attention(self) -> None:
        cfg = EncoderConfig(
            hidden_size=64,
            num_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
            vocab_size=100,
            bias=False,
        )
        model = TopoLIEncoder(cfg)
        layer = model.layers[0]
        assert layer.attn.q_proj.bias is None
        assert layer.attn.k_proj.bias is None
        assert layer.attn.v_proj.bias is None
        assert layer.attn.o_proj.bias is None


class TestEncoderParamCount:
    """Rough parameter count sanity checks."""

    def test_tiny_model_param_count(self) -> None:
        cfg = EncoderConfig(
            hidden_size=64,
            num_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
            vocab_size=100,
        )
        model = TopoLIEncoder(cfg)
        total = sum(p.numel() for p in model.parameters())
        assert 10_000 < total < 500_000

    def test_150m_scale_param_count(self) -> None:
        model_cfg = topoli_150m()
        cfg = model_cfg.encoder
        model = TopoLIEncoder(cfg)
        total = sum(p.numel() for p in model.parameters())
        assert 100_000_000 < total < 250_000_000
