"""Tests for fine-tuning trainer with topo-aware loss."""

from __future__ import annotations

import torch

from topoli.finetune.finetune_config import TopoLossSchedule
from topoli.finetune.trainer import FinetuneStep, TopoLIForRetrieval
from topoli.model.colbert_head import ColBERTHead
from topoli.model.encoder import TopoLIEncoder
from topoli.model.model_config import EncoderConfig
from topoli.model.pruning_head import PruningHead


def _tiny_model() -> TopoLIForRetrieval:
    cfg = EncoderConfig(
        hidden_size=64,
        num_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        vocab_size=100,
    )
    encoder = TopoLIEncoder(cfg)
    colbert_head = ColBERTHead(hidden_size=64, colbert_dim=32)
    pruning_head = PruningHead(hidden_size=64, hidden_dim=16)
    return TopoLIForRetrieval(encoder, colbert_head, pruning_head)


class TestTopoLIForRetrieval:
    """Full model composes encoder + heads."""

    def test_encode_returns_embeddings_and_scores(self) -> None:
        model = _tiny_model()
        input_ids = torch.randint(0, 100, (2, 8))
        embs, scores = model.encode(input_ids)
        assert embs.shape == (2, 8, 32)
        assert scores.shape == (2, 8, 1)

    def test_embeddings_are_l2_normalized(self) -> None:
        model = _tiny_model()
        input_ids = torch.randint(0, 100, (2, 8))
        embs, _ = model.encode(input_ids)
        norms = torch.norm(embs, dim=-1)
        torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)

    def test_pruning_scores_in_0_1(self) -> None:
        model = _tiny_model()
        input_ids = torch.randint(0, 100, (2, 8))
        _, scores = model.encode(input_ids)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

    def test_gradient_flows(self) -> None:
        model = _tiny_model()
        input_ids = torch.randint(0, 100, (2, 8))
        embs, scores = model.encode(input_ids)
        (embs.sum() + scores.sum()).backward()
        assert model.encoder.token_embedding.weight.grad is not None


class TestFinetuneStep:
    """Single fine-tuning step computes combined loss."""

    def test_retrieval_only_loss(self) -> None:
        model = _tiny_model()
        step_fn = FinetuneStep(model)

        q_ids = torch.randint(0, 100, (2, 8))
        pos_ids = torch.randint(0, 100, (2, 16))
        neg_ids = torch.randint(0, 100, (2, 3, 16))

        losses = step_fn(q_ids, pos_ids, neg_ids, step=0)
        assert "retrieval" in losses
        assert losses["retrieval"].item() > 0
        assert losses["topo_structure"].item() == 0.0
        assert losses["pruning_distill"].item() == 0.0

    def test_topo_losses_active_after_warmup(self) -> None:
        model = _tiny_model()
        schedule = TopoLossSchedule(
            warmup_end=0,
            ramp_end=1,
            full_end=100,
            anneal_end=200,
        )
        step_fn = FinetuneStep(model, topo_schedule=schedule)

        q_ids = torch.randint(0, 100, (2, 8))
        pos_ids = torch.randint(0, 100, (2, 16))
        neg_ids = torch.randint(0, 100, (2, 3, 16))

        losses = step_fn(q_ids, pos_ids, neg_ids, step=50)
        assert losses["topo_structure"].item() != 0.0

    def test_total_loss_is_sum(self) -> None:
        model = _tiny_model()
        step_fn = FinetuneStep(model)

        q_ids = torch.randint(0, 100, (2, 8))
        pos_ids = torch.randint(0, 100, (2, 16))
        neg_ids = torch.randint(0, 100, (2, 3, 16))

        losses = step_fn(q_ids, pos_ids, neg_ids, step=0)
        assert "total" in losses

    def test_gradient_flows_through_total(self) -> None:
        model = _tiny_model()
        step_fn = FinetuneStep(model)

        q_ids = torch.randint(0, 100, (2, 8))
        pos_ids = torch.randint(0, 100, (2, 16))
        neg_ids = torch.randint(0, 100, (2, 3, 16))

        losses = step_fn(q_ids, pos_ids, neg_ids, step=0)
        losses["total"].backward()
        assert model.encoder.token_embedding.weight.grad is not None
