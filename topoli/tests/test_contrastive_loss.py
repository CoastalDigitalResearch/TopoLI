"""Tests for ColBERT contrastive loss with differentiable MaxSim."""

from __future__ import annotations

import torch

from topoli.finetune.contrastive_loss import (
    colbert_info_nce,
    maxsim,
)


class TestMaxSim:
    """MaxSim computes per-query-token max similarities correctly."""

    def test_output_shape(self) -> None:
        query = torch.randn(2, 8, 128)  # (batch, q_len, dim)
        doc = torch.randn(2, 16, 128)  # (batch, d_len, dim)
        scores = maxsim(query, doc)
        assert scores.shape == (2,)

    def test_identical_embeddings_high_score(self) -> None:
        emb = torch.nn.functional.normalize(torch.randn(1, 8, 128), dim=-1)
        score = maxsim(emb, emb)
        # Each query token matches itself perfectly (cosine=1), sum = q_len
        torch.testing.assert_close(score, torch.tensor([8.0]), atol=0.01, rtol=0.01)

    def test_orthogonal_embeddings_low_score(self) -> None:
        q = torch.zeros(1, 2, 4)
        q[0, 0, 0] = 1.0
        q[0, 1, 1] = 1.0
        d = torch.zeros(1, 2, 4)
        d[0, 0, 2] = 1.0
        d[0, 1, 3] = 1.0
        score = maxsim(q, d)
        torch.testing.assert_close(score, torch.tensor([0.0]), atol=0.01, rtol=0.01)

    def test_gradient_flows_through_maxsim(self) -> None:
        q = torch.randn(2, 4, 128, requires_grad=True)
        d = torch.randn(2, 8, 128, requires_grad=True)
        score = maxsim(q, d)
        score.sum().backward()
        assert q.grad is not None
        assert d.grad is not None


class TestColBERTInfoNCE:
    """InfoNCE loss with MaxSim scoring."""

    def test_loss_is_scalar(self) -> None:
        query_embs = torch.randn(4, 8, 128)
        pos_doc_embs = torch.randn(4, 16, 128)
        neg_doc_embs = torch.randn(4, 3, 16, 128)  # 3 negatives per query
        loss = colbert_info_nce(query_embs, pos_doc_embs, neg_doc_embs)
        assert loss.shape == ()

    def test_loss_is_positive(self) -> None:
        query_embs = torch.randn(4, 8, 128)
        pos_doc_embs = torch.randn(4, 16, 128)
        neg_doc_embs = torch.randn(4, 3, 16, 128)
        loss = colbert_info_nce(query_embs, pos_doc_embs, neg_doc_embs)
        assert loss.item() > 0

    def test_loss_lower_when_positive_matches(self) -> None:
        q = torch.nn.functional.normalize(torch.randn(2, 4, 128), dim=-1)
        pos = q.clone()  # positive = identical to query
        neg = torch.nn.functional.normalize(torch.randn(2, 3, 4, 128), dim=-1)
        loss_good = colbert_info_nce(q, pos, neg)

        pos_bad = torch.nn.functional.normalize(torch.randn(2, 4, 128), dim=-1)
        loss_bad = colbert_info_nce(q, pos_bad, neg)

        assert loss_good.item() < loss_bad.item()

    def test_gradient_flows(self) -> None:
        q = torch.randn(2, 4, 128, requires_grad=True)
        pos = torch.randn(2, 8, 128, requires_grad=True)
        neg = torch.randn(2, 3, 8, 128, requires_grad=True)
        loss = colbert_info_nce(q, pos, neg)
        loss.backward()
        assert q.grad is not None
        assert pos.grad is not None
        assert neg.grad is not None

    def test_temperature_affects_loss(self) -> None:
        q = torch.randn(2, 4, 128)
        pos = torch.randn(2, 8, 128)
        neg = torch.randn(2, 3, 8, 128)
        loss_low_t = colbert_info_nce(q, pos, neg, temperature=0.01)
        loss_high_t = colbert_info_nce(q, pos, neg, temperature=1.0)
        assert loss_low_t.item() != loss_high_t.item()
