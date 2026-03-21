"""Fine-tuning trainer: ColBERT retrieval + topo-aware losses."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from topoli.finetune.contrastive_loss import colbert_info_nce
from topoli.finetune.finetune_config import TopoLossSchedule
from topoli.finetune.topo_distill import pruning_distill_loss
from topoli.finetune.topo_loss import topo_structure_loss
from topoli.model.colbert_head import ColBERTHead
from topoli.model.pruning_head import PruningHead


class TopoLIForRetrieval(nn.Module):
    """Full TopoLI model: encoder + ColBERT head + pruning head."""

    def __init__(
        self,
        encoder: nn.Module,
        colbert_head: ColBERTHead,
        pruning_head: PruningHead,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.colbert_head = colbert_head
        self.pruning_head = pruning_head

    def encode(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Encode input to ColBERT embeddings and pruning scores."""
        hidden = self.encoder(input_ids, attention_mask=attention_mask)
        embeddings = self.colbert_head(hidden)
        scores = self.pruning_head(hidden)
        return embeddings, scores


class FinetuneStep(nn.Module):
    """Single fine-tuning step combining retrieval + topo losses."""

    def __init__(
        self,
        model: TopoLIForRetrieval,
        topo_schedule: TopoLossSchedule | None = None,
        temperature: float = 0.05,
    ) -> None:
        super().__init__()
        self.model = model
        self.topo_schedule = topo_schedule or TopoLossSchedule()
        self.temperature = temperature

    def forward(
        self,
        query_ids: Tensor,
        pos_doc_ids: Tensor,
        neg_doc_ids: Tensor,
        step: int,
    ) -> dict[str, Tensor]:
        """Compute all losses for a single training step."""
        device = query_ids.device
        alpha, beta = self.topo_schedule.get_weights(step)

        q_embs, _ = self.model.encode(query_ids)
        pos_embs, pos_scores = self.model.encode(pos_doc_ids)

        batch_size, n_neg, neg_seq_len = neg_doc_ids.shape
        flat_neg_ids = neg_doc_ids.view(batch_size * n_neg, neg_seq_len)
        flat_neg_embs, _ = self.model.encode(flat_neg_ids)
        neg_embs = flat_neg_embs.view(batch_size, n_neg, neg_seq_len, -1)

        retrieval_loss = colbert_info_nce(
            q_embs,
            pos_embs,
            neg_embs,
            temperature=self.temperature,
        )

        zero = torch.tensor(0.0, device=device)

        if alpha > 0:
            topo_loss = topo_structure_loss(pos_embs.detach())
            topo_loss_weighted = alpha * topo_loss
        else:
            topo_loss_weighted = zero

        if beta > 0:
            dummy_targets = torch.zeros_like(pos_scores.squeeze(-1))
            dummy_mask = torch.ones_like(dummy_targets)
            distill_loss = pruning_distill_loss(pos_scores, dummy_targets, dummy_mask)
            distill_loss_weighted = beta * distill_loss
        else:
            distill_loss_weighted = zero

        total = retrieval_loss + topo_loss_weighted + distill_loss_weighted

        return {
            "total": total,
            "retrieval": retrieval_loss,
            "topo_structure": topo_loss_weighted,
            "pruning_distill": distill_loss_weighted,
        }
