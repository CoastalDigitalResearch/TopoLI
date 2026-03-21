"""TopoLI-1B transformer encoder."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from topoli.model.model_config import EncoderConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).type_as(x) * self.weight


def _rotary_embedding(dim: int, seq_len: int, theta: float = 10000.0) -> Tensor:
    """Compute rotary position embedding frequencies."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def _apply_rotary(x: Tensor, freqs: Tensor) -> Tensor:
    """Apply rotary embeddings to input tensor.

    x: (batch, seq, heads, head_dim)
    freqs: (seq, head_dim/2) complex
    """
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # freqs (seq, head_dim/2) -> (1, seq, 1, head_dim/2)
    freqs = freqs.unsqueeze(0).unsqueeze(2)
    x_rotated = x_complex * freqs
    return torch.view_as_real(x_rotated).flatten(-2).type_as(x)


class Attention(nn.Module):
    """Multi-head self-attention with RoPE."""

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=cfg.bias)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=cfg.bias)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=cfg.bias)
        self.o_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=cfg.bias)
        self.rope_theta = cfg.rope_theta

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)

        freqs = _rotary_embedding(self.head_dim, seq_len, self.rope_theta).to(x.device)
        q = _apply_rotary(q, freqs)
        k = _apply_rotary(k, freqs)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale

        if attention_mask is not None:
            mask_2d = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(~mask_2d, float("-inf"))

        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(attn_output)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        h, m = cfg.hidden_size, cfg.intermediate_size
        self.gate_proj = nn.Linear(h, m, bias=cfg.bias)
        self.up_proj = nn.Linear(h, m, bias=cfg.bias)
        self.down_proj = nn.Linear(m, h, bias=cfg.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerLayer(nn.Module):
    """Pre-norm transformer layer with attention + SwiGLU FFN."""

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden_size)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.hidden_size)
        self.ffn = SwiGLU(cfg)

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        h = x + self.attn(self.attn_norm(x), attention_mask=attention_mask)
        return h + self.ffn(self.ffn_norm(h))


class TopoLIEncoder(nn.Module):
    """TopoLI-1B transformer encoder."""

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [TransformerLayer(cfg) for _ in range(cfg.num_layers)]
        )
        self.final_norm = RMSNorm(cfg.hidden_size)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        x = self.token_embedding(input_ids)
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
        return self.final_norm(x)
