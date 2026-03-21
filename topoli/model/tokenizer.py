"""TopoLI tokenizer wrapper with ColBERT special tokens.

Uses a base tokenizer (default: BERT WordPiece) and adds the special tokens
needed for ColBERT-style late interaction: [Q] for queries, [D] for documents.

For production training, replace the base with a custom-trained BPE tokenizer
via `train_tokenizer.py`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict
from transformers import AutoTokenizer

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

SPECIAL_TOKENS: tuple[str, ...] = (
    "[CLS]",
    "[SEP]",
    "[PAD]",
    "[MASK]",
    "[Q]",
    "[D]",
)


class TopoLITokenizerConfig(BaseModel):
    """Tokenizer configuration."""

    model_config = ConfigDict(frozen=True)

    vocab_size: int = 32000
    base_tokenizer: str = "bert-base-uncased"
    max_length: int = 512


def build_tokenizer(cfg: TopoLITokenizerConfig) -> PreTrainedTokenizerBase:
    """Build a tokenizer with ColBERT special tokens added."""
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_tokenizer)
    new_tokens = [t for t in SPECIAL_TOKENS if t not in tokenizer.get_vocab()]
    if new_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    tokenizer.model_max_length = cfg.max_length
    return tokenizer
