"""Tests for TopoLI tokenizer wrapper."""

from __future__ import annotations

from topoli.model.tokenizer import (
    SPECIAL_TOKENS,
    TopoLITokenizerConfig,
    build_tokenizer,
)


class TestTokenizerConfig:
    """Tokenizer config is frozen and has correct defaults."""

    def test_default_vocab_size(self) -> None:
        cfg = TopoLITokenizerConfig()
        assert cfg.vocab_size == 32000

    def test_frozen(self) -> None:
        import pytest
        from pydantic import ValidationError

        cfg = TopoLITokenizerConfig()
        with pytest.raises(ValidationError):
            cfg.vocab_size = 50000  # type: ignore[misc]


class TestSpecialTokens:
    """Special tokens are defined for ColBERT-style retrieval."""

    def test_has_cls(self) -> None:
        assert "[CLS]" in SPECIAL_TOKENS

    def test_has_mask(self) -> None:
        assert "[MASK]" in SPECIAL_TOKENS

    def test_has_sep(self) -> None:
        assert "[SEP]" in SPECIAL_TOKENS

    def test_has_pad(self) -> None:
        assert "[PAD]" in SPECIAL_TOKENS

    def test_has_query_marker(self) -> None:
        assert "[Q]" in SPECIAL_TOKENS

    def test_has_document_marker(self) -> None:
        assert "[D]" in SPECIAL_TOKENS


class TestBuildTokenizer:
    """Tokenizer builder creates a working tokenizer from config."""

    def test_returns_tokenizer_with_special_tokens(self) -> None:
        cfg = TopoLITokenizerConfig()
        tokenizer = build_tokenizer(cfg)
        assert tokenizer.cls_token == "[CLS]"
        assert tokenizer.mask_token == "[MASK]"
        assert tokenizer.sep_token == "[SEP]"
        assert tokenizer.pad_token == "[PAD]"

    def test_encodes_text(self) -> None:
        cfg = TopoLITokenizerConfig()
        tokenizer = build_tokenizer(cfg)
        encoded = tokenizer.encode("hello world")
        assert isinstance(encoded, list)
        assert len(encoded) > 0

    def test_query_marker_in_vocab(self) -> None:
        cfg = TopoLITokenizerConfig()
        tokenizer = build_tokenizer(cfg)
        q_id = tokenizer.convert_tokens_to_ids("[Q]")
        assert isinstance(q_id, int)
        assert q_id != tokenizer.unk_token_id

    def test_document_marker_in_vocab(self) -> None:
        cfg = TopoLITokenizerConfig()
        tokenizer = build_tokenizer(cfg)
        d_id = tokenizer.convert_tokens_to_ids("[D]")
        assert isinstance(d_id, int)
        assert d_id != tokenizer.unk_token_id

    def test_round_trip_encode_decode(self) -> None:
        cfg = TopoLITokenizerConfig()
        tokenizer = build_tokenizer(cfg)
        text = "The Federal Reserve raised rates."
        ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        assert "federal" in decoded.lower()
        assert "reserve" in decoded.lower()
