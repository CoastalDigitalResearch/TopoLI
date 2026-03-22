"""Tests for data source configuration and provenance tracking."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from topoli.data.source_config import (
    DataSourceConfig,
    License,
    PassageRecord,
    SourceDomain,
    get_source_registry,
)


class TestLicenseEnum:
    """License values are the only ones we allow."""

    def test_apache_is_valid(self) -> None:
        assert License.APACHE_2_0.value == "Apache-2.0"

    def test_mit_is_valid(self) -> None:
        assert License.MIT.value == "MIT"

    def test_cc_by_is_valid(self) -> None:
        assert License.CC_BY_4_0.value == "CC-BY-4.0"

    def test_cc0_is_valid(self) -> None:
        assert License.CC0.value == "CC0-1.0"

    def test_odc_by_is_valid(self) -> None:
        assert License.ODC_BY.value == "ODC-By-1.0"

    def test_public_domain_is_valid(self) -> None:
        assert License.PUBLIC_DOMAIN.value == "Public Domain"


class TestDataSourceConfig:
    """Data source configs are frozen and validated."""

    def test_create_valid_source(self) -> None:
        src = DataSourceConfig(
            name="test_source",
            huggingface_id="org/dataset",
            license=License.APACHE_2_0,
            domain=SourceDomain.GENERAL,
            description="A test dataset",
        )
        assert src.name == "test_source"
        assert src.license == License.APACHE_2_0

    def test_frozen(self) -> None:
        src = DataSourceConfig(
            name="test",
            huggingface_id="org/ds",
            license=License.MIT,
            domain=SourceDomain.SCIENCE,
            description="test",
        )
        with pytest.raises(ValidationError):
            src.name = "changed"  # type: ignore[misc]

    def test_commercially_usable_property(self) -> None:
        src = DataSourceConfig(
            name="test",
            huggingface_id="org/ds",
            license=License.APACHE_2_0,
            domain=SourceDomain.GENERAL,
            description="test",
        )
        assert src.commercially_usable is True


class TestSourceRegistry:
    """Registry contains only verified permissive sources."""

    def test_registry_is_not_empty(self) -> None:
        registry = get_source_registry()
        assert len(registry) > 0

    def test_all_sources_are_commercially_usable(self) -> None:
        registry = get_source_registry()
        for src in registry:
            assert src.commercially_usable, (
                f"{src.name} has license {src.license.value} "
                "which is not commercially usable"
            )

    def test_no_cc_by_sa_in_registry(self) -> None:
        registry = get_source_registry()
        for src in registry:
            assert "SA" not in src.license.value, (
                f"{src.name} uses ShareAlike license {src.license.value}"
            )

    def test_no_nc_in_registry(self) -> None:
        registry = get_source_registry()
        for src in registry:
            assert "NC" not in src.license.value, (
                f"{src.name} uses NonCommercial license {src.license.value}"
            )

    def test_contains_triviaqa(self) -> None:
        registry = get_source_registry()
        names = {s.name for s in registry}
        assert "triviaqa" in names

    def test_contains_olc_legal(self) -> None:
        registry = get_source_registry()
        names = {s.name for s in registry}
        assert "olc_legal" in names

    def test_contains_olc_science(self) -> None:
        registry = get_source_registry()
        names = {s.name for s in registry}
        assert "olc_science" in names

    def test_contains_slimpajama(self) -> None:
        registry = get_source_registry()
        names = {s.name for s in registry}
        assert "slimpajama" in names


class TestPassageRecord:
    """Passage records carry full provenance."""

    def test_create_passage_record(self) -> None:
        record = PassageRecord(
            passage_id="gov_001",
            text="The Federal Reserve announced today...",
            source_name="kl3m_government",
            source_license=License.CC_BY_4_0,
            source_doc_id="fed_register_2024_001",
            char_count=38,
        )
        assert record.passage_id == "gov_001"
        assert record.source_license == License.CC_BY_4_0

    def test_passage_record_frozen(self) -> None:
        record = PassageRecord(
            passage_id="test",
            text="text",
            source_name="test",
            source_license=License.CC0,
            source_doc_id="doc1",
            char_count=4,
        )
        with pytest.raises(ValidationError):
            record.text = "changed"  # type: ignore[misc]

    def test_char_count_must_be_positive(self) -> None:
        with pytest.raises(ValidationError, match="char_count"):
            PassageRecord(
                passage_id="test",
                text="",
                source_name="test",
                source_license=License.CC0,
                source_doc_id="doc1",
                char_count=0,
            )
