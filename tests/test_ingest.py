"""Unit tests for knowledge base ingestion and metadata management.

Tests manifest serialization, metadata aggregation, chunk catalog previews,
summary statistics, and ingestion error handling.
"""

import json
from pathlib import Path

import pytest
from langchain_core.documents import Document

from src.config import Settings
from src.ingest import (
    _build_metadata_summary,
    _write_chunk_preview,
    _write_chunk_stats,
    load_manifest,
    run_ingestion,
    write_manifest,
)
from src.models import MetadataSummary


@pytest.fixture
def temp_settings(tmp_path: Path) -> Settings:
    settings = Settings(
        data_raw_dir=tmp_path / "raw",
        data_parsed_dir=tmp_path / "parsed",
        corpus_manifest_path=tmp_path / "manifest.csv",
        qdrant_collection_name="test_collection",
        qdrant_mode="memory",
    )
    settings.ensure_directories()
    return settings


# ---------------------------------------------------------------------------
# Metadata aggregation tests
# ---------------------------------------------------------------------------
class TestBuildMetadataSummary:
    def test_build_metadata_summary_aggregates_fields_correctly(
        self, temp_settings: Settings
    ) -> None:
        docs = [
            Document(
                page_content="Policy A",
                metadata={
                    "source": "data/raw/policy_a.md",
                    "doc_type": "handbook",
                    "year": 2024,
                    "program": "Undergraduate",
                },
            ),
            Document(
                page_content="Policy B",
                metadata={
                    "source": "data/raw/policy_b.md",
                    "doc_type": "syllabus",
                    "year": 2025,
                    "program": "Graduate",
                },
            ),
            Document(
                page_content="Policy C section 2",
                metadata={
                    "source": "data/raw/policy_a.md",
                    "doc_type": "handbook",
                    "year": 2024,
                    "program": "Undergraduate",
                },
            ),
        ]

        summary = _build_metadata_summary(
            settings=temp_settings,
            documents=docs,
            chunks_indexed=10,
        )

        assert summary.ready is True
        assert summary.collection_name == "test_collection"
        assert summary.files_indexed == 2  # policy_a and policy_b
        assert summary.sections_loaded == 3
        assert summary.chunks_indexed == 10
        assert summary.document_types == ["handbook", "syllabus"]
        assert summary.years == [2024, 2025]
        assert summary.programs == ["Graduate", "Undergraduate"]
        assert len(summary.sources) == 2

    def test_build_metadata_summary_handles_none_metadata(
        self, temp_settings: Settings
    ) -> None:
        docs = [
            Document(
                page_content="Unstructured note",
                metadata={"source": "note.txt", "year": None, "program": None},
            )
        ]
        summary = _build_metadata_summary(
            settings=temp_settings,
            documents=docs,
            chunks_indexed=1,
        )
        assert summary.document_types == []
        assert summary.years == []
        assert summary.programs == []
        assert summary.sources == ["note.txt"]


# ---------------------------------------------------------------------------
# Manifest serialization tests
# ---------------------------------------------------------------------------
class TestManifestRoundtrip:
    def test_write_and_load_manifest_roundtrip(self, temp_settings: Settings) -> None:
        summary = MetadataSummary(
            ready=True,
            collection_name="test_collection",
            qdrant_mode="memory",
            embedding_provider="mock",
            generation_provider="mock",
            retrieval_strategy="hybrid",
            files_indexed=5,
            sections_loaded=12,
            chunks_indexed=24,
            document_types=["handbook", "policy"],
            years=[2024, 2025],
            programs=["CS", "SE"],
            sources=["doc1.md", "doc2.md"],
        )

        manifest_path = write_manifest(temp_settings, summary)
        assert manifest_path.exists()

        # Check ISO timestamp was added
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert "indexed_at" in data

        loaded = load_manifest(temp_settings)
        assert loaded.ready is True
        assert loaded.collection_name == summary.collection_name
        assert loaded.files_indexed == summary.files_indexed
        assert loaded.chunks_indexed == summary.chunks_indexed
        assert loaded.document_types == summary.document_types

    def test_load_manifest_missing_returns_unready(self, temp_settings: Settings) -> None:
        summary = load_manifest(temp_settings)
        assert summary.ready is False
        assert summary.collection_name == "test_collection"
        assert summary.files_indexed == 0


# ---------------------------------------------------------------------------
# Chunk previews and stats tests
# ---------------------------------------------------------------------------
class TestChunkPreviewsAndStats:
    def test_write_chunk_preview_and_catalog(self, temp_settings: Settings) -> None:
        chunks = [
            Document(
                page_content="A" * 500,
                metadata={
                    "chunk_id": "chunk_001",
                    "source": "sample.md",
                    "doc_id": "doc_1",
                    "url": "https://example.edu/doc1",
                    "title": "Sample Document",
                    "section": "Section 1",
                    "page": 1,
                    "doc_type": "policy",
                    "year": 2025,
                    "program": "CS",
                },
            )
        ]

        _write_chunk_preview(temp_settings, chunks)

        preview_path = temp_settings.data_parsed_dir / "chunk_preview.jsonl"
        catalog_path = temp_settings.data_parsed_dir / "chunk_catalog.jsonl"

        assert preview_path.exists()
        assert catalog_path.exists()

        preview_data = json.loads(preview_path.read_text(encoding="utf-8").strip())
        catalog_data = json.loads(catalog_path.read_text(encoding="utf-8").strip())

        # Preview text is truncated to 400 chars
        assert len(preview_data["text"]) == 400
        # Catalog retains full 500 chars
        assert len(catalog_data["text"]) == 500
        assert catalog_data["chunk_id"] == "chunk_001"

    def test_write_chunk_stats(self, temp_settings: Settings) -> None:
        chunks = [
            Document(page_content="12345", metadata={"source": "s1.md", "doc_type": "typeA"}),
            Document(page_content="1234567890", metadata={"source": "s1.md", "doc_type": "typeA"}),
            Document(page_content="123", metadata={"source": "s2.md", "doc_type": "typeB"}),
        ]

        _write_chunk_stats(temp_settings, chunks)

        stats_path = temp_settings.data_parsed_dir / "chunk_stats.json"
        assert stats_path.exists()

        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        assert stats["total_chunks"] == 3
        assert stats["unique_sources"] == 2
        assert stats["doc_type_chunk_counts"] == {"typeA": 2, "typeB": 1}

        s1_stat = next(s for s in stats["source_stats"] if s["source"] == "s1.md")
        assert s1_stat["chunk_count"] == 2
        assert s1_stat["min_chars"] == 5
        assert s1_stat["max_chars"] == 10
        assert s1_stat["avg_chars"] == 7.5


# ---------------------------------------------------------------------------
# Ingestion failure edge cases
# ---------------------------------------------------------------------------
class TestIngestionEdgeCases:
    def test_run_ingestion_empty_input_directory_raises(
        self, temp_settings: Settings, tmp_path: Path
    ) -> None:
        empty_dir = tmp_path / "empty_dir"
        empty_dir.mkdir()

        with pytest.raises(ValueError, match="No supported documents were found"):
            run_ingestion(settings=temp_settings, input_dir=str(empty_dir))
