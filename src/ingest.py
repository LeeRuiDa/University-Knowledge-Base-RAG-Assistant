from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

from langchain_core.embeddings import Embeddings

from src.chunking import chunk_documents
from src.config import Settings, get_settings
from src.corpus import CorpusEntry, sync_manifest_corpus
from src.embed import get_embeddings
from src.loaders import load_documents_from_path
from src.models import IngestResponse, MetadataSummary
from src.retriever import index_documents

MANIFEST_FILENAME = "ingestion_manifest.json"
CHUNK_CATALOG_FILENAME = "chunk_catalog.jsonl"


def run_ingestion(
    settings: Settings,
    embeddings: Embeddings | None = None,
    input_dir: str | None = None,
    recreate: bool = True,
) -> IngestResponse:
    settings.ensure_directories()
    source_dir = Path(input_dir) if input_dir else settings.data_raw_dir
    corpus_entries: list[CorpusEntry] = []
    metadata_overrides: dict[str, dict[str, object]] | None = None

    if input_dir is None and settings.corpus_manifest_path.exists():
        corpus_entries, metadata_overrides = sync_manifest_corpus(settings)

    documents = load_documents_from_path(source_dir, metadata_overrides=metadata_overrides)
    if not documents:
        raise ValueError(f"No supported documents were found in {source_dir}")

    chunks = chunk_documents(
        documents=documents,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    embeddings = embeddings or get_embeddings(settings)
    indexed_count = index_documents(chunks, embeddings, settings, recreate=recreate)

    metadata_summary = _build_metadata_summary(
        settings=settings,
        documents=documents,
        chunks_indexed=indexed_count,
        corpus_entries=corpus_entries,
    )
    manifest_path = write_manifest(settings, metadata_summary)
    _write_chunk_preview(settings, chunks)
    _write_chunk_stats(settings, chunks)

    return IngestResponse(
        status="ok",
        input_dir=str(source_dir),
        files_indexed=metadata_summary.files_indexed,
        sections_loaded=metadata_summary.sections_loaded,
        chunks_indexed=indexed_count,
        collection_name=settings.qdrant_collection_name,
        metadata_path=str(manifest_path),
        corpus_manifest_path=metadata_summary.corpus_manifest_path,
    )


def write_manifest(settings: Settings, metadata: MetadataSummary) -> Path:
    manifest_path = settings.data_parsed_dir / MANIFEST_FILENAME
    payload = metadata.model_dump()
    payload["indexed_at"] = datetime.now(UTC).isoformat()
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def load_manifest(settings: Settings) -> MetadataSummary:
    manifest_path = settings.data_parsed_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        return MetadataSummary(
            ready=False,
            collection_name=settings.qdrant_collection_name,
            qdrant_mode=settings.qdrant_mode,
            embedding_provider=settings.embedding_provider,
            generation_provider=settings.generation_provider,
            retrieval_strategy=settings.retrieval_strategy,
            corpus_manifest_path=str(settings.corpus_manifest_path)
            if settings.corpus_manifest_path.exists()
            else None,
        )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload.pop("indexed_at", None)
    payload.setdefault("retrieval_strategy", settings.retrieval_strategy)
    return MetadataSummary(**payload)


def _build_metadata_summary(
    settings: Settings,
    documents: list,
    chunks_indexed: int,
    corpus_entries: list[CorpusEntry] | None = None,
) -> MetadataSummary:
    sources = sorted({str(document.metadata.get("source")) for document in documents})
    document_types = sorted(
        {
            str(document.metadata.get("doc_type"))
            for document in documents
            if document.metadata.get("doc_type")
        }
    )
    years = sorted(
        {
            int(document.metadata.get("year"))
            for document in documents
            if document.metadata.get("year") is not None
        }
    )
    programs = sorted(
        {
            str(document.metadata.get("program"))
            for document in documents
            if document.metadata.get("program")
        }
    )

    return MetadataSummary(
        ready=True,
        collection_name=settings.qdrant_collection_name,
        qdrant_mode=settings.qdrant_mode,
        embedding_provider=settings.embedding_provider,
        generation_provider=settings.generation_provider,
        retrieval_strategy=settings.retrieval_strategy,
        corpus_manifest_path=str(settings.corpus_manifest_path)
        if settings.corpus_manifest_path.exists()
        else None,
        files_indexed=len(sources),
        sections_loaded=len(documents),
        chunks_indexed=chunks_indexed,
        document_types=document_types,
        years=years,
        programs=programs,
        sources=sources,
    )


def _write_chunk_preview(settings: Settings, chunks: list) -> None:
    preview_path = settings.data_parsed_dir / "chunk_preview.jsonl"
    catalog_path = settings.data_parsed_dir / CHUNK_CATALOG_FILENAME
    lines = []
    catalog_lines = []
    for chunk in chunks:
        payload = {
            "chunk_id": chunk.metadata.get("chunk_id"),
            "source": chunk.metadata.get("source"),
            "doc_id": chunk.metadata.get("doc_id"),
            "url": chunk.metadata.get("url"),
            "title": chunk.metadata.get("title"),
            "section": chunk.metadata.get("section"),
            "page": chunk.metadata.get("page"),
            "doc_type": chunk.metadata.get("doc_type"),
            "year": chunk.metadata.get("year"),
            "program": chunk.metadata.get("program"),
            "text": chunk.page_content,
        }
        preview_payload = {**payload, "text": chunk.page_content[:400]}
        lines.append(json.dumps(preview_payload))
        catalog_lines.append(json.dumps(payload))
    preview_path.write_text("\n".join(lines), encoding="utf-8")
    catalog_path.write_text("\n".join(catalog_lines), encoding="utf-8")


def _write_chunk_stats(settings: Settings, chunks: list) -> None:
    stats_path = settings.data_parsed_dir / "chunk_stats.json"
    by_source: dict[str, list[int]] = {}
    by_doc_type: dict[str, int] = {}

    for chunk in chunks:
        source = str(chunk.metadata.get("source"))
        by_source.setdefault(source, []).append(len(chunk.page_content))
        doc_type = str(chunk.metadata.get("doc_type", "unknown"))
        by_doc_type[doc_type] = by_doc_type.get(doc_type, 0) + 1

    source_stats = []
    for source, lengths in sorted(by_source.items()):
        source_stats.append(
            {
                "source": source,
                "chunk_count": len(lengths),
                "avg_chars": round(mean(lengths), 2),
                "min_chars": min(lengths),
                "max_chars": max(lengths),
            }
        )

    payload = {
        "total_chunks": len(chunks),
        "unique_sources": len(by_source),
        "doc_type_chunk_counts": by_doc_type,
        "source_stats": source_stats,
    }
    stats_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest and index the university knowledge base.")
    parser.add_argument("--input-dir", default=None, help="Directory that contains raw documents.")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate the Qdrant collection before indexing.",
    )
    args = parser.parse_args()

    settings = get_settings()
    response = run_ingestion(
        settings=settings,
        embeddings=get_embeddings(settings),
        input_dir=args.input_dir,
        recreate=args.recreate,
    )
    print(response.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
