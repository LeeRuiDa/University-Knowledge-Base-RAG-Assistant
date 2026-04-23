from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.models import SearchFilters

TOKEN_PATTERN = re.compile(r"\b[a-zA-Z0-9][a-zA-Z0-9_-]*\b")


@dataclass(frozen=True)
class SparseChunk:
    chunk_id: str
    source: str
    text: str
    title: str
    doc_id: str | None = None
    url: str | None = None
    section: str | None = None
    page: int | None = None
    doc_type: str | None = None
    year: int | None = None
    program: str | None = None


class SparseChunkIndex:
    def __init__(self, chunks: list[SparseChunk]) -> None:
        self.chunks = chunks
        self._tokenized_corpus = [tokenize_text(self._searchable_text(chunk)) for chunk in chunks]
        self._bm25 = BM25Okapi(self._tokenized_corpus) if chunks else None

    def search(
        self,
        query: str,
        k: int,
        filters: SearchFilters | None = None,
    ) -> list[tuple[SparseChunk, float]]:
        if not self.chunks or self._bm25 is None:
            return []

        query_tokens = tokenize_text(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        candidates: list[tuple[SparseChunk, float]] = []
        for index, score in enumerate(scores):
            chunk = self.chunks[index]
            if filters and not _matches_filters(chunk, filters):
                continue
            if score <= 0:
                continue
            candidates.append((chunk, float(score)))

        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates[:k]

    @staticmethod
    def _searchable_text(chunk: SparseChunk) -> str:
        parts = [
            chunk.title,
            chunk.section or "",
            chunk.doc_type or "",
            chunk.program or "",
            chunk.text,
        ]
        return "\n".join(part for part in parts if part)


def load_sparse_index(catalog_path: Path) -> SparseChunkIndex:
    modified_ns = catalog_path.stat().st_mtime_ns
    return _load_sparse_index_cached(str(catalog_path), modified_ns)


@lru_cache(maxsize=4)
def _load_sparse_index_cached(catalog_path: str, modified_ns: int) -> SparseChunkIndex:
    del modified_ns
    path = Path(catalog_path)
    chunks: list[SparseChunk] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        chunks.append(
            SparseChunk(
                chunk_id=str(payload.get("chunk_id", "")),
                source=str(payload.get("source", "")),
                text=str(payload.get("text", "")),
                title=str(payload.get("title", "")),
                doc_id=_optional_str(payload.get("doc_id")),
                url=_optional_str(payload.get("url")),
                section=_optional_str(payload.get("section")),
                page=_optional_int(payload.get("page")),
                doc_type=_optional_str(payload.get("doc_type")),
                year=_optional_int(payload.get("year")),
                program=_optional_str(payload.get("program")),
            )
        )
    return SparseChunkIndex(chunks)


def tokenize_text(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def _matches_filters(chunk: SparseChunk, filters: SearchFilters) -> bool:
    if filters.doc_type and chunk.doc_type != filters.doc_type:
        return False
    if filters.year is not None and chunk.year != filters.year:
        return False
    return True


def _optional_str(value: object) -> str | None:
    return str(value) if value is not None else None


def _optional_int(value: object) -> int | None:
    return int(value) if value is not None else None
