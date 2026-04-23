from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SearchFilters(BaseModel):
    doc_type: str | None = Field(default=None, description="Filter to a single document type.")
    year: int | None = Field(default=None, description="Filter to a specific year.")


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    filters: SearchFilters | None = None


class SourceChunk(BaseModel):
    source_id: str
    chunk_id: str
    doc_id: str | None = None
    score: float | None = None
    source: str
    url: str | None = None
    title: str
    section: str | None = None
    page: int | None = None
    doc_type: str | None = None
    year: int | None = None
    program: str | None = None
    text: str


class AnswerResponse(BaseModel):
    question: str
    answer: str
    citations: list[str]
    grounded: bool
    warning: str | None = None
    filters_applied: SearchFilters | None = None
    sources: list[SourceChunk]


class IngestRequest(BaseModel):
    input_dir: str | None = None
    recreate: bool = True


class IngestResponse(BaseModel):
    status: Literal["ok"]
    input_dir: str
    files_indexed: int
    sections_loaded: int
    chunks_indexed: int
    collection_name: str
    metadata_path: str
    corpus_manifest_path: str | None = None


class MetadataSummary(BaseModel):
    ready: bool
    collection_name: str
    qdrant_mode: str
    embedding_provider: str
    generation_provider: str
    retrieval_strategy: str
    corpus_manifest_path: str | None = None
    files_indexed: int = 0
    sections_loaded: int = 0
    chunks_indexed: int = 0
    document_types: list[str] = Field(default_factory=list)
    years: list[int] = Field(default_factory=list)
    programs: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: Literal["ok"]
    ready: bool
    collection_name: str
    qdrant_mode: str
