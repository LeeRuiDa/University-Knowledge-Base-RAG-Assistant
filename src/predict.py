from __future__ import annotations

from functools import lru_cache

from src.answer import generate_answer
from src.config import Settings, get_settings
from src.embed import get_embeddings
from src.ingest import load_manifest, run_ingestion
from src.models import (
    AnswerResponse,
    HealthResponse,
    IngestResponse,
    MetadataSummary,
    SearchFilters,
)
from src.retriever import CorpusNotReadyError, retrieve_sources


class RAGAssistant:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.embeddings = get_embeddings(self.settings)

    def ingest(self, input_dir: str | None = None, recreate: bool = True) -> IngestResponse:
        return run_ingestion(
            settings=self.settings,
            embeddings=self.embeddings,
            input_dir=input_dir,
            recreate=recreate,
        )

    def metadata(self) -> MetadataSummary:
        return load_manifest(self.settings)

    def health(self) -> HealthResponse:
        manifest = self.metadata()
        return HealthResponse(
            status="ok",
            ready=manifest.ready,
            collection_name=self.settings.qdrant_collection_name,
            qdrant_mode=self.settings.qdrant_mode,
        )

    def ask(self, question: str, filters: SearchFilters | None = None) -> AnswerResponse:
        cleaned_question = question.strip()
        if len(cleaned_question) < 3:
            raise ValueError("Question must be at least 3 characters long.")
        if not self.metadata().ready:
            raise CorpusNotReadyError("The knowledge base is not indexed yet. Run ingestion first.")

        sources = retrieve_sources(
            question=cleaned_question,
            settings=self.settings,
            embeddings=self.embeddings,
            filters=filters,
        )
        return generate_answer(
            question=cleaned_question,
            sources=sources,
            settings=self.settings,
            filters=filters,
        )


@lru_cache(maxsize=1)
def get_rag_assistant() -> RAGAssistant:
    return RAGAssistant(get_settings())
