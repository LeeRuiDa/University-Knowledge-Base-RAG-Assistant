from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    project_name: str = "University Knowledge Base RAG Assistant"

    data_raw_dir: Path = Path("data/raw")
    data_parsed_dir: Path = Path("data/parsed")
    data_eval_dir: Path = Path("data/eval")
    reports_dir: Path = Path("reports")
    corpus_manifest_path: Path = Path("data/corpus_manifest.csv")

    qdrant_collection_name: str = "university_knowledge_base"
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_local_path: Path = Path(".qdrant")

    embedding_provider: str = "hash"
    generation_provider: str = "extractive"
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4.1-mini"
    openai_api_base: str | None = None

    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_embedding_model: str = "openai/text-embedding-3-small"
    openrouter_chat_model: str = "openai/gpt-4.1-mini"
    openrouter_http_referer: str | None = None
    openrouter_app_name: str | None = "University Knowledge Base Assistant"

    chunk_size: int = 700
    chunk_overlap: int = 80
    retrieval_k: int = 5
    retrieval_strategy: str = "hybrid"
    dense_retrieval_k: int = 10
    sparse_retrieval_k: int = 10
    hybrid_candidate_k: int = 15
    rrf_k: int = 60
    dense_rrf_weight: float = 1.0
    sparse_rrf_weight: float = 1.0
    metadata_boost_weight: float = 0.12
    max_chunks_per_doc: int = 2
    similarity_score_threshold: float | None = None
    answer_temperature: float = 0.0
    answer_max_tokens: int = 400

    def ensure_directories(self) -> None:
        self.data_raw_dir.mkdir(parents=True, exist_ok=True)
        self.data_parsed_dir.mkdir(parents=True, exist_ok=True)
        self.data_eval_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.qdrant_local_path.mkdir(parents=True, exist_ok=True)

    @property
    def qdrant_mode(self) -> str:
        return "server" if self.qdrant_url else "local"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
