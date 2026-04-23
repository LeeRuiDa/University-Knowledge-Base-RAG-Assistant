from __future__ import annotations

import hashlib
import math
import re

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from src.config import Settings


class HashEmbeddings(Embeddings):
    def __init__(self, dimension: int = 256) -> None:
        self.dimension = dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


def get_embeddings(settings: Settings) -> Embeddings:
    provider = settings.embedding_provider.lower()
    if provider == "openai":
        return OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            openai_api_key=settings.openai_api_key,
            openai_api_base=settings.openai_api_base,
        )
    if provider == "openrouter":
        return OpenAIEmbeddings(
            model=settings.openrouter_embedding_model,
            openai_api_key=settings.openrouter_api_key or settings.openai_api_key,
            openai_api_base=settings.openrouter_base_url,
            default_headers=_openrouter_headers(settings),
        )
    if provider == "hash":
        return HashEmbeddings()
    raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider}")


def _openrouter_headers(settings: Settings) -> dict[str, str]:
    headers = {"X-Title": settings.openrouter_app_name or settings.project_name}
    if settings.openrouter_http_referer:
        headers["HTTP-Referer"] = settings.openrouter_http_referer
    return headers
