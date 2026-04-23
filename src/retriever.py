from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from src.config import Settings
from src.models import SearchFilters, SourceChunk
from src.sparse_index import SparseChunk, load_sparse_index, tokenize_text

CHUNK_CATALOG_FILENAME = "chunk_catalog.jsonl"
COURSE_CODE_PATTERN = re.compile(r"\b[A-Z]{2,5}\s?\d{3}[A-Z]?\b")

QUERY_DOC_TYPE_RULES = {
    "internship_policy": (
        "internship",
        "intern",
        "co-op",
        "coop",
        "cpt",
        "faculty sponsor",
        "tech elective",
        "csce 495",
    ),
    "capstone_guidelines": (
        "capstone",
        "senior design",
        "showcase",
        "project sponsor",
    ),
    "registration_policy": (
        "withdraw",
        "withdrawal",
        "drop class",
        "add class",
        "cancel registration",
        "registration before the semester begins",
        "registration cancellation",
        "partial refund",
        "full refund",
    ),
    "attendance_policy": (
        "attendance",
        "absence",
        "first class meeting",
        "misses the first class",
        "missed the first class",
    ),
    "academic_calendar": (
        "academic calendar",
        "priority registration",
        "open registration",
        "semester",
        "deadline",
    ),
    "tuition_fees": (
        "tuition",
        "credit hour",
        "registration fee",
        "technology fee",
        "late registration fee",
    ),
    "billing_policy": (
        "hold",
        "delinquent",
        "financial responsibility",
        "econsent",
        "sanction",
    ),
    "payment_faq": (
        "payment",
        "bill",
        "online check",
        "credit card",
        "refund",
        "myred",
    ),
    "financial_aid": (
        "cost of attendance",
        "housing insecurity",
        "food insecurity",
        "emergency aid",
    ),
    "scholarship_policy": (
        "scholarship",
        "scholarships",
        "scholarship application",
        "financial aid",
        "sap",
        "satisfactory academic progress",
        "fafsa",
    ),
    "student_services": (
        "support",
        "tutoring",
        "wellbeing",
        "nest",
        "student organization",
    ),
    "degree_requirements": (
        "professional admission",
        "catalog year",
        "credit hours",
        "major requirements",
        "grade rules",
        "grade appeal",
        "college policy",
        "dispute",
    ),
    "course_catalog": (
        "prerequisite",
        "course description",
        "course details",
    ),
}


class CorpusNotReadyError(RuntimeError):
    """Raised when the vector store has not been built yet."""


class CorpusBusyError(RuntimeError):
    """Raised when the local Qdrant store is locked by another process."""


@dataclass
class RankedChunk:
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
    dense_rank: int | None = None
    dense_score: float | None = None
    sparse_rank: int | None = None
    sparse_score: float | None = None
    fused_score: float = 0.0
    rerank_score: float = 0.0
    metadata_boost: float = 0.0


def index_documents(
    documents: list[Document],
    embeddings: Embeddings,
    settings: Settings,
    recreate: bool = True,
) -> int:
    if not documents:
        raise ValueError("No chunked documents were provided for indexing.")

    try:
        client = QdrantClient(**_connection_kwargs(settings))
    except Exception as exc:
        _raise_if_local_storage_locked(exc, settings)
        raise
    try:
        if recreate:
            _delete_collection_if_present(client, settings.qdrant_collection_name)

        _ensure_collection(client, settings, embeddings)
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=settings.qdrant_collection_name,
            embedding=embeddings,
            retrieval_mode=RetrievalMode.DENSE,
        )
        vector_store.add_documents(
            documents=documents,
            ids=[_point_id_for_document(document) for document in documents],
        )
        _create_payload_indexes(vector_store, settings)
        return len(documents)
    finally:
        _close_client(client)


def retrieve_sources(
    question: str,
    settings: Settings,
    embeddings: Embeddings,
    filters: SearchFilters | None = None,
) -> list[SourceChunk]:
    strategy = settings.retrieval_strategy.lower()
    if strategy == "dense":
        dense_candidates = _dense_search(
            question,
            settings,
            embeddings,
            filters,
            settings.retrieval_k,
        )
        return _finalize_sources(dense_candidates, score_attr="dense_score")
    if strategy == "hybrid":
        return _hybrid_search(question, settings, embeddings, filters)
    raise ValueError(f"Unsupported retrieval strategy: {settings.retrieval_strategy}")


def _hybrid_search(
    question: str,
    settings: Settings,
    embeddings: Embeddings,
    filters: SearchFilters | None,
) -> list[SourceChunk]:
    dense_candidates = _dense_search(
        question,
        settings,
        embeddings,
        filters,
        settings.dense_retrieval_k,
    )

    catalog_path = settings.data_parsed_dir / CHUNK_CATALOG_FILENAME
    if not catalog_path.exists():
        return _finalize_sources(dense_candidates[: settings.retrieval_k], score_attr="dense_score")

    sparse_index = load_sparse_index(catalog_path)
    sparse_results = sparse_index.search(question, settings.sparse_retrieval_k, filters=filters)

    candidates = _fuse_candidates(question, dense_candidates, sparse_results, settings)
    reranked = _rerank_candidates(question, candidates, settings)
    final_candidates = _select_diverse_candidates(
        reranked,
        k=settings.retrieval_k,
        max_chunks_per_doc=settings.max_chunks_per_doc,
    )
    return _finalize_sources(final_candidates, score_attr="rerank_score")


def _dense_search(
    question: str,
    settings: Settings,
    embeddings: Embeddings,
    filters: SearchFilters | None,
    k: int,
) -> list[RankedChunk]:
    vector_store = _get_existing_vector_store(settings, embeddings)
    try:
        search_kwargs: dict[str, object] = {}
        qdrant_filter = _build_qdrant_filter(filters)
        if qdrant_filter is not None:
            search_kwargs["filter"] = qdrant_filter
        if settings.similarity_score_threshold is not None:
            search_kwargs["score_threshold"] = settings.similarity_score_threshold

        results = vector_store.similarity_search_with_relevance_scores(
            query=question,
            k=k,
            **search_kwargs,
        )
        return [
            RankedChunk(
                chunk_id=str(document.metadata.get("chunk_id", "")),
                source=str(document.metadata.get("source", "")),
                text=document.page_content.strip(),
                title=str(document.metadata.get("title", "")),
                doc_id=_optional_str(document.metadata.get("doc_id")),
                url=_optional_str(document.metadata.get("url")),
                section=_optional_str(document.metadata.get("section")),
                page=_optional_int(document.metadata.get("page")),
                doc_type=_optional_str(document.metadata.get("doc_type")),
                year=_optional_int(document.metadata.get("year")),
                program=_optional_str(document.metadata.get("program")),
                dense_rank=rank,
                dense_score=float(score),
            )
            for rank, (document, score) in enumerate(results, start=1)
        ]
    finally:
        _close_client(vector_store.client)


def _fuse_candidates(
    question: str,
    dense_candidates: list[RankedChunk],
    sparse_results: list[tuple[SparseChunk, float]],
    settings: Settings,
) -> list[RankedChunk]:
    merged: dict[str, RankedChunk] = {
        candidate.chunk_id: candidate for candidate in dense_candidates
    }

    for rank, (chunk, score) in enumerate(sparse_results, start=1):
        candidate = merged.get(chunk.chunk_id)
        if candidate is None:
            candidate = RankedChunk(
                chunk_id=chunk.chunk_id,
                source=chunk.source,
                text=chunk.text,
                title=chunk.title,
                doc_id=chunk.doc_id,
                url=chunk.url,
                section=chunk.section,
                page=chunk.page,
                doc_type=chunk.doc_type,
                year=chunk.year,
                program=chunk.program,
            )
            merged[chunk.chunk_id] = candidate
        candidate.sparse_rank = rank
        candidate.sparse_score = score

    preferred_doc_types = preferred_doc_types_for_query(question)
    for candidate in merged.values():
        candidate.fused_score = _rrf_score(candidate, settings)
        candidate.metadata_boost = _metadata_boost(candidate, preferred_doc_types, settings)
        candidate.fused_score += candidate.metadata_boost

    fused = sorted(merged.values(), key=lambda item: item.fused_score, reverse=True)
    return fused[: settings.hybrid_candidate_k]


def _rerank_candidates(
    question: str,
    candidates: list[RankedChunk],
    settings: Settings,
) -> list[RankedChunk]:
    if not candidates:
        return []

    query_tokens = set(tokenize_text(question))
    query_text = question.lower()
    course_codes = [
        code.lower().replace(" ", "")
        for code in COURSE_CODE_PATTERN.findall(question.upper())
    ]
    max_fused = max(candidate.fused_score for candidate in candidates) or 1.0

    for candidate in candidates:
        title_text = f"{candidate.title} {candidate.section or ''}".lower()
        candidate_tokens = set(tokenize_text(f"{title_text}\n{candidate.text}"))
        title_tokens = set(tokenize_text(title_text))

        overlap = _safe_divide(len(query_tokens & candidate_tokens), len(query_tokens) or 1)
        title_overlap = _safe_divide(len(query_tokens & title_tokens), len(query_tokens) or 1)
        phrase_bonus = _phrase_bonus(query_text, candidate)
        code_bonus = _course_code_bonus(course_codes, candidate)
        specificity_penalty = _specificity_penalty(candidate)
        fused_norm = candidate.fused_score / max_fused

        candidate.rerank_score = (
            (0.52 * fused_norm)
            + (0.22 * overlap)
            + (0.14 * title_overlap)
            + phrase_bonus
            + code_bonus
            + candidate.metadata_boost
            + specificity_penalty
        )

    return sorted(candidates, key=lambda item: item.rerank_score, reverse=True)


def _select_diverse_candidates(
    candidates: list[RankedChunk],
    k: int,
    max_chunks_per_doc: int,
) -> list[RankedChunk]:
    selected: list[RankedChunk] = []
    deferred: list[RankedChunk] = []
    counts: dict[str, int] = {}

    for candidate in candidates:
        key = candidate.doc_id or candidate.source
        if counts.get(key, 0) >= max_chunks_per_doc:
            deferred.append(candidate)
            continue
        selected.append(candidate)
        counts[key] = counts.get(key, 0) + 1
        if len(selected) == k:
            return selected

    for candidate in deferred:
        selected.append(candidate)
        if len(selected) == k:
            break

    return selected


def preferred_doc_types_for_query(question: str) -> set[str]:
    lowered = question.lower()
    matched: set[str] = set()
    for doc_type, phrases in QUERY_DOC_TYPE_RULES.items():
        if any(phrase in lowered for phrase in phrases):
            matched.add(doc_type)
    return matched


def _rrf_score(candidate: RankedChunk, settings: Settings) -> float:
    dense = (
        settings.dense_rrf_weight / (settings.rrf_k + candidate.dense_rank)
        if candidate.dense_rank is not None
        else 0.0
    )
    sparse = (
        settings.sparse_rrf_weight / (settings.rrf_k + candidate.sparse_rank)
        if candidate.sparse_rank is not None
        else 0.0
    )
    return dense + sparse


def _metadata_boost(
    candidate: RankedChunk,
    preferred_doc_types: set[str],
    settings: Settings,
) -> float:
    if not preferred_doc_types or not candidate.doc_type:
        return 0.0
    if candidate.doc_type in preferred_doc_types:
        return settings.metadata_boost_weight
    return 0.0


def _phrase_bonus(query_text: str, candidate: RankedChunk) -> float:
    searchable = f"{candidate.title} {candidate.section or ''} {candidate.text}".lower()
    bonus = 0.0
    phrases = [
        "internship credit",
        "senior design",
        "grade appeals",
        "registration cancellation",
        "first class meeting",
        "satisfactory academic progress",
        "academic calendar",
        "priority registration",
        "student account",
        "recommended form of payment",
        "credit card",
        "online check",
        "late payment fee",
    ]
    for phrase in phrases:
        if phrase in query_text and phrase in searchable:
            bonus += 0.08
    if "student account" in query_text and "student account" in candidate.title.lower():
        bonus += 0.06
    if "recommend" in query_text and "recommended form of payment" in searchable:
        bonus += 0.06
    return min(bonus, 0.24)


def _course_code_bonus(course_codes: list[str], candidate: RankedChunk) -> float:
    if not course_codes:
        return 0.0
    searchable = (
        f"{candidate.title} {candidate.section or ''} {candidate.text}"
    ).lower().replace(" ", "")
    return 0.12 if any(code in searchable for code in course_codes) else 0.0


def _specificity_penalty(candidate: RankedChunk) -> float:
    section = (candidate.section or "").strip().lower()
    title = candidate.title.strip().lower()
    if section in {"content", title}:
        return -0.05
    if section.startswith("step "):
        return -0.06
    if candidate.section is None:
        return -0.03
    return 0.0


def _candidate_to_source_chunk(candidate: RankedChunk, score: float | None) -> SourceChunk:
    return SourceChunk(
        source_id="",
        chunk_id=candidate.chunk_id,
        doc_id=candidate.doc_id,
        score=round(float(score), 4) if score is not None else None,
        source=candidate.source,
        url=candidate.url,
        title=candidate.title,
        section=candidate.section,
        page=candidate.page,
        doc_type=candidate.doc_type,
        year=candidate.year,
        program=candidate.program,
        text=candidate.text,
    )


def _finalize_sources(candidates: list[RankedChunk], score_attr: str) -> list[SourceChunk]:
    sources: list[SourceChunk] = []
    for index, candidate in enumerate(candidates, start=1):
        raw_score = getattr(candidate, score_attr) or candidate.fused_score or candidate.dense_score
        source = _candidate_to_source_chunk(candidate, raw_score)
        source.source_id = f"S{index}"
        sources.append(source)
    return sources


def _get_existing_vector_store(settings: Settings, embeddings: Embeddings) -> QdrantVectorStore:
    try:
        return QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            collection_name=settings.qdrant_collection_name,
            retrieval_mode=RetrievalMode.DENSE,
            **_connection_kwargs(settings),
        )
    except Exception as exc:  # pragma: no cover - exercised through service boundary
        _raise_if_local_storage_locked(exc, settings)
        raise CorpusNotReadyError(
            "The knowledge base is not indexed yet. Run ingestion first."
        ) from exc


def _connection_kwargs(settings: Settings) -> dict[str, object]:
    if settings.qdrant_url:
        kwargs: dict[str, object] = {"url": settings.qdrant_url}
        if settings.qdrant_api_key:
            kwargs["api_key"] = settings.qdrant_api_key
        return kwargs
    return {"path": str(settings.qdrant_local_path)}


def _build_qdrant_filter(filters: SearchFilters | None) -> qdrant_models.Filter | None:
    if filters is None:
        return None

    must: list[qdrant_models.FieldCondition] = []
    if filters.doc_type:
        must.append(
            qdrant_models.FieldCondition(
                key="metadata.doc_type",
                match=qdrant_models.MatchValue(value=filters.doc_type),
            )
        )
    if filters.year is not None:
        must.append(
            qdrant_models.FieldCondition(
                key="metadata.year",
                match=qdrant_models.MatchValue(value=filters.year),
            )
        )

    return qdrant_models.Filter(must=must) if must else None


def _create_payload_indexes(vector_store: QdrantVectorStore, settings: Settings) -> None:
    if settings.qdrant_mode == "local":
        return

    try:
        vector_store.client.create_payload_index(
            collection_name=settings.qdrant_collection_name,
            field_name="metadata.doc_type",
            field_schema="keyword",
        )
        vector_store.client.create_payload_index(
            collection_name=settings.qdrant_collection_name,
            field_name="metadata.year",
            field_schema="integer",
        )
    except Exception:
        return


def _ensure_collection(client: QdrantClient, settings: Settings, embeddings: Embeddings) -> None:
    if client.collection_exists(settings.qdrant_collection_name):
        return

    vector_size = len(embeddings.embed_query("dimension probe"))
    client.create_collection(
        collection_name=settings.qdrant_collection_name,
        vectors_config=qdrant_models.VectorParams(
            size=vector_size,
            distance=qdrant_models.Distance.COSINE,
        ),
    )


def _delete_collection_if_present(client: QdrantClient, collection_name: str) -> None:
    try:
        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)
    except Exception:
        try:
            client.delete_collection(collection_name)
        except Exception:
            return


def _close_client(client: QdrantClient) -> None:
    close = getattr(client, "close", None)
    if callable(close):
        close()


def _point_id_for_document(document: Document) -> str:
    chunk_id = str(document.metadata.get("chunk_id", ""))
    source = str(document.metadata.get("source", ""))
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}:{chunk_id}"))


def _raise_if_local_storage_locked(exc: Exception, settings: Settings) -> None:
    if settings.qdrant_mode != "local":
        return

    message = str(exc).lower()
    if "already accessed by another instance of qdrant client" not in message:
        return

    raise CorpusBusyError(
        "Local Qdrant storage is already in use by another process. "
        "Stop the other Streamlit/Uvicorn/Python session or set `QDRANT_URL` "
        "to use a shared Qdrant server."
    ) from exc


def _optional_str(value: object) -> str | None:
    return str(value) if value is not None else None


def _optional_int(value: object) -> int | None:
    return int(value) if value is not None else None


def _safe_divide(numerator: float, denominator: float) -> float:
    if math.isclose(denominator, 0.0):
        return 0.0
    return numerator / denominator
