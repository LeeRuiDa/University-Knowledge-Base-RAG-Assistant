from __future__ import annotations

from secrets import compare_digest
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.config import Settings, get_settings
from src.models import (
    AnswerResponse,
    AskRequest,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    MetadataSummary,
)
from src.predict import RAGAssistant, get_rag_assistant
from src.retriever import CorpusBusyError, CorpusNotReadyError

app = FastAPI(
    title="University Knowledge Base RAG Assistant",
    version="0.1.0",
    description="Grounded question answering over curated university documents.",
)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_frontend_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-Admin-Key"],
)


def get_pipeline() -> RAGAssistant:
    return get_rag_assistant()


def require_admin_key(
    x_admin_key: Annotated[str | None, Header(alias="X-Admin-Key")] = None,
    current_settings: Settings = Depends(get_settings),
) -> None:
    configured_key = current_settings.admin_api_key
    if configured_key is None:
        raise HTTPException(
            status_code=503,
            detail="Administrative API operations are disabled on this deployment.",
        )
    if x_admin_key is None or not compare_digest(
        x_admin_key,
        configured_key.get_secret_value(),
    ):
        raise HTTPException(status_code=401, detail="A valid administrative key is required.")


@app.get("/health", response_model=HealthResponse)
def health(pipeline: RAGAssistant = Depends(get_pipeline)) -> HealthResponse:
    return pipeline.health()


@app.get("/metadata", response_model=MetadataSummary)
def metadata(pipeline: RAGAssistant = Depends(get_pipeline)) -> MetadataSummary:
    return pipeline.metadata()


@app.post("/ask", response_model=AnswerResponse)
def ask(request: AskRequest, pipeline: RAGAssistant = Depends(get_pipeline)) -> AnswerResponse:
    try:
        return pipeline.ask(request.question, request.filters)
    except CorpusBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except CorpusNotReadyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="The assistant could not complete this request. Please try again.",
        ) from exc


@app.post("/ingest", response_model=IngestResponse)
def ingest(
    request: IngestRequest,
    _authorized: None = Depends(require_admin_key),
    pipeline: RAGAssistant = Depends(get_pipeline),
) -> IngestResponse:
    try:
        return pipeline.ingest(input_dir=request.input_dir, recreate=request.recreate)
    except CorpusBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/reindex", response_model=IngestResponse)
def reindex(
    request: IngestRequest,
    _authorized: None = Depends(require_admin_key),
    pipeline: RAGAssistant = Depends(get_pipeline),
) -> IngestResponse:
    try:
        return pipeline.ingest(input_dir=request.input_dir, recreate=True)
    except CorpusBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
