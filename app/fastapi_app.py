from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException

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


def get_pipeline() -> RAGAssistant:
    return get_rag_assistant()


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


@app.post("/ingest", response_model=IngestResponse)
def ingest(
    request: IngestRequest,
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
    pipeline: RAGAssistant = Depends(get_pipeline),
) -> IngestResponse:
    try:
        return pipeline.ingest(input_dir=request.input_dir, recreate=True)
    except CorpusBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
