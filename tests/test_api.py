from fastapi.testclient import TestClient

from app.fastapi_app import app, get_pipeline
from src.models import (
    AnswerResponse,
    HealthResponse,
    IngestResponse,
    MetadataSummary,
    SearchFilters,
    SourceChunk,
)


class FakeAssistant:
    def health(self) -> HealthResponse:
        return HealthResponse(
            status="ok",
            ready=True,
            collection_name="test_collection",
            qdrant_mode="local",
        )

    def metadata(self) -> MetadataSummary:
        return MetadataSummary(
            ready=True,
            collection_name="test_collection",
            qdrant_mode="local",
            embedding_provider="hash",
            generation_provider="extractive",
            retrieval_strategy="hybrid",
            files_indexed=2,
            sections_loaded=4,
            chunks_indexed=6,
            document_types=["graduation_requirements"],
            years=[2025],
            sources=["data/raw/example.md"],
        )

    def ask(self, question: str, filters: SearchFilters | None = None) -> AnswerResponse:
        return AnswerResponse(
            question=question,
            answer="You need 128 total credits. [S1]",
            citations=["S1"],
            grounded=True,
            warning=(
                "Grounded in 1 retrieved chunk. "
                "Verify the citation before treating it as policy."
            ),
            filters_applied=filters,
            sources=[
                SourceChunk(
                    source_id="S1",
                    chunk_id="chunk-1",
                    score=0.98,
                    source="data/raw/example.md",
                    title="Graduation Requirements",
                    section="Credit Requirements",
                    page=None,
                    doc_type="graduation_requirements",
                    year=2025,
                    text="To graduate, a student must complete at least 128 total credits.",
                )
            ],
        )

    def ingest(self, input_dir: str | None = None, recreate: bool = True) -> IngestResponse:
        return IngestResponse(
            status="ok",
            input_dir=input_dir or "data/raw",
            files_indexed=2,
            sections_loaded=4,
            chunks_indexed=6,
            collection_name="test_collection",
            metadata_path="data/parsed/ingestion_manifest.json",
        )


def test_health_endpoint() -> None:
    app.dependency_overrides[get_pipeline] = lambda: FakeAssistant()
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    app.dependency_overrides.clear()


def test_ask_endpoint() -> None:
    app.dependency_overrides[get_pipeline] = lambda: FakeAssistant()
    client = TestClient(app)

    response = client.post(
        "/ask",
        json={
            "question": "How many credits are required to graduate?",
            "filters": {"doc_type": "graduation_requirements", "year": 2025},
        },
    )

    assert response.status_code == 200
    assert response.json()["citations"] == ["S1"]
    app.dependency_overrides.clear()
