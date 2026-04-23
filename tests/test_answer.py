from src.answer import ABSTAIN_RESPONSE, generate_answer
from src.config import Settings
from src.models import SourceChunk


def test_extractive_answer_contains_citations() -> None:
    settings = Settings(generation_provider="extractive", embedding_provider="hash")
    sources = [
        SourceChunk(
            source_id="S1",
            chunk_id="chunk-1",
            score=0.95,
            source="data/raw/2025_graduation_requirements.md",
            title="Graduation Requirements",
            section="Credit Requirements",
            page=None,
            doc_type="graduation_requirements",
            year=2025,
            text="To graduate, a student must complete at least 128 total credits.",
        )
    ]

    response = generate_answer("How many credits are required to graduate?", sources, settings)

    assert "128 total credits" in response.answer
    assert response.citations == ["S1"]
    assert response.grounded is True


def test_generate_answer_abstains_without_sources() -> None:
    settings = Settings(generation_provider="extractive", embedding_provider="hash")

    response = generate_answer("When is convocation?", [], settings)

    assert response.answer == ABSTAIN_RESPONSE
    assert response.grounded is False

