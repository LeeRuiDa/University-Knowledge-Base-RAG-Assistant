from src.answer import ABSTAIN_RESPONSE, generate_answer
from src.answer_eval import (
    AnswerEvalRow,
    load_answer_eval_rows,
    score_citation_usefulness,
    score_refusal_behavior,
)
from src.config import Settings
from src.models import SourceChunk


def test_load_answer_eval_rows_splits_doc_ids_and_bools(tmp_path) -> None:
    eval_path = tmp_path / "answer_eval.csv"
    eval_path.write_text(
        (
            "question,expected_doc_ids,gold_answer,category,should_refuse,difficulty\n"
            '"Question one","doc-a|doc-b","Answer","policy",false,easy\n'
            '"Question two","","Refuse","refusal",true,medium\n'
        ),
        encoding="utf-8",
    )

    rows = load_answer_eval_rows(eval_path)

    assert rows[0].expected_doc_ids == ["doc-a", "doc-b"]
    assert rows[0].should_refuse is False
    assert rows[1].expected_doc_ids == []
    assert rows[1].should_refuse is True


def test_citation_usefulness_rewards_expected_docs() -> None:
    row = AnswerEvalRow(
        question="How many credits count as a tech elective?",
        expected_doc_ids=["unl_cs_internship_credit"],
        gold_answer="Three credits of CSCE 495 count as one technical elective.",
        category="internship",
        should_refuse=False,
        difficulty="easy",
    )
    sources = [
        {
            "source_id": "S1",
            "doc_id": "unl_cs_internship_credit",
            "text": "3 credits of CSCE 495 will count as one of your tech elective courses.",
        }
    ]

    score, note = score_citation_usefulness(
        row=row,
        answer_text="Three credits of CSCE 495 count as one technical elective.",
        response_citations=["S1"],
        sources=sources,
    )

    assert score == 3
    assert "expected documents" in note


def test_refusal_behavior_scores_exact_abstention_highly() -> None:
    score, note = score_refusal_behavior(
        should_refuse=True,
        answer_text=ABSTAIN_RESPONSE,
        citations=[],
    )

    assert score == 3
    assert "refused" in note.lower()


def test_generate_answer_keeps_abstention_ungrounded(monkeypatch) -> None:
    settings = Settings(generation_provider="openrouter", embedding_provider="hash")
    sources = [
        SourceChunk(
            source_id="S1",
            chunk_id="chunk-1",
            doc_id="doc-1",
            score=0.9,
            source="data/raw/example.md",
            title="Example",
            section="Policy",
            page=None,
            doc_type="policy",
            year=2025,
            text="Example text.",
        )
    ]

    monkeypatch.setattr("src.answer._generate_openrouter_answer", lambda *_args: ABSTAIN_RESPONSE)

    response = generate_answer("Unknown question?", sources, settings)

    assert response.answer == ABSTAIN_RESPONSE
    assert response.citations == []
    assert response.grounded is False


def test_generate_answer_normalizes_prefixed_abstention(monkeypatch) -> None:
    settings = Settings(generation_provider="openrouter", embedding_provider="hash")
    sources = [
        SourceChunk(
            source_id="S1",
            chunk_id="chunk-1",
            doc_id="doc-1",
            score=0.9,
            source="data/raw/example.md",
            title="Example",
            section="Policy",
            page=None,
            doc_type="policy",
            year=2025,
            text="Example text.",
        )
    ]

    monkeypatch.setattr(
        "src.answer._generate_openrouter_answer",
        lambda *_args: (
            "I don't know from the provided documents. "
            "None of the retrieved chunks mention this detail."
        ),
    )

    response = generate_answer("Unknown question?", sources, settings)

    assert response.answer == ABSTAIN_RESPONSE
    assert response.citations == []
    assert response.grounded is False
