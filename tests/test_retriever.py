from src.retriever import RankedChunk, _select_diverse_candidates, preferred_doc_types_for_query


def _candidate(doc_id: str, rerank_score: float) -> RankedChunk:
    return RankedChunk(
        chunk_id=f"{doc_id}-{rerank_score}",
        doc_id=doc_id,
        source=f"data/raw/{doc_id}.html",
        text="chunk text",
        title=doc_id,
        rerank_score=rerank_score,
    )


def test_query_doc_type_rules_cover_new_hardening_cases() -> None:
    registration = preferred_doc_types_for_query(
        "How can a student cancel registration before the semester begins without being charged?"
    )
    attendance = preferred_doc_types_for_query(
        "What may departments do if a student misses the first class meeting?"
    )
    grade_policy = preferred_doc_types_for_query(
        "What is the first step in a dispute involving a college policy or grade?"
    )

    assert "registration_policy" in registration
    assert "attendance_policy" in attendance
    assert "degree_requirements" in grade_policy


def test_select_diverse_candidates_limits_chunks_per_document() -> None:
    candidates = [
        _candidate("doc-a", 0.95),
        _candidate("doc-a", 0.90),
        _candidate("doc-b", 0.85),
        _candidate("doc-c", 0.80),
    ]

    selected = _select_diverse_candidates(candidates, k=3, max_chunks_per_doc=1)

    assert [candidate.doc_id for candidate in selected] == ["doc-a", "doc-b", "doc-c"]
