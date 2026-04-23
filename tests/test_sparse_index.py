import json

from src.models import SearchFilters
from src.sparse_index import load_sparse_index


def test_sparse_index_search_respects_filters(tmp_path) -> None:
    catalog_path = tmp_path / "chunk_catalog.jsonl"
    rows = [
        {
            "chunk_id": "chunk-1",
            "doc_id": "internship-doc",
            "source": "data/raw/internship.html",
            "title": "Internship Credit",
            "section": "Policy",
            "doc_type": "internship_policy",
            "year": 2025,
            "program": "computer_science_undergraduate",
            "text": "Students may earn internship credit through CSCE 495.",
        },
        {
            "chunk_id": "chunk-2",
            "doc_id": "degree-doc",
            "source": "data/raw/degree.html",
            "title": "Degree Requirements",
            "section": "Requirements",
            "doc_type": "degree_requirements",
            "year": 2025,
            "program": "computer_science_undergraduate",
            "text": "Students complete technical electives to graduate.",
        },
        {
            "chunk_id": "chunk-3",
            "doc_id": "billing-doc",
            "source": "data/raw/billing.html",
            "title": "Billing Policies",
            "section": "Policy",
            "doc_type": "billing_policy",
            "year": 2025,
            "program": "computer_science_undergraduate",
            "text": "Late payment fees apply to delinquent student accounts.",
        },
    ]
    catalog_path.write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )

    index = load_sparse_index(catalog_path)

    results = index.search("internship credit", k=2)
    filtered_results = index.search(
        "internship credit",
        k=2,
        filters=SearchFilters(doc_type="degree_requirements"),
    )

    assert results[0][0].doc_id == "internship-doc"
    assert filtered_results == []
