from pathlib import Path

from src.loaders import load_documents_from_path


def test_markdown_loader_extracts_title_and_metadata(tmp_path: Path) -> None:
    file_path = tmp_path / "2025_internship_policy.md"
    file_path.write_text(
        "# Internship Policy 2025\n\n## Eligibility\n\nStudents need 90 credits.\n",
        encoding="utf-8",
    )

    documents = load_documents_from_path(tmp_path)

    assert len(documents) >= 1
    assert documents[0].metadata["title"] == "Internship Policy 2025"
    assert documents[0].metadata["doc_type"] == "internship_rules"
    assert documents[0].metadata["year"] == 2025

