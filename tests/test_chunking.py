from langchain_core.documents import Document

from src.chunking import _make_chunk_id, chunk_documents


def test_chunking_preserves_metadata_and_adds_chunk_ids() -> None:
    text = "Graduation requirements " * 200
    documents = [
        Document(
            page_content=text,
            metadata={
                "source": "data/raw/2025_graduation_requirements.md",
                "title": "Graduation Requirements",
                "section": "Credit Requirements",
                "page": None,
                "doc_type": "graduation_requirements",
                "year": 2025,
            },
        )
    ]

    chunks = chunk_documents(documents, chunk_size=80, chunk_overlap=10)

    assert len(chunks) > 1
    assert all(chunk.metadata["source"] == documents[0].metadata["source"] for chunk in chunks)
    assert all(chunk.metadata["doc_type"] == "graduation_requirements" for chunk in chunks)
    assert all("chunk_id" in chunk.metadata for chunk in chunks)


def test_chunking_empty_document_list_returns_empty() -> None:
    chunks = chunk_documents([], chunk_size=80, chunk_overlap=10)
    assert chunks == []


def test_chunking_empty_content_returns_empty() -> None:
    documents = [
        Document(page_content="", metadata={"source": "empty.md"}),
        Document(page_content="   \n\n  ", metadata={"source": "whitespace.md"}),
    ]
    chunks = chunk_documents(documents, chunk_size=80, chunk_overlap=10)
    assert chunks == []


def test_chunking_short_text_produces_single_chunk() -> None:
    documents = [
        Document(
            page_content="Short policy statement.",
            metadata={"source": "short.md", "doc_type": "policy"},
        )
    ]
    chunks = chunk_documents(documents, chunk_size=100, chunk_overlap=10)
    assert len(chunks) == 1
    assert chunks[0].page_content == "Short policy statement."
    assert chunks[0].metadata["chunk_index"] == 0
    assert len(chunks[0].metadata["chunk_id"]) == 16


def test_chunking_missing_metadata_keys_defaults_safely() -> None:
    documents = [
        Document(
            page_content="Content with minimal empty metadata dict.",
            metadata={},
        )
    ]
    chunks = chunk_documents(documents, chunk_size=100, chunk_overlap=10)
    assert len(chunks) == 1
    assert "chunk_id" in chunks[0].metadata
    assert chunks[0].metadata["chunk_index"] == 0


def test_chunk_id_determinism_and_uniqueness() -> None:
    meta1 = {"source": "handbook.pdf", "page": 12, "section": "Admissions"}
    meta2 = {"source": "handbook.pdf", "page": 12, "section": "Admissions"}
    meta_diff = {"source": "handbook.pdf", "page": 13, "section": "Admissions"}

    id1 = _make_chunk_id(meta1, "Sample text content", 0)
    id2 = _make_chunk_id(meta2, "Sample text content", 0)
    id_diff_idx = _make_chunk_id(meta1, "Sample text content", 1)
    id_diff_meta = _make_chunk_id(meta_diff, "Sample text content", 0)
    id_diff_text = _make_chunk_id(meta1, "Different text content", 0)

    # Identical inputs produce identical hash
    assert id1 == id2
    assert len(id1) == 16

    # Any change in index, metadata, or text produces a different hash
    assert id1 != id_diff_idx
    assert id1 != id_diff_meta
    assert id1 != id_diff_text
