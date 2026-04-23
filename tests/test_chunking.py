from langchain_core.documents import Document

from src.chunking import chunk_documents


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

