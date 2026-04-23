from __future__ import annotations

import hashlib

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )

    split_docs = splitter.split_documents(documents)
    chunked: list[Document] = []
    for index, document in enumerate(split_docs):
        metadata = dict(document.metadata)
        metadata["chunk_index"] = index
        metadata["chunk_id"] = _make_chunk_id(metadata, document.page_content, index)
        chunked.append(Document(page_content=document.page_content, metadata=metadata))

    return chunked


def _make_chunk_id(metadata: dict[str, object], text: str, index: int) -> str:
    source = str(metadata.get("source", "unknown"))
    page = str(metadata.get("page", "na"))
    section = str(metadata.get("section", "na"))
    payload = f"{source}|{page}|{section}|{index}|{text[:120]}"
    digest = hashlib.sha1(payload.encode()).hexdigest()
    return digest[:16]
