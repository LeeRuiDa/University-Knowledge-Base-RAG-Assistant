from __future__ import annotations

import streamlit as st

from src.models import SearchFilters
from src.predict import get_rag_assistant
from src.retriever import CorpusBusyError, CorpusNotReadyError

st.set_page_config(
    page_title="University Knowledge Base RAG Assistant",
    page_icon=":material/school:",
    layout="wide",
)


def main() -> None:
    assistant = get_rag_assistant()
    metadata = assistant.metadata()

    st.title("University Knowledge Base RAG Assistant")
    st.caption("Grounded answers over curated university policy documents.")

    with st.sidebar:
        st.subheader("Knowledge Base")
        st.write(f"Collection: `{metadata.collection_name}`")
        st.write(f"Qdrant mode: `{metadata.qdrant_mode}`")
        st.write(f"Embeddings: `{metadata.embedding_provider}`")
        st.write(f"Answering: `{metadata.generation_provider}`")
        st.write(f"Retrieval: `{metadata.retrieval_strategy}`")
        st.write(f"Ready: `{metadata.ready}`")
        st.write(f"Files indexed: `{metadata.files_indexed}`")
        st.write(f"Chunks indexed: `{metadata.chunks_indexed}`")
        if metadata.programs:
            st.write("Programs:")
            for program in metadata.programs:
                st.write(f"- `{program}`")
        if metadata.corpus_manifest_path:
            st.caption(f"Manifest: `{metadata.corpus_manifest_path}`")

        if st.button("Ingest / Reindex Corpus", use_container_width=True):
            with st.spinner("Indexing the corpus..."):
                try:
                    response = assistant.ingest(recreate=True)
                except Exception as exc:  # pragma: no cover - UI path
                    st.error(str(exc))
                else:
                    st.success(
                        "Indexed "
                        f"{response.files_indexed} files and {response.chunks_indexed} chunks."
                    )
                    st.rerun()

        doc_types = ["All", *metadata.document_types]
        years = ["All", *[str(year) for year in metadata.years]]
        selected_doc_type = st.selectbox("Document type filter", options=doc_types)
        selected_year = st.selectbox("Year filter", options=years)

    filters = SearchFilters(
        doc_type=None if selected_doc_type == "All" else selected_doc_type,
        year=None if selected_year == "All" else int(selected_year),
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for item in st.session_state.chat_history:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
            if item["role"] == "assistant" and item.get("sources"):
                _render_sources(item["sources"], item.get("warning"))

    prompt = st.chat_input("Ask about graduation rules, fees, internships, or academic policies.")
    if not prompt:
        return

    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            response = assistant.ask(prompt, filters=filters)
        except CorpusBusyError as exc:
            st.error(str(exc))
            return
        except CorpusNotReadyError as exc:
            st.error(str(exc))
            return
        except Exception as exc:  # pragma: no cover - UI path
            st.error(str(exc))
            return

        st.markdown(response.answer)
        _render_sources(response.sources, response.warning)

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": response.answer,
            "warning": response.warning,
            "sources": [source.model_dump() for source in response.sources],
        }
    )


def _render_sources(sources: list[dict] | list, warning: str | None) -> None:
    if warning:
        st.caption(warning)

    with st.expander("Sources and retrieved chunks", expanded=True):
        for source in sources:
            if hasattr(source, "model_dump"):
                source = source.model_dump()
            label = f"{source['source_id']} | {source['title']}"
            if source.get("page") is not None:
                label = f"{label} | page {source['page']}"
            st.markdown(f"**{label}**")
            st.markdown(
                f"`{source.get('doc_type')}` | `{source.get('year')}` | "
                f"`{source.get('program')}` | `{source.get('source')}`"
            )
            if source.get("url"):
                st.markdown(f"[Open official source]({source['url']})")
            if source.get("section"):
                st.caption(f"Section: {source['section']}")
            st.write(source["text"])


if __name__ == "__main__":
    main()
