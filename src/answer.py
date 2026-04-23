from __future__ import annotations

import re

from langchain_openai import ChatOpenAI

from src.config import Settings
from src.models import AnswerResponse, SearchFilters, SourceChunk

ABSTAIN_RESPONSE = "I don't know from the provided documents."


def generate_answer(
    question: str,
    sources: list[SourceChunk],
    settings: Settings,
    filters: SearchFilters | None = None,
) -> AnswerResponse:
    if not sources:
        return AnswerResponse(
            question=question,
            answer=ABSTAIN_RESPONSE,
            citations=[],
            grounded=False,
            warning="No supporting chunks were retrieved for this question.",
            filters_applied=filters,
            sources=[],
        )

    provider = settings.generation_provider.lower()
    if provider == "openai":
        answer = _generate_openai_answer(question, sources, settings)
    elif provider == "openrouter":
        answer = _generate_openrouter_answer(question, sources, settings)
    elif provider == "extractive":
        answer = _generate_extractive_answer(question, sources)
    else:
        raise ValueError(f"Unsupported generation provider: {settings.generation_provider}")

    if _is_abstention(answer):
        return AnswerResponse(
            question=question,
            answer=ABSTAIN_RESPONSE,
            citations=[],
            grounded=False,
            warning="The assistant abstained because the retrieved evidence was insufficient.",
            filters_applied=filters,
            sources=sources,
        )

    citations = _extract_citations(answer) or [source.source_id for source in sources[:2]]
    if citations and not _extract_citations(answer):
        answer = f"{answer} {' '.join(f'[{citation}]' for citation in citations)}".strip()

    warning = _build_warning(sources, citations, answer)

    return AnswerResponse(
        question=question,
        answer=answer,
        citations=citations,
        grounded=answer != ABSTAIN_RESPONSE,
        warning=warning,
        filters_applied=filters,
        sources=sources,
    )


def _generate_openai_answer(question: str, sources: list[SourceChunk], settings: Settings) -> str:
    llm = ChatOpenAI(
        model=settings.openai_chat_model,
        temperature=settings.answer_temperature,
        max_tokens=settings.answer_max_tokens,
        openai_api_key=settings.openai_api_key,
        openai_api_base=settings.openai_api_base,
    )
    return _invoke_answer_llm(llm, question, sources)

def _generate_openrouter_answer(
    question: str,
    sources: list[SourceChunk],
    settings: Settings,
) -> str:
    llm = ChatOpenAI(
        model=settings.openrouter_chat_model,
        temperature=settings.answer_temperature,
        max_tokens=settings.answer_max_tokens,
        openai_api_key=settings.openrouter_api_key or settings.openai_api_key,
        openai_api_base=settings.openrouter_base_url,
        default_headers=_openrouter_headers(settings),
    )
    return _invoke_answer_llm(llm, question, sources)


def _invoke_answer_llm(
    llm: ChatOpenAI,
    question: str,
    sources: list[SourceChunk],
) -> str:
    system_prompt = (
        "You answer questions about university policies using only the provided context. "
        "If the answer is not supported by the context, reply exactly with: "
        f"{ABSTAIN_RESPONSE} "
        "Do not add any explanation or citations after that abstention sentence. "
        "Use only facts that are explicitly stated in the retrieved text. "
        "Do not infer alternative steps, timelines, or consequences "
        "unless the cited chunk says them. "
        "Prefer the shortest direct answer that fully addresses the question. "
        "If both a broad overview and a specific policy chunk support the answer, "
        "prefer the specific chunk. "
        "Every factual claim must cite one or more source ids like [S1] or [S2]. "
        "Keep the answer concise and practical."
    )
    human_prompt = (
        f"Question:\n{question}\n\n"
        f"Context:\n{_format_context(sources)}\n\n"
        "Answer with citations."
    )

    response = llm.invoke(
        [
            ("system", system_prompt),
            ("human", human_prompt),
        ]
    )

    return _message_to_text(response.content).strip()


def _generate_extractive_answer(question: str, sources: list[SourceChunk]) -> str:
    question_terms = {term for term in re.findall(r"\b[a-zA-Z0-9]{3,}\b", question.lower())}
    scored_sentences: list[tuple[int, str, str]] = []

    for source in sources[:3]:
        for sentence in _split_sentences(source.text):
            sentence_terms = {term for term in re.findall(r"\b[a-zA-Z0-9]{3,}\b", sentence.lower())}
            overlap = len(question_terms & sentence_terms)
            if overlap:
                scored_sentences.append((overlap, sentence, source.source_id))

    if not scored_sentences:
        fallback = _first_nonempty_sentence(sources[0].text)
        return f"{fallback} [{sources[0].source_id}]"

    scored_sentences.sort(key=lambda item: item[0], reverse=True)
    used_sources: list[str] = []
    chosen_sentences: list[str] = []
    seen_sentences: set[str] = set()

    for _, sentence, source_id in scored_sentences:
        if len(chosen_sentences) == 2:
            break
        normalized = sentence.strip().lower()
        if normalized in seen_sentences:
            continue
        seen_sentences.add(normalized)
        chosen_sentences.append(sentence)
        if source_id not in used_sources:
            used_sources.append(source_id)

    citation_suffix = " ".join(f"[{source_id}]" for source_id in used_sources)
    return f"{' '.join(chosen_sentences)} {citation_suffix}".strip()


def _format_context(sources: list[SourceChunk]) -> str:
    blocks = []
    for source in sources:
        header = [
            f"{source.source_id}",
            f"title: {source.title}",
            f"source: {source.source}",
        ]
        if source.doc_id:
            header.append(f"doc_id: {source.doc_id}")
        if source.section:
            header.append(f"section: {source.section}")
        if source.page is not None:
            header.append(f"page: {source.page}")
        if source.doc_type:
            header.append(f"doc_type: {source.doc_type}")
        if source.year is not None:
            header.append(f"year: {source.year}")
        if source.program:
            header.append(f"program: {source.program}")
        if source.url:
            header.append(f"url: {source.url}")
        header.append(f"excerpt: {source.text}")
        blocks.append("\n".join(header))

    return "\n\n".join(blocks)


def _extract_citations(answer: str) -> list[str]:
    seen: list[str] = []
    for citation in re.findall(r"\[(S\d+)\]", answer):
        if citation not in seen:
            seen.append(citation)
    return seen


def _build_warning(sources: list[SourceChunk], citations: list[str], answer: str) -> str:
    if answer == ABSTAIN_RESPONSE:
        return "The assistant abstained because the retrieved evidence was insufficient."
    if len(citations) == 1:
        return "Grounded in 1 retrieved chunk. Verify the citation before treating it as policy."
    return f"Retrieved {len(sources)} chunks and cited {len(citations)} sources."


def _message_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return " ".join(parts)
    return str(content)


def _split_sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]


def _first_nonempty_sentence(text: str) -> str:
    sentences = _split_sentences(text)
    return sentences[0] if sentences else text.strip()


def _is_abstention(answer: str) -> bool:
    return answer.strip().lower().startswith(ABSTAIN_RESPONSE.lower())


def _openrouter_headers(settings: Settings) -> dict[str, str]:
    headers = {"X-Title": settings.openrouter_app_name or settings.project_name}
    if settings.openrouter_http_referer:
        headers["HTTP-Referer"] = settings.openrouter_http_referer
    return headers
