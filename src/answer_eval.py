from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, Field
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness

from src.answer import ABSTAIN_RESPONSE
from src.config import Settings, get_settings
from src.predict import RAGAssistant

REFUSAL_PATTERNS = (
    "i don't know from the provided documents",
    "not found in the provided documents",
    "not in the provided documents",
    "not enough information",
    "insufficient information",
    "unable to determine",
)
MAX_EVAL_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 3.0
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "if",
    "in",
    "is",
    "it",
    "many",
    "may",
    "of",
    "on",
    "or",
    "the",
    "their",
    "there",
    "they",
    "this",
    "to",
    "what",
    "when",
    "which",
    "who",
    "with",
}
SUPPORTED_FAITHFULNESS_THRESHOLD = 0.8
SUPPORTED_COMPLETENESS_THRESHOLD = 2
SUPPORTED_CITATION_THRESHOLD = 2
REFUSAL_THRESHOLD = 3


@dataclass(frozen=True)
class AnswerEvalRow:
    question: str
    expected_doc_ids: list[str]
    gold_answer: str
    category: str
    should_refuse: bool
    difficulty: str


class CompletenessJudgment(BaseModel):
    score: int = Field(ge=0, le=3)
    rationale: str


def load_answer_eval_rows(eval_path: Path) -> list[AnswerEvalRow]:
    with eval_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[AnswerEvalRow] = []
        for row in reader:
            question = (row.get("question") or "").strip()
            if not question:
                continue
            rows.append(
                AnswerEvalRow(
                    question=question,
                    expected_doc_ids=_split_doc_ids(row.get("expected_doc_ids") or ""),
                    gold_answer=(row.get("gold_answer") or "").strip(),
                    category=(row.get("category") or "uncategorized").strip(),
                    should_refuse=_parse_bool(row.get("should_refuse") or "false"),
                    difficulty=(row.get("difficulty") or "").strip(),
                )
            )
        return rows


def run_answer_evaluation(
    settings: Settings,
    eval_rows: list[AnswerEvalRow],
    generation_provider: str,
    judge_provider: str,
) -> dict[str, object]:
    _validate_hosted_provider(generation_provider, "generation_provider")
    _validate_hosted_provider(judge_provider, "judge_provider")

    assistant = _assistant_for_mode(settings, generation_provider)
    completeness_judge = CompletenessJudge(settings, judge_provider)
    results: list[dict[str, object]] = []
    faithfulness_inputs: list[tuple[int, dict[str, object]]] = []

    for row in eval_rows:
        response = _ask_with_retry(assistant, row.question)
        answer_text = response.answer.strip()
        clean_answer = _strip_citations(answer_text)
        retrieved_doc_ids = [source.doc_id for source in response.sources if source.doc_id]
        cited_doc_ids = [
            source.doc_id
            for source in response.sources
            if source.source_id in response.citations and source.doc_id
        ]
        response_refused = _is_refusal_like(answer_text)

        completeness_score: int | None = None
        completeness_rationale: str | None = None
        if not row.should_refuse:
            judgment = completeness_judge.score(
                question=row.question,
                gold_answer=row.gold_answer,
                answer=clean_answer,
            )
            completeness_score = judgment.score
            completeness_rationale = judgment.rationale

        citation_score, citation_note = score_citation_usefulness(
            row=row,
            answer_text=clean_answer,
            response_citations=response.citations,
            sources=[source.model_dump() for source in response.sources],
        )
        refusal_score, refusal_note = score_refusal_behavior(
            should_refuse=row.should_refuse,
            answer_text=answer_text,
            citations=response.citations,
        )
        expected_doc_hit = bool(
            row.should_refuse or any(doc_id in row.expected_doc_ids for doc_id in retrieved_doc_ids)
        )
        expected_doc_cited = bool(
            row.should_refuse or any(doc_id in row.expected_doc_ids for doc_id in cited_doc_ids)
        )

        result = {
            "question": row.question,
            "category": row.category,
            "difficulty": row.difficulty,
            "should_refuse": row.should_refuse,
            "expected_doc_ids": row.expected_doc_ids,
            "gold_answer": row.gold_answer,
            "answer": answer_text,
            "clean_answer": clean_answer,
            "response_refused": response_refused,
            "retrieved_doc_ids": retrieved_doc_ids,
            "cited_doc_ids": cited_doc_ids,
            "citations": response.citations,
            "expected_doc_hit": expected_doc_hit,
            "expected_doc_cited": expected_doc_cited,
            "faithfulness": None,
            "completeness_score": completeness_score,
            "completeness_rationale": completeness_rationale,
            "citation_usefulness_score": citation_score,
            "citation_usefulness_note": citation_note,
            "refusal_behavior_score": refusal_score,
            "refusal_behavior_note": refusal_note,
            "warning": response.warning,
            "sources": [source.model_dump() for source in response.sources],
        }
        results.append(result)

        if not row.should_refuse:
            faithfulness_inputs.append(
                (
                    len(results) - 1,
                    {
                        "user_input": row.question,
                        "response": clean_answer,
                        "retrieved_contexts": [source.text for source in response.sources],
                    },
                )
            )

    faithfulness_samples = [item[1] for item in faithfulness_inputs]
    faithfulness_scores = score_faithfulness(
        settings,
        judge_provider,
        faithfulness_samples,
    )
    for (result_index, _), score in zip(faithfulness_inputs, faithfulness_scores, strict=True):
        results[result_index]["faithfulness"] = score

    summary = summarize_results(results)
    return {
        "summary": summary,
        "config": build_config_snapshot(settings, generation_provider, judge_provider),
        "results": results,
    }


def score_faithfulness(
    settings: Settings,
    judge_provider: str,
    samples: list[dict[str, object]],
) -> list[float]:
    if not samples:
        return []

    metric = Faithfulness(llm=_build_ragas_judge_llm(settings, judge_provider))
    scores: list[float] = []
    for sample in samples:
        result = _score_faithfulness_with_retry(metric, sample)
        scores.append(round(float(result.value), 4))
    return scores


class CompletenessJudge:
    def __init__(self, settings: Settings, provider: str) -> None:
        self._model = _provider_model(settings, provider)
        self._client = _build_sync_client(settings, provider)

    def score(
        self,
        question: str,
        gold_answer: str,
        answer: str,
    ) -> CompletenessJudgment:
        system_prompt = (
            "You grade answer completeness for a university policy assistant. "
            "Return JSON only with keys score and rationale. "
            "Use this rubric exactly: "
            "0 = misses the key requirement or is materially wrong; "
            "1 = partially complete but missing an important condition or detail; "
            "2 = complete enough for a student to act on, even if terse; "
            "3 = complete and concise, covering the key constraints without material errors. "
            "If the answer refuses when a factual answer was expected, score 0."
        )
        user_prompt = (
            f"Question:\n{question}\n\n"
            f"Reference answer:\n{gold_answer}\n\n"
            f"Assistant answer:\n{answer}\n"
        )

        response = None
        last_error: Exception | None = None
        for attempt in range(MAX_EVAL_ATTEMPTS):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    temperature=0,
                    max_tokens=220,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                break
            except Exception as exc:  # pragma: no cover - network retry path
                last_error = exc
                if attempt == MAX_EVAL_ATTEMPTS - 1:
                    raise
                time.sleep(RETRY_DELAY_SECONDS)
        if response is None:  # pragma: no cover - defensive
            raise RuntimeError("Completeness judge did not return a response.") from last_error
        content = response.choices[0].message.content or "{}"
        payload = _safe_json_loads(content)
        return CompletenessJudgment(
            score=max(0, min(3, int(payload.get("score", 0)))),
            rationale=str(payload.get("rationale", "")).strip() or "No rationale provided.",
        )


def score_citation_usefulness(
    row: AnswerEvalRow,
    answer_text: str,
    response_citations: list[str],
    sources: list[dict[str, object]],
) -> tuple[int, str]:
    cited_sources = [source for source in sources if source.get("source_id") in response_citations]
    if row.should_refuse:
        if _is_refusal_like(answer_text) and not response_citations:
            return 3, "The assistant refused without attaching unsupported citations."
        if _is_refusal_like(answer_text):
            return 2, "The assistant refused but still attached citations."
        return (
            0,
            "The assistant did not refuse and its citations are not useful for a refusal case.",
        )

    if not response_citations:
        return 0, "The answer contains no citations."
    if len(cited_sources) != len({citation for citation in response_citations}):
        return 0, "One or more cited source ids did not map to retrieved chunks."

    expected_doc_cited = any(
        source.get("doc_id") in row.expected_doc_ids for source in cited_sources
    )
    cited_text = "\n".join(str(source.get("text", "")) for source in cited_sources).lower()
    gold_terms = _key_terms(row.gold_answer)
    if not gold_terms:
        return (
            1 if expected_doc_cited else 0,
            "No key terms were available for lexical support checks.",
        )

    overlap = len(gold_terms & _key_terms(cited_text)) / len(gold_terms)
    number_terms = _number_terms(row.gold_answer)
    numbers_supported = not number_terms or all(number in cited_text for number in number_terms)

    if expected_doc_cited and overlap >= 0.5 and numbers_supported:
        return 3, "Citations point to expected documents and directly support the reference answer."
    if expected_doc_cited and overlap >= 0.25:
        return 2, "Citations are relevant, but the supporting overlap is only partial."
    if expected_doc_cited or overlap >= 0.25:
        return 1, "Citations are plausible but not specific enough."
    return 0, "Citations do not clearly support the reference answer."


def score_refusal_behavior(
    should_refuse: bool,
    answer_text: str,
    citations: list[str],
) -> tuple[int | None, str | None]:
    if not should_refuse:
        return None, None

    normalized = answer_text.strip().lower()
    if normalized == ABSTAIN_RESPONSE.lower() and not citations:
        return 3, "The assistant clearly refused using the required abstention message."
    if _is_refusal_like(answer_text) and not citations:
        return 3, "The assistant refused or hedged appropriately without inventing support."
    if _is_refusal_like(answer_text):
        return 2, "The assistant acknowledged uncertainty but still attached citations."
    if "?" in answer_text or "may" in normalized or "might" in normalized:
        return 1, "The answer hedged slightly but still attempted to answer."
    return 0, "The assistant answered a question that should have been refused."


def summarize_results(results: list[dict[str, object]]) -> dict[str, object]:
    supported = [result for result in results if not result["should_refuse"]]
    refusal = [result for result in results if result["should_refuse"]]

    supported_failures = [
        result
        for result in supported
        if (
            (
                result["faithfulness"] is not None
                and result["faithfulness"] < SUPPORTED_FAITHFULNESS_THRESHOLD
            )
            or (result["completeness_score"] or 0) < SUPPORTED_COMPLETENESS_THRESHOLD
            or result["citation_usefulness_score"] < SUPPORTED_CITATION_THRESHOLD
        )
    ]
    refusal_failures = [
        result for result in refusal if (result["refusal_behavior_score"] or 0) < REFUSAL_THRESHOLD
    ]
    failure_examples = (supported_failures + refusal_failures)[:10]

    failure_modes = Counter()
    for result in supported_failures:
        if (
            result["faithfulness"] is not None
            and result["faithfulness"] < SUPPORTED_FAITHFULNESS_THRESHOLD
        ):
            failure_modes["low_faithfulness"] += 1
        if (result["completeness_score"] or 0) < SUPPORTED_COMPLETENESS_THRESHOLD:
            failure_modes["incomplete_answer"] += 1
        if result["citation_usefulness_score"] < SUPPORTED_CITATION_THRESHOLD:
            failure_modes["weak_citations"] += 1
    for _result in refusal_failures:
        failure_modes["refusal_failure"] += 1

    by_category: dict[str, dict[str, object]] = {}
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for result in results:
        grouped[str(result["category"])].append(result)

    for category, items in sorted(grouped.items()):
        supported_items = [item for item in items if not item["should_refuse"]]
        refusal_items = [item for item in items if item["should_refuse"]]
        by_category[category] = {
            "count": len(items),
            "faithfulness_mean": _rounded_mean(item["faithfulness"] for item in supported_items),
            "completeness_mean": _rounded_mean(
                item["completeness_score"] for item in supported_items
            ),
            "citation_usefulness_mean": _rounded_mean(
                item["citation_usefulness_score"] for item in items
            ),
            "refusal_behavior_mean": _rounded_mean(
                item["refusal_behavior_score"] for item in refusal_items
            ),
        }

    return {
        "question_count": len(results),
        "supported_question_count": len(supported),
        "refusal_question_count": len(refusal),
        "faithfulness_mean": _rounded_mean(item["faithfulness"] for item in supported),
        "completeness_mean": _rounded_mean(item["completeness_score"] for item in supported),
        "citation_usefulness_mean": _rounded_mean(
            item["citation_usefulness_score"] for item in results
        ),
        "refusal_behavior_mean": _rounded_mean(
            item["refusal_behavior_score"] for item in refusal
        ),
        "supported_pass_rate": round(
            sum(
                int(
                    (item["faithfulness"] or 0) >= SUPPORTED_FAITHFULNESS_THRESHOLD
                    and (item["completeness_score"] or 0) >= SUPPORTED_COMPLETENESS_THRESHOLD
                    and item["citation_usefulness_score"] >= SUPPORTED_CITATION_THRESHOLD
                )
                for item in supported
            )
            / max(1, len(supported)),
            4,
        ),
        "correct_refusal_rate": round(
            sum(
                int((item["refusal_behavior_score"] or 0) >= REFUSAL_THRESHOLD)
                for item in refusal
            )
            / max(1, len(refusal)),
            4,
        ),
        "failure_mode_counts": dict(failure_modes),
        "failure_examples": [
            {
                "question": result["question"],
                "category": result["category"],
                "should_refuse": result["should_refuse"],
                "faithfulness": result["faithfulness"],
                "completeness_score": result["completeness_score"],
                "citation_usefulness_score": result["citation_usefulness_score"],
                "refusal_behavior_score": result["refusal_behavior_score"],
                "answer": result["answer"],
            }
            for result in failure_examples
        ],
        "by_category": by_category,
    }


def build_config_snapshot(
    settings: Settings,
    generation_provider: str,
    judge_provider: str,
) -> dict[str, object]:
    return {
        "collection_name": settings.qdrant_collection_name,
        "retrieval_strategy": settings.retrieval_strategy,
        "retrieval_k": settings.retrieval_k,
        "dense_retrieval_k": settings.dense_retrieval_k,
        "sparse_retrieval_k": settings.sparse_retrieval_k,
        "hybrid_candidate_k": settings.hybrid_candidate_k,
        "max_chunks_per_doc": settings.max_chunks_per_doc,
        "generation_provider": generation_provider,
        "generation_model": _provider_model(settings, generation_provider),
        "judge_provider": judge_provider,
        "judge_model": _provider_model(settings, judge_provider),
    }


def write_report(report: dict[str, object], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_path


def write_failure_analysis(report: dict[str, object], output_path: Path) -> Path:
    summary = report["summary"]
    lines = [
        "# Answer Quality Evaluation",
        "",
        "## Summary",
        "",
        f"- Questions: {summary['question_count']}",
        f"- Supported questions: {summary['supported_question_count']}",
        f"- Refusal questions: {summary['refusal_question_count']}",
        f"- Faithfulness mean: {summary['faithfulness_mean']}",
        f"- Completeness mean: {summary['completeness_mean']}",
        f"- Citation usefulness mean: {summary['citation_usefulness_mean']}",
        f"- Refusal behavior mean: {summary['refusal_behavior_mean']}",
        f"- Supported pass rate: {summary['supported_pass_rate']}",
        f"- Correct refusal rate: {summary['correct_refusal_rate']}",
        "",
        "## Failure Modes",
        "",
    ]
    failure_modes: dict[str, int] = summary["failure_mode_counts"]
    if failure_modes:
        for mode, count in sorted(failure_modes.items()):
            lines.append(f"- {mode}: {count}")
    else:
        lines.append("No failures.")
    lines.append("")
    lines.append("## Failed Examples")
    lines.append("")

    failed = [
        result
        for result in report["results"]
        if (
            (
                not result["should_refuse"]
                and (
                    (result["faithfulness"] or 0) < SUPPORTED_FAITHFULNESS_THRESHOLD
                    or (result["completeness_score"] or 0) < SUPPORTED_COMPLETENESS_THRESHOLD
                    or result["citation_usefulness_score"] < SUPPORTED_CITATION_THRESHOLD
                )
            )
            or (
                result["should_refuse"]
                and (result["refusal_behavior_score"] or 0) < REFUSAL_THRESHOLD
            )
        )
    ]
    if not failed:
        lines.append("None.")
        lines.append("")
    else:
        for result in failed[:10]:
            lines.append(f"### {result['question']}")
            lines.append(f"- Category: `{result['category']}`")
            if result["expected_doc_ids"]:
                lines.append(f"- Expected docs: `{', '.join(result['expected_doc_ids'])}`")
            lines.append(f"- Should refuse: `{result['should_refuse']}`")
            lines.append(f"- Faithfulness: `{result['faithfulness']}`")
            lines.append(f"- Completeness: `{result['completeness_score']}`")
            lines.append(f"- Citation usefulness: `{result['citation_usefulness_score']}`")
            lines.append(f"- Refusal behavior: `{result['refusal_behavior_score']}`")
            lines.append(f"- Answer: {result['answer']}")
            if result["completeness_rationale"]:
                lines.append(f"- Completeness rationale: {result['completeness_rationale']}")
            if result["citation_usefulness_note"]:
                lines.append(f"- Citation note: {result['citation_usefulness_note']}")
            if result["refusal_behavior_note"]:
                lines.append(f"- Refusal note: {result['refusal_behavior_note']}")
            top_docs = [
                source.get("doc_id")
                for source in result["sources"][:3]
                if source.get("doc_id")
            ]
            if top_docs:
                lines.append(f"- Top retrieved docs: `{', '.join(top_docs)}`")
            lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run answer-quality evaluation.")
    parser.add_argument(
        "--eval-file",
        default="data/eval/unl_cs_answer_eval_subset.csv",
        help="CSV file with the curated answer-evaluation subset.",
    )
    parser.add_argument(
        "--output",
        default="reports/answer_eval_results.json",
        help="JSON output path.",
    )
    parser.add_argument(
        "--markdown-output",
        default="reports/answer_failure_analysis.md",
        help="Markdown output path.",
    )
    parser.add_argument(
        "--generation-provider",
        default=None,
        help="Hosted provider used to generate answers, for example 'openrouter'.",
    )
    parser.add_argument(
        "--judge-provider",
        default=None,
        help="Hosted provider used for faithfulness and completeness judging.",
    )
    args = parser.parse_args()

    settings = get_settings()
    eval_rows = load_answer_eval_rows(Path(args.eval_file))
    generation_provider = args.generation_provider or settings.generation_provider
    judge_provider = args.judge_provider or generation_provider

    report = run_answer_evaluation(settings, eval_rows, generation_provider, judge_provider)
    output_path = write_report(report, Path(args.output))
    markdown_path = write_failure_analysis(report, Path(args.markdown_output))

    print(json.dumps(report["summary"], indent=2))
    print(f"Saved answer evaluation JSON to {output_path}")
    print(f"Saved answer evaluation Markdown to {markdown_path}")


def _assistant_for_mode(settings: Settings, generation_provider: str) -> RAGAssistant:
    eval_settings = settings.model_copy(update={"generation_provider": generation_provider})
    return RAGAssistant(settings=eval_settings)


def _build_ragas_judge_llm(settings: Settings, provider: str):
    client = _build_async_client(settings, provider)
    return llm_factory(_provider_model(settings, provider), client=client)


def _build_sync_client(settings: Settings, provider: str) -> OpenAI:
    kwargs = _client_kwargs(settings, provider)
    return OpenAI(**kwargs)


def _build_async_client(settings: Settings, provider: str) -> AsyncOpenAI:
    kwargs = _client_kwargs(settings, provider)
    return AsyncOpenAI(**kwargs)


def _client_kwargs(settings: Settings, provider: str) -> dict[str, object]:
    provider = provider.lower()
    if provider == "openrouter":
        headers = {"X-Title": settings.openrouter_app_name or settings.project_name}
        if settings.openrouter_http_referer:
            headers["HTTP-Referer"] = settings.openrouter_http_referer
        return {
            "api_key": settings.openrouter_api_key or settings.openai_api_key,
            "base_url": settings.openrouter_base_url,
            "default_headers": headers,
            "timeout": 120.0,
            "max_retries": 3,
        }
    if provider == "openai":
        kwargs: dict[str, object] = {
            "api_key": settings.openai_api_key,
            "timeout": 120.0,
            "max_retries": 3,
        }
        if settings.openai_api_base:
            kwargs["base_url"] = settings.openai_api_base
        return kwargs
    raise ValueError(f"Unsupported hosted provider: {provider}")


def _provider_model(settings: Settings, provider: str) -> str:
    provider = provider.lower()
    if provider == "openrouter":
        return settings.openrouter_chat_model
    if provider == "openai":
        return settings.openai_chat_model
    raise ValueError(f"Unsupported hosted provider: {provider}")


def _validate_hosted_provider(provider: str, field_name: str) -> None:
    if provider.lower() not in {"openai", "openrouter"}:
        raise ValueError(
            f"{field_name} must be 'openai' or 'openrouter' for hosted answer evaluation."
        )


def _split_doc_ids(raw_value: str) -> list[str]:
    return [part.strip() for part in raw_value.split("|") if part.strip()]


def _parse_bool(raw_value: str) -> bool:
    return raw_value.strip().lower() in {"1", "true", "yes", "y"}


def _strip_citations(text: str) -> str:
    return re.sub(r"\s*\[S\d+\]", "", text).strip()


def _is_refusal_like(answer_text: str) -> bool:
    normalized = answer_text.strip().lower()
    if normalized == ABSTAIN_RESPONSE.lower():
        return True
    return any(pattern in normalized for pattern in REFUSAL_PATTERNS)


def _key_terms(text: str) -> set[str]:
    terms: set[str] = set()
    for token in re.findall(r"\b[a-zA-Z0-9$./-]+\b", text.lower()):
        if token in STOPWORDS:
            continue
        if len(token) >= 4 or any(character.isdigit() for character in token):
            terms.add(token)
    return terms


def _number_terms(text: str) -> set[str]:
    return set(re.findall(r"(?:\$\d+(?:\.\d+)?)|\b\d+(?:\.\d+)?\b", text.lower()))


def _rounded_mean(values) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    if not cleaned:
        return None
    return round(mean(cleaned), 4)


def _safe_json_loads(content: str) -> dict[str, object]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}


def _ask_with_retry(assistant: RAGAssistant, question: str):
    last_error: Exception | None = None
    for attempt in range(MAX_EVAL_ATTEMPTS):
        try:
            return assistant.ask(question)
        except Exception as exc:  # pragma: no cover - network retry path
            last_error = exc
            if attempt == MAX_EVAL_ATTEMPTS - 1:
                raise
            time.sleep(RETRY_DELAY_SECONDS)
    raise RuntimeError("Answer generation failed after retries.") from last_error


def _score_faithfulness_with_retry(metric: Faithfulness, sample: dict[str, object]):
    last_error: Exception | None = None
    for attempt in range(MAX_EVAL_ATTEMPTS):
        try:
            return metric.score(**sample)
        except Exception as exc:  # pragma: no cover - network retry path
            last_error = exc
            if attempt == MAX_EVAL_ATTEMPTS - 1:
                raise
            time.sleep(RETRY_DELAY_SECONDS)
    raise RuntimeError("Faithfulness scoring failed after retries.") from last_error


if __name__ == "__main__":
    main()
