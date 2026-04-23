from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from src.config import Settings, get_settings
from src.predict import RAGAssistant


def load_eval_rows(eval_path: Path) -> list[dict[str, str]]:
    with eval_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [row for row in reader if row.get("question") and row.get("expected_doc_id")]


def run_evaluation(
    assistant: RAGAssistant,
    eval_rows: list[dict[str, str]],
    label: str,
) -> dict[str, object]:
    results: list[dict[str, object]] = []
    category_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"count": 0, "retrieval_hit": 0, "top1_hit": 0, "citation_hit": 0}
    )

    for row in eval_rows:
        question = row["question"].strip()
        expected_doc_id = row["expected_doc_id"].strip()
        response = assistant.ask(question)

        retrieved_doc_ids = [source.doc_id for source in response.sources if source.doc_id]
        top1_doc_id = retrieved_doc_ids[0] if retrieved_doc_ids else None
        cited_doc_ids = _cited_doc_ids(response)

        retrieval_hit = expected_doc_id in retrieved_doc_ids
        top1_hit = expected_doc_id == top1_doc_id
        citation_hit = expected_doc_id in cited_doc_ids

        category = row.get("category", "").strip() or "uncategorized"
        category_counts[category]["count"] += 1
        category_counts[category]["retrieval_hit"] += int(retrieval_hit)
        category_counts[category]["top1_hit"] += int(top1_hit)
        category_counts[category]["citation_hit"] += int(citation_hit)

        results.append(
            {
                "question": question,
                "category": category,
                "difficulty": row.get("difficulty", "").strip(),
                "expected_doc_id": expected_doc_id,
                "expected_section": row.get("expected_section", "").strip(),
                "gold_answer": row.get("gold_answer", "").strip(),
                "retrieval_hit": retrieval_hit,
                "top1_hit": top1_hit,
                "citation_hit": citation_hit,
                "failure_reason": _failure_reason(
                    retrieval_hit=retrieval_hit,
                    top1_hit=top1_hit,
                    citation_hit=citation_hit,
                ),
                "retrieved_doc_ids": retrieved_doc_ids,
                "cited_doc_ids": cited_doc_ids,
                "answer": response.answer,
                "top_sources": [
                    {
                        "doc_id": source.doc_id,
                        "title": source.title,
                        "section": source.section,
                        "score": source.score,
                    }
                    for source in response.sources[:3]
                ],
            }
        )

    total = len(results) or 1
    summary = {
        "label": label,
        "question_count": len(results),
        "retrieval_hit_rate": round(
            sum(int(item["retrieval_hit"]) for item in results) / total,
            4,
        ),
        "top1_hit_rate": round(
            sum(int(item["top1_hit"]) for item in results) / total,
            4,
        ),
        "citation_hit_rate": round(
            sum(int(item["citation_hit"]) for item in results) / total,
            4,
        ),
        "by_category": {
            category: {
                "count": counts["count"],
                "retrieval_hit_rate": round(counts["retrieval_hit"] / counts["count"], 4),
                "top1_hit_rate": round(counts["top1_hit"] / counts["count"], 4),
                "citation_hit_rate": round(counts["citation_hit"] / counts["count"], 4),
            }
            for category, counts in sorted(category_counts.items())
        },
        "failure_examples": [
            item
            for item in results
            if not item["retrieval_hit"] or not item["citation_hit"] or not item["top1_hit"]
        ][:10],
    }

    return {"summary": summary, "results": results}


def run_comparison(
    base_settings: Settings,
    eval_rows: list[dict[str, str]],
    generation_provider: str,
) -> dict[str, object]:
    dense_assistant = _assistant_for_mode(base_settings, "dense", generation_provider)
    hybrid_assistant = _assistant_for_mode(base_settings, "hybrid", generation_provider)

    dense_report = run_evaluation(dense_assistant, eval_rows, label="dense")
    hybrid_report = run_evaluation(hybrid_assistant, eval_rows, label="hybrid")

    dense_by_question = {result["question"]: result for result in dense_report["results"]}
    hybrid_by_question = {result["question"]: result for result in hybrid_report["results"]}

    fixed: list[dict[str, object]] = []
    remaining: list[dict[str, object]] = []
    regressions: list[dict[str, object]] = []

    for question in dense_by_question:
        dense_result = dense_by_question[question]
        hybrid_result = hybrid_by_question[question]

        dense_failed = _is_failure(dense_result)
        hybrid_failed = _is_failure(hybrid_result)

        record = {
            "question": question,
            "expected_doc_id": dense_result["expected_doc_id"],
            "dense_failure_reason": dense_result["failure_reason"],
            "hybrid_failure_reason": hybrid_result["failure_reason"],
            "dense_retrieved_doc_ids": dense_result["retrieved_doc_ids"],
            "hybrid_retrieved_doc_ids": hybrid_result["retrieved_doc_ids"],
            "dense_top_sources": dense_result["top_sources"],
            "hybrid_top_sources": hybrid_result["top_sources"],
        }

        if dense_failed and not hybrid_failed:
            fixed.append(record)
        elif dense_failed and hybrid_failed:
            remaining.append(record)
        elif not dense_failed and hybrid_failed:
            regressions.append(record)

    comparison = {
        "dense": dense_report["summary"],
        "hybrid": hybrid_report["summary"],
        "fixed_questions": fixed,
        "remaining_failures": remaining,
        "regressions": regressions,
    }
    return comparison


def write_report(report: dict[str, object], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_path


def write_comparison_markdown(comparison: dict[str, object], output_path: Path) -> Path:
    dense = comparison["dense"]
    hybrid = comparison["hybrid"]
    lines = [
        "# Retrieval Hardening Report",
        "",
        "## Summary",
        "",
        "| Mode | Retrieval Hit | Top-1 Hit | Citation Hit |",
        "| --- | ---: | ---: | ---: |",
        (
            f"| Dense | {dense['retrieval_hit_rate']:.4f} | {dense['top1_hit_rate']:.4f} "
            f"| {dense['citation_hit_rate']:.4f} |"
        ),
        (
            f"| Hybrid | {hybrid['retrieval_hit_rate']:.4f} | {hybrid['top1_hit_rate']:.4f} "
            f"| {hybrid['citation_hit_rate']:.4f} |"
        ),
        "",
        f"- Fixed failures: {len(comparison['fixed_questions'])}",
        f"- Remaining failures: {len(comparison['remaining_failures'])}",
        f"- Regressions: {len(comparison['regressions'])}",
        "",
    ]
    lines.extend(_markdown_section("Fixed By Hybrid", comparison["fixed_questions"]))
    lines.extend(_markdown_section("Remaining Failures", comparison["remaining_failures"]))
    lines.extend(_markdown_section("Regressions", comparison["regressions"]))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval evaluation.")
    parser.add_argument(
        "--eval-file",
        default="data/eval/unl_cs_policies_eval.csv",
        help="CSV file with evaluation questions and expected sources.",
    )
    parser.add_argument(
        "--output",
        default="reports/eval_results.json",
        help="Path to the JSON report file.",
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=["dense", "hybrid"],
        default=None,
        help="Run a single evaluation using the selected retrieval mode.",
    )
    parser.add_argument(
        "--generation-provider",
        default=None,
        help="Override the generation provider for evaluation, for example 'extractive'.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare dense and hybrid retrieval and write failure-analysis artifacts.",
    )
    parser.add_argument(
        "--comparison-output",
        default="reports/retrieval_comparison.json",
        help="JSON output path for dense vs hybrid comparison.",
    )
    parser.add_argument(
        "--comparison-markdown",
        default="reports/retrieval_failure_analysis.md",
        help="Markdown output path for readable dense vs hybrid failure analysis.",
    )
    args = parser.parse_args()

    settings = get_settings()
    eval_path = Path(args.eval_file)
    eval_rows = load_eval_rows(eval_path)
    generation_provider = args.generation_provider or settings.generation_provider

    if args.compare:
        comparison = run_comparison(settings, eval_rows, generation_provider)
        comparison_output = write_report(comparison, Path(args.comparison_output))
        markdown_output = write_comparison_markdown(comparison, Path(args.comparison_markdown))
        print(json.dumps(comparison["hybrid"], indent=2))
        print(f"Saved comparison JSON to {comparison_output}")
        print(f"Saved comparison Markdown to {markdown_output}")
        return

    retrieval_mode = args.retrieval_mode or settings.retrieval_strategy
    assistant = _assistant_for_mode(settings, retrieval_mode, generation_provider)
    report = run_evaluation(assistant, eval_rows, label=retrieval_mode)
    output_path = write_report(report, Path(args.output))

    print(json.dumps(report["summary"], indent=2))
    print(f"Saved evaluation report to {output_path}")


def _assistant_for_mode(
    settings: Settings,
    retrieval_strategy: str,
    generation_provider: str,
) -> RAGAssistant:
    eval_settings = settings.model_copy(
        update={
            "retrieval_strategy": retrieval_strategy,
            "generation_provider": generation_provider,
        }
    )
    return RAGAssistant(settings=eval_settings)


def _cited_doc_ids(response: object) -> list[str]:
    source_ids = set(getattr(response, "citations", []))
    seen: list[str] = []
    for source in getattr(response, "sources", []):
        if source.source_id in source_ids and source.doc_id and source.doc_id not in seen:
            seen.append(source.doc_id)
    return seen


def _failure_reason(retrieval_hit: bool, top1_hit: bool, citation_hit: bool) -> str:
    if not retrieval_hit:
        return "Expected document was not retrieved."
    if not top1_hit:
        return "Expected document was retrieved but ranked below another source."
    if not citation_hit:
        return "Expected document was retrieved but not cited in the answer."
    return "passed"


def _is_failure(result: dict[str, object]) -> bool:
    return not bool(result["retrieval_hit"] and result["top1_hit"] and result["citation_hit"])


def _markdown_section(title: str, items: list[dict[str, object]]) -> list[str]:
    lines = [f"## {title}", ""]
    if not items:
        lines.append("None.")
        lines.append("")
        return lines

    for item in items[:15]:
        lines.append(f"### {item['question']}")
        lines.append(f"- Expected doc: `{item['expected_doc_id']}`")
        lines.append(f"- Dense: {item['dense_failure_reason']}")
        lines.append(f"- Hybrid: {item['hybrid_failure_reason']}")
        lines.append(f"- Dense docs: `{', '.join(item['dense_retrieved_doc_ids'][:5])}`")
        lines.append(f"- Hybrid docs: `{', '.join(item['hybrid_retrieved_doc_ids'][:5])}`")
        lines.append("")

    return lines


if __name__ == "__main__":
    main()
