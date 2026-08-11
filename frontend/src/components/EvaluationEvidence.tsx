import { ArrowUpRight, BarChart3 } from "lucide-react";

const metrics = [
  {
    value: "94.9%",
    label: "Top-1 retrieval hit",
    detail: "59-question retrieval benchmark",
    comparison: "Dense-only baseline: 77.97%",
  },
  {
    value: "98.3%",
    label: "Citation hit",
    detail: "59-question retrieval benchmark",
  },
  {
    value: "100%",
    label: "Correct refusal",
    detail: "18-question hosted answer-quality subset",
  },
  {
    value: "98.2%",
    label: "Mean faithfulness",
    detail: "18-question hosted answer-quality subset",
  },
];

export function EvaluationEvidence() {
  return (
    <section className="evaluation-section" aria-labelledby="evaluation-title">
      <div className="section-heading-row">
        <div>
          <p className="eyebrow">
            <BarChart3 size={15} aria-hidden="true" /> Evaluation evidence
          </p>
          <h2 id="evaluation-title">Measured, not implied</h2>
        </div>
        <a
          className="text-link"
          href="https://github.com/LeeRuiDa/University-Knowledge-Base-RAG-Assistant/tree/main/reports"
          target="_blank"
          rel="noreferrer noopener"
        >
          Review reports <ArrowUpRight size={15} aria-hidden="true" />
        </a>
      </div>

      <div className="metric-grid">
        {metrics.map((metric) => (
          <article className="metric-card" key={metric.label}>
            <strong>{metric.value}</strong>
            <span>{metric.label}</span>
            <small>{metric.detail}</small>
            {metric.comparison ? <small className="metric-comparison">{metric.comparison}</small> : null}
          </article>
        ))}
      </div>
      <p className="evaluation-note">
        These are offline evaluation results, not live-service guarantees. Hybrid dense + BM25
        retrieval improved top-1 accuracy from 77.97% to 94.92% on the project benchmark.
      </p>
    </section>
  );
}
