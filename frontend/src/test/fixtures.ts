import type { AnswerResponse, HealthResponse, MetadataSummary } from "../types/api";

export const HEALTH: HealthResponse = {
  status: "ok",
  ready: true,
  collection_name: "university_knowledge_base",
  qdrant_mode: "server",
};

export const METADATA: MetadataSummary = {
  ready: true,
  collection_name: "university_knowledge_base",
  qdrant_mode: "server",
  embedding_provider: "openrouter",
  generation_provider: "openrouter",
  retrieval_strategy: "hybrid",
  corpus_manifest_path: "data/corpus_manifest.csv",
  files_indexed: 24,
  sections_loaded: 285,
  chunks_indexed: 383,
  document_types: ["internship_policy", "academic_calendar", "billing_policy"],
  years: [2025, 2026],
  programs: ["computer_science_undergraduate"],
  sources: ["internal/path/that-must-not-render.html"],
};

export const GROUNDED_ANSWER: AnswerResponse = {
  question: "How many credits of CSCE 495 count as one tech elective course?",
  answer: "Three credits of CSCE 495 count as one technical elective [S1].",
  citations: ["S1"],
  grounded: true,
  warning: "Grounded in 1 retrieved chunk. Verify the citation before treating it as policy.",
  filters_applied: { doc_type: null, year: null },
  sources: [
    {
      source_id: "S1",
      chunk_id: "internship-credit-1",
      doc_id: "unl_cs_internship_credit",
      score: 0.8933,
      source: "data/raw/unl/internship-credit.html",
      url: "https://computing.unl.edu/internship-credit/",
      title: "UNL Internship Credit for Computing Students",
      section: "Internship for Credit",
      page: null,
      doc_type: "internship_policy",
      year: 2025,
      program: "computer_science_undergraduate",
      text: "Three credits of CSCE 495 count as one technical elective course.",
    },
  ],
};

export const REFUSAL_ANSWER: AnswerResponse = {
  question: "What is the deadline to appeal a parking ticket at UNL?",
  answer: "I don't know from the provided documents.",
  citations: [],
  grounded: false,
  warning: "The assistant abstained because the retrieved evidence was insufficient.",
  filters_applied: { doc_type: null, year: null },
  sources: [],
};
