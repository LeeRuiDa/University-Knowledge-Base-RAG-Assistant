export interface SearchFilters {
  doc_type: string | null;
  year: number | null;
}

export interface AskRequest {
  question: string;
  filters: SearchFilters | null;
}

export interface SourceChunk {
  source_id: string;
  chunk_id: string;
  doc_id: string | null;
  score: number | null;
  source: string;
  url: string | null;
  title: string;
  section: string | null;
  page: number | null;
  doc_type: string | null;
  year: number | null;
  program: string | null;
  text: string;
}

export interface AnswerResponse {
  question: string;
  answer: string;
  citations: string[];
  grounded: boolean;
  warning: string | null;
  filters_applied: SearchFilters | null;
  sources: SourceChunk[];
}

export interface MetadataSummary {
  ready: boolean;
  collection_name: string;
  qdrant_mode: string;
  embedding_provider: string;
  generation_provider: string;
  retrieval_strategy: string;
  corpus_manifest_path: string | null;
  files_indexed: number;
  sections_loaded: number;
  chunks_indexed: number;
  document_types: string[];
  years: number[];
  programs: string[];
  sources: string[];
}

export interface HealthResponse {
  status: "ok";
  ready: boolean;
  collection_name: string;
  qdrant_mode: string;
}
