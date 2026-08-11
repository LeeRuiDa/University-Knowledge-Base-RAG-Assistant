import type {
  AnswerResponse,
  HealthResponse,
  MetadataSummary,
  SearchFilters,
  SourceChunk,
} from "../types/api";

export class ApiResponseError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ApiResponseError";
  }
}

function record(value: unknown, label: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new ApiResponseError(`${label} must be an object.`);
  }
  return value as Record<string, unknown>;
}

function string(value: unknown, label: string): string {
  if (typeof value !== "string") {
    throw new ApiResponseError(`${label} must be a string.`);
  }
  return value;
}

function boolean(value: unknown, label: string): boolean {
  if (typeof value !== "boolean") {
    throw new ApiResponseError(`${label} must be a boolean.`);
  }
  return value;
}

function number(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new ApiResponseError(`${label} must be a finite number.`);
  }
  return value;
}

function nullableString(value: unknown, label: string): string | null {
  return value === null ? null : string(value, label);
}

function nullableNumber(value: unknown, label: string): number | null {
  return value === null ? null : number(value, label);
}

function stringArray(value: unknown, label: string): string[] {
  if (!Array.isArray(value)) {
    throw new ApiResponseError(`${label} must be an array.`);
  }
  return value.map((item, index) => string(item, `${label}[${index}]`));
}

function numberArray(value: unknown, label: string): number[] {
  if (!Array.isArray(value)) {
    throw new ApiResponseError(`${label} must be an array.`);
  }
  return value.map((item, index) => number(item, `${label}[${index}]`));
}

function parseSearchFilters(value: unknown): SearchFilters {
  const data = record(value, "filters_applied");
  return {
    doc_type: nullableString(data.doc_type, "filters_applied.doc_type"),
    year: nullableNumber(data.year, "filters_applied.year"),
  };
}

function parseSourceChunk(value: unknown, index: number): SourceChunk {
  const data = record(value, `sources[${index}]`);
  return {
    source_id: string(data.source_id, `sources[${index}].source_id`),
    chunk_id: string(data.chunk_id, `sources[${index}].chunk_id`),
    doc_id: nullableString(data.doc_id, `sources[${index}].doc_id`),
    score: nullableNumber(data.score, `sources[${index}].score`),
    source: string(data.source, `sources[${index}].source`),
    url: nullableString(data.url, `sources[${index}].url`),
    title: string(data.title, `sources[${index}].title`),
    section: nullableString(data.section, `sources[${index}].section`),
    page: nullableNumber(data.page, `sources[${index}].page`),
    doc_type: nullableString(data.doc_type, `sources[${index}].doc_type`),
    year: nullableNumber(data.year, `sources[${index}].year`),
    program: nullableString(data.program, `sources[${index}].program`),
    text: string(data.text, `sources[${index}].text`),
  };
}

export function parseHealthResponse(value: unknown): HealthResponse {
  const data = record(value, "health response");
  const status = string(data.status, "status");
  if (status !== "ok") {
    throw new ApiResponseError("status must be 'ok'.");
  }
  return {
    status,
    ready: boolean(data.ready, "ready"),
    collection_name: string(data.collection_name, "collection_name"),
    qdrant_mode: string(data.qdrant_mode, "qdrant_mode"),
  };
}

export function parseMetadataSummary(value: unknown): MetadataSummary {
  const data = record(value, "metadata response");
  return {
    ready: boolean(data.ready, "ready"),
    collection_name: string(data.collection_name, "collection_name"),
    qdrant_mode: string(data.qdrant_mode, "qdrant_mode"),
    embedding_provider: string(data.embedding_provider, "embedding_provider"),
    generation_provider: string(data.generation_provider, "generation_provider"),
    retrieval_strategy: string(data.retrieval_strategy, "retrieval_strategy"),
    corpus_manifest_path: nullableString(
      data.corpus_manifest_path,
      "corpus_manifest_path",
    ),
    files_indexed: number(data.files_indexed, "files_indexed"),
    sections_loaded: number(data.sections_loaded, "sections_loaded"),
    chunks_indexed: number(data.chunks_indexed, "chunks_indexed"),
    document_types: stringArray(data.document_types, "document_types"),
    years: numberArray(data.years, "years"),
    programs: stringArray(data.programs, "programs"),
    sources: stringArray(data.sources, "sources"),
  };
}

export function parseAnswerResponse(value: unknown): AnswerResponse {
  const data = record(value, "answer response");
  if (!Array.isArray(data.sources)) {
    throw new ApiResponseError("sources must be an array.");
  }
  return {
    question: string(data.question, "question"),
    answer: string(data.answer, "answer"),
    citations: stringArray(data.citations, "citations"),
    grounded: boolean(data.grounded, "grounded"),
    warning: nullableString(data.warning, "warning"),
    filters_applied:
      data.filters_applied === null ? null : parseSearchFilters(data.filters_applied),
    sources: data.sources.map(parseSourceChunk),
  };
}
