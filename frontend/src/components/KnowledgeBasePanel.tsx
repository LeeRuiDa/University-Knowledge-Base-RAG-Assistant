import { BookOpen, ChevronDown, Database, Filter, Layers3 } from "lucide-react";

import type { MetadataSummary, SearchFilters } from "../types/api";

interface KnowledgeBasePanelProps {
  metadata: MetadataSummary | null;
  isLoading: boolean;
  filters: SearchFilters;
  onFiltersChange: (filters: SearchFilters) => void;
  onRetry: () => void;
}

function readable(value: string) {
  return value
    .replaceAll("_", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

export function KnowledgeBasePanel({
  metadata,
  isLoading,
  filters,
  onFiltersChange,
  onRetry,
}: KnowledgeBasePanelProps) {
  const activeFilterCount = Number(Boolean(filters.doc_type)) + Number(filters.year !== null);

  return (
    <details className="knowledge-panel">
      <summary>
        <span className="knowledge-summary-title">
          <Database size={18} aria-hidden="true" />
          Knowledge base
        </span>
        <span className="knowledge-summary-meta">
          {metadata ? `${metadata.files_indexed} documents · ${metadata.chunks_indexed} chunks` : "Corpus details"}
          {activeFilterCount ? <span className="filter-count">{activeFilterCount}</span> : null}
          <ChevronDown className="summary-chevron" size={17} aria-hidden="true" />
        </span>
      </summary>

      <div className="knowledge-content">
        {isLoading ? (
          <div className="panel-loading" aria-label="Loading knowledge base metadata">
            <span />
            <span />
            <span />
          </div>
        ) : metadata ? (
          <>
            <div className="filter-grid">
              <label>
                <span>
                  <Filter size={14} aria-hidden="true" /> Document type
                </span>
                <select
                  value={filters.doc_type ?? ""}
                  onChange={(event) =>
                    onFiltersChange({
                      ...filters,
                      doc_type: event.target.value || null,
                    })
                  }
                >
                  <option value="">All document types</option>
                  {metadata.document_types.map((documentType) => (
                    <option key={documentType} value={documentType}>
                      {readable(documentType)}
                    </option>
                  ))}
                </select>
              </label>

              <label>
                <span>
                  <Filter size={14} aria-hidden="true" /> Year
                </span>
                <select
                  value={filters.year ?? ""}
                  onChange={(event) =>
                    onFiltersChange({
                      ...filters,
                      year: event.target.value ? Number(event.target.value) : null,
                    })
                  }
                >
                  <option value="">All years</option>
                  {metadata.years.map((year) => (
                    <option key={year} value={year}>
                      {year}
                    </option>
                  ))}
                </select>
              </label>
            </div>

            <dl className="metadata-grid">
              <div>
                <dt><BookOpen size={15} aria-hidden="true" /> Readiness</dt>
                <dd>{metadata.ready ? "Indexed and ready" : "Not indexed"}</dd>
              </div>
              <div>
                <dt><Layers3 size={15} aria-hidden="true" /> Retrieval</dt>
                <dd>{readable(metadata.retrieval_strategy)}</dd>
              </div>
              <div>
                <dt>Embedding provider</dt>
                <dd>{readable(metadata.embedding_provider)}</dd>
              </div>
              <div>
                <dt>Generation provider</dt>
                <dd>{readable(metadata.generation_provider)}</dd>
              </div>
              <div>
                <dt>Indexed documents</dt>
                <dd>{metadata.files_indexed}</dd>
              </div>
              <div>
                <dt>Indexed chunks</dt>
                <dd>{metadata.chunks_indexed}</dd>
              </div>
            </dl>

            {metadata.programs.length ? (
              <div className="program-list">
                <span>Programs included</span>
                <div>
                  {metadata.programs.map((program) => (
                    <span className="metadata-chip" key={program}>
                      {readable(program)}
                    </span>
                  ))}
                </div>
              </div>
            ) : null}
          </>
        ) : (
          <div className="knowledge-unavailable">
            <p>Knowledge base metadata is unavailable.</p>
            <button type="button" className="secondary-button" onClick={onRetry}>
              Retry connection
            </button>
          </div>
        )}
      </div>
    </details>
  );
}
