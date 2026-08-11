import { ChevronDown, ExternalLink, FileText } from "lucide-react";
import { useState } from "react";

import type { SourceChunk } from "../types/api";
import { sourceElementId } from "../utils/sourceIds";

interface SourcePanelProps {
  messageId: string;
  sources: SourceChunk[];
}

function readable(value: string) {
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function validExternalUrl(value: string | null) {
  if (!value) return null;
  try {
    const url = new URL(value);
    return url.protocol === "http:" || url.protocol === "https:" ? url.toString() : null;
  } catch {
    return null;
  }
}

export function SourcePanel({ messageId, sources }: SourcePanelProps) {
  const [isOpen, setIsOpen] = useState(sources.length > 0);

  return (
    <details
      className="evidence-panel"
      open={isOpen}
      onToggle={(event) => setIsOpen(event.currentTarget.open)}
    >
      <summary>
        <span>
          <FileText size={17} aria-hidden="true" />
          Retrieved evidence
        </span>
        <span>
          {sources.length} {sources.length === 1 ? "source" : "sources"}
          <ChevronDown className="summary-chevron" size={16} aria-hidden="true" />
        </span>
      </summary>

      <div className="source-list">
        {sources.length ? (
          sources.map((source) => {
            const officialUrl = validExternalUrl(source.url);
            return (
              <article
                className="source-card"
                id={sourceElementId(messageId, source.source_id)}
                key={`${source.chunk_id}-${source.source_id}`}
                tabIndex={-1}
              >
                <div className="source-card-header">
                  <span className="source-id">{source.source_id}</span>
                  {source.score !== null ? (
                    <span className="source-score">score {source.score.toFixed(4)}</span>
                  ) : null}
                </div>
                <h4>{source.title}</h4>
                <div className="source-metadata">
                  {source.section ? <span>{source.section}</span> : null}
                  {source.page !== null ? <span>Page {source.page}</span> : null}
                  {source.year !== null ? <span>{source.year}</span> : null}
                  {source.doc_type ? <span>{readable(source.doc_type)}</span> : null}
                  {source.program ? <span>{readable(source.program)}</span> : null}
                </div>
                <p className="source-excerpt">{source.text}</p>
                {officialUrl ? (
                  <a
                    className="source-link"
                    href={officialUrl}
                    target="_blank"
                    rel="noreferrer noopener"
                  >
                    Open official source <ExternalLink size={14} aria-hidden="true" />
                  </a>
                ) : null}
              </article>
            );
          })
        ) : (
          <p className="empty-evidence">No source chunks were attached to this response.</p>
        )}
      </div>
    </details>
  );
}
