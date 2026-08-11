import { Check, CheckCircle2, Copy, ShieldAlert } from "lucide-react";
import { useState } from "react";
import ReactMarkdown from "react-markdown";
import rehypeSanitize from "rehype-sanitize";

import type { AssistantMessage as AssistantMessageType } from "../types/chat";
import { preferredScrollBehavior } from "../utils/motion";
import { sourceElementId } from "../utils/sourceIds";
import { SourcePanel } from "./SourcePanel";

interface AssistantMessageProps {
  message: AssistantMessageType;
}

function linkCitations(
  content: string,
  citations: string[],
  sourceIds: string[],
  messageId: string,
) {
  const availableSources = new Set(sourceIds);
  const validCitations = new Set(citations.filter((citation) => availableSources.has(citation)));
  return content.replace(/\[(S\d+)\]/g, (token, sourceId: string) => {
    if (!validCitations.has(sourceId)) return token;
    return `[${sourceId}](#${sourceElementId(messageId, sourceId)})`;
  });
}

export function AssistantMessage({ message }: AssistantMessageProps) {
  const [copied, setCopied] = useState(false);
  const { response } = message;
  const markdown = linkCitations(
    response.answer,
    response.citations,
    response.sources.map((source) => source.source_id),
    message.id,
  );

  async function copyAnswer() {
    try {
      await navigator.clipboard.writeText(response.answer);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      setCopied(false);
    }
  }

  return (
    <article
      className={`assistant-message ${response.grounded ? "assistant-message--grounded" : "assistant-message--refusal"}`}
      aria-label={response.grounded ? "Grounded assistant answer" : "Assistant refusal"}
    >
      <div className="assistant-message-header">
        <span className="answer-status">
          {response.grounded ? (
            <CheckCircle2 size={16} aria-hidden="true" />
          ) : (
            <ShieldAlert size={16} aria-hidden="true" />
          )}
          {response.grounded ? "Grounded answer" : "Boundary held"}
        </span>
        <button className="copy-button" type="button" onClick={copyAnswer}>
          {copied ? <Check size={15} aria-hidden="true" /> : <Copy size={15} aria-hidden="true" />}
          {copied ? "Copied" : "Copy"}
        </button>
      </div>

      <div className="answer-evidence-grid">
        <div className="answer-column">
          {!response.grounded ? (
            <div className="refusal-intro">
              <strong>Not enough evidence in the indexed corpus</strong>
              <p>
                The assistant deliberately avoided filling the gap with an unsupported policy.
                Try a question about graduation, internships, registration, fees, or student support.
              </p>
            </div>
          ) : null}

          <div className="markdown-answer">
            <ReactMarkdown
              rehypePlugins={[rehypeSanitize]}
              components={{
                a: ({ href, children, title }) => {
                  if (href?.startsWith("#source-")) {
                    return (
                      <a
                        href={href}
                        title={title}
                        className="citation-link"
                        onClick={(event) => {
                          event.preventDefault();
                          const target = document.getElementById(href.slice(1));
                          target?.scrollIntoView({
                            behavior: preferredScrollBehavior(),
                            block: "center",
                          });
                          target?.focus({ preventScroll: true });
                        }}
                      >
                        {children}
                      </a>
                    );
                  }
                  return (
                    <a
                      href={href}
                      title={title}
                      target="_blank"
                      rel="noreferrer noopener"
                    >
                      {children}
                    </a>
                  );
                },
              }}
            >
              {markdown}
            </ReactMarkdown>
          </div>

          {response.warning ? (
            <p className="answer-warning">
              <span aria-hidden="true" /> {response.warning}
            </p>
          ) : null}
        </div>

        <SourcePanel messageId={message.id} sources={response.sources} />
      </div>
    </article>
  );
}
