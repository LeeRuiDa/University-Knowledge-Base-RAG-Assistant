import { Search } from "lucide-react";

import type { ConversationMessage } from "../types/chat";
import { AssistantMessage } from "./AssistantMessage";

interface MessageListProps {
  messages: ConversationMessage[];
  isLoading: boolean;
  isSlow: boolean;
}

export function MessageList({ messages, isLoading, isSlow }: MessageListProps) {
  if (!messages.length && !isLoading) return null;

  return (
    <section className="conversation" aria-label="Conversation">
      {messages.map((message) =>
        message.role === "user" ? (
          <article className="user-message" key={message.id}>
            <p>{message.content}</p>
            {(message.filters.doc_type || message.filters.year !== null) && (
              <span className="message-filter-note">
                Filtered by {message.filters.doc_type?.replaceAll("_", " ") ?? "all documents"}
                {message.filters.year !== null ? ` · ${message.filters.year}` : ""}
              </span>
            )}
          </article>
        ) : (
          <AssistantMessage message={message} key={message.id} />
        ),
      )}

      {isLoading ? (
        <article className="searching-state" aria-label="Searching official sources">
          <div className="searching-icon" aria-hidden="true">
            <Search size={20} />
          </div>
          <div>
            <strong>Searching official sources…</strong>
            <p>
              {isSlow
                ? "This request is taking longer than usual. Retrieval and generation are still in progress."
                : "Running hybrid retrieval, ranking evidence, and preparing a grounded response."}
            </p>
          </div>
          <div className="loading-lines" aria-hidden="true">
            <span />
            <span />
            <span />
          </div>
        </article>
      ) : null}
    </section>
  );
}
