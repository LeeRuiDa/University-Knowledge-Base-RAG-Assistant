import { useEffect, useState } from "react";

import { parseAnswerResponse } from "../api/parsers";
import type { ConversationMessage } from "../types/chat";

const CONVERSATION_STORAGE_KEY = "university-policy-assistant-conversation";

function loadConversation(): ConversationMessage[] {
  try {
    const saved = window.localStorage.getItem(CONVERSATION_STORAGE_KEY);
    if (!saved) {
      return [];
    }
    const parsed: unknown = JSON.parse(saved);
    if (!Array.isArray(parsed)) {
      return [];
    }
    return parsed.filter(isConversationMessage);
  } catch {
    return [];
  }
}

function isConversationMessage(value: unknown): value is ConversationMessage {
  if (typeof value !== "object" || value === null) {
    return false;
  }
  const message = value as Record<string, unknown>;
  const hasBaseFields =
    typeof message.id === "string" &&
    typeof message.content === "string" &&
    typeof message.createdAt === "string";
  if (!hasBaseFields) return false;

  if (message.role === "user") {
    const filters = message.filters;
    return (
      typeof filters === "object" &&
      filters !== null &&
      "doc_type" in filters &&
      "year" in filters
    );
  }

  if (message.role === "assistant") {
    try {
      parseAnswerResponse(message.response);
      return true;
    } catch {
      return false;
    }
  }

  return false;
}

export function useConversation() {
  const [messages, setMessages] = useState<ConversationMessage[]>(loadConversation);

  useEffect(() => {
    try {
      window.localStorage.setItem(CONVERSATION_STORAGE_KEY, JSON.stringify(messages));
    } catch {
      // Persistence is optional when browser storage is unavailable or full.
    }
  }, [messages]);

  return { messages, setMessages };
}
