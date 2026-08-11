import type { AnswerResponse, SearchFilters } from "./api";

export interface UserMessage {
  id: string;
  role: "user";
  content: string;
  createdAt: string;
  filters: SearchFilters;
}

export interface AssistantMessage {
  id: string;
  role: "assistant";
  content: string;
  createdAt: string;
  response: AnswerResponse;
}

export type ConversationMessage = UserMessage | AssistantMessage;
