import type {
  AnswerResponse,
  AskRequest,
  HealthResponse,
  MetadataSummary,
} from "../types/api";
import {
  ApiResponseError,
  parseAnswerResponse,
  parseHealthResponse,
  parseMetadataSummary,
} from "./parsers";

export type ApiErrorKind =
  | "configuration"
  | "http"
  | "invalid-response"
  | "network";

export class ApiError extends Error {
  readonly status: number | null;
  readonly kind: ApiErrorKind;

  constructor(message: string, kind: ApiErrorKind, status: number | null = null) {
    super(message);
    this.name = "ApiError";
    this.kind = kind;
    this.status = status;
  }
}

export interface RAGApiClient {
  getHealth(signal?: AbortSignal): Promise<HealthResponse>;
  getMetadata(signal?: AbortSignal): Promise<MetadataSummary>;
  ask(request: AskRequest, signal?: AbortSignal): Promise<AnswerResponse>;
}

type Parser<T> = (value: unknown) => T;

export function createApiClient(baseUrl: string): RAGApiClient {
  const normalizedBaseUrl = baseUrl.trim().replace(/\/$/, "");

  async function request<T>(
    path: string,
    parser: Parser<T>,
    options: RequestInit = {},
  ): Promise<T> {
    if (!normalizedBaseUrl) {
      throw new ApiError(
        "VITE_API_BASE_URL is not configured for this frontend.",
        "configuration",
      );
    }

    let response: Response;
    try {
      response = await fetch(`${normalizedBaseUrl}${path}`, {
        ...options,
        headers: {
          Accept: "application/json",
          ...(options.body ? { "Content-Type": "application/json" } : {}),
          ...options.headers,
        },
      });
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        throw error;
      }
      throw new ApiError(
        "The backend is unavailable. Check the API URL and server status.",
        "network",
      );
    }

    const payload = await readJson(response);
    if (!response.ok) {
      throw new ApiError(errorDetail(payload, response.status), "http", response.status);
    }

    try {
      return parser(payload);
    } catch (error) {
      if (error instanceof ApiResponseError) {
        throw new ApiError(
          `The backend returned an unexpected response: ${error.message}`,
          "invalid-response",
          response.status,
        );
      }
      throw error;
    }
  }

  return {
    getHealth: (signal) => request("/health", parseHealthResponse, { signal }),
    getMetadata: (signal) => request("/metadata", parseMetadataSummary, { signal }),
    ask: (askRequest, signal) =>
      request("/ask", parseAnswerResponse, {
        method: "POST",
        body: JSON.stringify(askRequest),
        signal,
      }),
  };
}

async function readJson(response: Response): Promise<unknown> {
  try {
    return await response.json();
  } catch {
    if (response.ok) {
      throw new ApiError(
        "The backend returned a malformed response.",
        "invalid-response",
        response.status,
      );
    }
    return null;
  }
}

function errorDetail(payload: unknown, status: number): string {
  if (typeof payload === "object" && payload !== null && "detail" in payload) {
    const detail = (payload as { detail?: unknown }).detail;
    if (typeof detail === "string") {
      return detail;
    }
    if (Array.isArray(detail)) {
      const messages = detail
        .map((item) => {
          if (typeof item === "object" && item !== null && "msg" in item) {
            return String((item as { msg: unknown }).msg);
          }
          return null;
        })
        .filter((item): item is string => Boolean(item));
      if (messages.length) {
        return messages.join(" ");
      }
    }
  }
  return `The backend returned HTTP ${status}.`;
}

export const apiClient = createApiClient(import.meta.env.VITE_API_BASE_URL ?? "");
