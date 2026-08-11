import { AlertTriangle, RotateCcw, Trash2 } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { ApiError, apiClient, type RAGApiClient } from "./api/client";
import { AppHeader } from "./components/AppHeader";
import { Composer } from "./components/Composer";
import { EvaluationEvidence } from "./components/EvaluationEvidence";
import { ExamplePrompts } from "./components/ExamplePrompts";
import { KnowledgeBasePanel } from "./components/KnowledgeBasePanel";
import { MessageList } from "./components/MessageList";
import { useConversation } from "./hooks/useConversation";
import { useTheme } from "./hooks/useTheme";
import type { HealthResponse, MetadataSummary, SearchFilters } from "./types/api";
import type { ConversationMessage } from "./types/chat";
import { preferredScrollBehavior } from "./utils/motion";

interface AppProps {
  client?: RAGApiClient;
}

interface RequestErrorState {
  title: string;
  detail: string;
  question: string;
  filters: SearchFilters;
}

const EMPTY_FILTERS: SearchFilters = { doc_type: null, year: null };

function createId(prefix: string) {
  const random = Math.random().toString(36).slice(2, 10);
  return `${prefix}-${Date.now().toString(36)}-${random}`;
}

function requestError(error: unknown, question: string, filters: SearchFilters): RequestErrorState {
  if (error instanceof ApiError) {
    if (error.status === 409) {
      return { title: "Knowledge base not ready", detail: error.message, question, filters };
    }
    if (error.kind === "invalid-response") {
      return { title: "Unexpected backend response", detail: error.message, question, filters };
    }
    if (error.kind === "configuration") {
      return { title: "Frontend configuration required", detail: error.message, question, filters };
    }
    if (error.kind === "http" && error.status !== null && error.status < 500) {
      return { title: "Request rejected", detail: error.message, question, filters };
    }
    if (error.kind === "network") {
      return { title: "Backend unavailable", detail: error.message, question, filters };
    }
    return { title: "Assistant temporarily unavailable", detail: error.message, question, filters };
  }
  return {
    title: "Request failed",
    detail: "The assistant could not complete the request. Please try again.",
    question,
    filters,
  };
}

export default function App({ client = apiClient }: AppProps) {
  const { theme, toggleTheme } = useTheme();
  const { messages, setMessages } = useConversation();
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [metadata, setMetadata] = useState<MetadataSummary | null>(null);
  const [systemLoading, setSystemLoading] = useState(true);
  const [systemError, setSystemError] = useState(false);
  const [systemReload, setSystemReload] = useState(0);
  const [filters, setFilters] = useState<SearchFilters>(EMPTY_FILTERS);
  const [question, setQuestion] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isSlow, setIsSlow] = useState(false);
  const [requestStatus, setRequestStatus] = useState("");
  const [error, setError] = useState<RequestErrorState | null>(null);
  const activeRequest = useRef<{ id: string; controller: AbortController } | null>(null);
  const conversationEnd = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    setSystemLoading(true);
    setSystemError(false);

    Promise.all([
      client.getHealth(controller.signal),
      client.getMetadata(controller.signal),
    ])
      .then(([healthResponse, metadataResponse]) => {
        setHealth(healthResponse);
        setMetadata(metadataResponse);
      })
      .catch((loadError: unknown) => {
        if (!(loadError instanceof DOMException && loadError.name === "AbortError")) {
          setHealth(null);
          setMetadata(null);
          setSystemError(true);
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setSystemLoading(false);
      });

    return () => controller.abort();
  }, [client, systemReload]);

  useEffect(() => {
    if (!messages.length && !isLoading) return;
    conversationEnd.current?.scrollIntoView({
      behavior: preferredScrollBehavior(),
      block: "end",
    });
  }, [messages, isLoading]);

  async function runQuestion(
    nextQuestion: string,
    requestFilters: SearchFilters,
    appendUser = true,
  ) {
    const cleanedQuestion = nextQuestion.trim();
    if (cleanedQuestion.length < 3 || isLoading) return;

    const requestId = createId("request");
    const controller = new AbortController();
    activeRequest.current = { id: requestId, controller };
    setIsLoading(true);
    setIsSlow(false);
    setError(null);
    setRequestStatus("Searching official sources.");
    setQuestion("");

    if (appendUser) {
      const userMessage: ConversationMessage = {
        id: createId("user"),
        role: "user",
        content: cleanedQuestion,
        createdAt: new Date().toISOString(),
        filters: requestFilters,
      };
      setMessages((current) => [...current, userMessage]);
    }

    const slowTimer = window.setTimeout(() => {
      if (activeRequest.current?.id === requestId) {
        setIsSlow(true);
        setRequestStatus("The request is still in progress.");
      }
    }, 3500);

    try {
      const response = await client.ask(
        { question: cleanedQuestion, filters: requestFilters },
        controller.signal,
      );
      if (activeRequest.current?.id !== requestId) return;
      const assistantMessage: ConversationMessage = {
        id: createId("assistant"),
        role: "assistant",
        content: response.answer,
        createdAt: new Date().toISOString(),
        response,
      };
      setMessages((current) => [...current, assistantMessage]);
      setRequestStatus(
        response.grounded
          ? "Grounded answer ready with retrieved evidence."
          : "The assistant refused because the corpus did not contain enough evidence.",
      );
    } catch (requestFailure) {
      if (requestFailure instanceof DOMException && requestFailure.name === "AbortError") {
        setRequestStatus("Request cancelled.");
      } else if (activeRequest.current?.id === requestId) {
        setError(requestError(requestFailure, cleanedQuestion, requestFilters));
        setRequestStatus("The request failed.");
      }
    } finally {
      window.clearTimeout(slowTimer);
      if (activeRequest.current?.id === requestId) {
        activeRequest.current = null;
        setIsLoading(false);
        setIsSlow(false);
      }
    }
  }

  function submitComposer() {
    void runQuestion(question, filters);
  }

  function cancelRequest() {
    activeRequest.current?.controller.abort();
  }

  function clearConversation() {
    activeRequest.current?.controller.abort();
    activeRequest.current = null;
    setMessages([]);
    setError(null);
    setIsLoading(false);
    setQuestion("");
    setRequestStatus("Conversation cleared.");
  }

  const hasConversation = messages.length > 0 || isLoading;

  return (
    <div className="app-shell">
      <AppHeader
        health={health}
        isCheckingHealth={systemLoading}
        healthUnavailable={systemError}
        theme={theme}
        onToggleTheme={toggleTheme}
      />

      <main id="main-content" className="main-content">
        <div className="hero-row">
          <div>
            <p className="eyebrow">University of Nebraska–Lincoln · curated public corpus</p>
            <h1>
              Ask the policy.<br /> Inspect the evidence.
            </h1>
            <p className="hero-copy">
              A portfolio RAG system for undergraduate computer science requirements and student
              policies. Every supported answer is paired with the chunks retrieved from official
              public documents.
            </p>
          </div>
          <div className="scope-card">
            <span>Current corpus</span>
            <strong>24 official documents</strong>
            <p>
              Program requirements, internships, registration, academic calendars, billing,
              financial aid, and student support.
            </p>
          </div>
        </div>

        <EvaluationEvidence />

        <KnowledgeBasePanel
          metadata={metadata}
          isLoading={systemLoading}
          filters={filters}
          onFiltersChange={setFilters}
          onRetry={() => setSystemReload((value) => value + 1)}
        />

        {!hasConversation ? (
          <section className="welcome-section" aria-labelledby="examples-title">
            <div className="section-heading-row section-heading-row--welcome">
              <div>
                <p className="eyebrow">Start with a verified question</p>
                <h2 id="examples-title">Explore what the corpus knows</h2>
              </div>
              <p>
                The boundary test is intentionally outside the corpus and should produce a calm,
                explicit refusal.
              </p>
            </div>
            <ExamplePrompts
              onSelect={(selectedQuestion) => void runQuestion(selectedQuestion, filters)}
              disabled={isLoading}
            />
          </section>
        ) : (
          <div className="conversation-toolbar">
            <span>Conversation</span>
            <button type="button" className="text-button" onClick={clearConversation}>
              <Trash2 size={15} aria-hidden="true" /> Clear conversation
            </button>
          </div>
        )}

        <MessageList messages={messages} isLoading={isLoading} isSlow={isSlow} />

        {error ? (
          <div className="request-error" role="alert">
            <AlertTriangle size={20} aria-hidden="true" />
            <div>
              <strong>{error.title}</strong>
              <p>{error.detail}</p>
            </div>
            <button
              type="button"
              className="secondary-button"
              onClick={() => void runQuestion(error.question, error.filters, false)}
              disabled={isLoading}
            >
              <RotateCcw size={15} aria-hidden="true" /> Try again
            </button>
          </div>
        ) : null}

        <div ref={conversationEnd} />
      </main>

      <div className="composer-region">
        <Composer
          value={question}
          isLoading={isLoading}
          onChange={setQuestion}
          onSubmit={submitComposer}
          onCancel={cancelRequest}
        />
      </div>

      <div className="sr-only" aria-live="polite" aria-atomic="true">
        {requestStatus}
      </div>
    </div>
  );
}
