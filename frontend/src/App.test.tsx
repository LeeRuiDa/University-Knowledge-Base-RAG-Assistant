import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError, type RAGApiClient } from "./api/client";
import App from "./App";
import { GROUNDED_ANSWER, HEALTH, METADATA, REFUSAL_ANSWER } from "./test/fixtures";

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((promiseResolve) => {
    resolve = promiseResolve;
  });
  return { promise, resolve };
}

function testClient(overrides: Partial<RAGApiClient> = {}): RAGApiClient {
  return {
    getHealth: vi.fn().mockResolvedValue(HEALTH),
    getMetadata: vi.fn().mockResolvedValue(METADATA),
    ask: vi.fn().mockResolvedValue(GROUNDED_ANSWER),
    ...overrides,
  };
}

async function submitTypedQuestion(question: string) {
  const user = userEvent.setup();
  await user.type(screen.getByLabelText("Ask a question about university policies"), question);
  await user.click(screen.getByRole("button", { name: "Submit question" }));
}

describe("University Policy Assistant", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    window.localStorage.clear();
    document.documentElement.dataset.theme = "";
  });

  it("keeps the initial portfolio view at the top", async () => {
    render(<App client={testClient()} />);

    await screen.findByText("System ready");
    expect(Element.prototype.scrollIntoView).not.toHaveBeenCalled();
    expect(
      screen.getByRole("heading", {
        level: 1,
        name: /Ask the policy\. Inspect the evidence\./i,
      }),
    ).toBeInTheDocument();
  });

  it("shows an honest loading state and then renders grounded evidence", async () => {
    const pendingAnswer = deferred<typeof GROUNDED_ANSWER>();
    const client = testClient({ ask: vi.fn().mockReturnValue(pendingAnswer.promise) });
    const user = userEvent.setup();
    render(<App client={client} />);

    await user.click(
      screen.getByRole("button", {
        name: /How many credits of CSCE 495 count as one tech elective course/i,
      }),
    );
    expect(screen.getByText("Searching official sources…")).toBeInTheDocument();

    pendingAnswer.resolve(GROUNDED_ANSWER);

    expect(await screen.findByText("Grounded answer")).toBeInTheDocument();
    const citationLink = screen.getByRole("link", { name: "S1" });
    expect(citationLink).toHaveAttribute(
      "href",
      expect.stringContaining("#source-"),
    );
    expect(citationLink).not.toHaveAttribute("node");
    expect(
      screen.getByText("UNL Internship Credit for Computing Students"),
    ).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Open official source" })).toHaveAttribute(
      "href",
      "https://computing.unl.edu/internship-credit/",
    );
    await waitFor(() =>
      expect(window.localStorage.getItem("university-policy-assistant-conversation")).toContain(
        GROUNDED_ANSWER.answer,
      ),
    );
  });

  it("renders a refusal as a deliberate reliability boundary", async () => {
    const client = testClient({ ask: vi.fn().mockResolvedValue(REFUSAL_ANSWER) });
    render(<App client={client} />);

    await submitTypedQuestion(REFUSAL_ANSWER.question);

    expect(await screen.findByText("Boundary held")).toBeInTheDocument();
    expect(screen.getByText("Not enough evidence in the indexed corpus")).toBeInTheDocument();
    expect(screen.getByText(REFUSAL_ANSWER.warning!)).toBeInTheDocument();
    expect(screen.getByText("No source chunks were attached to this response.")).toBeInTheDocument();
  });

  it("populates and submits metadata-driven filters", async () => {
    const ask = vi.fn().mockResolvedValue(GROUNDED_ANSWER);
    const client = testClient({ ask });
    const user = userEvent.setup();
    render(<App client={client} />);

    await screen.findByText("System ready");
    expect(screen.queryByText("data/corpus_manifest.csv")).not.toBeInTheDocument();
    await user.click(screen.getByText("Knowledge base"));
    await user.selectOptions(screen.getByLabelText("Document type"), "internship_policy");
    await user.selectOptions(screen.getByLabelText("Year"), "2025");
    await submitTypedQuestion(GROUNDED_ANSWER.question);

    await waitFor(() =>
      expect(ask).toHaveBeenCalledWith(
        expect.objectContaining({
          filters: { doc_type: "internship_policy", year: 2025 },
        }),
        expect.any(AbortSignal),
      ),
    );
  });

  it("distinguishes a corpus-not-ready 409 from an application failure", async () => {
    const client = testClient({
      ask: vi.fn().mockRejectedValue(
        new ApiError("The knowledge base is not indexed yet. Run ingestion first.", "http", 409),
      ),
    });
    render(<App client={client} />);

    await submitTypedQuestion("How many credits are required to graduate?");

    const alert = await screen.findByRole("alert");
    expect(within(alert).getByText("Knowledge base not ready")).toBeInTheDocument();
    expect(within(alert).getByText(/not indexed yet/i)).toBeInTheDocument();
  });

  it("reports backend network failures without losing the question", async () => {
    const client = testClient({
      ask: vi.fn().mockRejectedValue(
        new ApiError("The backend is unavailable. Check the API URL and server status.", "network"),
      ),
    });
    render(<App client={client} />);

    await submitTypedQuestion("What late payment fee is assessed?");

    const alert = await screen.findByRole("alert");
    expect(within(alert).getByText("Backend unavailable")).toBeInTheDocument();
    expect(screen.getByText("What late payment fee is assessed?")).toBeInTheDocument();
  });

  it("cancels an in-flight request and restores the composer", async () => {
    const ask = vi.fn(
      (_request: Parameters<RAGApiClient["ask"]>[0], signal?: AbortSignal) =>
        new Promise<typeof GROUNDED_ANSWER>((_resolve, reject) => {
          signal?.addEventListener("abort", () => {
            reject(new DOMException("The operation was aborted.", "AbortError"));
          });
        }),
    );
    const user = userEvent.setup();
    render(<App client={testClient({ ask })} />);

    await submitTypedQuestion("When is priority registration for Fall Semester 2025?");
    await user.click(screen.getByRole("button", { name: "Cancel" }));

    await waitFor(() => expect(screen.queryByRole("button", { name: "Cancel" })).not.toBeInTheDocument());
    expect((ask.mock.calls[0]?.[1] as AbortSignal).aborted).toBe(true);
    expect(screen.getByLabelText("Ask a question about university policies")).toBeEnabled();
  });

  it("shows API validation errors as rejected requests", async () => {
    const client = testClient({
      ask: vi.fn().mockRejectedValue(
        new ApiError("String should have at least 3 characters", "http", 422),
      ),
    });
    render(<App client={client} />);

    await submitTypedQuestion("abc");

    const alert = await screen.findByRole("alert");
    expect(within(alert).getByText("Request rejected")).toBeInTheDocument();
    expect(within(alert).getByText(/at least 3 characters/i)).toBeInTheDocument();
  });

  it("persists the selected theme", async () => {
    const user = userEvent.setup();
    render(<App client={testClient()} />);

    await user.click(screen.getByRole("button", { name: "Switch to dark theme" }));

    expect(document.documentElement.dataset.theme).toBe("dark");
    expect(window.localStorage.getItem("university-policy-assistant-theme")).toBe("dark");
  });
});
