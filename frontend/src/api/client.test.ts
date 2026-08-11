import { afterEach, describe, expect, it, vi } from "vitest";

import { GROUNDED_ANSWER } from "../test/fixtures";
import { createApiClient } from "./client";

describe("RAG API client", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("sends the typed ask payload and parses a successful response", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(GROUNDED_ANSWER), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);
    const client = createApiClient("https://api.example.test/");

    const response = await client.ask({
      question: GROUNDED_ANSWER.question,
      filters: { doc_type: "internship_policy", year: 2025 },
    });

    expect(response.grounded).toBe(true);
    expect(fetchMock).toHaveBeenCalledWith(
      "https://api.example.test/ask",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("turns FastAPI validation details into a readable error", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({ detail: [{ msg: "String should have at least 3 characters" }] }),
          { status: 422, headers: { "Content-Type": "application/json" } },
        ),
      ),
    );
    const client = createApiClient("https://api.example.test");

    await expect(
      client.ask({ question: "no", filters: { doc_type: null, year: null } }),
    ).rejects.toMatchObject({
      status: 422,
      message: "String should have at least 3 characters",
    });
  });
});
