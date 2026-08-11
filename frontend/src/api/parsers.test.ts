import { describe, expect, it } from "vitest";

import { GROUNDED_ANSWER, HEALTH, METADATA } from "../test/fixtures";
import {
  ApiResponseError,
  parseAnswerResponse,
  parseHealthResponse,
  parseMetadataSummary,
} from "./parsers";

describe("API response parsers", () => {
  it("parses responses that match the Pydantic contract", () => {
    expect(parseHealthResponse(HEALTH)).toEqual(HEALTH);
    expect(parseMetadataSummary(METADATA)).toEqual(METADATA);
    expect(parseAnswerResponse(GROUNDED_ANSWER)).toEqual(GROUNDED_ANSWER);
  });

  it("rejects malformed answer source data", () => {
    const malformed = {
      ...GROUNDED_ANSWER,
      sources: [{ ...GROUNDED_ANSWER.sources[0], score: "high" }],
    };

    expect(() => parseAnswerResponse(malformed)).toThrow(ApiResponseError);
    expect(() => parseAnswerResponse(malformed)).toThrow("sources[0].score");
  });

  it("rejects invented or incomplete response fields", () => {
    expect(() => parseHealthResponse({ status: "ok", ready: true })).toThrow(
      "collection_name",
    );
  });
});
