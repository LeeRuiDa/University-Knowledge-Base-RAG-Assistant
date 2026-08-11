import { Send, Square } from "lucide-react";
import type { KeyboardEvent } from "react";

interface ComposerProps {
  value: string;
  isLoading: boolean;
  onChange: (value: string) => void;
  onSubmit: () => void;
  onCancel: () => void;
}

export function Composer({
  value,
  isLoading,
  onChange,
  onSubmit,
  onCancel,
}: ComposerProps) {
  const isValid = value.trim().length >= 3;

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      if (isValid && !isLoading) onSubmit();
    }
  }

  return (
    <div className="composer-shell">
      <div className="composer">
        <label className="sr-only" htmlFor="question-composer">
          Ask a question about university policies
        </label>
        <textarea
          id="question-composer"
          rows={2}
          value={value}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask about graduation, internships, registration, fees, or student support…"
          aria-describedby="composer-help"
          disabled={isLoading}
        />
        <div className="composer-actions">
          {isLoading ? (
            <button className="cancel-button" type="button" onClick={onCancel}>
              <Square size={14} fill="currentColor" aria-hidden="true" /> Cancel
            </button>
          ) : null}
          <button
            className="submit-button"
            type="button"
            onClick={onSubmit}
            disabled={!isValid || isLoading}
            aria-label="Submit question"
          >
            <Send size={18} aria-hidden="true" />
          </button>
        </div>
      </div>
      <p id="composer-help" className="composer-help">
        Enter to submit · Shift+Enter for a new line
      </p>
    </div>
  );
}
