import { ArrowRight, CalendarDays, CircleDollarSign, ShieldQuestion } from "lucide-react";

import { BOUNDARY_PROMPT, EXAMPLE_PROMPTS } from "../data/prompts";

const PROMPT_ICONS = [ArrowRight, CalendarDays, CircleDollarSign] as const;

interface ExamplePromptsProps {
  onSelect: (question: string) => void;
  disabled: boolean;
}

export function ExamplePrompts({ onSelect, disabled }: ExamplePromptsProps) {
  return (
    <div className="prompt-grid" aria-label="Verified example questions">
      {EXAMPLE_PROMPTS.map(({ question, label }, index) => {
        const Icon = PROMPT_ICONS[index] ?? ArrowRight;
        return (
          <button
            className="prompt-card"
            type="button"
            key={question}
            onClick={() => onSelect(question)}
            disabled={disabled}
          >
            <span className="prompt-card-icon" aria-hidden="true">
              <Icon size={17} />
            </span>
            <span>
              <strong>{label}</strong>
              <span>{question}</span>
            </span>
          </button>
        );
      })}

      <button
        className="prompt-card prompt-card--boundary"
        type="button"
        onClick={() => onSelect(BOUNDARY_PROMPT.question)}
        disabled={disabled}
      >
        <span className="prompt-card-icon" aria-hidden="true">
          <ShieldQuestion size={17} />
        </span>
        <span>
          <strong>{BOUNDARY_PROMPT.label}</strong>
          <span>{BOUNDARY_PROMPT.question}</span>
        </span>
      </button>
    </div>
  );
}
