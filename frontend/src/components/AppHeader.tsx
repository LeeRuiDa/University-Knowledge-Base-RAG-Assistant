import {
  BookOpenCheck,
  Github,
  Moon,
  ShieldCheck,
  Sun,
} from "lucide-react";

import type { Theme } from "../hooks/useTheme";
import type { HealthResponse } from "../types/api";

const REPOSITORY_URL =
  "https://github.com/LeeRuiDa/University-Knowledge-Base-RAG-Assistant";

interface AppHeaderProps {
  health: HealthResponse | null;
  isCheckingHealth: boolean;
  healthUnavailable: boolean;
  theme: Theme;
  onToggleTheme: () => void;
}

function statusLabel(
  health: HealthResponse | null,
  isChecking: boolean,
  unavailable: boolean,
) {
  if (isChecking) return { label: "Checking system", state: "checking" };
  if (unavailable) return { label: "Backend unavailable", state: "offline" };
  if (health?.ready) return { label: "System ready", state: "ready" };
  return { label: "Corpus not ready", state: "warning" };
}

export function AppHeader({
  health,
  isCheckingHealth,
  healthUnavailable,
  theme,
  onToggleTheme,
}: AppHeaderProps) {
  const systemStatus = statusLabel(health, isCheckingHealth, healthUnavailable);

  return (
    <header className="app-header">
      <div className="header-inner">
        <a className="brand" href="#main-content" aria-label="University Policy Assistant home">
          <span className="brand-mark" aria-hidden="true">
            <BookOpenCheck size={22} strokeWidth={1.8} />
          </span>
          <span className="brand-copy">
            <span className="brand-name">University Policy Assistant</span>
            <span className="brand-subtitle">
              Evidence-grounded answers over university policies
            </span>
          </span>
        </a>

        <div className="header-actions">
          <span className="project-label">
            <ShieldCheck size={14} aria-hidden="true" />
            Independent project
          </span>
          <span
            className={`system-status system-status--${systemStatus.state}`}
            aria-label={`System status: ${systemStatus.label}`}
          >
            <span className="status-dot" aria-hidden="true" />
            <span>{systemStatus.label}</span>
          </span>
          <a
            className="icon-button"
            href={REPOSITORY_URL}
            target="_blank"
            rel="noreferrer noopener"
            aria-label="Open the GitHub repository"
            title="GitHub repository"
          >
            <Github size={19} aria-hidden="true" />
          </a>
          <button
            className="icon-button"
            type="button"
            onClick={onToggleTheme}
            aria-label={`Switch to ${theme === "light" ? "dark" : "light"} theme`}
            title={`Switch to ${theme === "light" ? "dark" : "light"} theme`}
          >
            {theme === "light" ? (
              <Moon size={19} aria-hidden="true" />
            ) : (
              <Sun size={19} aria-hidden="true" />
            )}
          </button>
        </div>
      </div>
    </header>
  );
}
