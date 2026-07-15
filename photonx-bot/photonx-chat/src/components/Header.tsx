import type { Backend } from "../api/types";
import { BACKENDS } from "../api/backends";
import { MoonIcon, SunIcon } from "./icons";

interface Props {
  backendId: Backend["id"];
  onBackendChange: (id: Backend["id"]) => void;
  theme: "light" | "dark";
  onThemeToggle: () => void;
}

export function Header({
  backendId,
  onBackendChange,
  theme,
  onThemeToggle,
}: Props) {
  return (
    <header className="header">
      <div className="brand">
        <div className="logo">
          <img src="/photonx-logo.png" alt="PhotonX" />
        </div>
        <div>
          <div className="title">PhotonX Copilot</div>
          <div className="sub">Your PhotonX docs assistant</div>
        </div>
      </div>

      <div className="header-spacer" />

      <div
        className="toggle"
        role="group"
        aria-label="Choose backend"
        title="Switch which backend answers"
      >
        {BACKENDS.map((b) => (
          <button
            key={b.id}
            className={b.id === backendId ? "active" : ""}
            onClick={() => onBackendChange(b.id)}
            aria-pressed={b.id === backendId}
          >
            {b.label}
          </button>
        ))}
      </div>

      <button
        className="icon-btn"
        onClick={onThemeToggle}
        aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
        title="Toggle theme"
      >
        {theme === "dark" ? <SunIcon /> : <MoonIcon />}
      </button>
    </header>
  );
}
