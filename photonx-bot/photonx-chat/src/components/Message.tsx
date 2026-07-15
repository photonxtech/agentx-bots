import type { ReplyKind, Source } from "../api/types";
import { DocIcon } from "./icons";
import { Markdown } from "./Markdown";

export type Role = "user" | "bot";

export interface ChatMessage {
  id: string;
  role: Role;
  text: string;
  sources?: Source[];
  isError?: boolean;
  /** Backend reply kind: "answer" | "chat" | "redirect". */
  kind?: ReplyKind;
}

function SourceCards({ sources }: { sources: Source[] }) {
  return (
    <div className="sources">
      <div className="sources-label">
        {sources.length} source{sources.length !== 1 ? "s" : ""}
      </div>
      <div className="source-cards">
        {sources.map((s, i) => (
          <div className="source-card" key={`${s.document}-${s.page}-${i}`}>
            <span className="doc">
              <DocIcon />
              {s.document}
            </span>
            <span className="meta">
              Page {s.page}
              {s.section && s.section !== "N/A" ? ` · ${s.section}` : ""}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

/** Shown under a "redirect" (not-in-docs) reply so the user always has a next step. */
const REDIRECT_SUGGESTIONS = [
  "What services does PhotonX provide?",
  "What technologies does PhotonX use?",
  "Tell me about the PhotonX team.",
  "What do clients say about PhotonX?",
];

export function Message({
  message,
  onSuggest,
  suggestDisabled,
}: {
  message: ChatMessage;
  /** Click-to-send handler; when provided, redirect replies show suggestion chips. */
  onSuggest?: (question: string) => void;
  suggestDisabled?: boolean;
}) {
  const isBot = message.role === "bot";
  const showSuggestions =
    isBot && message.kind === "redirect" && !message.isError && !!onSuggest;
  const bubbleClass = [
    "bubble",
    message.isError ? "error" : "",
    // "redirect" (not-in-docs) gets a soft, non-alarming treatment; "chat"
    // (small talk) renders as a normal, friendly bubble.
    message.kind === "redirect" ? "refusal" : "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div className={`msg ${message.role}`}>
      <div className={`avatar ${isBot ? "bot" : "you"}`}>
        {isBot ? (
          <img src="/photonx-logo.png" alt="Buddy" />
        ) : (
          "You"
        )}
      </div>
      <div className="msg-body">
        <div className={bubbleClass}>
          {isBot && !message.isError ? (
            <Markdown text={message.text} />
          ) : (
            message.text
          )}
        </div>
        {isBot && message.sources && message.sources.length > 0 && (
          <SourceCards sources={message.sources} />
        )}
        {showSuggestions && (
          <div className="suggestions inline">
            {REDIRECT_SUGGESTIONS.map((q) => (
              <button
                key={q}
                onClick={() => onSuggest!(q)}
                disabled={suggestDisabled}
              >
                {q}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export function TypingBubble() {
  return (
    <div className="msg bot">
      <div className="avatar bot">
        <img src="/photonx-logo.png" alt="PhotonX Copilot" />
      </div>
      <div className="bubble">
        <div className="typing" aria-label="Assistant is typing">
          <span />
          <span />
          <span />
        </div>
      </div>
    </div>
  );
}
