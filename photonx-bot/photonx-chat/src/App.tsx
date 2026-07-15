import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import "./App.css";
import { api, ApiError } from "./api/client";
import { BACKENDS, DEFAULT_BACKEND_ID } from "./api/backends";
import type { Backend, StatsResponse } from "./api/types";
import { Composer } from "./components/Composer";
import { Header } from "./components/Header";
import { Message, TypingBubble, type ChatMessage } from "./components/Message";
import { StatusStrip, type HealthState } from "./components/StatusStrip";

const SUGGESTIONS = [
  "What services does PhotonX provide?",
  "Who founded PhotonX?",
  "What do clients say about PhotonX?",
];

function useTheme() {
  const [theme, setTheme] = useState<"light" | "dark">(() => {
    const saved = localStorage.getItem("theme");
    if (saved === "light" || saved === "dark") return saved;
    return window.matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark"
      : "light";
  });
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("theme", theme);
  }, [theme]);
  return { theme, toggle: () => setTheme((t) => (t === "dark" ? "light" : "dark")) };
}

export default function App() {
  const { theme, toggle } = useTheme();

  const [backendId, setBackendId] = useState<Backend["id"]>(DEFAULT_BACKEND_ID);
  const backend = useMemo(
    () => BACKENDS.find((b) => b.id === backendId)!,
    [backendId],
  );

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [asking, setAsking] = useState(false);
  const [health, setHealth] = useState<HealthState>("unknown");
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [ingesting, setIngesting] = useState(false);
  const [toast, setToast] = useState<{ text: string; error?: boolean } | null>(
    null,
  );

  const scrollRef = useRef<HTMLDivElement>(null);

  // Reflect the active backend as an accent on the root element.
  useEffect(() => {
    document.documentElement.setAttribute("data-accent", backendId);
  }, [backendId]);

  // Poll health + stats whenever the backend changes.
  const refreshStatus = useCallback(async () => {
    try {
      await api.health(backend.baseUrl);
      setHealth("ok");
      try {
        setStats(await api.stats(backend.baseUrl));
      } catch {
        setStats(null);
      }
    } catch {
      setHealth("down");
      setStats(null);
    }
  }, [backend.baseUrl]);

  useEffect(() => {
    setHealth("unknown");
    setStats(null);
    refreshStatus();
  }, [refreshStatus]);

  // Auto-scroll to the newest message.
  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages, asking]);

  // Auto-dismiss toasts.
  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(null), 3200);
    return () => clearTimeout(t);
  }, [toast]);

  const send = useCallback(
    async (question: string) => {
      const userMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: "user",
        text: question,
      };
      setMessages((m) => [...m, userMsg]);
      setAsking(true);

      try {
        const res = await api.ask(backend.baseUrl, question);
        setMessages((m) => [
          ...m,
          {
            id: crypto.randomUUID(),
            role: "bot",
            text: res.answer,
            sources: res.sources,
            kind: res.kind ?? "answer",
          },
        ]);
      } catch (err) {
        const detail =
          err instanceof ApiError ? err.message : "Something went wrong.";
        setMessages((m) => [
          ...m,
          { id: crypto.randomUUID(), role: "bot", text: detail, isError: true },
        ]);
        if (err instanceof ApiError && err.status === 0) setHealth("down");
      } finally {
        setAsking(false);
      }
    },
    [backend.baseUrl],
  );

  const ingest = useCallback(async () => {
    setIngesting(true);
    try {
      const res = await api.ingest(backend.baseUrl);
      setToast({
        text: `Indexed ${res.documents} document${
          res.documents !== 1 ? "s" : ""
        } · ${res.chunks} chunks`,
      });
      await refreshStatus();
    } catch (err) {
      const detail =
        err instanceof ApiError ? err.message : "Indexing failed.";
      setToast({ text: detail, error: true });
    } finally {
      setIngesting(false);
    }
  }, [backend.baseUrl, refreshStatus]);

  return (
    <div className="app">
      <Header
        backendId={backendId}
        onBackendChange={setBackendId}
        theme={theme}
        onThemeToggle={toggle}
      />
      <StatusStrip
        health={health}
        stats={stats}
        onIngest={ingest}
        ingesting={ingesting}
      />

      <div className="messages" ref={scrollRef}>
        {messages.length === 0 ? (
          <div className="empty">
            <div className="welcome-logo">
              <img src="/photonx-logo.png" alt="PhotonX" />
            </div>
            <h2>Hi, I'm Buddy 👋</h2>
            <p>
              I'm your PhotonX buddy — ask me anything about PhotonX's services,
              technology, team, or client feedback. Every answer comes straight
              from the official docs, with page-level citations, and I'll never
              make things up.
            </p>
            <p className="welcome-cta">Try one of these to get started:</p>
            <div className="suggestions">
              {SUGGESTIONS.map((q) => (
                <button key={q} onClick={() => send(q)} disabled={asking}>
                  {q}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((m) => (
            <Message
              key={m.id}
              message={m}
              onSuggest={send}
              suggestDisabled={asking}
            />
          ))
        )}
        {asking && <TypingBubble />}
      </div>

      <Composer onSend={send} disabled={asking} />

      {toast && (
        <div className={`toast ${toast.error ? "error" : ""}`} role="status">
          {toast.text}
        </div>
      )}
    </div>
  );
}
