import type { Backend } from "./types";

// The two backends this UI can switch between at runtime.
// Override the URLs with .env vars (VITE_MANUAL_URL / VITE_LANGCHAIN_URL).
export const BACKENDS: Backend[] = [
  {
    id: "manual",
    label: "Manual",
    baseUrl: import.meta.env.VITE_MANUAL_URL ?? "http://localhost:8000",
  },
  {
    id: "langchain",
    label: "LangChain",
    baseUrl: import.meta.env.VITE_LANGCHAIN_URL ?? "http://localhost:8001",
  },
];

export const DEFAULT_BACKEND_ID: Backend["id"] = "manual";
