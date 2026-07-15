// Types mirror the FastAPI backend contract (app/schemas/api.py).

export interface Source {
  document: string;
  page: number;
  section: string;
}

/** How the backend intends a reply to be presented. */
export type ReplyKind = "answer" | "chat" | "redirect";

export interface AskResponse {
  answer: string;
  sources: Source[];
  /** Optional for backwards-compat; older backends omit it (treated as "answer"). */
  kind?: ReplyKind;
}

export interface IngestResponse {
  status: string;
  documents: number;
  chunks: number;
}

export interface StatsResponse {
  collection: string;
  documents: number;
  chunks: number;
  embedding_model: string;
  llm_model: string;
}

export interface HealthResponse {
  status: string;
}

/** The two backends the frontend can talk to. */
export interface Backend {
  id: "manual" | "langchain";
  label: string;
  baseUrl: string;
}
