import type {
  AskResponse,
  HealthResponse,
  IngestResponse,
  StatsResponse,
} from "./types";

/** Raised for any non-2xx response, carrying the backend's `detail` message. */
export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
    this.name = "ApiError";
  }
}

async function request<T>(
  baseUrl: string,
  path: string,
  init?: RequestInit,
  timeoutMs = 60_000,
): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(`${baseUrl}${path}`, {
      ...init,
      signal: controller.signal,
      headers: { "Content-Type": "application/json", ...init?.headers },
    });

    if (!res.ok) {
      // FastAPI returns { detail: "..." } for handled errors.
      let detail = `Request failed (${res.status})`;
      try {
        const body = await res.json();
        if (body?.detail) detail = String(body.detail);
      } catch {
        /* non-JSON error body; keep the generic message */
      }
      throw new ApiError(res.status, detail);
    }

    return (await res.json()) as T;
  } catch (err) {
    if (err instanceof ApiError) throw err;
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new ApiError(0, "The request timed out. Is the backend running?");
    }
    throw new ApiError(
      0,
      "Could not reach the backend. Make sure it's running and CORS is enabled.",
    );
  } finally {
    clearTimeout(timer);
  }
}

export const api = {
  ask: (baseUrl: string, question: string) =>
    request<AskResponse>(baseUrl, "/ask", {
      method: "POST",
      body: JSON.stringify({ question }),
    }),

  ingest: (baseUrl: string) =>
    request<IngestResponse>(baseUrl, "/ingest", { method: "POST" }, 120_000),

  stats: (baseUrl: string) => request<StatsResponse>(baseUrl, "/stats"),

  health: (baseUrl: string) =>
    request<HealthResponse>(baseUrl, "/health", undefined, 5_000),
};
