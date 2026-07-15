import type { StatsResponse } from "../api/types";

export type HealthState = "unknown" | "ok" | "down";

interface Props {
  health: HealthState;
  stats: StatsResponse | null;
  onIngest: () => void;
  ingesting: boolean;
}

export function StatusStrip({ health, stats, onIngest, ingesting }: Props) {
  const healthLabel =
    health === "ok" ? "Online" : health === "down" ? "Offline" : "Checking…";

  return (
    <div className="status-strip">
      <span className="pill" title="Backend health">
        <span className={`dot ${health}`} />
        {healthLabel}
      </span>

      {stats && (
        <>
          <span className="pill" title="Distinct documents indexed">
            {stats.documents} doc{stats.documents !== 1 ? "s" : ""}
          </span>
          <span className="pill" title="Total chunks stored">
            {stats.chunks} chunks
          </span>
          <span className="pill" title="Chat model">
            {stats.llm_model}
          </span>
        </>
      )}

      <span className="grow" />

      <button
        className="ingest-btn"
        onClick={onIngest}
        disabled={ingesting || health !== "ok"}
        title="Re-index all PDFs in the backend's docs folder"
      >
        {ingesting ? "Indexing…" : "Re-index docs"}
      </button>
    </div>
  );
}
