"""
PhotonX Copilot - Streamlit Interface
A polished, chat-first landing experience over the hybrid RAG engine in rag_engine.py.
"""

import base64
import html
import json
import time
from pathlib import Path

import streamlit as st
from rag_engine import load_resources, ask
from llm_metrics import JUDGE_MODEL, METRICS, score_answer
from evaluate import DEFAULT_DATASET, build_summary, load_dataset, run_one

LOGO_PATH = Path(__file__).parent / "assets" / "photonx-logo.png"
LOGO_B64 = base64.b64encode(LOGO_PATH.read_bytes()).decode("utf-8")

st.set_page_config(
    page_title="PhotonX Copilot",
    page_icon=str(LOGO_PATH),
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg-deep: #0B0F19;
        --bg-panel: #131826;
        --bg-panel-hover: #171E30;
        --border: #232A3D;
        --accent-amber: #F2A93B;
        --accent-cyan: #4DD8E8;
        --text-primary: #EDEFF5;
        --text-muted: #8A93A6;
    }

    #MainMenu, footer { visibility: hidden; }
    /* `header` used to be hidden wholesale here. That also hid the chevron that
       opens the sidebar -- and since set_page_config sets
       initial_sidebar_state="collapsed", the sidebar became unreachable: no way
       to get at the "Score every answer" toggle at all. Hide the toolbar and
       the top decoration strip instead, which is what was actually unwanted,
       and leave the sidebar control clickable. */
    header[data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { visibility: hidden; height: 0; }
    [data-testid="stDecoration"] { display: none; }
    .stApp {
        background: radial-gradient(ellipse 80% 50% at 50% -10%, rgba(242,169,59,0.10), transparent),
                    radial-gradient(ellipse 60% 40% at 85% 15%, rgba(77,216,232,0.08), transparent),
                    var(--bg-deep);
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
    }
    .block-container { padding-top: 3rem; max-width: 760px; }

    h1, h2, h3, .hero-title { font-family: 'Space Grotesk', sans-serif; }

    /* Hero */
    .hero-wrap { text-align: center; margin-bottom: 2.2rem; }
    .hero-mark {
        display: inline-flex; align-items: center; justify-content: center;
        width: 52px; height: 52px; border-radius: 14px;
        background: var(--bg-panel);
        box-shadow: 0 0 32px rgba(242,169,59,0.35);
        margin-bottom: 14px;
        overflow: hidden;
        animation: pulse-glow 3.5s ease-in-out infinite;
    }
    .hero-mark img { width: 100%; height: 100%; object-fit: cover; display: block; }
    @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 0 24px rgba(242,169,59,0.30); }
        50% { box-shadow: 0 0 40px rgba(77,216,232,0.35); }
    }
    .hero-title {
        font-size: 2.1rem; font-weight: 700; margin: 0 0 6px 0;
        background: linear-gradient(90deg, #fff 40%, var(--accent-amber) 100%);
        -webkit-background-clip: text; background-clip: text; color: transparent;
    }
    .hero-sub { color: var(--text-muted); font-size: 0.98rem; margin: 0; }

    /* Suggestion buttons */
    div[data-testid="stButton"] > button {
        width: 100%; text-align: left; white-space: normal;
        background: var(--bg-panel) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.88rem !important;
        padding: 14px 16px !important;
        min-height: 76px;
        transition: all 0.18s ease;
    }
    div[data-testid="stButton"] > button:hover {
        border-color: var(--accent-amber) !important;
        background: var(--bg-panel-hover) !important;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(242,169,59,0.12);
    }

    /* Chat bubbles */
    div[data-testid="stChatMessage"] {
        background: var(--bg-panel);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 4px 6px;
        margin-bottom: 10px;
        animation: rise-in 0.35s ease;
    }
    @keyframes rise-in {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Sources -- a single collapsed expander instead of a row of dead-end
       chips. Opening it shows the actual excerpt each answer was pulled
       from, which is the practical version of "click through to that part
       of the document" given the source is a local .docx with no hosted
       page to deep-link to. */
    div[data-testid="stExpander"] {
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        background: var(--bg-panel) !important;
        margin-top: 10px !important;
    }
    div[data-testid="stExpander"] summary {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.76rem !important;
        color: var(--text-muted) !important;
    }
    .source-entry { margin-bottom: 10px; }
    .source-entry:last-child { margin-bottom: 0; }
    .source-heading {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.76rem; color: var(--accent-cyan);
        display: block; margin-bottom: 3px;
    }
    .source-excerpt {
        font-size: 0.85rem; color: var(--text-muted);
        line-height: 1.5; margin: 0;
    }

    div[data-testid="stChatInput"] textarea { font-family: 'Inter', sans-serif !important; }

    /* Per-answer RAG scores, computed by DeepEval in llm_metrics.py and
       rendered under the reply they belong to.
       Deliberately quiet -- a metadata strip, not a second headline
       competing with the answer. */
    .score-strip {
        display: flex; flex-wrap: wrap; gap: 7px; align-items: center;
        margin-top: 9px;
    }
    .score-chip {
        display: inline-flex; align-items: baseline; gap: 6px;
        border: 1px solid var(--border); border-radius: 999px;
        background: var(--bg-panel); padding: 3px 11px;
        font-family: 'JetBrains Mono', monospace; font-size: 0.7rem;
        color: var(--text-muted);
    }
    .score-chip b { font-weight: 500; font-size: 0.76rem; }
    .score-good b { color: #5FD08A; }
    .score-mid  b { color: var(--accent-amber); }
    .score-low  b { color: #E86A6A; }
    /* Marks the metric where a low number is the good outcome, so a green
       0.05 next to a green 0.95 doesn't read as a bug. */
    .score-chip i {
        font-style: normal; font-size: 0.62rem; color: var(--text-muted);
        opacity: 0.75;
    }
    .score-note { font-family: 'JetBrains Mono', monospace;
                  font-size: 0.68rem; color: var(--text-muted); }

    /* The judge's own justification per metric, inside the expander. */
    .why-row {
        display: grid; grid-template-columns: 1fr auto;
        gap: 2px 10px; margin-bottom: 12px;
    }
    .why-label { font-size: 0.8rem; color: var(--text-primary); }
    .why-score {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem; color: var(--accent-amber);
    }
    .why-bar {
        grid-column: 1 / -1; height: 4px; border-radius: 2px;
        background: var(--border); overflow: hidden;
    }
    .why-bar-fill {
        height: 100%; border-radius: 2px;
        background: linear-gradient(90deg, var(--accent-cyan), var(--accent-amber));
    }
    .why-reason {
        grid-column: 1 / -1;
        font-size: 0.76rem; color: var(--text-muted);
        line-height: 1.5; margin: 3px 0 0 0;
    }
    .why-foot {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem; color: var(--text-muted); line-height: 1.7;
        margin-top: 2px; padding-top: 9px; border-top: 1px solid var(--border);
    }

    /* System-level report card, rendered from eval_summary.json which
       evaluate.py writes. Aggregates over the whole benchmark set -- visually
       heavier than the per-answer chips because it says more. */
    .eval-row {
        display: grid; grid-template-columns: 1fr auto;
        gap: 3px 10px; margin-bottom: 13px;
    }
    .eval-label { font-size: 0.82rem; color: var(--text-primary); }
    .eval-label .eval-dir {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem; color: var(--text-muted); margin-left: 6px;
    }
    .eval-score {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem; color: var(--accent-amber);
    }
    .eval-bar {
        grid-column: 1 / -1; height: 4px; border-radius: 2px;
        background: var(--border); overflow: hidden;
    }
    .eval-bar-fill {
        height: 100%; border-radius: 2px;
        background: linear-gradient(90deg, var(--accent-cyan), var(--accent-amber));
    }
    .eval-foot {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem; color: var(--text-muted); line-height: 1.7;
        margin-top: 4px; padding-top: 9px; border-top: 1px solid var(--border);
    }
    .eval-foot code, .eval-empty code {
        font-family: 'JetBrains Mono', monospace; font-size: 0.76rem;
        color: var(--accent-cyan); background: rgba(77,216,232,0.08);
        padding: 1px 5px; border-radius: 4px;
    }
    .eval-empty { font-size: 0.82rem; color: var(--text-muted); line-height: 1.6; margin: 0; }

    /* Per-question breakdown -- which questions carry the average and which
       drag it down. An aggregate alone hides that. */
    .eval-perq {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem; color: var(--text-muted);
        margin: 14px 0 8px 0; text-transform: uppercase; letter-spacing: 0.06em;
    }
    .eval-q { margin-bottom: 11px; }
    .eval-q-text {
        font-size: 0.8rem; color: var(--text-primary);
        display: block; margin-bottom: 5px;
    }
    .eval-q-chips { display: flex; flex-wrap: wrap; gap: 5px; }
    .eval-q-chip {
        display: inline-flex; align-items: baseline; gap: 4px;
        border: 1px solid var(--border); border-radius: 999px;
        background: var(--bg-deep); padding: 2px 8px;
        font-family: 'JetBrains Mono', monospace; font-size: 0.64rem;
        color: var(--text-muted);
    }
    .eval-q-chip b { font-weight: 500; font-size: 0.7rem; }
    .eval-q-bad {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem; color: #E86A6A;
    }
    /* Breathing room so the report card doesn't crowd the last chat bubble. */
    .eval-anchor { margin-top: 18px; }
    </style>
    """,
    unsafe_allow_html=True,
)

SUGGESTED_QUESTIONS = [
    ("\U0001F4BC", "What services does PhotonX offer?"),
    ("\U0001F916", "What kind of AI work has PhotonX done?"),
    ("\U0001F91D", "How does PhotonX's engagement model work?"),
    ("\U0001F4C1", "What are some recent PhotonX projects?"),
]

# ---------------------------------------------------------------------------
# Resources (cached across reruns)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Warming up the copilot...")
def get_resources():
    try:
        return load_resources()
    except RuntimeError:
        # First run on a fresh deploy (Streamlit Cloud, HF Spaces, etc.) -- the
        # container has the repo's source_docs/ but no chroma_db/ yet, since
        # that's generated output, not something we commit. Build it once,
        # here, instead of requiring a manual `python ingest.py` step that's
        # easy to forget after every redeploy.
        import ingest
        with st.spinner("First run on this deployment: indexing PhotonX documents..."):
            ingest.run(source_dir=ingest.SOURCE_DIR)
        return load_resources()


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None
if "score_answers" not in st.session_state:
    # On by default; the sidebar toggle is the escape hatch when Groq starts
    # rate-limiting, since scoring adds one request per question.
    st.session_state.score_answers = True

if "eval_summary" not in st.session_state:
    # Holds the result of an in-app evaluation run for this browser session.
    # None means "nothing run here" -- render_evaluation() then falls back to a
    # committed eval_summary.json if one exists.
    st.session_state.eval_summary = None
if "run_eval" not in st.session_state:
    st.session_state.run_eval = None


@st.cache_data(show_spinner=False)
def get_eval_questions():
    """The benchmark set, read once. Cached because the sidebar needs its length
    on every rerun to bound the question-count input."""
    try:
        return load_dataset(DEFAULT_DATASET)
    except (OSError, SystemExit):
        return []


with st.sidebar:
    st.toggle(
        "Score every answer",
        key="score_answers",
        help="Scores all six RAG metrics on each reply with DeepEval. Each "
             "metric runs its own extraction and verdict calls, so this adds "
             "a few seconds per question and is the main consumer of your "
             "judge model's daily token allowance.",
    )


def render_eval_controls():
    """The Run-evaluation controls, rendered inside the report card in the main
    body rather than in the sidebar.

    Deliberately not in the sidebar: this page ships with the sidebar collapsed,
    so anything living only there is a step removed from being found. Putting the
    controls where the results appear means the button and its output are the
    same object on the page."""
    questions = get_eval_questions()
    if not questions:
        st.caption(
            f"`{DEFAULT_DATASET.name}` is missing or unreadable, so there is "
            "nothing to evaluate against."
        )
        return

    col_n, col_pause = st.columns(2)
    with col_n:
        n_eval = st.number_input(
            "Questions to run",
            min_value=1,
            max_value=len(questions),
            value=len(questions),
            key="eval_n",
            help="Start with 2 to confirm it works before spending a full run.",
        )
    with col_pause:
        pause = st.number_input(
            "Pause between (s)",
            min_value=0.0,
            max_value=15.0,
            value=0.0,
            step=0.5,
            key="eval_pause",
            help="Raise this if Groq starts returning rate-limit errors. Each "
                 "question costs one answer call plus a full DeepEval pass, "
                 "which is a dozen-plus small calls of its own.",
        )

    # ~60s per question: one answer generation plus a DeepEval pass, whose
    # six metrics each run their own extraction and verdict calls.
    est = int(n_eval) * 60 + int(n_eval) * pause
    st.caption(
        f"Answers and scores {int(n_eval)} of {len(questions)} benchmark "
        f"questions in this deployment. Roughly {est // 60}m {est % 60:.0f}s. "
        "Results stay in your session."
    )

    run_col, clear_col = st.columns(2)
    with run_col:
        if st.button("Run evaluation", use_container_width=True, type="primary"):
            # Record the request and rerun. The run itself happens earlier in
            # the script than this expander, so it cannot start until the next
            # pass -- and it needs to be there, above the report card, so the
            # progress log reads in order with the results it produces.
            st.session_state.run_eval = {"n": int(n_eval), "pause": float(pause)}
            st.rerun()
    with clear_col:
        if st.session_state.eval_summary:
            if st.button("Clear results", use_container_width=True):
                st.session_state.eval_summary = None
                st.rerun()


def queue_question(q: str):
    st.session_state.pending_query = q


def render_sources(sources: list[dict]):
    """One collapsed expander; opening it shows the excerpt each source
    contributed, so clicking actually surfaces the relevant document
    content instead of linking nowhere."""
    if not sources:
        return
    label = "Source" if len(sources) == 1 else "Sources"
    with st.expander(f"{label} ({len(sources)})"):
        parts = []
        for s in sources:
            heading = html.escape(s["label"][:80])
            excerpt = html.escape(s["excerpt"])
            parts.append(
                f'<div class="source-entry">'
                f'<span class="source-heading">{heading}</span>'
                f'<p class="source-excerpt">{excerpt}</p>'
                f"</div>"
            )
        st.markdown("".join(parts), unsafe_allow_html=True)


def _score_tone(value: float, direction: str) -> str:
    """Green/amber/red for one score. Any "lower is better" metric has its
    thresholds inverted rather than reusing the higher-is-better bands and
    colouring a good answer red."""
    if direction == "lower":
        return "score-good" if value <= 0.15 else ("score-mid" if value <= 0.30 else "score-low")
    return "score-good" if value >= 0.85 else ("score-mid" if value >= 0.70 else "score-low")


def render_answer_scores(scores: dict | None):
    """The RAG metrics for the one answer directly above, as computed by
    DeepEval in llm_metrics.score_answer.

    Two tiers on purpose: the chip strip is always visible so every answer
    carries its own numbers, and the expander holds DeepEval's stated reason
    for each one -- which is the part that makes a score arguable rather than
    something you either trust or don't.

    Renders whichever metrics actually scored, rather than a fixed six: a
    metric DeepEval could not complete is absent from `values` and is simply
    not drawn, instead of showing a zero it never measured."""
    if not scores:
        return
    if scores.get("error"):
        st.markdown(
            f'<p class="score-note">Answer scoring unavailable: '
            f'{html.escape(str(scores["error"]))}</p>',
            unsafe_allow_html=True,
        )
        return

    values = scores.get("scores") or {}
    reasons = scores.get("reasons") or {}
    if not values:
        return

    chips = []
    for key, label, direction, _needs_ref in METRICS:
        val = values.get(key)
        if val is None:
            continue
        arrow = ' <i>&darr;</i>' if direction == "lower" else ""
        chips.append(
            f'<span class="score-chip {_score_tone(val, direction)}">'
            f'{html.escape(label)} <b>{val:.2f}</b>{arrow}</span>'
        )
    if not chips:
        return

    st.markdown(
        '<div class="score-strip">' + "".join(chips) + "</div>",
        unsafe_allow_html=True,
    )

    with st.expander(f"How accurate is this? — DeepEval scores ({len(chips)} metrics)"):
        parts = []
        for key, label, direction, _needs_ref in METRICS:
            val = values.get(key)
            if val is None:
                continue
            # Every metric is defined on 0-1; clamp anyway so an out-of-range
            # value can't blow the bar past its track.
            pct = max(0.0, min(1.0, val)) * 100
            hint = "lower is better" if direction == "lower" else "higher is better"
            reason = reasons.get(key)
            parts.append(
                f'<div class="why-row">'
                f'<span class="why-label">{html.escape(label)}</span>'
                f'<span class="why-score">{val:.2f}</span>'
                f'<div class="why-bar"><div class="why-bar-fill" style="width:{pct:.1f}%"></div></div>'
                + (f'<p class="why-reason">{html.escape(reason)}</p>' if reason else "")
                + f'<p class="why-reason">({hint})</p>'
                f"</div>"
            )
        parts.append(
            '<div class="why-foot">'
            'Computed by <strong>DeepEval</strong> from this answer and the '
            f'context retrieved for it, using {html.escape(JUDGE_MODEL)} as the '
            'evaluation model.<br/>'
            'Each score is an algorithm, not an opinion: DeepEval extracts the '
            'individual claims, statements or entities in play, asks the model '
            'one yes/no verdict per item, and does the arithmetic itself.<br/>'
            'Context Precision, Context Recall, Context Entity Recall and Answer '
            'Correctness are defined against a reference answer, which a live '
            'question does not have - one is drafted from the retrieved context '
            'alone, without ever seeing the answer above, and used in its place. '
            'For the human-written-reference version, see the system evaluation '
            'at the bottom of the page.'
            "</div>"
        )
        st.markdown("".join(parts), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# System-level evaluation report card
# ---------------------------------------------------------------------------
EVAL_SUMMARY_PATH = Path(__file__).parent / "eval_summary.json"


def _summary_mtime() -> float:
    """0.0 when the file is absent. Used as the cache key below."""
    try:
        return EVAL_SUMMARY_PATH.stat().st_mtime
    except OSError:
        return 0.0


@st.cache_data(show_spinner=False)
def _read_eval_summary(mtime: float):
    """Aggregate scores from the last `python evaluate.py` run.

    Read from a file rather than computed here on purpose: a full run is a
    retrieval pass, an answer and a judge call per question, which is minutes
    of wall-clock and dozens of Groq requests -- too slow for a page load and
    enough to hit a rate limit. The app reads, evaluate.py writes.

    `mtime` is not used in the body -- it is the cache key. Caching on the
    filename alone meant a "file not found" result stuck for the life of the
    process, so an app that started before the summary existed would keep
    showing the empty state even after the file appeared. Keying on mtime
    invalidates the moment evaluate.py rewrites it. The argument must not start
    with an underscore: Streamlit excludes underscore-prefixed args from the
    hash, which would silently restore the old behaviour."""
    try:
        with open(EVAL_SUMMARY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def load_eval_summary():
    """An in-app run for this session wins over the committed file: if you just
    pressed the button, the numbers on screen should be the ones you asked
    for."""
    return st.session_state.get("eval_summary") or _read_eval_summary(_summary_mtime())


def run_evaluation_in_app(request: dict):
    """Answer and score the benchmark set inside this deployment.

    Runs in the main script body rather than a background thread on purpose:
    Streamlit's rerun model gives a thread no safe way to draw, and the script
    is allowed to run as long as the browser stays connected. The cost is that
    the page is busy until it finishes, which is why the progress UI reports
    every question as it lands instead of showing one opaque spinner.

    Returns True if results were stored, so the caller can rerun and let the
    sidebar redraw with them present.
    """
    questions = get_eval_questions()[: request["n"]]
    if not questions:
        st.warning("Nothing to evaluate.")
        return False

    pause = request["pause"]
    rows = []
    with st.status(
        f"Evaluating {len(questions)} questions in this deployment...",
        expanded=True,
    ) as status:
        bar = st.progress(0.0)
        for i, item in enumerate(questions, 1):
            st.write(f"**{i}/{len(questions)}** — {item['question']}")
            try:
                resources = get_resources()
            except Exception as e:
                status.update(label="Evaluation could not start", state="error")
                st.error(f"Could not load the index: {e}")
                return False
            row = run_one(resources, item)
            rows.append(row)

            if row["error"]:
                st.write(f"&nbsp;&nbsp;↳ :red[{row['error']}]")
            else:
                got = ", ".join(
                    f"{lbl.split()[-1]} {row['scores'][k]:.2f}"
                    for k, lbl, _d, _nr in METRICS
                    if k in row["scores"]
                )
                st.write(f"&nbsp;&nbsp;↳ :green[{got}]")
            bar.progress(i / len(questions))

            if pause and i < len(questions):
                time.sleep(pause)

        summary = build_summary(rows, DEFAULT_DATASET.name)
        summary["computed_in_app"] = True
        st.session_state.eval_summary = summary

        n_failed = sum(1 for r in rows if r["error"])
        if n_failed == len(rows):
            status.update(label="Every question failed to score", state="error")
        elif n_failed:
            status.update(
                label=f"Done — {len(rows) - n_failed} of {len(rows)} scored",
                state="complete",
            )
        else:
            status.update(label=f"Done — all {len(rows)} questions scored", state="complete")

    return True


def render_evaluation():
    """The whole system's numbers over the fixed question set in
    eval_dataset.json -- distinct from the per-answer chips above, which score
    only the reply they sit under.

    Rendered once at the bottom rather than repeated per message: these are
    corpus-level aggregates, and repeating them under each answer would read as
    a score for that answer."""
    summary = load_eval_summary()
    metrics = [m for m in (summary or {}).get("metrics", []) if m.get("score") is not None]

    st.markdown('<div class="eval-anchor"></div>', unsafe_allow_html=True)

    if not metrics:
        with st.expander("How good is this RAG system overall? — run the evaluation"):
            st.markdown(
                '<p class="eval-empty">No evaluation has been recorded yet. Press '
                '<strong>Run evaluation</strong> below to answer and score every '
                f'question in <code>{html.escape(DEFAULT_DATASET.name)}</code> with '
                'the real pipeline, right here in this deployment.<br/><br/>'
                'Results stay in your session. To make them the default every visitor '
                'sees, download the summary afterwards and commit it as '
                '<code>eval_summary.json</code> &mdash; or run '
                '<code>python evaluate.py</code> offline, which writes that file '
                'directly.</p>',
                unsafe_allow_html=True,
            )
            render_eval_controls()
        return

    n = summary.get("n_questions", 0)
    scored = summary.get("n_scored", n)
    with st.expander(f"How good is this RAG system overall? — {n} benchmark questions"):
        parts = []
        for m in metrics:
            score = float(m["score"])
            # Every metric is 0-1; clamp anyway so an out-of-range value can't
            # blow the bar past its track.
            pct = max(0.0, min(1.0, score)) * 100
            direction = "lower is better" if m.get("direction") == "lower" else "higher is better"
            n_scored = m.get("n_scored")
            detail = direction
            if n_scored is not None and n_scored != scored:
                # A mean over fewer questions than were run is worth showing
                # rather than hiding behind a confident-looking average.
                detail += f" &middot; n={n_scored}"
            parts.append(
                f'<div class="eval-row">'
                f'<span class="eval-label">{html.escape(str(m["label"]))}'
                f'<span class="eval-dir">{detail}</span></span>'
                f'<span class="eval-score">{score:.3f}</span>'
                f'<div class="eval-bar"><div class="eval-bar-fill" style="width:{pct:.1f}%"></div></div>'
                f"</div>"
            )

        in_app = bool(summary.get("computed_in_app"))
        foot = [
            f'{scored} of {n} questions scored from '
            f'<code>{html.escape(str(summary.get("dataset", "eval_dataset.json")))}</code> '
            f'on {html.escape(str(summary.get("generated_at", "?")))}',
            # Where the numbers came from is not a detail: a session run is
            # visible only to you, a committed one is what every visitor sees.
            'Computed in this deployment just now &mdash; this session only'
            if in_app
            else 'Read from the committed <code>eval_summary.json</code>',
            f'Answers: {html.escape(str(summary.get("answer_model", "?")))} '
            f'&middot; Scored by DeepEval using '
            f'{html.escape(str(summary.get("judge_model", "?")))}',
            'Every question here has a known-correct reference answer, so Context '
            'Precision, Context Recall, Context Entity Recall and Answer '
            'Correctness are measured against a human-written reference rather '
            'than a synthesized one &mdash; which a live question cannot offer.',
            'The chips under each reply score that reply alone, against a '
            'reference synthesized from its own retrieved context. These numbers '
            'are the whole system measured against fixed questions with '
            'human-written answers, which is what makes them comparable across '
            'changes to chunking, reranking or prompting.',
        ]
        parts.append('<div class="eval-foot">' + "<br/>".join(foot) + "</div>")
        st.markdown("".join(parts), unsafe_allow_html=True)

        if in_app:
            st.download_button(
                "Download eval_summary.json",
                data=json.dumps(summary, indent=2, ensure_ascii=False),
                file_name="eval_summary.json",
                mime="application/json",
                help="Commit this to the repo to make these the numbers every "
                     "visitor sees, instead of only this session.",
            )

        st.divider()
        render_eval_controls()

        rows = summary.get("questions") or []
        if rows:
            st.markdown('<p class="eval-perq">Per question</p>', unsafe_allow_html=True)
            for r in rows:
                scores = r.get("scores") or {}
                if r.get("error"):
                    badge = f'<span class="eval-q-bad">{html.escape(str(r["error"]))}</span>'
                else:
                    badge = " ".join(
                        f'<span class="eval-q-chip {_score_tone(scores[k], d)}">'
                        f'{html.escape(lbl.split()[-1])} <b>{scores[k]:.2f}</b></span>'
                        for k, lbl, d, _nr in METRICS
                        if k in scores
                    )
                st.markdown(
                    f'<div class="eval-q">'
                    f'<span class="eval-q-text">{html.escape(str(r.get("question", "")))}</span>'
                    f'<div class="eval-q-chips">{badge}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------------------
# Hero (only before the first message)
# ---------------------------------------------------------------------------
if not st.session_state.messages:
    st.markdown(
        f"""
        <div class="hero-wrap">
            <div class="hero-mark"><img src="data:image/png;base64,{LOGO_B64}" alt="PhotonX" /></div>
            <p class="hero-title">PhotonX Copilot</p>
            <p class="hero-sub">Ask anything about our services, projects, or how we work \u2014 answered straight from the source.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(2)
    for i, (icon, question) in enumerate(SUGGESTED_QUESTIONS):
        with cols[i % 2]:
            st.button(
                f"{icon}  {question}",
                key=f"suggest_{i}",
                on_click=queue_question,
                args=(question,),
            )

# ---------------------------------------------------------------------------
# Render existing conversation
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    avatar = str(LOGO_PATH) if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        render_sources(msg.get("sources", []))
        render_answer_scores(msg.get("scores"))

# ---------------------------------------------------------------------------
# Input (typed or from a suggestion click)
# ---------------------------------------------------------------------------
typed_query = st.chat_input("Ask PhotonX Copilot...")
query = st.session_state.pending_query or typed_query
st.session_state.pending_query = None

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant", avatar=str(LOGO_PATH)):
        try:
            resources = get_resources()
            chunks, stream = ask(resources, query, st.session_state.messages[:-1])
            full_answer = st.write_stream(stream)

            seen_keys, sources = set(), []
            for c in chunks:
                meta = c["metadata"]
                heading = meta.get("h2") or meta.get("h1") or ""
                key = (meta.get("source"), heading)
                if key not in seen_keys:
                    seen_keys.add(key)
                    label = meta.get("title", "Document")
                    if heading:
                        label += f" — {heading}"
                    excerpt = c["text"].strip().replace("\n", " ")
                    if len(excerpt) > 280:
                        excerpt = excerpt[:280].rsplit(" ", 1)[0] + "…"
                    sources.append({"label": label, "excerpt": excerpt})

            render_sources(sources)

            # Scored only after the answer is on screen, so DeepEval's calls
            # never delay the reply itself. Stored on the message so
            # Streamlit's reruns replay it instead of re-billing.
            scores = None
            if st.session_state.get("score_answers", True) and chunks:
                with st.spinner("Scoring this answer with DeepEval..."):
                    scores = score_answer(
                        question=query,
                        contexts=[c["text"] for c in chunks],
                        answer=full_answer,
                    ).as_dict()
                render_answer_scores(scores)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": full_answer,
                    "sources": sources,
                    "scores": scores,
                }
            )
        except RuntimeError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Something went wrong: {e}")


# Requested from the sidebar. Handled here, after the chat, so the progress log
# and the report card it produces appear in reading order at the bottom of the
# page rather than above the conversation.
if st.session_state.run_eval:
    request = st.session_state.run_eval
    st.session_state.run_eval = None  # cleared first: a rerun mid-run must not restart it
    if run_evaluation_in_app(request):
        # The sidebar was already drawn this pass, before any results existed,
        # so its "Clear results" button would not appear until the user next
        # interacted with something. Rerun to redraw it. Nothing is lost: the
        # transient progress log duplicates summary["questions"], which the
        # report card renders per question, errors included.
        st.rerun()

# Last thing on the page, so it sits under the newest answer and is present on
# every screen -- landing page and mid-conversation alike. st.chat_input is
# pinned to the viewport bottom by Streamlit regardless of call order, so this
# does not displace the input box.
render_evaluation()
