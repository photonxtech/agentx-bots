"""
PhotonX Copilot - Streamlit Interface
A polished, chat-first landing experience over the hybrid RAG engine in rag_engine.py.
"""

import base64
import html
from pathlib import Path

import streamlit as st
from rag_engine import load_resources, ask

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

    #MainMenu, footer, header { visibility: hidden; }
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

            st.session_state.messages.append(
                {"role": "assistant", "content": full_answer, "sources": sources}
            )
        except RuntimeError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Something went wrong: {e}")
