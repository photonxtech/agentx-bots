"""Multi-RAG — Streamlit frontend.

Flow:  each chat owns ONE file  ->  upload ingests (OCR + vision, cached)
       ->  chunk + tag with the chat id  ->  embed + index (persisted)
       ->  ask a question  ->  rewrite follow-ups
       ->  hybrid retrieve top-k *within this chat only*  ->  Groq answers.

Per-chat isolation: every chunk is tagged with the id of the chat it was
uploaded into, and retrieval is filtered to the active chat. A chat can never
see — or answer from — another chat's document.
"""

from __future__ import annotations

import os

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

import config
import chat_store
from rag import chunking, generator, ingestion
from rag.vectorstore import VectorStore

load_dotenv()

st.set_page_config(page_title="Multi-RAG", page_icon="📚", layout="wide")

UPLOAD_TYPES = ["pdf", "docx", "pptx", "txt", "md",
                "png", "jpg", "jpeg", "bmp", "tiff", "webp"]

# Read defaults from central configuration (UI settings pane removed)
model = config.DEFAULT_MODEL
temperature = config.DEFAULT_TEMPERATURE
top_k = config.TOP_K  # base; scales up automatically with number of indexed sources


# --------------------------------------------------------------------------- #
# Session state — the index is loaded from disk once per session. It holds
# every chat's chunks in one place, tagged by chat id; the UI only ever shows
# and queries the slice belonging to the active chat.
# --------------------------------------------------------------------------- #
if "store" not in st.session_state:
    st.session_state.store = VectorStore.load(config.INDEX_DIR)

if "chats" not in st.session_state:
    st.session_state.chats = chat_store.load_chats()

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None

# Per-chat uploader "nonce": bumped when a chat's file is removed so the
# file_uploader widget is re-created fresh (doesn't re-index the deleted file).
if "upload_nonce" not in st.session_state:
    st.session_state.upload_nonce = {}

# If there is no active chat but chats exist, default to the most recently updated one.
if not st.session_state.current_chat_id and st.session_state.chats:
    sorted_chats = sorted(
        st.session_state.chats.values(), key=lambda c: c["updated_at"], reverse=True
    )
    st.session_state.current_chat_id = sorted_chats[0]["id"]

# Guarantee an active chat always exists, so a file upload has a home.
if not st.session_state.current_chat_id:
    st.session_state.current_chat_id = chat_store.create_chat(
        st.session_state.chats, title="New Chat"
    )

current_chat = st.session_state.chats.get(st.session_state.current_chat_id)
messages = current_chat["messages"] if current_chat else []

cid = st.session_state.current_chat_id
store = st.session_state.store


def chat_file_name(store, chat_id: str) -> str | None:
    """The single file indexed for a chat, or None."""
    if not store:
        return None
    srcs = store.sources_for_chat(chat_id)
    return srcs[0] if srcs else None


def index_file_for_chat(upload, chat_id: str) -> None:
    """Ingest + chunk + tag + index one uploaded file into the given chat."""
    store = st.session_state.store or VectorStore()
    with st.status(f"Indexing **{upload.name}** …", expanded=True) as status:
        try:
            st.write(f"📥 Reading **{upload.name}** …")
            docs = ingestion.ingest_cached(upload.getvalue(), upload.name)
            chunks = chunking.chunk_documents(docs)
            # Tag every chunk with the owning chat so retrieval can be scoped.
            for c in chunks:
                c.meta["chat_id"] = chat_id
            st.write(f"✓ {len(chunks)} chunk(s) extracted")
        except Exception as e:
            status.update(label=f"⚠️ Could not read {upload.name}: {e}", state="error")
            return

        if not chunks:
            status.update(label="No extractable content found.", state="error")
            return

        st.write("🧠 Embedding + indexing …")
        store.add(chunks)
        store.save(config.INDEX_DIR)
        st.session_state.store = store
        status.update(
            label=f"✅ Indexed {upload.name} — {store.size_for_chat(chat_id)} "
                  f"chunk(s) for this chat. Saved to disk.",
            state="complete",
        )


def render_sources(sources: list[dict]) -> None:
    """Show retrieved chunks + the actual extracted images they refer to."""
    for src in sources:
        st.markdown(src["md"])
        for path in src.get("images", []):
            if os.path.exists(path):
                st.image(path, width=350)


# --------------------------------------------------------------------------- #
# Sidebar — chats + this chat's file
# --------------------------------------------------------------------------- #
with st.sidebar:
    if st.button("➕ New Chat", use_container_width=True):
        new_id = chat_store.create_chat(st.session_state.chats, title="New Chat")
        st.session_state.current_chat_id = new_id
        st.rerun()

    st.divider()

    # Recency Grouped Chat History
    grouped = chat_store.group_chats_by_recency(st.session_state.chats)
    if grouped:
        st.subheader("💬 Chat History")
        for label, chats_in_group in grouped.items():
            st.caption(label)
            for chat in chats_in_group:
                col_open, col_del = st.columns([0.85, 0.15])
                is_active = chat["id"] == st.session_state.current_chat_id
                btn_label = ("🟢 " if is_active else "💬 ") + chat["title"]

                # Load chat session on click
                if col_open.button(btn_label, key=f"open_{chat['id']}", use_container_width=True):
                    st.session_state.current_chat_id = chat["id"]
                    st.rerun()

                # Delete chat session on click — also drop its indexed file.
                if col_del.button("🗑️", key=f"del_{chat['id']}"):
                    if st.session_state.store:
                        st.session_state.store.remove_chat(chat["id"])
                        st.session_state.store.save(config.INDEX_DIR)
                    chat_store.delete_chat(st.session_state.chats, chat["id"])
                    if st.session_state.current_chat_id == chat["id"]:
                        st.session_state.current_chat_id = None
                    st.rerun()
        st.divider()

    # This chat's document (per-chat, not global).
    store = st.session_state.store
    this_file = chat_file_name(store, cid)
    if this_file:
        n = store.size_for_chat(cid)
        st.success(f"This chat's file:\n\n📄 `{this_file}`")
        st.caption(f"{n} chunk(s) indexed ({store.backend})")
    else:
        st.info("This chat has no file yet — upload one on the right.")

    if st.button("🗑️ Reset everything (all chats & files)"):
        if st.session_state.store:
            st.session_state.store.clear()
        else:
            # No live store; make one just to clear persisted data.
            try:
                VectorStore().clear()
            except Exception:
                pass
        st.session_state.store = None
        st.session_state.chats = {}
        st.session_state.current_chat_id = None
        st.session_state.upload_nonce = {}
        chat_store.save_chats({})
        st.rerun()


# --------------------------------------------------------------------------- #
# Main — this chat's single file (upload OR show + remove)
# --------------------------------------------------------------------------- #
st.title("📚 Multi-RAG")
st.caption("Each chat answers only from its own file — PDF, Word, PowerPoint, "
           "text, or image (OCR + vision) — grounded by Groq.")

store = st.session_state.store
this_file = chat_file_name(store, cid)

if this_file:
    col_a, col_b = st.columns([0.7, 0.3])
    with col_a:
        st.markdown(f"**📄 This chat's document:** `{this_file}`")
    with col_b:
        if st.button("🗑 Remove file", use_container_width=True):
            store.remove_source(this_file, chat_id=cid)
            store.save(config.INDEX_DIR)
            # Force a fresh uploader widget so the removed file isn't re-added.
            st.session_state.upload_nonce[cid] = st.session_state.upload_nonce.get(cid, 0) + 1
            st.rerun()
    st.caption("Remove this file to upload a different one (one file per chat).")
else:
    nonce = st.session_state.upload_nonce.get(cid, 0)
    upload = st.file_uploader(
        "📎 Upload ONE file for this chat — PDF, Word (.docx), PowerPoint "
        "(.pptx), text (.txt/.md), or an image (.png/.jpg/…).",
        type=UPLOAD_TYPES,
        accept_multiple_files=False,
        key=f"uploader_{cid}_{nonce}",
    )
    if upload is not None:
        index_file_for_chat(upload, cid)
        st.rerun()


# --------------------------------------------------------------------------- #
# Chat
# --------------------------------------------------------------------------- #
st.divider()

# Collect only user prompts and match them to their anchor ID
user_questions = [
    (msg["content"], f"msg-{idx}")
    for idx, msg in enumerate(messages)
    if msg["role"] == "user"
]

if user_questions:
    col_title, col_hist = st.columns([0.75, 0.25])
    with col_title:
        st.subheader("💬 Ask")
    with col_hist:
        with st.popover("🕒 Chat history", use_container_width=True):
            st.markdown('<p style="font-size:0.8rem;color:#888;margin:0 0 8px 0;">Click to scroll to question</p>', unsafe_allow_html=True)
            for q_text, anchor in user_questions:
                label = q_text[:50] + "..." if len(q_text) > 50 else q_text
                if st.button(label, key=f"hist_{anchor}", use_container_width=True):
                    st.session_state.scroll_to_anchor = anchor
                    st.rerun()
else:
    st.subheader("💬 Ask")

for global_idx, msg in enumerate(messages):
    anchor_id = f"msg-{global_idx}"

    # Place the invisible scroll target anchor in the DOM
    st.markdown(
        f'<div id="{anchor_id}" style="scroll-margin-top:80px;"></div>',
        unsafe_allow_html=True,
    )
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("metrics"):
            m = msg["metrics"]
            st.markdown(
                f'<p style="font-size:0.8rem;color:#888;margin-top:4px;margin-bottom:8px;">'
                f'⚡ Rewrite: {m["rewrite_ms"]}ms | '
                f'Embed & Search: {m["search_ms"]}ms | '
                f'TTFT: {m["ttft_ms"]}ms | '
                f'Generation: {m["generation_ms"]}ms</p>',
                unsafe_allow_html=True
            )
        if msg.get("sources"):
            with st.expander("Retrieved context"):
                render_sources(msg["sources"])

question = st.chat_input("Ask a question about this chat's document…")

if question:
    store = st.session_state.store
    # Block unless THIS chat has its own file — never fall back to another chat.
    if not store or store.size_for_chat(cid) == 0:
        st.error("Upload a file for this chat first — each chat answers only "
                 "from its own document.")
        st.stop()

    # History BEFORE this question
    history = list(messages)

    # Auto-generate title if this is the first message
    if not messages:
        title_text = question[:40] + "..." if len(question) > 40 else question
        current_chat["title"] = title_text

    # Append user question
    messages.append({"role": "user", "content": question})
    chat_store.update_chat_messages(st.session_state.chats, cid, messages)

    with st.chat_message("user"):
        st.markdown(question)

    import time

    # 1. Rewrite
    t_start_rewrite = time.perf_counter()
    search_q = generator.rewrite_query(question, history)
    t_rewrite = time.perf_counter() - t_start_rewrite

    if search_q.strip().lower() != question.strip().lower():
        st.caption(f"🔎 Searching for: _{search_q}_")

    # 2. Search (Embed + Retrieve) — scoped to THIS chat only.
    # Scale top_k with the number of sources in this chat (usually 1).
    num_sources = len(store.sources_for_chat(cid)) or 1
    effective_top_k = max(top_k, min(num_sources * 2, 30))

    t_start_search = time.perf_counter()
    hits = store.search(search_q, top_k=effective_top_k * 3, chat_id=cid)  # wider pool
    t_search = time.perf_counter() - t_start_search

    # --- Source-type & source-name filtering (within this chat's hits) ---
    import re as _re
    _q_lower = (question + " " + search_q).lower()

    _type_keywords = {
        "pptx":  ["ppt", "powerpoint", "presentation", "slide", "slides"],
        "pdf":   ["pdf"],
        "docx":  ["word", "docx", "document"],
        "image": ["image", "photo", "picture", "png", "jpg", "jpeg"],
        "txt":   ["text file", "txt", "markdown", "md"],
    }
    _wanted_kind = None
    for kind, kws in _type_keywords.items():
        if any(kw in _q_lower for kw in kws):
            _wanted_kind = kind
            break

    # Filename keyword match against THIS chat's sources only.
    _wanted_source = None
    if not _wanted_kind:
        q_words = set(_re.split(r"\W+", _q_lower))
        for src in store.sources_for_chat(cid):
            src_words = set(_re.split(r"[-_.\s]+", src.lower()))
            meaningful = {w for w in (src_words & q_words) if len(w) > 3}
            if len(meaningful) >= 2:
                _wanted_source = src
                break

    if _wanted_kind:
        _filtered = [(d, s) for d, s in hits if d.kind == _wanted_kind]
        hits = _filtered[:effective_top_k] if _filtered else hits[:effective_top_k]
    elif _wanted_source:
        _filtered = [(d, s) for d, s in hits if d.source == _wanted_source]
        hits = _filtered[:effective_top_k] if _filtered else hits[:effective_top_k]
    else:
        hits = hits[:effective_top_k]

    # Retrieval confidence: top hybrid score (0..1) → percentage
    top_score = hits[0][1] if hits else 0.0
    confidence_pct = int(round(min(top_score, 1.0) * 100))

    sources = []
    for i, (d, s) in enumerate(hits, 1):
        sources.append({
            "md": f"**[{i}]** `{d.source}` ({d.kind}, score {s:.2f})\n\n> {d.text[:400]}",
            "images": [p for p in (d.meta.get("images") or []) if p],
        })

    # Generate (streamed)
    with st.chat_message("assistant"):
        t_start_gen = time.perf_counter()
        metrics_container = {"ttft": None}

        def generator_wrapper():
            g = generator.answer(search_q, hits, model=model, temperature=temperature)
            for chunk in g:
                if metrics_container["ttft"] is None:
                    metrics_container["ttft"] = time.perf_counter() - t_start_gen
                yield chunk

        try:
            full = st.write_stream(generator_wrapper())
        except Exception as e:
            full = f"⚠️ Generation failed: {e}"
            st.error(full)

        t_generation = time.perf_counter() - t_start_gen
        ttft = metrics_container["ttft"]
        if ttft is None:
            ttft = t_generation

        metrics = {
            "rewrite_ms": int(t_rewrite * 1000),
            "search_ms": int(t_search * 1000),
            "ttft_ms": int(ttft * 1000),
            "generation_ms": int(t_generation * 1000),
            "confidence_pct": confidence_pct,
        }

        # Display metrics for current response immediately in UI
        st.markdown(
            f'<p style="font-size:0.8rem;color:#888;margin-top:4px;margin-bottom:8px;">'
            f'⚡ Rewrite: {metrics["rewrite_ms"]}ms | '
            f'Embed & Search: {metrics["search_ms"]}ms | '
            f'TTFT: {metrics["ttft_ms"]}ms | '
            f'Generation: {metrics["generation_ms"]}ms</p>',
            unsafe_allow_html=True
        )

        # Detect if it is an "I don't know" response
        is_dont_know = "I don't know" in full or "don't know" in full.lower()

        if not is_dont_know:
            with st.expander("Retrieved context"):
                render_sources(sources)

    # Append assistant response (omit sources if answer was not found)
    saved_sources = [] if is_dont_know else sources
    messages.append({
        "role": "assistant",
        "content": full,
        "sources": saved_sources,
        "metrics": metrics
    })
    chat_store.update_chat_messages(st.session_state.chats, cid, messages)
    st.rerun()

# --------------------------------------------------------------------------- #
# Scroll execution (placed at the end so it executes after DOM painting)
# --------------------------------------------------------------------------- #
target_anchor = st.session_state.pop("scroll_to_anchor", None)
if target_anchor:
    components.html(
        f"""
        <script>
        (function() {{
            function doScroll() {{
                var el = window.parent.document.getElementById("{target_anchor}");
                if (el) {{
                    el.scrollIntoView({{ behavior: "smooth", block: "start" }});
                }} else {{
                    setTimeout(doScroll, 150);
                }}
            }}
            setTimeout(doScroll, 400);
        }})();
        </script>
        """,
        height=0,
    )
