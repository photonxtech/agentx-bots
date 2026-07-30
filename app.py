import os
import streamlit as st
from dotenv import load_dotenv
from rag_utils import (
    process_pdf,
    build_hybrid_retriever,
    get_answer,
    load_sessions,
    create_session,
    update_session,
    add_pdf_to_session,
    append_chat_turn,
    delete_session,
    DATA_DIR,
)

load_dotenv()

st.set_page_config(page_title="WeNext FAQ Assistant", page_icon="📄", layout="centered")

# Inject custom CSS
st.markdown("""
<style>
[data-testid="stChatInput"] {
    border: 1px solid rgba(150, 150, 150, 0.4) !important;
    border-radius: 10px !important;
    box-shadow: none !important;
    transition: box-shadow 0.2s ease, border-color 0.2s ease;
    background-color: #FFFFFF !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: rgba(160, 160, 160, 0.6) !important;
    box-shadow: 0 0 6px rgba(160, 160, 160, 0.35) !important;
}
[data-testid="stChatInput"] textarea {
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    background: transparent !important;
}
[data-testid="stChatMessageAvatarAssistant"] {
    background-color: #FFFFFF !important;
    border: 2px solid #7C3AED !important;
}
[data-testid="stChatMessageAvatarAssistant"] svg,
[data-testid="stChatMessageAvatarAssistant"] * {
    color: #7C3AED !important;
    fill: #7C3AED !important;
}
[data-testid="stChatMessageAvatarUser"] {
    background-color: #FFFFFF !important;
    border: 2px solid #000000 !important;
}
[data-testid="stChatMessageAvatarUser"] svg,
[data-testid="stChatMessageAvatarUser"] * {
    color: #000000 !important;
    fill: #000000 !important;
}
[data-testid="stSidebar"] button {
    text-align: left !important;
}
/* Chip showing one attached PDF in the sidebar */
.doc-chip {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    border-radius: 8px;
    background: #F3E8FF;
    border: 1px solid #C4B5FD;
    margin-bottom: 6px;
    font-size: 13px;
    color: #4C1D95;
}
/* Small muted caption under the title */
.subtle-caption {
    color: #6B7280;
    font-size: 14px;
    margin-top: -8px;
}
</style>
""", unsafe_allow_html=True)

# --- Session state init ---
if "sessions" not in st.session_state:
    st.session_state.sessions = load_sessions()

if "current_session_id" not in st.session_state:
    if st.session_state.sessions:
        latest_id = sorted(
            st.session_state.sessions.items(),
            key=lambda kv: kv[1].get("created_at", ""),
            reverse=True
        )[0][0]
        st.session_state.current_session_id = latest_id
    else:
        st.session_state.current_session_id = create_session()
        st.session_state.sessions = load_sessions()

if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "retriever_session_id" not in st.session_state:
    st.session_state.retriever_session_id = None


def get_current_session():
    return st.session_state.sessions.get(st.session_state.current_session_id, {})


def switch_session(session_id):
    st.session_state.current_session_id = session_id
    st.session_state.retriever = None
    st.session_state.retriever_session_id = None


def start_new_chat():
    new_id = create_session()
    st.session_state.sessions = load_sessions()
    switch_session(new_id)


def process_uploaded_files(uploaded_files, session_id):
    """Saves + indexes any files not already attached to this session. Returns (n_files, n_pages, n_chunks)."""
    current = st.session_state.sessions.get(session_id, {})
    already_attached = set(current.get("pdf_names", []))

    new_files = [f for f in uploaded_files if f.name not in already_attached]
    if not new_files:
        return 0, 0, 0

    session_dir = DATA_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    total_pages, total_chunks = 0, 0
    for uploaded_file in new_files:
        save_path = session_dir / uploaded_file.name
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        num_pages, num_chunks = process_pdf(str(save_path), session_id)
        add_pdf_to_session(session_id, uploaded_file.name)
        total_pages += num_pages
        total_chunks += num_chunks

    st.session_state.sessions = load_sessions()
    # Force retriever rebuild since the index just changed
    st.session_state.retriever = build_hybrid_retriever(session_id)
    st.session_state.retriever_session_id = session_id

    return len(new_files), total_pages, total_chunks


# =========================== SIDEBAR ===========================
with st.sidebar:
    st.header("📄 WeNext FAQ Assistant")

    if st.button("➕ New Chat", use_container_width=True):
        start_new_chat()
        st.rerun()

    st.divider()

    current = get_current_session()
    pdf_names = current.get("pdf_names", [])

    st.subheader("Documents")

    if pdf_names:
        for name in pdf_names:
            st.markdown(f'<div class="doc-chip">📎 {name}</div>', unsafe_allow_html=True)

    # Always-available uploader — supports adding more PDFs to the same chat,
    # not just the first one. Duplicate filenames are skipped automatically.
    uploaded_files = st.file_uploader(
        "Add PDF(s)" if pdf_names else "Upload PDF(s) to start",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key=f"uploader_{st.session_state.current_session_id}",
    )

    if uploaded_files:
        with st.spinner("Processing PDF(s)... (splitting, embedding, indexing)"):
            n_files, n_pages, n_chunks = process_uploaded_files(
                uploaded_files, st.session_state.current_session_id
            )
        if n_files:
            st.success(f"Added {n_files} file(s): {n_pages} pages, {n_chunks} chunks.")
            st.rerun()

    st.divider()
    st.subheader("History")

    sessions_sorted = sorted(
        st.session_state.sessions.items(),
        key=lambda kv: kv[1].get("created_at", ""),
        reverse=True
    )

    if not sessions_sorted:
        st.caption("No past chats yet.")

    for sid, sdata in sessions_sorted:
        names = sdata.get("pdf_names", [])
        if not names:
            label = "New chat (no PDF yet)"
        elif len(names) == 1:
            label = names[0]
        else:
            label = f"{names[0]} +{len(names) - 1} more"
        is_active = sid == st.session_state.current_session_id

        col_select, col_delete = st.columns([5, 1])
        with col_select:
            prefix = "🟣 " if is_active else "💬 "
            if st.button(prefix + label, key=f"session_{sid}", use_container_width=True):
                if not is_active:
                    switch_session(sid)
                    st.rerun()
        with col_delete:
            if st.button("🗑️", key=f"del_{sid}"):
                was_active = sid == st.session_state.current_session_id
                delete_session(sid)
                st.session_state.sessions = load_sessions()
                if was_active:
                    if st.session_state.sessions:
                        remaining = sorted(
                            st.session_state.sessions.items(),
                            key=lambda kv: kv[1].get("created_at", ""),
                            reverse=True
                        )
                        switch_session(remaining[0][0])
                    else:
                        start_new_chat()
                st.rerun()

# =========================== MAIN AREA ===========================
st.title("WeNext FAQ Assistant")
st.markdown('<p class="subtle-caption">Upload one or more PDFs, then ask questions across all of them — follow-ups included.</p>', unsafe_allow_html=True)

current = get_current_session()
pdf_names = current.get("pdf_names", [])
chat_history = current.get("chat_history", [])


def render_sources(sources):
    with st.expander("Sources"):
        for i, src in enumerate(sources, start=1):
            # PyPDFLoader's page numbers are 0-indexed internally; +1 here converts
            # to the 1-indexed page numbers a human sees when reading the actual PDF.
            page_num = src["page"] + 1 if isinstance(src["page"], int) else src["page"]
            doc_label = src.get("source", "unknown")
            st.markdown(f"**Source {i} — {doc_label}, page {page_num}**")
            st.caption(src["text"][:300] + ("..." if len(src["text"]) > 300 else ""))
            st.divider()


for turn in chat_history:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        st.markdown(turn["answer"])
        if "sources" in turn:
            render_sources(turn["sources"])

if pdf_names:
    if st.session_state.retriever_session_id != st.session_state.current_session_id:
        with st.spinner("Loading document index..."):
            st.session_state.retriever = build_hybrid_retriever(st.session_state.current_session_id)
            st.session_state.retriever_session_id = st.session_state.current_session_id

    question = st.chat_input("Ask a question about your document(s)...")
    if question:
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, sources, token_usage = get_answer(
                    st.session_state.retriever,
                    question,
                    chat_history
                )
                st.markdown(answer)
                st.caption(
                    f"Tokens — prompt: {token_usage.get('prompt_tokens')}, "
                    f"completion: {token_usage.get('completion_tokens')}, "
                    f"total: {token_usage.get('total_tokens')}"
                )
                render_sources(sources)

        append_chat_turn(st.session_state.current_session_id, question, answer, sources)
        st.session_state.sessions = load_sessions()
        st.rerun()
else:
    st.info("Upload one or more PDFs from the sidebar to get started.")
