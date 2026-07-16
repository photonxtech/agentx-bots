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
    append_chat_turn,
    delete_session,
    DATA_DIR,
)

# Load env vars (GROQ_API_KEY) from .env
load_dotenv()

# Configure page title, icon, and layout
st.set_page_config(page_title="WeNext FAQ Assistant", page_icon="📄", layout="centered")

# Inject custom CSS for chat input, avatars, and sidebar buttons
st.markdown("""
<style>
/* Chat input outer container — the only visible box */
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

/* Inner textarea — no border, no background, no shadow of its own */
[data-testid="stChatInput"] textarea {
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    background: transparent !important;
}

/* AI (assistant) avatar */
[data-testid="stChatMessageAvatarAssistant"] {
    background-color: #FFFFFF !important;
    border: 2px solid #7C3AED !important;
}
[data-testid="stChatMessageAvatarAssistant"] svg,
[data-testid="stChatMessageAvatarAssistant"] * {
    color: #7C3AED !important;
    fill: #7C3AED !important;
}

/* Human (user) avatar */
[data-testid="stChatMessageAvatarUser"] {
    background-color: #FFFFFF !important;
    border: 2px solid #000000 !important;
}
[data-testid="stChatMessageAvatarUser"] svg,
[data-testid="stChatMessageAvatarUser"] * {
    color: #000000 !important;
    fill: #000000 !important;
}

/* History list buttons in the sidebar */
[data-testid="stSidebar"] button {
    text-align: left !important;
}
</style>
""", unsafe_allow_html=True)

# --- Session state init: load sessions.json once per browser session ---
if "sessions" not in st.session_state:
    st.session_state.sessions = load_sessions()

# --- Pick the most recent chat as active, or create one if none exist ---
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

# --- Init empty retriever cache (built lazily later) ---
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "retriever_session_id" not in st.session_state:
    st.session_state.retriever_session_id = None


# Fetch the active chat's data dict (pdf_name, chat_history, etc.)
def get_current_session():
    return st.session_state.sessions.get(st.session_state.current_session_id, {})


# Change active chat and invalidate the cached retriever
def switch_session(session_id):
    st.session_state.current_session_id = session_id
    st.session_state.retriever = None
    st.session_state.retriever_session_id = None


# Create and switch to a fresh empty chat session
def start_new_chat():
    new_id = create_session()
    st.session_state.sessions = load_sessions()
    switch_session(new_id)


# =========================== SIDEBAR ===========================
with st.sidebar:
    st.header("WeNext FAQ Assistant")

    # "New Chat" button -> creates session and forces immediate UI refresh
    if st.button("➕ New Chat", use_container_width=True):
        start_new_chat()
        st.rerun()

    st.divider()

    current = get_current_session()
    pdf_name = current.get("pdf_name")

    st.subheader("Document")
    if not pdf_name:
        # PDF uploader — only shown if this chat has no PDF yet (1 PDF per chat)
        uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"], label_visibility="collapsed")

        if uploaded_file is not None:
            session_id = st.session_state.current_session_id

            # Save uploaded PDF bytes to disk under data/<session_id>/
            session_dir = DATA_DIR / session_id
            session_dir.mkdir(parents=True, exist_ok=True)
            save_path = session_dir / uploaded_file.name
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Process PDF: chunk, embed, build vector DB + retriever
            with st.spinner("Processing PDF... (splitting, embedding, indexing)"):
                num_pages, num_chunks = process_pdf(str(save_path), session_id)
                update_session(session_id, pdf_name=uploaded_file.name)
                st.session_state.sessions = load_sessions()
                st.session_state.retriever = build_hybrid_retriever(session_id)
                st.session_state.retriever_session_id = session_id

            st.success(f"Processed {num_pages} pages into {num_chunks} chunks.")
            st.rerun()
    else:
        # PDF already attached -> just display its name
        st.info(f"Active document:  \n**{pdf_name}**")

    st.divider()
    st.subheader("History")

    # Sort all sessions newest-first for the sidebar list
    sessions_sorted = sorted(
        st.session_state.sessions.items(),
        key=lambda kv: kv[1].get("created_at", ""),
        reverse=True
    )

    if not sessions_sorted:
        st.caption("No past chats yet.")

    # Render each past chat as a select button + delete button
    for sid, sdata in sessions_sorted:
        label = sdata.get("pdf_name") or "New chat (no PDF yet)"
        is_active = sid == st.session_state.current_session_id

        col_select, col_delete = st.columns([5, 1])
        with col_select:
            # Select this chat -> switch session and refresh
            prefix = "🟣 " if is_active else "💬 "
            if st.button(prefix + label, key=f"session_{sid}", use_container_width=True):
                if not is_active:
                    switch_session(sid)
                    st.rerun()
        with col_delete:
            # Delete this chat -> remove data, then pick a new active session
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
st.caption("Upload a PDF, then ask questions about it.")

current = get_current_session()
pdf_name = current.get("pdf_name")
chat_history = current.get("chat_history", [])

# --- Replay full chat history for the active session (redrawn every rerun) ---
for turn in chat_history:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        st.markdown(turn["answer"])
        if "sources" in turn:
            # Collapsible source chunks with page numbers (converted to 1-indexed)
            with st.expander("Sources"):
                for i, src in enumerate(turn["sources"], start=1):
                    page_num = src["page"] + 1 if isinstance(src["page"], int) else src["page"]
                    st.markdown(f"**Source {i} — Page {page_num}**")
                    st.caption(src["text"][:300] + ("..." if len(src["text"]) > 300 else ""))
                    st.divider()

# --- Handle new question, only if a PDF exists for this chat ---
if pdf_name:
    # Rebuild retriever if it's stale (user switched to a different chat)
    if st.session_state.retriever_session_id != st.session_state.current_session_id:
        with st.spinner("Loading document index..."):
            st.session_state.retriever = build_hybrid_retriever(st.session_state.current_session_id)
            st.session_state.retriever_session_id = st.session_state.current_session_id

    # Chat input box at bottom of page; returns None unless submitted
    question = st.chat_input("Ask a question about the document...")
    if question:
        # Echo the user's message into the chat UI
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Core RAG call: retrieve -> build prompt -> call LLM -> parse answer
                answer, sources, token_usage = get_answer(
                    st.session_state.retriever,
                    question,
                    chat_history
                )
                st.markdown(answer)

                # Show token usage from the LLM response for cost visibility
                st.caption(
                    f"Tokens — prompt: {token_usage.get('prompt_tokens')}, "
                    f"completion: {token_usage.get('completion_tokens')}, "
                    f"total: {token_usage.get('total_tokens')}"
                )

                # Show retrieved source chunks used to answer
                with st.expander("Sources"):
                    for i, src in enumerate(sources, start=1):
                        page_num = src["page"] + 1 if isinstance(src["page"], int) else src["page"]
                        st.markdown(f"**Source {i} — Page {page_num}**")
                        st.caption(src["text"][:300] + ("..." if len(src["text"]) > 300 else ""))
                        st.divider()

        # Persist this turn to sessions.json, reload state, refresh UI
        append_chat_turn(st.session_state.current_session_id, question, answer, sources)
        st.session_state.sessions = load_sessions()
        st.rerun()
else:
    # No PDF yet -> prompt the user to upload one
    st.info("Upload a PDF from the sidebar to get started.")