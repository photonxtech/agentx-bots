"""FastAPI backend for Multi-RAG.

This package exposes the exact same RAG pipeline the Streamlit app uses
(`rag/` + `chat_store.py`), but over a REST API instead of a Streamlit UI.

Run it with:

    uvicorn api.main:app --reload --port 8000

(run from the `mutlirag/` directory so `config`, `chat_store`, and `rag`
resolve as top-level imports).
"""
