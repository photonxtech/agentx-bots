"""Diagnostic script for RAG retrieval (LangChain / Chroma backend)."""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.rag import (
    RELEVANCE_THRESHOLD,
    generate_rag_response,
    get_chunk_count,
    get_rag_status,
    get_relevant_context,
)

QUERY = "How do I build a WhatsApp automation"


def main() -> None:
    status = get_rag_status()
    print("RAG status:", status)
    print("Query:", QUERY)
    print()

    print("Chunks in collection:", get_chunk_count())
    print()

    chunks, min_dist = get_relevant_context(QUERY)
    print("Min distance:", min_dist)
    print("Threshold pass (<= %.2f):" % RELEVANCE_THRESHOLD, min_dist <= RELEVANCE_THRESHOLD)
    print()

    for i, c in enumerate(chunks[:5]):
        print(f"--- chunk {i + 1} dist={c['distance']:.4f} source={c['source']} page={c['page']} ---")
        print(c["text"][:400])
        print()

    print("--- RESPONSE ---")
    print(generate_rag_response(QUERY, []))
    print()

    pdf_path = os.path.join("data", "WeNext_Meta_WhatsApp_FAQ.pdf")
    if not os.path.exists(pdf_path):
        return

    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    print("--- PDF pages mentioning automation/build/whatsapp ---")
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if re.search(r"automation|whatsapp|build", text, re.I):
            print(f"=== Page {i + 1} ===")
            safe_text = text[:800].encode("ascii", errors="replace").decode("ascii")
            print(safe_text)
            print()


if __name__ == "__main__":
    main()