"""
ingest.py
Chunk the WeNext doc -> embed locally with sentence-transformers -> store in ChromaDB.

Run this once (or whenever the source .docx changes):
    python3 ingest.py

Requires (see requirements.txt):
    pip install python-docx chromadb sentence-transformers --break-system-packages
"""

import chromadb
from sentence_transformers import SentenceTransformer
from chunker import parse_docx_to_chunks

DOCX_PATH = "data/WeNext_AI_Features_Guide.docx"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "wenext_ai_features"
EMBED_MODEL = "all-MiniLM-L6-v2"   # small, fast, good enough for a demo (384-dim)


def main():
    print(f"Loading embedding model: {EMBED_MODEL} ...")
    model = SentenceTransformer(EMBED_MODEL)

    print(f"Parsing {DOCX_PATH} ...")
    chunks = parse_docx_to_chunks(DOCX_PATH)
    print(f"  -> {len(chunks)} chunks")

    print("Embedding chunks ...")
    texts = [f"{c.title}\n{c.text}" for c in chunks]  # prepend title for better retrieval signal
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    print(f"Writing to ChromaDB at ./{CHROMA_DIR} ...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # fresh collection each run, so re-running ingest.py after edits doesn't duplicate
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=[str(c.id) for c in chunks],
        embeddings=embeddings,
        documents=[c.text for c in chunks],
        metadatas=[
            {"title": c.title, "module": c.module, "heading_level": c.heading_level}
            for c in chunks
        ],
    )

    print(f"Done. {collection.count()} chunks stored in collection '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    main()
