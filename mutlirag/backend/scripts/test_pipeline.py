"""End-to-end test of the RAG pipeline, run from the project root.

Ingests the sample Excel + mixed PDF + standalone image, builds the index,
runs retrieval queries, and (if the Groq key is valid) generates an answer.
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")  # let the ✅/emoji print on Windows

from dotenv import load_dotenv

load_dotenv()

from rag import chunking, ingestion
from rag.vectorstore import VectorStore


def read(path):
    with open(path, "rb") as f:
        return f.read()


print("=" * 70)
print("STEP 1 — Ingestion")
print("=" * 70)

files = [
    "samples/employees.xlsx",
    "samples/mixed_report.pdf",
    "samples/code.png",
]

all_docs = []
for path in files:
    name = path.split("/")[-1]
    docs = ingestion.ingest(read(path), name)
    all_docs += docs
    print(f"\n[{name}] -> {len(docs)} document(s)")
    for d in docs:
        tag = f"(ocr={d.meta.get('ocr')}, imgs={len(d.meta.get('images', []))})" if d.kind == "pdf" else ""
        print(f"    - {d.kind} {tag}: {d.text[:90].replace(chr(10), ' ')}...")

chunks = chunking.chunk_documents(all_docs)
print(f"\nTotal chunks after chunking: {len(chunks)}")

print("\n" + "=" * 70)
print("STEP 2 — Embed + index")
print("=" * 70)
import shutil
shutil.rmtree("index_store_test", ignore_errors=True)
store = VectorStore(dirpath="index_store_test")
store.add(chunks)
print(f"Indexed {store.size} chunks using backend: {store.backend}")

print("\n" + "=" * 70)
print("STEP 3 — Retrieval (no LLM needed)")
print("=" * 70)
queries = [
    "What is the building access code?",          # lives inside the PDF's image (OCR)
    "Who works in Engineering and where?",         # lives in the Excel
    "How much did revenue grow in Q3?",            # lives in the PDF text layer
]
for q in queries:
    print(f"\nQ: {q}")
    for doc, score in store.search(q, top_k=2):
        print(f"   [{score:.3f}] ({doc.source}/{doc.kind}) {doc.text[:80].replace(chr(10),' ')}...")

print("\n" + "=" * 70)
print("STEP 4 — Generation via Groq")
print("=" * 70)
from rag import generator

try:
    q = "What is the building access code, and how long is it valid?"
    hits = store.search(q, top_k=4)
    print(f"Q: {q}\nA: ", end="", flush=True)
    for delta in generator.answer(q, hits):
        print(delta, end="", flush=True)
    print("\n\nGENERATION OK ✅")
except Exception as e:
    print(f"\n[skipped generation] {type(e).__name__}: {e}")

print("\nDONE ✅")
