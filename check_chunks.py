"""
Run this from your WeNextAI_FastAPI project root, in the ragenv environment:
    python check_chunks.py <session_id>

Bypasses BM25/vector/reranker entirely and checks the RAW chunk pickle for
this session — tells us definitively whether the "12 Techniques" content
was ever indexed at all, vs. indexed but never surfaced by retrieval.
"""
import pickle
import sys
from pathlib import Path

session_id = sys.argv[1] if len(sys.argv) > 1 else None
if not session_id:
    print("Usage: python check_chunks.py <session_id>")
    sys.exit(1)

CHUNKS_ROOT = Path("storage/chunks")
chunks_path = CHUNKS_ROOT / f"{session_id}.pkl"

if not chunks_path.exists():
    print(f"No chunk file found at {chunks_path}")
    sys.exit(1)

with open(chunks_path, "rb") as f:
    chunks = pickle.load(f)

print(f"Total chunks in this session: {len(chunks)}")

# 1. Does ANY chunk contain the page's distinctive text?
needle_variants = ["12 Techniques", "TECHNIQUE #1", "REFRAME TO", "Fredrickson"]
found_any = False
for i, c in enumerate(chunks):
    text = c.page_content
    for needle in needle_variants:
        if needle in text:
            found_any = True
            print(f"\nFOUND '{needle}' in chunk #{i}")
            print(f"  source={c.metadata.get('source')} page={c.metadata.get('page')} page_label={c.metadata.get('page_label')}")
            print(f"  content preview: {text[:200]!r}")

if not found_any:
    print("\n*** NOT FOUND in ANY chunk for this session. ***")
    print("This confirms the chunk was never indexed for this session —")
    print("not a retrieval-tuning issue. Check page 368 (0-indexed) specifically:")

# 2. Explicitly check what chunk(s), if any, came from page 368/369
print("\n--- All chunks with metadata page in [367, 368, 369, 370] ---")
matches = [c for c in chunks if c.metadata.get("page") in (367, 368, 369, 370)]
if not matches:
    print("NONE. No chunks at all exist for pages 367-370 in this session's index.")
else:
    for c in matches:
        print(f"  page={c.metadata.get('page')} page_label={c.metadata.get('page_label')} "
              f"preview={c.page_content[:150]!r}")
