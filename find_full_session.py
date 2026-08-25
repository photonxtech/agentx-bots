"""
Scans EVERY session's chunk pickle in storage/chunks/ and prints how many
chunks each one has, plus which source filename(s) it contains. Run this
from your project root:

    python find_full_session.py

This finds the session that actually has the FULL osw_data.pdf indexed
(should be roughly 1,900+ chunks for a 516-page document at chunk_size=450),
as opposed to a session with only a handful of chunks (wrong/partial file).
"""
import pickle
from pathlib import Path
from collections import Counter

CHUNKS_ROOT = Path("storage/chunks")

results = []
for path in sorted(CHUNKS_ROOT.glob("*.pkl")):
    if path.name.endswith("_pages.pkl"):
        continue  # skip the page-lookup pickles, only want chunk pickles
    session_id = path.stem
    try:
        with open(path, "rb") as f:
            chunks = pickle.load(f)
    except Exception as e:
        print(f"{session_id}: FAILED TO LOAD ({e})")
        continue

    sources = Counter(c.metadata.get("source", "unknown") for c in chunks)
    results.append((session_id, len(chunks), dict(sources)))

# Sort by chunk count, descending — the real osw_data.pdf session should be
# the biggest by far.
results.sort(key=lambda r: r[1], reverse=True)

print(f"{'session_id':<40} {'chunks':>8}   sources")
print("-" * 100)
for session_id, count, sources in results:
    print(f"{session_id:<40} {count:>8}   {sources}")
