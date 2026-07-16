import re


def dedupe_key(text: str) -> str:
    # Boilerplate (testimonials, CTAs) often reappears near-identically across many
    # pages, but chunk boundaries can shift slightly depending on surrounding content,
    # so exact-string matching alone misses some repeats. Comparing a normalized prefix
    # catches those near-duplicates without needing real fuzzy matching.
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    return normalized[:120]
