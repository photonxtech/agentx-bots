"""Simple recursive-ish text chunker.

Splits on paragraph/sentence boundaries where possible so chunks stay
semantically coherent, with a bit of overlap so context isn't lost at
chunk edges.
"""
 
import re


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    # Prefer splitting on paragraphs, then sentences, then hard cut.
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}" if current else para
            continue

        if current:
            chunks.append(current)

        if len(para) <= chunk_size:
            current = para
        else:
            # Paragraph itself too long: split on sentences.
            sentences = re.split(r"(?<=[.!?])\s+", para)
            current = ""
            for sent in sentences:
                if len(current) + len(sent) + 1 <= chunk_size:
                    current = f"{current} {sent}".strip()
                else:
                    if current:
                        chunks.append(current)
                    current = sent

    if current:
        chunks.append(current)

    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-overlap:]
            overlapped.append(f"{tail} {chunks[i]}".strip())
        chunks = overlapped

    return chunks