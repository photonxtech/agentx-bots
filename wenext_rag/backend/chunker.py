"""
chunker.py
Parses WeNext_AI_Features_Guide.docx into heading-aware chunks.
Each chunk = one section (H1/H2 level), with its full heading path preserved
as metadata (e.g. "4. Automation & Forms > 4.1 WhatsApp automation builder").

This mirrors the structure already validated in the earlier session
(30 chunks, clean hierarchy, ~200-3000 chars each).
"""

import docx
from dataclasses import dataclass, field
from typing import List


@dataclass
class Chunk:
    id: int
    title: str          # full heading path, e.g. "5. CRM & Leads > 5.1 Leads Board — AI lead analysis"
    text: str            # body text under that heading
    heading_level: int   # 1 or 2 (H1 = module, H2 = sub-feature)
    module: str = ""      # top-level module name only, e.g. "5. CRM & Leads"


def parse_docx_to_chunks(path: str) -> List[Chunk]:
    document = docx.Document(path)

    chunks: List[Chunk] = []
    heading_stack: List[str] = []   # tracks [H1, H2] currently active
    current_module = ""
    current_title = "Untitled"
    current_level = 1
    buffer: List[str] = []
    chunk_id = 0

    def flush():
        nonlocal chunk_id
        text = " | ".join(t.strip() for t in buffer if t.strip())
        if text:
            chunks.append(
                Chunk(
                    id=chunk_id,
                    title=current_title,
                    text=text,
                    heading_level=current_level,
                    module=current_module,
                )
            )
            chunk_id += 1
        buffer.clear()

    for para in document.paragraphs:
        style = para.style.name if para.style else ""
        text = para.text.strip()
        if not text:
            continue

        if style.startswith("Heading 1"):
            flush()
            heading_stack = [text]
            current_module = text
            current_title = text
            current_level = 1
        elif style.startswith("Heading 2"):
            flush()
            if len(heading_stack) >= 1:
                heading_stack = [heading_stack[0], text]
            else:
                heading_stack = [text]
            current_title = " > ".join(heading_stack)
            current_level = 2
        else:
            buffer.append(text)

    flush()  # last section

    # also sweep tables, in case any module uses a table for structured info
    for table in document.tables:
        rows_text = []
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                rows_text.append(" | ".join(cells))
        if rows_text:
            chunks.append(
                Chunk(
                    id=chunk_id,
                    title=f"{current_title} (table)",
                    text=" | ".join(rows_text),
                    heading_level=current_level,
                    module=current_module,
                )
            )
            chunk_id += 1

    return chunks


if __name__ == "__main__":
    chunks = parse_docx_to_chunks("data/WeNext_AI_Features_Guide.docx")
    print(f"Total chunks: {len(chunks)}\n")
    for c in chunks:
        preview = c.text[:110].replace("\n", " ")
        print(f"[{c.id}] {c.title}  ({len(c.text)} chars)")
        print(f"    {preview}...\n")
