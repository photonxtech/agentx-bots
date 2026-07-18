import argparse
import base64
import io
import uuid

import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.store_utils import (
    ID_KEY,
    IMAGES_DIR,
    get_caption_model,
    get_docstore,
    get_vectorstore,
)

MIN_IMAGE_SIDE = 120  # px -- filters out small icons/bullets/logos

CAPTION_PROMPT = (
    "Describe only the factual, informational content visible in this image "
    "in 2-4 sentences. If it's a chart or graph, state the trend and any "
    "specific values you can read. If it's a diagram, describe what it shows "
    "and how the parts connect. If it's purely decorative with no "
    "information (a logo, a divider, a background pattern), respond with "
    "exactly: NOT_INFORMATIVE"
)


def caption_image(model, image_bytes: bytes, mime_type: str) -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    message = HumanMessage(
        content=[
            {"type": "text", "text": CAPTION_PROMPT},
            {"type": "image_url", "image_url": f"data:{mime_type};base64,{b64}"},
        ]
    )
    response = model.invoke([message])
    return response.content.strip()


def process_pdf(pdf_path, vectorstore, docstore, caption_model, splitter):
    doc = fitz.open(pdf_path)
    text_docs, image_docs = [], []

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_number = page_index + 1

        # --- text ---
        page_text = page.get_text().strip()
        if page_text:
            for chunk in splitter.split_text(page_text):
                doc_id = str(uuid.uuid4())
                text_docs.append(
                    (
                        doc_id,
                        Document(
                            page_content=chunk,
                            metadata={
                                ID_KEY: doc_id,
                                "type": "text",
                                "source": pdf_path.name,
                                "page": page_number,
                            },
                        ),
                    )
                )

        # --- images ---
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            width, height = base_image.get("width", 0), base_image.get("height", 0)
            if width < MIN_IMAGE_SIDE or height < MIN_IMAGE_SIDE:
                continue  # likely an icon, bullet, or logo -- skip

            image_bytes = base_image["image"]
            ext = base_image.get("ext", "png")
            mime_type = f"image/{ext}"

            caption = caption_image(caption_model, image_bytes, mime_type)
            if caption == "NOT_INFORMATIVE":
                continue

            # save a copy to disk for reference/debugging
            IMAGES_DIR.mkdir(parents=True, exist_ok=True)
            image_name = f"{pdf_path.stem}_p{page_number}_{img_index}.{ext}"
            (IMAGES_DIR / image_name).write_bytes(image_bytes)

            doc_id = str(uuid.uuid4())
            b64_full = base64.b64encode(image_bytes).decode("utf-8")

            # Summary (caption) goes in the vectorstore -- this is what gets embedded/searched.
            image_docs.append(
                (
                    doc_id,
                    summary_doc := Document(
                        page_content=caption,
                        metadata={
                            ID_KEY: doc_id,
                            "type": "image",
                            "source": pdf_path.name,
                            "page": page_number,
                            "image_file": image_name,
                            "mime_type": mime_type,
                        },
                    ),
                    # Raw content goes in the docstore -- this is what the LLM sees at answer time.
                    Document(
                        page_content=b64_full,
                        metadata={
                            "type": "image",
                            "source": pdf_path.name,
                            "page": page_number,
                            "image_file": image_name,
                            "mime_type": mime_type,
                        },
                    ),
                )
            )

    # --- write everything out ---
    all_summary_docs = [d for _, d in text_docs] + [d for _, d, _ in image_docs]
    if all_summary_docs:
        vectorstore.add_documents(all_summary_docs)

    text_ids = [doc_id for doc_id, _ in text_docs]
    text_raw = [d for _, d in text_docs]
    image_ids = [doc_id for doc_id, _, _ in image_docs]
    image_raw = [raw for _, _, raw in image_docs]

    if text_ids:
        docstore.mset(list(zip(text_ids, text_raw)))
    if image_ids:
        docstore.mset(list(zip(image_ids, image_raw)))

    print(
        f"{pdf_path.name}: indexed {len(text_docs)} text chunks and "
        f"{len(image_docs)} images (skipped small/decorative images)."
    )


def main():
    load_dotenv()
    from backend.store_utils import clear_all_stores, PDFS_DIR, get_caption_model
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-dir", default=str(PDFS_DIR), help="Folder of PDFs to ingest")
    parser.add_argument("--clear", action="store_true", help="Clear the vectorstore and docstore first")
    args = parser.parse_args()

    from pathlib import Path

    vectorstore = get_vectorstore()
    docstore = get_docstore()

    if args.clear:
        print("Clearing stores...")
        clear_all_stores(vectorstore, docstore)
        print("Stores cleared.")

    pdf_dir = Path(args.pdf_dir)
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        if args.clear:
            return
        print(f"No PDFs found in {pdf_dir.resolve()}. Add some and re-run.")
        return

    caption_model = get_caption_model()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    for pdf_path in pdf_paths:
        process_pdf(pdf_path, vectorstore, docstore, caption_model, splitter)

    print("Done. Run `streamlit run frontend/app.py` to start chatting.")


if __name__ == "__main__":
    main()
