import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_ANONYMIZED_TELEMETRY"] = "False"

import logging
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.ERROR)

import base64
import uuid
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

try:
    from google.api_core.exceptions import ResourceExhausted
except ImportError:
    ResourceExhausted = None

from backend.ingest import process_pdf, caption_image
from backend.store_utils import (
    ID_KEY,
    PDFS_DIR,
    IMAGES_DIR,
    clear_all_stores,
    get_caption_model,
    get_chat_model,
    get_docstore,
    get_retriever,
    get_vectorstore,
    is_pdf_already_ingested,
    split_docs_by_type,
)

app = FastAPI(title="Multimodal DocuBot API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_no_cache_headers(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# Mount frontend directory for static assets (like style.css)
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    index_path = frontend_dir / "index.html"
    try:
        with open(index_path, encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read index.html: {e}")

VECTORSTORE = None
DOCSTORE = None
CAPTION_MODEL = None
CHAT_MODEL = None
RETRIEVER = None

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about the user's documents. "
    "You will be given retrieved text passages and, sometimes, retrieved images "
    "(charts, diagrams, screenshots) pulled directly from those documents. "
    "Read the images yourself -- don't just rely on any caption-like text near "
    "them. If the retrieved context doesn't contain the answer, say so plainly "
    "instead of guessing. When useful, mention which source/page an answer "
    "came from."
)

MAX_HISTORY_TURNS = 6

def build_prompt(inputs: dict) -> list:
    context = inputs["context"]  # {"texts": [...], "images": [...]}
    question = inputs["question"]
    chat_history = inputs.get("chat_history", [])

    context_text = "\n\n---\n\n".join(
        f"[{d.metadata.get('source', 'unknown')}, page {d.metadata.get('page', '?')}]\n{d.page_content}"
        for d in context["texts"]
    )

    human_content = []
    if context_text:
        human_content.append({"type": "text", "text": f"Retrieved text context:\n{context_text}"})

    for img_doc in context["images"]:
        mime_type = img_doc.metadata.get("mime_type", "image/png")
        human_content.append(
            {
                "type": "image_url",
                "image_url": f"data:{mime_type};base64,{img_doc.page_content}",
            }
        )
        human_content.append(
            {
                "type": "text",
                "text": f"(Image above is from {img_doc.metadata.get('source')}, "
                f"page {img_doc.metadata.get('page')})",
            }
        )

    if not context_text and not context["images"]:
        human_content.append(
            {
                "type": "text",
                "text": "Retrieved context: [No relevant context found in documents. Do not guess or assume answers. Say so plainly.]"
            }
        )

    human_content.append({"type": "text", "text": f"Question: {question}"})

    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    messages.extend(chat_history[-MAX_HISTORY_TURNS:])
    messages.append(HumanMessage(content=human_content))
    return messages

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        routes=app.routes,
    )

    for component in openapi_schema.get("components", {}).get("schemas", {}).values():
        for prop in component.get("properties", {}).values():
            if prop.get("contentMediaType"):
                prop.pop("contentMediaType", None)
                prop["format"] = "binary"
            items = prop.get("items")
            if isinstance(items, dict) and items.get("contentMediaType"):
                items.pop("contentMediaType", None)
                items["format"] = "binary"

    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

class QueryRequest(BaseModel):
    question: str
    chat_history: Optional[List[dict]] = None

def _messages_from_history(history: Optional[List[dict]]) -> List[HumanMessage | AIMessage]:
    if not history:
        return []
    converted = []
    for item in history:
        if not isinstance(item, dict):
            continue
        role = item.get("role", "user")
        content = item.get("content", "")
        if role == "assistant":
            converted.append(AIMessage(content=content))
        else:
            converted.append(HumanMessage(content=content))
    return converted

@app.on_event("startup")
async def startup_event():
    global VECTORSTORE, DOCSTORE, CAPTION_MODEL, CHAT_MODEL, RETRIEVER
    load_dotenv()
    
    # Ensure folders exist
    PDFS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    
    VECTORSTORE = get_vectorstore()
    DOCSTORE = get_docstore()
    CAPTION_MODEL = get_caption_model()
    CHAT_MODEL = get_chat_model()
    RETRIEVER = get_retriever(k=5)

@app.get("/health")
async def health():
    return {"status": "ok", "message": "Multimodal DocuBot backend is running"}

@app.get("/sources")
async def sources():
    # Retrieve all unique source names currently in vectorstore
    documents = VECTORSTORE._collection.get()
    sources = []
    for metadata in documents.get("metadatas", []):
        source = metadata.get("source") if metadata else None
        if source and source not in sources:
            sources.append(source)
    return {"sources": sources}

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}

@app.post("/ingest")
async def ingest_files(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    ingested = []
    already_ingested = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    for upload in files:
        filename = upload.filename
        if not filename:
            continue
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400, detail=f"'{filename}' has an unsupported file format. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
            )

        if is_pdf_already_ingested(filename, VECTORSTORE):
            already_ingested.append(filename)
            continue

        contents = await upload.read()

        if ext == ".pdf":
            # Save and process PDF
            destination = PDFS_DIR / filename
            destination.write_bytes(contents)
            process_pdf(destination, VECTORSTORE, DOCSTORE, CAPTION_MODEL, splitter)
        else:
            # Save copy to disk for reference
            destination = IMAGES_DIR / filename
            destination.write_bytes(contents)
            
            # Determine mime type
            mime_type = f"image/{ext[1:]}"
            if ext == ".jpg":
                mime_type = "image/jpeg"

            # Caption image
            caption = caption_image(CAPTION_MODEL, contents, mime_type)
            if caption == "NOT_INFORMATIVE":
                # Fallback description instead of skipping
                caption = f"Uploaded image {filename}. Description not available."

            doc_id = str(uuid.uuid4())
            b64_full = base64.b64encode(contents).decode("utf-8")

            # Summary document for Chroma search
            summary_doc = Document(
                page_content=caption,
                metadata={
                    ID_KEY: doc_id,
                    "type": "image",
                    "source": filename,
                    "page": 1,
                    "image_file": filename,
                    "mime_type": mime_type,
                }
            )

            # Raw base64 content for Docstore
            raw_doc = Document(
                page_content=b64_full,
                metadata={
                    "type": "image",
                    "source": filename,
                    "page": 1,
                    "image_file": filename,
                    "mime_type": mime_type,
                }
            )

            VECTORSTORE.add_documents([summary_doc])
            DOCSTORE.mset([(doc_id, raw_doc)])

        ingested.append(filename)

    return {"ingested": ingested, "already_ingested": already_ingested}

@app.post("/query")
async def query_docs(request: QueryRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    docs = RETRIEVER.invoke(question)
    context = split_docs_by_type(docs)
    chat_history = _messages_from_history(request.chat_history)

    messages = build_prompt(
        {
            "context": context,
            "question": question,
            "chat_history": chat_history,
        }
    )

    try:
        response = CHAT_MODEL.invoke(messages)
    except Exception as e:
        if ResourceExhausted is not None and isinstance(e, ResourceExhausted):
            raise HTTPException(
                status_code=503,
                detail=(
                    "AI quota exceeded. Please wait for quota reset or update your Google Gemini plan. "
                    "The document ingestion succeeded, but the query could not be answered right now."
                ),
            )
        raise HTTPException(
            status_code=500,
            detail=(
                "AI service error. Please check your API credentials, quota, and network connectivity. "
                "Original error: " + str(e)
            ),
        )

    answer = response.content if isinstance(response.content, str) else str(response.content)

    texts = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "?"),
            "text": doc.page_content,
        }
        for doc in context["texts"]
    ]

    images = []
    for doc in context["images"]:
        doc_id = doc.metadata.get(ID_KEY)
        raw_doc = None
        if doc_id is not None:
            raw_doc = DOCSTORE.get(doc_id)

        if raw_doc is not None:
            images.append(
                {
                    "source": raw_doc.metadata.get("source", "unknown"),
                    "page": raw_doc.metadata.get("page", "?"),
                    "mime_type": raw_doc.metadata.get("mime_type", "image/png"),
                    "image_data": raw_doc.page_content,
                }
            )
        else:
            images.append(
                {
                    "source": doc.metadata.get("source", "unknown"),
                    "page": doc.metadata.get("page", "?"),
                    "mime_type": doc.metadata.get("mime_type", "image/png"),
                    "image_data": doc.page_content,
                }
            )

    return {"answer": answer, "texts": texts, "images": images}

@app.post("/clear")
async def clear_index():
    clear_all_stores(VECTORSTORE, DOCSTORE)
    return {"status": "cleared"}