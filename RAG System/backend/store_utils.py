import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_ANONYMIZED_TELEMETRY"] = "False"

import logging
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.ERROR)

import pickle
from pathlib import Path

from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import LocalFileStore
from chromadb.config import Settings
from langchain.storage.encoder_backed import EncoderBackedStore
from langchain_core.documents import Document

# --- Paths ----------
ROOT_DIR = Path(__file__).parent.parent
PERSIST_DIR = ROOT_DIR / "chroma_db"
DOCSTORE_DIR = ROOT_DIR / "docstore"
DATA_DIR = ROOT_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
PDFS_DIR = DATA_DIR / "pdfs"

# --- Model names ------
CAPTION_MODEL = "gemini-3.1-flash-lite"
CHAT_MODEL = "gemini-3.5-flash"
EMBEDDING_MODEL = "models/gemini-embedding-001"

ID_KEY = "doc_id"


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)


def get_caption_model() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model=CAPTION_MODEL, temperature=0, max_retries=2)


def get_chat_model() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0.2, max_retries=2)


def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name="multimodal_rag",
        embedding_function=get_embeddings(),
        persist_directory=str(PERSIST_DIR),
        client_settings=Settings(anonymized_telemetry=False)
    )


def get_docstore() -> EncoderBackedStore:
    """A byte-store on disk, wrapped so it can hold arbitrary Document objects."""
    fs = LocalFileStore(str(DOCSTORE_DIR))
    return EncoderBackedStore(
        fs,
        key_encoder=lambda key: key,
        value_serializer=pickle.dumps,
        value_deserializer=pickle.loads,
    )


def get_retriever(k: int = 5) -> MultiVectorRetriever:
    return MultiVectorRetriever(
        vectorstore=get_vectorstore(),
        docstore=get_docstore(),
        id_key=ID_KEY,
        search_kwargs={"k": k},
    )


def split_docs_by_type(docs: list[Document]) -> dict:
    """Separate retrieved docs into text passages vs. images for prompt building."""
    out = {"texts": [], "images": []}
    for d in docs:
        if d.metadata.get("type") == "image":
            out["images"].append(d)
        else:
            out["texts"].append(d)
    return out


def is_pdf_already_ingested(pdf_name: str, vectorstore) -> bool:
    """Check if a PDF's chunks already exist in the vectorstore."""
    results = vectorstore.get(where={"source": pdf_name}, limit=1)
    return len(results.get("ids", [])) > 0


def delete_pdf_from_stores(pdf_name: str, vectorstore, docstore):
    """Cleanly delete all vector chunks and docstore entries associated with a PDF."""
    results = vectorstore.get(where={"source": pdf_name})
    doc_ids = []
    for meta in results.get("metadatas", []):
        if ID_KEY in meta:
            doc_ids.append(meta[ID_KEY])

    # Delete from docstore
    if doc_ids:
        docstore.mdelete(doc_ids)

    # Delete from vectorstore
    vectorstore.delete(where={"source": pdf_name})


def clear_all_stores(vectorstore, docstore):
    """Wipe all documents from both the vectorstore and the docstore."""
    # 1. Delete from vectorstore
    all_data = vectorstore._collection.get()
    ids = all_data.get("ids", [])
    if ids:
        vectorstore.delete(ids=ids)

    # 2. Delete from docstore
    doc_keys = list(docstore.yield_keys())
    if doc_keys:
        docstore.mdelete(doc_keys)


def get_all_sources(vectorstore) -> set:
    """Retrieve all unique PDF file names currently indexed in the vectorstore."""
    all_data = vectorstore._collection.get()
    sources = set()
    for meta in all_data.get("metadatas", []):
        if meta and "source" in meta:
            sources.add(meta["source"])
    return sources

