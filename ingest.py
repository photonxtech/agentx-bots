import os
import pickle
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load env vars (not actually used here, but kept for consistency with rest of project)
load_dotenv()

# Hardcoded paths — legacy script, no session/multi-PDF support

PERSIST_DIR = "chroma_db"
CHUNKS_PATH = "chunks.pkl"

def ingest_pdf(pdf_path):
    # Load the PDF into a list of LangChain Document objects (one per page)
    print("Loading PDF...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages.")

    # Split pages into smaller overlapping chunks for better retrieval
    print("Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    # Save chunks to disk so BM25 retriever can reload them later without re-parsing the PDF
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Saved raw chunks to {CHUNKS_PATH}")

    # Load the embedding model that converts text chunks into vectors
    print("Loading embedding model (downloads once, then caches locally)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Embed all chunks and persist them into a local Chroma vector database
    print("Storing chunks in Chroma vector DB...")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
  
    print(f"Done. Vector DB saved to ./{PERSIST_DIR}")
    