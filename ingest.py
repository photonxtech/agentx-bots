import os
import pickle
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables
load_dotenv()

# Paths
# Paths
STORAGE_DIR = Path("storage")
STORAGE_DIR.mkdir(exist_ok=True)

PERSIST_DIR = STORAGE_DIR / "chroma_db"
CHUNKS_PATH = STORAGE_DIR / "chunks.pkl"


def ingest_pdf(pdf_path):
    # -----------------------------
    # Load PDF
    # -----------------------------
    print("\n========== LOADING PDF ==========")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print(f"Loaded {len(documents)} pages.")

    # -----------------------------
    # Split into chunks
    # -----------------------------
    print("\n========== SPLITTING INTO CHUNKS ==========")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    # -----------------------------
    # DEBUG: Print every chunk
    # -----------------------------
    print("\n========== PRINTING CHUNKS ==========")

    for i, chunk in enumerate(chunks):
        print(f"\n------ CHUNK {i+1} ------")
        print("Page:", chunk.metadata.get("page"))
        print(chunk.page_content[:800])

    # -----------------------------
    # DEBUG: Search for verification section
    # -----------------------------
    print("\n========== SEARCHING FOR BUSINESS VERIFICATION ==========")

    found = False

    keywords = [
        "gst registration",
        "certificate of incorporation",
        "business verification",
        "business license",
        "trade license",
        "utility bill",
        "bank statement"
    ]

    for i, chunk in enumerate(chunks):
        text = chunk.page_content.lower()

        if any(keyword in text for keyword in keywords):
            found = True
            print(f"\n✅ FOUND IN CHUNK {i+1}")
            print("Page:", chunk.metadata.get("page"))
            print(chunk.page_content)

    if not found:
        print("\n❌ Business Verification section NOT found in extracted chunks.")

    print("\n============================================")

    # -----------------------------
    # Save chunks
    # -----------------------------
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"\nSaved chunks to {CHUNKS_PATH}")

    # -----------------------------
    # Embeddings
    # -----------------------------
    print("\nLoading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # -----------------------------
    # Create Vector DB
    # -----------------------------
    print("\nCreating Chroma vector database...")

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )

    print(f"\nVector DB saved to {PERSIST_DIR}")
    print("\n========== INGESTION COMPLETE ==========\n")

    return vectordb


if __name__ == "__main__":
    pdf_path = input("Enter PDF path: ").strip()
    ingest_pdf(pdf_path)