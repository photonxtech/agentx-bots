import os
import pickle
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough  # Sends the raw question straight into the prompt, without changing it
from langchain_core.output_parsers import StrOutputParser

# Load env vars (GROQ_API_KEY) from .env
load_dotenv()

# Hardcoded paths — legacy script, single global Chroma DB / chunk file, no sessions
PERSIST_DIR = "chroma_db"
CHUNKS_PATH = "chunks.pkl"

# Prompt template — no chat history support, single-turn only
PROMPT_TEMPLATE = """You are a helpful assistant answering questions about WeNext's Meta & WhatsApp Business Platform FAQ.
Use ONLY the context below to answer. If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}

Answer clearly and concisely:"""

# Join retrieved doc chunks into one text block for the prompt's {context}
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Build hybrid BM25 + vector retriever with reranking (single global index, no session_id)
def build_hybrid_retriever():
    # Load pre-chunked documents saved by ingest.py
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    # Keyword-based retriever (sparse/lexical matching)
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 10

    # Semantic retriever backed by the persisted Chroma vector DB
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    vector_retriever = vectordb.as_retriever(search_kwargs={"k": 10})

    # Merge BM25 + vector results, weighted 40%/60%
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6]
    )

    # Rerank merged candidates and keep only the top 4 most relevant
    compressor = FlashrankRerank(top_n=4)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever
    )

    return compression_retriever

def main():
    retriever = build_hybrid_retriever()

    # LLM setup — note: no token_usage tracking here (predecessor to get_answer())
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.5)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    # LCEL chain: retrieve+format -> fill prompt -> call LLM -> parse to string
    # (this is the "old" approach — replaced by the manual .invoke() chain in rag_utils.py
    # because LCEL's final output is just a string, losing access to response_metadata/tokens)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Simple terminal loop — no chat history, no persistence, no citations
    print("Hybrid RAG system ready. Type your question (or 'exit' to quit).\n")
    while True:
        question = input("Q: ")
        if question.lower() in ("exit", "quit"):
            break
        answer = rag_chain.invoke(question)
        print(f"\nA: {answer}\n")

if __name__ == "__main__":
    main()