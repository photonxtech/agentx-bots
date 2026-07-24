import logging
import os
import ollama
import streamlit as st
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
DOC_PATH = "./data/PhotonX_Company_Profile.pdf"
MODEL_NAME = "qwen2.5:3b"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "simple-rag"
PERSIST_DIRECTORY = "./chroma_db"


def ingest_pdf(doc_path):
  """Load PDF documents."""
  if os.path.exists(doc_path):
    loader = PyPDFLoader(file_path=doc_path)
    data = loader.load()
    logging.info(f"PDF loaded successfully. Loaded {len(data)} pages.")
    return data
  else:
    logging.error(f"PDF file not found at path: {doc_path}")
    st.error(f"PDF file not found at path: {doc_path}")
    return None


def split_documents(documents):
  """Split documents into smaller chunks."""
  from langchain_text_splitters import RecursiveCharacterTextSplitter

  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=800, chunk_overlap=150
  )
  chunks = text_splitter.split_documents(documents)
  logging.info(f"Documents split into {len(chunks)} chunks.")
  return chunks


@st.cache_resource(show_spinner=False)
def get_vector_db():
  """Load or create vector DB once per app session."""
  ollama.pull(EMBEDDING_MODEL)
  embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)

  if os.path.exists(PERSIST_DIRECTORY) and os.listdir(PERSIST_DIRECTORY):
    vector_db = Chroma(
        embedding_function=embedding,
        collection_name=VECTOR_STORE_NAME,
        persist_directory=PERSIST_DIRECTORY,
    )
    logging.info("Vector DB loaded from disk.")
  else:
    data = ingest_pdf(DOC_PATH)
    if not data:
      return None

    chunks = split_documents(data)
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        collection_name=VECTOR_STORE_NAME,
        persist_directory=PERSIST_DIRECTORY,
    )
    logging.info("Vector DB created and saved.")

  return vector_db


@st.cache_resource(show_spinner=False)
def get_llm():
  """Load LLM instance once."""
  return ChatOllama(model=MODEL_NAME, temperature=0.0)


def format_docs(docs):
  if not docs:
    return ""
  return "\n\n".join(doc.page_content for doc in docs)


def main():
  st.title("Document Assistant")

  # Pre-load resources
  with st.spinner("Initializing models and database..."):
    vector_db = get_vector_db()
    llm = get_llm()

  if vector_db is None:
    st.error("Could not load the vector database. Please check your PDF path.")
    return

  # Create chain once per interaction
  base_retriever = vector_db.as_retriever(
      search_type="similarity", search_kwargs={"k": 4}
  )

  QUERY_PROMPT = PromptTemplate(
      input_variables=["question"],
      template="""You are an AI assistant. Rephrase the given user question into 3 alternative search queries to retrieve relevant document passages. 
Provide ONLY the rephrased questions separated by newlines, with no intro text or numbers.

Original question: {question}""",
  )

  retriever = MultiQueryRetriever.from_llm(
      retriever=base_retriever, llm=llm, prompt=QUERY_PROMPT
  )

  template = """You are a document QA assistant. Answer the question using ONLY the CONTEXT below.

CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTIONS:
- If the CONTEXT is empty or does not contain enough information to answer, output EXACTLY this phrase:
"I cannot answer this question based on the provided document."
- Do not use outside knowledge.

ANSWER:"""

  prompt = ChatPromptTemplate.from_template(template)
  chain = (
      {
          "context": retriever | format_docs,
          "question": RunnablePassthrough(),
      }
      | prompt
      | llm
      | StrOutputParser()
  )

  # Form UI to handle re-runs properly
  with st.form("query_form"):
    user_input = st.text_input("Enter your question:", "")
    submitted = st.form_submit_button("Ask")

  if submitted and user_input:
    with st.spinner("Searching document & generating answer..."):
      try:
        response = chain.invoke(user_input)
        st.markdown("### Assistant Response")
        st.write(response)
      except Exception as e:
        st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
  main()