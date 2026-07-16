from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings , ChatGoogleGenerativeAI
from langchain_community.vectorstores import InMemoryVectorStore
import streamlit as st
from time import sleep

llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash")

if "vector_db" not in st.session_state:
    st.session_state.vector_db=None

if "messages" not in st.session_state:
    st.session_state.messages=[]

def document_process(path):
    ##doc loading
    loader=PyPDFLoader(path)
    docs=loader.load()

    ##splitting
    splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    docs=splitter.split_documents(docs)

    ##embedding and vector store
    embeddings=GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    vector_db=InMemoryVectorStore.from_documents(
        documents=docs,
        embedding=embeddings
    )
    st.session_state.vector_db=vector_db
    st.session_state.document_uploaded = True


st.subheader("Document QnA Chatbot - Ask Anything")

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

#### doc upload
if not st.session_state.document_uploaded:
    file=st.file_uploader(label="select your pdf file",type="pdf")
    if file:
        with open("uploaded_document.pdf","wb") as f:
            f.write(file.getvalue())

        with st.spinner("Processing document"):
            document_process("./uploaded_document.pdf")
        
        st.markdown("Document Processed succesfully...")
        sleep(2)
        st.rerun()

if st.session_state.document_uploaded and st.session_state.vector_db: ##chat ui

    for oneMessage in st.session_state.messages:
        role=oneMessage["role"]
        content=oneMessage["content"]
        st.chat_message(role).markdown(content)

    query=st.chat_input("Ask anythingg...")
    if query:
        st.session_state.messages.append({"role":"user","content":query})
        st.chat_message("user").markdown(query)
        documents=st.session_state.vector_db.similarity_search(query,k=2)
        context=""
        for doc in documents:
            context += doc.page_content + "\n\n"
        
        prompt=f"""You are a helpful assistant and you provide answers for user questions based on the provided context. context:{context},question:{query}"""
        result=llm.invoke(prompt)

        st.session_state.messages.append({"role":"ai","content":result.content})
        st.chat_message("ai").markdown(result.content)