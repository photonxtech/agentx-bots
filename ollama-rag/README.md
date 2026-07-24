# Ollama RAG Project

## Project Overview
This project is a local document-question answering application built with Python, Streamlit, LangChain, Chroma, and Ollama. It loads a PDF document, creates embeddings, stores them in a local vector database, and lets you ask questions about the document content.

The current app entry point is [app.py](app.py), and the project also includes a simple script example in [pdf-rag.py](pdf-rag.py).

## Setup Instructions
1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Make sure Ollama is installed and running locally.

4. Pull the required models if they are not already available:
   ```bash
   ollama pull qwen2.5:3b
   ollama pull nomic-embed-text
   ```

5. Place your PDF file in the data folder. The app expects a file at:
   ```text
   data/PhotonX_Company_Profile.pdf
   ```

## Required Environment Variables
No project-specific environment variables are required by default.

Optional:
- `OLLAMA_HOST` - if you are running Ollama on a different host or port, you can set this environment variable accordingly.

## How to Run the Project Locally
Start the Streamlit app with:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal (usually http://localhost:8501).

## API Endpoints
This project does not expose a separate REST API. It runs as a local Streamlit web app with a single interactive chat-style interface.

## Additional Notes and Dependencies
Key dependencies include:
- `streamlit`
- `ollama`
- `langchain`
- `langchain-community`
- `langchain-ollama`
- `chromadb`
- `pypdf`
- `unstructured`
- `sentence-transformers`

Notes:
- The app uses a local Chroma vector database stored in the [chroma_db](chroma_db) folder.
- The first run may take some time while embeddings and the vector store are created.
- If the PDF file is missing, the app will show an error message.
