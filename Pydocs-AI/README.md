# PyDocs AI 🐍

Your persistent, always-on Python documentation assistant — indexed once, queried forever.

## Project Overview

**PyDocs AI** is a Streamlit-based Retrieval Augmented Generation (RAG) application that provides an intelligent Q&A interface for Python documentation. It uses:

- **FAISS** (Facebook AI Similarity Search) for fast vector indexing
- **FastEmbed** for CPU-efficient embedding generation
- **LangChain** for document processing and retrieval orchestration
- **Google Generative AI (Gemini)** for response generation
- **Streamlit** for interactive web interface

### Key Features

- 📚 **One-time Indexing**: Build a FAISS vector index once, query forever
- 💬 **Multi-session Chat**: Persistent chat history saved locally
- 🔍 **Two-stage Retrieval**: MMR (Maximal Marginal Relevance) + cross-encoder reranking
- ⚡ **CPU-optimized**: Uses ONNX Runtime for fast inference without GPU
- 💾 **Resumable Indexing**: Crash-safe checkpoint system during index building
- 🏗️ **Structure-aware Chunking**: Respects Sphinx documentation sections and hierarchies

## Setup Instructions

### Prerequisites

- Python 3.10+
- Virtual environment (recommended: venv or conda)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/photonxtech/agentx-bots.git
   cd agentx-bots
   git checkout kalyan
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv312
   # On Windows:
   .\venv312\Scripts\Activate.ps1
   # On macOS/Linux:
   source venv312/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables** (see next section)

5. **Add Python documentation:**
   ```bash
   mkdir python-docs
   # Drop your .txt Python documentation files into this folder
   # (nested folders are supported)
   ```

## Required Environment Variables

Create a `.env` file in the project root with:

```env
GOOGLE_API_KEY=your_google_gemini_api_key_here
```

**How to get your Google API Key:**
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Click "Create API Key"
3. Copy and paste it into `.env`

**Note:** This file is gitignored and will not be committed.

## How to Run Locally

### First Time Setup (Index Building)

```bash
# Activate virtual environment
.\venv312\Scripts\Activate.ps1  # Windows

# Run the app
streamlit run app.py
```

The browser will open at `http://localhost:8501`

**First run will:**
1. Download embedding model (~130MB)
2. Scan `python-docs/` directory
3. Build FAISS index (~5-10 minutes depending on corpus size)
4. Create checkpoint files for resumable builds

### Subsequent Runs

Once indexed, the app loads instantly. Just run:
```bash
streamlit run app.py
```

### Re-index on Demand

Click the **🔁 Re-index** button in the sidebar to rebuild the index if documentation changes.

## Project Structure

```
UnderstandingRAG/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (gitignored)
├── .gitignore                  # Git ignore rules
├── python-docs/                # Your .txt documentation files
├── faiss_store_pydocs/         # Built FAISS index (gitignored, rebuilt on demand)
├── fastembed_cache/            # Downloaded model weights (gitignored, re-downloaded)
├── chat_sessions/              # User conversation history (gitignored, local only)
└── README.md                   # This file
```

## Key Components

### Constants & Configuration

- **CHUNK_SIZE**: 1200 characters per document chunk
- **EMBEDDING_MODEL**: BAAI/bge-small-en-v1.5 (384-dim, fast)
- **RERANKER_MODEL**: BAAI/bge-reranker-base (cross-encoder)
- **LLM**: Gemini 2.5 Flash

### Retrieval Pipeline

1. **Query Rewriting**: Uses LLM to convert follow-up questions into standalone queries
2. **MMR Retrieval**: Pulls diverse candidate pool from FAISS
3. **Cross-encoder Reranking**: Re-scores candidates with bge-reranker-base
4. **Response Generation**: Gemini answers using top-6 ranked chunks

### Chat Persistence

- Chat sessions saved to `chat_sessions/{session_id}.json`
- Each conversation is independent
- Stored locally (not synced to server)

## Web Interface

This is a **Streamlit app** with the following features:

### Main Features

- **Chat Input**: Ask questions about Python documentation
- **Sidebar Controls**:
  - 🧹 Reset Chat — Clear current conversation
  - 🔁 Re-index — Rebuild documentation index
  - ⚡ Load existing index — Skip hash verification for instant load
  - 💬 Chat Sessions — Browse and switch between saved conversations

### Output

- **Markdown Response**: LLM-generated answer
- **Source Citations**: Expandable section showing:
  - Chunk relevance scores
  - Source file paths
  - Documentation sections
  - Retrieved content preview

## Dependencies

All dependencies are listed in `requirements.txt`:

- **streamlit** — Web framework
- **langchain** — LLM orchestration
- **faiss-cpu** — Vector indexing
- **fastembed** — Fast embeddings
- **google-generativeai** — Gemini API
- **python-dotenv** — Environment variable management

See [requirements.txt](requirements.txt) for specific versions.

## Troubleshooting

### "No .txt files found"
- Ensure you added Python documentation to `python-docs/` folder
- Supported format: Sphinx-generated plain-text `.txt` files

### "Model download fails"
- Check internet connection
- `fastembed_cache/` may be corrupted — delete and retry

### "FAISS index unreadable"
- Delete `faiss_store_pydocs/` folder
- Run app again to rebuild index

### "Out of memory during indexing"
- Reduce `EMBED_BATCH_SIZE` from 64 to 32
- Modify in [app.py](app.py#L177)

## Performance Notes

- **Indexing Speed**: ~10 minutes for ~18K chunks on 8GB RAM
- **Query Speed**: ~2-3 seconds (rewrite + retrieval + inference)
- **Memory**: 3-4GB during indexing peak, ~1GB at runtime

## Deployment

### Recommended Platforms

- **[Streamlit Cloud](https://streamlit.io/cloud)** — Free, native Streamlit support
- **[Hugging Face Spaces](https://huggingface.co/spaces)** — Free tier available
- **[Railway](https://railway.app/)** — $5/month tier with persistent storage
- **[Render](https://render.com/)** — Free tier with limitations

**Not recommended:** Vercel (serverless, timeouts, ephemeral storage)

### Deploying to Streamlit Cloud

1. Push to GitHub branch
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Click "New app" → Connect your repo
4. Select branch `kalyan`
5. Set `GOOGLE_API_KEY` in Secrets
6. Deploy

## Additional Notes

### Gitignore Strategy

The following are intentionally ignored:

- `venv312/` — Virtual environment (regenerate with `pip install -r requirements.txt`)
- `faiss_store_pydocs/` — Built index (regenerate from python-docs/)
- `fastembed_cache/` — Model weights (auto-downloaded on first run)
- `chat_sessions/` — User data (local runtime state only)
- `.env` — Secrets (never commit credentials)

### Extending the App

To modify behavior:

1. **Change embedding model**: Update `EMBEDDING_MODEL_NAME` in app.py
2. **Adjust retrieval tuning**: Modify `RETRIEVAL_K`, `RERANK_POOL_K`, etc.
3. **Change LLM**: Replace `ChatGoogleGenerativeAI` in `get_llm()` function
4. **Custom prompts**: Edit the final prompt in the chat input section

---

**Built with ❤️ using Streamlit, LangChain, and FAISS**