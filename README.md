# agentx-bots

AgentX Bots — a small collection of tools and a demo RAG (Retrieval-Augmented Generation)
project under `WeNextAI_RAG` that demonstrates document ingestion, vector storage
with Chroma, and simple query / app interfaces.

**Summary**
- **Project:** WeNextAI_RAG (RAG demo with Chroma DB)
- **Language:** Python 3.12+
- **Virtualenv:** `ragenv` (included)

## Repository structure

- `WeNextAI_RAG/` — main demo app and scripts
	- `app.py` — demo application (UI / server)
	- `ingest.py` — ingest documents into Chroma vector DB
	- `query.py` — example query client against the vector DB
	- `rag_utils.py` — helper utilities for RAG workflows
	- `requirements.txt` — Python dependencies for the demo
- `chroma_db/` — local Chroma DB files produced by ingestion

## Quickstart

Prerequisites

- Python 3.12 (recommended)
- Git

Setup

1. Activate the provided virtual environment (recommended):

	 - macOS / Linux:

		 ```bash
		 source WeNextAI_RAG/ragenv/bin/activate
		 ```

	 - Or create/activate your own virtualenv and install requirements:

		 ```bash
		 python3 -m venv .venv
		 source .venv/bin/activate
		 pip install -r WeNextAI_RAG/requirements.txt
		 ```

2. Ingest documents (creates/updates `WeNextAI_RAG/chroma_db`):

	 ```bash
	 python WeNextAI_RAG/ingest.py
	 ```

3. Run an example query or the app:

	 ```bash
	 python WeNextAI_RAG/query.py
	 # or
	 python WeNextAI_RAG/app.py
	 ```

Notes

- The demo uses Chroma for vector storage; DB files are stored under
	`WeNextAI_RAG/chroma_db/` and under the top-level `chroma_db/` directory in
	the repository tree depending on how the demo was run.
- If an interactive UI is present in `app.py` it may use Streamlit or a simple
	web server. See the top of `WeNextAI_RAG/app.py` for exact run instructions.

## Configuration

- If the demo requires API keys (OpenAI, Hugging Face, etc.), set them as
	environment variables before running the scripts. Example:

	```bash
	export OPENAI_API_KEY="your_key_here"
	```

## Contributing

- Fork the repo, create a branch, make changes, and open a pull request.
- Please avoid committing large data files; add them to `.gitignore` if needed.

## License & Contact

This project is provided as-is. For questions or access issues, contact the
maintainer or the repository owner on GitHub: `photonxtech/agentx-bots`.

---
Updated: 2026-07-16