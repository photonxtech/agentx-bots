# KnowledgeBot

A personal knowledge-memory Slack bot. Share links, documents, or images in a Slack DM or channel, then ask questions about the information you have shared. The bot stores the extracted content locally and answers questions using Groq.

## Setup

### Prerequisites

- Python 3.11 or newer
- A Slack app configured for Socket Mode
- A Groq API key
- `uv` (recommended) or `pip`

### Install

From this directory:

```powershell
cd KnowledgeBot

# Recommended: install from pyproject.toml and uv.lock
uv sync

# Or create a virtual environment and install with pip
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install .
```

Copy the environment template and fill in your credentials:

```powershell
Copy-Item env.example .env
```

Do not commit `.env`. It is ignored by `.gitignore`.

## Required Environment Variables

Set these values in `.env`:

| Variable | Required | Description |
| --- | --- | --- |
| `SLACK_BOT_TOKEN` | Yes | Slack bot token, usually beginning with `xoxb-`. |
| `SLACK_APP_TOKEN` | Yes | Slack app-level token for Socket Mode, usually beginning with `xapp-`. |
| `GROQ_API_KEY` | Yes | API key from the Groq Console. |
| `GROQ_TEXT_MODEL` | No | Groq text model. Defaults to `openai/gpt-oss-120b`. |
| `GROQ_VISION_MODEL` | No | Groq vision model used for image processing. Defaults to `qwen/qwen3.8-27b`. |
| `CHROMA_PATH` | No | Local Chroma persistence directory. Defaults to `./chroma_data`. |
| `RELEVANCE_THRESHOLD` | No | Similarity threshold for retrieved knowledge. Defaults to `0.45`. |
| `REDIS_URL` | No | Redis connection URL for optional content caching. Without it, the bot still works. |

The Slack app must have the event subscriptions and OAuth scopes required to receive messages, read shared files, add/remove reactions, and post messages. Enable Socket Mode and use the resulting app-level token as `SLACK_APP_TOKEN`.

## Run Locally

With the virtual environment active, run:

```powershell
python app.py
```

A successful startup logs `Bot is running...`. Add the bot to a Slack channel or open a DM with it, then share a URL/file/image or ask a question about previously stored content.

## API Endpoints

This project does not expose HTTP API endpoints. It connects to Slack through Socket Mode and responds to Slack `message` and `app_mention` events.

## Additional Notes

- Chroma data is stored locally in `CHROMA_PATH` and is excluded from Git by default.
- Redis is an optional performance optimization. It caches extracted content so repeated URLs or files do not need to be processed again.
- Link and file ingestion may use PDF, Office document, image, and OCR dependencies. EasyOCR may download model files the first time it processes an image.
- Groq model names can change. Update `GROQ_TEXT_MODEL` or `GROQ_VISION_MODEL` if a configured model is unavailable.
- The bot answers from the knowledge associated with the Slack user who shared it, rather than treating the local database as a shared public knowledge base.
