"""Personal knowledge-memory Slack bot.

Drop a link, document, or image into a DM or channel it's in, and the bot
silently ingests it into your own private knowledge base. Ask a question
any time after (minutes, days, whenever) and it answers using only what
you personally shared, via Groq.
"""

import logging
import os
import warnings

from dotenv import load_dotenv
from groq import Groq
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from groq_client import Answerer
from ingest import find_urls, ingest_file, ingest_url, strip_urls_from_text
import redis_cache
from vector_store import MemoryStore

# Suppress noisy-but-harmless warnings from dependencies (deprecated torch
# quantization API used internally by EasyOCR, and pdfminer's font-metadata
# warnings when a PDF has an incomplete font descriptor). Neither affects
# extraction quality; they're just internal library chatter.
warnings.filterwarnings("ignore", message=".*torch.quantize_per_tensor.*")
logging.getLogger("pdfminer").setLevel(logging.ERROR)

# Bot's own activity log — what it did, not internal library noise.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("knowledgebot")

load_dotenv(override=True)

SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_APP_TOKEN = os.environ["SLACK_APP_TOKEN"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
GROQ_TEXT_MODEL = os.environ.get("GROQ_TEXT_MODEL", "openai/gpt-oss-120b")
GROQ_VISION_MODEL = os.environ.get("GROQ_VISION_MODEL", "qwen/qwen3.8-27b")
CHROMA_PATH = os.environ.get("CHROMA_PATH", "./chroma_data")
RELEVANCE_THRESHOLD = float(os.environ.get("RELEVANCE_THRESHOLD", "0.45"))

app = App(token=SLACK_BOT_TOKEN)
store = MemoryStore(persist_path=CHROMA_PATH, relevance_threshold=RELEVANCE_THRESHOLD)
raw_groq_client = Groq(api_key=GROQ_API_KEY)

if redis_cache.is_available():
    log.info("Redis content cache: ACTIVE — repeated links/files will be reused, not reprocessed")
else:
    log.info("Redis content cache: NOT ACTIVE — every link/file is processed fresh (see .env.example)")
answerer = Answerer(api_key=GROQ_API_KEY, model=GROQ_TEXT_MODEL)

# Slack tags certain message events with these subtypes; we want to
# ignore edits/deletions/etc. but NOT file_share, which is how every
# file/image upload arrives.
IGNORED_SUBTYPES = {
    "message_changed",
    "message_deleted",
    "message_replied",
    "bot_message",
    "channel_join",
    "channel_leave",
}
REACT_STORED = "white_check_mark"
REACT_WORKING = "hourglass_flowing_sand"
REACT_FAILED = "x"


def handle_ingestion(event, user_id: str, urls: list[str], files: list[dict], channel: str, ts: str) -> bool:
    """Stores any links/files found. Returns True if anything was stored."""
    if not urls and not files:
        return False

    app.client.reactions_add(channel=channel, timestamp=ts, name=REACT_WORKING)
    stored_any = False

    for url in urls:
        log.info(f"Ingesting link from user {user_id}: {url}")
        try:
            n = ingest_url(store, answerer, raw_groq_client, GROQ_VISION_MODEL, user_id, channel, url)
            log.info(f"Stored {n} chunks from link: {url}")
            stored_any = stored_any or n > 0
        except Exception as e:
            log.error(f"Failed to ingest url {url}: {e}")

    for f in files:
        filename = f.get("name", "unknown file")
        log.info(f"Ingesting file from user {user_id}: {filename}")
        try:
            n = ingest_file(
                store, answerer, raw_groq_client, GROQ_VISION_MODEL, user_id, channel,
                SLACK_BOT_TOKEN, f,
            )
            log.info(f"Stored {n} chunks from file: {filename}")
            stored_any = stored_any or n > 0
        except Exception as e:
            log.error(f"Failed to ingest file {filename}: {e}")

    app.client.reactions_remove(channel=channel, timestamp=ts, name=REACT_WORKING)
    app.client.reactions_add(
        channel=channel, timestamp=ts, name=REACT_STORED if stored_any else REACT_FAILED
    )
    return stored_any


def handle_question(user_id: str, text: str, channel: str, ingestion_just_failed: bool = False):
    log.info(f"Question from user {user_id}: {text!r}")
    hits = store.query(user_id=user_id, question=text, channel_id=channel, top_k=5)
    log.info(f"Retrieved {len(hits)} matching chunk(s) above the relevance threshold")
    answer = answerer.answer(text, hits)

    if ingestion_just_failed:
        # Don't let a failed ingestion masquerade as an answer about it —
        # say so explicitly, whatever the RAG answer ends up being.
        answer = (
            "⚠️ I couldn't read what you just shared, so this isn't based on it "
            "(the site may block scraping, or need JavaScript I can't run). "
            "Here's what I could answer from your other stored knowledge, if anything:\n\n"
            + answer
        )

    log.info(f"Answered user {user_id} ({len(answer)} chars)")
    # Posted as a fresh message, NOT threaded.
    app.client.chat_postMessage(channel=channel, text=answer)


def process_message(event, user_id: str, text: str, channel: str, ts: str):
    """Ingests any link/file in the message, AND answers any question that
    was sent alongside it in the same message — both, not either/or.
    """
    urls = find_urls(text)
    files = event.get("files", [])
    had_content_to_ingest = bool(urls or files)
    ingestion_succeeded = True

    if had_content_to_ingest:
        ingestion_succeeded = handle_ingestion(event, user_id, urls, files, channel, ts)
        # Whatever text is left after stripping the link markup out is a
        # question sent alongside the link/file, if there is one.
        question_text = strip_urls_from_text(text, urls)
    else:
        question_text = text.strip()

    # Ignore trivial leftovers (e.g. just punctuation) so a bare link drop
    # doesn't trigger a pointless "answer" with no real question in it.
    if len(question_text) > 2:
        handle_question(
            user_id, question_text, channel,
            ingestion_just_failed=had_content_to_ingest and not ingestion_succeeded,
        )


@app.event("message")
def on_message(event, say):
    # Ignore bot's own messages and edit/delete/etc. subtypes — but NOT
    # file_share, which is how file/image uploads arrive.
    if event.get("bot_id") or event.get("subtype") in IGNORED_SUBTYPES:
        return

    user_id = event.get("user")
    text = event.get("text", "")
    channel = event["channel"]
    ts = event["ts"]

    if not user_id:
        return

    process_message(event, user_id, text, channel, ts)


@app.event("app_mention")
def on_mention(event, say):
    user_id = event.get("user")
    channel = event["channel"]
    text = event.get("text", "")

    if not user_id:
        return

    process_message(event, user_id, text, channel, event["ts"])


if __name__ == "__main__":
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    log.info("Bot is running...")
    handler.start()