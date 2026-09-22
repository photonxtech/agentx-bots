"""Given a link or a downloaded Slack file, extract -> chunk -> store."""

import hashlib
import logging
import os
import re
import tempfile

from chunking import chunk_text
import extractors
import redis_cache

log = logging.getLogger("knowledgebot")

URL_RE = re.compile(r"https?://[^\s<>|]+")
SLACK_LINK_WRAPPER_RE = re.compile(r"<https?://[^>]*>")


def find_urls(text: str) -> list[str]:
    return URL_RE.findall(text or "")


def strip_urls_from_text(text: str, urls: list[str]) -> str:
    """Removes Slack's <url> / <url|label> markup (and any bare URLs) from
    a message, leaving behind whatever question text was sent alongside it.
    """
    cleaned = SLACK_LINK_WRAPPER_RE.sub("", text or "")
    for u in urls:
        cleaned = cleaned.replace(u, "")
    return cleaned.strip()


def is_shared_channel(channel_id: str) -> bool:
    """Slack DM channel ids start with 'D'; public/private channels start
    with 'C' or 'G'. Content shared in a DM is private to the sharer;
    content shared in a channel is visible to that channel.
    """
    return not (channel_id or "").startswith("D")


def ingest_url(
    store, answerer, groq_client, vision_model, user_id: str, channel_id: str, url: str
) -> int:
    content_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()

    if store.has_content(user_id, channel_id, content_hash):
        log.info(f"Already have this link for this user/scope — skipping duplicate: {url}")
        return 1  # nonzero so the caller correctly treats this as a success, not a failure

    cached = redis_cache.get_cached_url(url)
    if cached:
        log.info(f"Cache HIT for link: {url}")
        title, text, summary = cached["title"], cached["text"], cached["summary"]
    else:
        log.info(f"Cache MISS for link: {url}")
        title, text = extractors.extract_from_url(url, groq_client=groq_client, vision_model=vision_model)
        log.info(f"Extracted {len(text)} chars from link: {title!r}")
        summary = answerer.summarize(title, text)
        log.info("Generated summary chunk for link")
        redis_cache.set_cached_url(url, title, text, summary)

    chunks = [summary] + chunk_text(text)  # summary stored as its own retrievable chunk
    log.info(f"Chunked into {len(chunks)} piece(s) (including summary)")
    return store.add_document(
        user_id=user_id,
        chunks=chunks,
        source=url,
        source_type="link",
        title=title,
        channel_id=channel_id,
        is_shared=is_shared_channel(channel_id),
        content_hash=content_hash,
    )


def ingest_file(
    store, answerer, groq_client, vision_model, user_id: str, channel_id: str,
    bot_token: str, slack_file: dict,
) -> int:
    """slack_file is one entry from the Slack event's 'files' list."""
    file_url = slack_file["url_private_download"]
    filename = slack_file.get("name", "file")
    mimetype = slack_file.get("mimetype", "")
    ext = os.path.splitext(filename)[1].lower()

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = os.path.join(tmp_dir, filename)
        extractors.download_slack_file(file_url, bot_token, local_path)
        log.info(f"Downloaded {filename} from Slack")

        with open(local_path, "rb") as f:
            file_bytes = f.read()
        content_hash = hashlib.sha256(file_bytes).hexdigest()

        if store.has_content(user_id, channel_id, content_hash):
            log.info(f"Already have this exact file content for this user/scope — skipping duplicate: {filename}")
            return 1  # nonzero so the caller correctly treats this as a success, not a failure

        cached = redis_cache.get_cached_file(file_bytes)
        if cached:
            log.info(f"Cache HIT for file: {filename} (identical content seen before)")
            text, source_type, summary = cached["text"], cached["source_type"], cached["summary"]
        else:
            log.info(f"Cache MISS for file: {filename}")
            if mimetype.startswith("image/") or ext in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
                log.info("Running OCR + vision captioning on image...")
                text = extractors.extract_from_image(local_path, groq_client, vision_model)
                source_type = "image"
            elif ext == ".pdf":
                text = extractors.extract_from_pdf(local_path, groq_client=groq_client, vision_model=vision_model)
                source_type = "pdf"
            elif ext == ".docx":
                text = extractors.extract_from_docx(local_path)
                source_type = "docx"
            elif ext == ".csv":
                text = extractors.extract_from_csv(local_path)
                source_type = "csv"
            elif ext in (".xlsx", ".xls"):
                text = extractors.extract_from_excel(local_path)
                source_type = "excel"
            elif ext == ".pptx":
                text = extractors.extract_from_pptx(local_path)
                source_type = "pptx"
            elif ext in (".txt", ".md"):
                text = extractors.extract_from_txt(local_path)
                source_type = "txt"
            else:
                log.info(f"Skipping unsupported file type: {filename}")
                return 0

            log.info(f"Extracted {len(text)} chars from {source_type}: {filename}")
            summary = answerer.summarize(filename, text)
            log.info("Generated summary chunk for file")
            redis_cache.set_cached_file(file_bytes, text, source_type, summary)

        chunks = [summary] + chunk_text(text)
        log.info(f"Chunked into {len(chunks)} piece(s) (including summary)")
        return store.add_document(
            user_id=user_id,
            chunks=chunks,
            source=filename,
            source_type=source_type,
            title=filename,
            channel_id=channel_id,
            is_shared=is_shared_channel(channel_id),
            content_hash=content_hash,
        )