import asyncio
from datetime import datetime

from sqlalchemy.orm import Session

from app.chunker.splitter import chunk_text
from app.config import settings
from app.crawler.discover import discover_urls
from app.database.models import CrawlJob, CrawlJobStatus, Page, PageStatus, Website, WebsiteStatus
from app.embeddings.embedder import embed_texts
from app.extractor.renderer import extract_page_content
from app.llm.bm25_index import invalidate_bm25_index
from app.services.cache_service import bump_generation
from app.utils.hashing import hash_content
from app.utils.logging import get_logger
from app.utils.text_dedup import dedupe_key
from app.utils.url_utils import normalize_url
from app.vectordb.collection import (
    delete_page_vectors,
    delete_website_collection,
    get_existing_chunk_keys,
    upsert_page_chunks,
)

logger = get_logger(__name__)


async def _process_url(db: Session, website: Website, url: str, force: bool, job: CrawlJob) -> None:
    url = normalize_url(url)
    existing_page = db.query(Page).filter(Page.website_id == website.id, Page.url == url).first()

    extracted = await extract_page_content(url)
    if extracted is None:
        if existing_page:
            existing_page.status = PageStatus.failed
            existing_page.last_crawled_at = datetime.utcnow()
        else:
            db.add(Page(website_id=website.id, url=url, status=PageStatus.failed, last_crawled_at=datetime.utcnow()))
        job.pages_failed += 1
        db.commit()
        return

    title, text = extracted
    new_hash = hash_content(text)

    if existing_page and existing_page.content_hash == new_hash and not force:
        existing_page.last_crawled_at = datetime.utcnow()
        db.commit()
        return

    all_chunks = chunk_text(text)

    # Cross-page dedup at ingest: skip storing a chunk that's a near-duplicate of one
    # already indexed for a DIFFERENT page on this site (e.g. the same client testimonial
    # repeated across many pages). If a page's entire content turns out to duplicate
    # content already indexed elsewhere, it legitimately contributes zero new chunks —
    # that content is already searchable via the page that owns it.
    existing_keys = get_existing_chunk_keys(website.id, exclude_page_id=existing_page.id if existing_page else None)
    chunks = [c for c in all_chunks if dedupe_key(c) not in existing_keys]

    embeddings = embed_texts(chunks) if chunks else []

    if existing_page:
        delete_page_vectors(website.id, existing_page.id)
        page = existing_page
        page.title = title
        page.content_hash = new_hash
        page.status = PageStatus.indexed
        page.last_crawled_at = datetime.utcnow()
        page.indexed_at = datetime.utcnow()
    else:
        page = Page(
            website_id=website.id, url=url, title=title, content_hash=new_hash,
            status=PageStatus.indexed, last_crawled_at=datetime.utcnow(), indexed_at=datetime.utcnow(),
        )
        db.add(page)
    db.commit()
    db.refresh(page)

    if chunks:
        upsert_page_chunks(website.id, page.id, url, title, new_hash, chunks, embeddings, indexed_at=page.indexed_at.isoformat())

    job.pages_indexed += 1
    db.commit()


async def run_crawl(db: Session, website: Website, force: bool = False) -> CrawlJob:
    job = CrawlJob(website_id=website.id, status=CrawlJobStatus.running)
    db.add(job)
    website.status = WebsiteStatus.crawling
    db.commit()
    db.refresh(job)

    try:
        urls, sitemap_used = await discover_urls(
            website.url, website.max_pages, website.crawl_depth_limit, settings.crawler_delay_seconds,
        )
        job.pages_found = len(urls)
        website.sitemap_url = sitemap_used
        db.commit()

        semaphore = asyncio.Semaphore(settings.crawler_concurrency)

        async def _bounded(url: str) -> None:
            async with semaphore:
                await _process_url(db, website, url, force, job)

        await asyncio.gather(*[_bounded(url) for url in urls])

        normalized_current = [normalize_url(u) for u in urls]
        stale_pages = (
            db.query(Page)
            .filter(Page.website_id == website.id, Page.status != PageStatus.deleted)
            .filter(~Page.url.in_(normalized_current))
            .all()
        )
        for page in stale_pages:
            delete_page_vectors(website.id, page.id)
            page.status = PageStatus.deleted
        db.commit()

        job.status = CrawlJobStatus.completed
        job.finished_at = datetime.utcnow()
        website.status = WebsiteStatus.active
        website.last_synced_at = datetime.utcnow()
    except Exception as exc:
        logger.exception("Crawl failed for website %s", website.id)
        job.status = CrawlJobStatus.failed
        job.error_log = str(exc)
        job.finished_at = datetime.utcnow()
        website.status = WebsiteStatus.error

    db.commit()
    # Content may have changed regardless of outcome — rebuild BM25 and stop serving any
    # cached retrieval/response for this site next time it's asked about.
    invalidate_bm25_index(website.id)
    bump_generation(website.id)
    return job


def delete_website_data(website_id: int) -> None:
    delete_website_collection(website_id)
    invalidate_bm25_index(website_id)
    bump_generation(website_id)
