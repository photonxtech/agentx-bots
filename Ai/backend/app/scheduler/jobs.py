import asyncio

from app.database.models import Website, WebsiteStatus
from app.database.session import SessionLocal
from app.services.crawl_service import run_crawl
from app.utils.logging import get_logger

logger = get_logger(__name__)


def run_daily_sync() -> dict:
    db = SessionLocal()
    summary = {"total": 0, "succeeded": 0, "failed": 0}
    try:
        websites = db.query(Website).filter(Website.status != WebsiteStatus.crawling).all()
        summary["total"] = len(websites)
        for website in websites:
            try:
                asyncio.run(run_crawl(db, website, force=False))
                summary["succeeded"] += 1
            except Exception:
                logger.exception("Daily sync failed for website %s", website.id)
                summary["failed"] += 1
        logger.info("Daily sync summary: %s", summary)
        return summary
    finally:
        db.close()
