from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_admin, get_db
from app.api.schemas import CrawlJobRead
from app.database.models import AdminUser, CrawlJob, Website, WebsiteStatus
from app.database.session import SessionLocal
from app.services.crawl_service import run_crawl
from app.utils.logging import get_logger

router = APIRouter(tags=["crawl"])
logger = get_logger(__name__)


def _run_crawl_background(website_id: int, force: bool) -> None:
    import asyncio

    db = SessionLocal()
    try:
        website = db.query(Website).filter(Website.id == website_id).first()
        if website is None:
            logger.warning("Website %s vanished before background crawl started", website_id)
            return
        asyncio.run(run_crawl(db, website, force=force))
    finally:
        db.close()


def _trigger(website_id: int, force: bool, db: Session, background_tasks: BackgroundTasks) -> None:
    website = db.query(Website).filter(Website.id == website_id).first()
    if website is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Website not found")
    if website.status == WebsiteStatus.crawling:
        # A concurrent crawl/sync/reindex on the same website would run two writers
        # against the same ChromaDB collection at once, which corrupts its on-disk segment.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A crawl is already in progress for this website. Wait for it to finish before starting another.",
        )
    background_tasks.add_task(_run_crawl_background, website_id, force)


@router.post("/crawl/{website_id}", status_code=status.HTTP_202_ACCEPTED)
def crawl_website(
    website_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    _admin: AdminUser = Depends(get_current_admin),
) -> dict:
    _trigger(website_id, force=False, db=db, background_tasks=background_tasks)
    return {"detail": "Crawl started"}


@router.post("/sync/{website_id}", status_code=status.HTTP_202_ACCEPTED)
def sync_website(
    website_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    _admin: AdminUser = Depends(get_current_admin),
) -> dict:
    _trigger(website_id, force=False, db=db, background_tasks=background_tasks)
    return {"detail": "Sync started"}


@router.post("/reindex/{website_id}", status_code=status.HTTP_202_ACCEPTED)
def reindex_website(
    website_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    _admin: AdminUser = Depends(get_current_admin),
) -> dict:
    _trigger(website_id, force=True, db=db, background_tasks=background_tasks)
    return {"detail": "Reindex started"}


@router.get("/websites/{website_id}/crawl-jobs", response_model=list[CrawlJobRead])
def list_crawl_jobs(
    website_id: int,
    db: Session = Depends(get_db),
    _admin: AdminUser = Depends(get_current_admin),
) -> list[CrawlJob]:
    return (
        db.query(CrawlJob)
        .filter(CrawlJob.website_id == website_id)
        .order_by(CrawlJob.started_at.desc())
        .all()
    )
