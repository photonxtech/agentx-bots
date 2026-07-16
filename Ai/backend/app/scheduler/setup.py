from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from app.config import settings
from app.scheduler.jobs import run_daily_sync
from app.utils.logging import get_logger

logger = get_logger(__name__)

_scheduler: BackgroundScheduler | None = None


def start_scheduler() -> BackgroundScheduler:
    global _scheduler
    if _scheduler is not None:
        return _scheduler

    scheduler = BackgroundScheduler()
    scheduler.add_job(
        run_daily_sync,
        trigger=CronTrigger(hour=settings.daily_sync_hour, minute=0),
        id="daily_sync",
        replace_existing=True,
    )
    scheduler.start()
    logger.info("Scheduler started; daily sync scheduled at %02d:00", settings.daily_sync_hour)
    _scheduler = scheduler
    return scheduler


def shutdown_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
