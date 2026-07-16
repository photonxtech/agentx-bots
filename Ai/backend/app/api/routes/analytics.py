from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_current_admin, get_db
from app.api.schemas import AnalyticsRead
from app.database.models import AdminUser
from app.services.analytics_service import compute_analytics

router = APIRouter(tags=["analytics"])


@router.get("/analytics/{website_id}", response_model=AnalyticsRead)
def get_analytics(
    website_id: int,
    db: Session = Depends(get_db),
    _admin: AdminUser = Depends(get_current_admin),
) -> dict:
    return compute_analytics(db, website_id)
