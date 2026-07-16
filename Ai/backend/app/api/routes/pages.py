from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.api.schemas import PageRead
from app.database.models import Page

router = APIRouter(tags=["pages"])


@router.get("/pages/{website_id}", response_model=list[PageRead])
def list_pages(
    website_id: int,
    status: str | None = Query(default=None),
    db: Session = Depends(get_db),
) -> list[Page]:
    query = db.query(Page).filter(Page.website_id == website_id)
    if status:
        query = query.filter(Page.status == status)
    return query.order_by(Page.indexed_at.desc().nullslast()).all()
