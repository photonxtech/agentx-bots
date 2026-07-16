from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.deps import get_current_admin, get_db
from app.api.schemas import WebsiteCreate, WebsiteRead, WebsiteUpdate
from app.database.models import AdminUser, Website
from app.services.crawl_service import delete_website_data

router = APIRouter(tags=["websites"])


@router.post("/websites", response_model=WebsiteRead, status_code=status.HTTP_201_CREATED)
def create_website(
    payload: WebsiteCreate,
    db: Session = Depends(get_db),
    _admin: AdminUser = Depends(get_current_admin),
) -> Website:
    website = Website(
        url=str(payload.url),
        name=payload.name,
        logo_url=str(payload.logo_url) if payload.logo_url else None,
        crawl_depth_limit=payload.crawl_depth_limit,
        max_pages=payload.max_pages,
    )
    db.add(website)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Website URL already registered")
    db.refresh(website)
    return website


@router.get("/websites", response_model=list[WebsiteRead])
def list_websites(db: Session = Depends(get_db)) -> list[Website]:
    return db.query(Website).order_by(Website.created_at.desc()).all()


@router.patch("/websites/{website_id}", response_model=WebsiteRead)
def update_website(
    website_id: int,
    payload: WebsiteUpdate,
    db: Session = Depends(get_db),
    _admin: AdminUser = Depends(get_current_admin),
) -> Website:
    website = db.query(Website).filter(Website.id == website_id).first()
    if website is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Website not found")

    updates = payload.model_dump(exclude_unset=True)
    for field, value in updates.items():
        setattr(website, field, str(value) if field == "logo_url" and value else value)

    db.commit()
    db.refresh(website)
    return website


@router.delete("/website/{website_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_website(
    website_id: int,
    db: Session = Depends(get_db),
    _admin: AdminUser = Depends(get_current_admin),
) -> None:
    website = db.query(Website).filter(Website.id == website_id).first()
    if website is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Website not found")

    delete_website_data(website_id)
    db.delete(website)
    db.commit()
