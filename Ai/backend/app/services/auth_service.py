from sqlalchemy.orm import Session

from app.config import settings
from app.database.models import AdminUser
from app.utils.security import hash_password
from app.utils.logging import get_logger

logger = get_logger(__name__)


def seed_admin_user(db: Session) -> None:
    existing = db.query(AdminUser).filter(AdminUser.email == settings.admin_email).first()
    if existing:
        return
    admin = AdminUser(email=settings.admin_email, hashed_password=hash_password(settings.admin_password))
    db.add(admin)
    db.commit()
    logger.info("Seeded admin user %s", settings.admin_email)
