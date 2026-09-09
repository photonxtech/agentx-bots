"""
Database connection and session management.
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

load_dotenv()

def _database_url() -> str:
    """Resolve the connection string.

    DATABASE_URL wins when set, which covers local development and any host that
    hands over a ready-made URL.

    Otherwise assemble one from discrete DB_* parts. This exists for AWS: an
    RDS-managed secret stores `username`, `password`, `host`, `port` and
    `dbname` as separate JSON keys and has no assembled URL to inject, so the
    parts are the only thing the task definition can pass through without the
    password ever being written into an environment variable in plaintext.
    """
    url = os.getenv("DATABASE_URL", "").strip()
    if url:
        return url

    host = os.getenv("DB_HOST", "").strip()
    if host:
        from urllib.parse import quote_plus

        user = quote_plus(os.getenv("DB_USER", "postgres"))
        # Quoted because RDS generates passwords containing characters that
        # would otherwise be read as URL delimiters.
        password = quote_plus(os.getenv("DB_PASSWORD", ""))
        port = os.getenv("DB_PORT", "5432")
        name = os.getenv("DB_NAME", "rag_eval")
        return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{name}"

    return "postgresql://postgres:postgres@localhost:5432/rag_eval"


DATABASE_URL = _database_url()

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300,
    connect_args={
        "options": "-c statement_timeout=30000 -c idle_in_transaction_session_timeout=15000"
    } if "postgresql" in DATABASE_URL else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI dependency — yields a DB session and closes it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
