# Multi-Website AI Chatbot Platform Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a production-ready platform where an admin registers website URLs, the system crawls/extracts/chunks/embeds their content into per-website ChromaDB collections, and end users chat with a floating widget that answers only from that site's indexed content (with sources, confidence, and a fixed no-answer fallback), kept in sync daily.

**Architecture:** FastAPI backend (SQLAlchemy/SQLite for metadata, ChromaDB for vectors, LangChain + Groq for RAG, Playwright/Trafilatura/BeautifulSoup for extraction, APScheduler for daily sync) behind a thin routes→services→modules layering. React/Vite/TS/MUI frontend with two areas: `/admin/*` dashboard and `/` demo page hosting the chat widget.

**Tech Stack:** Python 3.11+, FastAPI, SQLAlchemy, SQLite, ChromaDB, LangChain, Groq API (`llama-3.3-70b-versatile`), HuggingFace `sentence-transformers/all-MiniLM-L6-v2`, Playwright, BeautifulSoup4, Trafilatura, APScheduler, PyJWT, passlib. React, Vite, TypeScript, MUI, React Query, Axios, Framer Motion, React Markdown.

Design spec: `docs/superpowers/specs/2026-07-09-multi-website-ai-chatbot-platform-design.md`

## Global Constraints

- Everything lives under `Ai/` (no Docker). Backend uses a local Python venv; frontend uses npm.
- Chunking: `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)` — exact values, non-negotiable per spec.
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` via LangChain `HuggingFaceEmbeddings`, loaded once as a process-wide singleton.
- LLM: Groq `llama-3.3-70b-versatile`. Must never answer outside retrieved context; on zero surviving chunks, return exactly `"I couldn't find that information on this website."` and skip the Groq call.
- Crawl defaults (env-overridable): `MAX_PAGES=500`, `MAX_CRAWL_DEPTH=5`, `CRAWLER_CONCURRENCY=2`, `CRAWLER_DELAY_SECONDS=1`.
- Similarity threshold for retrieval filtering: env-overridable, default `0.3` (cosine similarity, chunks below this are dropped before prompting).
- One ChromaDB collection per website, named `website_{id}`, persisted to `backend/data/chroma/`.
- SQLite file at `backend/data/app.db`. No Alembic — `Base.metadata.create_all()` on startup is sufficient for this scope.
- All secrets/config via `.env` (never hardcoded): `GROQ_API_KEY`, `JWT_SECRET`, `ADMIN_EMAIL`, `ADMIN_PASSWORD`, `CORS_ORIGINS`, plus the crawl/similarity env vars above.
- Admin-mutating routes require JWT bearer auth; `GET /websites`, `GET /pages/{id}`, `POST /chat`, conversation/feedback routes stay public.
- Every backend module gets structured logging via Python `logging` (module-level logger, configurable level via `LOG_LEVEL` env var).
- Network-facing calls (Playwright navigation, Groq requests, HF embedding calls on first load) go through a shared retry-with-backoff decorator.
- No automated tests for genuinely I/O-bound integration points (live Playwright rendering, real Groq streaming, ChromaDB persistence with the real embedding model, browser UI) — those get an explicit manual smoke-test step instead. Pure/logic-heavy code (URL filtering, hashing, chunking config, JWT, similarity/confidence math, HTML cleaning, route contracts with mocked services) gets real pytest unit/integration tests.
- Frontend has no test runner in the requested stack (Vite/React/TS/MUI/React Query/Axios/Framer Motion/React Markdown only) — UI tasks are verified by running the dev server and exercising the flow manually, per each task's step.

---

## Backend Tasks

### Task 1: Backend scaffold — venv, dependencies, config, logging

**Files:**
- Create: `Ai/backend/requirements.txt`
- Create: `Ai/backend/.env.example`
- Create: `Ai/backend/app/__init__.py`
- Create: `Ai/backend/app/config.py`
- Create: `Ai/backend/app/utils/__init__.py`
- Create: `Ai/backend/app/utils/logging.py`
- Create: `Ai/backend/.gitignore`

**Interfaces:**
- Produces: `app.config.settings` (a `Settings` instance), `app.utils.logging.get_logger(name: str) -> logging.Logger`. All later tasks import `from app.config import settings` and `from app.utils.logging import get_logger`.

- [ ] **Step 1: Create the venv and directory skeleton**

```bash
cd "Ai/backend"
python3 -m venv venv
source venv/bin/activate
mkdir -p app/database app/api/routes app/crawler app/extractor app/chunker app/embeddings app/vectordb app/llm app/scheduler app/services app/utils data
touch app/__init__.py app/database/__init__.py app/api/__init__.py app/api/routes/__init__.py \
      app/crawler/__init__.py app/extractor/__init__.py app/chunker/__init__.py app/embeddings/__init__.py \
      app/vectordb/__init__.py app/llm/__init__.py app/scheduler/__init__.py app/services/__init__.py app/utils/__init__.py
```

- [ ] **Step 2: Write `requirements.txt`**

```
fastapi==0.115.6
uvicorn[standard]==0.34.0
sqlalchemy==2.0.36
pydantic==2.10.4
pydantic-settings==2.7.1
python-dotenv==1.0.1
chromadb==0.5.23
langchain==0.3.13
langchain-community==0.3.13
langchain-huggingface==0.1.2
groq==0.15.0
sentence-transformers==3.3.1
playwright==1.49.1
beautifulsoup4==4.12.3
trafilatura==2.0.0
apscheduler==3.10.4
pyjwt==2.10.1
passlib[bcrypt]==1.7.4
python-multipart==0.0.20
httpx==0.28.1
tenacity==9.0.0
defusedxml==0.7.1
pytest==8.3.4
pytest-asyncio==0.25.0
```

Install:
```bash
pip install -r requirements.txt
playwright install chromium
```

- [ ] **Step 3: Write `.env.example`**

```
# Groq
GROQ_API_KEY=your-groq-api-key

# Auth
JWT_SECRET=change-this-to-a-random-secret
JWT_EXPIRE_MINUTES=480
ADMIN_EMAIL=admin@example.com
ADMIN_PASSWORD=change-this-password

# CORS
CORS_ORIGINS=http://localhost:5173

# Database / vector store
DATABASE_URL=sqlite:///./data/app.db
CHROMA_PERSIST_DIR=./data/chroma

# Crawl limits
MAX_PAGES=500
MAX_CRAWL_DEPTH=5
CRAWLER_CONCURRENCY=2
CRAWLER_DELAY_SECONDS=1

# RAG
SIMILARITY_THRESHOLD=0.3
TOP_K_CHUNKS=5
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
GROQ_MODEL=llama-3.3-70b-versatile

# Scheduler
DAILY_SYNC_HOUR=3

LOG_LEVEL=INFO
```

- [ ] **Step 4: Write `app/config.py`**

```python
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    groq_api_key: str
    jwt_secret: str
    jwt_expire_minutes: int = 480
    admin_email: str
    admin_password: str

    cors_origins: str = "http://localhost:5173"

    database_url: str = "sqlite:///./data/app.db"
    chroma_persist_dir: str = "./data/chroma"

    max_pages: int = 500
    max_crawl_depth: int = 5
    crawler_concurrency: int = 2
    crawler_delay_seconds: float = 1.0

    similarity_threshold: float = 0.3
    top_k_chunks: int = 5
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    groq_model: str = "llama-3.3-70b-versatile"

    daily_sync_hour: int = 3
    log_level: str = "INFO"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
```

- [ ] **Step 5: Write `app/utils/logging.py`**

```python
import logging
import sys

from app.config import settings

_CONFIGURED = False


def _configure_root() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        stream=sys.stdout,
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    _configure_root()
    return logging.getLogger(name)
```

- [ ] **Step 6: Write `.gitignore`**

```
venv/
__pycache__/
*.pyc
data/
.env
.pytest_cache/
```

- [ ] **Step 7: Verify config loads**

Copy `.env.example` to `.env`, fill in a placeholder `GROQ_API_KEY=test`, `JWT_SECRET=test-secret`, `ADMIN_EMAIL=admin@example.com`, `ADMIN_PASSWORD=test1234`, then run:

```bash
cd Ai/backend && source venv/bin/activate
python -c "from app.config import settings; print(settings.database_url, settings.max_pages)"
```
Expected: prints `sqlite:///./data/app.db 500` with no errors.

- [ ] **Step 8: Commit**

```bash
git add Ai/backend/requirements.txt Ai/backend/.env.example Ai/backend/.gitignore Ai/backend/app
git commit -m "feat: backend scaffold with config and logging"
```

---

### Task 2: Database models & session

**Files:**
- Create: `Ai/backend/app/database/session.py`
- Create: `Ai/backend/app/database/models.py`
- Test: `Ai/backend/tests/test_models.py`

**Interfaces:**
- Consumes: `app.config.settings.database_url`
- Produces: `app.database.session.engine`, `app.database.session.SessionLocal`, `app.database.session.get_db()` (FastAPI dependency, yields a `Session`), `app.database.session.init_db()` (calls `Base.metadata.create_all`). Models: `Base`, `Website`, `Page`, `Conversation`, `Message`, `AdminUser`, `CrawlJob` — exact field names below, used verbatim by every later backend task.

- [ ] **Step 1: Write `app/database/session.py`**

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings

connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


def get_db():
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    from app.database import models  # noqa: F401  (ensure models are registered)

    Base.metadata.create_all(bind=engine)
```

- [ ] **Step 2: Write `app/database/models.py`**

```python
import enum
from datetime import datetime

from sqlalchemy import JSON, DateTime, Enum, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.session import Base


class WebsiteStatus(str, enum.Enum):
    active = "active"
    crawling = "crawling"
    error = "error"


class PageStatus(str, enum.Enum):
    indexed = "indexed"
    failed = "failed"
    skipped = "skipped"
    deleted = "deleted"


class MessageRole(str, enum.Enum):
    user = "user"
    assistant = "assistant"


class Feedback(str, enum.Enum):
    up = "up"
    down = "down"


class CrawlJobStatus(str, enum.Enum):
    running = "running"
    completed = "completed"
    failed = "failed"


class Website(Base):
    __tablename__ = "websites"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    url: Mapped[str] = mapped_column(String(2048), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    logo_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    status: Mapped[WebsiteStatus] = mapped_column(Enum(WebsiteStatus), default=WebsiteStatus.active)
    sitemap_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    crawl_depth_limit: Mapped[int] = mapped_column(Integer, default=5)
    max_pages: Mapped[int] = mapped_column(Integer, default=500)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_synced_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    pages: Mapped[list["Page"]] = relationship(back_populates="website", cascade="all, delete-orphan")
    conversations: Mapped[list["Conversation"]] = relationship(back_populates="website", cascade="all, delete-orphan")
    crawl_jobs: Mapped[list["CrawlJob"]] = relationship(back_populates="website", cascade="all, delete-orphan")


class Page(Base):
    __tablename__ = "pages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    website_id: Mapped[int] = mapped_column(ForeignKey("websites.id"), nullable=False)
    url: Mapped[str] = mapped_column(String(2048), nullable=False)
    title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[PageStatus] = mapped_column(Enum(PageStatus), default=PageStatus.skipped)
    last_crawled_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    website: Mapped["Website"] = relationship(back_populates="pages")


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    website_id: Mapped[int] = mapped_column(ForeignKey("websites.id"), nullable=False)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False)
    title: Mapped[str] = mapped_column(String(255), default="New Chat")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    website: Mapped["Website"] = relationship(back_populates="conversations")
    messages: Mapped[list["Message"]] = relationship(back_populates="conversation", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id"), nullable=False)
    role: Mapped[MessageRole] = mapped_column(Enum(MessageRole), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    sources: Mapped[list | None] = mapped_column(JSON, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    feedback: Mapped[Feedback | None] = mapped_column(Enum(Feedback), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")


class AdminUser(Base):
    __tablename__ = "admin_users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)


class CrawlJob(Base):
    __tablename__ = "crawl_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    website_id: Mapped[int] = mapped_column(ForeignKey("websites.id"), nullable=False)
    status: Mapped[CrawlJobStatus] = mapped_column(Enum(CrawlJobStatus), default=CrawlJobStatus.running)
    pages_found: Mapped[int] = mapped_column(Integer, default=0)
    pages_indexed: Mapped[int] = mapped_column(Integer, default=0)
    pages_failed: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    error_log: Mapped[str | None] = mapped_column(Text, nullable=True)

    website: Mapped["Website"] = relationship(back_populates="crawl_jobs")
```

- [ ] **Step 3: Write the test**

```python
# Ai/backend/tests/test_models.py
import os
import tempfile

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture()
def db_session():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    engine = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False})
    from app.database.session import Base
    from app.database import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    os.remove(path)


def test_website_page_relationship(db_session):
    from app.database.models import Page, Website

    site = Website(url="https://example.com", name="Example")
    db_session.add(site)
    db_session.commit()

    page = Page(website_id=site.id, url="https://example.com/about", content_hash="abc123")
    db_session.add(page)
    db_session.commit()

    fetched = db_session.query(Website).first()
    assert fetched.pages[0].url == "https://example.com/about"


def test_conversation_message_relationship(db_session):
    from app.database.models import Conversation, Message, MessageRole, Website

    site = Website(url="https://example.com", name="Example")
    db_session.add(site)
    db_session.commit()

    convo = Conversation(website_id=site.id, session_id="s1")
    db_session.add(convo)
    db_session.commit()

    msg = Message(conversation_id=convo.id, role=MessageRole.user, content="Hi")
    db_session.add(msg)
    db_session.commit()

    assert db_session.query(Conversation).first().messages[0].content == "Hi"
```

- [ ] **Step 4: Run the tests**

```bash
cd Ai/backend && source venv/bin/activate
pytest tests/test_models.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add Ai/backend/app/database Ai/backend/tests/test_models.py
git commit -m "feat: SQLAlchemy models and session management"
```

---

### Task 3: Utils — hashing, retry decorator, URL filtering

**Files:**
- Create: `Ai/backend/app/utils/hashing.py`
- Create: `Ai/backend/app/utils/retry.py`
- Create: `Ai/backend/app/utils/url_utils.py`
- Test: `Ai/backend/tests/test_utils.py`

**Interfaces:**
- Produces: `hash_content(text: str) -> str` (SHA256 hex digest), `with_retry(max_attempts: int = 3, base_delay: float = 1.0)` (decorator factory using `tenacity`), `is_same_domain(url: str, base_domain: str) -> bool`, `is_ignorable_url(url: str) -> bool` (True for mailto/tel/images/video/pdf/zip/css/js), `normalize_url(url: str) -> str` (strips fragments/trailing slash for dedup). Used by the crawler (Task 5), extractor (Task 6), and sync service (Task 14).

- [ ] **Step 1: Write the failing tests**

```python
# Ai/backend/tests/test_utils.py
from app.utils.hashing import hash_content
from app.utils.url_utils import is_ignorable_url, is_same_domain, normalize_url


def test_hash_content_deterministic():
    assert hash_content("hello") == hash_content("hello")
    assert hash_content("hello") != hash_content("world")


def test_hash_content_matches_known_sha256():
    import hashlib

    assert hash_content("hello") == hashlib.sha256("hello".encode("utf-8")).hexdigest()


def test_is_same_domain():
    assert is_same_domain("https://example.com/about", "example.com") is True
    assert is_same_domain("https://sub.example.com/x", "example.com") is False
    assert is_same_domain("https://other.com/x", "example.com") is False


def test_is_ignorable_url_extensions():
    assert is_ignorable_url("https://example.com/photo.jpg") is True
    assert is_ignorable_url("https://example.com/doc.pdf") is True
    assert is_ignorable_url("https://example.com/app.zip") is True
    assert is_ignorable_url("https://example.com/style.css") is True
    assert is_ignorable_url("https://example.com/script.js") is True
    assert is_ignorable_url("https://example.com/about") is False


def test_is_ignorable_url_schemes():
    assert is_ignorable_url("mailto:hi@example.com") is True
    assert is_ignorable_url("tel:+123456789") is True
    assert is_ignorable_url("https://example.com/about") is False


def test_normalize_url_strips_fragment_and_trailing_slash():
    assert normalize_url("https://example.com/about/#section") == "https://example.com/about"
    assert normalize_url("https://example.com/about/") == "https://example.com/about"
    assert normalize_url("https://example.com") == "https://example.com"
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd Ai/backend && source venv/bin/activate
pytest tests/test_utils.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'app.utils.hashing'` (and similar).

- [ ] **Step 3: Write `app/utils/hashing.py`**

```python
import hashlib


def hash_content(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
```

- [ ] **Step 4: Write `app/utils/url_utils.py`**

```python
from urllib.parse import urldefrag, urlparse

IGNORED_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico", ".bmp",
    ".mp4", ".webm", ".mov", ".avi",
    ".pdf", ".zip", ".rar", ".tar", ".gz", ".7z",
    ".css", ".js", ".mjs",
    ".woff", ".woff2", ".ttf", ".eot",
)
IGNORED_SCHEMES = ("mailto", "tel", "javascript")


def is_ignorable_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme in IGNORED_SCHEMES:
        return True
    if parsed.scheme not in ("http", "https", ""):
        return True
    path = parsed.path.lower()
    return path.endswith(IGNORED_EXTENSIONS)


def is_same_domain(url: str, base_domain: str) -> bool:
    return urlparse(url).netloc.lower() == base_domain.lower()


def normalize_url(url: str) -> str:
    url, _ = urldefrag(url)
    if url.endswith("/") and urlparse(url).path != "/":
        url = url[:-1]
    return url


def get_domain(url: str) -> str:
    return urlparse(url).netloc.lower()
```

- [ ] **Step 5: Write `app/utils/retry.py`**

```python
from tenacity import retry, stop_after_attempt, wait_exponential

from app.utils.logging import get_logger

logger = get_logger(__name__)


def with_retry(max_attempts: int = 3, base_delay: float = 1.0):
    def decorator(func):
        wrapped = retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=base_delay, min=base_delay, max=base_delay * 10),
            reraise=True,
        )(func)
        return wrapped

    return decorator
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
pytest tests/test_utils.py -v
```
Expected: 7 passed.

- [ ] **Step 7: Commit**

```bash
git add Ai/backend/app/utils Ai/backend/tests/test_utils.py
git commit -m "feat: hashing, retry decorator, and URL filtering utils"
```

---

### Task 4: Auth — JWT, password hashing, login route, admin seeding

**Files:**
- Create: `Ai/backend/app/utils/security.py`
- Create: `Ai/backend/app/api/deps.py`
- Create: `Ai/backend/app/api/routes/auth.py`
- Create: `Ai/backend/app/services/auth_service.py`
- Test: `Ai/backend/tests/test_auth.py`

**Interfaces:**
- Consumes: `app.database.models.AdminUser`, `app.database.session.get_db`
- Produces: `hash_password(plain: str) -> str`, `verify_password(plain: str, hashed: str) -> bool`, `create_access_token(subject: str) -> str`, `decode_access_token(token: str) -> dict` (raises `jwt.InvalidTokenError` on failure), `app.api.deps.get_current_admin(...)` (FastAPI dependency raising `HTTPException(401)` if missing/invalid token — imported by every admin-protected route from Task 11 onward), `app.services.auth_service.seed_admin_user(db: Session) -> None`, `router` (APIRouter, mounted at `/auth`) exposing `POST /auth/login`.

- [ ] **Step 1: Write the failing tests**

```python
# Ai/backend/tests/test_auth.py
import pytest

from app.utils.security import create_access_token, decode_access_token, hash_password, verify_password


def test_password_hash_roundtrip():
    hashed = hash_password("s3cret!")
    assert hashed != "s3cret!"
    assert verify_password("s3cret!", hashed) is True
    assert verify_password("wrong", hashed) is False


def test_access_token_roundtrip():
    token = create_access_token("admin@example.com")
    payload = decode_access_token(token)
    assert payload["sub"] == "admin@example.com"


def test_decode_invalid_token_raises():
    import jwt

    with pytest.raises(jwt.InvalidTokenError):
        decode_access_token("not-a-real-token")
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_auth.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'app.utils.security'`.

- [ ] **Step 3: Write `app/utils/security.py`**

```python
from datetime import datetime, timedelta, timezone

import jwt
from passlib.context import CryptContext

from app.config import settings

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
JWT_ALGORITHM = "HS256"


def hash_password(plain: str) -> str:
    return _pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return _pwd_context.verify(plain, hashed)


def create_access_token(subject: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expire_minutes)
    payload = {"sub": subject, "exp": expire}
    return jwt.encode(payload, settings.jwt_secret, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict:
    return jwt.decode(token, settings.jwt_secret, algorithms=[JWT_ALGORITHM])
```

- [ ] **Step 4: Write `app/services/auth_service.py`**

```python
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
```

- [ ] **Step 5: Write `app/api/deps.py`**

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

import jwt

from app.database.session import get_db, SessionLocal
from app.database.models import AdminUser
from app.utils.security import decode_access_token

_bearer_scheme = HTTPBearer()


def get_current_admin(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
    db: Session = Depends(get_db),
) -> AdminUser:
    try:
        payload = decode_access_token(credentials.credentials)
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")

    admin = db.query(AdminUser).filter(AdminUser.email == payload["sub"]).first()
    if admin is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Admin not found")
    return admin


__all__ = ["get_db", "get_current_admin", "SessionLocal"]
```

- [ ] **Step 6: Write `app/api/routes/auth.py`**

```python
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.database.models import AdminUser
from app.utils.security import create_access_token, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    email: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


@router.post("/login", response_model=LoginResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)) -> LoginResponse:
    admin = db.query(AdminUser).filter(AdminUser.email == payload.email).first()
    if admin is None or not verify_password(payload.password, admin.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")
    token = create_access_token(admin.email)
    return LoginResponse(access_token=token)
```

- [ ] **Step 7: Run tests to verify they pass**

```bash
pytest tests/test_auth.py -v
```
Expected: 3 passed.

- [ ] **Step 8: Commit**

```bash
git add Ai/backend/app/utils/security.py Ai/backend/app/services/auth_service.py Ai/backend/app/api/deps.py Ai/backend/app/api/routes/auth.py Ai/backend/tests/test_auth.py
git commit -m "feat: JWT auth, login route, and admin seeding"
```

---

### Task 5: Crawler — sitemap discovery + recursive fallback + URL filtering

**Files:**
- Create: `Ai/backend/app/crawler/sitemap.py`
- Create: `Ai/backend/app/crawler/recursive.py`
- Create: `Ai/backend/app/crawler/discover.py`
- Test: `Ai/backend/tests/test_crawler.py`

**Interfaces:**
- Consumes: `app.utils.url_utils.{is_ignorable_url, is_same_domain, normalize_url, get_domain}`, `app.utils.retry.with_retry`
- Produces: `parse_sitemap_xml(xml_text: str, base_domain: str) -> list[str]` (pure), `extract_links(html: str, base_url: str, base_domain: str) -> list[str]` (pure), `async fetch_sitemap_urls(client: httpx.AsyncClient, base_url: str) -> list[str] | None`, `async recursive_crawl(client: httpx.AsyncClient, base_url: str, max_pages: int, max_depth: int, delay_seconds: float) -> list[str]`, `async discover_urls(base_url: str, max_pages: int, max_depth: int, delay_seconds: float) -> tuple[list[str], str | None]` (returns `(urls, sitemap_url_used_or_None)`). `discover_urls` is the single entry point consumed by the crawl service (Task 10).

- [ ] **Step 1: Write the failing tests (pure functions only)**

```python
# Ai/backend/tests/test_crawler.py
from app.crawler.recursive import extract_links
from app.crawler.sitemap import parse_sitemap_xml

SITEMAP_XML = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://example.com/</loc></url>
  <url><loc>https://example.com/about</loc></url>
  <url><loc>https://example.com/pricing.pdf</loc></url>
  <url><loc>https://external.com/other</loc></url>
</urlset>
"""

SITEMAP_INDEX_XML = """<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <sitemap><loc>https://example.com/sitemap-1.xml</loc></sitemap>
</sitemapindex>
"""

HTML_WITH_LINKS = """
<html><body>
  <nav><a href="/nav-link">Nav</a></nav>
  <a href="/about">About</a>
  <a href="https://example.com/pricing">Pricing</a>
  <a href="https://external.com/page">External</a>
  <a href="mailto:hi@example.com">Email</a>
  <a href="/logo.png">Logo</a>
  <a href="/about#team">About Team Anchor</a>
</body></html>
"""


def test_parse_sitemap_xml_filters_external_and_ignorable():
    urls = parse_sitemap_xml(SITEMAP_XML, base_domain="example.com")
    assert "https://example.com/about" in urls
    assert "https://example.com" in urls or "https://example.com/" in urls
    assert all("external.com" not in u for u in urls)
    assert all(not u.endswith(".pdf") for u in urls)


def test_parse_sitemap_xml_detects_sitemap_index():
    from app.crawler.sitemap import is_sitemap_index

    assert is_sitemap_index(SITEMAP_INDEX_XML) is True
    assert is_sitemap_index(SITEMAP_XML) is False


def test_extract_links_filters_and_dedupes():
    links = extract_links(HTML_WITH_LINKS, base_url="https://example.com/home", base_domain="example.com")
    assert "https://example.com/about" in links
    assert "https://example.com/pricing" in links
    assert not any("external.com" in link for link in links)
    assert not any(link.startswith("mailto:") for link in links)
    assert not any(link.endswith(".png") for link in links)
    # the fragment variant normalizes to the same URL already present, so no duplicate
    assert links.count("https://example.com/about") == 1
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_crawler.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'app.crawler.sitemap'`.

- [ ] **Step 3: Write `app/crawler/sitemap.py`**

```python
from defusedxml import ElementTree
from defusedxml.common import DefusedXmlException

import httpx

from app.utils.logging import get_logger
from app.utils.retry import with_retry
from app.utils.url_utils import is_ignorable_url, is_same_domain, normalize_url

logger = get_logger(__name__)
_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
# NOTE: sitemap XML comes from arbitrary admin-submitted third-party sites, so it is parsed
# with defusedxml (not the stdlib xml.etree) to prevent XXE / billion-laughs attacks.


def is_sitemap_index(xml_text: str) -> bool:
    try:
        root = ElementTree.fromstring(xml_text)
    except (ElementTree.ParseError, DefusedXmlException):
        return False
    tag = root.tag.rsplit("}", 1)[-1]
    return tag == "sitemapindex"


def parse_sitemap_xml(xml_text: str, base_domain: str) -> list[str]:
    try:
        root = ElementTree.fromstring(xml_text)
    except (ElementTree.ParseError, DefusedXmlException):
        logger.warning("Failed to parse sitemap XML")
        return []

    locs = [el.text.strip() for el in root.findall(".//sm:url/sm:loc", _NS) if el.text]
    urls = []
    for loc in locs:
        if is_ignorable_url(loc):
            continue
        if not is_same_domain(loc, base_domain):
            continue
        urls.append(normalize_url(loc))
    return list(dict.fromkeys(urls))


def parse_sitemap_index_locs(xml_text: str) -> list[str]:
    try:
        root = ElementTree.fromstring(xml_text)
    except (ElementTree.ParseError, DefusedXmlException):
        return []
    return [el.text.strip() for el in root.findall(".//sm:sitemap/sm:loc", _NS) if el.text]


@with_retry(max_attempts=3)
async def _fetch_text(client: httpx.AsyncClient, url: str) -> str | None:
    resp = await client.get(url, timeout=15.0, follow_redirects=True)
    if resp.status_code != 200:
        return None
    return resp.text


async def fetch_sitemap_urls(client: httpx.AsyncClient, base_url: str) -> list[str] | None:
    from app.utils.url_utils import get_domain

    sitemap_url = base_url.rstrip("/") + "/sitemap.xml"
    text = await _fetch_text(client, sitemap_url)
    if text is None:
        return None

    base_domain = get_domain(base_url)
    if is_sitemap_index(text):
        all_urls: list[str] = []
        for sub_sitemap in parse_sitemap_index_locs(text):
            sub_text = await _fetch_text(client, sub_sitemap)
            if sub_text:
                all_urls.extend(parse_sitemap_xml(sub_text, base_domain))
        return list(dict.fromkeys(all_urls)) or None

    urls = parse_sitemap_xml(text, base_domain)
    return urls or None
```

- [ ] **Step 4: Write `app/crawler/recursive.py`**

```python
import asyncio
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

from app.utils.logging import get_logger
from app.utils.retry import with_retry
from app.utils.url_utils import is_ignorable_url, is_same_domain, normalize_url

logger = get_logger(__name__)


def extract_links(html: str, base_url: str, base_domain: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    found = []
    for tag in soup.find_all("a", href=True):
        absolute = urljoin(base_url, tag["href"])
        if is_ignorable_url(absolute):
            continue
        if not is_same_domain(absolute, base_domain):
            continue
        found.append(normalize_url(absolute))
    return list(dict.fromkeys(found))


@with_retry(max_attempts=3)
async def _fetch_html(client: httpx.AsyncClient, url: str) -> str | None:
    resp = await client.get(url, timeout=15.0, follow_redirects=True)
    if resp.status_code != 200 or "text/html" not in resp.headers.get("content-type", ""):
        return None
    return resp.text


async def recursive_crawl(
    client: httpx.AsyncClient,
    base_url: str,
    max_pages: int,
    max_depth: int,
    delay_seconds: float,
) -> list[str]:
    from app.utils.url_utils import get_domain

    base_domain = get_domain(base_url)
    start = normalize_url(base_url)
    visited: set[str] = set()
    queue: list[tuple[str, int]] = [(start, 0)]
    discovered: list[str] = []

    while queue and len(discovered) < max_pages:
        url, depth = queue.pop(0)
        if url in visited or depth > max_depth:
            continue
        visited.add(url)

        html = await _fetch_html(client, url)
        await asyncio.sleep(delay_seconds)
        if html is None:
            continue

        discovered.append(url)
        if depth < max_depth:
            for link in extract_links(html, url, base_domain):
                if link not in visited:
                    queue.append((link, depth + 1))

    return discovered[:max_pages]
```

- [ ] **Step 5: Write `app/crawler/discover.py`**

```python
import httpx

from app.crawler.recursive import recursive_crawl
from app.crawler.sitemap import fetch_sitemap_urls
from app.utils.logging import get_logger

logger = get_logger(__name__)


async def discover_urls(
    base_url: str,
    max_pages: int,
    max_depth: int,
    delay_seconds: float,
) -> tuple[list[str], str | None]:
    async with httpx.AsyncClient(headers={"User-Agent": "Mozilla/5.0 (compatible; ChatbotCrawler/1.0)"}) as client:
        sitemap_urls = await fetch_sitemap_urls(client, base_url)
        if sitemap_urls:
            logger.info("Using sitemap for %s: %d URLs", base_url, len(sitemap_urls))
            return sitemap_urls[:max_pages], base_url.rstrip("/") + "/sitemap.xml"

        logger.info("No sitemap for %s, falling back to recursive crawl", base_url)
        urls = await recursive_crawl(client, base_url, max_pages, max_depth, delay_seconds)
        return urls, None
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
pytest tests/test_crawler.py -v
```
Expected: 3 passed.

- [ ] **Step 7: Manual smoke test (real network, not automated)**

```bash
cd Ai/backend && source venv/bin/activate
python -c "
import asyncio
from app.crawler.discover import discover_urls
urls, sitemap = asyncio.run(discover_urls('https://example.com', max_pages=20, max_depth=2, delay_seconds=0.5))
print('sitemap used:', sitemap)
print('found', len(urls), 'urls')
print(urls[:5])
"
```
Expected: prints a non-empty list of URLs (or an empty list gracefully, no exception) for a real reachable domain.

- [ ] **Step 8: Commit**

```bash
git add Ai/backend/app/crawler Ai/backend/tests/test_crawler.py
git commit -m "feat: sitemap discovery and recursive crawl fallback"
```

---

### Task 6: Content extractor — Playwright render + Trafilatura + BeautifulSoup fallback

**Files:**
- Create: `Ai/backend/app/extractor/cleaner.py`
- Create: `Ai/backend/app/extractor/content_extractor.py`
- Create: `Ai/backend/app/extractor/renderer.py`
- Test: `Ai/backend/tests/test_extractor.py`

**Interfaces:**
- Produces: `clean_html(html: str) -> BeautifulSoup` (strips nav/footer/script/style/cookie-banner/ad/social-icon elements, returns the cleaned soup), `extract_with_bs4(html: str) -> str` (cleaned visible text), `extract_content_from_html(html: str, url: str, min_length: int = 200) -> tuple[str, str]` (returns `(title, text)`; tries Trafilatura first, falls back to BeautifulSoup if the result is under `min_length` chars — this is the pure, fully-testable decision logic), `async render_page(url: str) -> str | None` (Playwright, network-idle wait, returns rendered HTML), `async extract_page_content(url: str, min_length: int = 200) -> tuple[str, str] | None` (renders then delegates to `extract_content_from_html`; returns `None` if rendering fails). Consumed by the crawl service (Task 10).

- [ ] **Step 1: Write the failing tests (pure functions only — no Playwright)**

```python
# Ai/backend/tests/test_extractor.py
from app.extractor.cleaner import clean_html, extract_with_bs4
from app.extractor.content_extractor import extract_content_from_html

NOISY_HTML = """
<html><head><title>About Us</title></head>
<body>
  <nav><a href="/">Home</a><a href="/about">About</a></nav>
  <div id="cookie-banner">We use cookies. <button>Accept</button></div>
  <header class="site-header">Acme Inc</header>
  <main>
    <h1>About Acme</h1>
    <p>Acme builds widgets for the modern era. Founded in 2010, Acme has shipped
    thousands of widgets to customers worldwide and continues to innovate on widget
    design every single year with a dedicated team of engineers and designers.</p>
  </main>
  <div class="social-icons"><a href="https://twitter.com/acme">Twitter</a></div>
  <footer>&copy; 2026 Acme Inc. All rights reserved.</footer>
  <script>console.log("tracking");</script>
  <style>.hidden { display: none; }</style>
</body></html>
"""

THIN_HTML = "<html><body><div>Hi</div></body></html>"


def test_clean_html_removes_boilerplate():
    soup = clean_html(NOISY_HTML)
    text = soup.get_text(" ", strip=True)
    assert "cookies" not in text.lower()
    assert "twitter" not in text.lower()
    assert "console.log" not in text
    assert "Acme builds widgets" in text


def test_extract_with_bs4_keeps_main_content():
    text = extract_with_bs4(NOISY_HTML)
    assert "Acme builds widgets" in text
    assert "Home" not in text
    assert "All rights reserved" not in text


def test_extract_content_from_html_uses_trafilatura_when_sufficient():
    title, text = extract_content_from_html(NOISY_HTML, url="https://acme.test/about", min_length=50)
    assert "Acme builds widgets" in text
    assert len(text) >= 50


def test_extract_content_from_html_falls_back_when_too_short():
    title, text = extract_content_from_html(THIN_HTML, url="https://acme.test/thin", min_length=200)
    assert text == "Hi" or "Hi" in text
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd Ai/backend && source venv/bin/activate
pytest tests/test_extractor.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'app.extractor.cleaner'`.

- [ ] **Step 3: Write `app/extractor/cleaner.py`**

```python
from bs4 import BeautifulSoup

REMOVE_TAGS = ["nav", "footer", "header", "script", "style", "noscript", "iframe", "form", "aside"]
REMOVE_SELECTOR_KEYWORDS = [
    "cookie", "consent", "banner", "advert", "ads-", "ad-slot", "social", "share-buttons",
    "sidebar", "popup", "modal", "newsletter", "breadcrumb",
]


def _matches_noise_keyword(tag) -> bool:
    haystack = " ".join([
        " ".join(tag.get("class", [])) if tag.get("class") else "",
        tag.get("id", "") or "",
    ]).lower()
    return any(keyword in haystack for keyword in REMOVE_SELECTOR_KEYWORDS)


def clean_html(html: str) -> BeautifulSoup:
    soup = BeautifulSoup(html, "html.parser")

    for tag_name in REMOVE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    for tag in soup.find_all(True):
        if _matches_noise_keyword(tag):
            tag.decompose()

    return soup


def extract_with_bs4(html: str) -> str:
    soup = clean_html(html)
    body = soup.find("body") or soup
    text = body.get_text("\n", strip=True)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)
```

- [ ] **Step 4: Write `app/extractor/content_extractor.py`**

```python
import trafilatura
from bs4 import BeautifulSoup

from app.extractor.cleaner import extract_with_bs4
from app.utils.logging import get_logger

logger = get_logger(__name__)


def _extract_title(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    h1 = soup.find("h1")
    return h1.get_text(strip=True) if h1 else ""


def extract_content_from_html(html: str, url: str, min_length: int = 200) -> tuple[str, str]:
    title = _extract_title(html)

    trafilatura_text = trafilatura.extract(html, url=url, favor_precision=True) or ""
    if len(trafilatura_text) >= min_length:
        return title, trafilatura_text

    logger.info("Trafilatura extraction too short (%d chars) for %s, falling back to BeautifulSoup", len(trafilatura_text), url)
    fallback_text = extract_with_bs4(html)
    return title, fallback_text or trafilatura_text
```

- [ ] **Step 5: Write `app/extractor/renderer.py`**

```python
from playwright.async_api import async_playwright

from app.utils.logging import get_logger
from app.utils.retry import with_retry

logger = get_logger(__name__)


@with_retry(max_attempts=2)
async def render_page(url: str, timeout_ms: int = 20000) -> str | None:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await page.goto(url, wait_until="networkidle", timeout=timeout_ms)
            html = await page.content()
            return html
        except Exception:
            logger.exception("Failed to render %s", url)
            return None
        finally:
            await browser.close()


async def extract_page_content(url: str, min_length: int = 200) -> tuple[str, str] | None:
    from app.extractor.content_extractor import extract_content_from_html

    html = await render_page(url)
    if html is None:
        return None
    return extract_content_from_html(html, url, min_length=min_length)
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
pytest tests/test_extractor.py -v
```
Expected: 4 passed.

- [ ] **Step 7: Manual smoke test for Playwright rendering (real browser, not automated)**

```bash
cd Ai/backend && source venv/bin/activate
python -c "
import asyncio
from app.extractor.renderer import extract_page_content
result = asyncio.run(extract_page_content('https://example.com'))
print(result[0] if result else None)
print((result[1][:300] if result else 'FAILED'))
"
```
Expected: prints a title and the first 300 chars of extracted body text, no exception.

- [ ] **Step 8: Commit**

```bash
git add Ai/backend/app/extractor Ai/backend/tests/test_extractor.py
git commit -m "feat: content extraction via Playwright, Trafilatura, and BeautifulSoup fallback"
```

---

### Task 7: Chunker

**Files:**
- Create: `Ai/backend/app/chunker/splitter.py`
- Test: `Ai/backend/tests/test_chunker.py`

**Interfaces:**
- Produces: `chunk_text(text: str) -> list[str]` (uses `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)` exactly per the Global Constraints). Consumed by the crawl service (Task 10).

- [ ] **Step 1: Write the failing test**

```python
# Ai/backend/tests/test_chunker.py
from app.chunker.splitter import chunk_text


def test_chunk_text_splits_long_text():
    text = "Sentence about widgets. " * 200  # well over 1000 chars
    chunks = chunk_text(text)
    assert len(chunks) > 1
    assert all(len(c) <= 1000 for c in chunks)


def test_chunk_text_short_text_single_chunk():
    text = "Short page content."
    chunks = chunk_text(text)
    assert chunks == ["Short page content."]


def test_chunk_text_has_overlap():
    text = "A" * 1500
    chunks = chunk_text(text)
    assert len(chunks) >= 2
    # consecutive chunks share overlapping characters
    assert chunks[0][-50:] in chunks[1] or chunks[1][:50] in chunks[0]


def test_chunk_text_empty_string_returns_empty_list():
    assert chunk_text("") == []
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd Ai/backend && source venv/bin/activate
pytest tests/test_chunker.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'app.chunker.splitter'`.

- [ ] **Step 3: Write `app/chunker/splitter.py`**

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


def chunk_text(text: str) -> list[str]:
    if not text.strip():
        return []
    return _splitter.split_text(text)
```

Add `langchain-text-splitters==0.3.4` to `requirements.txt` (it's a dependency of `langchain` but pin it explicitly since we import it directly) and `pip install -r requirements.txt` again.

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_chunker.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add Ai/backend/app/chunker Ai/backend/tests/test_chunker.py Ai/backend/requirements.txt
git commit -m "feat: text chunking with RecursiveCharacterTextSplitter"
```

---

### Task 8: Embeddings singleton

**Files:**
- Create: `Ai/backend/app/embeddings/embedder.py`
- Test: `Ai/backend/tests/test_embeddings.py`

**Interfaces:**
- Produces: `get_embedder() -> HuggingFaceEmbeddings` (process-wide singleton, model = `settings.embedding_model`), `embed_texts(texts: list[str]) -> list[list[float]]`, `embed_query(text: str) -> list[float]`. Consumed by the vector store (Task 9) and the chat/RAG module (Task 16).

- [ ] **Step 1: Write `app/embeddings/embedder.py`**

```python
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


@lru_cache
def get_embedder() -> HuggingFaceEmbeddings:
    logger.info("Loading embedding model %s", settings.embedding_model)
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    return get_embedder().embed_documents(texts)


def embed_query(text: str) -> list[float]:
    return get_embedder().embed_query(text)
```

- [ ] **Step 2: Write the test (real model load — first run downloads ~90MB, subsequent runs are cached)**

```python
# Ai/backend/tests/test_embeddings.py
from app.embeddings.embedder import embed_query, embed_texts, get_embedder


def test_embedder_is_singleton():
    assert get_embedder() is get_embedder()


def test_embed_texts_returns_correct_dimension_vectors():
    vectors = embed_texts(["hello world", "goodbye world"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 384  # all-MiniLM-L6-v2 output dimension
    assert len(vectors[1]) == 384


def test_embed_query_matches_dimension():
    vector = embed_query("what is this website about?")
    assert len(vector) == 384


def test_similar_texts_have_higher_cosine_similarity_than_dissimilar():
    import numpy as np

    v_dog = np.array(embed_query("dogs are loyal pets"))
    v_puppy = np.array(embed_query("puppies are great companions"))
    v_finance = np.array(embed_query("quarterly financial earnings report"))

    def cosine(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    assert cosine(v_dog, v_puppy) > cosine(v_dog, v_finance)
```

Add `numpy==2.2.1` to `requirements.txt` (test-only dependency here, also used by the vector store's similarity math in Task 9), `pip install -r requirements.txt`.

- [ ] **Step 3: Run the tests**

```bash
cd Ai/backend && source venv/bin/activate
pytest tests/test_embeddings.py -v
```
Expected: 4 passed (first run may take ~30-60s to download the model).

- [ ] **Step 4: Commit**

```bash
git add Ai/backend/app/embeddings Ai/backend/tests/test_embeddings.py Ai/backend/requirements.txt
git commit -m "feat: HuggingFace embedding singleton wrapper"
```

---

### Task 9: Vector store — ChromaDB per-website collections

**Files:**
- Create: `Ai/backend/app/vectordb/chroma_client.py`
- Create: `Ai/backend/app/vectordb/collection.py`
- Test: `Ai/backend/tests/test_vectordb.py`

**Interfaces:**
- Consumes: `app.config.settings.chroma_persist_dir`
- Produces: `get_chroma_client() -> chromadb.ClientAPI` (singleton), `collection_name(website_id: int) -> str` (returns `f"website_{website_id}"`), `get_or_create_collection(website_id: int)`, `upsert_page_chunks(website_id: int, page_id: int, url: str, title: str, content_hash: str, chunks: list[str], embeddings: list[list[float]]) -> None`, `delete_page_vectors(website_id: int, page_id: int) -> None`, `delete_website_collection(website_id: int) -> None`, `query_collection(website_id: int, query_embedding: list[float], top_k: int) -> list[dict]` (each dict: `{"text": str, "metadata": dict, "similarity": float}`, `similarity` = `1 - cosine_distance`, sorted descending). Consumed by the crawl service (Task 10), sync service (Task 14), and RAG module (Task 16).

- [ ] **Step 1: Write the failing tests (using precomputed fixture vectors, no real embedding model needed)**

```python
# Ai/backend/tests/test_vectordb.py
import shutil
import tempfile

import pytest


@pytest.fixture()
def temp_chroma(monkeypatch):
    tmp_dir = tempfile.mkdtemp()
    monkeypatch.setenv("CHROMA_PERSIST_DIR", tmp_dir)

    import app.vectordb.chroma_client as chroma_client_module

    chroma_client_module.get_chroma_client.cache_clear()
    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


def _vec(seed: float, dim: int = 8) -> list[float]:
    return [seed] * dim


def test_collection_name_format():
    from app.vectordb.collection import collection_name

    assert collection_name(1) == "website_1"
    assert collection_name(42) == "website_42"


def test_upsert_and_query_returns_similar_chunk(temp_chroma):
    from app.vectordb.collection import query_collection, upsert_page_chunks

    upsert_page_chunks(
        website_id=1,
        page_id=10,
        url="https://example.com/about",
        title="About",
        content_hash="hash1",
        chunks=["chunk about widgets", "chunk about pricing"],
        embeddings=[_vec(0.1), _vec(0.9)],
    )

    results = query_collection(website_id=1, query_embedding=_vec(0.1), top_k=1)
    assert len(results) == 1
    assert results[0]["text"] == "chunk about widgets"
    assert results[0]["metadata"]["page_id"] == 10
    assert results[0]["metadata"]["url"] == "https://example.com/about"
    assert results[0]["metadata"]["hash"] == "hash1"
    assert 0.0 <= results[0]["similarity"] <= 1.0


def test_upsert_replaces_on_same_ids(temp_chroma):
    from app.vectordb.collection import get_or_create_collection, upsert_page_chunks

    upsert_page_chunks(
        website_id=2, page_id=20, url="https://example.com/x", title="X",
        content_hash="h1", chunks=["old text"], embeddings=[_vec(0.2)],
    )
    upsert_page_chunks(
        website_id=2, page_id=20, url="https://example.com/x", title="X",
        content_hash="h2", chunks=["new text"], embeddings=[_vec(0.2)],
    )

    coll = get_or_create_collection(2)
    assert coll.count() == 1


def test_delete_page_vectors_removes_only_that_page(temp_chroma):
    from app.vectordb.collection import delete_page_vectors, get_or_create_collection, upsert_page_chunks

    upsert_page_chunks(website_id=3, page_id=30, url="u1", title="t1", content_hash="h", chunks=["a"], embeddings=[_vec(0.1)])
    upsert_page_chunks(website_id=3, page_id=31, url="u2", title="t2", content_hash="h", chunks=["b"], embeddings=[_vec(0.2)])

    delete_page_vectors(website_id=3, page_id=30)

    coll = get_or_create_collection(3)
    assert coll.count() == 1


def test_delete_website_collection_removes_collection(temp_chroma):
    from app.vectordb.chroma_client import get_chroma_client
    from app.vectordb.collection import delete_website_collection, upsert_page_chunks

    upsert_page_chunks(website_id=4, page_id=40, url="u", title="t", content_hash="h", chunks=["a"], embeddings=[_vec(0.1)])
    delete_website_collection(4)

    client = get_chroma_client()
    assert "website_4" not in [c.name for c in client.list_collections()]
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd Ai/backend && source venv/bin/activate
pytest tests/test_vectordb.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'app.vectordb.chroma_client'`.

- [ ] **Step 3: Write `app/vectordb/chroma_client.py`**

```python
from functools import lru_cache

import chromadb

from app.config import settings


@lru_cache
def get_chroma_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=settings.chroma_persist_dir)
```

- [ ] **Step 4: Write `app/vectordb/collection.py`**

```python
from app.vectordb.chroma_client import get_chroma_client


def collection_name(website_id: int) -> str:
    return f"website_{website_id}"


def get_or_create_collection(website_id: int):
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=collection_name(website_id),
        metadata={"hnsw:space": "cosine"},
    )


def upsert_page_chunks(
    website_id: int,
    page_id: int,
    url: str,
    title: str,
    content_hash: str,
    chunks: list[str],
    embeddings: list[list[float]],
) -> None:
    if not chunks:
        return
    collection = get_or_create_collection(website_id)
    ids = [f"{page_id}_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "website_id": website_id,
            "page_id": page_id,
            "title": title or "",
            "url": url,
            "chunk_id": i,
            "hash": content_hash,
        }
        for i in range(len(chunks))
    ]
    collection.upsert(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)


def delete_page_vectors(website_id: int, page_id: int) -> None:
    collection = get_or_create_collection(website_id)
    collection.delete(where={"page_id": page_id})


def delete_website_collection(website_id: int) -> None:
    client = get_chroma_client()
    existing = [c.name for c in client.list_collections()]
    if collection_name(website_id) in existing:
        client.delete_collection(collection_name(website_id))


def query_collection(website_id: int, query_embedding: list[float], top_k: int) -> list[dict]:
    collection = get_or_create_collection(website_id)
    if collection.count() == 0:
        return []

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
    )

    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]

    return [
        {"text": doc, "metadata": meta, "similarity": 1.0 - dist}
        for doc, meta, dist in zip(documents, metadatas, distances)
    ]
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_vectordb.py -v
```
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add Ai/backend/app/vectordb Ai/backend/tests/test_vectordb.py
git commit -m "feat: ChromaDB per-website collection wrapper"
```

---

### Task 10: Crawl service — orchestrates crawl, sync, and reindex

This single service function backs all three admin actions: **Crawl** (`force=False` on a fresh site), **Sync** (`force=False` on an existing site — indexes new/changed pages, deletes stale ones, skips unchanged), and **Reindex** (`force=True` — re-embeds every currently-discovered page regardless of hash).

**Files:**
- Create: `Ai/backend/app/services/crawl_service.py`
- Test: `Ai/backend/tests/test_crawl_service.py`

**Interfaces:**
- Consumes: `app.crawler.discover.discover_urls`, `app.extractor.renderer.extract_page_content`, `app.chunker.splitter.chunk_text`, `app.embeddings.embedder.embed_texts`, `app.vectordb.collection.{upsert_page_chunks, delete_page_vectors, delete_website_collection}`, `app.database.models.*`
- Produces: `async run_crawl(db: Session, website: Website, force: bool = False) -> CrawlJob`, `delete_website_data(website_id: int) -> None`. Consumed by the crawl/sync/reindex API routes (Task 12) and the scheduler (Task 14).

- [ ] **Step 1: Write the failing tests**

```python
# Ai/backend/tests/test_crawl_service.py
import os
import shutil
import tempfile

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture()
def db_session():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    engine = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False})
    from app.database.session import Base
    from app.database import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    os.remove(path)


@pytest.fixture()
def temp_chroma(monkeypatch):
    tmp_dir = tempfile.mkdtemp()
    monkeypatch.setenv("CHROMA_PERSIST_DIR", tmp_dir)
    import app.vectordb.chroma_client as chroma_client_module

    chroma_client_module.get_chroma_client.cache_clear()
    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


def _fake_embed_texts(texts):
    return [[float(len(t))] * 8 for t in texts]


@pytest.mark.asyncio
async def test_run_crawl_indexes_new_pages(db_session, temp_chroma, monkeypatch):
    from app.database.models import CrawlJobStatus, Page, PageStatus, Website
    import app.services.crawl_service as crawl_service

    website = Website(url="https://acme.test", name="Acme")
    db_session.add(website)
    db_session.commit()

    async def fake_discover_urls(*args, **kwargs):
        return (["https://acme.test/about", "https://acme.test/pricing"], None)

    async def fake_extract(url, min_length=200):
        return ("Title", f"Long enough content about {url} " * 20)

    monkeypatch.setattr(crawl_service, "discover_urls", fake_discover_urls)
    monkeypatch.setattr(crawl_service, "extract_page_content", fake_extract)
    monkeypatch.setattr(crawl_service, "embed_texts", _fake_embed_texts)

    job = await crawl_service.run_crawl(db_session, website)

    assert job.status == CrawlJobStatus.completed
    assert job.pages_found == 2
    assert job.pages_indexed == 2
    pages = db_session.query(Page).filter(Page.website_id == website.id).all()
    assert len(pages) == 2
    assert all(p.status == PageStatus.indexed for p in pages)


@pytest.mark.asyncio
async def test_run_crawl_skips_unchanged_and_deletes_stale(db_session, temp_chroma, monkeypatch):
    from app.database.models import Page, PageStatus, Website
    import app.services.crawl_service as crawl_service

    website = Website(url="https://acme.test", name="Acme")
    db_session.add(website)
    db_session.commit()

    async def fake_discover_urls_v1(*args, **kwargs):
        return (["https://acme.test/about", "https://acme.test/pricing"], None)

    async def fake_extract(url, min_length=200):
        return ("Title", f"Stable content for {url} " * 20)

    monkeypatch.setattr(crawl_service, "discover_urls", fake_discover_urls_v1)
    monkeypatch.setattr(crawl_service, "extract_page_content", fake_extract)
    monkeypatch.setattr(crawl_service, "embed_texts", _fake_embed_texts)
    await crawl_service.run_crawl(db_session, website)

    async def fake_discover_urls_v2(*args, **kwargs):
        return (["https://acme.test/about"], None)  # pricing page is now gone

    monkeypatch.setattr(crawl_service, "discover_urls", fake_discover_urls_v2)
    job2 = await crawl_service.run_crawl(db_session, website)

    assert job2.pages_indexed == 0  # about page unchanged -> skipped
    pricing = db_session.query(Page).filter(Page.url == "https://acme.test/pricing").first()
    assert pricing.status == PageStatus.deleted


@pytest.mark.asyncio
async def test_run_crawl_force_reindexes_unchanged_pages(db_session, temp_chroma, monkeypatch):
    from app.database.models import Website
    import app.services.crawl_service as crawl_service

    website = Website(url="https://acme.test", name="Acme")
    db_session.add(website)
    db_session.commit()

    async def fake_discover_urls(*args, **kwargs):
        return (["https://acme.test/about"], None)

    async def fake_extract(url, min_length=200):
        return ("Title", f"Stable content for {url} " * 20)

    monkeypatch.setattr(crawl_service, "discover_urls", fake_discover_urls)
    monkeypatch.setattr(crawl_service, "extract_page_content", fake_extract)
    monkeypatch.setattr(crawl_service, "embed_texts", _fake_embed_texts)
    await crawl_service.run_crawl(db_session, website)

    job2 = await crawl_service.run_crawl(db_session, website, force=True)
    assert job2.pages_indexed == 1  # forced re-index despite unchanged hash


@pytest.mark.asyncio
async def test_run_crawl_marks_failed_pages(db_session, temp_chroma, monkeypatch):
    from app.database.models import Page, PageStatus, Website
    import app.services.crawl_service as crawl_service

    website = Website(url="https://acme.test", name="Acme")
    db_session.add(website)
    db_session.commit()

    async def fake_discover_urls(*args, **kwargs):
        return (["https://acme.test/broken"], None)

    async def fake_extract(url, min_length=200):
        return None

    monkeypatch.setattr(crawl_service, "discover_urls", fake_discover_urls)
    monkeypatch.setattr(crawl_service, "extract_page_content", fake_extract)
    monkeypatch.setattr(crawl_service, "embed_texts", _fake_embed_texts)

    job = await crawl_service.run_crawl(db_session, website)

    assert job.pages_failed == 1
    page = db_session.query(Page).filter(Page.url == "https://acme.test/broken").first()
    assert page.status == PageStatus.failed
```

Add `pytest-asyncio==0.25.0` is already in `requirements.txt` (Task 1); create `Ai/backend/pytest.ini`:

```ini
[pytest]
asyncio_mode = auto
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd Ai/backend && source venv/bin/activate
pytest tests/test_crawl_service.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.crawl_service'`.

- [ ] **Step 3: Write `app/services/crawl_service.py`**

```python
import asyncio
from datetime import datetime

from sqlalchemy.orm import Session

from app.chunker.splitter import chunk_text
from app.config import settings
from app.crawler.discover import discover_urls
from app.database.models import CrawlJob, CrawlJobStatus, Page, PageStatus, Website, WebsiteStatus
from app.embeddings.embedder import embed_texts
from app.extractor.renderer import extract_page_content
from app.utils.hashing import hash_content
from app.utils.logging import get_logger
from app.utils.url_utils import normalize_url
from app.vectordb.collection import delete_page_vectors, delete_website_collection, upsert_page_chunks

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

    chunks = chunk_text(text)
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
        upsert_page_chunks(website.id, page.id, url, title, new_hash, chunks, embeddings)

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
    return job


def delete_website_data(website_id: int) -> None:
    delete_website_collection(website_id)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_crawl_service.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add Ai/backend/app/services/crawl_service.py Ai/backend/tests/test_crawl_service.py Ai/backend/pytest.ini
git commit -m "feat: crawl/sync/reindex orchestration service"
```

---

### Task 11: Website CRUD API

**Files:**
- Create: `Ai/backend/app/api/schemas.py`
- Create: `Ai/backend/app/api/routes/websites.py`
- Modify: `Ai/backend/app/api/deps.py` (no change needed — already exports `get_db`, `get_current_admin`)
- Test: `Ai/backend/tests/test_websites_api.py`

**Interfaces:**
- Consumes: `app.database.models.Website`, `app.services.crawl_service.delete_website_data`, `app.api.deps.{get_db, get_current_admin}`
- Produces: Pydantic schemas `WebsiteCreate`, `WebsiteUpdate`, `WebsiteRead` in `app/api/schemas.py` (grows with later tasks — Task 12/13/17/18/19 add more schemas to this same file). `router` (APIRouter, mounted with no prefix) exposing `POST /websites`, `GET /websites`, `PATCH /websites/{id}`, `DELETE /website/{id}`.

- [ ] **Step 1: Write the failing tests**

```python
# Ai/backend/tests/test_websites_api.py
import os
import tempfile

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture()
def client(monkeypatch):
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    engine = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False})

    from app.database.session import Base
    from app.database import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    TestSession = sessionmaker(bind=engine)

    from app.services.auth_service import seed_admin_user

    seed_db = TestSession()
    seed_admin_user(seed_db)  # lifespan seeds the *production* DB, not this temp one — seed it explicitly here
    seed_db.close()

    from app.main import app
    from app.api.deps import get_db

    def override_get_db():
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    os.remove(path)


def _admin_token(client) -> str:
    from app.config import settings

    resp = client.post("/auth/login", json={"email": settings.admin_email, "password": settings.admin_password})
    return resp.json()["access_token"]


def _auth_headers(client) -> dict:
    return {"Authorization": f"Bearer {_admin_token(client)}"}


def test_create_website_requires_auth(client):
    resp = client.post("/websites", json={"url": "https://acme.test", "name": "Acme"})
    assert resp.status_code == 403 or resp.status_code == 401


def test_create_and_list_website(client):
    resp = client.post("/websites", json={"url": "https://acme.test", "name": "Acme"}, headers=_auth_headers(client))
    assert resp.status_code == 201
    body = resp.json()
    assert body["url"] == "https://acme.test"
    assert body["status"] == "active"

    resp = client.get("/websites")
    assert resp.status_code == 200
    assert len(resp.json()) == 1


def test_update_website(client):
    headers = _auth_headers(client)
    created = client.post("/websites", json={"url": "https://acme.test", "name": "Acme"}, headers=headers).json()

    resp = client.patch(f"/websites/{created['id']}", json={"name": "Acme Corp"}, headers=headers)
    assert resp.status_code == 200
    assert resp.json()["name"] == "Acme Corp"


def test_delete_website(client):
    headers = _auth_headers(client)
    created = client.post("/websites", json={"url": "https://acme.test", "name": "Acme"}, headers=headers).json()

    resp = client.delete(f"/website/{created['id']}", headers=headers)
    assert resp.status_code == 204

    resp = client.get("/websites")
    assert resp.json() == []


def test_create_duplicate_url_rejected(client):
    headers = _auth_headers(client)
    client.post("/websites", json={"url": "https://acme.test", "name": "Acme"}, headers=headers)
    resp = client.post("/websites", json={"url": "https://acme.test", "name": "Acme 2"}, headers=headers)
    assert resp.status_code == 409
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd Ai/backend && source venv/bin/activate
pytest tests/test_websites_api.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'app.main'` (main.py doesn't exist until Task 19 — see note below).

> **Note:** `app.main` is created in Task 19. To run this task's tests standalone before Task 19 exists, temporarily create a minimal `app/main.py` with `app = FastAPI()` plus `app.include_router(auth.router)` and `app.include_router(websites.router)`; Task 19 will replace it with the full wiring. If executing tasks in order via subagent-driven-development, skip the temporary file and just proceed to Task 19 immediately after this task, then return to run this test.

- [ ] **Step 3: Write `app/api/schemas.py`**

```python
from datetime import datetime

from pydantic import BaseModel, ConfigDict, HttpUrl


class WebsiteCreate(BaseModel):
    url: HttpUrl
    name: str
    logo_url: HttpUrl | None = None
    crawl_depth_limit: int = 5
    max_pages: int = 500


class WebsiteUpdate(BaseModel):
    name: str | None = None
    logo_url: HttpUrl | None = None
    crawl_depth_limit: int | None = None
    max_pages: int | None = None


class WebsiteRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    url: str
    name: str
    logo_url: str | None
    status: str
    sitemap_url: str | None
    crawl_depth_limit: int
    max_pages: int
    created_at: datetime
    updated_at: datetime
    last_synced_at: datetime | None
```

- [ ] **Step 4: Write `app/api/routes/websites.py`**

```python
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
```

- [ ] **Step 5: Run tests to verify they pass (after Task 19's `app/main.py` exists, or the temporary stub)**

```bash
pytest tests/test_websites_api.py -v
```
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add Ai/backend/app/api/schemas.py Ai/backend/app/api/routes/websites.py Ai/backend/tests/test_websites_api.py
git commit -m "feat: website CRUD API"
```

---

### Task 12: Crawl / Sync / Reindex API + crawl status

**Files:**
- Create: `Ai/backend/app/api/routes/crawl.py`
- Modify: `Ai/backend/app/api/schemas.py` (append `CrawlJobRead`)
- Test: `Ai/backend/tests/test_crawl_api.py`

**Interfaces:**
- Consumes: `app.services.crawl_service.run_crawl`, `app.database.session.SessionLocal` (module-level import, so tests can monkeypatch it), `app.database.models.{Website, CrawlJob}`
- Produces: `router` (APIRouter, no prefix) exposing `POST /crawl/{website_id}`, `POST /sync/{website_id}`, `POST /reindex/{website_id}` (all admin, 202, kick off `run_crawl` as a background task with a fresh DB session), `GET /websites/{website_id}/crawl-jobs` (admin, list of `CrawlJobRead` ordered newest-first — backs "View Crawl Status").

- [ ] **Step 1: Write the failing tests**

```python
# Ai/backend/tests/test_crawl_api.py
import os
import tempfile

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture()
def client(monkeypatch):
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    engine = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False})

    from app.database.session import Base
    from app.database import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    TestSession = sessionmaker(bind=engine)

    from app.services.auth_service import seed_admin_user

    seed_db = TestSession()
    seed_admin_user(seed_db)  # lifespan seeds the *production* DB, not this temp one — seed it explicitly here
    seed_db.close()

    from app.main import app
    from app.api.deps import get_db
    import app.api.routes.crawl as crawl_module

    def override_get_db():
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    monkeypatch.setattr(crawl_module, "SessionLocal", TestSession)

    async def fake_run_crawl(db, website, force=False):
        from app.database.models import CrawlJob, CrawlJobStatus
        from datetime import datetime

        job = CrawlJob(
            website_id=website.id, status=CrawlJobStatus.completed,
            pages_found=3, pages_indexed=3 if not force else 3, pages_failed=0,
            finished_at=datetime.utcnow(),
        )
        db.add(job)
        db.commit()
        return job

    monkeypatch.setattr(crawl_module, "run_crawl", fake_run_crawl)

    with TestClient(app) as test_client:
        yield test_client

    os.remove(path)


def _auth_headers(client) -> dict:
    from app.config import settings

    token = client.post("/auth/login", json={"email": settings.admin_email, "password": settings.admin_password}).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _create_website(client, headers) -> int:
    return client.post("/websites", json={"url": "https://acme.test", "name": "Acme"}, headers=headers).json()["id"]


def test_crawl_endpoint_starts_job_and_returns_202(client):
    headers = _auth_headers(client)
    website_id = _create_website(client, headers)

    resp = client.post(f"/crawl/{website_id}", headers=headers)
    assert resp.status_code == 202

    jobs = client.get(f"/websites/{website_id}/crawl-jobs", headers=headers).json()
    assert len(jobs) == 1
    assert jobs[0]["pages_found"] == 3


def test_sync_endpoint(client):
    headers = _auth_headers(client)
    website_id = _create_website(client, headers)

    resp = client.post(f"/sync/{website_id}", headers=headers)
    assert resp.status_code == 202


def test_reindex_endpoint(client):
    headers = _auth_headers(client)
    website_id = _create_website(client, headers)

    resp = client.post(f"/reindex/{website_id}", headers=headers)
    assert resp.status_code == 202


def test_crawl_endpoint_404_for_unknown_website(client):
    headers = _auth_headers(client)
    resp = client.post("/crawl/999", headers=headers)
    assert resp.status_code == 404


def test_crawl_endpoint_requires_auth(client):
    website_id_resp = client.post("/websites", json={"url": "https://acme.test", "name": "Acme"}, headers=_auth_headers(client))
    resp = client.post(f"/crawl/{website_id_resp.json()['id']}")
    assert resp.status_code in (401, 403)
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd Ai/backend && source venv/bin/activate
pytest tests/test_crawl_api.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'app.api.routes.crawl'`.

- [ ] **Step 3: Append `CrawlJobRead` to `app/api/schemas.py`**

```python
class CrawlJobRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    website_id: int
    status: str
    pages_found: int
    pages_indexed: int
    pages_failed: int
    started_at: datetime
    finished_at: datetime | None
    error_log: str | None
```

- [ ] **Step 4: Write `app/api/routes/crawl.py`**

```python
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_admin, get_db
from app.api.schemas import CrawlJobRead
from app.database.models import AdminUser, CrawlJob, Website
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
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_crawl_api.py -v
```
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add Ai/backend/app/api/routes/crawl.py Ai/backend/app/api/schemas.py Ai/backend/tests/test_crawl_api.py
git commit -m "feat: crawl/sync/reindex endpoints and crawl status API"
```

---

### Task 13: Pages API — view indexed pages for a website

**Files:**
- Create: `Ai/backend/app/api/routes/pages.py`
- Modify: `Ai/backend/app/api/schemas.py` (append `PageRead`)
- Test: `Ai/backend/tests/test_pages_api.py`

**Interfaces:**
- Produces: `PageRead` schema, `router` exposing `GET /pages/{website_id}` (public — returns all `Page` rows for that website, newest-indexed first), with an optional `status` query filter (`indexed` / `failed` / `skipped` / `deleted`).

- [ ] **Step 1: Write the failing tests**

```python
# Ai/backend/tests/test_pages_api.py
import os
import tempfile

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture()
def client():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    engine = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False})

    from app.database.session import Base
    from app.database import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    TestSession = sessionmaker(bind=engine)

    from app.services.auth_service import seed_admin_user

    seed_db = TestSession()
    seed_admin_user(seed_db)  # lifespan seeds the *production* DB, not this temp one — seed it explicitly here
    seed_db.close()

    from app.main import app
    from app.api.deps import get_db

    def override_get_db():
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client, TestSession

    os.remove(path)


def _auth_headers(client) -> dict:
    from app.config import settings

    token = client.post("/auth/login", json={"email": settings.admin_email, "password": settings.admin_password}).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_list_pages_for_website(client):
    test_client, TestSession = client
    headers = _auth_headers(test_client)
    website_id = test_client.post("/websites", json={"url": "https://acme.test", "name": "Acme"}, headers=headers).json()["id"]

    from app.database.models import Page, PageStatus

    db = TestSession()
    db.add(Page(website_id=website_id, url="https://acme.test/about", title="About", status=PageStatus.indexed))
    db.add(Page(website_id=website_id, url="https://acme.test/broken", status=PageStatus.failed))
    db.commit()
    db.close()

    resp = test_client.get(f"/pages/{website_id}")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


def test_list_pages_filters_by_status(client):
    test_client, TestSession = client
    headers = _auth_headers(test_client)
    website_id = test_client.post("/websites", json={"url": "https://acme.test", "name": "Acme"}, headers=headers).json()["id"]

    from app.database.models import Page, PageStatus

    db = TestSession()
    db.add(Page(website_id=website_id, url="https://acme.test/about", status=PageStatus.indexed))
    db.add(Page(website_id=website_id, url="https://acme.test/broken", status=PageStatus.failed))
    db.commit()
    db.close()

    resp = test_client.get(f"/pages/{website_id}", params={"status": "failed"})
    assert resp.status_code == 200
    assert len(resp.json()) == 1
    assert resp.json()[0]["url"] == "https://acme.test/broken"


def test_list_pages_empty_for_unknown_website(client):
    test_client, _TestSession = client
    resp = test_client.get("/pages/999")
    assert resp.status_code == 200
    assert resp.json() == []
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd Ai/backend && source venv/bin/activate
pytest tests/test_pages_api.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'app.api.routes.pages'`.

- [ ] **Step 4: Append `PageRead` to `app/api/schemas.py`**

```python
class PageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    website_id: int
    url: str
    title: str | None
    status: str
    content_hash: str | None
    last_crawled_at: datetime | None
    indexed_at: datetime | None
```

- [ ] **Step 5: Write `app/api/routes/pages.py`**

```python
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
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
pytest tests/test_pages_api.py -v
```
Expected: 3 passed.

- [ ] **Step 7: Commit**

```bash
git add Ai/backend/app/api/routes/pages.py Ai/backend/app/api/schemas.py Ai/backend/tests/test_pages_api.py
git commit -m "feat: pages API for viewing indexed pages"
```

---

### Task 14: Scheduler — daily sync

**Files:**
- Create: `Ai/backend/app/scheduler/jobs.py`
- Create: `Ai/backend/app/scheduler/setup.py`
- Test: `Ai/backend/tests/test_scheduler.py`

**Interfaces:**
- Consumes: `app.services.crawl_service.run_crawl`, `app.database.session.SessionLocal`
- Produces: `run_daily_sync() -> dict` (`{"total": int, "succeeded": int, "failed": int}`), `start_scheduler() -> BackgroundScheduler`, `shutdown_scheduler() -> None`. Consumed by `app/main.py` startup/shutdown events (Task 19).

- [ ] **Step 1: Write the failing tests**

```python
# Ai/backend/tests/test_scheduler.py
import os
import tempfile

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture()
def db_setup(monkeypatch):
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    engine = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False})

    from app.database.session import Base
    from app.database import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    TestSession = sessionmaker(bind=engine)

    import app.scheduler.jobs as jobs_module

    monkeypatch.setattr(jobs_module, "SessionLocal", TestSession)

    yield TestSession
    os.remove(path)


def test_run_daily_sync_processes_all_non_crawling_websites(db_setup, monkeypatch):
    from app.database.models import Website, WebsiteStatus
    import app.scheduler.jobs as jobs_module

    db = db_setup()
    db.add(Website(url="https://a.test", name="A", status=WebsiteStatus.active))
    db.add(Website(url="https://b.test", name="B", status=WebsiteStatus.crawling))
    db.commit()
    db.close()

    calls = []

    async def fake_run_crawl(db, website, force=False):
        calls.append(website.url)

    monkeypatch.setattr(jobs_module, "run_crawl", fake_run_crawl)

    summary = jobs_module.run_daily_sync()

    assert summary["total"] == 1  # the "crawling" site is skipped
    assert summary["succeeded"] == 1
    assert calls == ["https://a.test"]


def test_run_daily_sync_counts_failures(db_setup, monkeypatch):
    from app.database.models import Website
    import app.scheduler.jobs as jobs_module

    db = db_setup()
    db.add(Website(url="https://a.test", name="A"))
    db.commit()
    db.close()

    async def failing_run_crawl(db, website, force=False):
        raise RuntimeError("boom")

    monkeypatch.setattr(jobs_module, "run_crawl", failing_run_crawl)

    summary = jobs_module.run_daily_sync()
    assert summary["failed"] == 1
    assert summary["succeeded"] == 0


def test_start_scheduler_registers_daily_job():
    from app.scheduler.setup import shutdown_scheduler, start_scheduler

    scheduler = start_scheduler()
    job = scheduler.get_job("daily_sync")
    assert job is not None
    shutdown_scheduler()
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd Ai/backend && source venv/bin/activate
pytest tests/test_scheduler.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'app.scheduler.jobs'`.

- [ ] **Step 3: Write `app/scheduler/jobs.py`**

```python
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
```

- [ ] **Step 4: Write `app/scheduler/setup.py`**

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_scheduler.py -v
```
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add Ai/backend/app/scheduler Ai/backend/tests/test_scheduler.py
git commit -m "feat: APScheduler daily sync job"
```

---

### Task 15: LLM / RAG — retrieval, prompt, confidence, Groq streaming

**Files:**
- Create: `Ai/backend/app/llm/prompts.py`
- Create: `Ai/backend/app/llm/retrieval.py`
- Create: `Ai/backend/app/llm/groq_client.py`
- Test: `Ai/backend/tests/test_llm.py`

**Interfaces:**
- Consumes: `app.embeddings.embedder.embed_query`, `app.vectordb.collection.query_collection`, `app.config.settings`
- Produces: `NO_ANSWER_MESSAGE = "I couldn't find that information on this website."` (exact string, reused by the chat API), `retrieve_relevant_chunks(website_id: int, question: str, top_k: int | None = None, threshold: float | None = None) -> list[dict]`, `compute_confidence(chunks: list[dict]) -> float` (0-100), `build_messages(question: str, context_chunks: list[dict]) -> list[dict]`, `async stream_completion(messages: list[dict]) -> AsyncIterator[str]` (yields text deltas from Groq). Consumed by the chat API (Task 16).

- [ ] **Step 1: Write the failing tests (retrieval + prompt + confidence — Groq streaming is manually smoke-tested in Step 5)**

```python
# Ai/backend/tests/test_llm.py
from app.llm.prompts import build_messages
from app.llm.retrieval import compute_confidence, retrieve_relevant_chunks


def test_build_messages_includes_context_and_question():
    chunks = [{"text": "Acme was founded in 2010.", "metadata": {"title": "About", "url": "https://acme.test/about"}}]
    messages = build_messages("When was Acme founded?", chunks)

    assert messages[0]["role"] == "system"
    assert "only" in messages[0]["content"].lower()
    assert messages[1]["role"] == "user"
    assert "Acme was founded in 2010." in messages[1]["content"]
    assert "When was Acme founded?" in messages[1]["content"]


def test_compute_confidence_empty_chunks_is_zero():
    assert compute_confidence([]) == 0.0


def test_compute_confidence_uses_best_similarity():
    chunks = [{"similarity": 0.42}, {"similarity": 0.81}, {"similarity": 0.5}]
    assert compute_confidence(chunks) == 81.0


def test_retrieve_relevant_chunks_filters_by_threshold(monkeypatch):
    import app.llm.retrieval as retrieval_module

    monkeypatch.setattr(retrieval_module, "embed_query", lambda text: [0.1] * 8)
    monkeypatch.setattr(
        retrieval_module,
        "query_collection",
        lambda website_id, query_embedding, top_k: [
            {"text": "a", "metadata": {}, "similarity": 0.9},
            {"text": "b", "metadata": {}, "similarity": 0.1},
        ],
    )

    results = retrieval_module.retrieve_relevant_chunks(website_id=1, question="q", threshold=0.3)
    assert len(results) == 1
    assert results[0]["text"] == "a"


def test_retrieve_relevant_chunks_returns_empty_when_all_below_threshold(monkeypatch):
    import app.llm.retrieval as retrieval_module

    monkeypatch.setattr(retrieval_module, "embed_query", lambda text: [0.1] * 8)
    monkeypatch.setattr(
        retrieval_module,
        "query_collection",
        lambda website_id, query_embedding, top_k: [{"text": "a", "metadata": {}, "similarity": 0.1}],
    )

    results = retrieval_module.retrieve_relevant_chunks(website_id=1, question="q", threshold=0.3)
    assert results == []
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd Ai/backend && source venv/bin/activate
pytest tests/test_llm.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'app.llm.prompts'`.

- [ ] **Step 3: Write `app/llm/prompts.py`**

```python
SYSTEM_PROMPT = (
    "You are an AI assistant that answers questions strictly using the provided website content. "
    "Only use information present in the CONTEXT section below — never use outside knowledge and never guess. "
    "If the context does not contain enough information to answer the question, respond with exactly: "
    "\"I couldn't find that information on this website.\" and nothing else. "
    "Keep answers concise and grounded only in the given context."
)


def build_messages(question: str, context_chunks: list[dict]) -> list[dict]:
    context_block = "\n\n".join(
        f"[Source: {chunk['metadata'].get('title') or chunk['metadata'].get('url')}]\n{chunk['text']}"
        for chunk in context_chunks
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"CONTEXT:\n{context_block}\n\nQUESTION: {question}"},
    ]
```

- [ ] **Step 4: Write `app/llm/retrieval.py`**

```python
from app.config import settings
from app.embeddings.embedder import embed_query
from app.vectordb.collection import query_collection

NO_ANSWER_MESSAGE = "I couldn't find that information on this website."


def retrieve_relevant_chunks(
    website_id: int,
    question: str,
    top_k: int | None = None,
    threshold: float | None = None,
) -> list[dict]:
    top_k = top_k if top_k is not None else settings.top_k_chunks
    threshold = threshold if threshold is not None else settings.similarity_threshold

    query_embedding = embed_query(question)
    results = query_collection(website_id, query_embedding, top_k)
    return [r for r in results if r["similarity"] >= threshold]


def compute_confidence(chunks: list[dict]) -> float:
    if not chunks:
        return 0.0
    best = max(chunk["similarity"] for chunk in chunks)
    return round(max(0.0, min(1.0, best)) * 100, 1)
```

- [ ] **Step 5: Write `app/llm/groq_client.py`**

```python
from collections.abc import AsyncIterator
from functools import lru_cache

from groq import AsyncGroq

from app.config import settings


@lru_cache
def get_groq_client() -> AsyncGroq:
    return AsyncGroq(api_key=settings.groq_api_key)


async def stream_completion(messages: list[dict]) -> AsyncIterator[str]:
    client = get_groq_client()
    stream = await client.chat.completions.create(
        model=settings.groq_model,
        messages=messages,
        temperature=0.2,
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
```

Manual smoke test (requires a real `GROQ_API_KEY` in `.env`, not automated):

```bash
cd Ai/backend && source venv/bin/activate
python -c "
import asyncio
from app.llm.groq_client import stream_completion

async def main():
    async for delta in stream_completion([{'role': 'user', 'content': 'Say hello in five words.'}]):
        print(delta, end='', flush=True)
    print()

asyncio.run(main())
"
```
Expected: prints a short streamed response, no exception.

- [ ] **Step 6: Run the automated tests to verify they pass**

```bash
pytest tests/test_llm.py -v
```
Expected: 5 passed.

- [ ] **Step 7: Commit**

```bash
git add Ai/backend/app/llm Ai/backend/tests/test_llm.py
git commit -m "feat: RAG retrieval, prompt construction, and Groq streaming client"
```

---

### Task 16: Chat API — SSE streaming endpoint

**Files:**
- Create: `Ai/backend/app/services/chat_service.py`
- Create: `Ai/backend/app/api/routes/chat.py`
- Modify: `Ai/backend/app/api/schemas.py` (append `ChatRequest`)
- Test: `Ai/backend/tests/test_chat_api.py`

**Interfaces:**
- Consumes: `app.llm.retrieval.{retrieve_relevant_chunks, compute_confidence, NO_ANSWER_MESSAGE}`, `app.llm.prompts.build_messages`, `app.llm.groq_client.stream_completion`, `app.database.models.{Conversation, Message, MessageRole}`
- Produces: `get_or_create_conversation(db, website_id, session_id, conversation_id) -> Conversation`, `async stream_chat_response(db, website_id, conversation, question) -> AsyncIterator[str]` (SSE-formatted `data: {...}\n\n` lines; event `type` is `"delta"` or `"done"`; the `done` event carries `sources`, `confidence`, `message_id`, `conversation_id`), `router` exposing `POST /chat` (public, `StreamingResponse`, media type `text/event-stream`).

- [ ] **Step 1: Write the failing tests**

```python
# Ai/backend/tests/test_chat_api.py
import json
import os
import tempfile

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture()
def client(monkeypatch):
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    engine = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False})

    from app.database.session import Base
    from app.database import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    TestSession = sessionmaker(bind=engine)

    from app.main import app
    from app.api.deps import get_db

    def override_get_db():
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client, TestSession

    os.remove(path)


def _parse_sse_events(text: str) -> list[dict]:
    events = []
    for block in text.strip().split("\n\n"):
        if block.startswith("data: "):
            events.append(json.loads(block[len("data: "):]))
    return events


def test_chat_streams_answer_with_sources_and_confidence(client, monkeypatch):
    test_client, TestSession = client
    from app.database.models import Website

    db = TestSession()
    website = Website(url="https://acme.test", name="Acme")
    db.add(website)
    db.commit()
    website_id = website.id
    db.close()

    import app.services.chat_service as chat_service_module

    fake_chunks = [
        {"text": "Acme was founded in 2010.", "metadata": {"page_id": 1, "url": "https://acme.test/about", "title": "About"}, "similarity": 0.8},
    ]
    monkeypatch.setattr(chat_service_module, "retrieve_relevant_chunks", lambda *a, **k: fake_chunks)

    async def fake_stream_completion(messages):
        for word in ["Acme ", "was ", "founded ", "in ", "2010."]:
            yield word

    monkeypatch.setattr(chat_service_module, "stream_completion", fake_stream_completion)

    resp = test_client.post("/chat", json={"website_id": website_id, "question": "When was Acme founded?", "session_id": "s1"})
    assert resp.status_code == 200

    events = _parse_sse_events(resp.text)
    deltas = [e for e in events if e["type"] == "delta"]
    done = [e for e in events if e["type"] == "done"][0]

    assert "".join(e["content"] for e in deltas) == "Acme was founded in 2010."
    assert done["confidence"] == 80.0
    assert done["sources"][0]["url"] == "https://acme.test/about"

    db = TestSession()
    from app.database.models import Message

    assert db.query(Message).count() == 2  # user + assistant
    db.close()


def test_chat_returns_fallback_when_no_relevant_chunks(client, monkeypatch):
    test_client, TestSession = client
    from app.database.models import Website

    db = TestSession()
    website = Website(url="https://acme.test", name="Acme")
    db.add(website)
    db.commit()
    website_id = website.id
    db.close()

    import app.services.chat_service as chat_service_module

    monkeypatch.setattr(chat_service_module, "retrieve_relevant_chunks", lambda *a, **k: [])

    async def fail_if_called(messages):
        raise AssertionError("Groq should not be called when there are no relevant chunks")
        yield  # pragma: no cover

    monkeypatch.setattr(chat_service_module, "stream_completion", fail_if_called)

    resp = test_client.post("/chat", json={"website_id": website_id, "question": "irrelevant?", "session_id": "s2"})
    events = _parse_sse_events(resp.text)
    done = [e for e in events if e["type"] == "done"][0]

    assert done["confidence"] == 0.0
    assert done["sources"] == []
    deltas = [e for e in events if e["type"] == "delta"]
    assert "".join(e["content"] for e in deltas) == "I couldn't find that information on this website."


def test_chat_reuses_existing_conversation(client, monkeypatch):
    test_client, TestSession = client
    from app.database.models import Website

    db = TestSession()
    website = Website(url="https://acme.test", name="Acme")
    db.add(website)
    db.commit()
    website_id = website.id
    db.close()

    import app.services.chat_service as chat_service_module

    monkeypatch.setattr(chat_service_module, "retrieve_relevant_chunks", lambda *a, **k: [])

    async def empty_stream(messages):
        return
        yield  # pragma: no cover

    monkeypatch.setattr(chat_service_module, "stream_completion", empty_stream)

    first = test_client.post("/chat", json={"website_id": website_id, "question": "q1", "session_id": "s3"})
    conversation_id = _parse_sse_events(first.text)[-1]["conversation_id"]

    test_client.post("/chat", json={"website_id": website_id, "question": "q2", "session_id": "s3", "conversation_id": conversation_id})

    db = TestSession()
    from app.database.models import Conversation, Message

    assert db.query(Conversation).count() == 1
    assert db.query(Message).filter(Message.conversation_id == conversation_id).count() == 4  # 2 user + 2 assistant
    db.close()
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd Ai/backend && source venv/bin/activate
pytest tests/test_chat_api.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.chat_service'`.

- [ ] **Step 3: Append `ChatRequest` to `app/api/schemas.py`**

```python
class ChatRequest(BaseModel):
    website_id: int
    question: str
    session_id: str
    conversation_id: int | None = None
```

- [ ] **Step 4: Write `app/services/chat_service.py`**

```python
import json

from sqlalchemy.orm import Session

from app.database.models import Conversation, Message, MessageRole
from app.llm.groq_client import stream_completion
from app.llm.prompts import build_messages
from app.llm.retrieval import NO_ANSWER_MESSAGE, compute_confidence, retrieve_relevant_chunks
from app.utils.logging import get_logger

logger = get_logger(__name__)


def get_or_create_conversation(
    db: Session, website_id: int, session_id: str, conversation_id: int | None
) -> Conversation:
    if conversation_id:
        conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
        if conversation:
            return conversation

    conversation = Conversation(website_id=website_id, session_id=session_id)
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return conversation


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _dedupe_sources(chunks: list[dict]) -> list[dict]:
    seen = set()
    deduped = []
    for chunk in chunks:
        meta = chunk["metadata"]
        if meta["url"] in seen:
            continue
        seen.add(meta["url"])
        deduped.append({"page_id": meta["page_id"], "url": meta["url"], "title": meta.get("title", "")})
    return deduped


async def stream_chat_response(db: Session, website_id: int, conversation: Conversation, question: str):
    db.add(Message(conversation_id=conversation.id, role=MessageRole.user, content=question))
    db.commit()

    chunks = retrieve_relevant_chunks(website_id, question)

    if not chunks:
        assistant_message = Message(
            conversation_id=conversation.id, role=MessageRole.assistant,
            content=NO_ANSWER_MESSAGE, sources=[], confidence=0.0,
        )
        db.add(assistant_message)
        db.commit()
        db.refresh(assistant_message)
        yield _sse({"type": "delta", "content": NO_ANSWER_MESSAGE})
        yield _sse({
            "type": "done", "sources": [], "confidence": 0.0,
            "message_id": assistant_message.id, "conversation_id": conversation.id,
        })
        return

    sources = _dedupe_sources(chunks)
    confidence = compute_confidence(chunks)
    messages = build_messages(question, chunks)

    full_text = ""
    try:
        async for delta in stream_completion(messages):
            full_text += delta
            yield _sse({"type": "delta", "content": delta})
    except Exception:
        logger.exception("Groq streaming failed for conversation %s", conversation.id)
        if not full_text:
            full_text = "Something went wrong generating a response. Please try again."
            yield _sse({"type": "delta", "content": full_text})

    assistant_message = Message(
        conversation_id=conversation.id, role=MessageRole.assistant,
        content=full_text, sources=sources, confidence=confidence,
    )
    db.add(assistant_message)
    db.commit()
    db.refresh(assistant_message)

    yield _sse({
        "type": "done", "sources": sources, "confidence": confidence,
        "message_id": assistant_message.id, "conversation_id": conversation.id,
    })
```

- [ ] **Step 5: Write `app/api/routes/chat.py`**

```python
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.api.schemas import ChatRequest
from app.services.chat_service import get_or_create_conversation, stream_chat_response

router = APIRouter(tags=["chat"])


@router.post("/chat")
def chat(payload: ChatRequest, db: Session = Depends(get_db)) -> StreamingResponse:
    conversation = get_or_create_conversation(db, payload.website_id, payload.session_id, payload.conversation_id)
    return StreamingResponse(
        stream_chat_response(db, payload.website_id, conversation, payload.question),
        media_type="text/event-stream",
    )
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
pytest tests/test_chat_api.py -v
```
Expected: 3 passed.

- [ ] **Step 7: Commit**

```bash
git add Ai/backend/app/services/chat_service.py Ai/backend/app/api/routes/chat.py Ai/backend/app/api/schemas.py Ai/backend/tests/test_chat_api.py
git commit -m "feat: SSE-streaming chat endpoint with RAG grounding"
```

---

### Task 17: Conversations + message feedback API

**Files:**
- Create: `Ai/backend/app/api/routes/conversations.py`
- Modify: `Ai/backend/app/api/schemas.py` (append `MessageRead`, `ConversationRead`, `ConversationDetailRead`, `FeedbackRequest`)
- Test: `Ai/backend/tests/test_conversations_api.py`

**Interfaces:**
- Produces: `router` exposing `GET /conversations?website_id&session_id` (public, list scoped to that session, newest first), `GET /conversations/{id}` (public, includes ordered `messages`), `DELETE /conversations/{id}` (public — end users can delete their own local conversations), `POST /messages/{id}/feedback` (public, body `{"feedback": "up" | "down"}`, backs thumbs up/down).

- [ ] **Step 1: Write the failing tests**

```python
# Ai/backend/tests/test_conversations_api.py
import os
import tempfile

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture()
def client():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    engine = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False})

    from app.database.session import Base
    from app.database import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    TestSession = sessionmaker(bind=engine)

    from app.main import app
    from app.api.deps import get_db

    def override_get_db():
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client, TestSession

    os.remove(path)


def _seed(TestSession):
    from app.database.models import Conversation, Message, MessageRole, Website

    db = TestSession()
    website = Website(url="https://acme.test", name="Acme")
    db.add(website)
    db.commit()

    convo = Conversation(website_id=website.id, session_id="s1", title="New Chat")
    db.add(convo)
    db.commit()

    msg = Message(conversation_id=convo.id, role=MessageRole.user, content="Hi")
    db.add(msg)
    db.commit()
    db.refresh(msg)
    ids = {"website_id": website.id, "conversation_id": convo.id, "message_id": msg.id}
    db.close()
    return ids


def test_list_conversations_scoped_to_session(client):
    test_client, TestSession = client
    ids = _seed(TestSession)

    resp = test_client.get("/conversations", params={"website_id": ids["website_id"], "session_id": "s1"})
    assert resp.status_code == 200
    assert len(resp.json()) == 1

    resp_other = test_client.get("/conversations", params={"website_id": ids["website_id"], "session_id": "other"})
    assert resp_other.json() == []


def test_get_conversation_detail_includes_messages(client):
    test_client, TestSession = client
    ids = _seed(TestSession)

    resp = test_client.get(f"/conversations/{ids['conversation_id']}")
    assert resp.status_code == 200
    assert len(resp.json()["messages"]) == 1
    assert resp.json()["messages"][0]["content"] == "Hi"


def test_delete_conversation(client):
    test_client, TestSession = client
    ids = _seed(TestSession)

    resp = test_client.delete(f"/conversations/{ids['conversation_id']}")
    assert resp.status_code == 204

    resp = test_client.get(f"/conversations/{ids['conversation_id']}")
    assert resp.status_code == 404


def test_submit_feedback(client):
    test_client, TestSession = client
    ids = _seed(TestSession)

    resp = test_client.post(f"/messages/{ids['message_id']}/feedback", json={"feedback": "up"})
    assert resp.status_code == 200
    assert resp.json()["feedback"] == "up"


def test_submit_invalid_feedback_rejected(client):
    test_client, TestSession = client
    ids = _seed(TestSession)

    resp = test_client.post(f"/messages/{ids['message_id']}/feedback", json={"feedback": "sideways"})
    assert resp.status_code == 422
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd Ai/backend && source venv/bin/activate
pytest tests/test_conversations_api.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'app.api.routes.conversations'`.

- [ ] **Step 3: Append schemas to `app/api/schemas.py`**

```python
from typing import Literal


class MessageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    role: str
    content: str
    sources: list[dict] | None
    confidence: float | None
    feedback: str | None
    created_at: datetime


class ConversationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    website_id: int
    session_id: str
    title: str
    created_at: datetime


class ConversationDetailRead(ConversationRead):
    messages: list[MessageRead]


class FeedbackRequest(BaseModel):
    feedback: Literal["up", "down"]
```

- [ ] **Step 4: Write `app/api/routes/conversations.py`**

```python
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.api.schemas import ConversationDetailRead, ConversationRead, FeedbackRequest, MessageRead
from app.database.models import Conversation, Message

router = APIRouter(tags=["conversations"])


@router.get("/conversations", response_model=list[ConversationRead])
def list_conversations(website_id: int, session_id: str, db: Session = Depends(get_db)) -> list[Conversation]:
    return (
        db.query(Conversation)
        .filter(Conversation.website_id == website_id, Conversation.session_id == session_id)
        .order_by(Conversation.created_at.desc())
        .all()
    )


@router.get("/conversations/{conversation_id}", response_model=ConversationDetailRead)
def get_conversation(conversation_id: int, db: Session = Depends(get_db)) -> Conversation:
    conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    conversation.messages.sort(key=lambda m: m.created_at)
    return conversation


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(conversation_id: int, db: Session = Depends(get_db)) -> None:
    conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    db.delete(conversation)
    db.commit()


@router.post("/messages/{message_id}/feedback", response_model=MessageRead)
def submit_feedback(message_id: int, payload: FeedbackRequest, db: Session = Depends(get_db)) -> Message:
    message = db.query(Message).filter(Message.id == message_id).first()
    if message is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Message not found")
    message.feedback = payload.feedback
    db.commit()
    db.refresh(message)
    return message
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_conversations_api.py -v
```
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add Ai/backend/app/api/routes/conversations.py Ai/backend/app/api/schemas.py Ai/backend/tests/test_conversations_api.py
git commit -m "feat: conversations history and message feedback API"
```

---

### Task 18: Analytics API

**Files:**
- Create: `Ai/backend/app/services/analytics_service.py`
- Create: `Ai/backend/app/api/routes/analytics.py`
- Modify: `Ai/backend/app/api/schemas.py` (append `AnalyticsRead`)
- Test: `Ai/backend/tests/test_analytics.py`

**Interfaces:**
- Produces: `compute_analytics(db: Session, website_id: int) -> dict` (keys: `total_conversations`, `total_messages`, `thumbs_up`, `thumbs_down`, `average_confidence`, `fallback_rate`, `recent_questions`), `router` exposing `GET /analytics/{website_id}` (admin).

- [ ] **Step 1: Write the failing tests**

```python
# Ai/backend/tests/test_analytics.py
import os
import tempfile

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture()
def db_session():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    engine = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False})
    from app.database.session import Base
    from app.database import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    os.remove(path)


def _seed(db):
    from app.database.models import Conversation, Message, MessageRole, Website
    from app.llm.retrieval import NO_ANSWER_MESSAGE

    website = Website(url="https://acme.test", name="Acme")
    db.add(website)
    db.commit()

    convo = Conversation(website_id=website.id, session_id="s1")
    db.add(convo)
    db.commit()

    db.add(Message(conversation_id=convo.id, role=MessageRole.user, content="What is Acme?"))
    db.add(Message(conversation_id=convo.id, role=MessageRole.assistant, content="Acme is a widget maker.", confidence=90.0, feedback="up"))
    db.add(Message(conversation_id=convo.id, role=MessageRole.user, content="What is the moon made of?"))
    db.add(Message(conversation_id=convo.id, role=MessageRole.assistant, content=NO_ANSWER_MESSAGE, confidence=0.0, feedback="down"))
    db.commit()
    return website.id


def test_compute_analytics_aggregates_correctly(db_session):
    from app.services.analytics_service import compute_analytics

    website_id = _seed(db_session)
    result = compute_analytics(db_session, website_id)

    assert result["total_conversations"] == 1
    assert result["total_messages"] == 4
    assert result["thumbs_up"] == 1
    assert result["thumbs_down"] == 1
    assert result["average_confidence"] == 45.0
    assert result["fallback_rate"] == 50.0
    assert "What is Acme?" in result["recent_questions"]


def test_compute_analytics_empty_website(db_session):
    from app.database.models import Website
    from app.services.analytics_service import compute_analytics

    website = Website(url="https://empty.test", name="Empty")
    db_session.add(website)
    db_session.commit()

    result = compute_analytics(db_session, website.id)
    assert result["total_conversations"] == 0
    assert result["total_messages"] == 0
    assert result["average_confidence"] == 0.0
    assert result["fallback_rate"] == 0.0
    assert result["recent_questions"] == []
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd Ai/backend && source venv/bin/activate
pytest tests/test_analytics.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.analytics_service'`.

- [ ] **Step 3: Write `app/services/analytics_service.py`**

```python
from sqlalchemy.orm import Session

from app.database.models import Conversation, Message, MessageRole
from app.llm.retrieval import NO_ANSWER_MESSAGE


def compute_analytics(db: Session, website_id: int) -> dict:
    total_conversations = db.query(Conversation).filter(Conversation.website_id == website_id).count()

    assistant_messages = (
        db.query(Message)
        .join(Conversation)
        .filter(Conversation.website_id == website_id, Message.role == MessageRole.assistant)
        .all()
    )
    user_messages = (
        db.query(Message)
        .join(Conversation)
        .filter(Conversation.website_id == website_id, Message.role == MessageRole.user)
        .order_by(Message.created_at.desc())
        .limit(10)
        .all()
    )
    total_messages = (
        db.query(Message).join(Conversation).filter(Conversation.website_id == website_id).count()
    )

    thumbs_up = sum(1 for m in assistant_messages if m.feedback and m.feedback.value == "up")
    thumbs_down = sum(1 for m in assistant_messages if m.feedback and m.feedback.value == "down")

    confidences = [m.confidence for m in assistant_messages if m.confidence is not None]
    average_confidence = round(sum(confidences) / len(confidences), 1) if confidences else 0.0

    fallback_count = sum(1 for m in assistant_messages if m.content == NO_ANSWER_MESSAGE)
    fallback_rate = round((fallback_count / len(assistant_messages)) * 100, 1) if assistant_messages else 0.0

    return {
        "total_conversations": total_conversations,
        "total_messages": total_messages,
        "thumbs_up": thumbs_up,
        "thumbs_down": thumbs_down,
        "average_confidence": average_confidence,
        "fallback_rate": fallback_rate,
        "recent_questions": [m.content for m in user_messages],
    }
```

- [ ] **Step 4: Append `AnalyticsRead` to `app/api/schemas.py`**

```python
class AnalyticsRead(BaseModel):
    total_conversations: int
    total_messages: int
    thumbs_up: int
    thumbs_down: int
    average_confidence: float
    fallback_rate: float
    recent_questions: list[str]
```

- [ ] **Step 5: Write `app/api/routes/analytics.py`**

```python
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
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
pytest tests/test_analytics.py -v
```
Expected: 2 passed.

- [ ] **Step 7: Commit**

```bash
git add Ai/backend/app/services/analytics_service.py Ai/backend/app/api/routes/analytics.py Ai/backend/app/api/schemas.py Ai/backend/tests/test_analytics.py
git commit -m "feat: chat analytics API"
```

---

### Task 19: `main.py` wiring — app assembly, CORS, lifespan, error handling

**Files:**
- Create: `Ai/backend/app/main.py`
- Test: `Ai/backend/tests/test_main.py`

**Interfaces:**
- Consumes: every router from Tasks 4, 11-13, 15-18; `app.database.session.{init_db, SessionLocal}`; `app.services.auth_service.seed_admin_user`; `app.scheduler.setup.{start_scheduler, shutdown_scheduler}`
- Produces: `app` (the FastAPI instance every test file in Tasks 11-13, 16-18 imports as `from app.main import app`).

> This task's file has already been referenced by every prior API test (`from app.main import app`). Once this task is done, re-run the full backend test suite (Step 4 below) to confirm everything integrates.

- [ ] **Step 1: Write the failing test**

```python
# Ai/backend/tests/test_main.py
def test_health_check():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


def test_cors_headers_present_for_allowed_origin():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        resp = client.get("/health", headers={"Origin": "http://localhost:5173"})
        assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_unhandled_exception_returns_500_json(monkeypatch):
    from fastapi.testclient import TestClient

    from app.main import app
    import app.api.routes.websites as websites_module

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(websites_module, "list_websites", boom)

    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.get("/websites")
        assert resp.status_code == 500
        assert resp.json() == {"detail": "Internal server error"}
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd Ai/backend && source venv/bin/activate
pytest tests/test_main.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'app.main'`.

- [ ] **Step 3: Write `app/main.py`**

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import analytics, auth, chat, conversations, crawl, pages, websites
from app.config import settings
from app.database.session import SessionLocal, init_db
from app.scheduler.setup import shutdown_scheduler, start_scheduler
from app.services.auth_service import seed_admin_user
from app.utils.logging import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    db = SessionLocal()
    try:
        seed_admin_user(db)
    finally:
        db.close()
    start_scheduler()
    logger.info("Application startup complete")
    yield
    shutdown_scheduler()
    logger.info("Application shutdown complete")


app = FastAPI(title="Multi-Website AI Chatbot Platform", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


app.include_router(auth.router)
app.include_router(websites.router)
app.include_router(crawl.router)
app.include_router(pages.router)
app.include_router(chat.router)
app.include_router(conversations.router)
app.include_router(analytics.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
```

- [ ] **Step 4: Run the entire backend test suite**

```bash
pytest -v
```
Expected: all tests across every task pass (roughly 45+ tests). Fix any integration mismatches surfaced now that every router is wired together (e.g., import cycles — if one appears between `app.main` and a router, move the shared dependency into `app/api/deps.py`, which every route already imports from).

- [ ] **Step 5: Manual smoke test of the real server**

```bash
cd Ai/backend && source venv/bin/activate
uvicorn app.main:app --reload --port 8000
```
In another terminal:
```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/auth/login -H "Content-Type: application/json" \
  -d '{"email":"admin@example.com","password":"test1234"}'
```
Expected: `{"status":"ok"}` then a JSON body containing `access_token`.

- [ ] **Step 6: Commit**

```bash
git add Ai/backend/app/main.py Ai/backend/tests/test_main.py
git commit -m "feat: wire FastAPI app with routers, CORS, lifespan, and error handling"
```

---

## Frontend Tasks

The frontend has no test runner in the requested stack, so each task substitutes `npm run build` (TypeScript type-check + Vite bundle) as its automated gate, plus a manual dev-server check for behavior — per the Global Constraints.

### Task 20: Frontend scaffold — Vite/React/TS, MUI theme, router, axios client, React Query

**Files:**
- Create: `Ai/frontend/` (via `npm create vite@latest`)
- Create: `Ai/frontend/.env.example`
- Create: `Ai/frontend/src/theme/theme.ts`
- Create: `Ai/frontend/src/context/ThemeModeContext.tsx`
- Create: `Ai/frontend/src/context/AuthContext.tsx`
- Create: `Ai/frontend/src/api/client.ts`
- Create: `Ai/frontend/src/api/types.ts`
- Create: `Ai/frontend/src/pages/DemoPage.tsx`
- Modify: `Ai/frontend/src/App.tsx`, `Ai/frontend/src/main.tsx`

**Interfaces:**
- Produces: `useThemeMode() -> { mode: 'light' | 'dark', toggleMode: () => void }`, `useAuth() -> { token: string | null, login: (t: string) => void, logout: () => void }`, `apiClient` (configured axios instance), `setAuthToken(token: string | null) -> void`, TS types `Website`, `Page`, `CrawlJob`, `ChatMessage`, `MessageSource`, `Conversation`, `ConversationDetail`, `AnalyticsSummary` in `src/api/types.ts` — imported by every later frontend task.

- [ ] **Step 1: Scaffold the Vite project and install dependencies**

```bash
cd "Ai"
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install @mui/material@^6 @mui/icons-material@^6 @emotion/react@^11 @emotion/styled@^11 \
  @tanstack/react-query@^5 axios@^1.7 framer-motion@^11 react-markdown@^9 react-router-dom@^6.28
```

- [ ] **Step 2: Write `.env.example`**

```
VITE_API_URL=http://localhost:8000
```
Copy it to `.env`.

- [ ] **Step 3: Write `src/theme/theme.ts`**

```typescript
import { createTheme, type Theme } from '@mui/material/styles'

export function buildTheme(mode: 'light' | 'dark'): Theme {
  return createTheme({
    palette: {
      mode,
      primary: { main: '#6366f1' },
      secondary: { main: '#22d3ee' },
      background: {
        default: mode === 'dark' ? '#0b0f19' : '#f5f7fb',
        paper: mode === 'dark' ? '#141826' : '#ffffff',
      },
    },
    shape: { borderRadius: 16 },
    typography: {
      fontFamily: '"Inter", "Segoe UI", Roboto, sans-serif',
    },
  })
}
```

- [ ] **Step 4: Write `src/context/ThemeModeContext.tsx`**

```typescript
import { createContext, useContext, useMemo, useState, type ReactNode } from 'react'
import { ThemeProvider, CssBaseline } from '@mui/material'
import { buildTheme } from '../theme/theme'

type ThemeMode = 'light' | 'dark'

interface ThemeModeContextValue {
  mode: ThemeMode
  toggleMode: () => void
}

const ThemeModeContext = createContext<ThemeModeContextValue | undefined>(undefined)

export function ThemeModeProvider({ children }: { children: ReactNode }) {
  const [mode, setMode] = useState<ThemeMode>(() => {
    const stored = localStorage.getItem('theme-mode')
    if (stored === 'light' || stored === 'dark') return stored
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
  })

  const toggleMode = () => {
    setMode((prev) => {
      const next: ThemeMode = prev === 'light' ? 'dark' : 'light'
      localStorage.setItem('theme-mode', next)
      return next
    })
  }

  const theme = useMemo(() => buildTheme(mode), [mode])

  return (
    <ThemeModeContext.Provider value={{ mode, toggleMode }}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        {children}
      </ThemeProvider>
    </ThemeModeContext.Provider>
  )
}

export function useThemeMode(): ThemeModeContextValue {
  const ctx = useContext(ThemeModeContext)
  if (!ctx) throw new Error('useThemeMode must be used within ThemeModeProvider')
  return ctx
}
```

- [ ] **Step 5: Write `src/api/client.ts`**

```typescript
import axios from 'axios'

export const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_URL ?? 'http://localhost:8000',
})

export function setAuthToken(token: string | null): void {
  if (token) {
    apiClient.defaults.headers.common.Authorization = `Bearer ${token}`
  } else {
    delete apiClient.defaults.headers.common.Authorization
  }
}
```

- [ ] **Step 6: Write `src/api/types.ts`**

```typescript
export interface Website {
  id: number
  url: string
  name: string
  logo_url: string | null
  status: 'active' | 'crawling' | 'error'
  sitemap_url: string | null
  crawl_depth_limit: number
  max_pages: number
  created_at: string
  updated_at: string
  last_synced_at: string | null
}

export interface Page {
  id: number
  website_id: number
  url: string
  title: string | null
  status: 'indexed' | 'failed' | 'skipped' | 'deleted'
  content_hash: string | null
  last_crawled_at: string | null
  indexed_at: string | null
}

export interface CrawlJob {
  id: number
  website_id: number
  status: 'running' | 'completed' | 'failed'
  pages_found: number
  pages_indexed: number
  pages_failed: number
  started_at: string
  finished_at: string | null
  error_log: string | null
}

export interface MessageSource {
  page_id: number
  url: string
  title: string
}

export interface ChatMessage {
  id: number
  role: 'user' | 'assistant'
  content: string
  sources: MessageSource[] | null
  confidence: number | null
  feedback: 'up' | 'down' | null
  created_at: string
}

export interface Conversation {
  id: number
  website_id: number
  session_id: string
  title: string
  created_at: string
}

export interface ConversationDetail extends Conversation {
  messages: ChatMessage[]
}

export interface AnalyticsSummary {
  total_conversations: number
  total_messages: number
  thumbs_up: number
  thumbs_down: number
  average_confidence: number
  fallback_rate: number
  recent_questions: string[]
}
```

- [ ] **Step 7: Write `src/context/AuthContext.tsx`**

```typescript
import { createContext, useContext, useState, type ReactNode } from 'react'
import { setAuthToken } from '../api/client'

interface AuthContextValue {
  token: string | null
  login: (token: string) => void
  logout: () => void
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [token, setToken] = useState<string | null>(() => {
    const stored = localStorage.getItem('admin_token')
    if (stored) setAuthToken(stored)
    return stored
  })

  const login = (newToken: string) => {
    localStorage.setItem('admin_token', newToken)
    setAuthToken(newToken)
    setToken(newToken)
  }

  const logout = () => {
    localStorage.removeItem('admin_token')
    setAuthToken(null)
    setToken(null)
  }

  return <AuthContext.Provider value={{ token, login, logout }}>{children}</AuthContext.Provider>
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used within AuthProvider')
  return ctx
}
```

- [ ] **Step 8: Write `src/pages/DemoPage.tsx` (placeholder — the widget is wired in here in Task 29)**

```typescript
import { Box, Typography } from '@mui/material'

export default function DemoPage() {
  return (
    <Box sx={{ p: 4 }}>
      <Typography variant="h4">Demo site</Typography>
      <Typography color="text.secondary">The chat widget will appear here.</Typography>
    </Box>
  )
}
```

- [ ] **Step 9: Rewrite `src/App.tsx`**

```typescript
import { Routes, Route } from 'react-router-dom'
import DemoPage from './pages/DemoPage'

function App() {
  return (
    <Routes>
      <Route path="/" element={<DemoPage />} />
    </Routes>
  )
}

export default App
```

- [ ] **Step 10: Rewrite `src/main.tsx`**

```typescript
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { BrowserRouter } from 'react-router-dom'
import App from './App'
import { ThemeModeProvider } from './context/ThemeModeContext'
import { AuthProvider } from './context/AuthContext'

const queryClient = new QueryClient({
  defaultOptions: { queries: { retry: 1, refetchOnWindowFocus: false } },
})

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <ThemeModeProvider>
        <AuthProvider>
          <BrowserRouter>
            <App />
          </BrowserRouter>
        </AuthProvider>
      </ThemeModeProvider>
    </QueryClientProvider>
  </StrictMode>,
)
```

- [ ] **Step 11: Run the build to verify types check and the bundle succeeds**

```bash
cd Ai/frontend
npm run build
```
Expected: builds successfully with no TypeScript errors.

- [ ] **Step 12: Manual dev-server check**

```bash
npm run dev
```
Open `http://localhost:5173` — expect to see "Demo site" text, no console errors. Toggle your OS light/dark mode and refresh — background should follow it.

- [ ] **Step 13: Commit**

```bash
git add Ai/frontend
git commit -m "feat: frontend scaffold with MUI theme, router, auth context, and API client"
```

---

### Task 21: Admin login page + protected layout shell

**Files:**
- Create: `Ai/frontend/src/admin/LoginPage.tsx`
- Create: `Ai/frontend/src/admin/RequireAdmin.tsx`
- Create: `Ai/frontend/src/admin/AdminLayout.tsx`
- Modify: `Ai/frontend/src/App.tsx`

**Interfaces:**
- Consumes: `useAuth`, `useThemeMode`, `apiClient` (Task 20)
- Produces: `<AdminLayout />` (renders an `<Outlet />` at `/admin/*` for nested pages — Tasks 22/24 add real routes for `websites` and `analytics`; this task adds temporary placeholder routes so the shell is independently testable), `<RequireAdmin>` (redirects to `/admin/login` when unauthenticated).

- [ ] **Step 1: Write `src/admin/LoginPage.tsx`**

```typescript
import { useState, type FormEvent } from 'react'
import { useNavigate } from 'react-router-dom'
import { Alert, Box, Button, Paper, TextField, Typography } from '@mui/material'
import { apiClient } from '../api/client'
import { useAuth } from '../context/AuthContext'

export default function LoginPage() {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const { login } = useAuth()
  const navigate = useNavigate()

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault()
    setError(null)
    setLoading(true)
    try {
      const { data } = await apiClient.post<{ access_token: string }>('/auth/login', { email, password })
      login(data.access_token)
      navigate('/admin')
    } catch {
      setError('Invalid email or password')
    } finally {
      setLoading(false)
    }
  }

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh', alignItems: 'center', justifyContent: 'center' }}>
      <Paper elevation={0} sx={{ p: 4, width: 360, borderRadius: 4 }} component="form" onSubmit={handleSubmit}>
        <Typography variant="h5" sx={{ mb: 2 }}>Admin Login</Typography>
        {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
        <TextField label="Email" fullWidth margin="normal" value={email} onChange={(e) => setEmail(e.target.value)} />
        <TextField label="Password" type="password" fullWidth margin="normal" value={password} onChange={(e) => setPassword(e.target.value)} />
        <Button type="submit" variant="contained" fullWidth sx={{ mt: 2 }} disabled={loading}>
          {loading ? 'Signing in…' : 'Sign in'}
        </Button>
      </Paper>
    </Box>
  )
}
```

- [ ] **Step 2: Write `src/admin/RequireAdmin.tsx`**

```typescript
import type { ReactNode } from 'react'
import { Navigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'

export default function RequireAdmin({ children }: { children: ReactNode }) {
  const { token } = useAuth()
  if (!token) return <Navigate to="/admin/login" replace />
  return <>{children}</>
}
```

- [ ] **Step 3: Write `src/admin/AdminLayout.tsx`**

```typescript
import { AppBar, Toolbar, Typography, Button, Box, Tabs, Tab, IconButton } from '@mui/material'
import Brightness4Icon from '@mui/icons-material/Brightness4'
import Brightness7Icon from '@mui/icons-material/Brightness7'
import { Outlet, useLocation, useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import { useThemeMode } from '../context/ThemeModeContext'

const TABS = [
  { label: 'Websites', path: '/admin/websites' },
  { label: 'Analytics', path: '/admin/analytics' },
]

export default function AdminLayout() {
  const { logout } = useAuth()
  const { mode, toggleMode } = useThemeMode()
  const location = useLocation()
  const navigate = useNavigate()

  const activeTab = TABS.findIndex((t) => location.pathname.startsWith(t.path))

  return (
    <Box>
      <AppBar position="static" elevation={0}>
        <Toolbar>
          <Typography variant="h6" sx={{ flexGrow: 1 }}>Chatbot Admin</Typography>
          <IconButton color="inherit" onClick={toggleMode}>
            {mode === 'dark' ? <Brightness7Icon /> : <Brightness4Icon />}
          </IconButton>
          <Button color="inherit" onClick={logout}>Logout</Button>
        </Toolbar>
        <Tabs value={activeTab === -1 ? 0 : activeTab} sx={{ px: 2 }}>
          {TABS.map((tab) => (
            <Tab key={tab.path} label={tab.label} onClick={() => navigate(tab.path)} />
          ))}
        </Tabs>
      </AppBar>
      <Box sx={{ p: 3 }}>
        <Outlet />
      </Box>
    </Box>
  )
}
```

- [ ] **Step 4: Rewrite `src/App.tsx` with admin routes (placeholders for `websites`/`analytics`, replaced in Tasks 22 & 24)**

```typescript
import { Navigate, Route, Routes } from 'react-router-dom'
import DemoPage from './pages/DemoPage'
import LoginPage from './admin/LoginPage'
import RequireAdmin from './admin/RequireAdmin'
import AdminLayout from './admin/AdminLayout'

function App() {
  return (
    <Routes>
      <Route path="/" element={<DemoPage />} />
      <Route path="/admin/login" element={<LoginPage />} />
      <Route
        path="/admin"
        element={
          <RequireAdmin>
            <AdminLayout />
          </RequireAdmin>
        }
      >
        <Route index element={<Navigate to="/admin/websites" replace />} />
        <Route path="websites" element={<div>Websites page coming soon</div>} />
        <Route path="analytics" element={<div>Analytics page coming soon</div>} />
      </Route>
    </Routes>
  )
}

export default App
```

- [ ] **Step 5: Run the build**

```bash
cd Ai/frontend
npm run build
```
Expected: succeeds with no TypeScript errors.

- [ ] **Step 6: Manual check against the running backend**

With the backend running (`uvicorn app.main:app --reload` from Task 19), run `npm run dev`, visit `http://localhost:5173/admin` — expect redirect to `/admin/login`. Log in with the seeded admin credentials from `.env` — expect redirect to `/admin/websites` showing the placeholder text, tabs and logout button visible, dark-mode toggle working. Click Logout — expect redirect back to `/admin/login`.

- [ ] **Step 7: Commit**

```bash
git add Ai/frontend/src/admin Ai/frontend/src/App.tsx
git commit -m "feat: admin login and protected layout shell"
```

---

### Task 22: Admin Websites page — CRUD, crawl/sync/reindex, indexed pages, crawl status

**Files:**
- Create: `Ai/frontend/src/api/hooks.ts` (React Query hooks for websites/pages/crawl-jobs/analytics/conversations — grows across Tasks 22-24)
- Create: `Ai/frontend/src/admin/WebsitesPage.tsx`
- Create: `Ai/frontend/src/admin/WebsiteFormDialog.tsx`
- Create: `Ai/frontend/src/admin/WebsiteDetailDrawer.tsx` (indexed pages list + crawl job history for one website)
- Modify: `Ai/frontend/src/App.tsx` (replace the `websites` placeholder route)

**Interfaces:**
- Produces: `useWebsites()`, `useCreateWebsite()`, `useUpdateWebsite()`, `useDeleteWebsite()`, `useCrawlWebsite()`, `useSyncWebsite()`, `useReindexWebsite()`, `usePages(websiteId)`, `useCrawlJobs(websiteId)` — all React Query hooks wrapping `apiClient` calls to the matching Task 11-13 endpoints, invalidating the `["websites"]` query key on any mutation.

- [ ] **Step 1: Write `src/api/hooks.ts`**

```typescript
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { apiClient } from './client'
import type { AnalyticsSummary, CrawlJob, Page, Website } from './types'

export function useWebsites() {
  return useQuery({
    queryKey: ['websites'],
    queryFn: async () => (await apiClient.get<Website[]>('/websites')).data,
  })
}

export function useCreateWebsite() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (payload: { url: string; name: string; logo_url?: string }) =>
      apiClient.post<Website>('/websites', payload),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['websites'] }),
  })
}

export function useUpdateWebsite() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ id, ...payload }: { id: number; name?: string; logo_url?: string }) =>
      apiClient.patch<Website>(`/websites/${id}`, payload),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['websites'] }),
  })
}

export function useDeleteWebsite() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: number) => apiClient.delete(`/website/${id}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['websites'] }),
  })
}

function useTriggerAction(path: (id: number) => string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: number) => apiClient.post(path(id)),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['websites'] }),
  })
}

export const useCrawlWebsite = () => useTriggerAction((id) => `/crawl/${id}`)
export const useSyncWebsite = () => useTriggerAction((id) => `/sync/${id}`)
export const useReindexWebsite = () => useTriggerAction((id) => `/reindex/${id}`)

export function usePages(websiteId: number | null) {
  return useQuery({
    queryKey: ['pages', websiteId],
    queryFn: async () => (await apiClient.get<Page[]>(`/pages/${websiteId}`)).data,
    enabled: websiteId !== null,
  })
}

export function useCrawlJobs(websiteId: number | null) {
  return useQuery({
    queryKey: ['crawl-jobs', websiteId],
    queryFn: async () => (await apiClient.get<CrawlJob[]>(`/websites/${websiteId}/crawl-jobs`)).data,
    enabled: websiteId !== null,
    refetchInterval: 5000,
  })
}

export function useAnalytics(websiteId: number | null) {
  return useQuery({
    queryKey: ['analytics', websiteId],
    queryFn: async () => (await apiClient.get<AnalyticsSummary>(`/analytics/${websiteId}`)).data,
    enabled: websiteId !== null,
  })
}
```

- [ ] **Step 2: Write `src/admin/WebsiteFormDialog.tsx`**

```typescript
import { useState, type FormEvent } from 'react'
import { Button, Dialog, DialogActions, DialogContent, DialogTitle, TextField } from '@mui/material'
import { useCreateWebsite } from '../api/hooks'

export default function WebsiteFormDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [url, setUrl] = useState('')
  const [name, setName] = useState('')
  const createWebsite = useCreateWebsite()

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault()
    createWebsite.mutate({ url, name }, { onSuccess: () => { setUrl(''); setName(''); onClose() } })
  }

  return (
    <Dialog open={open} onClose={onClose} fullWidth maxWidth="sm">
      <DialogTitle>Add Website</DialogTitle>
      <DialogContent component="form" id="website-form" onSubmit={handleSubmit} sx={{ display: 'flex', flexDirection: 'column', gap: 2, pt: 1 }}>
        <TextField label="Website URL" value={url} onChange={(e) => setUrl(e.target.value)} required fullWidth autoFocus />
        <TextField label="Display Name" value={name} onChange={(e) => setName(e.target.value)} required fullWidth />
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button type="submit" form="website-form" variant="contained" disabled={createWebsite.isPending}>
          {createWebsite.isPending ? 'Adding…' : 'Add'}
        </Button>
      </DialogActions>
    </Dialog>
  )
}
```

- [ ] **Step 3: Write `src/admin/WebsiteDetailDrawer.tsx`**

```typescript
import { Box, Chip, Drawer, List, ListItem, ListItemText, Typography } from '@mui/material'
import { usePages, useCrawlJobs } from '../api/hooks'
import type { Website } from '../api/types'

export default function WebsiteDetailDrawer({ website, onClose }: { website: Website | null; onClose: () => void }) {
  const { data: pages } = usePages(website?.id ?? null)
  const { data: jobs } = useCrawlJobs(website?.id ?? null)

  return (
    <Drawer anchor="right" open={website !== null} onClose={onClose}>
      <Box sx={{ width: 420, p: 3 }}>
        <Typography variant="h6">{website?.name}</Typography>
        <Typography variant="subtitle2" sx={{ mt: 2 }}>Crawl Jobs</Typography>
        <List dense>
          {jobs?.map((job) => (
            <ListItem key={job.id}>
              <ListItemText
                primary={`${job.status} — found ${job.pages_found}, indexed ${job.pages_indexed}, failed ${job.pages_failed}`}
                secondary={new Date(job.started_at).toLocaleString()}
              />
            </ListItem>
          ))}
        </List>
        <Typography variant="subtitle2" sx={{ mt: 2 }}>Indexed Pages ({pages?.length ?? 0})</Typography>
        <List dense>
          {pages?.map((page) => (
            <ListItem key={page.id}>
              <ListItemText primary={page.title || page.url} secondary={page.url} />
              <Chip size="small" label={page.status} color={page.status === 'indexed' ? 'success' : page.status === 'failed' ? 'error' : 'default'} />
            </ListItem>
          ))}
        </List>
      </Box>
    </Drawer>
  )
}
```

- [ ] **Step 4: Write `src/admin/WebsitesPage.tsx`**

```typescript
import { useState } from 'react'
import { Box, Button, Chip, IconButton, Paper, Stack, Table, TableBody, TableCell, TableHead, TableRow, Tooltip } from '@mui/material'
import AddIcon from '@mui/icons-material/Add'
import RefreshIcon from '@mui/icons-material/Refresh'
import SyncIcon from '@mui/icons-material/Sync'
import DeleteIcon from '@mui/icons-material/Delete'
import InfoIcon from '@mui/icons-material/Info'
import { useCrawlWebsite, useDeleteWebsite, useReindexWebsite, useSyncWebsite, useWebsites } from '../api/hooks'
import WebsiteFormDialog from './WebsiteFormDialog'
import WebsiteDetailDrawer from './WebsiteDetailDrawer'
import type { Website } from '../api/types'

const STATUS_COLOR: Record<Website['status'], 'success' | 'warning' | 'error'> = {
  active: 'success',
  crawling: 'warning',
  error: 'error',
}

export default function WebsitesPage() {
  const { data: websites, isLoading } = useWebsites()
  const [formOpen, setFormOpen] = useState(false)
  const [selected, setSelected] = useState<Website | null>(null)
  const crawl = useCrawlWebsite()
  const sync = useSyncWebsite()
  const reindex = useReindexWebsite()
  const remove = useDeleteWebsite()

  return (
    <Box>
      <Stack direction="row" justifyContent="space-between" sx={{ mb: 2 }}>
        <h2>Websites</h2>
        <Button startIcon={<AddIcon />} variant="contained" onClick={() => setFormOpen(true)}>
          Add Website
        </Button>
      </Stack>
      <Paper sx={{ borderRadius: 3 }}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Name</TableCell>
              <TableCell>URL</TableCell>
              <TableCell>Status</TableCell>
              <TableCell>Last Synced</TableCell>
              <TableCell align="right">Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {isLoading && (
              <TableRow><TableCell colSpan={5}>Loading…</TableCell></TableRow>
            )}
            {websites?.map((site) => (
              <TableRow key={site.id}>
                <TableCell>{site.name}</TableCell>
                <TableCell>{site.url}</TableCell>
                <TableCell><Chip size="small" label={site.status} color={STATUS_COLOR[site.status]} /></TableCell>
                <TableCell>{site.last_synced_at ? new Date(site.last_synced_at).toLocaleString() : 'Never'}</TableCell>
                <TableCell align="right">
                  <Tooltip title="Crawl"><IconButton onClick={() => crawl.mutate(site.id)}><RefreshIcon /></IconButton></Tooltip>
                  <Tooltip title="Sync"><IconButton onClick={() => sync.mutate(site.id)}><SyncIcon /></IconButton></Tooltip>
                  <Tooltip title="Reindex"><IconButton onClick={() => reindex.mutate(site.id)}><SyncIcon color="secondary" /></IconButton></Tooltip>
                  <Tooltip title="Details"><IconButton onClick={() => setSelected(site)}><InfoIcon /></IconButton></Tooltip>
                  <Tooltip title="Delete"><IconButton onClick={() => remove.mutate(site.id)}><DeleteIcon color="error" /></IconButton></Tooltip>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </Paper>
      <WebsiteFormDialog open={formOpen} onClose={() => setFormOpen(false)} />
      <WebsiteDetailDrawer website={selected} onClose={() => setSelected(null)} />
    </Box>
  )
}
```

- [ ] **Step 5: Wire the route in `src/App.tsx`** — replace `<Route path="websites" element={<div>Websites page coming soon</div>} />` with `<Route path="websites" element={<WebsitesPage />} />` and import it.

- [ ] **Step 6: Build and manually verify**

```bash
cd Ai/frontend && npm run build && npm run dev
```
With the backend running, log in, add a website, click Crawl, open Details and confirm the crawl job/pages lists populate (poll every 5s via `refetchInterval`), then delete it and confirm it disappears.

- [ ] **Step 7: Commit**

```bash
git add Ai/frontend/src
git commit -m "feat: admin websites page with CRUD, crawl actions, and detail drawer"
```

---

### Task 23: Admin Analytics page

**Files:**
- Create: `Ai/frontend/src/admin/AnalyticsPage.tsx`
- Modify: `Ai/frontend/src/App.tsx` (replace the `analytics` placeholder route)

**Interfaces:**
- Consumes: `useWebsites`, `useAnalytics` (Task 22)

- [ ] **Step 1: Write `src/admin/AnalyticsPage.tsx`**

```typescript
import { useState } from 'react'
import { Box, Card, CardContent, Grid, List, ListItem, ListItemText, MenuItem, Select, Typography } from '@mui/material'
import { useAnalytics, useWebsites } from '../api/hooks'

export default function AnalyticsPage() {
  const { data: websites } = useWebsites()
  const [websiteId, setWebsiteId] = useState<number | ''>('')
  const { data } = useAnalytics(websiteId === '' ? null : websiteId)

  return (
    <Box>
      <Typography variant="h5" sx={{ mb: 2 }}>Analytics</Typography>
      <Select value={websiteId} onChange={(e) => setWebsiteId(e.target.value as number)} displayEmpty sx={{ mb: 3, minWidth: 240 }}>
        <MenuItem value="" disabled>Select a website</MenuItem>
        {websites?.map((w) => <MenuItem key={w.id} value={w.id}>{w.name}</MenuItem>)}
      </Select>

      {data && (
        <Grid container spacing={2}>
          {[
            ['Conversations', data.total_conversations],
            ['Messages', data.total_messages],
            ['Thumbs Up', data.thumbs_up],
            ['Thumbs Down', data.thumbs_down],
            ['Avg Confidence', `${data.average_confidence}%`],
            ['Fallback Rate', `${data.fallback_rate}%`],
          ].map(([label, value]) => (
            <Grid item xs={6} md={4} key={label as string}>
              <Card sx={{ borderRadius: 3 }}>
                <CardContent>
                  <Typography variant="body2" color="text.secondary">{label}</Typography>
                  <Typography variant="h5">{value}</Typography>
                </CardContent>
              </Card>
            </Grid>
          ))}
          <Grid item xs={12}>
            <Card sx={{ borderRadius: 3 }}>
              <CardContent>
                <Typography variant="subtitle1" sx={{ mb: 1 }}>Recent Questions</Typography>
                <List dense>
                  {data.recent_questions.map((q, i) => <ListItem key={i}><ListItemText primary={q} /></ListItem>)}
                </List>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}
    </Box>
  )
}
```

- [ ] **Step 2: Wire the route** — replace the `analytics` placeholder in `App.tsx` with `<Route path="analytics" element={<AnalyticsPage />} />`.

- [ ] **Step 3: Build and manually verify**

```bash
cd Ai/frontend && npm run build && npm run dev
```
Select a website with existing conversations (chat with it first via Task 25's widget) and confirm the stat cards and recent questions populate.

- [ ] **Step 4: Commit**

```bash
git add Ai/frontend/src
git commit -m "feat: admin analytics page"
```

---

### Task 24: Chat widget — floating button, SSE streaming window, markdown, sidebar, actions

**Files:**
- Create: `Ai/frontend/src/widget/useChatSession.ts`
- Create: `Ai/frontend/src/widget/FloatingButton.tsx`
- Create: `Ai/frontend/src/widget/MessageBubble.tsx`
- Create: `Ai/frontend/src/widget/ChatInput.tsx`
- Create: `Ai/frontend/src/widget/ConversationSidebar.tsx`
- Create: `Ai/frontend/src/widget/ChatWindow.tsx`
- Create: `Ai/frontend/src/widget/ChatWidget.tsx`
- Modify: `Ai/frontend/src/index.css` (mobile full-screen override)

**Interfaces:**
- Produces: `<ChatWidget website={Website} />` — the single component Task 25 mounts on the demo page. Internally: `useChatSession(websiteId)` manages session id, conversation id, streaming message state, and parses the `POST /chat` SSE body via `fetch` + a manual `ReadableStream` reader (raw `EventSource` can't send a POST body, hence the manual parse).

- [ ] **Step 1: Write `src/widget/useChatSession.ts`**

```typescript
import { useCallback, useRef, useState } from 'react'
import { apiClient } from '../api/client'
import type { ChatMessage, MessageSource } from '../api/types'

function getSessionId(websiteId: number): string {
  const key = `chat-session-${websiteId}`
  let id = localStorage.getItem(key)
  if (!id) {
    id = crypto.randomUUID()
    localStorage.setItem(key, id)
  }
  return id
}

export interface StreamingMessage {
  id: number | 'streaming'
  role: 'user' | 'assistant'
  content: string
  sources?: MessageSource[]
  confidence?: number
  feedback?: 'up' | 'down' | null
}

export function useChatSession(websiteId: number) {
  const sessionId = useRef(getSessionId(websiteId)).current
  const [conversationId, setConversationId] = useState<number | null>(null)
  const [messages, setMessages] = useState<StreamingMessage[]>([])
  const [isStreaming, setIsStreaming] = useState(false)

  const sendMessage = useCallback(
    async (question: string) => {
      setMessages((prev) => [...prev, { id: Date.now(), role: 'user', content: question }])
      setMessages((prev) => [...prev, { id: 'streaming', role: 'assistant', content: '' }])
      setIsStreaming(true)

      const baseURL = apiClient.defaults.baseURL ?? ''
      const response = await fetch(`${baseURL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ website_id: websiteId, question, session_id: sessionId, conversation_id: conversationId }),
      })

      const reader = response.body?.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      if (!reader) {
        setIsStreaming(false)
        return
      }

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })

        const blocks = buffer.split('\n\n')
        buffer = blocks.pop() ?? ''

        for (const block of blocks) {
          if (!block.startsWith('data: ')) continue
          const event = JSON.parse(block.slice(6))

          if (event.type === 'delta') {
            setMessages((prev) =>
              prev.map((m) => (m.id === 'streaming' ? { ...m, content: m.content + event.content } : m)),
            )
          } else if (event.type === 'done') {
            setConversationId(event.conversation_id)
            setMessages((prev) =>
              prev.map((m) =>
                m.id === 'streaming'
                  ? { ...m, id: event.message_id, sources: event.sources, confidence: event.confidence, feedback: null }
                  : m,
              ),
            )
          }
        }
      }
      setIsStreaming(false)
    },
    [websiteId, sessionId, conversationId],
  )

  const clearChat = useCallback(() => {
    setMessages([])
    setConversationId(null)
  }, [])

  const loadConversation = useCallback(async (id: number) => {
    const { data } = await apiClient.get(`/conversations/${id}`)
    setConversationId(id)
    setMessages(data.messages.map((m: ChatMessage) => ({ ...m })))
  }, [])

  return { sessionId, conversationId, messages, isStreaming, sendMessage, clearChat, loadConversation }
}
```

- [ ] **Step 2: Write `src/widget/FloatingButton.tsx`**

```typescript
import { IconButton } from '@mui/material'
import ChatIcon from '@mui/icons-material/Chat'
import CloseIcon from '@mui/icons-material/Close'
import { motion } from 'framer-motion'

export default function FloatingButton({ open, onClick }: { open: boolean; onClick: () => void }) {
  return (
    <motion.div
      style={{ position: 'fixed', bottom: 24, right: 24, zIndex: 1300 }}
      whileHover={{ scale: 1.08 }}
      whileTap={{ scale: 0.95 }}
    >
      <IconButton
        onClick={onClick}
        sx={{
          width: 60, height: 60, bgcolor: 'primary.main', color: 'white',
          boxShadow: '0 8px 24px rgba(0,0,0,0.25)',
          '&:hover': { bgcolor: 'primary.dark' },
        }}
      >
        {open ? <CloseIcon /> : <ChatIcon />}
      </IconButton>
    </motion.div>
  )
}
```

- [ ] **Step 3: Write `src/widget/MessageBubble.tsx`**

```typescript
import { useState } from 'react'
import { Box, Chip, IconButton, Paper, Stack, Tooltip } from '@mui/material'
import ContentCopyIcon from '@mui/icons-material/ContentCopy'
import ThumbUpIcon from '@mui/icons-material/ThumbUp'
import ThumbDownIcon from '@mui/icons-material/ThumbDown'
import ReplayIcon from '@mui/icons-material/Replay'
import ReactMarkdown from 'react-markdown'
import { apiClient } from '../api/client'
import type { MessageSource } from '../api/types'

interface Props {
  id: number | 'streaming'
  role: 'user' | 'assistant'
  content: string
  sources?: MessageSource[]
  confidence?: number
  feedback?: 'up' | 'down' | null
  onRegenerate?: () => void
}

export default function MessageBubble({ id, role, content, sources, confidence, feedback, onRegenerate }: Props) {
  const [localFeedback, setLocalFeedback] = useState(feedback ?? null)
  const isUser = role === 'user'

  const submitFeedback = async (value: 'up' | 'down') => {
    if (typeof id !== 'number') return
    setLocalFeedback(value)
    await apiClient.post(`/messages/${id}/feedback`, { feedback: value })
  }

  return (
    <Box sx={{ display: 'flex', justifyContent: isUser ? 'flex-end' : 'flex-start', mb: 1.5 }}>
      <Paper
        elevation={0}
        sx={{
          maxWidth: '80%', p: 1.5, borderRadius: 3,
          bgcolor: isUser ? 'primary.main' : 'background.paper',
          color: isUser ? 'primary.contrastText' : 'text.primary',
        }}
      >
        <ReactMarkdown
          components={{
            code: ({ children, className }) => (
              <Box component="code" className={className} sx={{ fontFamily: 'monospace', bgcolor: 'action.hover', px: 0.5, borderRadius: 1 }}>
                {children}
              </Box>
            ),
          }}
        >
          {content}
        </ReactMarkdown>

        {!isUser && sources && sources.length > 0 && (
          <Stack direction="row" spacing={0.5} sx={{ mt: 1, flexWrap: 'wrap' }}>
            {sources.map((s) => (
              <Chip key={s.url} size="small" label={s.title || s.url} component="a" href={s.url} target="_blank" clickable />
            ))}
            {confidence !== undefined && <Chip size="small" variant="outlined" label={`${confidence}% confidence`} />}
          </Stack>
        )}

        {!isUser && (
          <Stack direction="row" spacing={0.5} sx={{ mt: 0.5 }}>
            <Tooltip title="Copy"><IconButton size="small" onClick={() => navigator.clipboard.writeText(content)}><ContentCopyIcon fontSize="inherit" /></IconButton></Tooltip>
            <Tooltip title="Good response"><IconButton size="small" color={localFeedback === 'up' ? 'primary' : 'default'} onClick={() => submitFeedback('up')}><ThumbUpIcon fontSize="inherit" /></IconButton></Tooltip>
            <Tooltip title="Bad response"><IconButton size="small" color={localFeedback === 'down' ? 'error' : 'default'} onClick={() => submitFeedback('down')}><ThumbDownIcon fontSize="inherit" /></IconButton></Tooltip>
            {onRegenerate && <Tooltip title="Regenerate"><IconButton size="small" onClick={onRegenerate}><ReplayIcon fontSize="inherit" /></IconButton></Tooltip>}
          </Stack>
        )}
      </Paper>
    </Box>
  )
}
```

- [ ] **Step 4: Write `src/widget/ChatInput.tsx`**

```typescript
import { useState, type KeyboardEvent } from 'react'
import { Box, Button, Chip, Stack, TextField } from '@mui/material'
import SendIcon from '@mui/icons-material/Send'

const SUGGESTED_QUESTIONS = ['What does this website offer?', 'How can I contact support?', 'What are your pricing plans?']

export default function ChatInput({ onSend, onClear, disabled }: { onSend: (text: string) => void; onClear: () => void; disabled: boolean }) {
  const [value, setValue] = useState('')

  const handleSend = () => {
    if (!value.trim()) return
    onSend(value.trim())
    setValue('')
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <Box sx={{ p: 1.5, borderTop: 1, borderColor: 'divider' }}>
      <Stack direction="row" spacing={1} sx={{ mb: 1, flexWrap: 'wrap' }}>
        {SUGGESTED_QUESTIONS.map((q) => (
          <Chip key={q} label={q} size="small" onClick={() => onSend(q)} clickable />
        ))}
        <Chip label="Clear chat" size="small" variant="outlined" onClick={onClear} />
      </Stack>
      <Stack direction="row" spacing={1}>
        <TextField
          fullWidth multiline maxRows={4} placeholder="Ask a question…" value={value}
          onChange={(e) => setValue(e.target.value)} onKeyDown={handleKeyDown} disabled={disabled} size="small"
        />
        <Button variant="contained" onClick={handleSend} disabled={disabled || !value.trim()}><SendIcon /></Button>
      </Stack>
    </Box>
  )
}
```

- [ ] **Step 5: Write `src/widget/ConversationSidebar.tsx`**

```typescript
import { Box, IconButton, List, ListItemButton, ListItemText, Stack, Tooltip } from '@mui/material'
import AddIcon from '@mui/icons-material/Add'
import DeleteIcon from '@mui/icons-material/Delete'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { apiClient } from '../api/client'
import type { Conversation } from '../api/types'

export default function ConversationSidebar({
  websiteId, sessionId, activeId, onSelect, onNewChat,
}: { websiteId: number; sessionId: string; activeId: number | null; onSelect: (id: number) => void; onNewChat: () => void }) {
  const qc = useQueryClient()
  const { data } = useQuery({
    queryKey: ['conversations', websiteId, sessionId],
    queryFn: async () =>
      (await apiClient.get<Conversation[]>('/conversations', { params: { website_id: websiteId, session_id: sessionId } })).data,
  })

  const handleDelete = async (id: number) => {
    await apiClient.delete(`/conversations/${id}`)
    qc.invalidateQueries({ queryKey: ['conversations', websiteId, sessionId] })
  }

  return (
    <Box sx={{ width: 200, borderRight: 1, borderColor: 'divider', overflowY: 'auto' }}>
      <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ p: 1 }}>
        <span>Chats</span>
        <Tooltip title="New chat"><IconButton size="small" onClick={onNewChat}><AddIcon fontSize="small" /></IconButton></Tooltip>
      </Stack>
      <List dense>
        {data?.map((c) => (
          <ListItemButton key={c.id} selected={c.id === activeId} onClick={() => onSelect(c.id)}>
            <ListItemText primary={c.title} secondary={new Date(c.created_at).toLocaleDateString()} />
            <IconButton size="small" onClick={(e) => { e.stopPropagation(); handleDelete(c.id) }}><DeleteIcon fontSize="inherit" /></IconButton>
          </ListItemButton>
        ))}
      </List>
    </Box>
  )
}
```

- [ ] **Step 6: Write `src/widget/ChatWindow.tsx`**

```typescript
import { useEffect, useRef, useState } from 'react'
import { Avatar, Box, IconButton, Paper, Stack, Typography } from '@mui/material'
import CloseIcon from '@mui/icons-material/Close'
import MenuIcon from '@mui/icons-material/Menu'
import { motion } from 'framer-motion'
import { useChatSession } from './useChatSession'
import MessageBubble from './MessageBubble'
import ChatInput from './ChatInput'
import ConversationSidebar from './ConversationSidebar'
import type { Website } from '../api/types'

export default function ChatWindow({ website, onClose }: { website: Website; onClose: () => void }) {
  const { sessionId, conversationId, messages, isStreaming, sendMessage, clearChat, loadConversation } = useChatSession(website.id)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)
  const lastQuestionRef = useRef<string | null>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSend = (text: string) => {
    lastQuestionRef.current = text
    sendMessage(text)
  }

  return (
    <motion.div
      className="chat-window-container"
      initial={{ opacity: 0, y: 40, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: 40, scale: 0.95 }}
      transition={{ duration: 0.25, ease: 'easeOut' }}
      style={{ position: 'fixed', bottom: 96, right: 24, zIndex: 1300, width: 380, height: 560 }}
    >
      <Paper sx={{
        display: 'flex', height: '100%', borderRadius: 4, overflow: 'hidden',
        backdropFilter: 'blur(16px)', boxShadow: '0 16px 48px rgba(0,0,0,0.3)',
      }}>
        {sidebarOpen && (
          <ConversationSidebar
            websiteId={website.id} sessionId={sessionId} activeId={conversationId}
            onSelect={loadConversation} onNewChat={clearChat}
          />
        )}
        <Box sx={{ display: 'flex', flexDirection: 'column', flex: 1, minWidth: 0 }}>
          <Stack direction="row" alignItems="center" spacing={1} sx={{ p: 1.5, borderBottom: 1, borderColor: 'divider' }}>
            <IconButton size="small" onClick={() => setSidebarOpen((v) => !v)}><MenuIcon fontSize="small" /></IconButton>
            <Avatar src={website.logo_url ?? undefined} sx={{ width: 32, height: 32 }}>{website.name[0]}</Avatar>
            <Box sx={{ flex: 1 }}>
              <Typography variant="subtitle2">{website.name}</Typography>
              <Stack direction="row" alignItems="center" spacing={0.5}>
                <Box sx={{ width: 8, height: 8, borderRadius: '50%', bgcolor: 'success.main' }} />
                <Typography variant="caption" color="text.secondary">AI Assistant · Online</Typography>
              </Stack>
            </Box>
            <IconButton size="small" onClick={onClose}><CloseIcon fontSize="small" /></IconButton>
          </Stack>

          <Box sx={{ flex: 1, overflowY: 'auto', p: 1.5 }}>
            {messages.map((m) => (
              <MessageBubble
                key={m.id}
                id={m.id}
                role={m.role}
                content={m.content || (isStreaming && m.id === 'streaming' ? '…' : '')}
                sources={m.sources}
                confidence={m.confidence}
                feedback={m.feedback}
                onRegenerate={
                  m.role === 'assistant' && lastQuestionRef.current
                    ? () => sendMessage(lastQuestionRef.current as string)
                    : undefined
                }
              />
            ))}
            <div ref={bottomRef} />
          </Box>

          <ChatInput onSend={handleSend} onClear={clearChat} disabled={isStreaming} />
        </Box>
      </Paper>
    </motion.div>
  )
}
```

- [ ] **Step 7: Write `src/widget/ChatWidget.tsx`**

```typescript
import { useState } from 'react'
import { AnimatePresence } from 'framer-motion'
import FloatingButton from './FloatingButton'
import ChatWindow from './ChatWindow'
import type { Website } from '../api/types'

export default function ChatWidget({ website }: { website: Website }) {
  const [open, setOpen] = useState(false)

  return (
    <>
      <AnimatePresence>
        {open && <ChatWindow key="chat-window" website={website} onClose={() => setOpen(false)} />}
      </AnimatePresence>
      <FloatingButton open={open} onClick={() => setOpen((v) => !v)} />
    </>
  )
}
```

- [ ] **Step 8: Append mobile override to `src/index.css`**

```css
@media (max-width: 480px) {
  .chat-window-container {
    top: 0 !important;
    left: 0 !important;
    right: 0 !important;
    bottom: 0 !important;
    width: 100vw !important;
    height: 100vh !important;
  }
  .chat-window-container .MuiPaper-root {
    border-radius: 0 !important;
  }
}
```

- [ ] **Step 9: Build**

```bash
cd Ai/frontend && npm run build
```
Expected: succeeds with no TypeScript errors. (The widget isn't mounted anywhere yet — that's Task 25 — so this only confirms it compiles.)

- [ ] **Step 10: Commit**

```bash
git add Ai/frontend/src/widget Ai/frontend/src/index.css
git commit -m "feat: chat widget with SSE streaming, markdown, sidebar, and message actions"
```

---

### Task 25: Wire widget into demo page, root run scripts, README

**Files:**
- Modify: `Ai/frontend/src/pages/DemoPage.tsx`
- Create: `Ai/backend/run.sh`
- Create: `Ai/frontend/run.sh`
- Create: `Ai/setup.sh`
- Create: `Ai/README.md`

- [ ] **Step 1: Rewrite `src/pages/DemoPage.tsx`**

```typescript
import { useState } from 'react'
import { Box, MenuItem, Select, Typography } from '@mui/material'
import { useWebsites } from '../api/hooks'
import ChatWidget from '../widget/ChatWidget'

export default function DemoPage() {
  const { data: websites } = useWebsites()
  const [selectedId, setSelectedId] = useState<number | ''>('')

  const selected = websites?.find((w) => w.id === selectedId) ?? websites?.[0]

  return (
    <Box sx={{ p: 4 }}>
      <Typography variant="h4">Demo site</Typography>
      <Typography color="text.secondary" sx={{ mb: 2 }}>
        This page simulates a customer's website with the chat widget embedded.
      </Typography>
      {websites && websites.length > 0 && (
        <Select value={selected?.id ?? ''} onChange={(e) => setSelectedId(e.target.value as number)} size="small">
          {websites.map((w) => (
            <MenuItem key={w.id} value={w.id}>{w.name}</MenuItem>
          ))}
        </Select>
      )}
      {websites && websites.length === 0 && (
        <Typography color="text.secondary">No websites yet — add one in the admin dashboard at /admin.</Typography>
      )}
      {selected && <ChatWidget website={selected} />}
    </Box>
  )
}
```

- [ ] **Step 2: Write `Ai/backend/run.sh`**

```bash
#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
source venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

- [ ] **Step 3: Write `Ai/frontend/run.sh`**

```bash
#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
npm run dev
```

- [ ] **Step 4: Write `Ai/setup.sh`**

```bash
#!/usr/bin/env bash
set -e
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "== Setting up backend =="
cd "$ROOT_DIR/backend"
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
playwright install chromium
[ -f .env ] || cp .env.example .env
deactivate

echo "== Setting up frontend =="
cd "$ROOT_DIR/frontend"
npm install
[ -f .env ] || cp .env.example .env

echo ""
echo "Setup complete."
echo "1. Edit Ai/backend/.env with your GROQ_API_KEY, JWT_SECRET, ADMIN_EMAIL, ADMIN_PASSWORD."
echo "2. Run ./backend/run.sh in one terminal."
echo "3. Run ./frontend/run.sh in another terminal."
echo "4. Open http://localhost:5173/admin to log in, and http://localhost:5173/ for the demo widget."
```

```bash
chmod +x Ai/setup.sh Ai/backend/run.sh Ai/frontend/run.sh
```

- [ ] **Step 5: Write `Ai/README.md`**

```markdown
# Multi-Website AI Chatbot Platform

Admins register website URLs; the platform crawls each site, extracts and chunks its content,
embeds it into a per-website ChromaDB collection, and serves a floating chat widget that answers
questions using only that website's indexed content — with sources, a confidence score, and a
fixed "I couldn't find that information on this website." fallback when nothing relevant is found.
A daily scheduler keeps every site's index in sync.

## Stack

- **Backend:** Python, FastAPI, SQLAlchemy + SQLite, ChromaDB, LangChain, Groq (`llama-3.3-70b-versatile`),
  HuggingFace `sentence-transformers/all-MiniLM-L6-v2`, Playwright, Trafilatura, BeautifulSoup4, APScheduler.
- **Frontend:** React, Vite, TypeScript, MUI, React Query, Axios, Framer Motion, React Markdown.

No Docker — everything runs from a local Python venv and Node/npm.

## Project layout

```
Ai/
├── backend/       FastAPI app (crawler, extractor, chunker, embeddings, vectordb, llm, scheduler, services, api)
├── frontend/      React admin dashboard + chat widget demo page
├── setup.sh       one-time setup for both
└── README.md
```

## Setup

```bash
cd Ai
./setup.sh
```

Then edit `backend/.env` (copied from `.env.example`) and set at minimum:

```
GROQ_API_KEY=...           # from console.groq.com
JWT_SECRET=...             # any long random string
ADMIN_EMAIL=admin@example.com
ADMIN_PASSWORD=...
```

## Running

```bash
./backend/run.sh     # http://localhost:8000  (terminal 1)
./frontend/run.sh    # http://localhost:5173   (terminal 2)
```

- Admin dashboard: `http://localhost:5173/admin` (log in with `ADMIN_EMAIL`/`ADMIN_PASSWORD`)
- Demo widget page: `http://localhost:5173/`

## Using it

1. In the admin dashboard, click **Add Website**, enter a URL and display name.
2. Click the crawl icon on that row — it discovers pages via `sitemap.xml` (or a recursive
   same-domain crawl if there's no sitemap), extracts clean text with Playwright + Trafilatura
   (falling back to BeautifulSoup), chunks it, embeds it, and stores it in that site's ChromaDB
   collection (`website_{id}`).
3. Click **Details** to see indexed pages and crawl job history.
4. Open `http://localhost:5173/`, pick the website from the dropdown, and chat with it via the
   floating button — answers are grounded only in that site's indexed content.
5. **Sync** re-crawls and indexes only new/changed pages, deleting vectors for pages no longer
   found. **Reindex** force re-embeds every currently-discovered page. A daily background job
   (default 03:00, `DAILY_SYNC_HOUR` in `.env`) runs the same sync automatically for every website.

## Configuration reference (`backend/.env`)

| Variable | Default | Meaning |
|---|---|---|
| `GROQ_API_KEY` | — | Groq API key (required) |
| `JWT_SECRET` | — | Signing secret for admin JWTs (required) |
| `ADMIN_EMAIL` / `ADMIN_PASSWORD` | — | Seeded admin login (required) |
| `CORS_ORIGINS` | `http://localhost:5173` | Comma-separated allowed frontend origins |
| `MAX_PAGES` | 500 | Max pages crawled per website |
| `MAX_CRAWL_DEPTH` | 5 | Max recursive-crawl link depth (ignored when a sitemap is used) |
| `CRAWLER_CONCURRENCY` | 2 | Concurrent page fetches/renders |
| `CRAWLER_DELAY_SECONDS` | 1 | Delay between requests to the same domain |
| `SIMILARITY_THRESHOLD` | 0.3 | Minimum cosine similarity for a chunk to be used as context |
| `TOP_K_CHUNKS` | 5 | Chunks retrieved per question |
| `DAILY_SYNC_HOUR` | 3 | Hour (0-23, server local time) the daily sync job runs |

## API summary

`POST /auth/login` · `POST /websites` · `GET /websites` · `PATCH /websites/{id}` · `DELETE /website/{id}` ·
`POST /crawl/{id}` · `POST /sync/{id}` · `POST /reindex/{id}` · `GET /websites/{id}/crawl-jobs` ·
`GET /pages/{website_id}` · `POST /chat` (SSE) · `GET /conversations` · `GET /conversations/{id}` ·
`DELETE /conversations/{id}` · `POST /messages/{id}/feedback` · `GET /analytics/{website_id}`

All admin-mutating routes require `Authorization: Bearer <token>` from `/auth/login`.

## Tests

```bash
cd backend && source venv/bin/activate && pytest -v
```

Pure logic (URL filtering, hashing, chunking, JWT, similarity/confidence, HTML cleaning) and API
contracts (with crawling/LLM calls mocked) are covered by pytest. Genuinely I/O-bound integration
points — live Playwright rendering and real Groq streaming — are verified with the manual smoke-test
commands documented in the implementation plan (`docs/superpowers/plans/`), since they need a real
browser / API key rather than a mock.

The frontend has no test runner in this stack; `npm run build` type-checks and bundles as the
automated gate for each change, verified manually in the browser alongside it.
```

- [ ] **Step 6: Build and run full end-to-end manual check**

```bash
cd Ai && ./setup.sh
# fill in backend/.env
./backend/run.sh &
./frontend/run.sh &
```
Visit `/admin`, add a real website, crawl it, then visit `/` and chat with it. Confirm: streaming text appears token-by-token, sources/confidence show on the answer, thumbs up/down and copy work, regenerate re-asks the last question, clear chat empties the window, sidebar shows the conversation and can start a new one / delete it, and the floating button animates open/closed on both a desktop-sized and a narrow (mobile-width) browser window.

- [ ] **Step 7: Commit**

```bash
git add Ai/frontend/src/pages/DemoPage.tsx Ai/backend/run.sh Ai/frontend/run.sh Ai/setup.sh Ai/README.md
git commit -m "feat: wire chat widget into demo page, add run scripts and README"
```

---
