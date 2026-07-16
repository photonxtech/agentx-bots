from datetime import datetime
from typing import Literal

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


class ChatRequest(BaseModel):
    website_id: int
    question: str
    session_id: str
    conversation_id: int | None = None
    regenerate: bool = False


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


class AnalyticsRead(BaseModel):
    total_conversations: int
    total_messages: int
    thumbs_up: int
    thumbs_down: int
    average_confidence: float
    fallback_rate: float
    recent_questions: list[str]


class EvalCase(BaseModel):
    question: str
    expected_keywords: list[str] = []
    expect_no_answer: bool = False


class EvalRequest(BaseModel):
    cases: list[EvalCase]


class EvalRunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    website_id: int
    total_cases: int
    passed_cases: int
    hallucination_count: int
    avg_confidence: float
    avg_latency_ms: float
    details: list[dict] | None
    started_at: datetime
    finished_at: datetime | None
