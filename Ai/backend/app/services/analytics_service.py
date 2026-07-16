from sqlalchemy import case, func
from sqlalchemy.orm import Session

from app.database.models import Conversation, Feedback, Message, MessageRole
from app.llm.retrieval import NO_ANSWER_MESSAGE


def compute_analytics(db: Session, website_id: int) -> dict:
    # Aggregate counts/sums/averages in SQL rather than pulling every message row into
    # Python to fold over — this endpoint's cost would otherwise grow unboundedly with a
    # site's chat history instead of staying a handful of constant-size queries.
    total_conversations = db.query(Conversation).filter(Conversation.website_id == website_id).count()

    total_messages = db.query(Message).join(Conversation).filter(Conversation.website_id == website_id).count()

    assistant_stats = (
        db.query(
            func.count(Message.id),
            func.sum(case((Message.feedback == Feedback.up, 1), else_=0)),
            func.sum(case((Message.feedback == Feedback.down, 1), else_=0)),
            func.avg(Message.confidence),
            func.sum(case((Message.content == NO_ANSWER_MESSAGE, 1), else_=0)),
        )
        .join(Conversation)
        .filter(Conversation.website_id == website_id, Message.role == MessageRole.assistant)
        .one()
    )
    assistant_count, thumbs_up, thumbs_down, avg_confidence, fallback_count = assistant_stats

    average_confidence = round(avg_confidence, 1) if avg_confidence is not None else 0.0
    fallback_rate = round((fallback_count / assistant_count) * 100, 1) if assistant_count else 0.0

    recent_questions = (
        db.query(Message.content)
        .join(Conversation)
        .filter(Conversation.website_id == website_id, Message.role == MessageRole.user)
        .order_by(Message.created_at.desc())
        .limit(10)
        .all()
    )

    return {
        "total_conversations": total_conversations,
        "total_messages": total_messages,
        "thumbs_up": int(thumbs_up or 0),
        "thumbs_down": int(thumbs_down or 0),
        "average_confidence": average_confidence,
        "fallback_rate": fallback_rate,
        "recent_questions": [content for (content,) in recent_questions],
    }
