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
