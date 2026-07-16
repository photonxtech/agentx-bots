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
