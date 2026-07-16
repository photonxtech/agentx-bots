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
        stream_chat_response(db, payload.website_id, conversation, payload.question, regenerate=payload.regenerate),
        media_type="text/event-stream",
    )
