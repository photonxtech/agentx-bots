import os
import json
import uuid
from datetime import datetime, date
import config

HISTORY_FILE = os.path.join(config.INDEX_DIR, "chat_history.json")

def load_chats() -> dict:
    if not os.path.exists(HISTORY_FILE):
        return {}
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_chats(chats: dict) -> None:
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(chats, f, indent=2, ensure_ascii=False)

def create_chat(chats: dict, title: str = "New Chat") -> str:
    chat_id = str(uuid.uuid4())
    now_str = datetime.now().isoformat()
    chats[chat_id] = {
        "id": chat_id,
        "title": title,
        "created_at": now_str,
        "updated_at": now_str,
        "messages": []
    }
    save_chats(chats)
    return chat_id

def delete_chat(chats: dict, chat_id: str) -> None:
    if chat_id in chats:
        del chats[chat_id]
        save_chats(chats)

def update_chat_messages(chats: dict, chat_id: str, messages: list) -> None:
    if chat_id in chats:
        chats[chat_id]["messages"] = messages
        chats[chat_id]["updated_at"] = datetime.now().isoformat()
        save_chats(chats)

def group_chats_by_recency(chats: dict) -> dict:
    groups = {
        "Today": [],
        "Yesterday": [],
        "Previous 7 Days": [],
        "Previous 30 Days": [],
        "Older": [],
    }
    today = date.today()
    # Sort chats so newest active chats appear first
    sorted_chats = sorted(chats.values(), key=lambda c: c["updated_at"], reverse=True)

    for chat in sorted_chats:
        try:
            updated = datetime.fromisoformat(chat["updated_at"]).date()
            delta = (today - updated).days
            if delta == 0:
                groups["Today"].append(chat)
            elif delta == 1:
                groups["Yesterday"].append(chat)
            elif delta <= 7:
                groups["Previous 7 Days"].append(chat)
            elif delta <= 30:
                groups["Previous 30 Days"].append(chat)
            else:
                groups["Older"].append(chat)
        except Exception:
            # Fallback if date is not parsed
            groups["Today"].append(chat)

    # Filter out empty buckets
    return {k: v for k, v in groups.items() if v}
