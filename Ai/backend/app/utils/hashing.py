import hashlib


def hash_content(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
