from langchain_text_splitters import RecursiveCharacterTextSplitter

_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


def chunk_text(text: str) -> list[str]:
    if not text.strip():
        return []
    return _splitter.split_text(text)
