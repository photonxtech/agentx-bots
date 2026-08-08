FROM python:3.11-slim

# System deps needed by chromadb / tokenizers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir pydantic-settings -r requirements.txt

COPY . .

# Directories that the app writes to at runtime
RUN mkdir -p data chroma_db

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
