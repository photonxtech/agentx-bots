FROM python:3.11-slim

# System deps for EasyOCR, PyMuPDF, psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full project so config.BASE_DIR and FRONTEND_DIR resolve correctly
COPY . .

# Data persistence: mount a volume at /data in production
ENV MULTIRAG_DATA_DIR=/data

EXPOSE 8000

# Run from /app/backend so `api`, `config`, `rag`, etc. resolve as top-level imports
WORKDIR /app/backend
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
