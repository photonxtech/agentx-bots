# Use Python 3.11 slim image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and force unbuffered stdout/stderr logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# Install essential system dependencies (for C-extensions, PostgreSQL drivers, and curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt to leverage Docker layer caching
COPY requirements.txt .

# Upgrade pip/setuptools and install python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --no-build-isolation -r requirements.txt

# Copy backend and frontend source directories
COPY backend ./backend
COPY frontend ./frontend

# Create upload directory and set environment default
RUN mkdir -p /app/uploads
ENV UPLOAD_DIR=/app/uploads

# Expose port 8000 for Easypanel
EXPOSE 8000

# Healthcheck for Easypanel/Traefik
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

# Launch Uvicorn server
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
