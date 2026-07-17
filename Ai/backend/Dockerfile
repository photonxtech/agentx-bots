FROM python:3.12.13-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8001

# Set the working directory
WORKDIR /app

# Install basic system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers and their OS dependencies
# Using chromium only by default to keep the image size smaller,
# but installing all dependencies it requires.
RUN playwright install --with-deps chromium

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 8001

# Run the application
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
