# ── Build stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Prevent .pyc files and enable unbuffered stdout/stderr for clean Render logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install OS-level dependencies required by PyMuPDF (fitz) and sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Create the documents directory in case it's empty (gitignored PDFs)
RUN mkdir -p documents

# Render injects PORT; default to 8000 for local docker runs
ENV PORT=8000

# Gunicorn with Uvicorn workers — production-grade ASGI server
# -w 1 because model loading is heavy; increase on paid plans
CMD gunicorn backend.app:app \
    --worker-class uvicorn.workers.UvicornWorker \
    --workers 1 \
    --bind 0.0.0.0:$PORT \
    --timeout 120 \
    --access-logfile - \
    --error-logfile -
