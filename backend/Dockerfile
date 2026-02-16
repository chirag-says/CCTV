# ============================================================================
# AI CCTV Surveillance System — Dockerfile
# Multi-stage build for production deployment
# ============================================================================

FROM python:3.11-slim AS base

# Install system dependencies for OpenCV and dlib (face_recognition)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopencv-dev \
    libboost-all-dev \
    libx11-dev \
    libatlas-base-dev \
    libgtk-3-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Production Stage ─────────────────────────────────────────
FROM base AS production

WORKDIR /app

# Copy application code
COPY . .

# Create snapshot directory
RUN mkdir -p ./snapshots

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8000/health'); r.raise_for_status()"

# Run the application
CMD ["python", "run.py"]
