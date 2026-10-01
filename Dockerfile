# ============================================================
# Vision Attribute Pipeline — Dockerfile
# Base: CUDA 12.1 + Python 3.10 (matches T4 / A10G on AWS)
# ============================================================

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV TORCH_HOME=/app/models/.torch_cache

# ── System dependencies ───────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-distutils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    curl \
    git \
    wget \
    ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Make python3.10 default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# ── Python dependencies ───────────────────────────────────────────────────
WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# ── Copy application code ─────────────────────────────────────────────────
COPY . .

# Create necessary directories
RUN mkdir -p \
    /tmp/thumbnails \
    /tmp/vap_logs \
    /app/models \
    /app/test_data \
    /app/vendor

# ── Health check script ───────────────────────────────────────────────────
RUN echo '#!/bin/sh\npython -c "import torch; import redis; import pymilvus; print(\"OK\")"' \
    > /usr/local/bin/healthcheck.sh && chmod +x /usr/local/bin/healthcheck.sh

# Default entrypoint — overridden by docker-compose per service
ENTRYPOINT ["python"]
CMD ["-m", "api.api_server"]
