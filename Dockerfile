# ================================
# Facial Paralysis Simulator
# Production-ready Dockerfile
# ================================

# ---- Base image (Python 3.10) ----
FROM python:3.10-slim

# ---- Metadata (optional) ----
LABEL maintainer="Anurag Jambhulkar <your.email@example.com>"
LABEL description="Facial Paralysis Simulator built with Gradio, MediaPipe, and OpenCV."

# ---- Set working directory ----
WORKDIR /app

# ---- Environment configuration ----
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1
ENV PORT=7860

# ---- Copy only requirements first (for build cache) ----
COPY requirements.txt /app/requirements.txt

# ---- Install system dependencies (fixes OpenCV & MediaPipe issues) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ---- Install Python dependencies ----
RUN pip install --upgrade pip wheel setuptools \
    && pip install --no-cache-dir -r /app/requirements.txt

# ---- Copy project files ----
COPY . /app

# ---- Create non-root user (optional but recommended) ----
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# ---- Expose the Gradio port ----
EXPOSE 7860

# ---- Start command ----
# Gradio will read the PORT environment variable in your app.py
CMD ["python", "app.py"]
