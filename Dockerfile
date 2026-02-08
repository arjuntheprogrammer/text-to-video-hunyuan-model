FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/models \
    HF_HUB_CACHE=/app/models \
    TRANSFORMERS_CACHE=/app/models \
    TORCH_HOME=/app/models \
    HF_TOKEN="" \
    HUGGING_FACE_HUB_TOKEN=""

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install -r /app/requirements.txt

COPY . /app

RUN mkdir -p /app/models /app/outputs

EXPOSE 8000 7860

ENTRYPOINT ["python3", "run.py"]
