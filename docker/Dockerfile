# 推理服务镜像（Ubuntu + CUDA）。训练请直接在宿主机跑，不用这个镜像。
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        libgl1 libglib2.0-0 \
        fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*
# fonts-noto-cjk 是必须的：标注图和 PDF 报告里的中文全靠它

WORKDIR /app
COPY requirements.txt .

# Blackwell(sm_120) 必须 cu128 及以上；3090(sm_86) 用同一套轮子即可
RUN python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 && \
    python3.11 -m pip install -r requirements.txt

COPY app/ ./app/
COPY configs/ ./configs/
COPY data/knowledge/ ./data/knowledge/
COPY models/ ./models/

ENV MOLD_STORAGE_DIR=/data/storage
VOLUME ["/data/storage"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD python3.11 -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/api/health')" || exit 1

CMD ["python3.11", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
