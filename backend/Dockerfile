# ============================================================================
# Stage 1: Base Image (CUDA 13.0.2 + Python 3.12 + Devel)
# ============================================================================
FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04 AS base

# 保持默认非交互设置
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    TORCH_CUDA_ARCH_LIST="9.0;10.0;12.0"

# ❌ 【删除换源命令】：不再使用阿里云/腾讯云，直接用官方源，配合您的 Clash 网络
# RUN sed -i ... 

# 1. 安装系统依赖
# 添加 --fix-missing 以防止网络波动导致的构建失败
RUN apt-get update && apt-get install -y --fix-missing --no-install-recommends \
    python3-pip python3-venv python3-dev \
    git wget curl vim build-essential \
    libgomp1 libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 \
    libgstreamer1.0-0 ffmpeg \
    antiword pandoc \
    libreoffice libreoffice-writer libreoffice-calc libreoffice-impress \
    fonts-noto-cjk fonts-noto-cjk-extra fonts-wqy-zenhei fonts-wqy-microhei \
    fontconfig \
    && fc-cache -fv \
    && rm -rf /var/lib/apt/lists/*

# 2. Python 软链接
RUN ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3

# ❌ 【删除 Pip 升级】：避免系统权限冲突报错
# RUN python -m pip install --upgrade pip ...

# ============================================================================
# Stage 2: 依赖安装
# ============================================================================
FROM base AS dependencies

WORKDIR /tmp
COPY backend/requirements.txt /tmp/requirements.txt

# Step 1: 安装 vLLM
# 使用 --break-system-packages 允许在系统环境安装
RUN --mount=type=cache,target=/root/.cache/pip \
    echo "Installing vLLM..." && \
    pip install "vllm>=0.15.1" \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu130 \
    --break-system-packages

# Step 2: 安装 PaddlePaddle
RUN --mount=type=cache,target=/root/.cache/pip \
    echo "Installing PaddlePaddle..." && \
    (pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu130/ --break-system-packages || \
     pip install --pre paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/cu130/ --break-system-packages)

# Step 3: 安装其他依赖
# ❌ 去掉国内 pip 源，直接走默认源（Clash 加速）
RUN --mount=type=cache,target=/root/.cache/pip \
    echo "Installing dependencies..." && \
    cat /tmp/requirements.txt | grep -vE "torch|paddle|vllm" > /tmp/cleaned_requirements.txt && \
    pip install -r /tmp/cleaned_requirements.txt \
    --break-system-packages

# Step 4: 安装 MinerU
RUN --mount=type=cache,target=/root/.cache/pip \
    echo "Installing MinerU..." && \
    pip install "mineru[core]>=2.7.6" --no-deps \
    --break-system-packages

# Step 5: 补齐工具
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install paddleocr[doc-parser] \
    transformers==4.57.6 tokenizers>=0.21.0 \
    fastapi uvicorn litserve aiohttp \
    'albumentations>=1.3.1,<1.4.0' 'albucore>=0.0.13,<0.0.17' \
    --break-system-packages

# Step 6: 验证
RUN python -c "import sys; print('Python:', sys.version)" && \
    python -c "import torch; print('✅ PyTorch:', torch.__version__)"

# ============================================================================
# Stage 3: Application
# ============================================================================
FROM dependencies AS application

WORKDIR /app
COPY backend/ /app/backend/
COPY pyproject.toml /app/

COPY scripts/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
COPY scripts/init-models.sh /usr/local/bin/init-models.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh /usr/local/bin/init-models.sh

RUN mkdir -p /app/models /app/data /app/uploads /app/output /app/logs
WORKDIR /app/backend
EXPOSE 8000 8001 8002

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["api", "python", "api_server.py"]
