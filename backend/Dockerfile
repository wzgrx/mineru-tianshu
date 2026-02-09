# ============================================================================
# Stage 1: Base Image (CUDA 13.0.2 + Python 3.12 + Devel)
# ============================================================================
FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    TORCH_CUDA_ARCH_LIST="9.0;10.0;12.0"

# ✅ [关键] 配置 APT 网络防御机制
# 解决 Clash 透明代理下 apt-get update 报 502/Hash Sum Mismatch 的问题
RUN echo 'Acquire::http::Pipeline-Depth "0";' > /etc/apt/apt.conf.d/99custom && \
    echo 'Acquire::http::No-Cache "true";' >> /etc/apt/apt.conf.d/99custom && \
    echo 'Acquire::BrokenProxy "true";' >> /etc/apt/apt.conf.d/99custom && \
    echo 'Acquire::Retries "5";' >> /etc/apt/apt.conf.d/99custom && \
    echo 'Acquire::http::Timeout "120";' >> /etc/apt/apt.conf.d/99custom

# 1. 安装全量系统依赖
# 保留了 LibreOffice 等大包，利用上面的网络配置确保下载成功
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

# ============================================================================
# Stage 2: 依赖安装
# ============================================================================
FROM base AS dependencies

WORKDIR /tmp
COPY backend/requirements.txt /tmp/requirements.txt

# Step 1: 安装 vLLM (增加重试参数)
RUN --mount=type=cache,target=/root/.cache/pip \
    echo "Installing vLLM..." && \
    pip install "vllm>=0.15.1" \
    --retries 10 --timeout 120 \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu130 \
    --break-system-packages

# Step 2: 安装 PaddlePaddle
RUN --mount=type=cache,target=/root/.cache/pip \
    echo "Installing PaddlePaddle..." && \
    (pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu130/ --break-system-packages || \
     pip install --pre paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/cu130/ --break-system-packages)

# Step 3: 安装常规依赖
# ⚠️ 注意：确保 requirements.txt 中已经删除了 doclayout-yolo
RUN --mount=type=cache,target=/root/.cache/pip \
    echo "Installing dependencies..." && \
    cat /tmp/requirements.txt | grep -vE "torch|paddle|vllm" > /tmp/cleaned_requirements.txt && \
    pip install -r /tmp/cleaned_requirements.txt \
    --retries 10 --timeout 120 \
    --break-system-packages

# Step 4: 安装 MinerU Core (绕过依赖检查)
# ⚠️ 核心操作：使用 --no-deps 防止 pip 读取 pyproject.toml 并报错
RUN --mount=type=cache,target=/root/.cache/pip \
    echo "Installing MinerU Core..." && \
    pip install "mineru[core]>=2.7.6" --no-deps \
    --retries 10 --timeout 120 \
    --break-system-packages

# Step 5: 强制“嫁接”正确的依赖版本
# ⚠️ 核心操作：手动安装兼容的 doclayout 版本和 albumentations
RUN --mount=type=cache,target=/root/.cache/pip \
    echo "Patching dependencies..." && \
    pip install doclayout-yolo==0.0.2b1 --no-deps --break-system-packages && \
    pip install "albumentations>=1.3.1,<1.4.0" "albucore>=0.0.13,<0.0.17" --break-system-packages

# Step 6: 补齐工具
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install paddleocr[doc-parser] \
    transformers==4.57.6 tokenizers>=0.21.0 \
    fastapi uvicorn litserve aiohttp \
    --break-system-packages

# Step 7: 验证环境
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
