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
    # 显式指定 5090 架构
    TORCH_CUDA_ARCH_LIST="9.0;10.0;12.0"

# 1. 安装系统依赖 (包含原版所有工具 + 编译工具)
RUN apt-get update && apt-get install -y --no-install-recommends \
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

# 3. 升级 pip
RUN python -m pip install --upgrade pip

# ============================================================================
# Stage 2: 依赖安装 (RTX 5090 兼容策略)
# ============================================================================
FROM base AS dependencies

WORKDIR /tmp
COPY backend/requirements.txt /tmp/requirements.txt

# Step 1: 安装 vLLM (>=0.15.1, 支持 RTX 5090)
# 它会自动处理 PyTorch 和 Triton 的依赖
RUN --mount=type=cache,target=/root/.cache/pip \
    echo "Installing vLLM >= 0.15.1..." && \
    pip install "vllm>=0.15.1" \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu130 \
    --break-system-packages

# Step 2: 安装 PaddlePaddle (优先 Stable)
RUN --mount=type=cache,target=/root/.cache/pip \
    echo "Installing PaddlePaddle..." && \
    (pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu130/ --break-system-packages || \
     pip install --pre paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/cu130/ --break-system-packages)

# Step 3: 安装 MinerU 依赖 (剔除冲突项)
RUN --mount=type=cache,target=/root/.cache/pip \
    echo "Installing base dependencies..." && \
    cat /tmp/requirements.txt | grep -vE "torch|paddle|vllm" > /tmp/cleaned_requirements.txt && \
    pip install -r /tmp/cleaned_requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --break-system-packages

# Step 4: 强制安装 MinerU Core (>=2.7.6)
# 这一步将 MinerU 安装到 vLLM 带来的 PyTorch 环境中
RUN --mount=type=cache,target=/root/.cache/pip \
    echo "Installing MinerU Core (Force)..." && \
    pip install "mineru[core]>=2.7.6" --no-deps \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --break-system-packages

# Step 5: 补齐其他库 & 锁定 Transformers
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install paddleocr[doc-parser] \
    transformers==4.57.6 tokenizers>=0.21.0 \
    fastapi uvicorn litserve aiohttp \
    'albumentations>=1.3.1,<1.4.0' 'albucore>=0.0.13,<0.0.17' \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --break-system-packages

# Step 6: 验证安装 (原版功能回归)
# 这一步在构建时检查环境是否健康
RUN python -c "import sys; print('Python:', sys.version)" && \
    python -c "import torch; print('✅ PyTorch:', torch.__version__)" && \
    python -c "import vllm; print('✅ vLLM:', vllm.__version__)" && \
    python -c "import transformers; print('✅ Transformers:', transformers.__version__)" && \
    python -c "import funasr; print('✅ FunASR:', funasr.__version__)" && \
    echo "⚠️  Paddle CUDA 验证将在运行时进行"

# ============================================================================
# Stage 3: Application
# ============================================================================
FROM dependencies AS application

WORKDIR /app
COPY backend/ /app/backend/
COPY pyproject.toml /app/

# 启动脚本
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
