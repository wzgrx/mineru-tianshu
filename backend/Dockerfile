# ============================================================================
# Stage 1: Base Image (CUDA 13.0.2 + Python 3.12 + Devel)
# ============================================================================
FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04 AS base

# ✅ [关键] 替换为阿里云镜像源 (防止 apt-get 500/timeout 错误)
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list.d/ubuntu.sources && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list.d/ubuntu.sources

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    # RTX 5090 (Blackwell) 架构支持
    TORCH_CUDA_ARCH_LIST="9.0;10.0;12.0"

# ✅ [关键] 配置 APT 网络防御机制
RUN echo 'Acquire::http::Pipeline-Depth "0";' > /etc/apt/apt.conf.d/99custom && \
    echo 'Acquire::http::No-Cache "true";' >> /etc/apt/apt.conf.d/99custom && \
    echo 'Acquire::BrokenProxy "true";' >> /etc/apt/apt.conf.d/99custom && \
    echo 'Acquire::Retries "5";' >> /etc/apt/apt.conf.d/99custom && \
    echo 'Acquire::http::Timeout "120";' >> /etc/apt/apt.conf.d/99custom

# 1. 安装全量系统依赖 (增加 || true 容错)
RUN apt-get update || true && apt-get install -y --fix-missing --no-install-recommends \
    python3-pip python3-venv python3-dev \
    git wget curl vim build-essential \
    libgomp1 libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 \
    libgstreamer1.0-0 ffmpeg \
    libmagic1 \
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

# Step 1: 安装 vLLM (使用用户指定版本 0.15.1+)
# ✅ 增加清华源 fallback
RUN --mount=type=cache,target=/root/.cache/pip \
    echo "Installing vLLM..." && \
    pip install "vllm>=0.15.1" \
    --retries 10 --timeout 120 \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --break-system-packages

# Step 2: 安装 PaddlePaddle (使用用户指定版本 3.3.0)
RUN --mount=type=cache,target=/root/.cache/pip \
    echo "Installing PaddlePaddle..." && \
    pip install paddlepaddle-gpu==3.3.0 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cu130/ \
    --break-system-packages

# Step 3: 安装常规依赖
# ✅ [修复] grep 仅过滤 paddlepaddle 框架，保留 paddleocr
# 原来的 "grep -vE 'torch|paddle|vllm'" 会把 paddleocr 也删掉，导致后续 import 失败
RUN --mount=type=cache,target=/root/.cache/pip \
    echo "Installing dependencies..." && \
    cat /tmp/requirements.txt | grep -vE "torch|paddlepaddle|vllm" > /tmp/cleaned_requirements.txt && \
    pip install -r /tmp/cleaned_requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --retries 10 --timeout 120 \
    --break-system-packages

# Step 4: 安装 MinerU Core (绕过依赖检查)
# ⚠️ 核心操作：使用 --no-deps 防止 pip 读取 pyproject.toml 并自动降级我们的库
RUN --mount=type=cache,target=/root/.cache/pip \
    echo "Installing MinerU Core..." && \
    pip install "mineru[core]>=2.7.6" --no-deps \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --retries 10 --timeout 120 \
    --break-system-packages

# Step 5: 强制“嫁接”正确的依赖版本 (Conflict-Free Fix)
# ✅ [关键修复] 使用 doclayout-yolo 0.0.4 配合 albumentations 2.x
# ✅ [新增] 卸载可能被自动安装的 GUI 版 opencv-python，只保留 headless 版
RUN --mount=type=cache,target=/root/.cache/pip \
    echo "Patching dependencies for MinerU/YOLO compat..." && \
    pip install doclayout-yolo==0.0.4 --no-deps --break-system-packages \
    -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install "albumentations>=2.0.8" "albucore>=0.0.20" --break-system-packages \
    -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip uninstall -y opencv-python || true

# Step 6: 验证环境
RUN python -c "import sys; print('Python:', sys.version)" && \
    python -c "import torch; print('✅ PyTorch:', torch.__version__)" && \
    python -c "import paddleocr; print('✅ PaddleOCR:', paddleocr.__version__)" && \
    python -c "import doclayout_yolo; print('✅ DocLayout-YOLO:', doclayout_yolo.__version__)"

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
