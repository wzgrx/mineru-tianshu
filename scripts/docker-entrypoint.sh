#!/bin/bash
# Tianshu - Docker Entrypoint Script
# Container startup script for initialization and health checks

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Log functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ============================================================================
# Environment check
# ============================================================================
check_environment() {
    local service_type=$1

    log_info "Checking environment configuration..."

    # Check Python version
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    log_info "Python version: $PYTHON_VERSION"

    # Check CUDA
    if command -v nvidia-smi &> /dev/null; then
        log_success "NVIDIA GPU detected"
        nvidia-smi --query-gpu=gpu_name,driver_version,memory.total --format=csv,noheader
    else
        log_warning "NVIDIA GPU or driver not detected"
    fi

    # Check necessary environment variables (only API Server needs JWT)
    if [ "$service_type" != "worker" ] && [ "$service_type" != "mcp" ]; then
        if [ -z "$JWT_SECRET_KEY" ]; then
            log_error "JWT_SECRET_KEY is not set! Please configure in .env"
            exit 1
        fi

        if [ "$JWT_SECRET_KEY" = "CHANGE_THIS_TO_A_SECURE_RANDOM_STRING_IN_PRODUCTION" ]; then
            log_warning "JWT_SECRET_KEY is using default value, must be changed for production!"
        fi
    fi

    # ✅ [新增] vLLM 显存检查
    if [ "$service_type" = "worker" ]; then
        if [ -z "$VLLM_GPU_MEMORY_UTILIZATION" ]; then
            log_warning "VLLM_GPU_MEMORY_UTILIZATION not set. Defaulting to 0.7 to prevent OOM."
            export VLLM_GPU_MEMORY_UTILIZATION=0.7
        else
            log_info "vLLM Memory Limit: $VLLM_GPU_MEMORY_UTILIZATION"
        fi
    fi
}

# ============================================================================
# Directory initialization
# ============================================================================
initialize_directories() {
    log_info "Initializing directory structure..."

    mkdir -p /app/models
    mkdir -p /app/data/uploads
    mkdir -p /app/data/output
    mkdir -p /app/logs

    log_success "Directory structure initialized"
}

# ============================================================================
# MinerU Configuration Generator
# ============================================================================
generate_mineru_config() {
    log_info "Generating MinerU configuration (magic-pdf.json)..."

    # 这里的路径必须与 docker-compose.yml 中的挂载路径一致
    # 我们挂载的是: - /mnt/d/.../mineru:/app/models/mineru
    MODEL_DIR="/app/models/mineru"

    # 检查模型目录是否存在
    if [ ! -d "$MODEL_DIR" ]; then
        log_warning "MinerU model directory not found at $MODEL_DIR. Creating empty directory..."
        mkdir -p "$MODEL_DIR"
    else
        log_info "MinerU models found: $(ls $MODEL_DIR | tr '\n' ' ')"
    fi

    # 生成 magic-pdf.json 到用户主目录 (/root)
    # 这是 MinerU 识别本地模型的唯一方式
    cat > /root/magic-pdf.json <<EOF
{
  "models-dir": "${MODEL_DIR}",
  "device-mode": "cuda",
  "table-config": {
    "model": "TableMaster",
    "is_table_recog_enable": true,
    "max_time": 400
  },
  "layout-config": {
    "model": "doclayout_yolo"
  },
  "formula-config": {
    "mfd_model": "yolo_v8_mfd",
    "mfr_model": "unimernet_small",
    "enable": true
  }
}
EOF
    # 为了兼容性，同时也生成 mineru.json (新版可能用这个名字)
    cp /root/magic-pdf.json /root/mineru.json

    log_success "MinerU configuration generated at /root/magic-pdf.json pointing to ${MODEL_DIR}"
}

# ============================================================================
# PaddleOCR Configuration Check (新增)
# ============================================================================
check_paddleocr_config() {
    log_info "Checking PaddleOCR models..."
    
    # 检查默认缓存路径
    if [ -d "/root/.paddleocr" ]; then
        log_success "PaddleOCR cache directory found (/root/.paddleocr)"
        # 简单列出 whl 目录下的模型
        if [ -d "/root/.paddleocr/whl" ]; then
             log_info "Available PaddleOCR models: $(ls /root/.paddleocr/whl)"
        fi
    else
        log_warning "PaddleOCR cache directory NOT found! Models will be downloaded at runtime."
    fi

    # 检查大模型挂载 (vLLM用)
    if [ -d "/app/models/paddleocr-vl-v1.5" ]; then
        log_success "PaddleOCR-VL-1.5 (vLLM) model found"
    else
        log_warning "PaddleOCR-VL-1.5 model NOT found at /app/models/paddleocr-vl-v1.5"
    fi
}

# ============================================================================
# Database initialization
# ============================================================================
initialize_database() {
    log_info "Checking database..."

    DB_PATH=${DATABASE_PATH:-/app/data/db/mineru_tianshu.db}
    
    # 确保数据库目录存在
    mkdir -p $(dirname "$DB_PATH")

    if [ -f "$DB_PATH" ]; then
        log_success "Database exists: $DB_PATH"
    else
        log_info "First run, database will be automatically created"
    fi
}

# ============================================================================
# GPU check
# ============================================================================
check_gpu() {
    log_info "Checking GPU availability..."

    # Check PyTorch
    if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
        GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
        log_success "PyTorch CUDA detected: $GPU_NAME"
    else
        log_warning "PyTorch CUDA NOT detected!"
    fi

    # Check PaddlePaddle
    if python -c "import paddle; print(paddle.device.is_compiled_with_cuda())" | grep -q "True"; then
        log_success "PaddlePaddle CUDA detected"
    else
        log_warning "PaddlePaddle CUDA NOT detected!"
    fi
}

# ============================================================================
# Main entry point
# ============================================================================
main() {
    log_info "=========================================="
    log_info "Tianshu Starting (RTX 5090 Optimized)..."
    log_info "=========================================="

    # First determine service type
    SERVICE_TYPE=${1:-api}

    # Run checks (pass service type)
    check_environment "$SERVICE_TYPE"
    initialize_directories
    initialize_database
    
    # ✅ 生成 MinerU 配置
    generate_mineru_config
    
    # ✅ 检查 PaddleOCR 配置
    check_paddleocr_config

    # Execute different checks based on service type
    if [ "$SERVICE_TYPE" = "worker" ]; then
        log_info "Startup type: LitServe Worker"
        check_gpu
        shift  # Remove first argument (service type)
    elif [ "$SERVICE_TYPE" = "mcp" ]; then
        log_info "Startup type: MCP Server"
        shift  # Remove first argument (service type)
    else
        log_info "Startup type: API Server"
        # If first argument is "api", also need to remove it
        if [ "$1" = "api" ]; then
            shift
        fi
    fi

    log_info "=========================================="
    log_success "Initialization complete, starting service..."
    log_info "=========================================="

    # Execute the passed command
    exec "$@"
}

# Catch signals for graceful shutdown
trap 'log_warning "Received termination signal, shutting down..."; exit 0' SIGTERM SIGINT

# Execute main function
main "$@"
