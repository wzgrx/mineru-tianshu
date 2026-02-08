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
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ============================================================================
# Environment check
# ============================================================================
check_environment() {
    local service_type=$1
    log_info "Checking environment configuration..."

    # Check CUDA
    if command -v nvidia-smi &> /dev/null; then
        log_success "NVIDIA GPU detected"
    else
        log_warning "NVIDIA GPU or driver not detected"
    fi

    # Check JWT (API only)
    if [ "$service_type" != "worker" ] && [ "$service_type" != "mcp" ]; then
        if [ -z "$JWT_SECRET_KEY" ]; then
            log_error "JWT_SECRET_KEY is not set! Please configure in .env"
            exit 1
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
    # PaddleOCR 缓存目录
    mkdir -p /root/.paddlex
}

# ============================================================================
# Smart Model Management (核心修改：自动检测与下载)
# ============================================================================
manage_models() {
    log_info "Starting Smart Model Management..."

    # 容器内挂载点 (对应宿主机 D:\aiworkspace\models\mineru)
    MINERU_DIR="/app/models/mineru"
    
    # 确保目录存在 (Docker挂载会自动创建，但为了保险)
    if [ ! -d "$MINERU_DIR" ]; then
        mkdir -p "$MINERU_DIR"
    fi

    # ---------------------------------------------------------
    # 1. 检测现有模型 (支持多种层级结构)
    # ---------------------------------------------------------
    MODEL_READY=false
    FINAL_MODEL_PATH=""

    # 检查路径 A: .../PDF-Extract-Kit-1.0/models/Layout/...
    if [ -f "$MINERU_DIR/PDF-Extract-Kit-1.0/models/Layout/doclayout_yolo/best.pt" ]; then
        FINAL_MODEL_PATH="$MINERU_DIR/PDF-Extract-Kit-1.0/models"
        MODEL_READY=true
        log_success "Found models in sub-directory: $FINAL_MODEL_PATH"
        
    # 检查路径 B: .../opendatalab/PDF-Extract-Kit-1.0/models/Layout/... (ModelScope 默认结构)
    elif [ -f "$MINERU_DIR/opendatalab/PDF-Extract-Kit-1.0/models/Layout/doclayout_yolo/best.pt" ]; then
        FINAL_MODEL_PATH="$MINERU_DIR/opendatalab/PDF-Extract-Kit-1.0/models"
        MODEL_READY=true
        log_success "Found models in ModelScope cache dir: $FINAL_MODEL_PATH"

    # 检查路径 C: .../models/Layout/... (直接解压)
    elif [ -f "$MINERU_DIR/models/Layout/doclayout_yolo/best.pt" ]; then
        FINAL_MODEL_PATH="$MINERU_DIR/models"
        MODEL_READY=true
        log_success "Found models in models dir: $FINAL_MODEL_PATH"
        
    # 检查路径 D: .../Layout/... (完全扁平)
    elif [ -f "$MINERU_DIR/Layout/doclayout_yolo/best.pt" ]; then
        FINAL_MODEL_PATH="$MINERU_DIR"
        MODEL_READY=true
        log_success "Found models in root dir: $FINAL_MODEL_PATH"
    fi

    # ---------------------------------------------------------
    # 2. 如果没找到，自动下载 (ModelScope)
    # ---------------------------------------------------------
    if [ "$MODEL_READY" = false ]; then
        log_warning "Models missing in $MINERU_DIR"
        log_info "🚀 Starting auto-download from ModelScope (China)..."
        
        # 使用 Python 调用 modelscope 下载
        python3 -c "
import os
try:
    from modelscope.hub.snapshot_download import snapshot_download
    print('Downloading PDF-Extract-Kit-1.0 to $MINERU_DIR ...')
    # cache_dir 指定为挂载目录，这样会下载到 D 盘
    path = snapshot_download('opendatalab/PDF-Extract-Kit-1.0', cache_dir='$MINERU_DIR')
    print(f'Download success: {path}')
except ImportError:
    print('ModelScope library not found!')
    exit(1)
except Exception as e:
    print(f'Download failed: {e}')
    exit(1)
"
        if [ $? -eq 0 ]; then
            log_success "Download completed successfully!"
            # 下载后重新探测路径 (ModelScope 通常下载到 opendatalab/... 下)
            if [ -d "$MINERU_DIR/opendatalab/PDF-Extract-Kit-1.0/models" ]; then
                FINAL_MODEL_PATH="$MINERU_DIR/opendatalab/PDF-Extract-Kit-1.0/models"
            else
                # 暴力搜索 best.pt 重新定位
                FOUND=$(find "$MINERU_DIR" -name "best.pt" | grep "doclayout_yolo" | head -n 1)
                if [ -n "$FOUND" ]; then
                    # ../../.. 回退到 models 目录
                    FINAL_MODEL_PATH=$(dirname $(dirname $(dirname "$FOUND")))
                fi
            fi
        else
            log_error "Auto-download failed. Please check network or download manually."
            # 失败后防止 Crash，指向根目录
            FINAL_MODEL_PATH="$MINERU_DIR"
        fi
    else
        log_info "Models exist. Skipping download."
    fi

    # ---------------------------------------------------------
    # 3. 生成配置文件
    # ---------------------------------------------------------
    if [ -z "$FINAL_MODEL_PATH" ]; then FINAL_MODEL_PATH="$MINERU_DIR"; fi
    
    log_info "Generating configuration pointing to: $FINAL_MODEL_PATH"

    cat > /root/magic-pdf.json <<EOF
{
  "models-dir": "${FINAL_MODEL_PATH}",
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
    cp /root/magic-pdf.json /root/mineru.json
    chmod 644 /root/magic-pdf.json
}

# ============================================================================
# Database initialization
# ============================================================================
initialize_database() {
    log_info "Checking database..."
    DB_PATH=${DATABASE_PATH:-/app/data/db/mineru_tianshu.db}
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
    if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
        log_success "PyTorch CUDA detected"
    else
        log_warning "PyTorch CUDA NOT detected!"
    fi
}

# ============================================================================
# Main entry point
# ============================================================================
main() {
    log_info "=========================================="
    log_info "Tianshu Starting (Smart Model Mode)..."
    log_info "=========================================="

    SERVICE_TYPE=${1:-api}

    check_environment "$SERVICE_TYPE"
    initialize_directories
    initialize_database
    
    # ✅ 执行智能模型管理 (检测 -> 下载 -> 配置)
    manage_models

    if [ "$SERVICE_TYPE" = "worker" ]; then
        log_info "Startup type: LitServe Worker"
        check_gpu
        shift 
    elif [ "$SERVICE_TYPE" = "mcp" ]; then
        log_info "Startup type: MCP Server"
        shift
    else
        log_info "Startup type: API Server"
        if [ "$1" = "api" ]; then shift; fi
    fi

    log_info "=========================================="
    log_success "Initialization complete, starting service..."
    log_info "=========================================="

    exec "$@"
}

trap 'log_warning "Received termination signal, shutting down..."; exit 0' SIGTERM SIGINT
main "$@"
