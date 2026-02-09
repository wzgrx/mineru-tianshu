#!/usr/bin/env python3
"""
模型预下载脚本 (最终完整版)
"""
import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from loguru import logger

logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")

MODELS = {
    "mineru": {
        "name": "MinerU PDF-Extract-Kit",
        "repo_id": "opendatalab/PDF-Extract-Kit-1.0",
        "source": "huggingface",
        "target_dir": "mineru/",
        "description": "PDF OCR and layout analysis models",
        "required": True
    },
    # ✅ 【修复】添加 PaddleOCR-VL 大模型下载配置
    "paddleocr_vl": {
        "name": "PaddleOCR-VL-1.5-0.9B",
        "model_id": "PaddlePaddle/PaddleOCR-VL-1.5-0.9B", 
        "source": "modelscope",
        "target_dir": "paddlex/PaddleOCR-VL-1.5/", # 对应 Docker 挂载路径
        "description": "Vision-Language Model for Document Parsing (Required for vLLM)",
        "required": True
    },
    "sensevoice": {
        "name": "SenseVoice Audio Recognition",
        "model_id": "iic/SenseVoiceSmall",
        "source": "modelscope",
        "target_dir": "sensevoice/",
        "description": "Multi-language speech recognition model",
        "required": True
    },
    "paraformer": {
        "name": "Paraformer Speaker Diarization",
        "model_id": "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "source": "modelscope",
        "target_dir": "paraformer/",
        "description": "Speaker diarization and VAD model",
        "required": False
    },
    "yolo11": {
        "name": "YOLO11x Watermark Detection",
        "repo_id": "corzent/yolo11x_watermark_detection",
        "filename": "best.pt",
        "source": "huggingface",
        "target_dir": "watermark_models/",
        "description": "Watermark detection model",
        "required": False
    },
    "lama": {
        "name": "LaMa Watermark Inpainting",
        "auto_download": True, 
        "description": "Will be downloaded by simple_lama_inpainting on first use",
        "required": False
    },
    "paddleocr_core": {
        "name": "PaddleOCR Core Models",
        "auto_download": True, 
        "description": "Standard OCR models (~/.paddleocr)",
        "required": False
    }
}

def download_from_huggingface(repo_id, target_dir, filename=None):
    try:
        from huggingface_hub import snapshot_download, hf_hub_download
        hf_endpoint = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
        os.environ.setdefault("HF_ENDPOINT", hf_endpoint)

        if filename:
            path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=str(Path(target_dir).parent), local_dir=str(target_dir), resume_download=True)
        else:
            path = snapshot_download(repo_id=repo_id, local_dir=str(target_dir), resume_download=True)
        return path
    except Exception as e:
        logger.error(f"   ❌ Download failed: {e}")
        return None

def download_from_modelscope(model_id, target_dir):
    try:
        from modelscope import snapshot_download
        logger.info(f"   Downloading from ModelScope: {model_id}")
        path = snapshot_download(model_id, local_dir=str(target_dir), revision="master")
        return path
    except Exception as e:
        logger.error(f"   ❌ Download failed: {e}")
        return None

def main(output_dir, selected_models=None, force=False):
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Output: {output_path}")

    models_to_download = MODELS
    if selected_models:
        s_list = [m.strip() for m in selected_models.split(",")]
        models_to_download = {k: v for k, v in MODELS.items() if k in s_list}

    for name, config in models_to_download.items():
        logger.info(f"📦 [{name.upper()}] {config['name']}")
        if config.get("auto_download"): continue
        
        target = output_path / config["target_dir"]
        if not force and target.exists() and any(target.iterdir()):
            logger.info(f"   ✅ Exists: {target}")
            continue

        if config["source"] == "huggingface":
            download_from_huggingface(config["repo_id"], target, config.get("filename"))
        elif config["source"] == "modelscope":
            download_from_modelscope(config["model_id"], target)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="./models", help="Output directory")
    parser.add_argument("--models", help="Specific models")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(args.output, args.models, args.force)
