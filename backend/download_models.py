#!/usr/bin/env python3
"""
模型预下载脚本 - 为 CPU/GPU 离线部署准备所有必需模型

功能:
1. 下载 MinerU 模型 (PDF解析)
2. 下载 PaddleOCR-VL-1.5-0.9B (用于 vLLM 加速或本地推理) [新增]
3. 下载 SenseVoice 音频识别模型
4. 下载 Paraformer 说话人分离模型
5. 下载 YOLO11 水印检测模型
6. 触发 PaddleOCR 常规小模型自动下载 (可选)
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from loguru import logger

# 配置日志
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")

# 模型配置
MODELS = {
    "mineru": {
        "name": "MinerU PDF-Extract-Kit",
        "repo_id": "opendatalab/PDF-Extract-Kit-1.0",
        "source": "huggingface",
        "target_dir": "mineru/", # 建议统一放在 models/mineru 下
        "description": "PDF OCR and layout analysis models",
        "required": True
    },
    # 【新增】PaddleOCR-VL 大模型 (用于 vLLM)
    "paddleocr_vl": {
        "name": "PaddleOCR-VL-1.5-0.9B",
        "model_id": "PaddlePaddle/PaddleOCR-VL-1.5-0.9B", # ModelScope ID
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
        "description": "Watermark detection model for document processing",
        "required": False
    },
    "lama": {
        "name": "LaMa Watermark Inpainting",
        "auto_download": True,
        "description": "Will be downloaded by simple_lama_inpainting on first use",
        "required": False
    },
    # 常规 OCR 模型 (检测/识别/表格)
    # 这些通常由 paddleocr 库自动管理，但为了离线环境，建议手动下载 .paddleocr 目录
    "paddleocr_core": {
        "name": "PaddleOCR Core Models (v4/v5)",
        "auto_download": True, 
        "description": "Standard OCR models. For offline use, ensure ~/.paddleocr is populated.",
        "required": False
    }
}


def download_from_huggingface(repo_id, target_dir, filename=None):
    """从 HuggingFace 下载模型"""
    try:
        from huggingface_hub import snapshot_download, hf_hub_download

        # 配置镜像（国内加速）
        hf_endpoint = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
        os.environ.setdefault("HF_ENDPOINT", hf_endpoint)

        if filename:
            # 下载单个文件
            logger.info(f"   Downloading file: {filename}")
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=str(Path(target_dir).parent), # HF 缓存机制特殊，这里简化处理
                local_dir=str(target_dir), # 直接下载到指定目录
                resume_download=True
            )
        else:
            # 下载整个仓库
            logger.info(f"   Downloading repository: {repo_id}")
            path = snapshot_download(
                repo_id=repo_id,
                local_dir=str(target_dir), # 直接下载到指定目录，不使用软链接缓存
                resume_download=True
            )

        return path

    except ImportError:
        logger.error("   ❌ huggingface_hub not installed. Install: pip install huggingface-hub")
        return None
    except Exception as e:
        logger.error(f"   ❌ Download failed: {e}")
        return None


def download_from_modelscope(model_id, target_dir):
    """从 ModelScope 下载模型"""
    try:
        from modelscope import snapshot_download

        logger.info(f"   Downloading from ModelScope: {model_id}")
        # ModelScope 默认会下载到 cache 目录，我们指定 local_dir
        path = snapshot_download(
            model_id,
            local_dir=str(target_dir),
            revision="master"
        )

        return path

    except ImportError:
        logger.error("   ❌ modelscope not installed. Install: pip install modelscope")
        return None
    except Exception as e:
        logger.error(f"   ❌ Download failed: {e}")
        return None


def verify_model_files(path, model_name):
    """验证模型文件完整性"""
    if not path or not Path(path).exists():
        return False

    path_obj = Path(path)

    # 检查关键文件（根据不同模型类型）
    if model_name == "mineru":
        # MinerU 应该包含 .safetensors 或 .bin 文件
        has_model = any(path_obj.rglob("*.safetensors")) or any(path_obj.rglob("*.bin"))
        if not has_model:
            logger.warning(f"   ⚠️  No model files (.safetensors/.bin) found in {path}")
            return False

    elif model_name == "paddleocr_vl":
        # 检查 model.pdparams
        if not any(path_obj.rglob("*.pdparams")):
            logger.warning(f"   ⚠️  No PaddleOCR-VL model files found in {path}")
            return False

    elif model_name in ["sensevoice", "paraformer"]:
        # ModelScope 模型应该包含配置文件
        config_file = path_obj / "configuration.json"
        if not config_file.exists():
            # 尝试查找其他配置文件
            config_file = path_obj / "config.json"
        if not config_file.exists():
            logger.warning(f"   ⚠️  No configuration file found in {path}")
            return False

    elif model_name == "yolo11":
        # YOLO 模型应该是 .pt 文件
        if not str(path).endswith(".pt"):
            logger.warning(f"   ⚠️  Invalid YOLO model file: {path}")
            return False

    logger.info(f"   ✅ Model files verified")
    return True


def get_directory_size(path):
    """获取目录大小（MB）"""
    if not path or not Path(path).exists():
        return 0

    path_obj = Path(path)
    if path_obj.is_file():
        return path_obj.stat().st_size / (1024 * 1024)

    total_size = 0
    for file_path in path_obj.rglob("*"):
        if file_path.is_file():
            total_size += file_path.stat().st_size

    return total_size / (1024 * 1024)


def check_model_exists(output_path, config, name):
    """检查模型是否已存在"""
    target_dir = output_path / config["target_dir"]

    if not target_dir.exists():
        return False, "Directory not found"

    # 根据不同模型类型检查关键文件
    if name == "mineru":
        has_model = any(target_dir.rglob("*.safetensors")) or any(target_dir.rglob("*.bin"))
        return has_model, "Model files found" if has_model else "Model files missing"
    
    elif name == "paddleocr_vl":
        has_model = any(target_dir.rglob("*.pdparams"))
        return has_model, "Paddle params found" if has_model else "Model files missing"

    elif name in ["sensevoice", "paraformer"]:
        config_files = list(target_dir.rglob("configuration.json"))
        if not config_files:
            config_files = list(target_dir.rglob("config.json"))
        return bool(config_files), "Config found" if config_files else "Config missing"

    elif name == "yolo11":
        pt_files = list(target_dir.rglob("*.pt"))
        return bool(pt_files), f"{len(pt_files)} .pt files found" if pt_files else "No .pt files"

    if any(target_dir.iterdir()):
        return True, "Files found"

    return False, "Directory empty"


def main(output_dir, selected_models=None, force=False):
    """主函数"""
    logger.info("=" * 60)
    logger.info("🚀 Tianshu Model Download Script (Updated)")
    logger.info("=" * 60)

    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"📁 Output directory: {output_path}")
    if force:
        logger.info("⚠️  Force mode: Will re-download existing models")
    logger.info("")

    # 筛选要下载的模型
    models_to_download = MODELS
    if selected_models:
        selected_list = [m.strip() for m in selected_models.split(",")]
        models_to_download = {k: v for k, v in MODELS.items() if k in selected_list}
        logger.info(f"📋 Selected models: {', '.join(models_to_download.keys())}")
    else:
        logger.info(f"📋 Downloading all models ({len(MODELS)} total)")

    logger.info("")

    manifest = {
        "created": datetime.now().isoformat(),
        "platform": "cpu",
        "output_dir": str(output_path),
        "models": {},
        "total_size_mb": 0
    }

    total_downloaded = 0
    total_skipped = 0
    total_failed = 0

    # 下载每个模型
    for name, config in models_to_download.items():
        logger.info(f"📦 [{name.upper()}] {config['name']}")
        logger.info(f"   {config['description']}")

        try:
            # 检查是否是自动下载的模型
            if config.get("auto_download"):
                logger.info(f"   ℹ️  {name} will be downloaded automatically on first run")
                manifest["models"][name] = {
                    "name": config["name"],
                    "status": "auto_download",
                    "description": config["description"]
                }
                logger.info("")
                continue

            # 创建目标目录
            target = output_path / config["target_dir"]
            target.mkdir(parents=True, exist_ok=True)

            # 检查模型是否已存在（除非使用 --force）
            if not force:
                exists, reason = check_model_exists(output_path, config, name)
                if exists:
                    size_mb = get_directory_size(target)
                    logger.info(f"   ✅ Already exists ({size_mb:.1f} MB)")
                    logger.info(f"   📂 Path: {target}")
                    manifest["models"][name] = {
                        "name": config["name"],
                        "status": "already_exists",
                        "size_mb": round(size_mb, 2),
                        "path": str(target),
                        "description": config["description"]
                    }
                    manifest["total_size_mb"] += size_mb
                    total_skipped += 1
                    logger.info("")
                    continue
                else:
                    logger.info(f"   ℹ️  Not found: {reason}")

            # 下载模型
            logger.info(f"   ⬇️  Downloading...")
            path = None
            if config["source"] == "huggingface":
                path = download_from_huggingface(
                    config["repo_id"],
                    str(target),
                    config.get("filename")
                )
            elif config["source"] == "modelscope":
                path = download_from_modelscope(config["model_id"], str(target))

            if path:
                # 验证下载
                if verify_model_files(path, name):
                    size_mb = get_directory_size(path)
                    manifest["models"][name] = {
                        "name": config["name"],
                        "status": "downloaded",
                        "path": str(path),
                        "size_mb": round(size_mb, 2),
                        "description": config["description"]
                    }
                    manifest["total_size_mb"] += size_mb
                    logger.info(f"   ✅ Downloaded successfully ({size_mb:.1f} MB)")
                    logger.info(f"   📂 Path: {path}")
                    total_downloaded += 1
                else:
                    manifest["models"][name] = {
                        "name": config["name"],
                        "status": "verification_failed",
                        "path": str(path) if path else None,
                        "description": config["description"]
                    }
                    total_failed += 1
            else:
                manifest["models"][name] = {
                    "name": config["name"],
                    "status": "download_failed",
                    "error": "Download failed",
                    "description": config["description"]
                }
                total_failed += 1

        except Exception as e:
            logger.error(f"   ❌ Error downloading {name}: {e}")
            manifest["models"][name] = {
                "name": config["name"],
                "status": "error",
                "error": str(e),
                "description": config["description"]
            }
            total_failed += 1

        logger.info("")

    # 保存清单
    manifest_file = output_path / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # 输出总结
    logger.info("=" * 60)
    logger.info("📊 Download Summary")
    logger.info("=" * 60)
    logger.info(f"✅ Successfully downloaded: {total_downloaded} models")
    if total_skipped > 0:
        logger.info(f"⏭️  Skipped (already exists): {total_skipped} models")
    logger.info(f"❌ Failed: {total_failed} models")
    logger.info(f"💾 Total size: {manifest['total_size_mb']:.1f} MB")
    logger.info(f"📄 Manifest saved to: {manifest_file}")
    logger.info("")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download all models for Tianshu CPU offline deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output",
        default="./models-offline",
        help="Output directory for downloaded models (default: ./models-offline)"
    )
    parser.add_argument(
        "--models",
        help="Comma-separated list of models to download (default: all). Available: "
             + ", ".join(MODELS.keys())
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download all models, even if they already exist"
    )

    args = parser.parse_args()

    try:
        exit_code = main(args.output, args.models, args.force)
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Download interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
