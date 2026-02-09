"""
MinerU Pipeline Engine
单例模式，每个进程只加载一次模型
"""

import os
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
from threading import Lock
from loguru import logger
import img2pdf


class MinerUPipelineEngine:
    """
    MinerU Pipeline 引擎

    特性：
    - 单例模式
    - 封装 MinerU 的 do_parse 调用
    - 延迟加载（避免过早初始化模型）
    - 支持 PDF 和图片（自动转换）
    - 自动处理输出路径和结果解析
    - 线程安全
    """

    _instance: Optional["MinerUPipelineEngine"] = None
    _lock = Lock()
    _pipeline = None  # 这里的 pipeline 实际上是 do_parse 函数
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, device: str = "cuda:0"):
        """
        初始化引擎

        Args:
            device: 设备 (cuda:0, cuda:1 等)
        """
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            self.device = device
            # 简单的设备ID提取
            if "cuda:" in device:
                self.gpu_id = device.split(":")[-1]
            else:
                self.gpu_id = "0"
            
            # 设置环境变量以确保 MinerU 使用正确的 GPU
            os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_id

            self._initialized = True
            logger.info(f"🔧 MinerU Pipeline Engine initialized on {device}")

    def _load_pipeline(self):
        """延迟加载 MinerU 管道 (do_parse)"""
        if self._pipeline is not None:
            return self._pipeline

        with self._lock:
            if self._pipeline is not None:
                return self._pipeline

            logger.info("=" * 60)
            logger.info("📥 Loading MinerU Pipeline (do_parse)...")
            logger.info("=" * 60)

            try:
                # 延迟导入 do_parse，避免过早初始化模型
                from mineru.cli.common import do_parse

                self._pipeline = do_parse

                logger.info("=" * 60)
                logger.info("✅ MinerU Pipeline loaded successfully!")
                logger.info("=" * 60)

                return self._pipeline

            except ImportError:
                logger.error("❌ Failed to import mineru.cli.common.do_parse. Is mineru installed?")
                raise
            except Exception as e:
                logger.error(f"❌ Error loading MinerU pipeline: {e}")
                raise

    def cleanup(self):
        """清理显存"""
        try:
            from mineru.utils.model_utils import clean_memory
            clean_memory()
            logger.debug("🧹 MinerU: Memory cleanup completed")
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Memory cleanup warning: {e}")

    def parse(self, file_path: str, output_path: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        处理文件

        Args:
            file_path: 输入文件路径
            output_path: 输出目录路径 (任务根目录)
            options: 处理选项

        Returns:
            包含结果的字典
        """
        options = options or {}
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        file_path_obj = Path(file_path)
        file_stem = file_path_obj.stem
        file_ext = file_path_obj.suffix.lower()

        # 加载管道
        do_parse_func = self._load_pipeline()

        temp_pdf_path = None

        try:
            # 读取文件为字节
            with open(file_path, "rb") as f:
                file_bytes = f.read()

            # 处理图片输入: 转 PDF
            if file_ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
                logger.info(f"🖼️ Converting image to PDF: {file_path_obj.name}")
                try:
                    pdf_bytes = img2pdf.convert(file_bytes)
                    # 临时保存这个转换后的 PDF，因为 MinerU 内部有些逻辑依赖文件名
                    # 为了避免并发冲突，使用原始文件名但加 .pdf 后缀
                    temp_pdf_name = f"{file_stem}.pdf"
                    # 这里我们不实际写入磁盘再读，直接传递 bytes 给 do_parse 即可
                    # 但为了逻辑统一，变量名保持一致
                    file_name_for_mineru = temp_pdf_name
                except Exception as e:
                    logger.error(f"❌ Image conversion failed: {e}")
                    raise ValueError(f"Failed to convert image to PDF: {e}")
            else:
                # PDF 文件
                pdf_bytes = file_bytes
                file_name_for_mineru = file_path_obj.name

            # 获取语言设置 (MinerU 仅支持 'ch' 或 'en')
            lang = options.get("lang", "ch")
            if lang not in ["ch", "en"]:
                lang = "ch"

            logger.info(f"🚀 Running MinerU do_parse on: {file_name_for_mineru} (Lang: {lang})")

            # 调用 MinerU (do_parse)
            # 注意: output_dir 必须是一个存在的目录
            do_parse_func(
                pdf_file_names=[file_name_for_mineru],  # 文件名列表
                pdf_bytes_list=[pdf_bytes],       # 文件字节列表
                p_lang_list=[lang],               # 语言列表
                output_dir=str(output_dir),       # 输出目录
                output_format="md_json",          # 强制输出 md 和 json
                end_page_id=options.get("end_page_id", None), # 默认处理所有页
                layout_mode=options.get("layout_mode", True),
                formula_enable=options.get("formula_enable", True),
                table_enable=options.get("table_enable", True),
            )

            # --- 结果解析 ---
            # MinerU 通常会在 output_dir 下创建一个与文件名(不含后缀)同名的子目录
            # 例如输入 a.pdf，输出在 output_dir/a/auto/a.md
            
            # 使用 file_stem (去除后缀的文件名) 来定位子目录
            # 注意: 如果是图片转 PDF，file_stem 应该也是原始图片的文件名(不含后缀)
            expected_subdir = output_dir / Path(file_name_for_mineru).stem
            
            # 查找 Markdown 文件
            # 优先在 expected_subdir 中查找，找不到则全目录搜索
            md_files = list(expected_subdir.rglob("*.md"))
            if not md_files:
                md_files = list(output_dir.rglob("*.md"))

            if md_files:
                # 排序，取最短路径的 md 文件（通常是主文件，而非readme）
                md_files.sort(key=lambda p: len(str(p)))
                md_file = md_files[0]
                
                logger.info(f"✅ Found MinerU output: {md_file}")
                content = md_file.read_text(encoding="utf-8")

                # 实际的结果目录 (包含 images, layout.json 等)
                actual_result_dir = md_file.parent

                # 查找 content_list.json
                json_files = list(actual_result_dir.glob("*_content_list.json"))
                
                result = {
                    "markdown": content,
                    "result_path": str(actual_result_dir), # 返回包含资源的目录
                }

                if json_files:
                    json_file = json_files[0]
                    try:
                        with open(json_file, "r", encoding="utf-8") as f:
                            result["json_content"] = json.load(f)
                            result["json_path"] = str(json_file)
                    except Exception as e:
                        logger.warning(f"Failed to load JSON: {e}")

                return result

            else:
                # 失败处理：打印目录结构以便调试
                logger.error(f"❌ MinerU output not found in {output_dir}")
                logger.error("Directory content:")
                for f in output_dir.rglob("*"):
                    logger.error(f"  - {f.relative_to(output_dir)}")
                
                raise FileNotFoundError("MinerU failed to generate markdown output")

        except Exception as e:
            logger.error(f"MinerU Processing Failed: {e}")
            raise

        finally:
            self.cleanup()


# 全局单例
_engine = None

def get_engine() -> MinerUPipelineEngine:
    """获取全局引擎实例"""
    global _engine
    if _engine is None:
        _engine = MinerUPipelineEngine()
    return _engine
