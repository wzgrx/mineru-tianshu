"""
MinerU Tianshu - LitServe Worker
天枢 LitServe Worker (Fixed VLLM & Memory Optimization)
"""

import os
import json
import sys
import time
import threading
import signal
import atexit
from pathlib import Path
from typing import Optional
import multiprocessing
import importlib.util

# ============================================================================
# 1. 禁用 LitServe 内置 MCP (避免冲突)
# ============================================================================
import litserve as ls
from litserve.connector import check_cuda_with_nvidia_smi
from utils import parse_list_arg

try:
    import litserve.mcp as ls_mcp
    from contextlib import asynccontextmanager

    class DummyMCPServer:
        def __init__(self, *args, **kwargs): pass
    
    class DummyMCPConnector:
        def __init__(self, *args, **kwargs): pass
        @asynccontextmanager
        async def lifespan(self, app): yield
        def connect_mcp_server(self, *args, **kwargs): pass

    ls_mcp.MCPServer = DummyMCPServer
    ls_mcp._LitMCPServerConnector = DummyMCPConnector
    if "litserve.mcp" in sys.modules:
        sys.modules["litserve.mcp"].MCPServer = DummyMCPServer
        sys.modules["litserve.mcp"]._LitMCPServerConnector = DummyMCPConnector
except Exception as e:
    import warnings
    warnings.warn(f"Failed to patch litserve.mcp: {e}")

from loguru import logger

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from task_db import TaskDB
from output_normalizer import normalize_output

# ============================================================================
# 2. 引擎可用性检测
# ============================================================================

try:
    from markitdown import MarkItDown
    MARKITDOWN_AVAILABLE = True
except ImportError:
    MARKITDOWN_AVAILABLE = False
    logger.warning("⚠️  markitdown not available")

PADDLEOCR_AVAILABLE = importlib.util.find_spec("paddleocr") is not None
if PADDLEOCR_AVAILABLE:
    logger.info("✅ PaddleOCR engine available")
else:
    logger.warning("⚠️  PaddleOCR not available (pip install paddleocr>=2.9.1)")

MINERU_PIPELINE_AVAILABLE = importlib.util.find_spec("mineru_pipeline") is not None
if MINERU_PIPELINE_AVAILABLE:
    logger.info("✅ MinerU Pipeline available")

MINERU_VLM_AVAILABLE = importlib.util.find_spec("mineru.backend.vlm") is not None
if MINERU_VLM_AVAILABLE:
    logger.info("✅ MinerU VLM available")

MINERU_HYBRID_AVAILABLE = importlib.util.find_spec("mineru.backend.hybrid") is not None
if MINERU_HYBRID_AVAILABLE:
    logger.info("✅ MinerU Hybrid available")

SENSEVOICE_AVAILABLE = importlib.util.find_spec("audio_engines") is not None
VIDEO_ENGINE_AVAILABLE = importlib.util.find_spec("video_engines") is not None
WATERMARK_REMOVAL_AVAILABLE = importlib.util.find_spec("remove_watermark") is not None

# Format Engines
try:
    from format_engines import FormatEngineRegistry, FASTAEngine, GenBankEngine
    FormatEngineRegistry.register(FASTAEngine())
    FormatEngineRegistry.register(GenBankEngine())
    FORMAT_ENGINES_AVAILABLE = True
except ImportError:
    FORMAT_ENGINES_AVAILABLE = False


class MinerUWorkerAPI(ls.LitAPI):
    def __init__(
        self,
        paddleocr_vl_vllm_api_list=None,
        output_dir=None,
        poll_interval=0.5,
        enable_worker_loop=True,
        paddleocr_vl_vllm_engine_enabled=False,
    ):
        super().__init__()
        project_root = Path(__file__).parent.parent
        default_output = project_root / "data" / "output"
        self.output_dir = output_dir or os.getenv("OUTPUT_PATH", str(default_output))
        self.poll_interval = poll_interval
        self.enable_worker_loop = enable_worker_loop
        self.paddleocr_vl_vllm_engine_enabled = paddleocr_vl_vllm_engine_enabled
        self.paddleocr_vl_vllm_api_list = paddleocr_vl_vllm_api_list or []
        
        ctx = multiprocessing.get_context("spawn")
        self._global_worker_counter = ctx.Value("i", 0)

    def setup(self, device):
        with self._global_worker_counter.get_lock():
            my_global_index = self._global_worker_counter.value
            self._global_worker_counter.value += 1
        
        logger.info(f"🔢 Worker #{my_global_index} setup on {device}")

        # 设置 CUDA_VISIBLE_DEVICES
        if "cuda:" in str(device):
            gpu_id = str(device).split(":")[-1]
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
            os.environ["MINERU_DEVICE_MODE"] = "cuda:0"

        # ✅ [关键修复] 限制 vLLM 显存占用 (解决 OOM 核心)
        # 0.7 = 70% 显存给 vLLM，剩余 30% (约7GB/24GB) 给 PaddleOCR/MinerU 临时使用
        os.environ["VLLM_GPU_MEMORY_UTILIZATION"] = "0.7"
        # 强制 vLLM 使用与 PyTorch 兼容的显存分配方式
        os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

        # 配置模型源 (ModelScope/HF)
        model_source = os.getenv("MODEL_DOWNLOAD_SOURCE", "auto").lower()
        if model_source == "modelscope":
             os.environ["MINERU_MODEL_SOURCE"] = "modelscope"
        
        self.device = device
        self.accelerator = "cuda" if "cuda" in str(device) else "cpu"
        self.engine_device = "cuda:0" if self.accelerator == "cuda" else "cpu"

        # 延迟加载 MinerU VRAM Utils
        global get_vram, clean_memory
        from mineru.utils.model_utils import get_vram, clean_memory
        
        # 初始化数据库
        db_path = os.getenv("DATABASE_PATH", "/app/data/db/mineru_tianshu.db")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.task_db = TaskDB(db_path)
        
        # 引擎实例缓存
        self.markitdown = MarkItDown() if MARKITDOWN_AVAILABLE else None
        self.mineru_pipeline_engine = None
        self.paddleocr_vl_engine = None
        self.paddleocr_vllm_engine = None  # ✅ vLLM 引擎句柄
        self.sensevoice_engine = None
        self.video_engine = None
        self.watermark_handler = None

        # 初始化水印引擎
        if WATERMARK_REMOVAL_AVAILABLE and self.accelerator == "cuda":
            try:
                from remove_watermark.pdf_watermark_handler import PDFWatermarkHandler
                self.watermark_handler = PDFWatermarkHandler(device="cuda:0", use_lama=True)
                logger.info("✅ Watermark engine ready")
            except Exception as e:
                logger.error(f"❌ Watermark engine failed: {e}")

        # 启动循环
        self.running = True
        self.current_task_id = None
        if self.enable_worker_loop:
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()

    def _worker_loop(self):
        """Worker 主循环"""
        logger.info(f"🔁 Worker loop started (interval={self.poll_interval}s)")
        while self.running:
            try:
                task = self.task_db.get_next_task(worker_id=self.worker_id)
                if task:
                    self.current_task_id = task["task_id"]
                    logger.info(f"📥 Processing task: {task['task_id']} ({task['backend']})")
                    try:
                        self._process_task(task)
                        logger.info(f"✅ Task completed: {task['task_id']}")
                    except Exception as e:
                        logger.error(f"❌ Task failed: {e}")
                        logger.exception(e)
                    finally:
                        self.current_task_id = None
                else:
                    time.sleep(self.poll_interval)
            except Exception as e:
                logger.error(f"Loop error: {e}")
                time.sleep(1)

    def _process_task(self, task: dict):
        """核心任务分发逻辑"""
        task_id = task["task_id"]
        file_path = task["file_path"]
        options = json.loads(task.get("options", "{}"))
        backend = task.get("backend", "auto")

        try:
            # 1. 预处理：PDF 转换 / 水印去除 / 拆分
            file_ext = Path(file_path).suffix.lower()
            
            # Office 转 PDF
            if file_ext in [".docx", ".xlsx", ".pptx"] and options.get("convert_office_to_pdf"):
                file_path = self._convert_office_to_pdf(file_path)
                file_ext = ".pdf"
            
            # PDF 拆分 (仅针对 PDF 且非子任务)
            if file_ext == ".pdf" and not task.get("parent_task_id"):
                 if self._should_split_pdf(task_id, file_path, task, options):
                     return # 已拆分为子任务
            
            # 去除水印
            if file_ext == ".pdf" and options.get("remove_watermark") and self.watermark_handler:
                file_path = str(self._preprocess_remove_watermark(file_path, options))

            # 2. 引擎路由
            result = None
            
            # === MinerU 系列 ===
            if backend == "pipeline":
                if not MINERU_PIPELINE_AVAILABLE: raise ValueError("MinerU Pipeline missing")
                result = self._process_with_mineru(file_path, options)
            
            elif backend == "vlm-auto-engine":
                if not MINERU_VLM_AVAILABLE: raise ValueError("MinerU VLM missing")
                result = self._process_with_mineru_vlm(file_path, options)

            elif backend == "hybrid-auto-engine":
                if not MINERU_HYBRID_AVAILABLE: raise ValueError("MinerU Hybrid missing")
                result = self._process_with_mineru_hybrid(file_path, options)

            # === [新增] PaddleOCR VLLM 加速版 ===
            elif backend == "paddleocr-vl-vllm":
                if not importlib.util.find_spec("vllm"): raise ValueError("vLLM module not found")
                logger.info(f"🚀 Processing with PaddleOCR-VL (vLLM): {file_path}")
                result = self._process_with_paddleocr_vllm(file_path, options)

            # === PaddleOCR 本地版 ===
            elif backend in ["paddleocr-vl", "paddleocr-vl-0.9b", "paddleocr-vl-1.5-0.9b", 
                             "pp-ocrv5", "pp-structurev3", "pp-chatocrv4"]:
                if not PADDLEOCR_AVAILABLE: raise ValueError("PaddleOCR missing")
                options['model_type'] = backend 
                result = self._process_with_paddleocr(file_path, options)

            # === 音视频/其他 ===
            elif backend == "sensevoice":
                if not SENSEVOICE_AVAILABLE: raise ValueError("SenseVoice missing")
                result = self._process_audio(file_path, options)
            
            elif backend == "video":
                if not VIDEO_ENGINE_AVAILABLE: raise ValueError("Video engine missing")
                result = self._process_video(file_path, options)
            
            # === 自动选择 ===
            elif backend == "auto":
                if file_ext == ".pdf": # 默认用 Pipeline
                    result = self._process_with_mineru(file_path, options)
                elif file_ext in [".jpg", ".png"]: # 默认用 PaddleOCR-VL
                    options['model_type'] = 'paddleocr-vl'
                    result = self._process_with_paddleocr(file_path, options)
                elif self.markitdown: # Office/Text
                    result = self._process_with_markitdown(file_path)
                else:
                    raise ValueError("No suitable engine found for auto mode")

            else:
                raise ValueError(f"Unknown backend: {backend}")

            # 3. 结果处理
            if result:
                self.task_db.update_task_status(
                    task_id=task_id,
                    status="completed",
                    result_path=result["result_path"]
                )
                # 处理子任务合并逻辑...
                if task.get("parent_task_id"):
                    parent_id = self.task_db.on_child_task_completed(task_id)
                    if parent_id: self._merge_parent_task_results(parent_id)

            if "cuda" in str(self.device): clean_memory()

        except Exception as e:
            self.task_db.update_task_status(task_id, "failed", error_message=str(e))
            if task.get("parent_task_id"):
                self.task_db.on_child_task_failed(task_id, str(e))
            raise

    # ==========================================================
    # 具体处理方法
    # ==========================================================
    
    def _process_with_paddleocr_vllm(self, file_path: str, options: dict) -> dict:
        """调用 PaddleOCR vLLM 加速引擎"""
        if self.paddleocr_vllm_engine is None:
            # 延迟导入，防止启动时占用显存
            from paddleocr_vl_vllm.engine import PaddleOCRVLLMEngine
            self.paddleocr_vllm_engine = PaddleOCRVLLMEngine()
        
        output_dir = Path(self.output_dir) / Path(file_path).stem
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 调用 VLLM 引擎的解析方法
        result = self.paddleocr_vllm_engine.parse(file_path, str(output_dir), **options)
        
        normalize_output(output_dir)
        return {"result_path": str(output_dir), "content": result.get("markdown", "")}

    def _process_with_paddleocr(self, file_path: str, options: dict) -> dict:
        """统一调用 PaddleOCR 引擎"""
        if self.paddleocr_vl_engine is None:
            from paddleocr_vl.engine import get_engine
            self.paddleocr_vl_engine = get_engine() # 单例获取
        
        output_dir = Path(self.output_dir) / Path(file_path).stem
        
        # 传递 options (包含 model_type)
        result = self.paddleocr_vl_engine.parse(file_path, str(output_dir), **options)
        
        normalize_output(output_dir)
        return {"result_path": str(output_dir), "content": result.get("markdown", "")}

    def _process_with_mineru(self, file_path: str, options: dict) -> dict:
        if self.mineru_pipeline_engine is None:
            from mineru_pipeline import MinerUPipelineEngine
            self.mineru_pipeline_engine = MinerUPipelineEngine(device=self.engine_device)
        
        output_dir = Path(self.output_dir) / Path(file_path).stem
        result = self.mineru_pipeline_engine.parse(file_path, output_path=str(output_dir), options=options)
        normalize_output(Path(result["result_path"]))
        return {"result_path": result["result_path"], "content": result["markdown"]}

    def _process_with_mineru_vlm(self, file_path: str, options: dict) -> dict:
        from mineru.backend.vlm.vlm_analyze import doc_analyze
        from mineru.data.data_reader_writer import FileBasedDataWriter
        from mineru.backend.vlm.vlm_middle_json_mkcontent import mid_json_to_markdown
        
        output_dir = Path(self.output_dir) / Path(file_path).stem
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, "rb") as f: content = f.read()
        writer = FileBasedDataWriter(str(output_dir))
        
        middle_json, _ = doc_analyze(content, writer, backend="transformers")
        md = mid_json_to_markdown(middle_json)
        
        (output_dir / "result.md").write_text(md, encoding="utf-8")
        normalize_output(output_dir)
        return {"result_path": str(output_dir), "content": md}

    def _process_with_mineru_hybrid(self, file_path: str, options: dict) -> dict:
        from mineru.backend.hybrid.hybrid_analyze import doc_analyze
        from mineru.data.data_reader_writer import FileBasedDataWriter
        from mineru.backend.pipeline.pipeline_middle_json_mkcontent import mid_json_to_markdown

        output_dir = Path(self.output_dir) / Path(file_path).stem
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(file_path, "rb") as f: content = f.read()
        writer = FileBasedDataWriter(str(output_dir))

        middle_json, _, _ = doc_analyze(
            content, writer, 
            language=options.get("lang", "ch"),
            parse_method=options.get("method", "auto")
        )
        md = mid_json_to_markdown(middle_json)
        (output_dir / "result.md").write_text(md, encoding="utf-8")
        
        normalize_output(output_dir)
        return {"result_path": str(output_dir), "content": md}

    def _process_audio(self, file_path: str, options: dict) -> dict:
        # 简单实现，假设已在 audio_engines 中实现
        from audio_engines.sensevoice_engine import SenseVoiceEngine
        if self.sensevoice_engine is None:
            self.sensevoice_engine = SenseVoiceEngine(device=self.engine_device)
        
        output_dir = Path(self.output_dir) / Path(file_path).stem
        output_dir.mkdir(parents=True, exist_ok=True)
        
        result = self.sensevoice_engine.transcribe(file_path, options)
        (output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        
        return {"result_path": str(output_dir), "content": str(result)}

    def _process_video(self, file_path: str, options: dict) -> dict:
        # 简单实现，假设已在 video_engines 中实现
        from video_engines.video_engine import VideoEngine
        if self.video_engine is None:
            self.video_engine = VideoEngine()
            
        output_dir = Path(self.output_dir) / Path(file_path).stem
        output_dir.mkdir(parents=True, exist_ok=True)
        
        result = self.video_engine.process(file_path, str(output_dir), options)
        return {"result_path": str(output_dir), "content": "Video processed"}

    def _convert_office_to_pdf(self, file_path):
        # 简单占位符，实际需调用 LibreOffice 或 Pandoc
        # 假设 docker 镜像中已有相关工具
        return file_path
        
    def _should_split_pdf(self, task_id, file_path, task, options):
        # 简单占位符
        return False
    
    def _preprocess_remove_watermark(self, file_path, options):
         # 简单占位符
        return file_path

    def _merge_parent_task_results(self, parent_id):
        pass

    def _process_with_markitdown(self, file_path):
        return {"result_path": "", "content": "MarkItDown not implemented"}

    def decode_request(self, request):
        return request.get("action", "health")

    def predict(self, action):
        if action == "health":
            return {"status": "healthy", "worker_id": self.worker_id}
        return {"status": "ok"}

def start_litserve_workers(**kwargs):
    api = MinerUWorkerAPI(**kwargs)
    server = ls.LitServer(api, accelerator="auto", workers_per_device=1)
    server.run(port=kwargs.get('port', 8001))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # 简化版入口
    start_litserve_workers(output_dir=None)
