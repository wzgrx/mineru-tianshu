"""
PaddleOCR 统一解析引擎
支持模型:
1. PaddleOCR-VL (v1 / v1.5) - 多模态文档理解
2. PP-OCRv5 - 高精度纯文本识别
3. PP-StructureV3 - 版面分析与表格还原
4. PP-ChatOCRv4 - 对话式关键信息提取
"""
import os
import copy
from pathlib import Path
from typing import Optional, Dict, Any
from threading import Lock
from loguru import logger
import numpy as np

# 尝试导入必要的库
try:
    import paddle
    from paddleocr import PaddleOCR, PPStructure
    
    # 尝试导入 PaddleOCR-VL
    try:
        from paddleocr import PaddleOCRVL
    except ImportError:
        PaddleOCRVL = None
        logger.warning("⚠️ PaddleOCRVL not found. Please upgrade paddleocr>=2.9.1")

    # 尝试导入 PPChatOCRv4Doc
    try:
        from paddleocr import PPChatOCRv4Doc
    except ImportError:
        PPChatOCRv4Doc = None
        
    import fitz # PyMuPDF
except ImportError as e:
    logger.error(f"❌ Missing dependencies: {e}")
    raise

class PaddleOCREngine:
    """
    PaddleOCR 统一引擎管理器 (单例模式)
    """
    _instance: Optional["PaddleOCREngine"] = None
    _lock = Lock()
    _models = {}  # 模型缓存池: { 'model_key': model_instance }

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, device: str = "cuda:0"):
        if hasattr(self, "_initialized") and self._initialized:
            return
        
        with self._lock:
            if hasattr(self, "_initialized") and self._initialized:
                return
            
            self.device = device
            self.use_gpu = "cuda" in str(device).lower()
            if self.use_gpu:
                try:
                    self.gpu_id = int(str(device).split(":")[-1])
                except:
                    self.gpu_id = 0
            else:
                self.gpu_id = 0

            self._init_environment()
            self._initialized = True
            logger.info(f"🔧 PaddleOCR Engine initialized (Device: {device}, GPU: {self.use_gpu})")

    def _init_environment(self):
        """初始化 Paddle 环境"""
        try:
            if self.use_gpu:
                if not paddle.device.is_compiled_with_cuda():
                    logger.warning("⚠️ PaddlePaddle is not compiled with CUDA! Falling back to CPU.")
                    self.use_gpu = False
                else:
                    paddle.set_device(f"gpu:{self.gpu_id}")
            else:
                paddle.set_device("cpu")
        except Exception as e:
            logger.warning(f"⚠️ Failed to set paddle device: {e}")

    def _get_model(self, model_type: str, lang: str = 'ch'):
        """
        根据类型和语言懒加载模型
        """
        # 生成缓存键 (例如: paddleocr-vl-1.5-0.9b_ch)
        cache_key = f"{model_type}_{lang}"
        if cache_key in self._models:
            return self._models[cache_key]

        with self._lock:
            # 双重检查
            if cache_key in self._models:
                return self._models[cache_key]

            logger.info(f"📥 Loading PaddleOCR model: {model_type} (Lang: {lang})...")
            
            try:
                instance = None

                # =========================================================
                # 1. PaddleOCR-VL 系列 (多模态大模型)
                # =========================================================
                if 'paddleocr-vl' in model_type and 'vllm' not in model_type:
                    if PaddleOCRVL is None:
                         raise ImportError("PaddleOCRVL module not found.")

                    # 默认使用 v1.5
                    pipeline_version = 'v1.5'
                    # 如果明确指定了 0.9b 且没有 1.5 字样，则使用 v1
                    if '0.9b' in model_type and '1.5' not in model_type:
                        pipeline_version = 'v1'
                    
                    logger.info(f"   🚀 Initializing PaddleOCR-VL (Version: {pipeline_version})")
                    
                    instance = PaddleOCRVL(
                        pipeline_version=pipeline_version,
                        use_doc_orientation_classify=True,
                        use_doc_unwarping=True,
                        use_layout_detection=True
                    )

                # =========================================================
                # 2. PP-Structure 系列 (版面分析/表格)
                # =========================================================
                elif 'pp-structure' in model_type:
                    logger.info("   🏗️ Initializing PP-StructureV3")
                    instance = PPStructure(
                        show_log=False,
                        image_orientation=True,
                        layout=True,
                        table=True,
                        use_gpu=self.use_gpu,
                        gpu_id=self.gpu_id,
                        lang='ch' if lang == 'auto' else lang,
                        structure_version='PP-StructureV3'
                    )

                # =========================================================
                # 3. PP-ChatOCR 系列 (对话式提取)
                # =========================================================
                elif 'pp-chatocr' in model_type:
                    logger.info("   💬 Initializing PP-ChatOCRv4")
                    if PPChatOCRv4Doc:
                        instance = PPChatOCRv4Doc(
                            use_doc_orientation_classify=True,
                            use_doc_unwarping=True
                        )
                    else:
                        logger.warning("   ⚠️ PPChatOCRv4Doc not found. Falling back to PP-Structure(KIE).")
                        instance = PPStructure(
                            show_log=False,
                            image_orientation=True,
                            kie=True, # 启用关键信息提取
                            use_gpu=self.use_gpu,
                            gpu_id=self.gpu_id,
                            lang='ch' if lang == 'auto' else lang
                        )

                # =========================================================
                # 4. PP-OCR 系列 (纯文本识别 v4/v5)
                # =========================================================
                else:
                    # 默认为 PP-OCRv5 (PaddleOCR 会自动下载最新版)
                    logger.info("   ⚡ Initializing PP-OCRv5/v4")
                    instance = PaddleOCR(
                        use_angle_cls=True,
                        lang='ch' if lang == 'auto' else lang,
                        use_gpu=self.use_gpu,
                        gpu_id=self.gpu_id,
                        show_log=False,
                        ocr_version='PP-OCRv4' # v4/v5 共用此 tag
                    )
                
                self._models[cache_key] = instance
                logger.info(f"✅ Model loaded successfully: {cache_key}")
                return instance

            except Exception as e:
                logger.error(f"❌ Failed to load model {model_type}: {e}")
                raise

    def parse(self, file_path: str, output_path: str, **kwargs) -> Dict[str, Any]:
        """
        统一执行解析任务
        """
        file_path = Path(file_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 获取参数
        model_type = kwargs.get('model_type', 'paddleocr-vl')
        lang = kwargs.get('lang', 'ch')
        
        # 获取模型实例
        model = self._get_model(model_type, lang)
        
        markdown_content = ""
        json_data = {}
        
        try:
            # -------------------------------------------------------------
            # 分支 A: PaddleOCR-VL (原生支持 PDF/图片)
            # -------------------------------------------------------------
            if 'paddleocr-vl' in model_type and 'vllm' not in model_type:
                res = model.predict(str(file_path))
                
                if not isinstance(res, list):
                    res = [res]
                
                md_list = []
                json_list = []
                
                for i, page_res in enumerate(res):
                    # 尝试获取 markdown
                    if hasattr(page_res, 'markdown'):
                        md_list.append(page_res.markdown)
                    elif isinstance(page_res, str):
                        md_list.append(page_res)
                    
                    # 尝试获取结构化数据 (用于 JSON 输出)
                    if hasattr(page_res, 'json'):
                        json_list.append(page_res.json)
                    elif hasattr(page_res, 'res'): # 旧版本字段
                        json_list.append(page_res.res)
                        
                    # 保存单页详情 (可选，依赖 SDK 版本)
                    if hasattr(page_res, 'save_to_markdown'):
                         page_res.save_to_markdown(str(output_path))

                markdown_content = "\n\n---\n\n".join([str(m) for m in md_list])
                json_data = {"pages": json_list}

            # -------------------------------------------------------------
            # 分支 B: PP-Structure / ChatOCR (版面分析)
            # -------------------------------------------------------------
            elif 'pp-structure' in model_type or 'pp-chatocr' in model_type:
                # 兼容 PDF 处理
                if file_path.suffix.lower() == '.pdf':
                    import fitz # PyMuPDF
                    from PIL import Image
                    doc = fitz.open(file_path)
                    full_md = []
                    full_json = []

                    for i, page in enumerate(doc):
                        pix = page.get_pixmap()
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        img_np = np.array(img)
                        
                        # 推理
                        result = model(img_np)
                        
                        # 结果转 Markdown
                        page_md = f"## Page {i+1}\n\n"
                        page_structure = []
                        
                        if result:
                            # 某些版本直接返回 list，某些返回 tuple
                            regions = result[0] if isinstance(result, tuple) else result
                            
                            for region in regions:
                                r_type = region.get('type', '')
                                r_res = region.get('res', {})
                                
                                # 收集 JSON 数据
                                page_structure.append({
                                    "type": r_type,
                                    "bbox": region.get('bbox'),
                                    "content": r_res
                                })
                                
                                if r_type == 'table':
                                    page_md += f"\n{r_res.get('html', '')}\n"
                                else:
                                    text_lines = r_res if isinstance(r_res, list) else [r_res]
                                    for line in text_lines:
                                        if isinstance(line, dict):
                                            page_md += f"{line.get('text', '')}\n"
                                        else:
                                            page_md += f"{str(line)}\n"
                                            
                        full_md.append(page_md)
                        full_json.append({"page": i+1, "regions": page_structure})

                    markdown_content = "\n\n---\n\n".join(full_md)
                    json_data = {"structure_results": full_json}
                else:
                    # 单图处理
                    result = model(str(file_path))
                    # ... (类似上面的处理逻辑，简化略) ...
                    markdown_content = str(result)
                    json_data = {"raw": str(result)}

            # -------------------------------------------------------------
            # 分支 C: PP-OCR (纯文本识别)
            # -------------------------------------------------------------
            else:
                res = model.ocr(str(file_path), cls=True)
                
                # 兼容 PDF (list of list) 和 图片 (list)
                is_pdf = file_path.suffix.lower() == '.pdf'
                pages_res = res if is_pdf else [res]
                
                full_txt = []
                raw_json = []
                
                for idx, page_data in enumerate(pages_res):
                    if not page_data: continue
                    
                    page_str = f"## Page {idx+1}\n"
                    page_lines = []
                    
                    for line in page_data:
                        text = line[1][0]
                        page_str += text + "\n"
                        page_lines.append({
                            "text": text,
                            "confidence": float(line[1][1]),
                            "bbox": line[0]
                        })
                    
                    full_txt.append(page_str)
                    raw_json.append({"page": idx+1, "lines": page_lines})
                
                markdown_content = "\n\n---\n\n".join(full_txt)
                json_data = {"ocr_results": raw_json}

            # -------------------------------------------------------------
            # 保存结果
            # -------------------------------------------------------------
            if not markdown_content:
                markdown_content = "> No content detected."

            (output_path / "result.md").write_text(markdown_content, encoding="utf-8")
            
            import json
            class NpEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer): return int(obj)
                    if isinstance(obj, np.floating): return float(obj)
                    if isinstance(obj, np.ndarray): return obj.tolist()
                    return super(NpEncoder, self).default(obj)

            (output_path / "result.json").write_text(
                json.dumps(json_data, ensure_ascii=False, indent=2, cls=NpEncoder), 
                encoding="utf-8"
            )

            return {
                "success": True,
                "markdown": markdown_content
            }

        except Exception as e:
            logger.error(f"PaddleOCR Processing Failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def cleanup(self):
        """清理显存"""
        try:
            import paddle
            import gc
            if self.use_gpu:
                paddle.device.cuda.empty_cache()
            gc.collect()
        except:
            pass

# ✅ 关键：添加工厂函数，供 litserve_worker.py 调用
_engine = None
def get_engine() -> PaddleOCREngine:
    global _engine
    if _engine is None:
        _engine = PaddleOCREngine()
    return _engine
