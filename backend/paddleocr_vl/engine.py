"""
PaddleOCR 统一解析引擎 (适配 PaddleOCR 3.0+ / PaddleOCR-VL 1.5)
"""
import os
from pathlib import Path
from typing import Optional, Dict, Any
from threading import Lock
from loguru import logger
import numpy as np

# 尝试导入必要的库
try:
    import paddle
    from paddleocr import PaddleOCR
    try:
        from paddleocr import PaddleOCRVL
    except ImportError:
        PaddleOCRVL = None
        logger.warning("⚠️ PaddleOCRVL not found. Please upgrade paddleocr>=2.9.1")

    try:
        from paddleocr import PPStructureV3
    except ImportError:
        PPStructureV3 = None
        from paddleocr import PPStructure
        logger.warning("⚠️ PPStructureV3 class not found, using PPStructure compatibility mode.")

    try:
        from paddleocr import PPChatOCRv4Doc
    except ImportError:
        PPChatOCRv4Doc = None
        
    import fitz # PyMuPDF
except ImportError as e:
    logger.error(f"❌ Missing dependencies: {e}. Please run: pip install 'paddleocr>=2.9.1' pymupdf")
    raise

# ✅ [关键修复] 类名修改为 PaddleOCRVLEngine 以匹配 __init__.py
class PaddleOCRVLEngine:
    """
    PaddleOCR 引擎管理器 - 单例模式
    """
    _instance: Optional["PaddleOCRVLEngine"] = None
    _lock = Lock()
    _models = {} 

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, device: str = "cuda:0"):
        if hasattr(self, "_initialized") and self._initialized: return
        
        with self._lock:
            if hasattr(self, "_initialized") and self._initialized: return
            
            self.device = device
            self.use_gpu = "cuda" in str(device).lower()
            self._init_global_device()
            self._initialized = True
            logger.info(f"🔧 PaddleOCR Engine initialized (Device: {device})")

    def _init_global_device(self):
        try:
            if self.use_gpu:
                if not paddle.device.is_compiled_with_cuda():
                    logger.warning("⚠️ PaddlePaddle CUDA not found! Falling back to CPU.")
                    self.use_gpu = False
                    paddle.set_device("cpu")
                    self.device = "cpu"
                else:
                    paddle.set_device(self.device)
            else:
                paddle.set_device("cpu")
        except Exception as e:
            logger.warning(f"⚠️ Failed to set paddle device: {e}")

    def _get_model(self, model_type: str, lang: str = 'ch'):
        cache_key = f"{model_type}_{lang}"
        if cache_key in self._models: return self._models[cache_key]

        with self._lock:
            if cache_key in self._models: return self._models[cache_key]
            logger.info(f"📥 Loading PaddleOCR model: {model_type} (Lang: {lang})...")
            
            instance = None
            try:
                # 1. PaddleOCR-VL
                if 'paddleocr-vl' in model_type and 'vllm' not in model_type:
                    if PaddleOCRVL is None:
                        raise ImportError("PaddleOCRVL not available.")
                    ver = 'v1.5'
                    if '0.9b' in model_type and '1.5' not in model_type: ver = 'v1'
                    logger.info(f"    🚀 Mode: PaddleOCR-VL (Version: {ver})")
                    instance = PaddleOCRVL(
                        pipeline_version=ver,
                        device=self.device,
                        use_doc_orientation_classify=True,
                        use_doc_unwarping=True,
                        use_layout_detection=True
                    )

                # 2. PP-StructureV3
                elif 'pp-structure' in model_type:
                    logger.info("    🏗️ Mode: PP-StructureV3")
                    if PPStructureV3:
                        instance = PPStructureV3(
                            use_doc_orientation_classify=True,
                            use_doc_unwarping=True,
                            lang='ch' if lang=='auto' else lang,
                            device=self.device
                        )
                    else:
                        instance = PPStructure(
                            show_log=False,
                            image_orientation=True,
                            structure_version='PP-StructureV2',
                            use_gpu=self.use_gpu,
                            lang='ch' if lang=='auto' else lang
                        )

                # 3. PP-ChatOCRv4
                elif 'pp-chatocr' in model_type:
                    logger.info("    💬 Mode: PP-ChatOCRv4")
                    if PPChatOCRv4Doc:
                        instance = PPChatOCRv4Doc(
                            use_doc_orientation_classify=True,
                            use_doc_unwarping=True,
                            device=self.device
                        )
                    else:
                        instance = PPStructure(structure_version='PP-StructureV2')

                # 4. PP-OCRv5
                else: 
                    logger.info("    ⚡ Mode: PP-OCRv5")
                    instance = PaddleOCR(
                        use_angle_cls=True,
                        use_doc_orientation_classify=True,
                        lang='ch' if lang=='auto' else lang,
                        ocr_version='PP-OCRv4' 
                    )
                
                self._models[cache_key] = instance
                return instance
            except Exception as e:
                logger.error(f"❌ Load model failed: {e}")
                raise

    def parse(self, file_path: str, output_path: str, **kwargs) -> Dict[str, Any]:
        file_path = Path(file_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        model_type = kwargs.get('model_type', 'paddleocr-vl')
        lang = kwargs.get('lang', 'ch')
        
        model = self._get_model(model_type, lang)
        markdown_content = ""
        json_data = {}

        try:
            if ('paddleocr-vl' in model_type and 'vllm' not in model_type) or \
               ('pp-structure' in model_type) or \
               ('pp-chatocr' in model_type and PPChatOCRv4Doc):
                
                if 'pp-chatocr' in model_type and PPChatOCRv4Doc and isinstance(model, PPChatOCRv4Doc):
                    res = model.visual_predict(str(file_path))
                    markdown_content = "> Visual Analysis Completed."
                    json_data = {"visual_info": str(res)} 
                else:
                    logger.info(f"    Predicting with {model_type}...")
                    res = model.predict(input=str(file_path))
                    pages_res = list(res) if hasattr(res, '__iter__') else [res]
                    
                    if 'paddleocr-vl' in model_type and hasattr(model, 'restructure_pages'):
                         try:
                             pages_res = model.restructure_pages(pages_res, merge_table=True)
                         except: pass

                    md_list = []
                    json_list = []
                    for idx, p in enumerate(pages_res):
                        if hasattr(p, 'save_to_markdown'):
                            p.save_to_markdown(str(output_path))
                        if hasattr(p, 'markdown'): md_list.append(p.markdown)
                        if hasattr(p, 'json'): json_list.append(p.json)
                    
                    # 尝试读取生成的 markdown 文件
                    saved_mds = sorted(list(output_path.glob("*.md")))
                    content_list = [f.read_text(encoding='utf-8') for f in saved_mds if f.name != "result.md"]
                    
                    if content_list:
                        markdown_content = "\n\n---\n\n".join(content_list)
                    elif md_list:
                        markdown_content = "\n\n---\n\n".join([str(m) for m in md_list])
                    json_data = {"pages": json_list}

            else:
                # PP-OCRv5 fallback
                logger.info("    Running PP-OCRv5...")
                from PIL import Image
                imgs = []
                if file_path.suffix.lower() == '.pdf':
                    doc = fitz.open(file_path)
                    for page in doc:
                        pix = page.get_pixmap(dpi=200)
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        imgs.append(np.array(img))
                else:
                    imgs.append(str(file_path))

                full_md = []
                raw_res = []
                for i, img_input in enumerate(imgs):
                    res = model.ocr(img_input, cls=True)
                    page_md = f"## Page {i+1}\n"
                    if res and res[0]:
                        for line in res[0]:
                            page_md += line[1][0] + "\n"
                    full_md.append(page_md)
                    raw_res.append(str(res))
                
                markdown_content = "\n\n---\n\n".join(full_md)
                json_data = {"ocr_raw": raw_res}

            if not markdown_content: markdown_content = "(No content detected)"
            (output_path / "result.md").write_text(markdown_content, encoding="utf-8")
            
            return {"success": True, "markdown": markdown_content}

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise

    def cleanup(self):
        try:
            import paddle, gc
            if self.use_gpu: paddle.device.cuda.empty_cache()
            gc.collect()
        except: pass

_engine = None
def get_engine() -> PaddleOCRVLEngine:
    global _engine
    if _engine is None: _engine = PaddleOCRVLEngine()
    return _engine
