"""
PaddleOCR 统一解析引擎 (最终修复版)
支持: PaddleOCR-VL (v1/v1.5), PP-OCRv5, PP-StructureV3, PP-ChatOCRv4
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
    from paddleocr import PaddleOCR, PPStructure, PaddleOCRVL
    # 导入 PyMuPDF 用于 PDF 转图片 (PP-OCR 需要)
    import fitz 
except ImportError as e:
    logger.error(f"❌ Missing dependencies: {e}. Please run: pip install paddleocr>=2.9.1 pymupdf")
    raise

class PaddleOCREngine:
    """
    PaddleOCR 引擎管理器 - 单例模式
    """
    _instance: Optional["PaddleOCREngine"] = None
    _lock = Lock()
    _models = {} 

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, device: str = "cuda:0"):
        # 防止重复初始化
        if hasattr(self, "_initialized") and self._initialized: return
        
        with self._lock:
            if hasattr(self, "_initialized") and self._initialized: return
            
            self.device = device
            self.use_gpu = "cuda" in str(device).lower()
            
            # 解析 GPU ID
            if self.use_gpu:
                try:
                    self.gpu_id = int(str(device).split(":")[-1])
                except:
                    self.gpu_id = 0
            else:
                self.gpu_id = 0
            
            self._init_env()
            self._initialized = True
            logger.info(f"🔧 PaddleOCR Engine initialized (Device: {device}, GPU: {self.use_gpu})")

    def _init_env(self):
        """初始化 Paddle 环境"""
        try:
            if self.use_gpu:
                if not paddle.device.is_compiled_with_cuda():
                    logger.warning("⚠️ PaddlePaddle CUDA not found! Falling back to CPU.")
                    self.use_gpu = False
                else:
                    paddle.set_device(f"gpu:{self.gpu_id}")
            else:
                paddle.set_device("cpu")
        except Exception as e:
            logger.warning(f"⚠️ Failed to set paddle device: {e}")

    def _get_model(self, model_type: str, lang: str = 'ch'):
        """
        根据类型和语言懒加载模型实例
        """
        cache_key = f"{model_type}_{lang}"
        if cache_key in self._models: return self._models[cache_key]

        with self._lock:
            if cache_key in self._models: return self._models[cache_key]
            logger.info(f"📥 Loading PaddleOCR model: {model_type} (Lang: {lang})...")
            
            instance = None
            try:
                # =========================================================
                # 1. PaddleOCR-VL (多模态大模型)
                # =========================================================
                if 'paddleocr-vl' in model_type and 'vllm' not in model_type:
                    ver = 'v1.5' # 默认最新
                    # 如果指定了0.9b且没指定1.5，则使用旧版 v1
                    if '0.9b' in model_type and '1.5' not in model_type: 
                        ver = 'v1'
                    
                    logger.info(f"   🚀 Initializing PaddleOCR-VL (Version: {ver})")
                    instance = PaddleOCRVL(
                        pipeline_version=ver,
                        use_doc_orientation_classify=True,
                        use_doc_unwarping=True,
                        use_layout_detection=True
                    )
                
                # =========================================================
                # 2. PP-Structure (版面分析/表格)
                # =========================================================
                elif 'pp-structure' in model_type or 'pp-chatocr' in model_type:
                    logger.info("   🏗️ Initializing PP-StructureV3")
                    instance = PPStructure(
                        show_log=False, 
                        image_orientation=True,
                        layout=True,
                        table=True, 
                        use_gpu=self.use_gpu,
                        gpu_id=self.gpu_id,
                        lang='ch' if lang=='auto' else lang,
                        structure_version='PP-StructureV3'
                    )
                
                # =========================================================
                # 3. PP-OCR (纯文本识别)
                # =========================================================
                else: 
                    logger.info("   ⚡ Initializing PP-OCRv5/v4")
                    # PaddleOCR 会自动下载最新版 (v4/v5 共用权重)
                    instance = PaddleOCR(
                        use_angle_cls=True,
                        lang='ch' if lang=='auto' else lang,
                        use_gpu=self.use_gpu,
                        gpu_id=self.gpu_id,
                        show_log=False,
                        ocr_version='PP-OCRv4' 
                    )
                
                self._models[cache_key] = instance
                logger.info(f"✅ Model {model_type} loaded successfully")
                return instance
            except Exception as e:
                logger.error(f"❌ Load model failed: {e}")
                raise

    def parse(self, file_path: str, output_path: str, **kwargs) -> Dict[str, Any]:
        """
        执行解析任务
        """
        file_path = Path(file_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 从 kwargs 获取参数 (由 litserve_worker 传递)
        model_type = kwargs.get('model_type', 'paddleocr-vl')
        lang = kwargs.get('lang', 'ch')
        
        model = self._get_model(model_type, lang)
        
        markdown_content = ""
        json_data = {}

        try:
            # -------------------------------------------------------------
            # 分支 A: PaddleOCR-VL (原生支持 PDF/图片)
            # -------------------------------------------------------------
            if 'paddleocr-vl' in model_type and 'vllm' not in model_type:
                # 预测
                res = model.predict(str(file_path))
                if not isinstance(res, list): res = [res]
                
                md_list = []
                json_list = []
                for p in res:
                    # 提取 markdown
                    if hasattr(p, 'markdown'): md_list.append(p.markdown)
                    elif isinstance(p, str): md_list.append(p)
                    
                    # 提取 JSON 结构
                    if hasattr(p, 'json'): json_list.append(p.json)
                    elif hasattr(p, 'res'): json_list.append(p.res)
                
                markdown_content = "\n\n---\n\n".join([str(m) for m in md_list])
                json_data = {"pages": json_list}

            # -------------------------------------------------------------
            # 分支 B: 其他模型 (需要手动将 PDF 转图片)
            # -------------------------------------------------------------
            else:
                from PIL import Image
                
                # 1. 准备图片列表
                imgs = []
                if file_path.suffix.lower() == '.pdf':
                    # 使用 PyMuPDF (fitz) 读取 PDF
                    doc = fitz.open(file_path)
                    for page in doc:
                        # 渲染为图片 (dpi=200 兼顾速度与精度)
                        pix = page.get_pixmap(dpi=200)
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        imgs.append(np.array(img)) # 转为 numpy 数组供 Paddle 使用
                else:
                    # 如果是图片，直接传递路径
                    imgs.append(str(file_path))

                full_res = []
                full_md = []

                # 2. 逐页推理
                for i, img_input in enumerate(imgs):
                    page_md = f"## Page {i+1}\n\n"
                    
                    # PP-Structure 推理
                    if 'pp-structure' in model_type or 'pp-chatocr' in model_type:
                        res = model(img_input)
                        # 兼容不同版本的返回格式 (list or tuple)
                        if isinstance(res, tuple): res = res[0]
                        
                        if res:
                            for region in res:
                                r_type = region.get('type', '')
                                r_res = region.get('res', {})
                                
                                if r_type == 'table': 
                                    page_md += f"\n{r_res.get('html', '')}\n"
                                else:
                                    # 文本区域
                                    lines = r_res if isinstance(r_res, list) else [r_res]
                                    for line in lines:
                                        if isinstance(line, dict): 
                                            page_md += line.get('text', '') + "\n"
                        full_res.append(str(res)) # 简化存储

                    # PP-OCR 推理
                    else:
                        res = model.ocr(img_input, cls=True)
                        # res[0] 是该页的结果列表
                        if res and res[0]:
                            for line in res[0]:
                                # line格式: [bbox, (text, score)]
                                text = line[1][0]
                                page_md += text + "\n"
                        full_res.append(str(res))
                    
                    full_md.append(page_md)

                markdown_content = "\n\n---\n\n".join(full_md)
                json_data = {"raw_results": full_res}

            # -------------------------------------------------------------
            # 保存结果
            # -------------------------------------------------------------
            if not markdown_content: 
                markdown_content = "> No text content detected."

            (output_path / "result.md").write_text(markdown_content, encoding="utf-8")
            
            # 尝试保存 JSON (忽略 numpy 序列化错误)
            try:
                import json
                # 简单的 fallback encoder
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
            except Exception as je:
                logger.warning(f"Failed to save JSON: {je}")

            return {"success": True, "markdown": markdown_content}

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def cleanup(self):
        """清理显存"""
        try:
            import paddle, gc
            if self.use_gpu: 
                paddle.device.cuda.empty_cache()
            gc.collect()
        except: pass

# 全局单例
_engine = None
def get_engine() -> PaddleOCREngine:
    global _engine
    if _engine is None: _engine = PaddleOCREngine()
    return _engine
