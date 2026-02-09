"""
PaddleOCR 统一解析引擎 (最终修复版 - 适配 PaddleOCR 3.0+)
支持模型:
1. PaddleOCR-VL (v1 / v1.5) - 多模态文档理解
2. PP-OCRv5 - 高精度纯文本识别 (支持 109 种语言)
3. PP-StructureV3 - 版面分析与表格还原 (使用新版 API)
4. PP-ChatOCRv4 - 智能信息提取 (基础视觉模式)
"""
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
from threading import Lock
from loguru import logger
import numpy as np

# 尝试导入必要的库
try:
    import paddle
    # 基础 OCR
    from paddleocr import PaddleOCR
    
    # 尝试导入 3.x 新增/更新的类
    try:
        from paddleocr import PaddleOCRVL
    except ImportError:
        PaddleOCRVL = None
        logger.warning("⚠️ PaddleOCRVL not found. Please upgrade paddleocr>=2.9.1")

    try:
        from paddleocr import PPStructureV3
    except ImportError:
        PPStructureV3 = None
        # 尝试使用旧版兼容
        from paddleocr import PPStructure
        logger.warning("⚠️ PPStructureV3 class not found, using PPStructure compatibility mode.")

    try:
        from paddleocr import PPChatOCRv4Doc
    except ImportError:
        PPChatOCRv4Doc = None
        
    import fitz # PyMuPDF, 用于 PDF 转图片
except ImportError as e:
    logger.error(f"❌ Missing dependencies: {e}. Please run: pip install 'paddleocr>=2.9.1' pymupdf")
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
        if hasattr(self, "_initialized") and self._initialized: return
        
        with self._lock:
            if hasattr(self, "_initialized") and self._initialized: return
            
            self.device = device
            self.use_gpu = "cuda" in str(device).lower()
            if self.use_gpu:
                try:
                    self.gpu_id = int(str(device).split(":")[-1])
                except: self.gpu_id = 0
            else: self.gpu_id = 0
            
            self._init_env()
            self._initialized = True
            logger.info(f"🔧 PaddleOCR Engine initialized (Device: {device}, GPU: {self.use_gpu})")

    def _init_env(self):
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
        """根据类型和语言懒加载模型实例"""
        # 缓存键
        cache_key = f"{model_type}_{lang}"
        if cache_key in self._models: return self._models[cache_key]

        with self._lock:
            if cache_key in self._models: return self._models[cache_key]
            logger.info(f"📥 Loading PaddleOCR model: {model_type} (Lang: {lang})...")
            
            instance = None
            try:
                # =========================================================
                # 1. PaddleOCR-VL 系列 (v1 / v1.5)
                # =========================================================
                if 'paddleocr-vl' in model_type and 'vllm' not in model_type:
                    if PaddleOCRVL is None:
                        raise ImportError("PaddleOCRVL not available. Check paddleocr version.")
                    
                    # 版本判断
                    ver = 'v1.5' # 默认最新
                    if '0.9b' in model_type and '1.5' not in model_type:
                        ver = 'v1'
                    
                    logger.info(f"   🚀 Mode: PaddleOCR-VL (Version: {ver})")
                    
                    # 【修复】移除不支持的 models_dir 参数，仅使用官方支持的参数
                    instance = PaddleOCRVL(
                        pipeline_version=ver,
                        use_doc_orientation_classify=True,
                        use_doc_unwarping=True,
                        use_layout_detection=True
                    )

                # =========================================================
                # 2. PP-StructureV3 (版面分析)
                # =========================================================
                elif 'pp-structure' in model_type:
                    logger.info("   🏗️ Mode: PP-StructureV3")
                    if PPStructureV3:
                        instance = PPStructureV3(
                            use_doc_orientation_classify=True,
                            use_doc_unwarping=True,
                            use_gpu=self.use_gpu,
                            lang='ch' if lang=='auto' else lang
                        )
                    else:
                        # 降级兼容旧版
                        from paddleocr import PPStructure
                        instance = PPStructure(
                            show_log=False,
                            image_orientation=True,
                            structure_version='PP-StructureV3',
                            use_gpu=self.use_gpu,
                            lang='ch' if lang=='auto' else lang
                        )

                # =========================================================
                # 3. PP-ChatOCRv4 (智能提取)
                # =========================================================
                elif 'pp-chatocr' in model_type:
                    logger.info("   💬 Mode: PP-ChatOCRv4")
                    if PPChatOCRv4Doc:
                        # ChatOCR 基础初始化，Visual Predict 不需要 key
                        instance = PPChatOCRv4Doc(
                            use_doc_orientation_classify=True,
                            use_doc_unwarping=True
                        )
                    else:
                        logger.warning("⚠️ PPChatOCRv4Doc not found. Falling back to PP-Structure.")
                        from paddleocr import PPStructure
                        instance = PPStructure(structure_version='PP-StructureV3')

                # =========================================================
                # 4. PP-OCRv5 (通用 OCR)
                # =========================================================
                else: 
                    logger.info("   ⚡ Mode: PP-OCRv5")
                    # PaddleOCR 3.x 会自动下载最新的 v4/v5 模型
                    instance = PaddleOCR(
                        use_angle_cls=True,
                        use_doc_orientation_classify=True,
                        lang='ch' if lang=='auto' else lang,
                        use_gpu=self.use_gpu,
                        show_log=False,
                        ocr_version='PP-OCRv4' # v4 tag 兼容 v5
                    )
                
                self._models[cache_key] = instance
                return instance
            except Exception as e:
                logger.error(f"❌ Load model failed: {e}")
                raise

    def parse(self, file_path: str, output_path: str, **kwargs) -> Dict[str, Any]:
        """
        统一解析入口
        """
        file_path = Path(file_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        model_type = kwargs.get('model_type', 'paddleocr-vl')
        lang = kwargs.get('lang', 'ch')
        
        model = self._get_model(model_type, lang)
        
        markdown_content = ""
        json_data = {}

        try:
            # === 分支 A: 产线类模型 (PaddleOCR-VL, StructureV3, ChatOCR) ===
            # 这些模型原生支持 .predict(input=path) 且支持 PDF
            if ('paddleocr-vl' in model_type and 'vllm' not in model_type) or \
               ('pp-structure' in model_type) or \
               ('pp-chatocr' in model_type and PPChatOCRv4Doc):
                
                # 1. ChatOCR 特殊处理
                if 'pp-chatocr' in model_type and PPChatOCRv4Doc and isinstance(model, PPChatOCRv4Doc):
                    logger.info("   Running ChatOCR visual_predict...")
                    # visual_predict 返回视觉信息，不进行 LLM 对话
                    res = model.visual_predict(str(file_path))
                    markdown_content = "> PP-ChatOCRv4 Visual Analysis Completed.\n> (To ask questions, configure LLM/API Key)"
                    json_data = {"visual_info": str(res)} 
                    
                # 2. VL 和 StructureV3 标准处理
                else:
                    logger.info(f"   Predicting with {model_type}...")
                    res = model.predict(input=str(file_path))
                    
                    # 转换为列表 (如果只返回单个结果)
                    pages_res = list(res) if hasattr(res, '__iter__') else [res]
                    
                    # === 关键优化：使用官方 API 进行页面重构/合并 ===
                    # PaddleOCR-VL 1.5 支持 restructure_pages
                    if 'paddleocr-vl' in model_type and hasattr(model, 'restructure_pages'):
                         try:
                             logger.info("   Restructuring pages (merging tables)...")
                             # merge_table=True 合并跨页表格
                             pages_res = model.restructure_pages(pages_res, merge_table=True)
                         except Exception as e:
                             logger.warning(f"Restructure pages failed: {e}")

                    # PP-StructureV3 支持 concatenate_markdown_pages
                    elif 'pp-structure' in model_type and hasattr(model, 'concatenate_markdown_pages'):
                        try:
                            # 提取 markdown 信息列表
                            md_list_struct = []
                            for p in pages_res:
                                if hasattr(p, 'markdown'):
                                    md_list_struct.append(p.markdown)
                            
                            if md_list_struct:
                                logger.info("   Concatenating markdown pages (StructureV3)...")
                                full_md = model.concatenate_markdown_pages(md_list_struct)
                                # 覆盖下面的逐页拼接逻辑
                                markdown_content = full_md
                        except Exception as e:
                            logger.warning(f"Concatenate markdown failed: {e}")

                    # === 逐页保存与 JSON 收集 (Fallback) ===
                    md_list_fallback = []
                    json_list = []
                    
                    for idx, p in enumerate(pages_res):
                        # 尝试使用 SDK 自带保存方法
                        if hasattr(p, 'save_to_markdown'):
                            p.save_to_markdown(str(output_path))
                        
                        # 收集内容
                        if hasattr(p, 'markdown'): md_list_fallback.append(p.markdown)
                        elif isinstance(p, dict) and 'markdown' in p: md_list_fallback.append(p['markdown'])
                        
                        if hasattr(p, 'json'): json_list.append(p.json)
                        elif isinstance(p, dict): json_list.append(p)
                    
                    # 如果没有通过 concatenate_markdown_pages 生成内容，则使用 fallback 拼接
                    if not markdown_content and md_list_fallback:
                        # 尝试读取 SDK 保存的文件
                        saved_md_files = sorted(list(output_path.glob("*.md")))
                        read_mds = []
                        for f in saved_md_files:
                            if f.name != "result.md": 
                                read_mds.append(f.read_text(encoding='utf-8'))
                        
                        if read_mds:
                            markdown_content = "\n\n---\n\n".join(read_mds)
                        else:
                            markdown_content = "\n\n---\n\n".join([str(m) for m in md_list_fallback])

                    json_data = {"pages": json_list}

            # === 分支 B: 纯 OCR 模型 (PP-OCRv5) ===
            else:
                logger.info("   Running PP-OCRv5...")
                from PIL import Image
                imgs = []
                
                # 手动 PDF 转图片
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
                            text = line[1][0]
                            page_md += text + "\n"
                    
                    full_md.append(page_md)
                    raw_res.append(str(res))
                
                markdown_content = "\n\n---\n\n".join(full_md)
                json_data = {"ocr_raw": raw_res}

            # === 最终保存 ===
            if not markdown_content: markdown_content = "(No content detected)"
            (output_path / "result.md").write_text(markdown_content, encoding="utf-8")
            
            try:
                import json
                class NpEncoder(json.JSONEncoder):
                    def default(self, obj):
                        if isinstance(obj, np.integer): return int(obj)
                        if isinstance(obj, np.floating): return float(obj)
                        if isinstance(obj, np.ndarray): return obj.tolist()
                        return super(NpEncoder, self).default(obj)
                (output_path / "result.json").write_text(json.dumps(json_data, ensure_ascii=False, indent=2, cls=NpEncoder), encoding="utf-8")
            except: pass

            return {"success": True, "markdown": markdown_content}

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def cleanup(self):
        try:
            import paddle, gc
            if self.use_gpu: paddle.device.cuda.empty_cache()
            gc.collect()
        except: pass

_engine = None
def get_engine() -> PaddleOCREngine:
    global _engine
    if _engine is None: _engine = PaddleOCREngine()
    return _engine
