import fitz  # PyMuPDF
from PIL import Image
import os

def convert_pdf_to_images(pdf_path, dpi=200):
    """
    将 PDF 转换为 PIL Image 列表
    
    Args:
        pdf_path (str): PDF 文件路径
        dpi (int): 输出图像的分辨率
        
    Returns:
        list[PIL.Image]: 图像对象列表
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    images = []
    
    for page in doc:
        # 使用 PyMuPDF 将页面渲染为像素图
        pix = page.get_pixmap(dpi=dpi)
        # 转换为 PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
        
    doc.close()
    return images
