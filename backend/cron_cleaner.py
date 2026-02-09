"""
Tianshu Auto Cleaner
自动清理超过指定时间（默认24小时）的上传文件和输出结果
"""
import os
import time
from pathlib import Path
from loguru import logger

# 配置
UPLOAD_DIR = "/app/data/uploads"
OUTPUT_DIR = "/app/data/output"
MAX_AGE_SECONDS = 24 * 3600  # 24小时

def cleanup_directory(directory):
    if not os.path.exists(directory):
        return
    
    logger.info(f"🧹 Scanning {directory}...")
    now = time.time()
    count = 0
    size_freed = 0
    
    for item in Path(directory).rglob("*"):
        if item.is_file():
            try:
                # 检查最后修改时间
                if now - item.stat().st_mtime > MAX_AGE_SECONDS:
                    size = item.stat().st_size
                    item.unlink() # 删除文件
                    size_freed += size
                    count += 1
            except Exception as e:
                logger.error(f"Failed to delete {item}: {e}")
    
    # 清理空目录
    for item in Path(directory).rglob("*"):
        if item.is_dir() and not any(item.iterdir()):
            try:
                item.rmdir()
            except: pass

    if count > 0:
        logger.info(f"✅ Cleaned {count} files, freed {size_freed / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    logger.info("🕒 Starting cleanup job")
    cleanup_directory(UPLOAD_DIR)
    cleanup_directory(OUTPUT_DIR)
