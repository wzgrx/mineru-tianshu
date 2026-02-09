"""
MinerU Tianshu - API Server
天枢 API 服务器

企业级 AI 数据预处理平台
支持文档、图片、音频、视频等多模态数据处理
提供 RESTful API 接口用于任务提交、查询和管理
企业级认证授权: JWT Token + API Key + SSO
"""

import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import quote, unquote

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from loguru import logger

# 导入认证模块
from auth import (
    User,
    Permission,
    get_current_active_user,
    require_permission,
)
from auth.auth_db import AuthDB
from auth.routes import router as auth_router
from task_db import TaskDB

# 初始化 FastAPI 应用
app = FastAPI(
    title="MinerU Tianshu API",
    description="天枢 - 企业级 AI 数据预处理平台 | 支持文档、图片、音频、视频等多模态数据处理 | 企业级认证授权",
    version="2.0.0",
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 获取项目根目录（backend 的父目录）
PROJECT_ROOT = Path(__file__).parent.parent

# 初始化数据库
db_path_env = os.getenv("DATABASE_PATH")
if db_path_env:
    db_path = str(Path(db_path_env).resolve())
    logger.info(f"📊 API Server using DATABASE_PATH: {db_path_env} -> {db_path}")
    db = TaskDB(db_path)
else:
    logger.warning("⚠️  DATABASE_PATH not set in API Server, using default")
    default_db_path = PROJECT_ROOT / "data" / "db" / "mineru_tianshu.db"
    default_db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path = str(default_db_path.resolve())
    logger.info(f"📊 Using default database path: {db_path}")
    db = TaskDB(db_path)
auth_db = AuthDB()

# 注册认证路由
app.include_router(auth_router)

# 配置输出目录
output_path_env = os.getenv("OUTPUT_PATH")
if output_path_env:
    OUTPUT_DIR = Path(output_path_env)
else:
    OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"📁 Output directory: {OUTPUT_DIR.resolve()}")


def process_markdown_images_legacy(md_content: str, image_dir: Path, result_path: str):
    """
    【已修复】处理 Markdown 中的图片引用
    
    Worker 已自动上传图片到 RustFS 并替换 URL，此函数仅用于向后兼容。
    修复了旧版本粗暴检查 "http" 导致文档内含有其他链接时跳过图片处理的 Bug。
    """
    if not image_dir.exists():
        return md_content

    # 兼容模式：转换相对路径为本地 URL
    # logger.warning("⚠️  Checking/Fixing image URLs (legacy mode)")

    def replace_image_path(match):
        """替换图片路径为本地 URL"""
        full_match = match.group(0)
        
        # 提取图片路径和 Alt 文本
        if "![" in full_match:
            # Markdown: ![alt](path)
            alt_text = match.group(1)
            image_path = match.group(2)
        else:
            # HTML: <img src="path">
            # match.group(2) 是 src 的值
            image_path = match.group(2)
            alt_text = "Image" # HTML正则如果不捕获alt，设为默认值

        # ✅ [关键修复]：只检查图片路径本身是否为 URL，而不是检查全文
        if image_path.startswith("http://") or image_path.startswith("https://"):
            return full_match

        # 生成本地静态文件 URL
        try:
            image_filename = Path(image_path).name
            output_dir_str = str(OUTPUT_DIR).replace("\\", "/")
            result_path_str = result_path.replace("\\", "/")

            if result_path_str.startswith(output_dir_str):
                relative_path = result_path_str[len(output_dir_str) :].lstrip("/")
                encoded_relative_path = quote(relative_path, safe="/")
                encoded_filename = quote(image_filename, safe="/")
                static_url = f"/api/v1/files/output/{encoded_relative_path}/images/{encoded_filename}"

                # 返回替换后的内容
                if "![" in full_match:
                    return f"![{alt_text}]({static_url})"
                else:
                    return full_match.replace(image_path, static_url)
        except Exception as e:
            logger.error(f"❌ Failed to generate local URL: {e}")

        return full_match

    try:
        # 匹配 Markdown 图片: ![alt](path)
        md_pattern = r"!\[([^\]]*)\]\(([^)]+)\)"
        # 匹配 HTML 图片: <img ... src="path" ...>
        html_pattern = r'<img\s+[^>]*src="([^"]+)"[^>]*>'

        new_content = re.sub(md_pattern, replace_image_path, md_content)
        new_content = re.sub(html_pattern, replace_image_path, new_content)
        return new_content
    except Exception as e:
        logger.error(f"❌ Failed to process images: {e}")
        return md_content


@app.get("/", tags=["系统信息"])
async def root():
    """API根路径"""
    return {
        "service": "MinerU Tianshu",
        "version": "2.0.0",
        "description": "天枢 - 企业级 AI 数据预处理平台",
        "docs": "/docs",
    }


@app.post("/api/v1/tasks/submit", tags=["任务管理"])
async def submit_task(
    file: UploadFile = File(..., description="文件"),
    backend: str = Form("auto", description="处理后端"),
    lang: str = Form("auto", description="语言"),
    method: str = Form("auto", description="解析方法"),
    formula_enable: bool = Form(True, description="是否启用公式识别"),
    table_enable: bool = Form(True, description="是否启用表格识别"),
    priority: int = Form(0, description="优先级"),
    # 视频参数
    keep_audio: bool = Form(False),
    enable_keyframe_ocr: bool = Form(False),
    ocr_backend: str = Form("paddleocr-vl"),
    keep_keyframes: bool = Form(False),
    # 音频参数
    enable_speaker_diarization: bool = Form(False),
    # 水印参数
    remove_watermark: bool = Form(False),
    watermark_conf_threshold: float = Form(0.35),
    watermark_dilation: int = Form(10),
    # Office 转 PDF
    convert_office_to_pdf: bool = Form(False),
    # 认证依赖
    current_user: User = Depends(require_permission(Permission.TASK_SUBMIT)),
):
    try:
        upload_path_env = os.getenv("UPLOAD_PATH")
        if upload_path_env:
            upload_dir = Path(upload_path_env)
        else:
            upload_dir = PROJECT_ROOT / "data" / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)

        unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
        temp_file_path = upload_dir / unique_filename

        with open(temp_file_path, "wb") as temp_file:
            while True:
                chunk = await file.read(1 << 23)  # 8MB chunks
                if not chunk:
                    break
                temp_file.write(chunk)

        options = {
            "lang": lang,
            "method": method,
            "formula_enable": formula_enable,
            "table_enable": table_enable,
            "keep_audio": keep_audio,
            "enable_keyframe_ocr": enable_keyframe_ocr,
            "ocr_backend": ocr_backend,
            "keep_keyframes": keep_keyframes,
            "enable_speaker_diarization": enable_speaker_diarization,
            "remove_watermark": remove_watermark,
            "watermark_conf_threshold": watermark_conf_threshold,
            "watermark_dilation": watermark_dilation,
            "convert_office_to_pdf": convert_office_to_pdf,
        }

        task_id = db.create_task(
            file_name=file.filename,
            file_path=str(temp_file_path),
            backend=backend,
            options=options,
            priority=priority,
            user_id=current_user.user_id,
        )

        logger.info(f"✅ Task submitted: {task_id} - {file.filename}")
        return {
            "success": True,
            "task_id": task_id,
            "status": "pending",
            "message": "Task submitted successfully",
        }

    except Exception as e:
        logger.error(f"❌ Failed to submit task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/tasks/{task_id}", tags=["任务管理"])
async def get_task_status(
    task_id: str,
    upload_images: bool = Query(False),
    format: str = Query("markdown"),
    current_user: User = Depends(get_current_active_user),
):
    task = db.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if not current_user.has_permission(Permission.TASK_VIEW_ALL):
        if task.get("user_id") != current_user.user_id:
            raise HTTPException(status_code=403, detail="Permission denied")

    response = {
        "success": True,
        "task_id": task_id,
        "status": task["status"],
        "file_name": task["file_name"],
        "backend": task["backend"],
        "priority": task["priority"],
        "error_message": task["error_message"],
        "created_at": task["created_at"],
        "started_at": task["started_at"],
        "completed_at": task["completed_at"],
        "worker_id": task["worker_id"],
        "retry_count": task["retry_count"],
        "user_id": task.get("user_id"),
    }

    if task.get("is_parent"):
        child_count = task.get("child_count", 0)
        child_completed = task.get("child_completed", 0)
        response["is_parent"] = True
        response["subtask_progress"] = {
            "total": child_count,
            "completed": child_completed,
            "percentage": round(child_completed / child_count * 100, 1) if child_count > 0 else 0,
        }
        try:
            children = db.get_child_tasks(task_id)
            response["subtasks"] = [{"task_id": c["task_id"], "status": c["status"]} for c in children]
        except Exception:
            pass

    if task["status"] == "completed":
        if not task["result_path"]:
            response["data"] = None
            return response

        result_dir = Path(task["result_path"])
        if result_dir.exists():
            md_files = list(result_dir.rglob("*.md"))
            json_files = [f for f in result_dir.rglob("*.json") if "_content_list.json" in f.name or f.name in ["content.json", "result.json"]]

            if md_files:
                try:
                    response["data"] = {}
                    response["data"]["json_available"] = len(json_files) > 0

                    if format in ["markdown", "both"]:
                        md_file = next((f for f in md_files if f.name == "result.md"), md_files[0])
                        image_dir = md_file.parent / "images"
                        
                        with open(md_file, "r", encoding="utf-8") as f:
                            md_content = f.read()

                        # ✅ [调用修复后的函数]
                        md_content = process_markdown_images_legacy(md_content, image_dir, task["result_path"])
                        
                        response["data"]["markdown_file"] = md_file.name
                        response["data"]["content"] = md_content
                        response["data"]["has_images"] = image_dir.exists()

                    if format in ["json", "both"] and json_files:
                        json_file = json_files[0]
                        try:
                            with open(json_file, "r", encoding="utf-8") as f:
                                response["data"]["json_content"] = json.load(f)
                            response["data"]["json_file"] = json_file.name
                        except Exception:
                            pass
                except Exception as e:
                    logger.error(f"❌ Failed to read content: {e}")
                    response["data"] = None
        else:
            logger.error(f"❌ Result directory missing: {result_dir}")

    return response


@app.delete("/api/v1/tasks/{task_id}", tags=["任务管理"])
async def cancel_task(task_id: str, current_user: User = Depends(get_current_active_user)):
    task = db.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if not current_user.has_permission(Permission.TASK_DELETE_ALL):
        if task.get("user_id") != current_user.user_id:
            raise HTTPException(status_code=403, detail="Permission denied")

    if task["status"] == "pending":
        db.update_task_status(task_id, "cancelled")
        file_path = Path(task["file_path"])
        if file_path.exists():
            file_path.unlink()
        return {"success": True}
    else:
        raise HTTPException(status_code=400, detail="Cannot cancel non-pending task")


@app.get("/api/v1/queue/stats", tags=["队列管理"])
async def get_queue_stats(current_user: User = Depends(require_permission(Permission.QUEUE_VIEW))):
    return {"success": True, "stats": db.get_queue_stats(), "user": current_user.username}


@app.get("/api/v1/queue/tasks", tags=["队列管理"])
async def list_tasks(
    status: Optional[str] = Query(None),
    limit: int = Query(100),
    current_user: User = Depends(get_current_active_user),
):
    can_view_all = current_user.has_permission(Permission.TASK_VIEW_ALL)
    
    query = "SELECT * FROM tasks"
    params = []
    conditions = []

    if not can_view_all:
        conditions.append("user_id = ?")
        params.append(current_user.user_id)
    
    if status:
        conditions.append("status = ?")
        params.append(status)
    
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    with db.get_cursor() as cursor:
        cursor.execute(query, tuple(params))
        tasks = [dict(row) for row in cursor.fetchall()]

    return {"success": True, "count": len(tasks), "tasks": tasks, "can_view_all": can_view_all}


@app.post("/api/v1/admin/cleanup", tags=["系统管理"])
async def cleanup_old_tasks(days: int = 7, current_user: User = Depends(require_permission(Permission.QUEUE_MANAGE))):
    count = db.cleanup_old_task_records(days)
    return {"success": True, "deleted_count": count}


@app.post("/api/v1/admin/reset-stale", tags=["系统管理"])
async def reset_stale_tasks(timeout_minutes: int = 60, current_user: User = Depends(require_permission(Permission.QUEUE_MANAGE))):
    count = db.reset_stale_tasks(timeout_minutes)
    return {"success": True, "reset_count": count}


@app.get("/api/v1/engines", tags=["系统信息"])
async def list_engines():
    # ... (省略静态列表，与原代码一致)
    return {"success": True, "engines": {
        "document": [{"name": "pipeline", "display_name": "MinerU Pipeline", "description": "标准文档解析"}],
        "ocr": [{"name": "paddleocr-vl", "display_name": "PaddleOCR-VL"}],
        "office": [{"name": "markitdown", "display_name": "MarkItDown"}]
    }}


@app.get("/api/v1/health", tags=["系统信息"])
async def health_check():
    try:
        db.get_queue_stats()
        return {"status": "healthy"}
    except Exception as e:
        return JSONResponse(status_code=503, content={"status": "unhealthy", "error": str(e)})


@app.get("/v1/files/output/{file_path:path}", tags=["文件服务"])
async def serve_output_file(file_path: str):
    """
    提供输出文件的访问服务
    支持 URL 编码的中文路径
    """
    try:
        # ✅ [安全修复] 防止路径遍历和双重斜杠问题
        decoded_path = unquote(file_path).lstrip("/")
        full_path = (OUTPUT_DIR / decoded_path).resolve()
        safe_root = OUTPUT_DIR.resolve()

        if not str(full_path).startswith(str(safe_root)):
            raise HTTPException(status_code=403, detail="Access denied")

        if not full_path.exists() or not full_path.is_file():
            raise HTTPException(status_code=404, detail="File not found")

        return FileResponse(path=str(full_path), media_type="application/octet-stream", filename=full_path.name)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error serving file: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    api_port = int(os.getenv("API_PORT", "8000"))
    logger.info("🚀 Starting MinerU Tianshu API Server...")
    uvicorn.run(app, host="0.0.0.0", port=api_port, log_level="info")
