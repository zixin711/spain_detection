"""FastAPI 后台服务（设计第 8 章）。

启动：
    uvicorn app.main:app --host 0.0.0.0 --port 8000
手机在同一局域网下访问 http://<本机IP>:8000 即可。
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import socket
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import inference, pipeline, report, storage
from .config import ROOT, STORAGE_DIR, grading_config

MAX_FILES = 6
MAX_BYTES = 20 * 1024 * 1024

app = FastAPI(title="墙面霉菌检测系统", version="0.3.0")
_executor = ThreadPoolExecutor(max_workers=1)   # GPU/模型独占，串行执行避免 CUDA 上下文竞争
_STATIC = Path(__file__).parent / "static"


def _port() -> str:
    """从命令行里读 uvicorn 的 --port，读不到就按默认 8000。"""
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == "--port" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--port="):
            return a.split("=", 1)[1]
    return os.environ.get("PORT", "8000")


def _print_qr(url: str) -> None:
    """在终端里打印二维码，手机扫一下就能打开，省得手输 IP。"""
    try:
        import qrcode

        qr = qrcode.QRCode(border=1)
        qr.add_data(url)
        qr.make(fit=True)
        qr.print_ascii(invert=True)
    except Exception:
        pass


@app.on_event("startup")
def _startup() -> None:
    try:
        storage.init_db()
    except Exception as exc:
        print(f"\n  ✗ 存储初始化失败：{exc}")
        print("    如果报的是 disk I/O error，说明 storage 目录在网络盘/挂载盘上，")
        print("    SQLite 不支持。换个本地路径即可：")
        print("      export MOLD_STORAGE_DIR=~/mold-storage\n")
        raise

    url = f"http://{_lan_ip()}:{_port()}"
    print("\n  墙面霉菌检测服务已启动")
    print(f"  手机访问： {url}   （手机需与本机在同一个 Wi-Fi）")
    print(f"  白光模型： {inference.backend_name('white')}")
    print(f"  紫外模型： {inference.backend_name('uv')}")
    print(f"  存储目录： {STORAGE_DIR}\n")
    _print_qr(url)


def _lan_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((_STATIC / "index.html").read_text(encoding="utf-8"))


app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


@app.get("/api/health")
def health() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "status": "ok",
        "grading_config": grading_config().get("version"),
        "models": {"white": inference.backend_name("white"), "uv": inference.backend_name("uv")},
        "records": storage.count_records(),
        "lan_ip": _lan_ip(),
    }
    try:
        import torch

        info["gpu"] = {
            "available": torch.cuda.is_available(),
            "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "capability": ".".join(map(str, torch.cuda.get_device_capability(0))) if torch.cuda.is_available() else None,
        }
    except Exception:
        info["gpu"] = {"available": False, "name": None, "capability": None}
    return info


@app.post("/api/detect")
async def detect(
    images: List[UploadFile] = File(...),
    modality_hint: Optional[str] = Form(None),      # JSON 数组字符串，如 ["white","uv"]
    px_per_cm: Optional[float] = Form(None),
) -> JSONResponse:
    if not images:
        raise HTTPException(400, "请至少上传一张照片")
    if len(images) > MAX_FILES:
        raise HTTPException(400, f"一次最多上传 {MAX_FILES} 张照片")

    payload: List[tuple] = []
    for up in images:
        blob = await up.read()
        if len(blob) > MAX_BYTES:
            raise HTTPException(400, f"{up.filename} 超过 {MAX_BYTES // 1024 // 1024}MB 上限")
        if not blob:
            raise HTTPException(400, f"{up.filename} 是空文件")
        payload.append((up.filename or "upload.jpg", blob))

    hints: List[Optional[str]] = [None] * len(payload)
    if modality_hint:
        try:
            parsed = json.loads(modality_hint)
            if isinstance(parsed, list):
                for i, v in enumerate(parsed[:len(hints)]):
                    hints[i] = v if v in ("white", "uv") else None
        except json.JSONDecodeError:
            pass

    loop = asyncio.get_running_loop()
    try:
        result, originals, annotated = await loop.run_in_executor(
            _executor, pipeline.detect, payload, hints, px_per_cm)
    except Exception as exc:
        raise HTTPException(500, f"检测失败：{type(exc).__name__}: {exc}") from exc

    await loop.run_in_executor(_executor, storage.save, result, originals, annotated)
    return JSONResponse(result)


@app.get("/api/records")
def records(limit: int = 30, offset: int = 0) -> Dict[str, Any]:
    return {"total": storage.count_records(), "items": storage.list_records(limit, offset)}


@app.get("/api/records/{record_id}")
def record(record_id: str) -> Dict[str, Any]:
    data = storage.load(record_id)
    if data is None:
        raise HTTPException(404, "记录不存在")
    return data


@app.get("/api/records/{record_id}/annotated.png")
def annotated(record_id: str, src: int = 0) -> FileResponse:
    path = storage.record_dir(record_id) / f"src{src}_annotated.png"
    if not path.exists():
        raise HTTPException(404, "标注图不存在")
    return FileResponse(str(path), media_type="image/png",
                        filename=f"{record_id}_src{src}_annotated.png")


@app.get("/api/records/{record_id}/source.jpg")
def source(record_id: str, src: int = 0) -> FileResponse:
    path = storage.record_dir(record_id) / f"src{src}.jpg"
    if not path.exists():
        raise HTTPException(404, "原图不存在")
    return FileResponse(str(path), media_type="image/jpeg")


@app.get("/api/records/{record_id}/instances/{instance_id}")
def instance_detail(record_id: str, instance_id: int) -> Dict[str, Any]:
    data = storage.load(record_id)
    if data is None:
        raise HTTPException(404, "记录不存在")
    for inst in data["instances"]:
        if inst["id"] == instance_id:
            return {"instance": inst, "level": data["summary"]["level"],
                    "recommendations": data["recommendations"]}
    raise HTTPException(404, "实例不存在")


@app.get("/api/records/{record_id}/instances/{instance_id}/crop.jpg")
def instance_crop(record_id: str, instance_id: int) -> StreamingResponse:
    data = storage.load(record_id)
    if data is None:
        raise HTTPException(404, "记录不存在")
    inst = next((i for i in data["instances"] if i["id"] == instance_id), None)
    if inst is None:
        raise HTTPException(404, "实例不存在")

    src = storage.record_dir(record_id) / f"src{inst['source_index']}.jpg"
    img = cv2.imread(str(src))
    if img is None:
        raise HTTPException(404, "原图不存在")

    from .render import crop_instance

    crop = crop_instance(img, inst)
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise HTTPException(500, "裁剪失败")
    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/jpeg")


@app.get("/api/records/{record_id}/report.pdf")
def pdf_report(record_id: str) -> FileResponse:
    data = storage.load(record_id)
    if data is None:
        raise HTTPException(404, "记录不存在")
    d = storage.record_dir(record_id)
    out = d / "report.pdf"
    if not out.exists():
        try:
            report.build(data, d, out)
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc
    return FileResponse(str(out), media_type="application/pdf", filename=f"{record_id}_report.pdf")


@app.post("/api/records/{record_id}/feedback")
def feedback(record_id: str, kind: str = Form(...), instance_id: Optional[int] = Form(None),
             comment: str = Form("")) -> Dict[str, str]:
    if storage.load(record_id) is None:
        raise HTTPException(404, "记录不存在")
    storage.add_feedback(record_id, instance_id, kind, comment)
    return {"status": "ok"}


@app.delete("/api/records/{record_id}")
def delete_record(record_id: str) -> Dict[str, str]:
    if not storage.delete(record_id):
        raise HTTPException(404, "记录不存在")
    return {"status": "deleted"}
