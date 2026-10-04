"""记录存储：文件落盘 + SQLite 元数据（标准库 sqlite3，零额外依赖）。"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .config import STORAGE_DIR

CST = timezone(timedelta(hours=8))
_DB_LOCK = threading.Lock()


def _db_path() -> Path:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    return STORAGE_DIR / "records.db"


def init_db() -> None:
    with _DB_LOCK, sqlite3.connect(_db_path()) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS records (
                record_id   TEXT PRIMARY KEY,
                created_at  TEXT NOT NULL,
                level       TEXT NOT NULL,
                level_range TEXT NOT NULL,
                i_mold      REAL NOT NULL,
                modalities  TEXT NOT NULL,
                n_instances INTEGER NOT NULL,
                confidence  REAL NOT NULL,
                dir         TEXT NOT NULL
            )""")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                record_id TEXT NOT NULL,
                instance_id INTEGER,
                kind TEXT NOT NULL,
                comment TEXT,
                created_at TEXT NOT NULL
            )""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_records_created ON records(created_at DESC)")


def new_record_id() -> str:
    import uuid

    return datetime.now(CST).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]


def record_dir(record_id: str) -> Path:
    day = record_id.split("-")[0]
    return STORAGE_DIR / day / record_id


def save(result: Dict[str, Any], images: List[np.ndarray], annotated: List[np.ndarray]) -> Path:
    rid = result["record_id"]
    d = record_dir(rid)
    d.mkdir(parents=True, exist_ok=True)

    for i, img in enumerate(images):
        cv2.imwrite(str(d / f"src{i}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    for i, img in enumerate(annotated):
        cv2.imwrite(str(d / f"src{i}_annotated.png"), img)
    if annotated:
        from .render import thumbnail

        cv2.imwrite(str(d / "thumb.jpg"), thumbnail(annotated[0]), [cv2.IMWRITE_JPEG_QUALITY, 85])

    with open(d / "result.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)

    s = result["summary"]
    with _DB_LOCK, sqlite3.connect(_db_path()) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO records VALUES (?,?,?,?,?,?,?,?,?)",
            (rid, result["created_at"], s["level"], "-".join(s["level_range"]), s["i_mold"],
             ",".join(sorted({i["modality"] for i in result["inputs"]})),
             len(result["instances"]), s["confidence"], str(d)),
        )
    return d


def list_records(limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    with _DB_LOCK, sqlite3.connect(_db_path()) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM records ORDER BY created_at DESC LIMIT ? OFFSET ?", (limit, offset)
        ).fetchall()
    return [dict(r) for r in rows]


def count_records() -> int:
    with _DB_LOCK, sqlite3.connect(_db_path()) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM records").fetchone()[0])


def load(record_id: str) -> Optional[Dict[str, Any]]:
    path = record_dir(record_id) / "result.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def delete(record_id: str) -> bool:
    import shutil

    d = record_dir(record_id)
    if d.exists():
        shutil.rmtree(d)
    with _DB_LOCK, sqlite3.connect(_db_path()) as conn:
        cur = conn.execute("DELETE FROM records WHERE record_id=?", (record_id,))
        return cur.rowcount > 0


def add_feedback(record_id: str, instance_id: Optional[int], kind: str, comment: str = "") -> None:
    with _DB_LOCK, sqlite3.connect(_db_path()) as conn:
        conn.execute(
            "INSERT INTO feedback (record_id, instance_id, kind, comment, created_at) VALUES (?,?,?,?,?)",
            (record_id, instance_id, kind, comment, datetime.now(CST).isoformat(timespec="seconds")),
        )
