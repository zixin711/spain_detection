#!/usr/bin/env bash
# 一键起服务：建虚拟环境 → 装依赖 → 启动。重复执行是安全的。
#   ./run.sh              默认 8000 端口
#   ./run.sh 8080         指定端口
set -euo pipefail
cd "$(dirname "$0")"

PORT="${1:-8000}"
PY="${PYTHON:-python3}"

if [ ! -d .venv ]; then
  echo "▸ 建虚拟环境 .venv"
  "$PY" -m venv .venv
fi
source .venv/bin/activate

if ! python -c "import fastapi, cv2, yaml" 2>/dev/null; then
  echo "▸ 安装依赖（只有第一次会花时间）"
  pip install --quiet --upgrade pip
  pip install --quiet -r requirements.txt
fi

# SQLite 不能放在网络盘/挂载盘上，检测一下，必要时换到本地
: "${MOLD_STORAGE_DIR:=$PWD/storage}"
export MOLD_STORAGE_DIR
mkdir -p "$MOLD_STORAGE_DIR"
if ! python - <<'PYCHK' 2>/dev/null
import os, sqlite3, pathlib
p = pathlib.Path(os.environ["MOLD_STORAGE_DIR"]) / ".probe.db"
sqlite3.connect(p).execute("create table if not exists t(x)")
p.unlink(missing_ok=True)
PYCHK
then
  export MOLD_STORAGE_DIR="$HOME/.mold-detect/storage"
  mkdir -p "$MOLD_STORAGE_DIR"
  echo "▸ 项目目录不支持 SQLite，存储改到 $MOLD_STORAGE_DIR"
fi

export PYTHONUNBUFFERED=1
echo "▸ 启动（Ctrl+C 退出）"
exec python -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
