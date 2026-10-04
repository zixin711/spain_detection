#!/usr/bin/env python3
"""极简本地复核页：给清洗管线挑出来的图快速打标（键盘 1-9 选类别，← → 翻页）。

    python scripts/review_server.py            # http://127.0.0.1:8010
结果写入 data/review/labels.json，prepare_dataset.py 下次运行时会读取。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import DATA_DIR, labels_uv, labels_white  # noqa: E402

REVIEW = DATA_DIR / "review"
QUEUE = REVIEW / "queue.json"
LABELS = REVIEW / "labels.json"

PAGE = """<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>样本复核</title><style>
body{background:#111;color:#eee;font-family:system-ui,-apple-system,"PingFang SC",sans-serif;margin:0;padding:20px}
img{max-width:100%;max-height:62vh;display:block;margin:0 auto;border-radius:8px}
.bar{display:flex;gap:8px;flex-wrap:wrap;justify-content:center;margin-top:16px}
button{background:#222;border:1px solid #444;color:#eee;border-radius:8px;padding:10px 14px;font-size:15px;cursor:pointer}
button:hover{background:#2c2c2c}
.meta{text-align:center;color:#888;font-size:13px;margin:10px 0}
kbd{background:#333;border-radius:4px;padding:1px 5px;font-size:12px}
</style></head><body>
<div class="meta" id="meta"></div><img id="img" alt=""><div class="bar" id="bar"></div>
<div class="meta">按数字键选类别，<kbd>←</kbd><kbd>→</kbd> 翻页，<kbd>0</kbd> 丢弃</div>
<script>
const CLASSES = __CLASSES__;
let items = [], labels = {}, i = 0;
async function boot(){
  items = await (await fetch('/queue')).json();
  labels = await (await fetch('/labels')).json();
  const bar = document.getElementById('bar');
  CLASSES.forEach((c, k) => {
    const b = document.createElement('button');
    b.textContent = `${k+1}. ${c.cn}`;
    b.onclick = () => mark(c.name);
    bar.appendChild(b);
  });
  const skip = document.createElement('button');
  skip.textContent = '0. 丢弃'; skip.onclick = () => mark('__discard__'); bar.appendChild(skip);
  render();
}
function render(){
  if(!items.length){document.getElementById('meta').textContent='队列为空';return;}
  i = Math.max(0, Math.min(i, items.length-1));
  const it = items[i];
  document.getElementById('img').src = '/img?path=' + encodeURIComponent(it.image);
  const done = Object.keys(labels).length;
  document.getElementById('meta').textContent =
    `${i+1} / ${items.length}　已标 ${done}　原因：${it.reason}　当前：${labels[it.image] || '未标'}`;
}
async function mark(name){
  const it = items[i]; labels[it.image] = name;
  await fetch('/label', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({image: it.image, label: name})});
  i++; render();
}
document.addEventListener('keydown', e => {
  if(e.key === 'ArrowRight'){i++; render();}
  else if(e.key === 'ArrowLeft'){i--; render();}
  else if(e.key === '0'){mark('__discard__');}
  else if(/^[1-9]$/.test(e.key)){const c = CLASSES[+e.key-1]; if(c) mark(c.name);}
});
boot();
</script></body></html>"""


def main() -> int:
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from urllib.parse import parse_qs, urlparse

    if not QUEUE.exists():
        print(f"没有复核队列：{QUEUE}\n先跑 python scripts/prepare_dataset.py")
        return 1

    classes = [{"name": m["name"], "cn": m.get("cn", m["name"])}
               for _, m in sorted(labels_white()["classes"].items())]
    classes += [{"name": m["name"], "cn": m.get("cn", m["name"])}
                for _, m in sorted(labels_uv()["classes"].items())][:2]
    page = PAGE.replace("__CLASSES__", json.dumps(classes, ensure_ascii=False))

    def read_labels():
        return json.loads(LABELS.read_text(encoding="utf-8")) if LABELS.exists() else {}

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/":
                self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
            elif u.path == "/queue":
                self._send(200, QUEUE.read_bytes())
            elif u.path == "/labels":
                self._send(200, json.dumps(read_labels(), ensure_ascii=False).encode("utf-8"))
            elif u.path == "/img":
                p = Path(parse_qs(u.query).get("path", [""])[0])
                if p.exists() and DATA_DIR.resolve() in p.resolve().parents:
                    self._send(200, p.read_bytes(), "image/jpeg")
                else:
                    self._send(404, b"{}")
            else:
                self._send(404, b"{}")

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n) or b"{}")
            labels = read_labels()
            labels[data["image"]] = data["label"]
            LABELS.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
            self._send(200, b'{"ok":true}')

        def log_message(self, *a):
            pass

    print("复核页：http://127.0.0.1:8010　（Ctrl+C 退出）")
    HTTPServer(("127.0.0.1", 8010), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
