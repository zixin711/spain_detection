# 部署与运行

三种场景，按你当下的需要挑一个看：

| 场景 | 看哪节 | 大概要多久 |
|---|---|---|
| 现在就想在 Mac 上拍照试试效果 | [一、Mac 上跑起来](#一mac-上跑起来最快) | 5 分钟 |
| 在 Ubuntu + 显卡机器上训练模型并长期跑服务 | [二、Ubuntu + NVIDIA GPU](#二ubuntu--nvidia-gpu) | 30 分钟 |
| 打成镜像交付 | [三、Docker](#三docker-部署) | 15 分钟 |

**不需要模型权重也能跑。** 没有训练好的权重时，系统自动降级成传统 CV 基线，
全链路（拍照 → 检测 → 分级 → 建议 → 标注图 → PDF）照常工作，方便先把手机端体验联调通。

---

## 一、Mac 上跑起来（最快）

```bash
cd ~/Desktop/work2026/spain
./run.sh
```

就这一条。脚本会自动建 `.venv`、装依赖、检查存储目录、启动服务。重复执行是安全的。

想换端口：`./run.sh 8080`

如果你想手动来：

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

启动后终端会打印：

```
  墙面霉菌检测服务已启动
  手机访问： http://192.168.1.23:8000   （手机需与本机在同一个 Wi-Fi）
  白光模型： baseline (no weights)
  紫外模型： baseline (no weights)
  存储目录： /Users/david/Desktop/work2026/spain/storage
```

下面还会打一个二维码，**手机相机扫一下直接就打开了**，不用手输 IP。

### 手机怎么用

1. 手机连**同一个 Wi-Fi**（这一步最容易出错，5G 网络下访问不到）
2. 扫二维码，或浏览器输那个 `http://192.168.x.x:8000`
3. 点"拍照 / 选择照片"，会直接调起系统相机
4. 每张照片可以手动标"白光照 / 紫光照"，也可以留"自动判别"
5. 点"开始检测"

**不需要 HTTPS**。页面用 `<input type="file" capture>` 调起系统相机，只有做实时取景
（`getUserMedia`）才强制要证书。

### Mac 上的注意事项

- **不要装 `requirements-gpu.txt`**，`onnxruntime-gpu` 在 macOS 上没有 wheel，装了直接失败
- 标注图和 PDF 里的中文依赖系统字体，macOS 自带 PingFang，开箱即用
- Apple Silicon 上也能训练（`--device mps`），但只适合小规模试验，正式训练还是上 GPU 机器

---

## 二、Ubuntu + NVIDIA GPU

### 1. 系统依赖

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip \
                    libgl1 libglib2.0-0 fonts-noto-cjk
```

`fonts-noto-cjk` 必装，否则标注图和 PDF 报告里的中文会变成方块。

### 2. Python 环境与 PyTorch

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

# ⚠️ 先单独装 torch，按 CUDA 版本选 index-url，不要让 pip 自己解依赖
# RTX PRO 5000 Blackwell 是 sm_120，必须 cu128 及以上；3090(sm_86) 用同一套轮子
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt          # 服务本体
pip install -r requirements-train.txt    # 训练与数据管线
pip install -r requirements-gpu.txt      # GPU 推理加速（可选）
```

验一下卡认出来没有：

```bash
python -c "import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

打印 `no kernel image is available for execution on the device` 就是装成了 cu118/cu121 的轮子，重装 cu128 版本。

### 3. 准备数据并训练

```bash
export ROBOFLOW_API_KEY=你的 private key      # 见 docs/DATA_DOWNLOAD.md
python scripts/download_datasets.py
python scripts/prepare_dataset.py             # → data/processed/{white,uv}/

# 两张卡架构不同，不要混合 DDP，并行跑两个实验吞吐更高
python scripts/train.py --domain white --device 1   # RTX PRO 5000 48GB 主训练
python scripts/train.py --domain uv    --device 0   # 3090 跑紫外专家
```

训完导出推理引擎。**TensorRT 引擎不跨架构通用，两张卡各跑一次**：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/export_model.py --domain white --formats onnx engine  # → seg_white_sm86.engine
CUDA_VISIBLE_DEVICES=1 python scripts/export_model.py --domain white --formats engine       # → seg_white_sm120.engine
```

服务启动时按实测卡的 compute capability 自动挑对应引擎，挑不到退 ONNX，再退 `.pt`，
最后退传统 CV 基线。

### 4. 起服务

```bash
# 绑到 3090，把 PRO 5000 留给训练
CUDA_VISIBLE_DEVICES=0 python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 5. 开机自启（systemd）

`/etc/systemd/system/mold-detect.service`：

```ini
[Unit]
Description=Wall Mold Detection Service
After=network.target

[Service]
Type=simple
User=你的用户名
WorkingDirectory=/opt/spain
Environment="CUDA_VISIBLE_DEVICES=0"
Environment="MOLD_STORAGE_DIR=/var/lib/mold-detect/storage"
Environment="PYTHONUNBUFFERED=1"
ExecStart=/opt/spain/.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now mold-detect
sudo journalctl -u mold-detect -f
```

---

## 三、Docker 部署

需要先装 [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)。

```bash
cd docker
docker compose up -d --build
docker compose logs -f
```

`models/` 与 `configs/` 是只读挂载，**换权重或调阈值不用重建镜像**，重启容器即可：

```bash
docker compose restart
```

镜像里已经装了 `fonts-noto-cjk`，中文渲染开箱可用。

---

## 四、公网访问（可选）

局域网够用就别开公网。确实需要时前置 nginx：

```nginx
server {
    listen 443 ssl;
    server_name mold.example.com;
    ssl_certificate     /etc/letsencrypt/live/mold.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/mold.example.com/privkey.pem;

    client_max_body_size 32m;       # 手机原图较大，必须放开
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 120s;    # 推理排队时不要过早断开
    }
}
```

注意：当前设计是**单机单用户 + 本地历史，没有账号体系**。放到公网前先想清楚谁能访问。

---

## 五、常见问题

| 现象 | 原因与处理 |
|---|---|
| `pip install` 报 onnxruntime-gpu 找不到 | 在 Mac 上装了 `requirements-gpu.txt`，那个只给 NVIDIA 机器用，跳过即可 |
| 启动报 `disk I/O error` | storage 目录在网络盘/挂载盘上，SQLite 不支持。`export MOLD_STORAGE_DIR=~/mold-storage`（`run.sh` 会自动处理） |
| 手机打不开页面 | 九成是没连同一个 Wi-Fi；其次检查 Mac 的防火墙是否拦了 8000 端口 |
| 标注图里中文变方块 | Ubuntu 上没装 CJK 字体：`sudo apt install fonts-noto-cjk` |
| `no kernel image is available` | PyTorch 轮子的 CUDA 版本低于显卡架构，换 cu128 |
| 检测结果全是 UNKNOWN | 看返回里的 `unknown_reasons`，通常是照片过于模糊、墙面占比过低，或光源判别不确定 |
| 紫外照被判成白光 | 前端手动指定光源；再把这张图放进 `data/modality_clf/uv/`，跑 `train_modality_clf.py` 重新标定阈值 |
| 面积占比明显偏大/偏小 | 返回里 `wall_area_fallback: true` 说明墙面分割失败退回了整幅画面，换个角度重拍 |
| iPhone 上传 HEIC 报错 | 确认装了 `pillow-heif`；或在 iPhone 设置里把相机格式改成"兼容性最佳" |
| PDF 报告 503 | 没装 reportlab：`pip install reportlab` |
| 干净墙面被判成有霉 | 传统 CV 基线的已知问题（实测误报率约 40%），训好模型后会大幅下降 |
