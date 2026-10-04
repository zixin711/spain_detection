# GPU 服务器上的安装与训练

面向 Ubuntu + NVIDIA 双卡（RTX 3090 24GB / RTX PRO 5000 Blackwell 48GB）。
每一步都给了"怎么验证这步成了"，别一路装到底最后发现卡不认。

全流程速查（细节见下面各节）：

```bash
# 一次性
sudo apt install -y python3.11 python3.11-venv python3-pip libgl1 libglib2.0-0 fonts-noto-cjk
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt -r requirements-train.txt
python scripts/check_env.py --train          # 自检，有问题会直接告诉你怎么修

# 数据
export ROBOFLOW_API_KEY=你的private key
python scripts/download_datasets.py
python scripts/prepare_dataset.py

# 训练
python scripts/train.py --domain white --device 1
python scripts/train.py --domain uv    --device 0     # 需要先自采紫外数据

# 评估与导出
python scripts/evaluate.py --domain white
CUDA_VISIBLE_DEVICES=1 python scripts/export_model.py --domain white --formats onnx engine
CUDA_VISIBLE_DEVICES=0 python scripts/export_model.py --domain white --formats engine

# 上线
CUDA_VISIBLE_DEVICES=0 python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## 一、驱动与 CUDA

Blackwell（RTX PRO 5000，compute capability **sm_120**）需要较新的驱动，
R570 分支起支持，生产上建议用 R580 分支。3090（sm_86）同一套驱动兼容。

```bash
nvidia-smi
```

应该同时列出两张卡。关注三个数字：

```
Driver Version: 580.xx        ← 低于 570 先升级驱动
CUDA Version: 12.8            ← 这是驱动支持的上限，不是你装的 CUDA Toolkit
两张卡都在列表里
```

驱动没装或版本太低：

```bash
sudo apt install -y nvidia-driver-580        # 或 ubuntu-drivers autoinstall
sudo reboot
```

**不需要单独装 CUDA Toolkit。** PyTorch 的 wheel 自带运行时，只要驱动够新就行。
只有要自己编译 CUDA 扩展或用 TensorRT 命令行工具时才需要完整 Toolkit。

---

## 二、Python 环境与 PyTorch

```bash
sudo apt install -y python3.11 python3.11-venv python3-pip libgl1 libglib2.0-0 fonts-noto-cjk

cd /path/to/spain
python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
```

`fonts-noto-cjk` 必装，否则标注图和 PDF 报告里的中文会变成方块。
`libgl1` / `libglib2.0-0` 是 OpenCV 的运行时依赖，服务器版 Ubuntu 通常没有。

### PyTorch 一定要单独装

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

**这一步是整套流程里最容易出错的地方。** 直接 `pip install torch` 装到的可能是
CPU 版或 cu118/cu126 版，在 sm_120 的卡上一跑就报：

```
CUDA error: no kernel image is available for execution on the device
```

装完立刻验证：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); \
print([(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())])"
```

期望输出类似：

```
2.x.x  12.8  True
[(0, 'NVIDIA GeForce RTX 3090', (8, 6)), (1, 'NVIDIA RTX PRO 5000 Blackwell', (12, 0))]
```

`torch.version.cuda` 必须是 **12.8 或更高**，`(12, 0)` 就是 sm_120。

### 其余依赖

```bash
pip install -r requirements.txt          # 服务本体
pip install -r requirements-train.txt    # ultralytics / roboflow / sklearn 等
pip install -r requirements-gpu.txt      # onnxruntime-gpu，可选
```

### 自检

```bash
python scripts/check_env.py --train
```

一次性检查 Python 版本、依赖、中文字体、存储目录、驱动、每张卡的架构与 torch 的匹配性、
训练依赖、数据集规模。它会**真的跑一次 GPU 矩阵乘**，而不是只看 `is_available()`——
架构不匹配时 `is_available()` 会返回 True，一算才报错。

有阻塞性问题会打 `✗` 并给出修复命令，退出码非 0，方便串在部署脚本里。

---

## 三、准备数据

```bash
export ROBOFLOW_API_KEY=你的private key      # Private Key，不是 rf_ 开头的 Publishable Key
python scripts/download_datasets.py
```

三个 Roboflow 数据集共约 4900 张图（详见 `docs/DATA_DOWNLOAD.md`）。
下完会打印每个数据集实际下到多少张，**0 张会明确报失败**。

```bash
python scripts/prepare_dataset.py
```

清洗管线：格式统一 → pHash 跨数据集去重 → 模糊/曝光过滤 → 打模态标签 →
类名映射 → 框转掩膜 → 按场景分组 8:1:1 划分。产出 `data/processed/{white,uv}/`。

几个实用参数：

```bash
python scripts/prepare_dataset.py --limit 400    # 只处理 400 张，快速验证流程（几十秒）
python scripts/prepare_dataset.py --no-sam       # 跳过 SAM2，检测框退化成矩形掩膜
python scripts/prepare_dataset.py --dry-run      # 只统计不落盘
```

### 关于 SAM2

公开数据集给的是**检测框**，本项目要的是**分割掩膜**。装了 ultralytics 后
清洗脚本会自动下载 `sam2_b.pt`，用框作为 prompt 生成掩膜，并按 IoU 与面积合理性过滤。

不装 SAM2 也能训，但框会退化成矩形掩膜，分割精度会明显打折。**建议装**：
第一次运行会自动下模型权重（几百 MB），之后走本地缓存。

### 期望看到什么

```
清洗统计：
  modality_declared_white 4800
  ann_box_rect            12000
  duplicate               60
  blurry                  40

white 数据集 → data/processed/white
  划分：{'train': 3900, 'val': 480, 'test': 490}
  各类实例数：{'mold_black': 9000, 'substrate_damage': 1500, 'damp_stain': 400}
```

**关键是"各类实例数"不能是空的。** 如果显示"全是背景负样本"，说明类名映射没生效，
去 `configs/label_map_white.yaml` 的 `source_map` 里补上对应的原始类名。

### 人工复核（可选）

```bash
python scripts/review_server.py     # http://127.0.0.1:8010
```

给清洗时挑出来的没标签的图快速打标，键盘 1-9 选类别，← → 翻页。

---

## 四、训练白光专家

```bash
python scripts/train.py --domain white --device 1
```

开训前脚本会先做一轮检查并打印数据集规模，**发现数据为空、没有标注实例、
验证集为空这三种情况会直接拒绝开始**，免得跑两小时才发现白跑。想只做检查：

```bash
python scripts/train.py --domain white --dry-run
```

### 参数与显存

| 参数 | RTX PRO 5000 48GB | RTX 3090 24GB | 说明 |
|---|---|---|---|
| `--model` | `yolo11l-seg.pt` | `yolo11m-seg.pt` | 显存不够就降一档 |
| `--imgsz` | 1024 | 1024 | 霉斑小目标多，比默认 640 收益明显 |
| `--batch` | 16 | 8 | OOM 就减半 |
| `--epochs` | 300 | 300 | 配合 `patience=50` 早停 |
| `--cache` | `disk` | `disk` | `ram` 更快但吃内存约「图数 × 3MB」，4900 张要 ~15GB |
| `--workers` | 8 | 8 | CPU 核少就调低 |

固定的超参（写在 `scripts/train.py` 里）：AdamW、`lr0=0.001`、`weight_decay=0.0005`、
warmup 3 epoch、`cos_lr`、`close_mosaic=15`、`copy_paste=0.3`、AMP。

**增强策略是分域的**，这点很重要：

- 白光：色相抖动幅度很小（`hsv_h=0.010`）—— 霉斑的颜色是关键判据，抖大了等于毁掉特征
- 紫外：**完全不做色相抖动**（`hsv_h=0.0`），改做亮度/曝光抖动，模拟紫光手电的距离衰减
- 两者都**不做上下翻转** —— 霉菌有重力和结露方向的先验

### 时间预估

约 4000 张训练图、`yolo11l-seg`、imgsz 1024、batch 16，在 48GB 卡上大致
**每 epoch 一两分钟，300 epoch 十小时上下**。这只是量级参考，实际取决于卡的状态、
数据加载速度和是否开缓存——**跑起来看前几个 epoch 的实际耗时再推算**，
`patience=50` 通常会让它在 300 epoch 之前就停。

建议挂在 tmux / screen 里跑：

```bash
tmux new -s train-white
python scripts/train.py --domain white --device 1
# Ctrl+B 然后按 D 脱离，ssh 断了也不影响
tmux attach -t train-white
```

### 中断续训

```bash
python scripts/train.py --domain white --device 1 --resume
```

### 训练产物

- 过程与曲线：`runs/seg/seg_white/`
- 最优权重：`runs/seg/seg_white/weights/best.pt`
- 脚本会自动把 best.pt 复制到 `models/seg_white.pt`，**服务重启后自动生效**

---

## 五、训练紫外专家

紫外数据**公开数据集里没有，必须自采**。采集协议见 `docs/UV_CAPTURE_PROTOCOL.md`。

在拿到自采数据之前这一步跑不了，脚本会明确告诉你数据集还没生成。
数据放到 `data/raw/uv_self/`（YOLO 格式，images/labels 同级）后：

```bash
python scripts/prepare_dataset.py --modality uv
python scripts/train.py --domain uv --device 0
```

紫外专家是 6 类（按荧光形态而不是表型分类），其中 `fluor_artifact` 是最重要的负类——
荧光增白剂、塑料件、涂料助剂在 395nm 下都会亮，不建这个类误报率会高到没法用。

再强调一次实测结论：**395nm 下霉菌不发黄绿荧光，而是比墙面更暗**，
菌落前沿是亮白蕾丝状细丝。标注时按"暗斑本体 + 亮丝前沿"来圈。

---

## 六、光源判别器（可选）

规则版（统计量 + 线性打分）零训练成本，在 25 张真实白光样张上 25/25 全对。
积累了足够样本后可以换成逻辑回归：

```bash
# 数据放到 data/modality_clf/{white,uv}/
python scripts/train_modality_clf.py                  # 先看规则版准确率
python scripts/train_modality_clf.py --sklearn --write  # 训练并保存 models/modality_clf.json
```

保存后 `app/modality.py` 会自动优先加载它，接口不变。

---

## 七、评估

```bash
python scripts/evaluate.py --domain white
```

输出分割 mAP、以及**负样本误报清单**（在纯负样本图上的假阳性，按置信度排序，
写到 `runs/false_positives_white.json`）。这份清单是数据回流的抓手——
误报最凶的那些场景，就是下一轮该补数据的地方。

验收目标：

| 指标 | 目标 |
|---|---|
| 白光 mask mAP50 | ≥ 0.60 |
| 紫外 mask mAP50 | ≥ 0.55 |
| 纯负样本假阳性率 | ≤ 8%（传统 CV 基线实测约 40%，这是训模型的主要动机） |
| 等级一致性加权 Kappa | ≥ 0.7，低估 ≥2 级的漏判率 ≤ 2% |

等级一致性需要人工标注的对照表（两列 `human_level,system_level`）：

```bash
python scripts/evaluate.py --grading data/eval/level_labels.csv
```

---

## 八、导出推理引擎

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/export_model.py --domain white --formats onnx engine
CUDA_VISIBLE_DEVICES=0 python scripts/export_model.py --domain white --formats engine
```

**TensorRT 引擎不跨 GPU 架构通用。** 两张卡必须各 build 一份，
脚本会把架构写进文件名（`seg_white_sm86.engine` / `seg_white_sm120.engine`），
服务启动时按实测卡自动挑选。

加载优先级：`_sm{arch}.engine` → `.engine` → `.onnx` → `.pt` → 传统 CV 基线。
ONNX 保留作为换卡、CI 和没有 TensorRT 环境时的兜底。

---

## 九、上线

权重就位后重启服务即可，代码不用动：

```bash
ls models/
# seg_white.pt  seg_white.onnx  seg_white_sm86.engine  seg_white_sm120.engine

CUDA_VISIBLE_DEVICES=0 python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
curl -s localhost:8000/api/health | python -m json.tool
```

`/api/health` 会报出实际加载了哪个后端：

```json
"models": {"white": "ultralytics:seg_white_sm86.engine", "uv": "baseline (no weights)"}
```

看到 `baseline (no weights)` 就说明权重没被找到，检查 `models/` 下的文件名。

开机自启的 systemd unit 见 `docs/DEPLOY.md`。

---

## 十、训练完别忘了重新标定阈值

模型换了之后，检出的面积口径会变，分级阈值要跟着重标：

```bash
python scripts/calibrate_grading.py            # 先看拟合效果
python scripts/calibrate_grading.py --write    # 回写 configs/grading.yaml
```

当前阈值是用 5 组 L1~L5 配对样张标的，每级只有 1 组，统计意义很弱。
样本补到每级 20~30 组后再跑一次，并把 `configs/grading.yaml` 里的
`index_check.calibrated` 改成 `true`（届时指数校验才允许按实际差距抬级，
现在为了安全最多只抬 1 级）。

---

## 十一、两张卡怎么安排

架构不同（sm_86 vs sm_120），**混合 DDP 会被慢卡拖住**，同步开销还大。
并行跑两个独立实验的吞吐明显更高：

| 卡 | 用途 |
|---|---|
| RTX PRO 5000 48GB（device 1） | 主训练：白光大模型、大 batch 大分辨率、长实验 |
| RTX 3090 24GB（device 0） | 紫外专家、消融实验；训练完成后常驻跑推理服务 |

```bash
tmux new -s white -d 'python scripts/train.py --domain white --device 1'
tmux new -s uv    -d 'python scripts/train.py --domain uv    --device 0'
watch -n 5 nvidia-smi
```

什么时候才该上 DDP：等你有两张**相同架构**的卡时，用
`--device 0,1`，ultralytics 会自动起 DDP。

---

## 十二、常见报错

| 报错 | 原因 | 处理 |
|---|---|---|
| `no kernel image is available for execution on the device` | torch 的 CUDA 版本低于显卡架构 | 重装 cu128：`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --force-reinstall` |
| `CUDA out of memory` | batch 或 imgsz 太大 | `--batch` 减半；还不行就 `--imgsz 800` 或换小一档模型 |
| `ImportError: libGL.so.1` | 服务器版 Ubuntu 缺 OpenCV 运行时依赖 | `sudo apt install -y libgl1 libglib2.0-0` |
| 训练前检查报"一个标注实例都没有" | 类名没被 source_map 覆盖 | 看 `data/raw/*/data.yaml` 里的原始类名，补进 `configs/label_map_white.yaml` |
| 训练前检查报"验证集为空" | 样本太少，按场景分组后没分到验证集 | 补数据后重跑 prepare_dataset.py |
| `prepare_dataset` 报"有 N 个文件删不掉" | 上一轮产物没清干净，新旧数据会混 | `rm -rf data/processed` 后重跑 |
| 内存被吃光 / OOM killer | `--cache ram` 在大数据集上要十几 GB | 改 `--cache disk` 或 `--cache none` |
| mAP 一直是 0 | 标签坐标不对，或类别索引错位 | 抽查 `data/processed/white/labels/train/*.txt`，坐标应当是 0~1 的归一化值 |
| 标注图中文变方块 | 缺 CJK 字体 | `sudo apt install fonts-noto-cjk` |
| ssh 断开训练就停 | 没挂后台 | 用 tmux / screen，或做成 systemd service |
