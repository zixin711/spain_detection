# 数据集获取指引

`python scripts/download_datasets.py` 会自动下载所有不需要凭据的数据源，
需要凭据或需要人工申请的会打印指引。本文件是完整清单与手工步骤。

---

## 已自动下载到本地

| 数据集 | 位置 | 数量 | 说明 |
|---|---|---|---|
| `bd3_samples` | `data/raw/bd3_samples/` | 25 张 | BD3 仓库公开的每类 5 张样例（Algae / stain / peeling / spalling / normal）。**仅够冒烟测试**，仓库未声明 license，不要用它训练发布版模型。 |

---

## 需要免费 API key：Roboflow Universe

### 第一步 · 拿 Private API Key

1. 打开 <https://app.roboflow.com> 注册（邮箱或 Google 账号，免费）
2. 注册时会让你建一个 workspace，随便起个名字，类型选 Public / 免费的那档就行
3. 打开 <https://app.roboflow.com/settings/api>
4. 页面上有**两个 key，别拿错**：

   | Key 类型 | 长什么样 | 用途 | 能不能下数据集 |
   |---|---|---|---|
   | **Private API Key** | 一串随机字符 | 服务端操作，能读写工作区的数据与模型 | **✅ 就用这个** |
   | Publishable API Key | `rf_` 开头 | 浏览器端推理，可以公开嵌在网页里 | ❌ 只能推理和下模型权重 |

5. 点 Private API Key 旁边的 **Copy**

> Private Key 等同于密码，别提交进 git。万一泄露了，同一个页面有 **Roll API Key** 可以作废重发。

### 第二步 · 配置并下载

```bash
export ROBOFLOW_API_KEY=粘贴你的 private key      # 想长期生效就写进 ~/.zshrc 或 ~/.bashrc
pip install roboflow

python scripts/download_datasets.py --only mould_detection_v3 salford_mold_detection building_defect_on_walls
```

脚本会自动取每个项目的最新版本，下到 `data/raw/<数据集名>/`，并写一份 `manifest.json` 记录来源与许可。
成功时会打印实际下到了多少张图；如果是 0 张会明确报失败，不会假装成功。

几个常见情况：

- 填了 `rf_` 开头的 key → 脚本直接告诉你拿错了
- 某个数据集已经下过 → 自动跳过，加 `--force` 可以强制重下
- 下完 0 张图 → 通常是该项目在 Universe 上只公开了模型没公开数据集，
  或者你的 workspace 还没接受它的许可条款（先在网页上手动点一次 Download 就会弹条款）

### 下哪三个（都是 CC BY 4.0，可商用，需署名）

| 数据集 | 规模 | 类别 | 在本项目里的用途 |
|---|---|---|---|
| [mould-detection-V3](https://universe.roboflow.com/mould-detection/mould-detection-v3) | 984 图 / 7 版本 | `mould` | **正样本主源，必下** |
| [mold-detection-project](https://universe.roboflow.com/university-of-salford-d3dwy/mold-detection-project)（曼彻斯特索尔福德大学） | 475 图 / 2 版本 | `mold` | 补充正样本，室内场景 |
| [building-defect-on-walls](https://universe.roboflow.com/builddef2/building-defect-on-walls) | 472 图 / 3 版本 | `crack` `mold` `peeling_paint` `stairstep_crack` `water_seepage` | **性价比最高**，一份数据同时喂三种类别 |

第三个之所以最值：它的五个类正好覆盖了本项目三种角色——

```
mold                                  → mold_black        正样本
water_seepage                         → damp_stain        负类，专门压水渍误报
peeling_paint / crack / stairstep_crack → substrate_damage  L5 的结构破坏判据
```

映射规则写在 `configs/label_map_white.yaml` 的 `source_map` 里，清洗脚本会自动套用。

> **注意**：`substrate_damage` 只有在**同时检出霉菌**时才触发 L5。一面干净墙上的裂缝是建筑缺陷，
> 不是霉菌等级，系统会把它作为附带发现报出来，但不会因此判 L5。

### 如果脚本跑不通

网页手工下载完全等价：打开数据集页面 → 右上角 **Download this Dataset** → 格式选
**YOLOv8** → 下载 zip → 解压到 `data/raw/<数据集名>/`（保持 `train/images`、`train/labels` 的目录结构）。

### 还想再多要一点数据

`damp wall` 系列没有稳定的单一项目 id，需要先在
<https://universe.roboflow.com/search?q=class%3Adamp+wall> 搜索，挑到合适的项目后，
把它的 workspace / project 按 `configs/datasets.yaml` 里 `mould_detection_v3` 的格式补进去
（`version` 直接写 `latest`），再跑脚本。挑的时候逐个核对许可，不是所有项目都是 CC BY 4.0。

### 许可与署名

CC BY 4.0 允许商用，但要求署名。用了哪些数据集请登记到 `docs/DATA_LICENSES.md`，
并在产品的"关于"页保留出处，例如：

> 本产品的检测模型部分基于以下开放数据集训练：mould-detection-V3、mold-detection-project、
> building-defect-on-walls（均来自 Roboflow Universe，CC BY 4.0）。

## 需要 Kaggle 凭据

```bash
# 1. https://www.kaggle.com/settings → API → Create New Token，下载 kaggle.json
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
pip install kagglehub

# 2. 下载
python scripts/download_datasets.py --only kaggle_mold_detection
```

---

## 需要联系作者：BD3 完整数据集

完整 BD3（3,965 图：Algae 624 / Stain 521 / Peeling 520 / Spalling 500 / Normal 600 等）
**没有在 GitHub 上提供下载链接**，仓库里只有每类 5 张样例。获取方式：

1. 阅读 BuildSys'24 论文 <https://dl.acm.org/doi/10.1145/3671127.3698789>，按文中联系方式向作者申请
2. 或在 <https://github.com/Praveenkottari/BD3-Dataset> 提 issue 询问数据分发链接

拿到压缩包后解压到 `data/raw/bd3_full/`，清洗管线会自动识别。
它的 `Peeling` 与 `Spalling` 是本项目 `substrate_damage` 类（L5 判据）最好的来源，
`Normal` 与 `Stain` 是压误报用的负样本。

---

## 需要按论文指引获取：无人机建筑表面缺陷数据集

Scientific Data 论文 <https://www.nature.com/articles/s41597-025-06318-5>，
按论文 **Data Availability** 段落给出的仓库链接下载，解压到 `data/raw/uav_defects/`。

---

## 必须自采：395nm 紫外数据

**公开数据集里不存在紫外荧光的墙面霉菌数据，这部分只能自己拍。** 好消息是不需要成对拍摄。

采集协议见 `docs/UV_CAPTURE_PROTOCOL.md`。目标规模：

| 类别 | 目标张数 |
|---|---|
| `fluor_point` 点状微弱荧光 | ≥ 200 |
| `fluor_local` 局域亮荧光 | ≥ 200 |
| `fluor_band` 条带/块状强荧光 | ≥ 200 |
| `fluor_saturated` 大面积饱和荧光 | ≥ 200 |
| `fluor_chaotic` 杂乱深度荧光 | ≥ 200 |
| `fluor_artifact` 荧光干扰负类 | ≥ 300 |

拍好后放到 `data/raw/uv_self/`，标注成 YOLO-seg 格式（labels 目录同级）。

另外需要 **150~300 组白光/紫外配对样本** 放到 `data/paired_calib/{white,uv}/`（同名文件配对），
只用于标定两套面积阈值的映射关系，不参与训练。跑 `python scripts/calibrate_grading.py --write` 完成标定。

---

## 关于网上那些"墙面霉菌数据集"仓库

GitHub 上能搜到若干名为 *"墙面裂缝 霉菌 渗水 检测数据集 VOC YOLO 格式"* 的仓库
（如 `lonlonago/*` 系列）。这些仓库里**只有几张预览拼图和一个 README**，
真实数据需要到第三方付费渠道购买，不是开源数据集。已在本项目中排除，不建议使用：
来源与授权不清，用于训练会给后续商用埋雷。

---

## 处理流程

```bash
python scripts/download_datasets.py          # 下载
python scripts/prepare_dataset.py            # 清洗 → data/processed/{white,uv}/
python scripts/review_server.py              # 人工复核没有标签的图（可选）
python scripts/train.py --domain white --device 0
python scripts/train.py --domain uv    --device 1
```
