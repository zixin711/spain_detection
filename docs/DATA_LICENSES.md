# 数据集出处与许可登记

产品发布前必须核对此表，CC BY 4.0 的数据集要求在产品中保留署名。
`scripts/download_datasets.py` 会在每个数据集目录写 `manifest.json` 记录实际下载信息。

| 数据集 | 出处 | 许可 | 署名要求 | 用途 | 状态 |
|---|---|---|---|---|---|
| mould-detection-V3（984 图 / 1 类 mould） | https://universe.roboflow.com/mould-detection/mould-detection-v3 | CC BY 4.0 ✅已核实 | 需署名 | 白光正样本主源 | 待下载（需 Private API Key） |
| mold-detection-project（475 图 / 1 类 mold，索尔福德大学） | https://universe.roboflow.com/university-of-salford-d3dwy/mold-detection-project | CC BY 4.0 ✅已核实 | 需署名 | 白光正样本 | 待下载 |
| building-defect-on-walls（472 图 / 5 类） | https://universe.roboflow.com/builddef2/building-defect-on-walls | CC BY 4.0 ✅已核实 | 需署名 | mold + damp_stain + substrate_damage | 待下载 |
| damp wall 系列 | https://universe.roboflow.com/search?q=class%3Adamp+wall | 逐项目核对 | 逐项目核对 | 负类 damp_stain | 待挑选 |
| Kaggle mold-detection-dataset | https://www.kaggle.com/datasets/quasaronrails/mold-detection-dataset | Kaggle 条款 | 待核对 | 白光正样本 | 待下载 |
| BD3（完整） | https://github.com/Praveenkottari/BD3-Dataset · https://dl.acm.org/doi/10.1145/3671127.3698789 | 仓库未声明，需向作者确认 | 需引用论文 | substrate_damage + 负样本 | 需联系作者 |
| BD3 样例 25 张 | 同上仓库 `sample images/class_images` | 未声明 | — | **仅冒烟测试，不得用于发布模型** | 已下载 |
| UAV 建筑表面缺陷 | https://www.nature.com/articles/s41597-025-06318-5 | 开放获取，按论文声明 | 需引用论文 | 补充 | 待下载 |
| 自采紫外数据 | 本项目自采 | 自有 | — | 紫外专家训练 | 待采集 |
| 配对标定子集 | 本项目自采 | 自有 | — | 阈值标定 | 待采集 |

## 产品内署名文案（示例）

> 本产品的检测模型部分基于以下开放数据集训练：mould-detection-V3、mold-detection-project、
> building-defect-on-walls（均来自 Roboflow Universe，CC BY 4.0）。
