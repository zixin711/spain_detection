# 示例输出

这些示例由 `data/paired_calib/` 里的 **L3 真实配对样张**生成。

| 文件 | 说明 |
|---|---|
| `annotated_white.jpg` | 白光图的标注结果：半透明掩膜 + 轮廓 + 编号 + 右上角等级角标 + 左上角光源标记 |
| `annotated_uv.jpg` | 紫外图的标注结果（可以直观看到霉斑在 395nm 下是**暗区**，不是亮的荧光） |
| `result_white_only.json` | 仅白光输入的完整返回（`channels.H` 为 null，`confirmation_hint` 提示补紫光照） |
| `result_uv_only.json` | 仅紫外输入的完整返回（`channels.L` 为 null，权重重归一化成 H .53 / M .47） |
| `result_paired_L3.json` | 双模态输入的完整返回（三个通道都有值，用原始权重 .40/.35/.25） |

重点看 JSON 里这几个字段，它们体现了"按可用证据分级"的设计：

```
summary.channels          缺失的通道是 null，不是 0
summary.channel_weights   按模态重归一化后的权重
summary.level_range       该模态能确定的等级区间
summary.needs_confirmation + confirmation_hint   补拍引导
summary.rules_hit         命中了哪几条硬规则，便于排查
summary.index_level       I_mold 指数对应的等级，用于交叉校验
```
