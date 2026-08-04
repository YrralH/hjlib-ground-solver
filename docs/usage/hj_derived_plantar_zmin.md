# HJ 派生的脚底 ZMin 地面高度

当调用方已经有同一序列的左右脚掌逐帧最低高度，并希望采用绝对脚底 `zmin`
作为水平地面高度代理时，使用：

```python
from hjlib_ground_solver import estimate_hj_derived_plantar_zmin_ground

result = estimate_hj_derived_plantar_zmin_ground(
    left_per_frame_minimum_height_in_meter=left_height,
    right_per_frame_minimum_height_in_meter=right_height,
)

ground_height = result.ground_height_in_meter
assert result.provenance == 'hj_derived_nonofficial'
```

输入必须是同 shape、同 dtype 的一维 NumPy 数组；支持 `float32` / `float64`，
要求 `T >= 1` 且全部有限。函数不修改输入。

返回值除高度外还包含：

- 左右脚各自的序列最小高度；
- 按“高度、左脚优先、较早 track frame”决定的 selected side/frame；
- pooled 左右脚 track 中严格等于全局最小值的样本数；
- 不可由调用方修改的 provenance `hj_derived_nonofficial`。

这里的 frame index 是输入 track 内的局部索引。若 track 是裁剪窗口，调用方必须
自行映射到 native frame；函数不会假定 AMASS、FPS 或裁剪策略。

该函数只做绝对 `min`，不会偷偷加入 percentile、outlier peeling、速度门、脚趾
offset 或零值 fallback。它也不推断左右脚 contact。

## 重要语义

这个结果是 HJ 根据人体拟合 mesh 派生的代理，不是 AMASS 官方 ground 或 contact
标注。后续生成 foot-contact 标签时，应另行结合离地高度、脚底速度、持续时间及
不确定状态；不能仅因某帧等于该高度就称为官方接触真值。

## 如何选择

| 手里的输入 / 目的 | 使用入口 |
| --- | --- |
| 左右 plantar `(T,)` 高度，明确采用非官方 absolute zmin | `estimate_hj_derived_plantar_zmin_ground` |
| full mesh，只想计算 lower-envelope 候选 | `compute_per_frame_mesh_minimum_height` + `summarize_mesh_lower_envelope` |
| plantar 高度和速度，想复现 common-domain HuMoR-style comparator | `estimate_static_foot_plantar_humor_baseline` |
| 想获得 AMASS 官方 ground/contact | 当前不存在该入口；不要使用 derived API 冒充 |
