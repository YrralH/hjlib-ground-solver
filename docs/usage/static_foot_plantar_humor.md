# Plantar HuMoR-style static-foot baseline

当调用方已经有同一 plantar vertex subset 的左右脚最低高度与 interval median
speed 时，用这个入口做共同观测域的 phase-1 高度聚类：

```python
from hjlib_ground_solver import (
    Static_Foot_Plantar_HuMoR_Config,
    estimate_static_foot_plantar_humor_baseline,
)

config = Static_Foot_Plantar_HuMoR_Config(
    maximum_contact_speed_in_meter_per_second=0.5,
    dbscan_epsilon_in_meter=0.005,
    dbscan_minimum_sample_count=3,
)
result = estimate_static_foot_plantar_humor_baseline(
    left_height_in_meter,                              # (T,)
    right_height_in_meter,                             # (T,)
    left_interval_median_speed_in_meter_per_second,    # (T-1,)
    right_interval_median_speed_in_meter_per_second,   # (T-1,)
    config,
)
```

四个数组必须同为 float32 或同为 float64、有限、长度匹配；speed 必须非负。
interval `i -> i+1` 分配给 frame `i`，最后一个 interval 重复给末帧。contact gate
使用 strict `< maximum_contact_speed...`。

结果状态：

- `candidate`：至少一个非 noise DBSCAN cluster；
- `no_contact_samples`：速度门未留下样本；
- `noise_only`：有样本，但只有 DBSCAN `-1` noise。

只有 `candidate_height_in_meter` 非空时才有候选。它直接取最低非-noise cluster
的高度 median，不减 HuMoR toe-joint 的 1 cm offset。每个 cluster 同时给出
minimum/median/maximum/span 与最大相邻高度 gap，用于发现 DBSCAN density
chaining；宽 cluster 仍是候选证据不足，而不是自动通过。

## 选择哪个 static-foot API

| 输入与目的 | API |
| --- | --- |
| 原生 root/left-toe/right-toe 3D tracks，要求精确复现 HuMoR 历史规则 | `estimate_static_foot_humor_baseline` |
| 已有共同 plantar 高度/速度 tracks，要与 foot-only zmin 公平比较 | `estimate_static_foot_plantar_humor_baseline` |

后者不是最终 robust 或 multilevel solver，也不是 AMASS ground truth。它排除
noise，但尚未要求 cluster compactness、时间连续性或显式 unknown/multilevel
判定。
