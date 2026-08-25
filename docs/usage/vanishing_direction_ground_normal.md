# Vanishing-Direction Ground Normal

## 我有 line→VP source

以下入口都要求 calibrated `Camera_Intrinsics`，并显式假设 local ground normal
平行于 gravity/scene vertical：

```python
from hjlib_ground_solver import (
    solve_ground_normal_by_vertical_vp_selection,
    solve_ground_normal_by_orthogonal_consensus,
    solve_ground_normal_by_role_aware_orthogonal_consensus,
    solve_ground_normal_from_vanishing_directions,
)

simple = solve_ground_normal_by_vertical_vp_selection(source, intrinsics)
discrete = solve_ground_normal_by_orthogonal_consensus(
    full_sources,
    intrinsics,
    discrete_config,
)
role_aware = solve_ground_normal_by_role_aware_orthogonal_consensus(
    role_aware_sources,
    intrinsics,
    discrete_config,
)
formal = solve_ground_normal_from_vanishing_directions(
    full_sources,
    intrinsics,
    robust_config,
)
```

`simple`、`discrete` 与 `role_aware` 的 `ground_normal_camera` 总是存在；selector
找不到合法 winner 时直接传播 `ValueError`。`formal` 允许
`ground_normal_camera=None`，并在 nested ledger 中记录 all-rejected candidates。

## 我有 upright-person top/bottom pixels

```python
from hjlib_ground_solver import fit_person_vertical_direction_evidence

people = fit_person_vertical_direction_evidence(
    top_xy_px,
    bottom_xy_px,
    observation_weights,
    intrinsics,
    source_id='people',
    image_record_id='scene:people-all-frames',
    prop_filter=0.24,
    times_filter=2,
)
```

输入至少三条 finite、non-zero-length、同一 fixed-camera pixel frame 的 standing /
near-upright body-axis observations。`prop_filter` 是每轮 discard fraction；trimming
membership 不加权，weights 只进入最终 retained SVD。结果 `source` 含一个 VP cluster，
可由调用方包装为 `Role_Aware_Vanishing_Direction_Source(..., False)`。

## Picking between options

| 输入与需求 | 入口 |
| --- | --- |
| 一路 source，复现 max camera-y baseline | `solve_ground_normal_by_vertical_vp_selection` |
| full sources，离散 3-degree consensus | `solve_ground_normal_by_orthogonal_consensus` |
| full sources + vertical-only source | `solve_ground_normal_by_role_aware_orthogonal_consensus` |
| continuous distribution/refinement | `solve_ground_normal_from_vanishing_directions` |
| upright-person pixels → vertical evidence | `fit_person_vertical_direction_evidence` |

所有 solve 只返回 normal，不返回 plane offset `D`、camera height 或 slope。
