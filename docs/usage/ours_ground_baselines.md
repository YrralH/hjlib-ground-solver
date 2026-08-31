# 按是否给定 K 求 Ground Normal、camera 与 offset

```python
from hjlib_ground_solver import (
    Ground_Offset_Observations,
    solve_ground_normal,
    solve_ground_offset,
)

normal_result = solve_ground_normal(line_vp_source, intrinsics)
observations = Ground_Offset_Observations(
    top_xy_px=shoulder_midpoints,
    bottom_xy_px=ankle_midpoints,
    confidence=person_confidence,
    ankle_ratio=ankle_pair_distance_over_bbox_width,
)
offset_result = solve_ground_offset(
    observations,
    normal_result.ground_normal_camera,
    intrinsics,
)
plane_camera = offset_result.plane_camera_abcd
```

若 K 未知，但已知 `fx=fy` 且光心居中，可从同一张图的 line→VP 与 person evidence
联合求 camera、GN 和 D：

```python
from hjlib_ground_solver import (
    Ground_Offset_Observations,
    solve_ground_normal_and_camera,
    solve_ground_offset,
)

camera_normal_result = solve_ground_normal_and_camera(line_vp_source)
observations = Ground_Offset_Observations(
    top_xy_px=shoulder_midpoints,
    bottom_xy_px=ankle_midpoints,
    confidence=person_confidence,
    ankle_ratio=ankle_pair_distance_over_bbox_width,
)
offset_result = solve_ground_offset(
    observations,
    camera_normal_result.ground_normal_camera,
    camera_normal_result.camera_intrinsics,
)
intrinsics = camera_normal_result.camera_intrinsics
plane_camera = offset_result.plane_camera_abcd
```

三个入口的默认 ID 分别是 `ground_normal_baseline001`、
`ground_offset_baseline001` 与 `ground_normal_and_camera_baseline001`。可先调用
`ground_normal_config()`、`ground_offset_config()` 或
`ground_normal_and_camera_config()` 查看 frozen config；不要在调用处复制数值。

offset 使用 strict `confidence > 4.3`、strict `ankle_ratio < 0.20`、unweighted
observations 与 `H_prior=1.27 m`。`select_ground_offset_observations(observations,
ground_offset_config())` 可在求 D 前单独取得 full-length retained mask。

| 你已有的输入 | 调用 |
| --- | --- |
| line→VP source + K | `solve_ground_normal` |
| top/bottom + confidence/ankle ratio + GN + K | `solve_ground_offset` |
| line→VP，K 未知但 centered square-pixel | `solve_ground_normal_and_camera`，返回 K+GN |
| 上述 K+GN 再加 person observations | 显式后接 `solve_ground_offset` |
| 想自己调 H、filter 或 weighting | 使用低层研究入口，不得仍称 `ground_offset_baseline001` |

所有 pixel coordinates 必须与 K 处于同一个 uncropped image frame。GN 必须是 float64
camera-up unit vector；offset 只提供绝对尺度所需的 fixed height prior，不是单目无先验
恢复 scale。
