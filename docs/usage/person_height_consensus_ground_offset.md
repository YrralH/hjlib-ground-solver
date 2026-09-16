# 从 Identity-Aware 2D Pose 求 Ankle Plane

> **状态：incomplete，修复前不要使用。** 此 V1 API 用一个 scene mean
> equivalent-height prior 固定尺度，同时允许每个 identity 有不同 inferred height；这套
> joint constraint 未被接受为目标方法。下面只记录历史 V1 调用方式，不能据此发布新结果。

先选好 pose population，并估计一个 camera-space Ground Normal。四组 joint-pair
array 都是 float64 `(N, 2, 2)`，中间轴依次放 left/right，且必须与 K 使用同一个
uncropped pixel frame。

```python
from hjlib_ground_solver import (
    Person_Height_Consensus_Config,
    Person_Height_Consensus_Observations,
    solve_ground_offset_by_person_height_consensus,
)

observations = Person_Height_Consensus_Observations(
    person_ids=person_ids,
    frame_indices=frame_indices,
    shoulder_xy_px=shoulder_pairs,
    hip_xy_px=hip_pairs,
    knee_xy_px=knee_pairs,
    ankle_xy_px=ankle_pairs,
)
result = solve_ground_offset_by_person_height_consensus(
    observations,
    ground_normal_camera,
    camera_intrinsics,
    retained_person_trimmed_mean_equivalent_height_m=1.28,
    config=Person_Height_Consensus_Config(
        minimum_valid_frames_per_person=24,
        minimum_retained_person_count=3,
    ),
)
plane_camera_abcd = result.plane_camera_abcd
```

`(person_id, frame_index)` 必须唯一，并已按 person/frame 升序排列。solver 拒绝
nonfinite inputs；退化的有限 pose 逐行记为 invalid。每层从两端各裁掉
`floor(0.05 N)` 个值；少于 20 个值时该层不裁。

mean height 只负责确定绝对尺度。改变它会同比缩放 `D` 和所有 person height，
不会改变 dimensionless `H/D`。必须先在 development data 上确定 mean height，
再执行 transfer solve。

| 已有输入 | 使用入口 |
| --- | --- |
| 具名 pose、identity、K、GN 与 frozen scene mean height | `solve_ground_offset_by_person_height_consensus` |
| 只有独立 top/bottom rows，且要走 registered Ours filters | `solve_ground_offset` |
| physical 3D ankle tracks，只做历史 distribution analysis | 已废弃的 `infer_person_ankle_plane_distribution` |
