# 使用文档 —— hjlib-ground-solver

按"你手里有什么输入"组织。所有 public API 都从顶层 `hjlib_ground_solver` 直接导入。

## 一句话索引

本仓只有一个 usage 维度（按输入类型选求解入口），故不拆子页，全部列在本页。

## 决策树：我有什么 → 调哪个

```
我有 top/bottom 2D 关键点 (一群站立的人) + 相机 K
    └─ solve_ground_param_by_top_bottom_given_K   (estimate_ground, 主入口)
       内部链路: get_KN_with_filter -> solve_D_search
                 (底层可单独调: get_KN / get_bias_from_2D_ground_normal /
                  get_projection_loss / uv_to_xyz_via_ground_torch)

我有一批 3D 点 (落在地面上)
    ├─ 要地面 mesh (4 顶点矩形)         -> get_ground_by_points_on_the_ground
    └─ 要地面参数 (A,B,C,D)             -> get_ground_by_points_on_the_ground_lstsq
                                          (或更底层 compute_plane_parameters_by_positions_hj
                                           / compute_plane_normal_by_positions)

我有一批 pillars (起点 + 方向, 例如人体竖直向上向量)
    └─ get_ground_by_pillars_on_the_ground       -> 地面 mesh

我有一批 SMPL verts (+ SMPL_Full 模型)
    └─ get_ground_by_smpls_on_the_ground         -> 地面 mesh
       (策略: smpl_direction_maxmin / normal_direction_maxmin / normal_direction_lstsq)

我有一段已实现 full-body mesh vertices，想检查最低包络
    ├─ compute_per_frame_mesh_minimum_height      -> 每帧一个最低高度
    ├─ summarize_mesh_lower_envelope              -> absolute + retained-coverage candidates
    └─ peel_separated_mesh_lower_envelope_prefixes
       -> 迭代剥离局部分离的最低 value-prefix；过多轮次/移除量显式 unstable
       (注意: candidate 不是 semantic ground / ground truth)

我有 native-rate +Z-up 的 root / left-toe / right-toe 3D tracks
    └─ estimate_static_foot_humor_baseline
       -> 精确 HuMoR static-foot height-cluster comparator + 完整证据
       (注意: displacement 只是宽松候选门；cluster 才决定高度)

我有一张深度图 + 相机 K
    ├─ get_pixel_features_of_depth_map -> filter_vertical_and_horizontal_features  (边缘特征)
    ├─ cluster_pixel_features                                                       (像素聚类)
    ├─ get_depth_map_by_ground_np / _tensor   (给定地面参数 -> 渲染深度图)
    └─ get_ground_main_area_by_depth_map / get_list_ground_area_by_depth_map
       (注: depth-map 求解里的 L-BFGS 在 monolith 已被注释禁用, 见 design/migration.md DEAD-3)

我有世界坐标系 + up axis (y/z)
    ├─ get_ground_param_in_world_space               -> (A,B,C,D)
    ├─ get_ground_param_in_world_space_with_extrinsic -> 经 RT 转到相机系再解
    └─ get_ground_geometry_in_world_space             -> trimesh.Trimesh 地面网格

我有多帧 2D HVIP + 每人 2D 关键点 + RT/K
    └─ get_3d_info_from_hvip_2d   -> (3D world HVIP, 每帧地面参数, torso 2D)
```

## 公共契约

- 地面参数统一是 `(4,)` 的 `np.ndarray`，含义 `(A, B, C, D)`，平面 `Ax + By + Cz + D = 0`；
  法向 `[A, B, C]` 通常归一化，约定指向 `y<0`（见 `compute_plane_parameters_by_positions_hj`）。
- 地面 mesh 统一返回 `(verts, faces)`，`verts` 为 `(4, 3)`、`faces` 为 `(2, 3)`
  （`get_ground_geometry_in_world_space` 例外，直接返回 `trimesh.Trimesh`）。
- 像素坐标统一 `(u, v)`；齐次列向量约定 `(3, N)` 每列 `[u, v, 1]`（estimate_ground 内部）。

### Mesh lower envelope

```python
from hjlib_ground_solver import (
    compute_per_frame_mesh_minimum_height,
    summarize_mesh_lower_envelope,
)

per_frame_minimum = compute_per_frame_mesh_minimum_height(
    vertices,          # floating torch.Tensor, (B, V, 3), finite
    up_axis_index=2,
)
summary = summarize_mesh_lower_envelope(
    per_frame_minimum.detach().cpu().numpy(),
    retained_coverages=(1.0, 0.999, 0.995, 0.99),
    frame_rate_in_hz=120.0,
)
```

对 `T` 帧、coverage `c`，结果使用离散 order statistic：
`d = T - ceil(c*T)`、`E_c = sorted(m)[d]`。coverage 计数通过
`Decimal(str(c))` 计算，不做插值；短序列若 `d == 0`，该行就与 absolute
minimum 相同。每行同时给出相对 absolute minimum 的 delta、实际 retained
fraction 和 strict-below 最长连续段。

该 API 不做 body-model forward、不复制整段 mesh 到 CPU、不推断 contact，也不把
输出分类为 ground plane。调用方负责 mesh realization 与 semantic acceptance。

### Mesh lower-envelope low-prefix peeling

```python
from hjlib_ground_solver import (
    Mesh_Lower_Envelope_Peeling_Config,
    peel_separated_mesh_lower_envelope_prefixes,
)

config = Mesh_Lower_Envelope_Peeling_Config(
    maximum_candidate_fraction_per_round_decimal='0.02',
    maximum_candidate_frame_count_per_round=24,
    minimum_retained_frame_count=32,
    reference_gap_window_size=32,
    minimum_boundary_gap_in_meter=0.001,
    minimum_gap_ratio=5.0,
    maximum_round_count=3,
    maximum_total_removed_fraction_decimal='0.05',
    maximum_total_removed_frame_count=60,
)
result = peel_separated_mesh_lower_envelope_prefixes(
    per_frame_minimum_height_in_meter,
    config,
    frame_rate_in_hz=120.0,
)
```

该方法对高度与原始 frame index 做一次 lexsort，然后从当前最低值开始按 prefix
长度递增检查 boundary gap。reference 是 boundary 上方固定 `w` 个 gap slot 中的
正 gap 中位数；不会越过 slot window 搜集正 gap。第一个同时通过 absolute gap 与
gap-ratio 的 prefix 被整组剥离，ties 不拆分，然后重复。

只有 `status == 'stable_candidate'` 时
`accepted_candidate_height_in_meter` 才非 `None`。若下一轮仍有 eligible prefix，
但已达到 `maximum_round_count` 或总移除 budget，结果分别为
`unstable_maximum_round_count` / `unstable_removal_budget`，proposal 被记录但不应用。
每个 applied/blocked proposal 携带 native frame indices、gap/reference/ratio 与最长
连续 run，便于诊断与可视化。

这不是概率检验或通用离群算法。它无法仅凭 scalar heights 区分 penetration、有效
手/膝/躺姿接触、楼梯或平台；输出仍是 outlier-peeled lower-envelope candidate，
不是 semantic ground。

### Static-foot HuMoR baseline

```python
from hjlib_ground_solver import estimate_static_foot_humor_baseline

result = estimate_static_foot_humor_baseline(
    root_position_in_meter=root,            # (T, 3), float32/float64, finite
    left_toe_position_in_meter=left_toe,    # same dtype/shape, +Z up
    right_toe_position_in_meter=right_toe,
    frame_rate_in_hz=120.0,
)
```

这是钉在 HuMoR commit `fc6ef84f...` 的精确比较器，调用方不能调阈值。它先以
strict `< 0.005 m/native-frame` 排除粗大运动，再对保留的左脚趾、右脚趾高度
合并运行一维 DBSCAN。候选高度取最低 label median 减 `0.01 m`。因此：

- 逐帧位移只决定样本能否进入聚类，不排序已保留样本；
- 精确 parity 故意保留 DBSCAN `-1` noise 被当成一个普通 cluster 的上游缺陷；
- `upstream_zero_fallback` 与 `upstream_terrain_rejection` 的
  `accepted_candidate_height_in_meter` 都是 `None`；
- `upstream_candidate` 也只表示未被上游规则拒绝，不是 AMASS 官方 ground truth。

`samples` 保留 left-before-right 顺序、native frame index、height 和 label；
`clusters` 保留去重 root frames、两种 median、sample count、selected/terrain 证据。
该结果可直接 `dataclasses.asdict`，但不含 body-model joint index 或 AMASS 路径语义。

## 常见坑

- `solve_ground_param_by_top_bottom_given_K(flag_opt=True)` 会 `raise NotImplementedError`
  （monolith 标注 opt 路径不稳定，仅保留 grid-search）。
- `get_3d_info_from_hvip_2d` 末尾断言恢复出的 world HVIP `z ≈ 0`（假设 up_axis='z'、
  地面过世界原点）；若相机/地面约定不符会触发 assert。
- depth-map 系列函数依赖 hjlib-geometry 的 `get_depth_of_points_via_ground`；
  `get_ground_main_area_by_depth_map` 当前等价于返回硬编码初值（refinement 被禁用）。
- lower-envelope reducer 会拒绝任意坐标中的 NaN/Inf；不能靠某一轴上的有限最小值
  隐藏坏 vertex。`retained_coverages` 必须是唯一 Python `float` tuple 且含 `1.0`。
- peeling 的两个 fraction 是 exact decimal string；短序列的 effective total budget
  可以 floor 到 0，此时检测到的第一组会以 unstable removal-budget 返回，不会应用。
- static-foot HuMoR 的 `0.005` 是每 native frame 位移，不是 m/s；不要在调用前
  默默重采样或按 FPS 归一化，否则就不再是这个 exact baseline。
