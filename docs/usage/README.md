# 使用文档 —— hjlib-ground-solver

按"你手里有什么输入"组织。所有 public API 都从顶层 `hjlib_ground_solver` 直接导入。

## 一句话索引

- [density_balanced_rcr.md](density_balanced_rcr.md)：从 top/bottom 2D observations 构造可检查的 automatic-KDE 或 fixed-kNN inverse-density 权重并运行 weighted RCR。
- [vertex_subset_observation.md](vertex_subset_observation.md)：对显式局部 mesh indices 做可分块高度与 median speed 观测。
- [static_foot_plantar_humor.md](static_foot_plantar_humor.md)：在共同 plantar 高度/速度域上做 HuMoR-style phase-1 聚类。
- [hj_derived_plantar_zmin.md](hj_derived_plantar_zmin.md)：从左右 plantar 高度取明确非官方的 HJ-derived zmin 地面代理。
- [vanishing_direction_ground_normal.md](vanishing_direction_ground_normal.md)：从 line→VP sources 或 upright-person lines 求 locally-horizontal camera-space Ground Normal。
- [ours_ground_baselines.md](ours_ground_baselines.md)：按 frozen Ours Ground baseline 分别求 GN、offset D，或在 centered square-pixel 假设下联合求 camera+GN。
- 其余 ground solver 入口按输入类型列在本页。

## 决策树：我有什么 → 调哪个

```
我有 top/bottom 2D 关键点 (一群站立的人) + 相机 K
    ├─ 已有 camera-up GN，使用 frozen Ours Ground offset -> solve_ground_offset
    ├─ 直接等权求解 -> solve_ground_param_by_top_bottom_given_K
    └─ 想降低空间重复观测的支配 -> compute_ground_observation_kde_density
                                      (固定局部尺度才用 compute_ground_observation_density)
                                      + observation_weights=...
       内部链路: get_KN_with_filter -> solve_D_search
                 (底层可单独调: get_KN / get_bias_from_2D_ground_normal /
                  get_projection_loss / uv_to_xyz_via_ground_torch)

我有 line→VP，K 未知但 fx=fy/光心居中
    └─ solve_ground_normal_and_camera
       -> centered Camera_Intrinsics + camera-up GN
       再有 top/bottom observations 时显式后接 solve_ground_offset -> plane

我有一路或多路独立 line→VP associations + calibrated K，且地面局部水平
    ├─ 一路 source，使用 frozen exact simple-probe baseline
    │  └─ solve_ground_normal
    ├─ 一路 source，复用 max |camera y| baseline
    │  └─ solve_ground_normal_by_vertical_vp_selection
    ├─ 一路或多路 full sources，复用 discrete orthogonal consensus
    │  └─ solve_ground_normal_by_orthogonal_consensus
    ├─ full sources + 至多一路 vertical-only evidence
    │  └─ solve_ground_normal_by_role_aware_orthogonal_consensus
    ├─ 已预选的一路或多路 vertical 2D segments，要求每条线等权重新拟合
    │  └─ solve_ground_normal_by_equal_weight_vertical_lines
    ├─ 已预选的多路 vertical 2D segments，要求固定每路 source 总份额
    │  └─ solve_ground_normal_by_source_weighted_vertical_lines
    └─ 要 continuous robust fusion
       └─ solve_ground_normal_from_vanishing_directions
    均返回 camera-space unit Ground Normal + 原 selector ledger
    (只解 normal，不解 D/相机高度；sloped ground 不适用)

我有同一 fixed camera 下站立/近直立人的 shoulder-midpoint / ankle-midpoint pixels
    └─ fit_person_vertical_direction_evidence
       -> weighted RCR retained lines + one checked vertical VP source
       (作为 vertical-only source 参与 role-aware solve)

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

我有一段 mesh vertices + 明确的局部 vertex indices（例如脚掌）
    └─ compute_vertex_subset_observation_chunk
       -> chunkable 每帧最低高度 + interval median per-vertex speed + owned carry
       (topology 不在本仓；脚掌 indices 由 hjlib-smpl.foot_support 负责)

我有 native-rate +Z-up 的 root / left-toe / right-toe 3D tracks
    └─ estimate_static_foot_humor_baseline
       -> 精确 HuMoR static-foot height-cluster comparator + 完整证据
       (注意: displacement 只是宽松候选门；cluster 才决定高度)

我有 left/right plantar minimum height (T) + interval median speed (T-1)
    └─ estimate_static_foot_plantar_humor_baseline
       -> physical-speed gate + non-noise 1D DBSCAN + cluster span evidence
       (共同域比较 baseline；不是最终 robust/ground truth)

我有 left/right plantar minimum height (T)，决定使用非官方 absolute zmin
    └─ estimate_hj_derived_plantar_zmin_ground
       -> frozen HJ-derived nonofficial height + selected side/track-frame evidence
       (不推断 contact；不是 AMASS 官方 ground)

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

## Common API table

| 输入 / 目标 | 主入口 | 返回 |
| --- | --- | --- |
| top/bottom pixels + K | `solve_ground_param_by_top_bottom_given_K` | camera-frame plane `(4,)` + dimensionless objective |
| bottom pixels + K + provisional normal | `compute_ground_observation_kde_density` / `compute_ground_observation_density` | immutable density/weight evidence |
| 3D ground points | `get_ground_by_points_on_the_ground_lstsq` | plane `(4,)` |
| pillars / SMPL verts | `get_ground_by_pillars_on_the_ground` / `get_ground_by_smpls_on_the_ground` | ground mesh `(verts, faces)` |
| depth map + K | `get_ground_main_area_by_depth_map` | ground-area result；当前 refinement 受 migration 限制 |
| 2D HVIP + RT/K | `get_3d_info_from_hvip_2d` | world HVIP + per-frame plane + torso 2D |
| independent line→VP sources + K | `solve_ground_normal_from_vanishing_directions` | camera-space unit normal or `None` + full direction-fusion ledger |
| one line→VP source + K | `solve_ground_normal_by_vertical_vp_selection` | unit normal + support/camera-y/margin receipt |
| full line→VP sources + K | `solve_ground_normal_by_orthogonal_consensus` | unit normal + discrete winner/runner ledger |
| full + vertical-only line→VP sources + K | `solve_ground_normal_by_role_aware_orthogonal_consensus` | unit normal + role-aware discrete ledger |
| preselected vertical 2D segments + K | `solve_ground_normal_by_equal_weight_vertical_lines` | equal-line TLS unit normal + source/hash/count/scatter ledger |
| preselected vertical 2D segment sources + fixed fractions + K | `solve_ground_normal_by_source_weighted_vertical_lines` | source-total-weighted TLS unit normal + fraction/source/hash/count/scatter ledger |
| upright-person top/bottom pixels + weights + K | `fit_person_vertical_direction_evidence` | checked one-VP source + direction receipt |
| one line→VP source + K, frozen Ours Ground method | `solve_ground_normal` | registered config + simple-probe receipt + camera-up unit GN |
| top/bottom/confidence/ankle ratio + GN + K | `solve_ground_offset` | registered selection receipt + float64 camera-frame plane |
| line→VP，centered square-pixel K 未知 | `solve_ground_normal_and_camera` | registered camera receipt + camera-up GN |

## 公共契约

- 地面参数统一是 `(4,)` 的 `np.ndarray`，含义 `(A, B, C, D)`，平面 `Ax + By + Cz + D = 0`；
  法向 `[A, B, C]` 通常归一化，约定指向 `y<0`（见 `compute_plane_parameters_by_positions_hj`）。
- 地面 mesh 统一返回 `(verts, faces)`，`verts` 为 `(4, 3)`、`faces` 为 `(2, 3)`
  （`get_ground_geometry_in_world_space` 例外，直接返回 `trimesh.Trimesh`）。
- 像素坐标统一 `(u, v)`；齐次列向量约定 `(3, N)` 每列 `[u, v, 1]`（estimate_ground 内部）。
- top/bottom RCR 至少需要 3 个有限、非退化 observation。它用两轮低 angular-bias
  trimming 估计有限 vertical vanishing point，再在显式
  `[distance_min, distance_max)` metre grid 上搜索 plane distance；命中 search boundary
  会失败。返回的第二项是选择 distance 时使用的 dimensionless
  `relative-length + normalized-pixel` objective，不是 metre error。精确水平相机的
  vertical vanishing point 位于无穷远，当前入口会明确拒绝而非返回 NaN。
- `observation_weights` 是 keyword-only 可选 `(N,) float64` finite positive 数组。两轮 angular
  trimming 仍由未加权 observations 固定 membership，权重只进入最终 normal SVD
  与完整 observation population 的 D objective。`D_init/device/flag` 的历史 positional
  slots 保留；新增 distance bounds 与 weights 不占用旧位置。
- vanishing-direction GN 入口显式假设 local ground normal 与 gravity/scene vertical 平行；
  architectural vertical 不等于 gravity 或 ground 有 slope 时不适用。它不返回 plane offset
  `D`，也不从单图恢复 camera height。
- person-line evidence 还要求站立/近直立 body axis 表示 gravity，并且全部 observations
  已在同一 uncropped fixed-camera pixel frame；mixed crop/resize/K 不适用。
- equal-weight vertical-line 入口不会替调用方判断哪些 lines 是 vertical。调用方先完成
  method-specific preselection；随后所有保留 segments 逐线等权，不按 source 数、segment
  length 或 GT 重加权。返回值假设 local ground horizontal，只解 normal，不解 `D`。
- source-weighted vertical-line 入口使用 caller 固定的 source 总份额，每路内部仍逐线等权；
  它同样要求每路 lines 已独立预选为同一个物理 scene-vertical/gravity axis，不做 clustering、
  trimming、fraction tuning 或 GT selection。返回值只解 locally-horizontal normal。

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

### Static-foot plantar HuMoR baseline

共同 plantar-domain 的调用与两种 static-foot API 选择见
[static_foot_plantar_humor.md](static_foot_plantar_humor.md)。它使用 m/s 速度门、
排除 DBSCAN noise、取消 toe offset，并暴露 cluster span/gap；不要把它与上面的
精确历史 comparator 混为同一算法。

## 常见坑

- `solve_ground_param_by_top_bottom_given_K(flag_opt=True)` 会 `raise NotImplementedError`
  （monolith 标注 opt 路径不稳定，仅保留 grid-search）。
- `solve_ground_param_by_top_bottom_given_K` 的尺度由 `H_prior` 提供；默认
  `1.35 m` 是 fixed-height RCR baseline，而不是从单目关键点独立恢复的绝对人体高度。
- inverse-density 权重只重新分配贡献，不删除 observations，也不保证 normal、D
  与组合的 ground-effect 指标会同时改善；必须分别诊断。
- 本次同时修正了旧 trim 的 sorted-position mask 与 threshold-tie bug。因此当前
  unweighted/`None` 是 corrected control，不是 pre-fix implementation 的 byte parity。
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
