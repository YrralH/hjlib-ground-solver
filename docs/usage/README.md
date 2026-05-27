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

## 常见坑

- `solve_ground_param_by_top_bottom_given_K(flag_opt=True)` 会 `raise NotImplementedError`
  （monolith 标注 opt 路径不稳定，仅保留 grid-search）。
- `get_3d_info_from_hvip_2d` 末尾断言恢复出的 world HVIP `z ≈ 0`（假设 up_axis='z'、
  地面过世界原点）；若相机/地面约定不符会触发 assert。
- depth-map 系列函数依赖 hjlib-geometry 的 `get_depth_of_points_via_ground`；
  `get_ground_main_area_by_depth_map` 当前等价于返回硬编码初值（refinement 被禁用）。
