# Migration record —— hjlib-ground-solver

Port type: **file-mapping**（保留 monolith 文件级结构，逐文件小改）。
Equivalence model（uniform parity + intentional divergence）见
[migration_protocol.md](../../../hjlibm/docs/hjlib_standard/migration_protocol.md)。

## 1. Source

- monolith repo: `~/Repo/dynamic_hvip`，子树 `lib_dynamic_hvip/ground/`（solver 侧）。
- frozen sha: **`2bc42db4`**（只读，不改）。
- capture date: 2026-05-27。
- 本次 scope = **Tier-1**（逐文件依赖 2026-05-27 核实）。Tier-2 见 [handoff.md](handoff.md)。

## 2. Destination

- lib: `hjlib-ground-solver`，src `src/hjlib_ground_solver/`。
- initial commit: 见 git log（Phase 1 开仓）。
- deps: `hjlib-geometry` `269c7c21` + `hjlib-smpl` `62940b5c`。

## 3. What was ported（file-level mapping）

| monolith 源 (`lib_dynamic_hvip/ground/`) | 本仓 dest (`src/hjlib_ground_solver/`) | 改动 |
|---|---|---|
| `segment_area/cluster_feature.py` | `segment_area/cluster_feature.py` | 清死 import |
| `segment_area/filter_depth_map_feature.py` | `segment_area/filter_depth_map_feature.py` | 清死 import；f-string→percent |
| `segment_area/get_depth_map_by_ground.py` | `segment_area/get_depth_map_by_ground.py` | `get_depth_of_points_via_ground` 改 import 自 hjlib-geometry |
| `segment_area/get_ground_by_depth_map.py` | `segment_area/get_ground_by_depth_map.py` | intra-lib import 改名；DEAD-3 去禁用的 L-BFGS scaffolding |
| `get_ground_geometry/by_pillars.py` | `get_ground_geometry/by_pillars.py` | 中文注释译英 |
| `get_ground_geometry/by_points.py` | `get_ground_geometry/by_points.py` | 内联 `get_valid_filter_mask_by_max_value`；中文注释译英 |
| `get_ground_geometry/by_smpl.py` | `get_ground_geometry/by_smpl.py` | 清死 import；SMPL_Full + skeleton helper 改 import 自 hjlib-smpl；`convert_..._on_ground_points` 自 hjlib-geometry；annotation 修正（见 FIX-1） |
| `get_ground_geometry/by_wolrd_space.py` | `get_ground_geometry/by_world_space.py` | **DIV-1** 文件名 typo 修正；两个 import 均为死 import 已删 |
| `get_ground_param/by_world_space.py` | `get_ground_param/by_world_space.py` | `solve_ground_from_3_points` 改 import 自 hjlib-geometry |
| `hvip/get_3d_info_from_hvip_2d.py` | `hvip/get_3d_info_from_hvip_2d.py` | 大量死 import 清理（见 §5）；`assert_is_R`→hjlib-geometry `assert_is_rotmat`（DIV-2）；活 dep 改 import 自 hjlib-geometry/hjlib-smpl |
| `estimate_ground/by_kp_rcr/compute_KN_by_vertical_lines.py` | `estimate_ground/by_kp_rcr/compute_KN_by_vertical_lines.py` | 内联 3 个 utils；删未用 import + 死 else-branch（见 §5）；`KN` 重赋值改名避 strict |
| `estimate_ground/by_kp_rcr/solve_by_top_bot/project_loss.py` | 同路径 | 加返回类型注解（无逻辑改动） |
| `estimate_ground/by_kp_rcr/solve_by_top_bot/search_D.py` | 同路径 | 删死 import `reverse_project_via_ground_tensor`；`D` 重赋值改 lowercase 避 strict |
| `estimate_ground/by_kp_rcr/solve_by_top_bot/process_solve_by_top_bot_given_K.py` | 同路径 | 删死 import（Lib_SMPL / vis_*） |

### 跨仓 import 重定向（活依赖）

| monolith 符号 | 现 import 自 |
|---|---|
| `ground.reverse_project.reverse_project.get_depth_of_points_via_ground` | `hjlib_geometry` |
| `ground.reverse_project.reverse_project.reverse_project_via_ground` | `hjlib_geometry` |
| `ground.get_ground_param.by_points_on_ground.solve_ground_from_3_points` | `hjlib_geometry` |
| `ground.get_ground_geometry.by_param.convert_ground_parameters_to_4_verts_mesh_with_on_ground_points` | `hjlib_geometry` |
| `utils.utils_np.assert_is_R` | `hjlib_geometry.assert_is_rotmat`（DIV-2，等价替代） |
| `smpl.smpl_lib.SMPL_Full` | `hjlib_smpl` |
| `smpl.skeleton.get_rough_pillars_and_from_smpl_verts_batch` | `hjlib_smpl` |
| `smpl.skeleton.get_2d_torso_center_from_2d_joint` | `hjlib_smpl` |

## 4. What was NOT ported

| 项 | 原因 |
|---|---|
| `ground/estimate_ground/for_fixed_camera_video_by_rcr.py` | **Tier-2**，真耦合 monolith `detection.{filter, remove_duplication}`。见 [handoff.md](handoff.md) / DRU-9 |
| `ground/hvip/{get_hvip_by_linear, get_hvip_by_ray_cast}.py` | 纯 use（签名输入含 `ground_parameters`）→ 归 hjlib-geometry（DRU-10），本仓不 port；需要时从 hjlib-geometry import |
| `ground/estimate_ground/by_kp_rcr/estimate_ground_by_human_top_bot.py` | 0 行空文件，drop |
| `ground/{reverse_project/*, transform.py, get_ground_geometry/by_param.py, get_ground_param/by_points_on_ground.py}` | use 侧，已在 hjlib-geometry（DRU-7 等） |

## 5. 死代码清理（逐条；非 divergence —— 删的都是从未被运行/引用的代码）

| id | 文件 | 删了什么 |
|---|---|---|
| DEAD-1 | `hvip/get_3d_info_from_hvip_2d.py` | 死 import：`vis.color.LIST_COLOR_BGR_36`、`vis.vis_2d.{vis_bbox, vis_points}`、`render.persp_project.project_3D_points`、`constants_skeleton.{get_index_transform, COCO_17, COCO_BODY_WITH_FOOT_23}`、`local_setting`，及 shutil/os/sys/json/tqdm/Enum/pickle/ic/cv/trimesh —— body 全不引用（grep 核实） |
| DEAD-2 | `estimate_ground/.../process_solve_by_top_bot_given_K.py` | 死 import：`smpl.smpl_lib as Lib_SMPL`、`vis.{vis_smpl, vis_2d, vis_3d}`，及 trimesh/cv/ic/pickle/tqdm |
| DEAD-3 | `segment_area/get_ground_by_depth_map.py` | `get_ground_main_area_by_depth_map` 内的 L-BFGS scaffolding（`closure` / `optimizer` / 喂给它的 tensor 转换）—— monolith 里 `optimizer.step` 已被注释禁用，函数实际只返回硬编码初值。删后 **observable 行为完全一致**（仍返回同一 `np.array([0.034178, -0.899, -0.14, 20])`）。`forward_ground_parameters_to_depth_map` / `get_depth_map_loss` 作为独立 public 函数保留 |
| DEAD-4 | `estimate_ground/.../compute_KN_by_vertical_lines.py` | 未用 import `utils_py.{generate_info, filter_by_valid_mask}`；`get_KN_with_filter` 里 `if True: ... else: filter_by_value(...)` 的死 else 分支（引用从未定义的 `filter_by_value` / `_values`），collapse 为 if-body |
| DEAD-5 | `get_ground_geometry/by_wolrd_space.py` | 两个从未被函数体引用的 import（`get_ground_param_in_world_space`、`convert_..._on_ground_points`）；该文件实为 leaf（numpy+trimesh） |
| DEAD-6 | `segment_area/cluster_feature.py` 等 | 各文件 boilerplate 死 import（os/sys/json/icecream/pickle/未用 typing/未用 torch/sklearn.KMeans 等） |

## 6. Intentional API divergences

| id | 改动 | 为什么 |
|---|---|---|
| DIV-1 | 文件 `by_wolrd_space.py` → `by_world_space.py`（typo 修正） | 家族干净命名；新目录下不与 `get_ground_param/by_world_space.py` 冲突。public 函数名 `get_ground_geometry_in_world_space` 不变 |
| DIV-2 | `utils_np.assert_is_R(R)` → `hjlib_geometry.assert_is_rotmat(R)` | monolith `assert_is_R` 已在 hjlib-geometry 拆为 `check_orthogonal` + `assert_is_rotmat`（DRU-7）；后者是等价的正交+det 检查，直接复用通用 leaf 而非内联 |

注：DIV 不写 parity（行为不同）；Phase 2 用 behavior 覆盖。

## 7. Bug fixes during the port

| id | 位置 | 问题 | 修复 |
|---|---|---|---|
| FIX-1 | `get_ground_geometry/by_smpl.py` | monolith `get_ground_by_smpls_on_the_ground` 形参注解 `list_smpl_verts: List[str]` 错误（实际是 SMPL verts `np.ndarray` 列表，且 hjlib-smpl 的 `get_rough_pillars_and_from_smpl_verts_batch` 签名要求 `List[np.ndarray]`） | 注解改为 `List[np.ndarray]`。纯类型注解修正，运行期行为不变 |

## 8. Where verification lives

`Code_as_Libs/hjlib-migration-tests/ground-solver/`（新 session 建，per
migration_protocol.md step 13-14）。本仓不依赖 monolith。

## 9. Migration test status

- [x] Phase 1 — code lives in new lib; pyright strict 0 errors + 16 smoke green;
      initial commit landed.
- [ ] Phase 2 — parity + behavior tests green in
      `hjlib-migration-tests/ground-solver/`; `(2bc42db4, <new_lib_sha>)` pinned
      in that subdir's README.md; `grep -rn 'lib_dynamic_hvip' .../ground-solver/behavior/`
      returns empty.
- [ ] Phase 3 — absorbed; `behavior/` + per-lib `conftest.py` + `local_setting_test.py`
      + readers moved into `hjlib-ground-solver/test/`; `parity/` + `divergence/`
      deleted; `hjlib-migration-tests/ground-solver/` subdir removed
      (absorb date: YYYY-MM-DD).
