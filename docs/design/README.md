# 设计文档 —— hjlib-ground-solver

本仓**唯一** onboarding 入口（无 `docs/CLAUDE.md`）。改本仓前先读这里。

## 1. Scope

**做什么**：地面的"求解 / 主动推断"侧 —— 从 SMPL pillars、top-bottom 关键点、
深度图等观测推断地面参数 / 几何；并从 2D HVIP 反推 3D 信息（含其内部的地面现算）。
另外提供不声称 ground truth 的 full-mesh / vertex-subset observation、
static-foot comparison candidates，以及 provenance 固定为
`hj_derived_nonofficial` 的 plantar-zmin 高度代理。

**不做什么**：

- 地面的"被动使用"（已知地面后的 reverse_project / transform / by_param 等）
  → 在 [`hjlib-geometry`](../../../hjlib-geometry)（沿 DRU-7 use/solve 切分）。
- `hvip/{get_hvip_by_linear, get_hvip_by_ray_cast}`（纯 use，输入已含 ground_parameters）
  → 归 hjlib-geometry（DRU-10），本仓不 port，需要时从 hjlib-geometry import。
- `for_fixed_camera_video_by_rcr.py`（Tier-2，耦合 monolith detection）→ 见
  [handoff.md](handoff.md)（DRU-9），本次未迁。
- SMPL forward / fitting：来自 hjlib-smpl。

## 2. Repo layout

```
hjlib-ground-solver/
├── README.md
├── pyproject.toml              deps = hjlib-geometry + hjlib-smpl (+ numpy/torch/cv2/trimesh/scipy/sklearn)
├── pyrightconfig.json          strict; 关掉 5 条第三方 stub 噪声规则 (见 §4)
├── .gitignore
├── src/hjlib_ground_solver/
│   ├── __init__.py             顶层 re-export 全部 public 函数
│   ├── segment_area/
│   │   ├── cluster_feature.py              cluster_pixel_features (k-means 像素聚类)
│   │   ├── filter_depth_map_feature.py     深度图边缘特征 + 阈值过滤
│   │   ├── get_depth_map_by_ground.py      地面参数 -> 深度图 (调 hjlib-geometry)
│   │   └── get_ground_by_depth_map.py      深度图 -> 地面 (L-BFGS 已禁用, 见 DEAD-3)
│   ├── get_ground_geometry/
│   │   ├── by_pillars.py        起点+方向 pillars -> 地面 mesh
│   │   ├── by_points.py         3D 点 -> 地面 mesh / 参数 (内联 utils, 见 §3)
│   │   ├── by_smpl.py           SMPL verts -> 地面 mesh (调 hjlib-smpl + hjlib-geometry)
│   │   └── by_world_space.py    世界系轴对齐地面 trimesh (原文件名 typo by_wolrd_space, 见 DIV-1)
│   ├── get_ground_param/
│   │   └── by_world_space.py    世界系/经 RT 的地面参数 (调 hjlib-geometry solve_ground_from_3_points)
│   ├── hvip/
│   │   └── get_3d_info_from_hvip_2d.py     2D HVIP + RT/K -> 3D world HVIP + 地面 (含 solve)
│   └── estimate_ground/
│       ├── by_mesh_lower_envelope.py        full-mesh per-frame minima + exact coverage candidates
│       ├── by_mesh_lower_envelope_peeling.py iterative separated low-prefix peeling
│       ├── by_static_foot_humor.py          exact HuMoR static-foot height clustering
│       ├── by_vertex_subset_observation.py  chunkable local-vertex height/speed tracks
│       ├── by_static_foot_plantar_humor.py  common-domain plantar HuMoR-style comparator
│       ├── by_hj_derived_plantar_zmin.py    explicitly nonofficial absolute plantar zmin
│       └── by_kp_rcr/
│           ├── compute_KN_by_vertical_lines.py  竖直线消失点 KN + 过滤 (内联 3 个 utils)
│           └── solve_by_top_bot/
│               ├── project_loss.py               投影 loss (torch)
│               ├── search_D.py                   grid-search 解地面距离 D
│               └── process_solve_by_top_bot_given_K.py  顶层入口 solve_ground_param_by_top_bottom_given_K
├── test_smoke/                 topic smoke + master runner + clean_test_data
├── test/                       数据依赖测试占位 (Phase 2 前主要落 hjlib-migration-tests)
└── docs/{usage,design}/
```

## 3. 关键设计点

- **port 风格 = file-mapping**：保留 monolith 文件级结构，逐文件小改（修 import +
  清死 import + 内联小 utils）。逐条记录见 [migration.md](migration.md)。
- **内联的 utils**（不开 `utils_*.py`，遵家族禁 utils-module 约定）：
  - `by_points.py` 内联 `utils_py.get_valid_filter_mask_by_max_value`
  - `compute_KN_by_vertical_lines.py` 内联 `utils_np.{assert_zeros, filter_column_vectors_by_list_valid_mask}`
    + `utils_py.get_valid_filter_mask_by_max_value`
- **跨仓边界**：本仓只直接 dep `hjlib-geometry` + `hjlib-smpl`；skeleton / camera /
  vis-2d 作为 hjlib-smpl 的传递依赖在 env 里，但**不在** `[tool.hjlibm.deps]` 直接声明。

## 4. Family conventions inherited

- pyright strict 默认 —— [pyright-strict-default](../../../../.claude/projects/-data3-hj-home-hj-Repo-Code-as-Libs/memory/feedback_pyright_strict_default.md)
- 字符串 percent-style 单引号 / 注释英文标点 / 4 空格缩进 / 禁 `utils_*.py` 命名
  —— 见 `Code_as_Libs/CLAUDE.md` + family memory。
- 测试两棵树（test_smoke + test）—— [test_layout.md](../../../hjlibm/docs/hjlib_standard/test_layout.md)，
  本仓实例见 [test.md](test.md)。

### pyright 配置说明

`typeCheckingMode = 'strict'`，并关掉 5 条第三方 stub 噪声规则
（`reportMissingTypeStubs` / `reportUnknownMemberType` / `reportUnknownVariableType` /
`reportUnknownArgumentType` / `reportPrivateImportUsage`），即 family 标准的
ladder level 3，根因与处理标准见
[pyright_stub_noise.md](../../../hjlibm/docs/hjlib_standard/pyright_stub_noise.md)。
本仓触发库：torch / cv2 / trimesh / smplx 的弱 stub 在几乎每个 numpy/torch
调用点产生噪声，压垮真实 strict 信号。其余 strict 规则全开，0 errors。

## 5. State of the world

- pyright: **strict, 0 errors**（见 §4 的规则豁免）。
- 测试: `test_smoke/` master 全绿；`get_ground_by_smpls_on_the_ground`
  需真实 SMPL 模型，留给数据依赖测试（见 [test.md](test.md)）。
- remote: <https://github.com/YrralH/hjlib-ground-solver>
- deps: hjlib-geometry `fe58e07c` + hjlib-smpl `64f4f49b`（当前 pyproject pin，随 sibling
  commit 落地用 `hjlibm version` bump）。

## 6. What's open

- **AMASS mesh lower-envelope task**: the implemented/reviewed Layered Design residence is
  [`tasks/amass-ground-zmin-family/`](tasks/amass-ground-zmin-family/). It owns
  the reusable zmin/retained-coverage core; AMASS reading and configured
  operation orchestration remain outside this package's public API.
- **AMASS zmin outlier-peeling task**: the implemented/reviewed Layered Design residence is
  [`tasks/amass-ground-zmin-outlier-peeling/`](tasks/amass-ground-zmin-outlier-peeling/).
  Mathematical and Code Architecture plus final logic/implementation/consistency
  closure are independently reviewed; public core, synthetic smoke, and
  authoritative bounded AMASS v3 evidence are complete.
- **AMASS static-foot HuMoR baseline**: the reviewed Layered Design residence is
  [`tasks/amass-ground-static-foot-humor-baseline/`](tasks/amass-ground-static-foot-humor-baseline/).
  The reusable exact comparator, immutable evidence records, synthetic oracle
  smoke, and public exports are implemented; the configured AMASS operation
  and bounded visual artifact remain in `hjlib-dataset-raw`.
- **AMASS foot-sole domain and lower envelope**: the completed Layered Design
  residence is
  [`tasks/amass-ground-foot-sole-domain/`](tasks/amass-ground-foot-sole-domain/).
  It puts lower-envelope and future robust static-foot work on one plantar
  heel/mid-foot/toe observation domain while preserving historical full-mesh
  and exact-HuMoR comparators. Public subset observation, SMPL-H/SMPL-X mask
  composition, synthetic/model-backed tests, and raw post-review v5 evidence
  are complete; 3/5 mm masks pass to the separate robust-contact task.
- **AMASS static-foot robust**: phase 1 is implemented/reviewed at
  [`tasks/amass-ground-static-foot-robust/`](tasks/amass-ground-static-foot-robust/).
  It adds a common-domain plantar-HuMoR baseline with physical-speed gating,
  non-noise DBSCAN selection, and explicit cluster compactness evidence. Raw v7
  shows broad 21--69 mm selected spans, so final compactness/temporal and
  multilevel/unknown semantics remain open.
- **AMASS HJ-derived plantar zmin ground**: the active Layered Design residence
  is [`tasks/amass-ground-hj-derived-plantar-zmin/`](tasks/amass-ground-hj-derived-plantar-zmin/).
  It turns the user-selected plantar zmin proxy into an explicitly HJ-derived,
  nonofficial public result; implementation is gated on layer review.
- **Phase 2 (parity + behavior)**：留给新 session 在 `hjlib-migration-tests/ground-solver/`
  落地。本仓 migration.md 已建 Phase 1/2/3 checkbox。
- **Tier-2 / DRU-9 / DRU-10**：见 [handoff.md](handoff.md)。
- **`get_ground_by_smpls_on_the_ground` 数据测试**：需 SMPL_Full 模型，long-term pending，
  待第一份真实 SMPL 序列可用（与 dataset-raw-uplift 链路相关）。
