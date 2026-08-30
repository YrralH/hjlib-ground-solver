# 设计文档 —— hjlib-ground-solver

本仓**唯一** onboarding 入口（无 `docs/CLAUDE.md`）。改本仓前先读这里。

> `estimate_ground/person_ankle_plane/` 的 distribution-first V1 已暂时废弃并冻结，
> 只保留历史实现和 regression。当前 ankle-ground 方法不在本仓沿 V1 继续开发；唯一
> active 路线见 `hjlib-dataset-std` Campaign 02 的 temporal low-basin/getG task。

## 1. Scope

**做什么**：地面的"求解 / 主动推断"侧 —— 从 SMPL pillars、top-bottom 关键点、
深度图等观测推断地面参数 / 几何；并从 2D HVIP 反推 3D 信息（含其内部的地面现算）。
另外提供不声称 ground truth 的 full-mesh / vertex-subset observation、
static-foot comparison candidates，以及 provenance 固定为
`hj_derived_nonofficial` 的 plantar-zmin 高度代理。
在 explicit locally-horizontal scene assumption 下，本仓还负责把 camera-solver
选出的 single-source / discrete-consensus / robust calibrated vertical 解释为
camera-space Ground Normal，并从 fixed-camera upright-person top/bottom lines 构造
vertical evidence；不解 slope、offset 或 camera height。

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
├── pyproject.toml              deps = hjlib-camera + camera-solver + geometry + smpl (+ third-party runtime)
├── pyrightconfig.json          strict; 关掉 5 条第三方 stub 噪声规则 (见 Family conventions)
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
│       ├── person_ankle_plane/                per-person local clusters -> global height hypotheses -> typed status
│       ├── observation_density.py              provisional-plane exact-LOO KDE / kNN density + immutable weight evidence
│       ├── by_mesh_lower_envelope.py        full-mesh per-frame minima + exact coverage candidates
│       ├── by_mesh_lower_envelope_peeling.py iterative separated low-prefix peeling
│       ├── by_static_foot_humor.py          exact HuMoR static-foot height clustering
│       ├── by_vertex_subset_observation.py  chunkable local-vertex height/speed tracks
│       ├── by_static_foot_plantar_humor.py  common-domain plantar HuMoR-style comparator
│       ├── by_hj_derived_plantar_zmin.py    explicitly nonofficial absolute plantar zmin
│       ├── by_vanishing_direction.py        single-source / robust camera vertical -> locally-horizontal GN
│       ├── by_orthogonal_vanishing_direction.py discrete/role-aware consensus -> locally-horizontal GN
│       ├── by_person_vertical_lines.py       weighted person top/bottom RCR -> one checked VP source
│       ├── by_equal_vertical_lines.py         equal-line / source-total-weighted vertical lines -> locally-horizontal GN
│       ├── ground_normal_contract.py          shared immutable/unit/exact-winner GN invariant
│       ├── ours_baseline.py                   registered given-camera GN / offset baselines
│       └── by_kp_rcr/
│           ├── compute_KN_by_vertical_lines.py  竖直线消失点 KN + 过滤 (内联 3 个 utils)
│           ├── observation_weight.py            optional NumPy/torch positive-weight boundary validation
│           └── solve_by_top_bot/
│               ├── project_loss.py               投影 loss (torch)
│               ├── search_D.py                   grid-search 解地面距离 D
│               └── process_solve_by_top_bot_given_K.py  顶层入口 solve_ground_param_by_top_bottom_given_K
├── test_smoke/                 topic smoke + master runner + clean_test_data
├── test/                       数据依赖测试占位 (Phase 2 前主要落 hjlib-migration-tests)
└── docs/{usage,design}/
```

## 3. Must-read

1. [migration.md](migration.md) —— monolith file mapping、intentional divergence 与 migration status。
2. [observation_density.md](observation_density.md) —— density IR、weighted RCR 与 owner boundary。
3. [test.md](test.md) —— portable/data-dependent 两棵测试树在本仓的实例。
4. [handoff.md](handoff.md) —— deferred Tier-2 与跨仓 handoff。
5. [tasks/equal_line_vertical_ground_normal/README.md](tasks/equal_line_vertical_ground_normal/README.md) —— equal-per-line vertical fit 的 Ground 解释。
6. [tasks/source_weighted_vertical_ground_normal/README.md](tasks/source_weighted_vertical_ground_normal/README.md) —— source-total-weighted vertical fit 的 Ground 解释。
7. [ours_ground_baselines.md](ours_ground_baselines.md) —— frozen Ours Ground GN/offset IDs、selection seam 与 orientation contract。
8. [dataset-std task: person ankle-plane distribution V1](../../../hjlib-dataset-std/docs/design/tasks/person_ankle_plane_distribution/README.md)
   —— 已迁移的 failed predecessor task design history、数学契约与测试标准；本仓暂留
   V1 实现，待 replacement design 后统一清理。

## 4. 关键设计点

- **port 风格 = file-mapping**：保留 monolith 文件级结构，逐文件小改（修 import +
  清死 import + 内联小 utils）。逐条记录见 [migration.md](migration.md)。
- **内联的 utils**（不开 `utils_*.py`，遵家族禁 utils-module 约定）：
  - `by_points.py` 内联 `utils_py.get_valid_filter_mask_by_max_value`
  - `compute_KN_by_vertical_lines.py` 内联 `utils_np.{assert_zeros, filter_column_vectors_by_list_valid_mask}`
    + `utils_py.get_valid_filter_mask_by_max_value`
- **跨仓边界**：本仓 direct dep `hjlib-camera`（public K type）、
  `hjlib-camera-solver`（line/VP calibration 与 direction selection owner）、`hjlib-geometry`（ground use）与
  `hjlib-smpl`（body observations）；skeleton / vis-2d 仍为传递依赖。
- **density-balanced RCR**：method-neutral provisional-plane density、immutable
  intermediate 与 weighted solver contract 见
  [observation_density.md](observation_density.md)。dataset selection 与结果评估不进本仓。
- **Per-person ankle-plane distribution V1 implementation**：本仓暂时保留从一个 physical person
  的 world-ankle runs + oriented source planes 到 immutable local/global evidence
  和 typed result 的数值层。physical-scene/person identity、canonical-view
  selection、export I/O、config selection 和结果评估都在上层。算法不会 refit
  normal，也不会把一个 static local episode 自动提升为 scalar ground；这个
  identifiability boundary 是 V1 public contract，不是 dataset special case。其
  task residence 已转到 `hjlib-dataset-std`，新设计不由本页预先决定 code home。

## 5. Family conventions inherited

- pyright strict 与 suppression ladder —— [pyright_stub_noise.md](../../../hjlibm/docs/hjlib_standard/pyright_stub_noise.md)
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

## 6. State of the world

- pyright: **strict, 0 errors**（见 §5 的规则豁免）。
- 测试: `test_smoke/` **142 passed**；`get_ground_by_smpls_on_the_ground`
  需真实 SMPL 模型，留给数据依赖测试（见 [test.md](test.md)）。
- density/weighted RCR：公开 API、immutable evidence 与 synthetic hand-oracle
  smoke 已实现；VirtualCrowd real operation 由 `hjlib-evaluation` 持有。
- per-person ankle-plane distribution：public immutable contracts、local product
  metric clustering、complete-link hypotheses、recurrence/transition status、20
  focused mathematical tests 已实现并通过专项 review。Campaign 06 的首轮
  common-grid application 因 single-run recurrence identifiability 返回
  `method_indeterminate`；这不改变 API 的 typed non-candidate behavior。
- **Ground Normal from vanishing-direction evidence**: robust interpretation is implemented and reviewed
  at [`tasks/vanishing_direction_ground_normal/`](tasks/vanishing_direction_ground_normal/);
  single-source/discrete/role-aware interpretations plus person-line evidence
  are implemented at
  [`tasks/vanishing_direction_ground_method_ownership/`](tasks/vanishing_direction_ground_method_ownership/).
  Equal-line and fixed source-total weighted vertical-line interpretations are
  implemented at [`tasks/equal_line_vertical_ground_normal/`](tasks/equal_line_vertical_ground_normal/)
  and [`tasks/source_weighted_vertical_ground_normal/`](tasks/source_weighted_vertical_ground_normal/).
  Registered given-camera stages and centered-focal Ours Ground composition are implemented at
  [ours_ground_baselines.md](ours_ground_baselines.md); empirical selection is
  owned by `hjlib-experiments` Campaign 04. Synthetic contracts and an 8/8
  VirtualCrowd V4 artifact replay are green.
  This repo owns the locally-horizontal ground method while camera-solver owns
  direction selection and geometry owns rasterization.
- remote: <https://github.com/YrralH/hjlib-ground-solver>
- deps: hjlib-camera + hjlib-camera-solver + hjlib-geometry + hjlib-smpl；精确
  revisions 只以 `pyproject.toml` 的当前 pins 为准，不在设计摘要复制第二份。

## 7. What's open

- **Ours Ground continuation**: the identity-aware offset experiment remains
  pending and must reuse the frozen selection seam without changing
  `ground_offset_baseline001`.

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
