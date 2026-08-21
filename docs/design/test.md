# 测试布局 —— hjlib-ground-solver

家族测试两棵树政策见
[test_layout.md](../../../hjlibm/docs/hjlib_standard/test_layout.md)；本页只记本仓实例。

## test_smoke/（无外部数据，处处可跑）

```
test_smoke/
├── test_segment_area.py            cluster_pixel_features / depth feature / depth map / depth->ground
├── test_ground_geometry.py         by_pillars / by_points / plane fits / world_space geometry
├── test_ground_param_and_hvip.py   get_ground_param_in_world_space(_with_extrinsic) / get_3d_info_from_hvip_2d
├── test_estimate_ground.py         get_KN / get_bias / get_KN_with_filter / projection_loss / uv_to_xyz / solve_D_search / 顶层入口
├── test_observation_density.py     provisional-plane coordinates / exact-LOO KDE / kNN density / clipping / ESS / validation / immutability
├── test_mesh_lower_envelope.py     exact coverage / contamination / ties / run / reducer / re-export
├── test_mesh_lower_envelope_peeling.py  first eligible low-prefix / iteration / budgets / oracle
├── test_static_foot_humor.py       literal upstream oracle / cluster priority / terrain / validation
├── test_static_foot_plantar_humor.py physical-speed gate / noise exclusion / span / validation
├── test_hj_derived_plantar_zmin.py explicit nonofficial provenance / zmin ties / validation
├── test_vertex_subset_observation.py chunk equality / union / median / carry / autograd / overflow
├── test_all_func.py                master runner (import 各 smoke_test_*)
└── clean_test_data.py              LIST_PATH_CLEAN (当前空, 全程 in-memory 无产物)
```

- 每个 topic 文件同时暴露 `test_*`（pytest 发现）与一个 `smoke_test_<topic>()` 编排入口。
- 合成输入策略：
  - depth / cluster：随机 numpy / torch 张量。
  - 几何：散布在 `z=0` 平面附近的 3D 点（**带小 z jitter**，否则 lstsq 离群过滤退化）。
  - estimate_ground / hvip：构造"相机俯视 `z=0` 地面"的 RT + K，把站立行人的
    top/bottom 3D 点投影到 2D（**加 ~1.5px 抖动**，否则消失点过滤会因 bias 全相等而清空）。
    `get_3d_info_from_hvip_2d` 用主点像素 → 反投影回世界原点，断言 `z≈0`。
  - weighted RCR：hand-reduced weighted normal/D objective、all-one/`None` parity、
    两轮 trim weight alignment、所有公开 low-level validation 与 top-level forwarding。
  - observation density：exact LOO/chunk oracle、Scott covariance closure、nonuniform grid hand oracle、provisional-normal scale/sign invariance、
    duplicate radius floor、clip/mean-one/ESS closure、输入 independence、
    read-only record 与 degenerate/invalid failure。
  - mesh lower envelope：合成 `(B,V,3)` torch mesh 与 `T=1000` contamination
    oracle；覆盖 exact Decimal discard count、short-sequence 离散边界、ties、连续/
    稀疏低尾、autograd、dtype/device、输入不变和三级 public re-export。
  - mesh lower-envelope peeling：isolated/tied/two-group low prefixes、first-eligible
    selection、fixed-slot empty reference、ties、round/removal equality 与 stop priority、
    temporal permutation、极端 finite 派生 overflow、immutable records、三级 re-export，
    以及 100 个 deterministic randomized series 对 repeated-sort oracle。
  - static-foot HuMoR：float32/64 literal upstream oracle、strict displacement gate、
    sub-threshold 位移不参与候选排序、left-before-right pooling、duplicate root frame
    去重、pure/mixed noise、equal-median label tie、terrain 三重 strict boundary、
    transform invariant、derived overflow、immutable records 与三级 re-export。
  - static-foot plantar HuMoR：strict physical-speed gate、terminal interval
    repeat、left-before-right pooling、noise-only/no-contact statuses、最低 median、
    density-chain span/gap、boundary-safe vertical shift、immutable records、输入不变、
    validation 与三级 re-export。
  - vertex-subset observation：hand oracle、odd/even median、single-frame、整段与
    两 chunk + carry exact equality、左右 union height、float32/64、CPU/CUDA、
    autograd（含 previous carry）、input/storage independence、derived overflow、
    deterministic repeat 与三级 re-export。

运行：`pytest test_smoke/` 或 `python test_smoke/test_all_func.py`。

## test/（需真实数据，缺失时 FAIL 不 skip）

当前为占位。**数据依赖契约的验证主要落在
`hjlib-migration-tests/ground-solver/`（Phase 2）**，Phase 3 absorb 时再 move 回本 `test/`。

唯一明确需要真实数据的 public API：

- `get_ground_by_smpls_on_the_ground` —— 需 `SMPL_Full` 模型权重（hjlib-smpl 提供模型加载）
  + 一批真实 SMPL verts。smoke 无法合成，故未覆盖；待 dataset-raw-uplift 出第一份
  真实 SMPL 序列后，在 migration-tests / 本 `test/` 落地。

## Phase 2 验证去处

`Code_as_Libs/hjlib-migration-tests/ground-solver/`（新 session 建），含 `parity/` +
`behavior/` + `divergence/DIVERGENCE.md`，per migration_protocol.md。本仓 migration.md
的 "Migration test status" 三 checkbox 跟踪 Phase 1/2/3。
