# Handoff —— hjlib-ground-solver 未迁部分

本次开仓（2026-05-27）只迁 **Tier-1**。以下为留给后续 session 的跨仓 TODO，
均已同步到 `migration_progress/index.md` 的 Cross-lib TODO 表。

## DRU-9 —— Tier-2: `for_fixed_camera_video_by_rcr.py`

monolith `ground/estimate_ground/for_fixed_camera_video_by_rcr.py` 是 RCR pipeline 的
**业务入口**（固定相机视频按 RCR 求地面），是 ground solver 子树里唯一真耦合
`detection` 的文件。

- 真活依赖（grep 核实）：
  - `detection.remove_duplication.remove_duplicate_keypoints_kdtree`
  - `detection.filter.filter_by_straight_pose`
  - 本仓已迁的 `solve_ground_param_by_top_bottom_given_K`（intra）
- 另含一批死/重 import（vis_*/pyrender/dataset.prod.multi_dynamics）+ `local_setting`，
  与 Tier-1 同类，迁时按需清。
- `detection` 两函数的下游缺口仅 `similarity_kp`（其余 dep 几乎全已迁）。

三选项（开仓 / 触发时拍板）：
1. 等 `hjlib-detection` 开仓后，本仓 dep 之；
2. 把那 2 个 detection 函数 lift / inline 进本仓（缺口 `similarity_kp` 一并 lift）；
3. 该文件是 pipeline 业务入口，按调用方需求再 lift（可能根本不进 lib）。

## DRU-10 —— `hvip/{get_hvip_by_linear, get_hvip_by_ray_cast}` 归 hjlib-geometry

经 2026-05-27 逐行核实为**纯 use**（签名输入已含 `ground_parameters`，不含求解）：
- `get_hvip_by_ray_cast` 仅 dep 已迁的 `by_param`；`get_hvip_by_linear` 仅 numpy。
- 应延续 DRU-7 的 use/solve 切分，**增量 port 到已 Phase 2 verified 的 hjlib-geometry**
  （需补 parity case），**不进本仓**。本仓需要时 `from hjlib_geometry import ...`。
- `hvip/get_3d_info_from_hvip_2d`（body 含 solve，现算地面）已留本仓（Tier-1）。

## DRU-8 —— 仓名拍板

✅ **closed** —— 仓名定为 `hjlib-ground-solver`（本仓即此）。

## `get_ground_by_smpls_on_the_ground` 数据测试

需真实 SMPL_Full 模型 + verts，smoke 无法覆盖。待 dataset-raw-uplift 出第一份真实
SMPL 序列后，在 `hjlib-migration-tests/ground-solver/`（Phase 2）/ 本仓 `test/` 落地。
