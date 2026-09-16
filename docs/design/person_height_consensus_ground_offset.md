# Person-Height Consensus Ground Offset

> **Method state: incomplete; do not use before redesign.** The implementation
> consistently realizes V1, but V1's one-scene-mean-height scale constraint was
> not an accepted definition for jointly handling different latent per-person
> heights. This document now describes frozen historical architecture.

## 契约

`estimate_ground/by_person_height_consensus/` 负责 dataset-independent NumPy
几何：从已经选好的具名双侧 shoulder/hip/knee/ankle 2D observations，得到
camera-frame Ankle Plane。power-8 measurement 与 public result 保留原 observation
轴；退化 pose 以 `NaN` 加 measurement validity mask 表示，result 再给出完整的
projective-geometry validity mask、逐人 support/eligibility、逐人 `H/D`、scene
aggregation mask、等效身高与 plane。

Dataset loading、skeleton slot lookup、confidence/window/motion selection、Ground
Normal estimation、mean-height calibration、GT support 与 evaluation 都由 caller
负责，因此 solver 不依赖 tracking 或 VirtualCrowd。

## 算法决定

每条 observation 以 torso length 加两条 articulated leg-chain length 的稳定 power-8
mean，定义 virtual-upright I-Pose Height。两只 ankle 分别正交投影到以 shoulder
中点为起点、由 `K @ ground_normal` 决定的 image line；两个投影点的中点是
corrected bottom。解析 projective inversion 给出有效的 `H/D`。

每个人内部对 frame ratios 等权取 middle-90% mean；eligible people 再按人等权取
middle-90% mean。caller 提供 trim 后 retained people 的平均 equivalent height，
用 `D = H_mean / q_scene` 确定原本不可识别的 metric scale。scene trimming 后仍须
达到 `minimum_retained_person_count`。

power-8 height 是 walking pose 的 heuristic projected upright equivalent，不声称是
精确 anthropometric height。无效 projective geometry 逐行失效；person 或 scene
support 不足时 fail loudly。

## 扩展边界

height construction 的 observation 语义变化时，应增加 sibling construction。
只有 identity aggregation 与 scale recovery 的 invariants 不变时才留在本 package。
新的 dataset selector、joint vocabulary 或 calibration objective 属于 dataset / experiment
owner。实现任务记录见
[tasks/person_height_consensus_ground_offset](tasks/person_height_consensus_ground_offset/README.md)。
