# Person-Height Consensus Ground Offset

State: **incomplete; do not use before method redesign**, user disposition
2026-09-15. The reviewed implementation below is retained as frozen historical
V1 evidence. Its one-scene-mean-height scale constraint was implemented before
the intended joint constraint over different per-person heights was accepted.

## Requirements

`hjlib-ground-solver` 负责 dataset-independent geometry 与 aggregation。实现住在
`src/hjlib_ground_solver/estimate_ground/by_person_height_consensus/`，作为 additive
public API；不得修改 `solve_ground_offset`、`ground_offset_baseline001` 或已废弃的
`person_ankle_plane` V1。

输入是同一 uncropped pixel frame 中的具名双侧 shoulder/hip/knee/ankle 2D points、
person/frame identity、nonsingular K、camera-up unit GN，以及 positive retained-person
trimmed-mean equivalent-height prior。dataset loading、window selection、GT ground、
evaluation 与 hyperparameter calibration 留在 library 外部。

## Mathematical Architecture

对每条 pose，以 shoulder/hip midpoint 定义 torso。左右 leg length 各为
hip-to-knee 加 knee-to-ankle，power-8 I-Pose Height 为：

```text
h = torso + ((left_leg^8 + right_leg^8) / 2)^(1/8).
```

令 `v = K n`，`s` 为 shoulder midpoint。projective vertical image direction 为
`d = (v_xy - s v_z) / ||v_xy - s v_z||`，方向符号朝 observed ankle midpoint。
两只 ankle 分别正交投影到 `{s + alpha d}`，两个投影点的中点是 corrected bottom
`b`。该构造只使用 2D joints、K 与 supplied GN。

power-8 `h` 是 walking pose 的 virtual-upright heuristic，不是精确 physical 3D
height。对 `r = K^-1 [b_x,b_y,1]`、plane `n dot X + D = 0` 和 `q = H/D`，解析式为：

```text
q = h * (-1 / (n dot r)) / (||v_xy - b v_z|| - h v_z).
```

使用固定 `1e-12` geometry epsilon。非正 I-Pose Height、shoulder 处未定义的
vertical direction、非 positive-depth ray、非正 denominator/ratio 或 virtual-top depth
都只令该 observation 失效；measurement 与 result 分别以 `NaN + valid mask` 保留原轴。

每个人对有效 frame ratios 排序，从每端裁 `floor(0.05 N)` 后等 frame 求 mean。
eligible people 再按人等权执行同样的 middle-90% mean。scene trimming 后必须仍达到
caller-supplied minimum retained-person count。给定 retained-person trimmed mean height
`H_mean` 后：

```text
D = H_mean / q_scene
H_person = D * q_person.
```

没有 `H_mean` 时绝对 scale 不可识别。输入 `(person_id, frame_id)` 唯一且按升序；
result 检查 observation/person masks、support counts、ratios、heights 与 metric scale
互相一致。

## Code Architecture

- `contract.py`：immutable named-joint observations、config、measurement/result contracts；
  不复制 COCO vocabulary。
- `ipose.py`：stable power-8 I-Pose 与 corrected bottom；不做 confidence selection。
- `solve.py`：projective inversion、两级 middle-90% consensus 与 mean-height scale。
- package 与 repository `__init__.py`：re-export additive API。

模块不 import dataset、tracking、evaluation、experiment 或 visualization owner；实现为
deterministic CPU NumPy。

## Smoke-Test Standard

synthetic virtual segment 必须闭合已知 `H/D`，同时明确这不证明 real-pose anthropometric
accuracy。测试覆盖 equal-person weighting、tail count、退化 row 逐行失效、post-trim
minimum、support masks、corrected bottom，以及旧 registered offset config 不变。

## Modification History

- 2026-09-14：建立数学与 code architecture；专项 review 澄清 projective vertical、
  virtual-upright 语义、identity/support 与 stable power mean。
- 2026-09-15：最终 behavior review 补齐退化 row mask、post-trim minimum 与 result
  cross-field invariants。
