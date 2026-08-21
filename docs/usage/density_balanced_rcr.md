# 用 density-balanced weights 求 top/bottom RCR 地面

手里已有同一相机下的一组 `(N,2) float64` top/bottom pixels 与 `(3,3)
`float64` 内参时，可以先等权求一个 provisional normal，再建立相对空间密度并把
权重传回 RCR：

```python
from hjlib_ground_solver import (
    compute_ground_observation_kde_density,
    solve_ground_param_by_top_bottom_given_K,
)

provisional_plane, unused_objective = solve_ground_param_by_top_bottom_given_K(
    top_xy_px,
    bottom_xy_px,
    camera_K,
    H_prior=1.35,
)
density = compute_ground_observation_kde_density(
    bottom_xy_px,
    camera_K,
    provisional_plane[:3].astype('float64'),
)
weighted_plane, weighted_objective = solve_ground_param_by_top_bottom_given_K(
    top_xy_px,
    bottom_xy_px,
    camera_K,
    H_prior=1.35,
    observation_weights=density.normalized_observation_weights,
)
```

`Ground_Observation_KDE_Density` 按输入顺序保留 provisional unit-plane
coordinates、Scott-bandwidth full-covariance kernel、exact leave-one-out log
density、clip 前后 inverse-density weights、mean-one weights 和 effective sample
size。默认 pre-normalization clip 是 `[0.25,4.0]`。
所有数组为 owned、read-only `float64`。

需要显式固定局部尺度的历史探索时，也可调用
`compute_ground_observation_density(..., neighbor_count=k)` 得到 kNN record；KDE
是不需要手选 `k` 的当前通用入口。两者都不做 finite-support boundary correction。

该函数不选择 observations、不读取 dataset、不使用 GT，也不迭代 normal/density。
RCR 的两轮 angular trimming 仍保持未加权 membership；同一权重进入最终 normal
fit 与 D objective。比较不同 density methods 时，应固定完全相同的 observations。
这是一种 density-coefficient balancing heuristic；未归一化 homogeneous line scale
仍影响最终 normal-fit leverage，所以不能把 ESS 或 mean-one weights 解读成严格的
最终空间贡献均匀性。

| 我有的数据 / 目标 | 调用 |
| --- | --- |
| top/bottom + K，直接等权 baseline | `solve_ground_param_by_top_bottom_given_K(...)` |
| bottom + K + provisional normal，自动 bandwidth KDE | `compute_ground_observation_kde_density(...)` |
| bottom + K + provisional normal，固定 kNN 尺度 | `compute_ground_observation_density(..., neighbor_count=k)` |
| density record，运行 weighted RCR | `solve_ground_param_by_top_bottom_given_K(..., observation_weights=record.normalized_observation_weights)` |
| 只需历史行为 | 不传 `observation_weights` 或显式传 `None` |
