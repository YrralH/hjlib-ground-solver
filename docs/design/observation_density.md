# Observation Density and Weighted RCR

## Contract

`compute_ground_observation_kde_density` 是无需手选邻居数的通用入口。它在同一
provisional unit-plane 2D coordinates 上，以 sample full covariance 和 Scott factor
`N^(-1/6)` 定义 Gaussian kernel，计算 diagonal-excluded exact LOO log density，再在
log space 形成 inverse-density weights。实现用 centered Cholesky whitening 与
`[<=256,N]` chunked `cdist`，不会持久化 `N x N` matrix。record 会从 stored
coordinates 独立重算 Scott covariance、clip、normalization 与 ESS closure。full-rank
covariance 是显式前提；finite-support boundary bias 不做校正。

下面的 kNN contract 保留给需要显式局部尺度的探索与历史结果。

`compute_ground_observation_density` 把 ordered bottom-image observations 通过
camera rays 投到由 provisional unit normal 定义、距原点 1 个任意单位的共同平面。
它在该平面的 deterministic orthonormal 2D basis 上用 `cKDTree` 计算第 `k` 个邻居
半径：

```
rho_i = k / (pi * r_i^2)
relative_i = (1 / rho_i) / median(1 / rho)
weight_i = clip(relative_i, lower, upper) / mean(clip(relative, lower, upper))
```

零半径 duplicate 使用正半径中位数的 `1e-6` 作为 floor；若全部 raw radii 为零但
存在多个 unique locations，则改用 unique-location nearest-neighbor distance 的中位数。
只有一个 unique location 的 spatially collapsed population 才失败。record 完整保存 raw/effective radius、density、relative/clip/normalized
weights、normalization 与 ESS，构造时重算 closure，并 defensive-copy 为 read-only。

## Weighted solver boundary

`observation_weights` 是 RCR public functions 的 keyword-only optional 参数。旧
`D_init/device/flag` positional slots 保持原位；新 distance bounds 与 weights 只允许
keyword 传入。weights 必须是与 observations 对齐的 finite positive `(N,) float64` NumPy array；torch loss
边界要求相同 dtype/device。历史 `None` 路径保持不变。

两轮 angular-bias trimming 故意不加权并先固定 membership；对应 weights 用同一
masks 对齐。最后的 vanishing-line nullspace 对 `sqrt(weight_i) * line_i` 做 SVD。
D search 在完整原始 population 上对 relative-length 与 normalized-pixel losses 做
weighted mean。权重的全局 scale 不改变结果。

这里均衡的是 empirical-density coefficient，不是最终 SVD leverage 的严格等分。
继承的 homogeneous line magnitude 仍会影响 normal fit；ESS 也只描述 weights 的集中度。

## Ownership and extension

本仓只持有 method-neutral density IR 与 numerical fitting。dataset-specific joint/
bbox confidence selection、GT、sampling、result reduction 与 visualization 属调用方。
新增 density method 时应返回独立可检查 evidence，而不是只返回 opaque weights；
若需要 iterative fixed point 或 learned density，应建立新的 sibling design，不能默默
改变本 one-pass contract。
