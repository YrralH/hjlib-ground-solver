# 显式 vertex subset 的高度与速度观测

## 何时使用

调用方已拥有 `(B,V,3)` mesh 和一个明确的 vertex index tensor，需要按 chunk
计算每帧最低高度、相邻帧 median per-vertex speed，并把最后一帧作为下一 chunk
carry 时，使用 `compute_vertex_subset_observation_chunk`。本 API 不理解 SMPL
topology；脚掌 index 可由 `hjlib-smpl.foot_support` 提供。

```python
import torch

from hjlib_ground_solver.estimate_ground.by_vertex_subset_observation import (
    compute_vertex_subset_observation_chunk,
)

indices = torch.tensor(plantar_indices, dtype=torch.long, device=vertices.device)
chunk = compute_vertex_subset_observation_chunk(
    vertices,
    indices,
    up_axis_index=2,
    frame_rate_in_hz=60.0,
    previous_vertex_positions=previous,
    previous_vertex_indices=previous_indices,
)
previous = chunk.final_vertex_positions.detach()
previous_indices = chunk.final_vertex_indices
```

返回值：

| 字段 | shape | 含义 |
|---|---:|---|
| `minimum_height` | `(B,)` | subset 在指定 up axis 的每帧 minimum |
| `interval_median_speed` | `(B-1,)` 或有 carry 时 `(B,)` | 每个时间间隔内，各 vertex 3D speed 的 ordered median |
| `final_vertex_positions` | `(N,3)` | 独立 clone；供下一 chunk 使用，不保留整块 storage |
| `final_vertex_indices` | `(N,)` | 独立 ordered identity；下一 chunk 必须原样交回 |

偶数 `N` 的 median 是两个中间值的安全 midpoint。函数保留 autograd；inference
caller 自行决定是否对 carry `detach()`。输入必须 finite floating tensor；index
必须同 device、1D nonempty unique `torch.long`；FPS 必须是 finite-positive Python
`int` 或 `float`（排除 `bool`）。carry 的 positions 和 ordered indices 必须
成对提供且 identity/order 完全相同。所有派生高度、距离、速度也必须 finite。

## 选择入口

| 输入 | 使用 |
|---|---|
| full mesh，确实要全身最低点 | `compute_per_frame_mesh_minimum_height` |
| 明确的脚掌/手/局部 mesh indices | `compute_vertex_subset_observation_chunk` |
| 只有 SMPL family，需要先找脚掌 indices | 先用 `hjlib-smpl.foot_support` |
| 已有 scalar height track，要低尾分析 | `summarize_mesh_lower_envelope` / peeling |
