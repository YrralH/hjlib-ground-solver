# hjlib-ground-solver

地面"求解"侧（the ground *solving* side）：从 SMPL pillars / top-bottom 关键点 /
深度图等输入**主动推断**地面参数与几何，并从 2D HVIP 反推 3D 信息。从 monolith
`lib_dynamic_hvip/ground/` 的 solver 子树迁出（file-mapping port）。

与 [`hjlib-geometry`](../hjlib-geometry) 的边界：地面的**被动使用**（reverse_project /
transform / by_param 等已知地面后的操作）在 hjlib-geometry；本仓只装**求解**侧。

## 安装

```bash
conda activate hjlib_py312
cd hjlib-ground-solver
pip install -e .
```

依赖两个 sibling 包（已 `pip install -e .` 到同一 env）：`hjlib-geometry`、`hjlib-smpl`。

## 最小示例

从一组站立行人的 top（双肩中点）/ bottom（双踝中点）2D 关键点 + 相机内参，
求地面参数 `(A, B, C, D)`（`Ax + By + Cz + D = 0`，相机坐标系）：

```python
import numpy as np
from hjlib_ground_solver import solve_ground_param_by_top_bottom_given_K

array_top = ...     # (N, 2) pixel, 双肩中点
array_bottom = ...  # (N, 2) pixel, 双踝中点
K = ...             # (3, 3) 相机内参

ground, loss = solve_ground_param_by_top_bottom_given_K(
    array_top, array_bottom, K, H_prior=1.35
)
assert ground.shape == (4,)
```

## 文档

- 怎么调用：[docs/usage/README.md](docs/usage/README.md)
- 怎么修改 / 设计点：[docs/design/README.md](docs/design/README.md)
- 迁移记录（file-mapping + 死代码清理）：[docs/design/migration.md](docs/design/migration.md)
- 未迁部分（Tier-2 / DRU-9）：[docs/design/handoff.md](docs/design/handoff.md)

## 链接

- GitHub remote: <https://github.com/YrralH/hjlib-ground-solver>
- 家族入口与约定：[`Code_as_Libs/CLAUDE.md`](../CLAUDE.md)
