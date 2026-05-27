'''Smoke tests for the estimate_ground (by_kp_rcr) public API.

Synthesizes a set of upright people standing on the world z=0 plane and
projects their bottom (ankle) / top (shoulder) points into a camera looking
down, so the top-bottom segments form physically-consistent vertical lines.
'''

from typing import Tuple

import numpy as np
import torch

from hjlib_ground_solver import (
    get_KN,
    get_bias_from_2D_ground_normal,
    get_KN_with_filter,
    get_projection_loss,
    uv_to_xyz_via_ground_torch,
    solve_D_search,
    solve_ground_param_by_top_bottom_given_K,
)


def make_K() -> np.ndarray:
    return np.array([[500.0, 0.0, 256.0], [0.0, 500.0, 256.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def make_R_T() -> Tuple[np.ndarray, np.ndarray]:
    R_w2c = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float32)
    C = np.array([0.0, 0.0, 3.0], dtype=np.float32)
    T = -R_w2c @ C
    return R_w2c, T


def project(points_world: np.ndarray, R: np.ndarray, T: np.ndarray, K: np.ndarray) -> np.ndarray:
    p_cam = points_world @ R.T + T
    uv = p_cam @ K.T
    uv = uv / uv[:, 2:3]
    return uv[:, 0:2].astype(np.float32)


def make_top_bottom(n: int = 24, h_prior: float = 1.35) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(7)
    xy = (np.random.rand(n, 2) - 0.5) * 2.0
    bottom_world = np.concatenate([xy, np.zeros((n, 1))], axis=1)
    top_world = bottom_world + np.array([0.0, 0.0, h_prior])
    R, T = make_R_T()
    K = make_K()
    array_bottom = project(bottom_world, R, T, K)
    array_top = project(top_world, R, T, K)
    # small pixel jitter so the bias-based outlier filter is non-degenerate
    array_bottom = (array_bottom + np.random.randn(n, 2).astype(np.float32) * 1.5).astype(np.float32)
    array_top = (array_top + np.random.randn(n, 2).astype(np.float32) * 1.5).astype(np.float32)
    return array_top, array_bottom


def to_homogeneous_cols(points: np.ndarray) -> np.ndarray:
    # (N, 2) -> (3, N) with each column [u, v, 1]
    return np.insert(points, 2, values=1.0, axis=1).T


def test_get_KN_and_bias() -> None:
    array_top, array_bottom = make_top_bottom()
    xt = to_homogeneous_cols(array_top)
    xb = to_homogeneous_cols(array_bottom)
    KN = get_KN(xb, xt)
    assert isinstance(KN, np.ndarray)
    assert KN.shape == (3,)
    assert abs(KN[2] - 1.0) < 1e-5
    list_bias = get_bias_from_2D_ground_normal(xt, xb, KN)
    assert len(list_bias) == xt.shape[1]


def test_get_KN_with_filter() -> None:
    array_top, array_bottom = make_top_bottom()
    xt = to_homogeneous_cols(array_top)
    xb = to_homogeneous_cols(array_bottom)
    KN = get_KN_with_filter(xb, xt, prop_filter=0.24, times_filter=2)
    assert isinstance(KN, np.ndarray)
    assert KN.shape == (3,)


def test_projection_loss_and_uv_to_xyz() -> None:
    array_top, array_bottom = make_top_bottom()
    xt = torch.from_numpy(to_homogeneous_cols(array_top)).to(torch.float32)
    xb = torch.from_numpy(to_homogeneous_cols(array_bottom)).to(torch.float32)
    ret = get_projection_loss(xb, xt, xt)
    assert len(ret) == 3
    _, _, loss_pixel = ret
    assert float(loss_pixel.item()) >= 0.0

    K = torch.from_numpy(make_K())
    ground = torch.tensor([0.0, 0.0, -1.0, 3.0], dtype=torch.float32)
    xyz = uv_to_xyz_via_ground_torch(xb, ground, K)
    assert xyz.shape == (3, xb.shape[1])


def test_solve_D_search() -> None:
    array_top, array_bottom = make_top_bottom()
    xb = to_homogeneous_cols(array_bottom)
    xt = to_homogeneous_cols(array_top)
    ground_normal = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    ret = solve_D_search(xb, xt, ground_normal, make_K(), H_prior=1.35)
    assert len(ret) == 2
    ground, loss = ret
    assert ground.shape == (4,)
    assert float(loss) >= 0.0


def test_solve_ground_param_by_top_bottom_given_K() -> None:
    array_top, array_bottom = make_top_bottom()
    ground, loss = solve_ground_param_by_top_bottom_given_K(
        array_top, array_bottom, make_K(), H_prior=1.35
    )
    assert ground.shape == (4,)
    assert float(loss) >= 0.0


def smoke_test_estimate_ground() -> None:
    test_get_KN_and_bias()
    test_get_KN_with_filter()
    test_projection_loss_and_uv_to_xyz()
    test_solve_D_search()
    test_solve_ground_param_by_top_bottom_given_K()
    print('[OK] estimate_ground')


if __name__ == '__main__':
    smoke_test_estimate_ground()
