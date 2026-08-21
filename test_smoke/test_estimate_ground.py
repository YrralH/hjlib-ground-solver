'''Smoke tests for the estimate_ground (by_kp_rcr) public API.

Synthesizes a set of upright people standing on the world z=0 plane and
projects their bottom (ankle) / top (shoulder) points into a camera looking
down, so the top-bottom segments form physically-consistent vertical lines.
'''

from typing import Tuple
from unittest.mock import patch

import numpy as np
import pytest
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


def make_top_bottom(
        n: int = 24,
        h_prior: float = 1.35,
        jitter_px: float = 1.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(7)
    xy = (np.random.rand(n, 2) - 0.5) * 2.0
    bottom_world = np.concatenate([xy, np.zeros((n, 1))], axis=1)
    top_world = bottom_world + np.array([0.0, 0.0, h_prior])
    R, T = make_R_T()
    K = make_K()
    array_bottom = project(bottom_world, R, T, K)
    array_top = project(top_world, R, T, K)
    # small pixel jitter so the bias-based outlier filter is non-degenerate
    array_bottom = (
        array_bottom + np.random.randn(n, 2).astype(np.float32) * jitter_px
    ).astype(np.float32)
    array_top = (
        array_top + np.random.randn(n, 2).astype(np.float32) * jitter_px
    ).astype(np.float32)
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


def test_get_KN_with_filter_is_permutation_invariant() -> None:
    array_top, array_bottom = make_top_bottom(80)
    xt = to_homogeneous_cols(array_top)
    xb = to_homogeneous_cols(array_bottom)
    permutation = np.random.default_rng(19).permutation(xt.shape[1])

    KN = get_KN_with_filter(xb, xt, prop_filter=0.24, times_filter=2)
    KN_permuted = get_KN_with_filter(
        xb[:, permutation],
        xt[:, permutation],
        prop_filter=0.24,
        times_filter=2,
    )

    np.testing.assert_allclose(KN_permuted, KN, rtol=1e-6, atol=1e-6)


def test_get_KN_with_filter_retains_equal_bias_oracle() -> None:
    array_top, array_bottom = make_top_bottom(80, jitter_px=0.0)
    KN = get_KN_with_filter(
        to_homogeneous_cols(array_bottom),
        to_homogeneous_cols(array_top),
        prop_filter=0.24,
        times_filter=2,
    )
    np.testing.assert_allclose(KN, np.array([256.0, 256.0, 1.0]), atol=1e-4)


def test_weighted_KN_and_trim_weight_alignment() -> None:
    array_top, array_bottom = make_top_bottom(80)
    xt = to_homogeneous_cols(array_top)
    xb = to_homogeneous_cols(array_bottom)
    weights = np.linspace(0.2, 3.0, 80, dtype=np.float64)
    A = np.cross(xb.T, xt.T)
    unused_u, unused_s, vh = np.linalg.svd(
        A * np.sqrt(weights)[:, None],
        full_matrices=False,
    )
    del unused_u, unused_s
    expected = vh[-1] / vh[-1, 2]
    actual = get_KN(xb, xt, observation_weights=weights)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    filtered = get_KN_with_filter(
        xb,
        xt,
        prop_filter=0.24,
        times_filter=2,
        flag_ret_filtered_result=True,
    )
    assert isinstance(filtered, tuple)
    unused_kn, xb_filtered, xt_filtered = filtered
    del unused_kn
    retained_indices = []
    for column in range(xb_filtered.shape[1]):
        matches = np.flatnonzero(np.all(xb.T == xb_filtered[:, column], axis=1))
        assert matches.size == 1
        retained_indices.append(int(matches[0]))
    expected_filtered = get_KN(
        xb_filtered,
        xt_filtered,
        observation_weights=weights[np.asarray(retained_indices, dtype=np.int64)],
    )
    actual_filtered = get_KN_with_filter(
        xb,
        xt,
        prop_filter=0.24,
        times_filter=2,
        observation_weights=weights,
    )
    np.testing.assert_allclose(actual_filtered, expected_filtered, rtol=1e-12, atol=1e-12)

    scaled = get_KN_with_filter(
        xb,
        xt,
        prop_filter=0.24,
        times_filter=2,
        observation_weights=weights * 19.0,
    )
    np.testing.assert_allclose(scaled, actual_filtered, rtol=1e-10, atol=1e-10)


def test_get_KN_rejects_rank_degenerate_and_infinite_vp() -> None:
    bottom_repeated = np.tile(
        np.array([[100.0], [200.0], [1.0]], dtype=np.float64),
        (1, 6),
    )
    top_repeated = np.tile(
        np.array([[100.0], [150.0], [1.0]], dtype=np.float64),
        (1, 6),
    )
    with pytest.raises(ValueError, match='rank-degenerate'):
        get_KN(bottom_repeated, top_repeated)

    x = np.arange(6, dtype=np.float64) * 50.0 + 100.0
    y = np.arange(6, dtype=np.float64) * 20.0 + 200.0
    bottom_parallel = np.stack([x, y, np.ones_like(x)])
    top_parallel = np.stack([x, y - 50.0, np.ones_like(x)])
    with pytest.raises(ValueError, match='at infinity'):
        get_KN(bottom_parallel, top_parallel)


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

    K_skew = K.clone()
    K_skew[0, 1] = 125.0
    xyz_skew = uv_to_xyz_via_ground_torch(xb, ground, K_skew)
    plane_residual = ground[:3] @ xyz_skew + ground[3]
    torch.testing.assert_close(plane_residual, torch.zeros_like(plane_residual))


def test_weighted_projection_loss_matches_hand_reduction() -> None:
    xb = torch.tensor(
        [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]],
        dtype=torch.float32,
    )
    xt = torch.tensor(
        [[0.0, 0.0], [2.0, 4.0], [1.0, 1.0]],
        dtype=torch.float32,
    )
    pred = torch.tensor(
        [[0.0, 0.0], [1.0, 3.0], [1.0, 1.0]],
        dtype=torch.float32,
    )
    weights = torch.tensor([1.0, 3.0], dtype=torch.float32)
    weighted_ret = get_projection_loss(
        xb,
        xt,
        pred,
        observation_weights=weights,
    )
    assert len(weighted_ret) == 3
    unused_vec, loss_mod, loss_pixel = weighted_ret
    del unused_vec
    expected_each = torch.tensor([0.5, 0.25], dtype=torch.float32)
    expected = torch.sum(weights * expected_each) / torch.sum(weights)
    torch.testing.assert_close(loss_mod, expected)
    torch.testing.assert_close(loss_pixel, expected)
    assert not torch.isclose(expected, torch.mean(expected_each))
    scaled_ret = get_projection_loss(
        xb,
        xt,
        pred,
        observation_weights=weights * 7.0,
    )
    assert len(scaled_ret) == 3
    torch.testing.assert_close(scaled_ret[1], loss_mod)
    torch.testing.assert_close(scaled_ret[2], loss_pixel)


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


def test_solve_ground_recovers_exact_oracle() -> None:
    array_top, array_bottom = make_top_bottom(80, jitter_px=0.0)
    ground, objective = solve_ground_param_by_top_bottom_given_K(
        array_top,
        array_bottom,
        make_K(),
        H_prior=1.35,
    )
    if ground[2] > 0.0:
        ground = -ground
    np.testing.assert_allclose(ground[:3], np.array([0.0, 0.0, -1.0]), atol=1e-5)
    assert abs(float(ground[3]) - 3.0) < 1e-5
    assert float(objective) < 1e-5


def test_all_one_weighted_solver_matches_unweighted() -> None:
    array_top, array_bottom = make_top_bottom(80)
    unweighted_plane, unweighted_objective = solve_ground_param_by_top_bottom_given_K(
        array_top,
        array_bottom,
        make_K(),
        H_prior=1.35,
    )
    weighted_plane, weighted_objective = solve_ground_param_by_top_bottom_given_K(
        array_top,
        array_bottom,
        make_K(),
        H_prior=1.35,
        observation_weights=np.ones(80, dtype=np.float64),
    )
    np.testing.assert_allclose(weighted_plane, unweighted_plane, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        weighted_objective,
        unweighted_objective,
        rtol=1e-6,
        atol=1e-6,
    )


def test_solve_ground_rejects_invalid_and_boundary_cases() -> None:
    array_top, array_bottom = make_top_bottom(8)
    with pytest.raises(ValueError, match='at least three'):
        solve_ground_param_by_top_bottom_given_K(
            array_top[:2],
            array_bottom[:2],
            make_K(),
        )
    degenerate_top = array_bottom.copy()
    with pytest.raises(ValueError, match='nondegenerate'):
        solve_ground_param_by_top_bottom_given_K(
            degenerate_top,
            array_bottom,
            make_K(),
        )

    xb = to_homogeneous_cols(array_bottom)
    xt = to_homogeneous_cols(array_top)
    with pytest.raises(ValueError, match='search boundary'):
        solve_D_search(
            xb,
            xt,
            np.array([0.0, 0.0, -1.0], dtype=np.float32),
            make_K(),
            distance_min=100.0,
            distance_max=101.0,
            distance_step=0.1,
        )


def test_solve_ground_forwards_device() -> None:
    array_top, array_bottom = make_top_bottom(8)
    device = torch.device('cuda:5')
    path = (
        'hjlib_ground_solver.estimate_ground.by_kp_rcr.solve_by_top_bot.'
        'process_solve_by_top_bot_given_K.solve_D_search'
    )
    with patch(path) as solve_mock:
        solve_mock.return_value = (
            np.array([0.0, 0.0, -1.0, 3.0], dtype=np.float32),
            np.array(0.0, dtype=np.float32),
        )
        solve_ground_param_by_top_bottom_given_K(
            array_top,
            array_bottom,
            make_K(),
            device_solve=device,
        )
    assert solve_mock.call_args.kwargs['device'] == device


def test_solve_ground_forwards_weights_to_normal_and_distance() -> None:
    array_top, array_bottom = make_top_bottom(8)
    weights = np.linspace(0.5, 1.5, 8, dtype=np.float64)
    process_module = (
        'hjlib_ground_solver.estimate_ground.by_kp_rcr.solve_by_top_bot.'
        'process_solve_by_top_bot_given_K'
    )
    with (
            patch('%s.get_KN_with_filter' % process_module) as normal_mock,
            patch('%s.solve_D_search' % process_module) as distance_mock,
        ):
        normal_mock.return_value = np.array([256.0, 256.0, 1.0], dtype=np.float64)
        distance_mock.return_value = (
            np.array([0.0, 0.0, -1.0, 3.0], dtype=np.float32),
            np.array(0.0, dtype=np.float32),
        )
        solve_ground_param_by_top_bottom_given_K(
            array_top,
            array_bottom,
            make_K(),
            observation_weights=weights,
        )
    np.testing.assert_array_equal(
        normal_mock.call_args.kwargs['observation_weights'],
        weights,
    )
    np.testing.assert_array_equal(
        distance_mock.call_args.kwargs['observation_weights'],
        weights,
    )


def test_low_level_weight_validation() -> None:
    array_top, array_bottom = make_top_bottom(8)
    xb = to_homogeneous_cols(array_bottom)
    xt = to_homogeneous_cols(array_top)
    wrong_dtype = np.ones(8, dtype=np.float32)
    with pytest.raises(TypeError, match='float64'):
        get_KN(xb, xt, observation_weights=wrong_dtype)
    invalid = np.ones(8, dtype=np.float64)
    invalid[0] = 0.0
    with pytest.raises(ValueError, match='positive'):
        get_KN_with_filter(xb, xt, observation_weights=invalid)
    with pytest.raises(TypeError, match='float64'):
        solve_D_search(
            xb,
            xt,
            np.array([0.0, 0.0, -1.0], dtype=np.float32),
            make_K(),
            observation_weights=wrong_dtype,
        )
    with pytest.raises(ValueError, match='positive'):
        solve_ground_param_by_top_bottom_given_K(
            array_top,
            array_bottom,
            make_K(),
            observation_weights=invalid,
        )
    with pytest.raises(ValueError, match='positive'):
        get_projection_loss(
            torch.from_numpy(xb).to(torch.float32),
            torch.from_numpy(xt).to(torch.float32),
            torch.from_numpy(xt).to(torch.float32),
            observation_weights=torch.tensor(
                [0.0] + [1.0] * 7,
                dtype=torch.float32,
            ),
        )


def test_legacy_positional_and_explicit_none_parity() -> None:
    array_top, array_bottom = make_top_bottom(24)
    xb = to_homogeneous_cols(array_bottom)
    xt = to_homogeneous_cols(array_top)
    np.testing.assert_array_equal(get_KN(xb, xt, False), get_KN(xb, xt, False, None))
    np.testing.assert_array_equal(
        get_KN_with_filter(xb, xt, 0.24, 2, False),
        get_KN_with_filter(xb, xt, 0.24, 2, False, None),
    )
    torch_xb = torch.from_numpy(xb).to(torch.float32)
    torch_xt = torch.from_numpy(xt).to(torch.float32)
    projection_old = get_projection_loss(torch_xb, torch_xt, torch_xt, False, 0.9)
    projection_none = get_projection_loss(
        torch_xb,
        torch_xt,
        torch_xt,
        False,
        0.9,
        None,
    )
    assert len(projection_old) == len(projection_none) == 3
    torch.testing.assert_close(projection_old[1], projection_none[1])
    torch.testing.assert_close(projection_old[2], projection_none[2])
    ground_normal = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    distance_old = solve_D_search(
        xb,
        xt,
        ground_normal,
        make_K(),
        1.35,
        10.0,
        False,
        0.9,
        torch.device('cpu'),
    )
    distance_none = solve_D_search(
        xb,
        xt,
        ground_normal,
        make_K(),
        1.35,
        10.0,
        False,
        0.9,
        torch.device('cpu'),
        observation_weights=None,
    )
    assert len(distance_old) == len(distance_none) == 2
    np.testing.assert_array_equal(distance_old[0], distance_none[0])
    np.testing.assert_array_equal(distance_old[1], distance_none[1])
    plane_old, objective_old = solve_ground_param_by_top_bottom_given_K(
        array_top,
        array_bottom,
        make_K(),
        1.35,
        10.0,
        torch.device('cpu'),
        False,
    )
    plane_none, objective_none = solve_ground_param_by_top_bottom_given_K(
        array_top,
        array_bottom,
        make_K(),
        1.35,
        10.0,
        torch.device('cpu'),
        False,
        observation_weights=None,
    )
    np.testing.assert_array_equal(plane_old, plane_none)
    np.testing.assert_array_equal(objective_old, objective_none)


def smoke_test_estimate_ground() -> None:
    test_get_KN_and_bias()
    test_get_KN_with_filter()
    test_get_KN_with_filter_is_permutation_invariant()
    test_get_KN_with_filter_retains_equal_bias_oracle()
    test_weighted_KN_and_trim_weight_alignment()
    test_get_KN_rejects_rank_degenerate_and_infinite_vp()
    test_projection_loss_and_uv_to_xyz()
    test_weighted_projection_loss_matches_hand_reduction()
    test_solve_D_search()
    test_solve_ground_param_by_top_bottom_given_K()
    test_solve_ground_recovers_exact_oracle()
    test_all_one_weighted_solver_matches_unweighted()
    test_solve_ground_rejects_invalid_and_boundary_cases()
    test_solve_ground_forwards_device()
    test_solve_ground_forwards_weights_to_normal_and_distance()
    test_low_level_weight_validation()
    test_legacy_positional_and_explicit_none_parity()
    print('[OK] estimate_ground')


if __name__ == '__main__':
    smoke_test_estimate_ground()
