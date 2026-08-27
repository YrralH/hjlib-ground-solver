'''Smokes for the registered given-camera ground baselines.'''

import numpy as np
from numpy.typing import NDArray
import pytest
from typing import cast

from hjlib_camera import Camera_Intrinsics
from hjlib_camera_solver import (
    Image_Line_Segments,
    Vanishing_Direction_Source,
    Vanishing_Point_Association,
)
from hjlib_ground_solver import (
    Ground_Normal_And_Camera_Config,
    Ground_Normal_And_Camera_Result,
    Ground_Normal_Config,
    Ground_Normal_Result,
    Ground_Offset_Config,
    Ground_Offset_Observations,
    Ground_Offset_Result,
    Ground_Offset_Selection,
    ground_normal_config,
    ground_offset_config,
    ground_normal_and_camera_config,
    select_ground_offset_observations,
    solve_D_search,
    solve_ground_normal,
    solve_ground_offset,
    solve_ground_normal_and_camera,
)
import hjlib_ground_solver as ground_solver_module


def make_intrinsics() -> Camera_Intrinsics:
    return Camera_Intrinsics(np.array([
        [600.0, 0.0, 320.0],
        [0.0, 600.0, 240.0],
        [0.0, 0.0, 1.0],
    ]), (640, 480))


def unit(value:NDArray[np.float64]) -> NDArray[np.float64]:
    return value / np.linalg.norm(value)


def endpoints_through_vp(vp_h:NDArray[np.float64], count:int) -> NDArray[np.float64]:
    if abs(float(vp_h[2])) < 1e-12:
        direction = unit(vp_h[:2])
        centers = np.array([
            [320.0, 80.0 + 320.0 * index / max(count - 1, 1)]
            for index in range(count)
        ])
        return np.stack((centers - 120.0 * direction, centers + 120.0 * direction), axis=1)
    vp_xy = vp_h[:2] / vp_h[2]
    base_y = 430.0 if vp_xy[1] < 240.0 else 50.0
    bases = np.stack(
        (np.linspace(80.0, 560.0, count), np.full((count,), base_y)),
        axis=1,
    )
    return np.stack((vp_xy[None] + 0.6 * (bases - vp_xy[None]), bases), axis=1)


def make_direction_source(
        camera_ready:bool = False,
    ) -> tuple[Vanishing_Direction_Source, NDArray[np.float64]]:
    vertical = unit(np.array([0.0, -0.81, 0.6], dtype=np.float64))
    direction_values = [
        vertical,
        np.array([1.0, 0.0, 0.0]),
        unit(np.array([0.0, vertical[2], -vertical[1]])),
    ]
    support_values = [5, 7, 8]
    if camera_ready:
        direction_values.append(unit(np.array([
            0.2,
            0.45,
            0.45 * -vertical[1] / vertical[2],
        ])))
        support_values.append(6)
    directions = np.stack(direction_values)
    supports = tuple(support_values)
    pixel_vps = (make_intrinsics().K @ directions.T).T
    endpoints = np.concatenate(tuple(
        endpoints_through_vp(vp, support)
        for vp, support in zip(pixel_vps, supports, strict=True)
    ))
    lines = Image_Line_Segments('ours-ground-normal', (640, 480), endpoints)
    association = Vanishing_Point_Association(
        'ours-ground-normal',
        lines.line_segments_sha256,
        np.repeat(np.arange(len(supports), dtype=np.int64), supports),
        pixel_vps,
    )
    return Vanishing_Direction_Source('elsed', association, lines), vertical


def make_offset_observations(
        normal:NDArray[np.float64],
        distance:float = 4.0,
    ) -> Ground_Offset_Observations:
    xz = np.array([[-2.0, 20.0], [0.0, 25.0], [2.0, 30.0]])
    y = (-distance - normal[0] * xz[:, 0] - normal[2] * xz[:, 1]) / normal[1]
    bottoms_3d = np.stack((xz[:, 0], y, xz[:, 1]), axis=1)
    tops_3d = bottoms_3d + 1.27 * normal[None]
    intrinsics = make_intrinsics()
    bottom_xy, _ = intrinsics.project_points_in_camera_frame(bottoms_3d)
    top_xy, _ = intrinsics.project_points_in_camera_frame(tops_3d)
    return Ground_Offset_Observations(
        top_xy,
        bottom_xy,
        np.full((3,), 5.0),
        np.full((3,), 0.1),
    )


def test_registered_configs_are_exact_and_unknown_ids_fail() -> None:
    normal = ground_normal_config()
    assert normal.baseline.value == 'ground_normal_baseline001'
    assert normal.camera_solver_config.minimum_cluster_support == 5
    assert normal.camera_solver_config.minimum_abs_camera_y == 0.8
    assert normal.camera_solver_config.orthogonality_tolerance_deg == 3.0
    assert normal.camera_solver_config.residual_gate_px == 0.25
    assert normal.camera_solver_config.minimum_retained_support == 5
    assert normal.camera_solver_config.maximum_refit_iterations == 20
    offset = ground_offset_config()
    assert offset.baseline.value == 'ground_offset_baseline001'
    assert offset.confidence_threshold_strict_gt == 4.3
    assert offset.ankle_ratio_threshold_strict_lt == 0.20
    assert offset.height_prior_m == 1.27
    assert (offset.distance_min_m, offset.distance_max_m, offset.distance_step_m) == (
        -5.0, 80.0, 0.1,
    )
    camera = ground_normal_and_camera_config()
    assert camera.baseline.value == 'ground_normal_and_camera_baseline001'
    assert camera.camera_solver_config.vertical_config == normal.camera_solver_config
    assert camera.camera_solver_config.minimum_orthogonal_neighbor_count == 2
    assert camera.camera_solver_config.maximum_focal_refit_iterations == 20
    with pytest.raises(ValueError, match='legal values'):
        ground_normal_config('unknown')
    with pytest.raises(ValueError, match='legal values'):
        ground_offset_config('unknown')
    with pytest.raises(ValueError, match='legal values'):
        ground_normal_and_camera_config('unknown')


def test_ground_normal_baseline_owns_exact_camera_up_result() -> None:
    source, expected = make_direction_source()
    result = solve_ground_normal(source, make_intrinsics())
    np.testing.assert_allclose(result.ground_normal_camera, expected, atol=1e-12)
    np.testing.assert_array_equal(
        result.ground_normal_camera,
        result.direction_result.direction_camera_up,
    )
    assert not result.ground_normal_camera.flags.writeable
    with pytest.raises(TypeError, match='constructed by'):
        Ground_Normal_Config()
    with pytest.raises(TypeError, match='constructed by'):
        Ground_Normal_Result()


def test_ground_offset_selection_is_strict_bound_and_immutable() -> None:
    observations = Ground_Offset_Observations(
        np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0]]),
        np.zeros((5, 2)),
        np.array([4.31, 5.0, 6.0, 4.3, 7.0]),
        np.array([0.19, 0.10, 0.00, 0.10, 0.20]),
    )
    config = ground_offset_config()
    selection = select_ground_offset_observations(observations, config)
    np.testing.assert_array_equal(selection.retained_mask, [True, True, True, False, False])
    assert selection.observations is observations
    assert selection.config is config
    assert not selection.retained_mask.flags.writeable
    assert not observations.top_xy_px.flags.writeable
    with pytest.raises(TypeError, match='constructed by'):
        Ground_Offset_Config()
    with pytest.raises(TypeError, match='constructed by'):
        Ground_Offset_Selection()
    with pytest.raises(TypeError, match='constructed by'):
        Ground_Offset_Result()


def test_ground_offset_preserves_float64_normal_and_solves_positive_D() -> None:
    normal = unit(np.array([0.001234567890123, -0.992345678901234, 0.123456789012345]))
    observations = make_offset_observations(normal)
    result = solve_ground_offset(observations, normal, make_intrinsics())
    np.testing.assert_array_equal(result.ground_normal_camera, normal)
    np.testing.assert_array_equal(result.plane_camera_abcd[:3], normal)
    assert result.plane_camera_abcd[3] == pytest.approx(4.0, abs=0.051)
    assert result.plane_camera_abcd[3] > 0.0
    assert np.isfinite(result.objective)
    assert not result.plane_camera_abcd.flags.writeable


def test_low_level_preserve_flag_is_opt_in_and_backward_compatible() -> None:
    normal = unit(np.array([0.001234567890123, -0.992345678901234, 0.123456789012345]))
    observations = make_offset_observations(normal)
    ones = np.ones((observations.top_xy_px.shape[0], 1), dtype=np.float64)
    xt = np.concatenate((observations.top_xy_px, ones), axis=1).T
    xb = np.concatenate((observations.bottom_xy_px, ones), axis=1).T
    legacy = solve_D_search(
        xb,
        xt,
        normal,
        make_intrinsics().K,
        H_prior=1.27,
    )[0]
    preserved = solve_D_search(
        xb,
        xt,
        normal,
        make_intrinsics().K,
        H_prior=1.27,
        preserve_ground_normal_orientation=True,
    )[0]
    assert legacy.dtype == np.float32
    np.testing.assert_allclose(legacy[:3], -normal, atol=1e-7)
    assert preserved.dtype == np.float64
    np.testing.assert_array_equal(preserved[:3], normal)
    with pytest.raises(ValueError, match='Python bool'):
        solve_D_search(
            xb,
            xt,
            normal,
            make_intrinsics().K,
            preserve_ground_normal_orientation=cast(bool, np.bool_(True)),
        )


def test_ground_offset_rejects_bad_support_and_nonpositive_winner() -> None:
    with pytest.raises(ValueError, match='positive pixel length'):
        select_ground_offset_observations(
            Ground_Offset_Observations(
                np.ones((3, 2)),
                np.ones((3, 2)),
                np.full((3,), 5.0),
                np.full((3,), 0.1),
            ),
            ground_offset_config(),
        )
    normal = unit(np.array([0.1, -0.99, 0.05]))
    observations = make_offset_observations(normal, distance=-1.0)
    with pytest.raises(ValueError, match='positive solved D'):
        solve_ground_offset(observations, normal, make_intrinsics())


def test_ground_normal_and_camera_then_explicit_offset() -> None:
    source, vertical = make_direction_source(camera_ready=True)
    result = solve_ground_normal_and_camera(source)
    assert result.camera_result.focal_px == pytest.approx(600.0, abs=1e-9)
    np.testing.assert_allclose(result.ground_normal_camera, vertical, atol=1e-12)
    assert not hasattr(result, 'plane_camera_abcd')
    offset_result = solve_ground_offset(
        make_offset_observations(vertical),
        result.ground_normal_camera,
        result.camera_intrinsics,
    )
    assert offset_result.plane_camera_abcd[3] == pytest.approx(4.0, abs=0.051)


def test_ground_normal_and_camera_closed_constructors_and_no_aliases() -> None:
    with pytest.raises(TypeError, match='constructed by'):
        Ground_Normal_And_Camera_Config()
    with pytest.raises(TypeError, match='constructed by'):
        Ground_Normal_And_Camera_Result()
    for old_name in (
        'Ground_Camera_Baseline',
        'Ground_Camera_Config',
        'Ground_Camera_Observations',
        'Ground_Camera_Result',
        'ground_camera_config',
        'solve_ground_camera',
    ):
        assert not hasattr(ground_solver_module, old_name)


def smoke_test_ours_ground_baselines() -> None:
    test_registered_configs_are_exact_and_unknown_ids_fail()
    test_ground_normal_baseline_owns_exact_camera_up_result()
    test_ground_offset_selection_is_strict_bound_and_immutable()
    test_ground_offset_preserves_float64_normal_and_solves_positive_D()
    test_low_level_preserve_flag_is_opt_in_and_backward_compatible()
    test_ground_offset_rejects_bad_support_and_nonpositive_winner()
    test_ground_normal_and_camera_then_explicit_offset()
    test_ground_normal_and_camera_closed_constructors_and_no_aliases()


if __name__ == '__main__':
    smoke_test_ours_ground_baselines()
    print('smoke_test_ours_ground_baselines: OK')
