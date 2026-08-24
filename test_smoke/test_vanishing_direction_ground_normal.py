'''Smokes for the locally-horizontal vanishing-direction GN wrapper.'''

import math
from unittest.mock import patch

import numpy as np
import pytest

from hjlib_camera import Camera_Intrinsics
from hjlib_camera_solver import (
    Image_Line_Segments,
    Robust_Vertical_Direction_Config,
    Vanishing_Direction_Source,
    Vanishing_Point_Association,
    select_vertical_direction_by_robust_fusion,
)
from hjlib_ground_solver import (
    Vanishing_Direction_Ground_Normal_Result,
    solve_ground_normal_from_vanishing_directions,
)
import hjlib_ground_solver.estimate_ground.by_vanishing_direction as wrapper_module


def unit(value:np.ndarray) -> np.ndarray:
    return value / np.linalg.norm(value)


def make_intrinsics() -> Camera_Intrinsics:
    return Camera_Intrinsics(np.array([
        [100.0, 0.0, 80.0],
        [0.0, 100.0, 80.0],
        [0.0, 0.0, 1.0],
    ]), (160, 160))


def make_config() -> Robust_Vertical_Direction_Config:
    return Robust_Vertical_Direction_Config(
        (16, 16), (4, 4), 256, 100_000, 2,
        2.0, 1.0, 5.0, 5.0, 2.0, 20.0, 2,
        0.2, 0.2, 0.2, 0.2, 0.2,
        0.0, 20.0, 0.5, 0.5, 2.0, 2.0, 2.0, 1.0,
        1e-10, 100, 1e-6, 1e-12, 1e-12,
    )


def make_directions() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertical = unit(np.array([0.0, -0.8, 0.6]))
    horizontal_1 = np.array([1.0, 0.0, 0.0])
    tangent = unit(np.cross(vertical, horizontal_1))
    horizontal_2 = unit(
        math.cos(math.radians(30.0)) * horizontal_1
        + math.sin(math.radians(30.0)) * tangent
    )
    return vertical, horizontal_1, horizontal_2


def make_source(
        directions:np.ndarray,
        record_id:str = 'ground-normal',
    ) -> Vanishing_Direction_Source:
    segment_sets = tuple(
        np.array([
            [[10.0, 15.0 + 35.0 * index], [55.0, 15.0 + 35.0 * index]],
            [[75.0, 19.0 + 35.0 * index], [135.0, 19.0 + 35.0 * index]],
        ])
        for index in range(directions.shape[0])
    )
    endpoints = np.concatenate(segment_sets, axis=0)
    labels = np.concatenate(tuple(
        np.full((2,), index, dtype=np.int64)
        for index in range(directions.shape[0])
    ))
    lines = Image_Line_Segments(record_id, (160, 160), endpoints)
    association = Vanishing_Point_Association(
        record_id,
        lines.line_segments_sha256,
        labels,
        (make_intrinsics().K @ directions.T).T,
    )
    return Vanishing_Direction_Source('synthetic', association, lines)


def test_wrapper_delegates_once_and_returns_independent_winner() -> None:
    source = make_source(np.stack(make_directions()))
    original = wrapper_module.select_vertical_direction_by_robust_fusion
    with patch.object(
            wrapper_module,
            'select_vertical_direction_by_robust_fusion',
            wraps=original,
        ) as mocked:
        result = solve_ground_normal_from_vanishing_directions(
            (source,), make_intrinsics(), make_config(),
        )
    assert mocked.call_count == 1
    assert result.direction_fusion_result.status == 'success'
    winner_index = result.direction_fusion_result.winner_candidate_index
    assert winner_index is not None
    winner_direction = result.direction_fusion_result.candidates[
        winner_index
    ].refined_direction_camera_up
    assert winner_direction is not None
    assert result.ground_normal_camera is not None
    np.testing.assert_array_equal(result.ground_normal_camera, winner_direction)
    assert result.ground_normal_camera is not winner_direction
    assert not result.ground_normal_camera.flags.writeable
    with pytest.raises(ValueError):
        result.ground_normal_camera.setflags(write=True)


def test_constructor_defensively_copies_and_rejects_forgery() -> None:
    source = make_source(np.stack(make_directions()))
    nested = select_vertical_direction_by_robust_fusion(
        (source,), make_intrinsics(), make_config(),
    )
    assert nested.winner_candidate_index is not None
    winner = nested.candidates[nested.winner_candidate_index]
    assert winner.refined_direction_camera_up is not None
    mutable = np.array(winner.refined_direction_camera_up, copy=True)
    result = Vanishing_Direction_Ground_Normal_Result(mutable, nested)
    mutable[0] += 1.0
    np.testing.assert_array_equal(
        result.ground_normal_camera,
        winner.refined_direction_camera_up,
    )
    with pytest.raises(ValueError, match='unit norm'):
        Vanishing_Direction_Ground_Normal_Result(
            winner.refined_direction_camera_up * 2.0,
            nested,
        )
    with pytest.raises(ValueError, match='exactly equal'):
        Vanishing_Direction_Ground_Normal_Result(
            -winner.refined_direction_camera_up,
            nested,
        )
    with pytest.raises(ValueError, match='requires a Ground Normal'):
        Vanishing_Direction_Ground_Normal_Result(None, nested)


def test_all_rejected_preserves_ledger_and_maps_to_none() -> None:
    vertical, horizontal, _ = make_directions()
    source = make_source(np.stack((vertical, horizontal)), 'all-rejected')
    result = solve_ground_normal_from_vanishing_directions(
        (source,), make_intrinsics(), make_config(),
    )
    assert result.ground_normal_camera is None
    assert result.direction_fusion_result.status == 'no_accepted_candidate'
    assert result.direction_fusion_result.candidates
    with pytest.raises(ValueError, match='requires ground_normal_camera=None'):
        Vanishing_Direction_Ground_Normal_Result(
            np.array([0.0, -1.0, 0.0]),
            result.direction_fusion_result,
        )


def test_camera_solver_input_failure_propagates() -> None:
    source = make_source(np.stack(make_directions()))
    wrong_intrinsics = Camera_Intrinsics(np.eye(3), (80, 80))
    with pytest.raises(ValueError, match='image sizes differ'):
        solve_ground_normal_from_vanishing_directions(
            (source,), wrong_intrinsics, make_config(),
        )


def test_sloped_ground_is_explicitly_out_of_scope() -> None:
    assert wrapper_module.__doc__ is not None
    assert 'Sloped ground is not supported' in wrapper_module.__doc__


def smoke_test_vanishing_direction_ground_normal() -> None:
    test_wrapper_delegates_once_and_returns_independent_winner()
    test_constructor_defensively_copies_and_rejects_forgery()
    test_all_rejected_preserves_ledger_and_maps_to_none()
    test_camera_solver_input_failure_propagates()
    test_sloped_ground_is_explicitly_out_of_scope()


if __name__ == '__main__':
    smoke_test_vanishing_direction_ground_normal()
    print('smoke_test_vanishing_direction_ground_normal: OK')
