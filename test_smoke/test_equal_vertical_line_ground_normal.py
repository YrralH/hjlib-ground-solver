'''Smokes for equal-weight vertical-line Ground Normal interpretation.'''

from unittest.mock import patch

import numpy as np
import pytest

import hjlib_ground_solver
from hjlib_camera import Camera_Intrinsics
from hjlib_camera_solver import Equal_Weight_Image_Line_Source, Image_Line_Segments
from hjlib_ground_solver import (
    Equal_Weight_Vertical_Line_Ground_Normal_Result,
    solve_ground_normal_by_equal_weight_vertical_lines,
)
from hjlib_ground_solver.estimate_ground import (
    Equal_Weight_Vertical_Line_Ground_Normal_Result as Estimate_Result,
    solve_ground_normal_by_equal_weight_vertical_lines as estimate_solve,
)
from hjlib_ground_solver.estimate_ground.by_equal_vertical_lines import (
    Equal_Weight_Vertical_Line_Ground_Normal_Result as Module_Result,
    solve_ground_normal_by_equal_weight_vertical_lines as module_solve,
)


def make_inputs() -> tuple[Equal_Weight_Image_Line_Source, Camera_Intrinsics]:
    intrinsics = Camera_Intrinsics(
        np.array([
            [500.0, 0.0, 320.0],
            [0.0, 500.0, 240.0],
            [0.0, 0.0, 1.0],
        ]),
        (640, 480),
    )
    vp_x = 320.0
    bottom_xs = (100.0, 260.0, 500.0)
    endpoints = np.array([
        [
            [vp_x + (bottom_x - vp_x) * 0.4, 100.0],
            [bottom_x, 400.0],
        ]
        for bottom_x in bottom_xs
    ])
    lines = Image_Line_Segments('scene:vertical', (640, 480), endpoints)
    return Equal_Weight_Image_Line_Source('vertical', 'fixed.camera', lines), intrinsics


def test_wrapper_delegates_once_and_preserves_direction() -> None:
    source, intrinsics = make_inputs()
    native = solve_ground_normal_by_equal_weight_vertical_lines((source,), intrinsics)
    with patch(
            'hjlib_ground_solver.estimate_ground.by_equal_vertical_lines.'
            'fit_axial_direction_by_equal_weight_image_lines',
            return_value=native.direction_result,
        ) as fit:
        result = solve_ground_normal_by_equal_weight_vertical_lines((source,), intrinsics)
    fit.assert_called_once_with((source,), intrinsics)
    np.testing.assert_array_equal(
        result.ground_normal_camera,
        result.direction_result.direction_camera_up,
    )
    assert not result.ground_normal_camera.flags.writeable
    with pytest.raises(ValueError):
        result.ground_normal_camera.setflags(write=True)


def test_owner_failure_propagates_without_fallback() -> None:
    source, intrinsics = make_inputs()
    sentinel = ValueError('sentinel camera TLS failure')
    with patch(
            'hjlib_ground_solver.estimate_ground.by_equal_vertical_lines.'
            'fit_axial_direction_by_equal_weight_image_lines',
            side_effect=sentinel,
        ) as fit:
        with pytest.raises(ValueError) as caught:
            solve_ground_normal_by_equal_weight_vertical_lines((source,), intrinsics)
    assert caught.value is sentinel
    fit.assert_called_once_with((source,), intrinsics)


def test_result_rejects_wrong_or_disagreeing_direction() -> None:
    source, intrinsics = make_inputs()
    native = solve_ground_normal_by_equal_weight_vertical_lines((source,), intrinsics)
    with pytest.raises(ValueError, match='direction_result'):
        Equal_Weight_Vertical_Line_Ground_Normal_Result(
            native.ground_normal_camera,
            object(),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match='exactly equal'):
        Equal_Weight_Vertical_Line_Ground_Normal_Result(
            np.array([1.0, 0.0, 0.0]),
            native.direction_result,
        )


def test_three_level_public_exports() -> None:
    assert Module_Result is Estimate_Result
    assert Estimate_Result is Equal_Weight_Vertical_Line_Ground_Normal_Result
    assert hjlib_ground_solver.Equal_Weight_Vertical_Line_Ground_Normal_Result is (
        Equal_Weight_Vertical_Line_Ground_Normal_Result
    )
    assert hjlib_ground_solver.solve_ground_normal_by_equal_weight_vertical_lines is (
        solve_ground_normal_by_equal_weight_vertical_lines
    )
    assert module_solve is estimate_solve
    assert estimate_solve is solve_ground_normal_by_equal_weight_vertical_lines


def smoke_test_equal_vertical_line_ground_normal() -> None:
    test_wrapper_delegates_once_and_preserves_direction()
    test_owner_failure_propagates_without_fallback()
    test_result_rejects_wrong_or_disagreeing_direction()
    test_three_level_public_exports()


if __name__ == '__main__':
    smoke_test_equal_vertical_line_ground_normal()
    print('smoke_test_equal_vertical_line_ground_normal: OK')
