'''Smoke tests for the explicitly nonofficial HJ-derived plantar zmin.'''

from dataclasses import FrozenInstanceError
from typing import Any, Callable, cast

import numpy as np

import hjlib_ground_solver
import hjlib_ground_solver.estimate_ground as estimate_ground
import hjlib_ground_solver.estimate_ground.by_hj_derived_plantar_zmin as implementation
from hjlib_ground_solver.estimate_ground.by_hj_derived_plantar_zmin import (
    HJ_Derived_Plantar_ZMin_Ground_Provenance,
    HJ_Derived_Plantar_ZMin_Ground_Result,
    HJ_Derived_Plantar_ZMin_Ground_Side,
    estimate_hj_derived_plantar_zmin_ground,
)


def expect_value_error(operation: Callable[[], object]) -> None:
    try:
        operation()
    except ValueError:
        return
    raise AssertionError('operation did not raise ValueError')


def test_selection_ties_and_provenance() -> None:
    left = np.asarray((0.2, -0.3, -0.3, 0.1), dtype=np.float64)
    right = np.asarray((-0.3, 0.0, -0.3, 0.2), dtype=np.float64)
    result = estimate_hj_derived_plantar_zmin_ground(left, right)
    assert result.provenance == 'hj_derived_nonofficial'
    assert result.input_dtype == 'float64'
    assert result.frame_count == 4
    assert result.left_minimum_height_in_meter == -0.3
    assert result.right_minimum_height_in_meter == -0.3
    assert result.ground_height_in_meter == -0.3
    assert result.selected_side == 'left'
    assert result.selected_frame_index_within_input_track == 1
    assert result.tied_global_minimum_sample_count == 4

    right_lower = estimate_hj_derived_plantar_zmin_ground(
        left,
        np.asarray((0.0, -0.4, -0.4, 0.1), dtype=np.float64),
    )
    assert right_lower.selected_side == 'right'
    assert right_lower.selected_frame_index_within_input_track == 1
    assert right_lower.tied_global_minimum_sample_count == 2

    signed_zero = estimate_hj_derived_plantar_zmin_ground(
        np.asarray((-0.0, 1.0), dtype=np.float64),
        np.asarray((+0.0, 2.0), dtype=np.float64),
    )
    assert signed_zero.selected_side == 'left'
    assert signed_zero.selected_frame_index_within_input_track == 0
    assert signed_zero.tied_global_minimum_sample_count == 2


def test_dtype_shift_immutability_and_rounding_collision() -> None:
    for dtype in (np.float32, np.float64):
        left = np.asarray((0.0, -0.25, 0.5), dtype=dtype)
        right = np.asarray((0.25, 0.0, 0.75), dtype=dtype)
        left_before = left.copy()
        right_before = right.copy()
        result = estimate_hj_derived_plantar_zmin_ground(left, right)
        shift = dtype(0.125)
        shifted = estimate_hj_derived_plantar_zmin_ground(
            left + shift,
            right + shift,
        )
        assert shifted.ground_height_in_meter \
            == result.ground_height_in_meter + float(shift)
        assert shifted.selected_side == result.selected_side
        assert shifted.selected_frame_index_within_input_track \
            == result.selected_frame_index_within_input_track
        assert shifted.tied_global_minimum_sample_count \
            == result.tied_global_minimum_sample_count
        assert np.array_equal(left, left_before)
        assert np.array_equal(right, right_before)

    lower = np.float32(1.0)
    upper = np.nextafter(lower, np.float32(2.0))
    original = estimate_hj_derived_plantar_zmin_ground(
        np.asarray((lower, upper), dtype=np.float32),
        np.asarray((upper, upper), dtype=np.float32),
    )
    large = np.float32(2 ** 25)
    collided = estimate_hj_derived_plantar_zmin_ground(
        np.asarray((lower, upper), dtype=np.float32) + large,
        np.asarray((upper, upper), dtype=np.float32) + large,
    )
    assert original.tied_global_minimum_sample_count == 1
    assert collided.tied_global_minimum_sample_count == 4


def test_frozen_exports_and_validation() -> None:
    result = estimate_hj_derived_plantar_zmin_ground(
        np.asarray((0.0,), dtype=np.float64),
        np.asarray((1.0,), dtype=np.float64),
    )
    try:
        setattr(result, 'ground_height_in_meter', 1.0)
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError('result must be frozen')
    public_symbols = (
        (
            'HJ_Derived_Plantar_ZMin_Ground_Provenance',
            HJ_Derived_Plantar_ZMin_Ground_Provenance,
        ),
        ('HJ_Derived_Plantar_ZMin_Ground_Result', HJ_Derived_Plantar_ZMin_Ground_Result),
        (
            'HJ_Derived_Plantar_ZMin_Ground_Side',
            HJ_Derived_Plantar_ZMin_Ground_Side,
        ),
        ('estimate_hj_derived_plantar_zmin_ground', estimate_hj_derived_plantar_zmin_ground),
    )
    for name, symbol in public_symbols:
        assert getattr(implementation, name) is symbol
        assert getattr(estimate_ground, name) is symbol
        assert getattr(hjlib_ground_solver, name) is symbol

    values = np.asarray((0.0, 1.0), dtype=np.float64)
    expect_value_error(lambda: estimate_hj_derived_plantar_zmin_ground(
        cast(Any, (0.0, 1.0)), values,
    ))
    expect_value_error(lambda: estimate_hj_derived_plantar_zmin_ground(
        values[:, None], values,
    ))
    expect_value_error(lambda: estimate_hj_derived_plantar_zmin_ground(
        values[:0], values[:0],
    ))
    expect_value_error(lambda: estimate_hj_derived_plantar_zmin_ground(
        values, values[:1],
    ))
    expect_value_error(lambda: estimate_hj_derived_plantar_zmin_ground(
        values.astype(np.float32), values,
    ))
    expect_value_error(lambda: estimate_hj_derived_plantar_zmin_ground(
        values.astype(np.float16), values.astype(np.float16),
    ))
    expect_value_error(lambda: estimate_hj_derived_plantar_zmin_ground(
        cast(Any, values.astype(np.int64)),
        cast(Any, values.astype(np.int64)),
    ))
    nonfinite = values.copy()
    nonfinite[0] = np.nan
    expect_value_error(lambda: estimate_hj_derived_plantar_zmin_ground(
        nonfinite, values,
    ))


def smoke_test_hj_derived_plantar_zmin() -> None:
    test_selection_ties_and_provenance()
    test_dtype_shift_immutability_and_rounding_collision()
    test_frozen_exports_and_validation()


if __name__ == '__main__':
    smoke_test_hj_derived_plantar_zmin()
    print('test_hj_derived_plantar_zmin: PASS')
