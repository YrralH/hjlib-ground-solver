'''Explicitly nonofficial HJ-derived ground from plantar height tracks.'''

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray


type ARRAY_F = NDArray[np.floating[Any]]
type HJ_Derived_Plantar_ZMin_Ground_Provenance = Literal[
    'hj_derived_nonofficial',
]
type HJ_Derived_Plantar_ZMin_Ground_Side = Literal['left', 'right']


@dataclass(frozen=True, slots=True)
class HJ_Derived_Plantar_ZMin_Ground_Result:
    '''Immutable derived height and deterministic minimum identity evidence.'''

    input_dtype: str
    frame_count: int
    left_minimum_height_in_meter: float
    right_minimum_height_in_meter: float
    ground_height_in_meter: float
    selected_side: HJ_Derived_Plantar_ZMin_Ground_Side
    selected_frame_index_within_input_track: int
    tied_global_minimum_sample_count: int
    provenance: HJ_Derived_Plantar_ZMin_Ground_Provenance = field(
        default='hj_derived_nonofficial',
        init=False,
    )


def validate_hj_derived_plantar_zmin_ground_input(
        left_per_frame_minimum_height_in_meter: object,
        right_per_frame_minimum_height_in_meter: object,
    ) -> tuple[ARRAY_F, ARRAY_F]:
    '''Validate same-shape finite float32/float64 plantar height tracks.'''
    if (
            type(left_per_frame_minimum_height_in_meter) is not np.ndarray
            or type(right_per_frame_minimum_height_in_meter) is not np.ndarray
        ):
        raise ValueError('left and right plantar height tracks must be numpy arrays')
    left = left_per_frame_minimum_height_in_meter
    right = right_per_frame_minimum_height_in_meter
    if left.ndim != 1 or left.size <= 0 or right.shape != left.shape:
        raise ValueError('left and right plantar height tracks must share nonempty shape (T,)')
    supported_dtypes = (np.dtype(np.float32), np.dtype(np.float64))
    if left.dtype not in supported_dtypes or right.dtype != left.dtype:
        raise ValueError('plantar height tracks must share dtype float32 or float64')
    if not bool(np.isfinite(left).all()) or not bool(np.isfinite(right).all()):
        raise ValueError('plantar height tracks must contain only finite values')
    return left, right


def estimate_hj_derived_plantar_zmin_ground(
        left_per_frame_minimum_height_in_meter: ARRAY_F,
        right_per_frame_minimum_height_in_meter: ARRAY_F,
    ) -> HJ_Derived_Plantar_ZMin_Ground_Result:
    '''Estimate an HJ-derived nonofficial ground height as absolute plantar zmin.'''
    left, right = validate_hj_derived_plantar_zmin_ground_input(
        left_per_frame_minimum_height_in_meter,
        right_per_frame_minimum_height_in_meter,
    )
    left_minimum = float(np.min(left))
    right_minimum = float(np.min(right))
    if left_minimum <= right_minimum:
        ground_height = left_minimum
        selected_side: HJ_Derived_Plantar_ZMin_Ground_Side = 'left'
        selected_frame = int(np.flatnonzero(left == left_minimum)[0])
    else:
        ground_height = right_minimum
        selected_side = 'right'
        selected_frame = int(np.flatnonzero(right == right_minimum)[0])
    tied_count = (
        int(np.count_nonzero(left == ground_height))
        + int(np.count_nonzero(right == ground_height))
    )
    return HJ_Derived_Plantar_ZMin_Ground_Result(
        input_dtype=str(left.dtype),
        frame_count=int(left.size),
        left_minimum_height_in_meter=left_minimum,
        right_minimum_height_in_meter=right_minimum,
        ground_height_in_meter=ground_height,
        selected_side=selected_side,
        selected_frame_index_within_input_track=selected_frame,
        tied_global_minimum_sample_count=tied_count,
    )


__all__ = [
    'HJ_Derived_Plantar_ZMin_Ground_Provenance',
    'HJ_Derived_Plantar_ZMin_Ground_Result',
    'HJ_Derived_Plantar_ZMin_Ground_Side',
    'estimate_hj_derived_plantar_zmin_ground',
]
