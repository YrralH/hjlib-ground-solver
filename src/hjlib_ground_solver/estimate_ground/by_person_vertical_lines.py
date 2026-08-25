'''Build one vertical-direction evidence source from upright person image lines.'''

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

from hjlib_camera import Camera_Intrinsics
from hjlib_camera_solver import (
    Image_Line_Segments,
    Vanishing_Direction_Source,
    Vanishing_Point_Association,
    Vertical_Vanishing_Direction_Result,
    select_vertical_vanishing_direction,
)
from hjlib_ground_solver.estimate_ground.by_kp_rcr.compute_KN_by_vertical_lines import (
    get_KN_with_filter,
)


def owned_real_float64_array(
        name:str,
        value:object,
    ) -> NDArray[np.float64]:
    if not isinstance(value, np.ndarray):
        raise ValueError('%s must be a NumPy array' % name)
    if not np.issubdtype(value.dtype, np.number) or np.issubdtype(
            value.dtype,
            np.bool_,
        ) or np.issubdtype(value.dtype, np.complexfloating):
        raise ValueError('%s must have real non-Boolean numeric dtype' % name)
    owned = np.array(value, dtype=np.float64, copy=True)
    if not bool(np.isfinite(owned).all()):
        raise ValueError('%s must be finite' % name)
    return owned


@dataclass(frozen=True, slots=True, init=False)
class Person_Vertical_Direction_Evidence_Result:
    source:Vanishing_Direction_Source
    direction_result:Vertical_Vanishing_Direction_Result
    retained_observation_count:int

    def __init__(
            self,
            source:Vanishing_Direction_Source,
            intrinsics:Camera_Intrinsics,
        ) -> None:
        if type(source) is not Vanishing_Direction_Source:
            raise ValueError('source must be a Vanishing_Direction_Source')
        if type(intrinsics) is not Camera_Intrinsics:
            raise ValueError('intrinsics must be a Camera_Intrinsics')
        retained_count = int(source.line_segments.endpoints_xy.shape[0])
        if retained_count < 3:
            raise ValueError('person vertical evidence must retain at least three lines')
        if source.association.pixel_vps_h.shape != (1, 3):
            raise ValueError('person vertical evidence must contain exactly one VP')
        if source.association.labels.shape != (retained_count,):
            raise ValueError('person vertical evidence label count disagrees with lines')
        if not bool(np.all(source.association.labels == 0)):
            raise ValueError('person vertical evidence labels must all select cluster zero')
        direction_result = select_vertical_vanishing_direction(
            source.association,
            source.line_segments,
            intrinsics,
            min_support_count=1,
        )
        if direction_result.cluster_index != 0:
            raise ValueError('person vertical evidence must select cluster zero')
        if direction_result.support_count != retained_count:
            raise ValueError('person vertical evidence support count disagrees with lines')
        object.__setattr__(self, 'source', source)
        object.__setattr__(self, 'direction_result', direction_result)
        object.__setattr__(self, 'retained_observation_count', retained_count)


def fit_person_vertical_direction_evidence(
        top_xy_px:NDArray[np.generic],
        bottom_xy_px:NDArray[np.generic],
        observation_weights:NDArray[np.generic],
        intrinsics:Camera_Intrinsics,
        source_id:str,
        image_record_id:str,
        prop_filter:float,
        times_filter:int,
    ) -> Person_Vertical_Direction_Evidence_Result:
    if type(intrinsics) is not Camera_Intrinsics:
        raise ValueError('intrinsics must be a Camera_Intrinsics')
    if type(prop_filter) is not float or not math.isfinite(prop_filter):
        raise ValueError('prop_filter must be a finite Python float')
    if not 0 < prop_filter < 1:
        raise ValueError('prop_filter must be in (0, 1)')
    if type(times_filter) is not int or not 1 <= times_filter <= 9:
        raise ValueError('times_filter must be a Python integer in [1, 9]')

    top = owned_real_float64_array('top_xy_px', top_xy_px)
    bottom = owned_real_float64_array('bottom_xy_px', bottom_xy_px)
    weights = owned_real_float64_array('observation_weights', observation_weights)
    if top.ndim != 2 or top.shape[1:] != (2,) or bottom.shape != top.shape:
        raise ValueError('top/bottom arrays must have equal shape (N, 2)')
    observation_count = int(top.shape[0])
    if observation_count < 3:
        raise ValueError('at least three person observations are required')
    if weights.shape != (observation_count,):
        raise ValueError('observation_weights must have shape (N,)')
    if not bool(np.all(weights > 0)):
        raise ValueError('observation_weights must be strictly positive')
    if bool(np.any(np.linalg.norm(top - bottom, axis=1) == 0)):
        raise ValueError('each person top/bottom segment must have non-zero length')

    top_h = np.concatenate(
        (top, np.ones((observation_count, 1), dtype=np.float64)),
        axis=1,
    ).T
    bottom_h = np.concatenate(
        (bottom, np.ones((observation_count, 1), dtype=np.float64)),
        axis=1,
    ).T
    fitted = get_KN_with_filter(
        bottom_h,
        top_h,
        prop_filter=prop_filter,
        times_filter=times_filter,
        flag_ret_filtered_result=True,
        observation_weights=weights,
    )
    if not isinstance(fitted, tuple):
        raise RuntimeError('filtered KN fit did not return retained lines')
    pixel_vp_h, bottom_filtered_h, top_filtered_h = fitted
    endpoints = np.stack(
        (bottom_filtered_h[:2].T, top_filtered_h[:2].T),
        axis=1,
    )
    lines = Image_Line_Segments(image_record_id, intrinsics.image_size, endpoints)
    association = Vanishing_Point_Association(
        image_record_id,
        lines.line_segments_sha256,
        np.zeros((endpoints.shape[0],), dtype=np.int64),
        np.asarray(pixel_vp_h, dtype=np.float64)[None],
    )
    source = Vanishing_Direction_Source(source_id, association, lines)
    return Person_Vertical_Direction_Evidence_Result(source, intrinsics)


__all__:list[str] = [
    'Person_Vertical_Direction_Evidence_Result',
    'fit_person_vertical_direction_evidence',
]
