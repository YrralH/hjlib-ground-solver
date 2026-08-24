'''Ground Normal interpretation for locally horizontal scenes.

Sloped ground is not supported: this wrapper interprets the robust calibrated
camera vertical as a Ground Normal only under the explicit assumption that the
local ground normal is parallel to gravity/scene vertical.
'''

from dataclasses import dataclass
import math
from typing import Sequence, cast

import numpy as np
from numpy.typing import NDArray

from hjlib_camera import Camera_Intrinsics
from hjlib_camera_solver import (
    Robust_Vertical_Direction_Config,
    Robust_Vertical_Direction_Result,
    Vanishing_Direction_Source,
    select_vertical_direction_by_robust_fusion,
)


@dataclass(frozen=True, slots=True)
class Vanishing_Direction_Ground_Normal_Result:
    ground_normal_camera:NDArray[np.float64] | None
    direction_fusion_result:Robust_Vertical_Direction_Result

    def __post_init__(self) -> None:
        if type(self.direction_fusion_result) is not Robust_Vertical_Direction_Result:
            raise ValueError(
                'direction_fusion_result must be Robust_Vertical_Direction_Result'
            )
        if self.direction_fusion_result.status == 'no_accepted_candidate':
            if self.ground_normal_camera is not None:
                raise ValueError('failed direction result requires ground_normal_camera=None')
            return
        winner_index = self.direction_fusion_result.winner_candidate_index
        if winner_index is None:
            raise ValueError('successful direction result lacks winner index')
        winner_direction = self.direction_fusion_result.candidates[
            winner_index
        ].refined_direction_camera_up
        if winner_direction is None:
            raise ValueError('successful direction winner lacks refined direction')
        if self.ground_normal_camera is None:
            raise ValueError('successful direction result requires a Ground Normal')
        value = np.asarray(self.ground_normal_camera)
        if not np.issubdtype(value.dtype, np.number) or np.issubdtype(
                value.dtype,
                np.bool_,
            ) or np.issubdtype(value.dtype, np.complexfloating):
            raise ValueError('ground_normal_camera must have real numeric dtype')
        owned = np.asarray(value, dtype=np.float64)
        if owned.shape != (3,) or not np.all(np.isfinite(owned)):
            raise ValueError('ground_normal_camera must be a finite (3,) vector')
        norm = float(np.linalg.norm(owned))
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError('ground_normal_camera must have unit norm')
        immutable = cast(
            NDArray[np.float64],
            np.frombuffer(owned.tobytes(), dtype=np.float64).reshape((3,)),
        )
        if not np.array_equal(immutable, winner_direction):
            raise ValueError('ground_normal_camera must exactly equal winning direction')
        object.__setattr__(self, 'ground_normal_camera', immutable)


def solve_ground_normal_from_vanishing_directions(
        sources:Sequence[Vanishing_Direction_Source],
        intrinsics:Camera_Intrinsics,
        config:Robust_Vertical_Direction_Config,
    ) -> Vanishing_Direction_Ground_Normal_Result:
    direction_result = select_vertical_direction_by_robust_fusion(
        sources,
        intrinsics,
        config,
    )
    if direction_result.status == 'no_accepted_candidate':
        return Vanishing_Direction_Ground_Normal_Result(None, direction_result)
    winner_index = direction_result.winner_candidate_index
    if winner_index is None:
        raise RuntimeError('successful direction result lacks winner index')
    direction = direction_result.candidates[
        winner_index
    ].refined_direction_camera_up
    if direction is None:
        raise RuntimeError('successful direction winner lacks refined direction')
    return Vanishing_Direction_Ground_Normal_Result(direction, direction_result)


__all__:list[str] = [
    'Vanishing_Direction_Ground_Normal_Result',
    'solve_ground_normal_from_vanishing_directions',
]
