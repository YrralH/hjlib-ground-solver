'''Single-source and robust VP Ground Normal interpretation.

Sloped ground is not supported: these wrappers interpret a selected calibrated
camera vertical as a Ground Normal only under the explicit assumption that the
local ground normal is parallel to gravity/scene vertical.
'''

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from hjlib_camera import Camera_Intrinsics
from hjlib_camera_solver import (
    Robust_Vertical_Direction_Config,
    Robust_Vertical_Direction_Result,
    Vertical_Vanishing_Direction_Result,
    Vanishing_Direction_Source,
    select_vertical_direction_by_robust_fusion,
    select_vertical_vanishing_direction,
)
from hjlib_ground_solver.estimate_ground.ground_normal_contract import (
    checked_ground_normal,
)


@dataclass(frozen=True, slots=True)
class Vertical_VP_Selection_Ground_Normal_Result:
    ground_normal_camera:NDArray[np.float64]
    direction_result:Vertical_Vanishing_Direction_Result

    def __post_init__(self) -> None:
        if type(self.direction_result) is not Vertical_Vanishing_Direction_Result:
            raise ValueError(
                'direction_result must be a Vertical_Vanishing_Direction_Result'
            )
        object.__setattr__(
            self,
            'ground_normal_camera',
            checked_ground_normal(
                self.ground_normal_camera,
                self.direction_result.direction_camera_up,
            ),
        )


def solve_ground_normal_by_vertical_vp_selection(
        source:Vanishing_Direction_Source,
        intrinsics:Camera_Intrinsics,
        min_support_count:int = 5,
    ) -> Vertical_VP_Selection_Ground_Normal_Result:
    direction_result = select_vertical_vanishing_direction(
        source.association,
        source.line_segments,
        intrinsics,
        min_support_count,
    )
    return Vertical_VP_Selection_Ground_Normal_Result(
        direction_result.direction_camera_up,
        direction_result,
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
        object.__setattr__(
            self,
            'ground_normal_camera',
            checked_ground_normal(self.ground_normal_camera, winner_direction),
        )


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
    'Vertical_VP_Selection_Ground_Normal_Result',
    'solve_ground_normal_by_vertical_vp_selection',
    'Vanishing_Direction_Ground_Normal_Result',
    'solve_ground_normal_from_vanishing_directions',
]
