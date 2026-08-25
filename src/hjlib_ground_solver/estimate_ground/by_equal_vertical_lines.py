'''Locally-horizontal Ground Normal from equal-line or source-weighted evidence.'''

from dataclasses import dataclass
from typing import Sequence

from numpy.typing import NDArray
import numpy as np

from hjlib_camera import Camera_Intrinsics
from hjlib_camera_solver import (
    Equal_Weight_Axial_Direction_Result,
    Equal_Weight_Image_Line_Source,
    Source_Weighted_Axial_Direction_Result,
    Source_Weighted_Image_Line_Source,
    fit_axial_direction_by_equal_weight_image_lines,
    fit_axial_direction_by_source_weighted_image_lines,
)
from hjlib_ground_solver.estimate_ground.ground_normal_contract import (
    checked_ground_normal,
)


@dataclass(frozen=True, slots=True)
class Equal_Weight_Vertical_Line_Ground_Normal_Result:
    ground_normal_camera:NDArray[np.float64]
    direction_result:Equal_Weight_Axial_Direction_Result

    def __post_init__(self) -> None:
        if type(self.direction_result) is not Equal_Weight_Axial_Direction_Result:
            raise ValueError(
                'direction_result must be an Equal_Weight_Axial_Direction_Result',
            )
        object.__setattr__(
            self,
            'ground_normal_camera',
            checked_ground_normal(
                self.ground_normal_camera,
                self.direction_result.direction_camera_up,
            ),
        )


def solve_ground_normal_by_equal_weight_vertical_lines(
        sources:Sequence[Equal_Weight_Image_Line_Source],
        intrinsics:Camera_Intrinsics,
    ) -> Equal_Weight_Vertical_Line_Ground_Normal_Result:
    direction_result = fit_axial_direction_by_equal_weight_image_lines(
        sources,
        intrinsics,
    )
    return Equal_Weight_Vertical_Line_Ground_Normal_Result(
        direction_result.direction_camera_up,
        direction_result,
    )


@dataclass(frozen=True, slots=True)
class Source_Weighted_Vertical_Line_Ground_Normal_Result:
    ground_normal_camera:NDArray[np.float64]
    direction_result:Source_Weighted_Axial_Direction_Result

    def __post_init__(self) -> None:
        if type(self.direction_result) is not Source_Weighted_Axial_Direction_Result:
            raise ValueError(
                'direction_result must be a Source_Weighted_Axial_Direction_Result',
            )
        object.__setattr__(
            self,
            'ground_normal_camera',
            checked_ground_normal(
                self.ground_normal_camera,
                self.direction_result.direction_camera_up,
            ),
        )


def solve_ground_normal_by_source_weighted_vertical_lines(
        sources:Sequence[Source_Weighted_Image_Line_Source],
        intrinsics:Camera_Intrinsics,
    ) -> Source_Weighted_Vertical_Line_Ground_Normal_Result:
    direction_result = fit_axial_direction_by_source_weighted_image_lines(
        sources,
        intrinsics,
    )
    return Source_Weighted_Vertical_Line_Ground_Normal_Result(
        direction_result.direction_camera_up,
        direction_result,
    )


__all__:list[str] = [
    'Equal_Weight_Vertical_Line_Ground_Normal_Result',
    'solve_ground_normal_by_equal_weight_vertical_lines',
    'Source_Weighted_Vertical_Line_Ground_Normal_Result',
    'solve_ground_normal_by_source_weighted_vertical_lines',
]
