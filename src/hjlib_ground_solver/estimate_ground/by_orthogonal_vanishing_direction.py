'''Locally-horizontal Ground Normal from discrete orthogonal VP consensus.'''

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from hjlib_camera import Camera_Intrinsics
from hjlib_camera_solver import (
    Orthogonal_Consensus_Config,
    Orthogonal_Consensus_Result,
    Role_Aware_Vanishing_Direction_Source,
    Vanishing_Direction_Source,
    select_vertical_direction_by_orthogonal_consensus,
    select_vertical_direction_by_role_aware_orthogonal_consensus,
)
from hjlib_ground_solver.estimate_ground.ground_normal_contract import (
    checked_ground_normal,
)


@dataclass(frozen=True, slots=True)
class Orthogonal_Consensus_Ground_Normal_Result:
    ground_normal_camera:NDArray[np.float64]
    direction_consensus_result:Orthogonal_Consensus_Result

    def __post_init__(self) -> None:
        if type(self.direction_consensus_result) is not Orthogonal_Consensus_Result:
            raise ValueError(
                'direction_consensus_result must be an Orthogonal_Consensus_Result'
            )
        object.__setattr__(
            self,
            'ground_normal_camera',
            checked_ground_normal(
                self.ground_normal_camera,
                self.direction_consensus_result.winner.direction_camera_up,
            ),
        )


def solve_ground_normal_by_orthogonal_consensus(
        sources:Sequence[Vanishing_Direction_Source],
        intrinsics:Camera_Intrinsics,
        config:Orthogonal_Consensus_Config,
    ) -> Orthogonal_Consensus_Ground_Normal_Result:
    direction_result = select_vertical_direction_by_orthogonal_consensus(
        sources,
        intrinsics,
        config,
    )
    return Orthogonal_Consensus_Ground_Normal_Result(
        direction_result.winner.direction_camera_up,
        direction_result,
    )


def solve_ground_normal_by_role_aware_orthogonal_consensus(
        sources:Sequence[Role_Aware_Vanishing_Direction_Source],
        intrinsics:Camera_Intrinsics,
        config:Orthogonal_Consensus_Config,
    ) -> Orthogonal_Consensus_Ground_Normal_Result:
    direction_result = select_vertical_direction_by_role_aware_orthogonal_consensus(
        sources,
        intrinsics,
        config,
    )
    return Orthogonal_Consensus_Ground_Normal_Result(
        direction_result.winner.direction_camera_up,
        direction_result,
    )


__all__:list[str] = [
    'Orthogonal_Consensus_Ground_Normal_Result',
    'solve_ground_normal_by_orthogonal_consensus',
    'solve_ground_normal_by_role_aware_orthogonal_consensus',
]
