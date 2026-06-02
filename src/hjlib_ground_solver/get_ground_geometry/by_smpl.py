from typing import List, Tuple

import numpy as np

from hjlib_smpl.smpl_full_54 import SMPL_Full
from hjlib_smpl.skeleton_helpers import get_rough_pillars_and_from_smpl_verts_batch
from hjlib_geometry import convert_ground_parameters_to_4_verts_mesh_with_on_ground_points

from hjlib_ground_solver.get_ground_geometry.by_pillars import get_ground_by_pillars_on_the_ground
from hjlib_ground_solver.get_ground_geometry.by_points import (
    get_ground_by_points_on_the_ground,
    get_ground_by_points_on_the_ground_lstsq,
)


def get_ground_by_smpls_on_the_ground(
        list_smpl_verts: List[np.ndarray],
        smpl_model: SMPL_Full,
        strategy_get_ground: str = 'normal_direction_maxmin',
        ratio_border: float = 0.25
    ) -> Tuple[np.ndarray, np.ndarray]:
    list_direction, list_position = get_rough_pillars_and_from_smpl_verts_batch(list_smpl_verts, smpl_model)
    array_direction = np.array(list_direction)
    array_position = np.array(list_position)

    N = array_direction.shape[0]
    assert array_position.shape == (N, 3), array_position.shape
    assert array_direction.shape == (N, 3), array_direction.shape

    if strategy_get_ground == 'smpl_direction_maxmin':
        verts_ground, faces_ground = get_ground_by_pillars_on_the_ground(
            array_position,
            array_direction,
            ratio_border=ratio_border,
            ratio_height=0.5
        )
    elif strategy_get_ground == 'normal_direction_maxmin':
        verts_ground, faces_ground = get_ground_by_points_on_the_ground(
            array_position,
            ratio_border=ratio_border,
            ratio_height=0.5
        )
    elif strategy_get_ground == 'normal_direction_lstsq':
        ground_parameter = get_ground_by_points_on_the_ground_lstsq(array_position, ratio_filter_outliers=0.05)
        verts_ground, faces_ground = convert_ground_parameters_to_4_verts_mesh_with_on_ground_points(
            ground_parameter,
            array_position,
            ratio_border=ratio_border
        )
    else:
        raise NotImplementedError(strategy_get_ground)

    return verts_ground, faces_ground
