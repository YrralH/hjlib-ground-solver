from typing import Tuple

import numpy as np
import torch

from hjlib_ground_solver.estimate_ground.by_kp_rcr.compute_KN_by_vertical_lines import get_KN_with_filter
from hjlib_ground_solver.estimate_ground.by_kp_rcr.observation_weight import (
    validated_numpy_observation_weights,
)
from hjlib_ground_solver.estimate_ground.by_kp_rcr.solve_by_top_bot.search_D import solve_D_search


def solve_ground_param_by_top_bottom_given_K(
    array_top: np.ndarray,
    array_bottom: np.ndarray,
    K: np.ndarray,
    H_prior: float = 1.35,
    D_init: float = 10.0,
    device_solve: torch.device = torch.device('cpu'),
    flag_opt: bool = False,
    *,
    distance_min: float = -5.0,
    distance_max: float = 80.0,
    distance_step: float = 0.1,
    observation_weights: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    @param array_top: (N, 2) in pixel, usually the middle of two shoulder keypoints
    @param array_bottom: (N, 2) in pixel, usually the middle of two ankle keypoints
    @param K: (3, 3) camera intrinsic matrix
    @param H_prior: prior of human height in meter, for 1.7meter human, the ditance from shoulder to ankle is about 1.35 meter
    @param D_init: compatibility-only legacy argument; the grid search does not use it
    @param distance_min: inclusive lower distance-search bound in meters
    @param distance_max: exclusive upper distance-search bound in meters
    @param distance_step: distance-search grid step in meters
    @param device_solve: device to use for torch optimization
    @return: homogeneous camera-frame ground plane and the dimensionless
        objective used to select its distance
    '''

    N = array_top.shape[0]
    assert array_top.shape == (N, 2), array_top.shape
    assert array_bottom.shape == (N, 2), array_bottom.shape
    assert K.shape == (3, 3), K.shape
    if N < 3:
        raise ValueError('RCR ground solving requires at least three observations')
    weights = validated_numpy_observation_weights(observation_weights, N)
    if (
            not bool(np.isfinite(array_top).all())
            or not bool(np.isfinite(array_bottom).all())
            or not bool(np.isfinite(K).all())
        ):
        raise ValueError('RCR inputs must be finite')
    segment_lengths = np.linalg.norm(array_top - array_bottom, axis=1)
    if bool(np.any(segment_lengths <= 0.0)):
        raise ValueError('RCR top-bottom observations must be nondegenerate')
    determinant = float(np.linalg.det(K))
    if not np.isfinite(determinant) or abs(determinant) <= 1e-12:
        raise ValueError('RCR camera intrinsics must be nonsingular')

    array_top_homogeneous = np.insert(array_top, 2, values=1.0, axis=1)  # (N, 3)
    array_bottom_homogeneous = np.insert(array_bottom, 2, values=1.0, axis=1)  # (N, 3)
    assert array_top_homogeneous.shape == (N, 3), array_top_homogeneous.shape
    assert array_bottom_homogeneous.shape == (N, 3), array_bottom_homogeneous.shape
    KN = get_KN_with_filter(
        array_bottom_homogeneous.T,
        array_top_homogeneous.T,
        prop_filter=0.24,
        times_filter=2,
        observation_weights=weights,
    )
    assert isinstance(KN, np.ndarray)
    ground_normal: np.ndarray = np.linalg.solve(K, KN)
    normal_norm = float(np.linalg.norm(ground_normal, ord=2))
    if not np.isfinite(normal_norm) or normal_norm <= 0.0:
        raise ValueError('RCR ground normal must be finite and nondegenerate')
    ground_normal = ground_normal / normal_norm
    assert ground_normal.shape == (3,), ground_normal.shape

    if flag_opt:
        raise NotImplementedError('do not use opt for now, since it is not stable')
    else:
        ret = solve_D_search(
            xb=array_bottom_homogeneous.T,
            xt=array_top_homogeneous.T,
            ground_normal=ground_normal,
            cam_para=K,
            H_prior=H_prior,
            D_init=D_init,
            distance_min=distance_min,
            distance_max=distance_max,
            distance_step=distance_step,
            flag_ret_filter_mask=False,
            device=device_solve,
            observation_weights=weights,
        )
        assert len(ret) == 2, len(ret)
        ground, loss = ret
        assert ground.shape == (4,), ground.shape
        assert loss >= 0, loss

    return ground, loss
