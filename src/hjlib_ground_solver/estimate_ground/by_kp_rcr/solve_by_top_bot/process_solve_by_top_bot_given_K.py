from typing import Tuple

import numpy as np
import torch

from hjlib_ground_solver.estimate_ground.by_kp_rcr.compute_KN_by_vertical_lines import get_KN_with_filter
from hjlib_ground_solver.estimate_ground.by_kp_rcr.solve_by_top_bot.search_D import solve_D_search


def solve_ground_param_by_top_bottom_given_K(
    array_top: np.ndarray,
    array_bottom: np.ndarray,
    K: np.ndarray,
    H_prior: float = 1.35,
    D_init: float = 10.0,
    device_solve: torch.device = torch.device('cpu'),
    flag_opt: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    @param array_top: (N, 2) in pixel, usually the middle of two shoulder keypoints
    @param array_bottom: (N, 2) in pixel, usually the middle of two ankle keypoints
    @param K: (3, 3) camera intrinsic matrix
    @param H_prior: prior of human height in meter, for 1.7meter human, the ditance from shoulder to ankle is about 1.35 meter
    @param D_init: initial value of distance from camera to ground plane in meter, usually not matter, but do not be too strange
    @param device_solve: device to use for torch optimization
    '''

    N = array_top.shape[0]
    assert array_top.shape == (N, 2), array_top.shape
    assert array_bottom.shape == (N, 2), array_bottom.shape
    assert K.shape == (3, 3), K.shape

    array_top_homogeneous = np.insert(array_top, 2, values=1.0, axis=1)  # (N, 3)
    array_bottom_homogeneous = np.insert(array_bottom, 2, values=1.0, axis=1)  # (N, 3)
    assert array_top_homogeneous.shape == (N, 3), array_top_homogeneous.shape
    assert array_bottom_homogeneous.shape == (N, 3), array_bottom_homogeneous.shape
    KN = get_KN_with_filter(
        array_bottom_homogeneous.T,
        array_top_homogeneous.T,
        prop_filter=0.24,
        times_filter=2
    )
    assert isinstance(KN, np.ndarray)
    ground_normal: np.ndarray = np.matmul(np.linalg.inv(K), KN)
    ground_normal = ground_normal / np.linalg.norm(ground_normal, ord=2)
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
            flag_ret_filter_mask=False
        )
        assert len(ret) == 2, len(ret)
        ground, loss = ret
        assert ground.shape == (4,), ground.shape
        assert loss >= 0, loss

    return ground, loss
