from typing import Optional, Tuple, Union

import numpy as np
import torch

from hjlib_ground_solver.estimate_ground.by_kp_rcr.solve_by_top_bot.project_loss import get_projection_loss
from hjlib_ground_solver.estimate_ground.by_kp_rcr.observation_weight import (
    validated_numpy_observation_weights,
)


def uv_to_xyz_via_ground_torch(uv: torch.Tensor, ground: torch.Tensor, cam_in: torch.Tensor) -> torch.Tensor:
    '''
    input:
    Note all in the same device
        * uv [3, n] each col is [u, v, 1]
        * ground [4] A, B, C, D
        * cam in [3, 3]
    output
        * xyz [3, n] each col is [x, y, z]
    '''
    assert len(uv.shape) == 2
    assert uv.shape[0] == 3

    num = uv.shape[1]
    rays = torch.linalg.solve(cam_in, uv)
    denominators = torch.sum(ground[:3, None] * rays, dim=0)
    if bool(torch.any(torch.abs(denominators) <= 1e-8).item()):
        raise ValueError('camera ray is parallel or near-parallel to ground')
    scales = -ground[3] / denominators
    xyz = rays * scales[None, :]
    assert xyz.shape == (3, num), xyz.shape
    if not bool(torch.isfinite(xyz).all().item()):
        raise ValueError('ground intersection must be finite')
    return xyz


def solve_D_search(
    xb: np.ndarray,
    xt: np.ndarray,
    ground_normal: np.ndarray,
    cam_para: np.ndarray,
    H_prior: float = 1.5,
    D_init: float = 10.0,
    flag_ret_filter_mask: bool = False,
    ratio_filter_keep: float = 0.9,
    device: torch.device = torch.device('cpu'),
    *,
    distance_min: float = -5.0,
    distance_max: float = 80.0,
    distance_step: float = 0.1,
    observation_weights: np.ndarray | None = None,
    preserve_ground_normal_orientation: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:

    del D_init

    if type(preserve_ground_normal_orientation) is not bool:
        raise ValueError('preserve_ground_normal_orientation must be a Python bool')

    num_observations = xb.shape[1]
    weights = validated_numpy_observation_weights(
        observation_weights,
        num_observations,
    )

    tensor_dtype = (
        torch.float64 if preserve_ground_normal_orientation else torch.float32
    )
    ground_normal_solver = np.array(ground_normal, copy=True)
    if not preserve_ground_normal_orientation and ground_normal_solver[2] > 0:
        ground_normal_solver = -1.0 * ground_normal_solver
    ground_normal_tensor = torch.from_numpy(ground_normal_solver).to(tensor_dtype).to(device)
    cam_para_tensor = torch.from_numpy(np.array(cam_para, copy=True)).to(tensor_dtype).to(device)
    xb_tensor = torch.from_numpy(np.array(xb, copy=True)).to(tensor_dtype).to(device)
    xt_tensor = torch.from_numpy(np.array(xt, copy=True)).to(tensor_dtype).to(device)
    weights_tensor = (
        None
        if weights is None
        else torch.from_numpy(weights).to(tensor_dtype).to(device)
    )

    def forward(
            D: Union[torch.Tensor, float],
            flag_ret_filter_mask: bool = False,
            ratio_filter_keep: float = 0.9
        ) -> Union[
            Tuple[int, torch.Tensor, torch.Tensor],
            Tuple[int, torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
        ]:
        ground_tensor = torch.zeros((4), dtype=tensor_dtype, device=device)
        ground_tensor[0:3] = ground_normal_tensor
        ground_tensor[3] = D

        # decide "depth"
        xyz_bot = uv_to_xyz_via_ground_torch(xb_tensor, ground_tensor, cam_para_tensor)
        # get person
        xyz_top = (xyz_bot.permute(1, 0) + (ground_normal_tensor * H_prior)).permute(1, 0)
        # project to image
        uv_top = torch.matmul(cam_para_tensor, xyz_top)
        uv_top = uv_top / uv_top[2, :]

        if flag_ret_filter_mask:
            ret = get_projection_loss(
                xb_gt=xb_tensor,
                xt_gt=xt_tensor,
                xt_pred=uv_top,
                flag_ret_filter_mask=flag_ret_filter_mask,
                ratio_filter_keep=ratio_filter_keep,
                observation_weights=weights_tensor,
            )
            assert len(ret) == 4, len(ret)
            loss_vec, loss_mod, loss_pixel, mask = ret
            return loss_vec, loss_mod, loss_pixel, mask
        else:
            ret = get_projection_loss(
                xb_gt=xb_tensor,
                xt_gt=xt_tensor,
                xt_pred=uv_top,
                observation_weights=weights_tensor,
            )
            assert len(ret) == 3, len(ret)
            loss_vec, loss_mod, loss_pixel = ret
            return loss_vec, loss_mod, loss_pixel
    K1 = 0
    K2 = 1
    K3 = 1

    if not (
            np.isfinite(distance_min)
            and np.isfinite(distance_max)
            and np.isfinite(distance_step)
            and distance_min < distance_max
            and distance_step > 0.0
        ):
        raise ValueError('distance search bounds and step are invalid')
    distance_candidates = np.arange(
        distance_min,
        distance_max,
        distance_step,
        dtype=np.float64,
    )
    if distance_candidates.size < 3:
        raise ValueError('distance search requires at least three candidates')
    list_loss: list[float] = []
    for d_cand in distance_candidates:
        ret = forward(float(d_cand))
        assert len(ret) == 3, len(ret)
        loss_vec, loss_mod, loss_pixel = ret
        loss = K1 * loss_vec + K2 * loss_mod + K3 * loss_pixel
        list_loss.append(loss.item())

    index_best = int(np.argmin(list_loss))
    if index_best in (0, distance_candidates.size - 1):
        raise ValueError('best ground distance lies on the search boundary')
    d_best = float(distance_candidates[index_best])

    if flag_ret_filter_mask:
        ret = forward(d_best, flag_ret_filter_mask=flag_ret_filter_mask, ratio_filter_keep=ratio_filter_keep)
        assert len(ret) == 4, len(ret)
        loss_vec, loss_mod, loss_pixel, mask = ret
    else:
        ret = forward(d_best)
        assert len(ret) == 3, len(ret)
        loss_vec, loss_mod, loss_pixel = ret
        mask = None

    loss_ret = loss_mod + loss_pixel

    ground_tensor = torch.zeros((4), dtype=tensor_dtype, device=device)
    ground_tensor[0:3] = ground_normal_tensor
    ground_tensor[3] = d_best

    if flag_ret_filter_mask:
        assert mask is not None
        return ground_tensor.cpu().numpy(), loss_ret.cpu().numpy(), mask.cpu().numpy()
    else:
        return ground_tensor.cpu().numpy(), loss_ret.cpu().numpy()
