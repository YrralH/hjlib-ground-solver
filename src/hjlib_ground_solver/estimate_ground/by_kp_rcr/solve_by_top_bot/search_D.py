from typing import Optional, Tuple, Union

import numpy as np
import torch

from hjlib_ground_solver.estimate_ground.by_kp_rcr.solve_by_top_bot.project_loss import get_projection_loss


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

    fx = cam_in[0, 0]
    fy = cam_in[1, 1]
    cx = cam_in[0, 2]
    cy = cam_in[1, 2]

    num = uv.shape[1]

    # get rid of for-loop
    term_1 = ground[0] * (1 / fx) * (uv[0, :] - cx)
    term_2 = ground[1] * (1 / fy) * (uv[1, :] - cy)
    term_3 = ground[2]
    new_z = - ground[3] / (term_1 + term_2 + term_3)
    assert new_z.shape == (num,), new_z.shape

    xyz = torch.matmul(torch.inverse(cam_in), (uv * new_z))
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
    device: torch.device = torch.device('cpu')
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:

    if ground_normal[2] > 0:
        ground_normal = -1.0 * ground_normal
    ground_normal_tensor = torch.from_numpy(ground_normal).to(torch.float32).to(device)
    cam_para_tensor = torch.from_numpy(cam_para).to(torch.float32).to(device)
    xb_tensor = torch.from_numpy(xb).to(torch.float32).to(device)
    xt_tensor = torch.from_numpy(xt).to(torch.float32).to(device)

    def forward(
            D: Union[torch.Tensor, float],
            flag_ret_filter_mask: bool = False,
            ratio_filter_keep: float = 0.9
        ) -> Union[
            Tuple[int, torch.Tensor, torch.Tensor],
            Tuple[int, torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
        ]:
        ground_tensor = torch.zeros((4)).to(torch.float32).to(device)
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
                ratio_filter_keep=ratio_filter_keep
            )
            assert len(ret) == 4, len(ret)
            loss_vec, loss_mod, loss_pixel, mask = ret
            return loss_vec, loss_mod, loss_pixel, mask
        else:
            ret = get_projection_loss(
                xb_gt=xb_tensor,
                xt_gt=xt_tensor,
                xt_pred=uv_top
            )
            assert len(ret) == 3, len(ret)
            loss_vec, loss_mod, loss_pixel = ret
            return loss_vec, loss_mod, loss_pixel
    K1 = 0
    K2 = 1
    K3 = 1

    list_loss: list[float] = []
    list_d: list[float] = []
    for d_cand in np.arange(-5, 80, 0.1):
        ret = forward(float(d_cand))
        assert len(ret) == 3, len(ret)
        loss_vec, loss_mod, loss_pixel = ret
        loss = K1 * loss_vec + K2 * loss_mod + K3 * loss_pixel
        list_d.append(float(d_cand))
        list_loss.append(loss.item())

    d_best = list_d[int(np.argmin(list_loss))]

    K1_ret = 0
    K2_ret = 0
    K3_ret = 1
    if flag_ret_filter_mask:
        ret = forward(d_best, flag_ret_filter_mask=flag_ret_filter_mask, ratio_filter_keep=ratio_filter_keep)
        assert len(ret) == 4, len(ret)
        loss_vec, loss_mod, loss_pixel, mask = ret
    else:
        ret = forward(d_best)
        assert len(ret) == 3, len(ret)
        loss_vec, loss_mod, loss_pixel = ret
        mask = None

    loss_ret = K1_ret * loss_vec + K2_ret * loss_mod + K3_ret * loss_pixel

    ground_tensor = torch.zeros((4)).to(torch.float32).to(device)
    ground_tensor[0:3] = ground_normal_tensor
    ground_tensor[3] = d_best

    if flag_ret_filter_mask:
        assert mask is not None
        return ground_tensor.cpu().numpy(), loss_ret.cpu().numpy(), mask.cpu().numpy()
    else:
        return ground_tensor.cpu().numpy(), loss_ret.cpu().numpy()
