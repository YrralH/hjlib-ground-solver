from typing import Optional, Tuple, Union

import torch

from hjlib_ground_solver.estimate_ground.by_kp_rcr.observation_weight import (
    validated_torch_observation_weights,
)


def get_projection_loss(
        xb_gt: torch.Tensor,
        xt_gt: torch.Tensor,
        xt_pred: torch.Tensor,
        flag_ret_filter_mask: bool = False,
        ratio_filter_keep: float = 0.9,
        observation_weights: torch.Tensor | None = None,
    ) -> Union[
        Tuple[int, torch.Tensor, torch.Tensor],
        Tuple[int, torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
    ]:
    '''
    input :
        * xb_gt [3, n] each col [u, v, 1]
        * xt_gt [3, n] each col [u, v, 1]
        * xt_pred [3, n] each col [`u, v, 1]
    '''

    '''
    k is the direction vector of (xt_pred - xb_gt)
    n is perpendicular to vec

    then xt_pred = a*k + b*n (a, b is scalar)
    loss = f(b) + f(a - 1.45)
    '''
    N = xb_gt.shape[1]

    # ignore 1 of [u, v, 1]
    xb_gt = xb_gt[0:2, :]
    xt_gt = xt_gt[0:2, :]
    xt_pred = xt_pred[0:2, :]

    loss_vec = 0

    vecs_pg = xt_gt - xb_gt
    vecs_pq = xt_pred - xb_gt
    bias = vecs_pg - vecs_pq

    mods_bias = torch.norm(bias, p=2, dim=0)
    mods_pg = torch.norm(vecs_pg, p=2, dim=0)
    mods_pq = torch.norm(vecs_pq, p=2, dim=0)

    assert mods_bias.shape == torch.Size([N]), mods_bias.shape
    assert mods_pg.shape == torch.Size([N]), mods_pg.shape
    assert mods_pq.shape == torch.Size([N]), mods_pq.shape

    losses_mod = torch.abs(mods_pg - mods_pq) / mods_pg
    losses_pixel = mods_bias / mods_pg

    assert losses_mod.shape == torch.Size([N]), losses_mod.shape
    assert losses_pixel.shape == torch.Size([N]), losses_pixel.shape

    weights = validated_torch_observation_weights(
        observation_weights,
        N,
        losses_mod.dtype,
        losses_mod.device,
    )

    if flag_ret_filter_mask:
        # consider losses_pixel only
        mask = losses_pixel < torch.quantile(losses_pixel, ratio_filter_keep)
    else:
        mask = None

    if weights is None:
        loss_mod = torch.mean(losses_mod)
        loss_pixel = torch.mean(losses_pixel)
    else:
        weight_sum = torch.sum(weights)
        loss_mod = torch.sum(weights * losses_mod) / weight_sum
        loss_pixel = torch.sum(weights * losses_pixel) / weight_sum

    assert loss_mod.shape == torch.Size([]), loss_mod.shape
    assert loss_pixel.shape == torch.Size([]), loss_pixel.shape

    if flag_ret_filter_mask:
        return loss_vec, loss_mod, loss_pixel, mask
    else:
        return loss_vec, loss_mod, loss_pixel
