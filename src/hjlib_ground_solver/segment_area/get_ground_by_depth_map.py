from typing import Tuple, List, Optional

import numpy as np
import torch

from hjlib_ground_solver.segment_area.get_depth_map_by_ground import get_depth_map_by_ground_tensor, get_depth_map_by_ground_np


def forward_ground_parameters_to_depth_map(
        ground_parameters: torch.Tensor,
        camera_K: torch.Tensor,
        image_size_wh: Tuple[int, int],
    ) -> torch.Tensor:

    assert ground_parameters.shape == (4,), ground_parameters.shape
    assert camera_K.shape == (3, 3), camera_K.shape
    assert len(image_size_wh) == 2, image_size_wh

    ground_parameters_normalized = torch.zeros(4, dtype=torch.float32)
    ground_parameters_normalized = ground_parameters / torch.norm(ground_parameters[0:3])
    depth_map_pred = get_depth_map_by_ground_tensor(ground_parameters_normalized, camera_K, image_size_wh)
    return depth_map_pred


def get_depth_map_loss(
        depth_map_pred: torch.Tensor,
        depth_map_ref: torch.Tensor,
        mask_valid: Optional[torch.Tensor],
        ratio_topk: float = 0.5
    ) -> torch.Tensor:
    '''
    @param depth_map_pred: torch.Tensor, shape (H, W), dtype float32
    @param depth_map_ref: torch.Tensor, shape (H, W), dtype float32
    @return: loss: torch.Tensor, scalar
    '''

    H, W = depth_map_pred.shape
    assert depth_map_pred.shape == (H, W), depth_map_pred.shape
    assert depth_map_ref.shape == (H, W), depth_map_ref.shape
    if mask_valid is None:
        mask_valid = torch.ones(depth_map_pred.shape, dtype=torch.bool)
    assert mask_valid.shape == depth_map_pred.shape, (mask_valid.shape, depth_map_pred.shape)

    assert depth_map_pred.shape == depth_map_ref.shape, (depth_map_pred.shape, depth_map_ref.shape)
    # get norm of all pixels, loss_depth_all_pixel.shape should be (H, W)
    loss_depth_all_pixel = torch.abs(depth_map_pred[mask_valid] - depth_map_ref[mask_valid])

    NUM_PIXEL_VALID = loss_depth_all_pixel.shape[0]
    assert loss_depth_all_pixel.shape == (NUM_PIXEL_VALID,), loss_depth_all_pixel.shape

    # only consider the top-k of pixels with the smallest absolute depth difference
    _, indices_topk = torch.topk(loss_depth_all_pixel.view(-1), int(NUM_PIXEL_VALID * ratio_topk), largest=False)

    loss_depth_avg = torch.mean(torch.abs(loss_depth_all_pixel.reshape(NUM_PIXEL_VALID)[indices_topk]))

    return loss_depth_avg


def get_ground_main_area_by_depth_map(
        depth_map: np.ndarray,
        camera_K: np.ndarray,
        mask_valid: Optional[np.ndarray],
        ratio_topk: float = 0.5
    ) -> np.ndarray:
    '''
    @param depth_map: np.ndarray, shape (H, W)
    @param camera_K: np.ndarray, shape (3, 3), dtype float32, camera intrinsic matrix
    @return: mask np.ndarray, shape (H, W), dtype bool
    '''

    assert camera_K.shape == (3, 3), camera_K.shape

    if mask_valid is None:
        mask_valid = np.ones(depth_map.shape, dtype=bool)

    assert mask_valid.shape == depth_map.shape, (mask_valid.shape, depth_map.shape)

    # NOTE: the L-BFGS refinement was already commented out in the monolith source,
    # so this function returns the hardcoded initial guess unchanged. The dead
    # closure / optimizer / tensor-setup scaffolding is dropped here (DEAD-3 in
    # docs/design/migration.md); observable behavior is identical. The forward /
    # loss helpers it referenced remain as standalone public functions.
    ground_parameters = np.array([0.034178, -0.899, -0.14, 20], dtype=np.float32)  # Example ground parameters (a, b, c, d)

    return ground_parameters


def get_list_ground_area_by_depth_map(depth_map: np.ndarray, camera_K: np.ndarray) -> List[np.ndarray]:
    '''
    strategy:
    1. Get ground parameters by depth map
    2. remove the pixels that are on the ground plane
    3. back to 1 to get the next ground area
    '''

    H, W = depth_map.shape
    assert depth_map.shape == (H, W), depth_map.shape
    assert camera_K.shape == (3, 3), camera_K.shape

    mask_remaining = np.ones(depth_map.shape, dtype=bool)

    MIN_RATIO_ONE_GROUND = 0.2
    RATIO_TOPK = 0.5  # Ratio of top-k pixels to consider for ground fitting
    ALLOWANCE_RATIO_BELONG_TO_GROUND = 0.2  # Allowance for depth difference to consider a pixel as belonging to the ground

    # first remove depth < 0.1
    mask_remaining = mask_remaining & (depth_map > 0.1)

    list_ground_parameters: List[np.ndarray] = []

    for count_ground in range(10):

        if np.sum(mask_remaining) < (H * W * MIN_RATIO_ONE_GROUND):
            break

        print('Finding ground area %d, remaining pixels ratio: %.2f' % (count_ground, np.sum(mask_remaining) / (H * W)))

        ground_parameters_current = get_ground_main_area_by_depth_map(depth_map, camera_K, mask_remaining, ratio_topk=RATIO_TOPK)

        depth_pred_current = get_depth_map_by_ground_np(ground_parameters_current, camera_K, (W, H))
        depth_diff_relative = np.abs(depth_map - depth_pred_current) / (depth_map + 1e-6)  # Avoid division by zero
        mask_ground_current = depth_diff_relative < ALLOWANCE_RATIO_BELONG_TO_GROUND

        mask_remaining = mask_remaining & ~mask_ground_current

        list_ground_parameters.append(ground_parameters_current)

    return list_ground_parameters
