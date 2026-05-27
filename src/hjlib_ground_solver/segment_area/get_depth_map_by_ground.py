from typing import Tuple

import numpy as np
import torch

from hjlib_geometry import get_depth_of_points_via_ground


def get_depth_map_by_ground_tensor(ground_parameters: torch.Tensor, camera_K: torch.Tensor, image_size_wh: Tuple[int, int]) -> torch.Tensor:
    '''
    @param ground_parameters: torch.Tensor, shape (4,), dtype float32, ground parameters (a, b, c, d)
    @param camera_K: torch.Tensor, shape (3, 3), dtype float32, camera intrinsic matrix
    @param image_size_wh: Tuple[int, int], (W, H)
    @return: depth_map: torch.Tensor, shape (H, W), dtype float32
    '''

    W, H = image_size_wh
    assert camera_K.shape == (3, 3), camera_K.shape

    # Create a grid of pixel coordinates
    x = torch.arange(W).view(1, -1).repeat(H, 1)  # Shape (H, W)
    y = H - torch.arange(H).view(-1, 1).repeat(1, W)  # Shape (H, W)

    points_2d_all_pixel = torch.zeros((H, W, 2), dtype=torch.float32)
    points_2d_all_pixel[:, :, 0] = x  # u
    points_2d_all_pixel[:, :, 1] = y  # v

    depth_all_points = get_depth_of_points_via_ground(points_2d_all_pixel.reshape(H * W, 2), ground_parameters, camera_K)
    assert isinstance(depth_all_points, torch.Tensor), type(depth_all_points)
    assert depth_all_points.shape == (H * W,), depth_all_points.shape
    depth_map = depth_all_points.reshape(H, W)
    return depth_map


def get_depth_map_by_ground_np(ground_parameters: np.ndarray, camera_K: np.ndarray, image_size_wh: Tuple[int, int]) -> np.ndarray:

    ground_parameters_tensor = torch.from_numpy(ground_parameters).to(torch.float32)
    camera_K_tensor = torch.from_numpy(camera_K).to(torch.float32)

    depth_map_tensor = get_depth_map_by_ground_tensor(ground_parameters_tensor, camera_K_tensor, image_size_wh)

    return depth_map_tensor.numpy()  # Convert back to numpy array
