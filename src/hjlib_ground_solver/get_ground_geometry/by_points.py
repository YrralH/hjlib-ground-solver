from typing import List, Tuple

import numpy as np

from hjlib_ground_solver.get_ground_geometry.by_pillars import get_ground_by_pillars_on_the_ground


def get_valid_filter_mask_by_max_value(list_values: np.ndarray, max_value: float) -> List[bool]:
    # inlined from monolith utils_py.get_valid_filter_mask_by_max_value (caller passes the bias ndarray)
    list_mask_valid = [bool(value < max_value) for value in list_values]
    assert len(list_mask_valid) == len(list_values)
    return list_mask_valid


def compute_plane_normal_by_positions(position: np.ndarray) -> np.ndarray:
    X = position[:, 0]
    Y = position[:, 1]
    Z = position[:, 2]
    # fit a plane equation Z = aX + bY + c
    A = np.c_[X, Y, np.ones(X.shape[0])]
    b = Z
    coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    # normal is (a, b, -1)
    normal = np.array([coeffs[0], coeffs[1], -1.0])
    normal /= np.linalg.norm(normal)

    if normal[1] > 0:
        normal = -1 * normal

    return normal


def compute_plane_parameters_by_positions_hj(position: np.ndarray) -> np.ndarray:
    '''
    @param position: np.ndarray, shape=(N, 3)
    @return ground: np.ndarray, shape=(4,)
    '''
    # construct the matrix A
    '''
    A:
    | x1 y1 z1 1 |
    | x2 y2 z2 1 |
    ...
    | xn yn zn 1 |
    '''
    A = np.hstack((position, np.ones((position.shape[0], 1))))

    # To solve AX = 0, where X are the coefficients of the plane
    ____, s, v = np.linalg.svd(A)  # solve by SVD
    X = v[np.argmin(s), :]

    ground = X / np.linalg.norm(X[:3])
    if ground[1] > 0:
        ground = -1 * ground
    return ground


def get_ground_by_points_on_the_ground_lstsq(
        points: np.ndarray,
        ratio_filter_outliers: float = 0.15
    ) -> np.ndarray:
    N_POINTS = points.shape[0]
    assert points.shape == (N_POINTS, 3), points.shape

    ground_init = compute_plane_parameters_by_positions_hj(points)
    assert ground_init.shape == (4,), ground_init.shape
    bias = np.hstack((points, np.ones((points.shape[0], 1)))) @ ground_init.T
    assert bias.shape == (N_POINTS,), bias.shape

    max_value = sorted(bias)[int(N_POINTS * (1 - ratio_filter_outliers))]
    list_mask = get_valid_filter_mask_by_max_value(bias, max_value)
    assert len(list_mask) == N_POINTS, len(list_mask)

    points_filtered = points[np.array(list_mask)]
    N_POINTS_FILTERED = points_filtered.shape[0]
    assert points_filtered.shape == (N_POINTS_FILTERED, 3), points_filtered.shape

    ground_filtered = compute_plane_parameters_by_positions_hj(points_filtered)
    assert ground_filtered.shape == (4,), ground_filtered.shape

    return ground_filtered


def get_ground_by_points_on_the_ground(
        points: np.ndarray,
        ratio_border: float = 0.5,
        ratio_height: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
    '''
    '''
    N = points.shape[0]
    assert points.shape == (N, 3), points.shape

    ground_normal = compute_plane_parameters_by_positions_hj(points)[0:3]
    assert ground_normal.shape == (3,), ground_normal.shape

    ground_normal_repeat = np.repeat(ground_normal[np.newaxis, :], N, axis=0)
    assert ground_normal_repeat.shape == (N, 3), ground_normal_repeat.shape

    verts, faces = get_ground_by_pillars_on_the_ground(
        position=points,
        direction=ground_normal_repeat,
        ratio_border=ratio_border,
        ratio_height=ratio_height
    )

    return verts, faces
