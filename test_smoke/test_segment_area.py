'''Smoke tests for the segment_area public API (synthetic numpy / torch input).'''

import numpy as np
import torch

from hjlib_ground_solver import (
    cluster_pixel_features,
    get_pixel_features_of_depth_map,
    filter_vertical_and_horizontal_features,
    get_depth_map_by_ground_tensor,
    get_depth_map_by_ground_np,
    forward_ground_parameters_to_depth_map,
    get_depth_map_loss,
    get_ground_main_area_by_depth_map,
    get_list_ground_area_by_depth_map,
)


def make_K() -> np.ndarray:
    return np.array([[500.0, 0.0, 16.0], [0.0, 500.0, 16.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def make_ground_param() -> np.ndarray:
    # a downward-tilted plane in front of the camera (normalized normal)
    g = np.array([0.0, -0.9, -0.1, 20.0], dtype=np.float32)
    g[:3] = g[:3] / np.linalg.norm(g[:3])
    return g


def test_cluster_pixel_features() -> None:
    np.random.seed(0)
    features = np.random.rand(8, 10, 3).astype(np.float32)
    for encoding in ['none', 'y', 'x', 'xy']:
        list_mask = cluster_pixel_features(features, position_encoding=encoding)
        assert isinstance(list_mask, list)
        assert all(m.shape == (8, 10) for m in list_mask)


def test_depth_features() -> None:
    np.random.seed(1)
    depth_map = np.random.rand(16, 20).astype(np.float32) * 5.0
    features = get_pixel_features_of_depth_map(depth_map)
    assert features.shape == (16, 20, 2)
    list_mask = filter_vertical_and_horizontal_features(features)
    assert len(list_mask) == 2
    assert all(m.shape == (16, 20) for m in list_mask)


def test_depth_map_by_ground() -> None:
    K = make_K()
    g = make_ground_param()
    W, H = 20, 16
    dm_np = get_depth_map_by_ground_np(g, K, (W, H))
    assert dm_np.shape == (H, W)
    dm_t = get_depth_map_by_ground_tensor(
        torch.from_numpy(g), torch.from_numpy(K), (W, H)
    )
    assert dm_t.shape == (H, W)
    assert np.allclose(dm_np, dm_t.numpy(), atol=1e-4)


def test_forward_and_loss() -> None:
    K = torch.from_numpy(make_K())
    g = torch.from_numpy(make_ground_param())
    W, H = 20, 16
    dm_pred = forward_ground_parameters_to_depth_map(g, K, (W, H))
    assert dm_pred.shape == (H, W)
    dm_ref = dm_pred.clone()
    loss = get_depth_map_loss(dm_pred, dm_ref, None, ratio_topk=0.5)
    assert loss.shape == torch.Size([])
    assert float(loss.item()) >= 0.0


def test_ground_area_by_depth_map() -> None:
    K = make_K()
    g = make_ground_param()
    W, H = 20, 16
    depth_map = get_depth_map_by_ground_np(g, K, (W, H))
    depth_map = np.abs(depth_map) + 0.5  # keep strictly positive
    params = get_ground_main_area_by_depth_map(depth_map, K, None)
    assert params.shape == (4,)
    list_params = get_list_ground_area_by_depth_map(depth_map, K)
    assert isinstance(list_params, list)
    assert all(p.shape == (4,) for p in list_params)


def smoke_test_segment_area() -> None:
    test_cluster_pixel_features()
    test_depth_features()
    test_depth_map_by_ground()
    test_forward_and_loss()
    test_ground_area_by_depth_map()
    print('[OK] segment_area')


if __name__ == '__main__':
    smoke_test_segment_area()
