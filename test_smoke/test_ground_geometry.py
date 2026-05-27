'''Smoke tests for the get_ground_geometry public API (synthetic numpy input).

NOTE: get_ground_by_smpls_on_the_ground requires a real SMPL_Full model (model
weights on disk), so it is covered by the data-dependent / migration tests, not
here. See docs/design/test.md.
'''

import numpy as np

from hjlib_ground_solver import (
    get_ground_by_pillars_on_the_ground,
    compute_plane_normal_by_positions,
    compute_plane_parameters_by_positions_hj,
    get_ground_by_points_on_the_ground_lstsq,
    get_ground_by_points_on_the_ground,
    get_ground_geometry_in_world_space,
)


def make_points_on_plane(n: int = 30) -> np.ndarray:
    # points scattered near the world z=0 plane, with up = +z. A small z jitter
    # keeps the lstsq outlier filter non-degenerate (bias must vary across points).
    np.random.seed(2)
    xy = (np.random.rand(n, 2) - 0.5) * 4.0
    z = np.random.randn(n, 1) * 0.02
    return np.concatenate([xy, z], axis=1).astype(np.float64)


def test_by_pillars() -> None:
    np.random.seed(3)
    n = 30
    position = make_points_on_plane(n)
    direction = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
    direction = direction + np.random.randn(n, 3) * 0.02
    verts, faces = get_ground_by_pillars_on_the_ground(position, direction, ratio_border=0.5, ratio_height=0.0)
    assert verts.shape == (4, 3)
    assert faces.shape == (2, 3)


def test_plane_fits() -> None:
    points = make_points_on_plane()
    normal = compute_plane_normal_by_positions(points)
    assert normal.shape == (3,)
    assert abs(np.linalg.norm(normal) - 1.0) < 1e-5
    ground = compute_plane_parameters_by_positions_hj(points)
    assert ground.shape == (4,)
    assert abs(np.linalg.norm(ground[:3]) - 1.0) < 1e-5


def test_by_points() -> None:
    points = make_points_on_plane()
    verts, faces = get_ground_by_points_on_the_ground(points, ratio_border=0.5, ratio_height=0.0)
    assert verts.shape == (4, 3)
    assert faces.shape == (2, 3)
    ground = get_ground_by_points_on_the_ground_lstsq(points, ratio_filter_outliers=0.15)
    assert ground.shape == (4,)


def test_world_space_geometry() -> None:
    for up_axis in ['y', 'z']:
        mesh = get_ground_geometry_in_world_space(up_axis=up_axis, height=0.0, half_size=3.0)
        assert mesh.vertices.shape == (4, 3)
        assert mesh.faces.shape == (2, 3)


def smoke_test_ground_geometry() -> None:
    test_by_pillars()
    test_plane_fits()
    test_by_points()
    test_world_space_geometry()
    print('[OK] get_ground_geometry')


if __name__ == '__main__':
    smoke_test_ground_geometry()
