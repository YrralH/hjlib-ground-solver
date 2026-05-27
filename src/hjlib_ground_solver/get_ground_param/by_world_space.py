import numpy as np

from hjlib_geometry import solve_ground_from_3_points


def get_ground_param_in_world_space(
        up_axis: str = 'y',
        height: float = 0.0
    ) -> np.ndarray:

    # ^ Warning: dont make ground_param to int
    if up_axis == 'y':
        ground_param = np.array([0, 1.0, 0, -height], dtype=np.float32)
    elif up_axis == 'z':
        ground_param = np.array([0, 0, 1.0, -height], dtype=np.float32)
    else:
        raise NotImplementedError(up_axis)

    assert ground_param.shape == (4,), ground_param.shape

    return ground_param


def get_ground_param_in_world_space_with_extrinsic(
        RT: np.ndarray,
        up_axis: str = 'y',
        height: float = 0.0
    ) -> np.ndarray:

    # ^ Warning: dont make points_on_ground to int
    if up_axis == 'y':
        points_on_ground = np.array([
            [0.0, height, 0],
            [1.0, height, 0],
            [0.0, height, 1]
        ], dtype=np.float32)
    elif up_axis == 'z':
        points_on_ground = np.array([
            [0.0, 0, height],
            [1.0, 0, height],
            [0.0, 1, height]
        ], dtype=np.float32)
    else:
        raise NotImplementedError(up_axis)

    assert RT.shape == (4, 4), RT.shape
    R = RT[:3, :3]
    T = RT[:3, 3]

    points_on_ground_camera_space = points_on_ground @ R.T + T
    ground_param = solve_ground_from_3_points(points_on_ground_camera_space)

    return ground_param
