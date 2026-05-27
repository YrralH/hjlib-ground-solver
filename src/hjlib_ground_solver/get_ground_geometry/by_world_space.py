import numpy as np
import trimesh


def get_ground_geometry_in_world_space(
        up_axis: str = 'y',
        height: float = 0.0,
        half_size: float = 3.0,
    ) -> trimesh.Trimesh:
    if up_axis == 'y':
        points_on_ground = np.array([
            [-half_size, height, half_size],
            [half_size, height, half_size],
            [half_size, height, -half_size],
            [-half_size, height, -half_size]
        ])
    elif up_axis == 'z':
        points_on_ground = np.array([
            [-half_size, half_size, height],
            [half_size, half_size, height],
            [half_size, -half_size, height],
            [-half_size, -half_size, height]
        ])
    else:
        raise NotImplementedError(up_axis)

    faces = np.array([
        [0, 1, 2],
        [0, 2, 3]
    ])

    mesh_ground = trimesh.Trimesh(vertices=points_on_ground, faces=faces)

    return mesh_ground
