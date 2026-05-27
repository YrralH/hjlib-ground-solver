'''Smoke tests for get_ground_param + hvip public API (synthetic numpy input).'''

import numpy as np

from hjlib_ground_solver import (
    get_ground_param_in_world_space,
    get_ground_param_in_world_space_with_extrinsic,
    get_3d_info_from_hvip_2d,
)


def make_camera_looking_down() -> np.ndarray:
    # camera at world (0, 0, 3) looking toward -z; world z=0 is the ground plane.
    # R_world_to_camera maps world axes into camera frame (proper rotation, det=+1).
    R_w2c = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float32)
    C = np.array([0.0, 0.0, 3.0], dtype=np.float32)
    T = -R_w2c @ C
    RT = np.eye(4, dtype=np.float32)
    RT[:3, :3] = R_w2c
    RT[:3, 3] = T
    return RT


def make_K() -> np.ndarray:
    return np.array([[500.0, 0.0, 256.0], [0.0, 500.0, 256.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def test_ground_param_world_space() -> None:
    for up_axis in ['y', 'z']:
        g = get_ground_param_in_world_space(up_axis=up_axis, height=0.0)
        assert g.shape == (4,)
    RT = make_camera_looking_down()
    g_ext = get_ground_param_in_world_space_with_extrinsic(RT, up_axis='z', height=0.0)
    assert g_ext.shape == (4,)


def test_get_3d_info_from_hvip_2d() -> None:
    RT = make_camera_looking_down()
    K = make_K()
    NUM_FRAME, NUM_PERSON = 1, 1
    array_valid = np.ones((NUM_FRAME, NUM_PERSON), dtype=bool)
    # hvip_2d = principal point => reverse-projects to world origin (z=0)
    array_hvip_2d = np.array([[[256.0, 256.0]]], dtype=np.float32)
    array_torso_2d = np.array([[[256.0, 256.0]]], dtype=np.float32)
    RTs = RT[None, :, :]
    Ks = K[None, :, :]

    hvip_3d_world, ground_all, torso_all = get_3d_info_from_hvip_2d(
        array_valid, array_hvip_2d, array_torso_2d, RTs, Ks
    )
    assert hvip_3d_world.shape == (NUM_FRAME, NUM_PERSON, 3)
    assert ground_all.shape == (NUM_FRAME, 4)
    assert torso_all.shape == (NUM_FRAME, NUM_PERSON, 2)
    # the recovered world HVIP must lie on the ground (z == 0 by construction)
    assert abs(float(hvip_3d_world[0, 0, 2])) < 1e-4


def smoke_test_ground_param_and_hvip() -> None:
    test_ground_param_world_space()
    test_get_3d_info_from_hvip_2d()
    print('[OK] get_ground_param + hvip')


if __name__ == '__main__':
    smoke_test_ground_param_and_hvip()
