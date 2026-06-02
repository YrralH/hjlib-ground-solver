from typing import Tuple

import numpy as np

from hjlib_geometry import reverse_project_via_ground, assert_is_rotmat
from hjlib_smpl.skeleton_helpers import get_2d_torso_center_from_2d_joint

from hjlib_ground_solver.get_ground_param.by_world_space import get_ground_param_in_world_space_with_extrinsic


def get_3d_info_from_hvip_2d(
    array_valid: np.ndarray,
    array_hvip_2d: np.ndarray,
    array_keypoints_17_or_torso_2d: np.ndarray,
    RTs: np.ndarray,
    Ks: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    NUM_FRAME = array_valid.shape[0]
    NUM_PERSON = array_valid.shape[1]
    assert array_valid.shape == (NUM_FRAME, NUM_PERSON), (array_valid.shape, NUM_FRAME, NUM_PERSON)
    assert array_hvip_2d.shape == (NUM_FRAME, NUM_PERSON, 2), (array_hvip_2d.shape, NUM_FRAME, NUM_PERSON)
    assert array_keypoints_17_or_torso_2d.shape in [(NUM_FRAME, NUM_PERSON, 17, 2), (NUM_FRAME, NUM_PERSON, 2)], \
        (array_keypoints_17_or_torso_2d.shape, NUM_FRAME, NUM_PERSON)
    assert RTs.shape == (NUM_FRAME, 4, 4), RTs.shape
    assert Ks.shape == (NUM_FRAME, 3, 3), Ks.shape

    array_hvip_3d_world = np.zeros((NUM_FRAME, NUM_PERSON, 3), dtype=np.float32)

    GROUND_ALL = np.zeros((NUM_FRAME, 4), dtype=np.float32)
    TORSO_2D_ALL = np.zeros((NUM_FRAME, NUM_PERSON, 2), dtype=np.float32)

    for index_frame in range(NUM_FRAME):
        RT_one_frame = RTs[index_frame, :, :]
        assert_is_rotmat(RT_one_frame[:3, :3])

        K_one_frame = Ks[index_frame, :, :]
        assert K_one_frame.shape == (3, 3), K_one_frame.shape

        ground = get_ground_param_in_world_space_with_extrinsic(RT_one_frame, up_axis='z', height=0.0)

        GROUND_ALL[index_frame, :] = ground

        K = Ks[index_frame, :, :]
        for index_person in range(NUM_PERSON):
            hvip_2d = array_hvip_2d[index_frame, index_person, :]
            if not array_valid[index_frame, index_person]:
                continue

            assert not np.any(np.isnan(hvip_2d)), (index_frame, index_person, hvip_2d)

            array_keypoints_17_or_torso_2d_one_person = array_keypoints_17_or_torso_2d[index_frame, index_person]
            if array_keypoints_17_or_torso_2d_one_person.shape == (17, 2):
                torso_2d_one_person = get_2d_torso_center_from_2d_joint(
                    joints_2d=array_keypoints_17_or_torso_2d_one_person
                )
            else:
                assert array_keypoints_17_or_torso_2d_one_person.shape == (2,), array_keypoints_17_or_torso_2d_one_person.shape
                torso_2d_one_person = array_keypoints_17_or_torso_2d_one_person

            TORSO_2D_ALL[index_frame, index_person, :] = torso_2d_one_person
            array_valid[index_frame, index_person] = True

            hvip_3d_camera = reverse_project_via_ground(hvip_2d.reshape(1, 2), ground, K)
            RT_world_to_camera = RT_one_frame
            RT_camera_to_world = np.linalg.inv(RT_world_to_camera)
            hvip_3d_world_one_frame_one_person = hvip_3d_camera @ RT_camera_to_world[:3, :3].T + RT_camera_to_world[:3, 3]
            hvip_3d_world_one_frame_one_person = hvip_3d_world_one_frame_one_person.reshape(3)
            assert abs(hvip_3d_world_one_frame_one_person[2] - 0) < 1e-5, hvip_3d_world_one_frame_one_person[2]
            hvip_3d_world_one_frame_one_person[2] = 0
            array_hvip_3d_world[index_frame, index_person, :] = hvip_3d_world_one_frame_one_person

    return array_hvip_3d_world, GROUND_ALL, TORSO_2D_ALL
