'''Smoke tests for person-height consensus Ground Offset estimation.'''

import numpy as np
from numpy.typing import NDArray

from hjlib_ground_solver import (
    Person_Height_Consensus_Config,
    Person_Height_Consensus_Observations,
    compute_power8_ipose_measurements,
    ground_offset_config,
    solve_ground_offset_by_person_height_consensus,
)


type Float_Array = NDArray[np.float64]


K = np.array([
    [1000.0, 0.0, 0.0],
    [0.0, 1000.0, 0.0],
    [0.0, 0.0, 1.0],
], dtype=np.float64)
NORMAL = np.array([0.0, -1.0, 0.0], dtype=np.float64)


def pose_for_ratio(q: float, bottom_y: float, x: float = 0.0) -> Float_Array:
    height = q * bottom_y
    shoulder_y = bottom_y - height
    hip_y = shoulder_y + 0.2 * height
    knee_y = hip_y + 0.3 * height
    ankle_y = bottom_y
    return np.array([
        [[x - 5.0, shoulder_y], [x + 5.0, shoulder_y]],
        [[x - 5.0, hip_y], [x + 5.0, hip_y]],
        [[x - 5.0, knee_y], [x + 5.0, knee_y]],
        [[x - 5.0, ankle_y], [x + 5.0, ankle_y]],
    ], dtype=np.float64)


def observations(
        person_ids: list[int],
        frame_indices: list[int],
        ratios: list[float],
    ) -> Person_Height_Consensus_Observations:
    poses = np.stack([pose_for_ratio(q, 500.0) for q in ratios])
    return Person_Height_Consensus_Observations(
        np.asarray(person_ids, dtype=np.int64),
        np.asarray(frame_indices, dtype=np.int64),
        poses[:, 0],
        poses[:, 1],
        poses[:, 2],
        poses[:, 3],
    )


def test_virtual_projection_inverse_and_equal_person_weight() -> None:
    person_ids = [0] * 11 + [1]
    frames = list(range(11)) + [0]
    ratios = [0.2] * 11 + [0.4]
    result = solve_ground_offset_by_person_height_consensus(
        observations(person_ids, frames, ratios),
        NORMAL,
        K,
        1.2,
        Person_Height_Consensus_Config(0.05, 1, 2),
    )
    np.testing.assert_allclose(
        result.observation_height_over_offset,
        np.asarray(ratios),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(result.person_height_over_offset, [0.2, 0.4])
    np.testing.assert_allclose(result.scene_height_over_offset, 0.3)
    np.testing.assert_allclose(result.plane_camera_abcd, [0.0, -1.0, 0.0, 4.0])
    np.testing.assert_allclose(result.person_equivalent_height_m, [0.8, 1.6])


def test_middle90_and_support_masks() -> None:
    person_ids = [0] * 20 + [1]
    frames = list(range(20)) + [0]
    ratios = [0.01] + [0.2] * 18 + [2.0] + [0.4]
    result = solve_ground_offset_by_person_height_consensus(
        observations(person_ids, frames, ratios),
        NORMAL,
        K,
        1.2,
        Person_Height_Consensus_Config(0.05, 2, 1),
    )
    np.testing.assert_allclose(result.person_height_over_offset[0], 0.2)
    assert result.person_valid_frame_counts.tolist() == [20, 1]
    assert result.person_eligible_mask.tolist() == [True, False]
    assert result.scene_person_retained_mask.tolist() == [True, False]


def test_corrected_bottom_uses_projective_vertical() -> None:
    pose = pose_for_ratio(0.2, 500.0)
    pose[3, 0, 0] = -10.0
    pose[3, 1, 0] = 50.0
    value = Person_Height_Consensus_Observations(
        np.array([0], dtype=np.int64),
        np.array([0], dtype=np.int64),
        pose[None, 0],
        pose[None, 1],
        pose[None, 2],
        pose[None, 3],
    )
    measured = compute_power8_ipose_measurements(value, K, NORMAL)
    assert float(np.mean(pose[3, :, 0])) == 20.0
    np.testing.assert_allclose(measured.corrected_bottom_xy_px[0, 0], 0.0)
    np.testing.assert_allclose(measured.corrected_bottom_xy_px[0, 1], 500.0)


def test_invalid_geometry_and_old_baseline_regression() -> None:
    value = observations([0], [0], [0.2])
    try:
        solve_ground_offset_by_person_height_consensus(
            value,
            NORMAL,
            K,
            1.2,
            Person_Height_Consensus_Config(0.05, 2, 1),
        )
    except ValueError as error:
        assert 'insufficient retained people' in str(error)
    else:
        raise AssertionError('insufficient support must fail')
    old = ground_offset_config('ground_offset_baseline001')
    assert old.height_prior_m == 1.27
    assert old.distance_step_m == 0.1
    assert old.confidence_threshold_strict_gt == 4.3


def smoke_test_person_height_consensus_ground_offset() -> None:
    test_virtual_projection_inverse_and_equal_person_weight()
    test_middle90_and_support_masks()
    test_corrected_bottom_uses_projective_vertical()
    test_invalid_geometry_and_old_baseline_regression()


if __name__ == '__main__':
    smoke_test_person_height_consensus_ground_offset()
    print('[OK] person_height_consensus_ground_offset')
