'''Synthetic contract tests for the exact HuMoR static-foot comparator.'''

from dataclasses import FrozenInstanceError
import math
from typing import Any, Callable, cast

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import DBSCAN

import hjlib_ground_solver as package_root
import hjlib_ground_solver.estimate_ground as estimate_subpackage
from hjlib_ground_solver import (
    Static_Foot_HuMoR_Result,
    estimate_static_foot_humor_baseline,
)
from hjlib_ground_solver.estimate_ground.by_static_foot_humor import (
    STATIC_FOOT_HUMOR_CONFIG,
)


type ARRAY_F = NDArray[np.floating[Any]]


def expect_value_error(operation: Callable[[], object]) -> None:
    try:
        operation()
    except ValueError:
        return
    raise AssertionError('operation must raise ValueError')


def literal_upstream_oracle(
        root: ARRAY_F,
        left_toe: ARRAY_F,
        right_toe: ARRAY_F,
        frame_rate_in_hz: float,
    ) -> tuple[float, float, bool, NDArray[np.int64]]:
    '''Literal independent translation of the pinned HuMoR source subset.'''
    left_velocity = np.linalg.norm(left_toe[1:] - left_toe[:-1], axis=1)
    right_velocity = np.linalg.norm(right_toe[1:] - right_toe[:-1], axis=1)
    left_velocity = np.concatenate((left_velocity, left_velocity[-1:]))
    right_velocity = np.concatenate((right_velocity, right_velocity[-1:]))
    left_static = left_velocity < 0.005
    right_static = right_velocity < 0.005
    static_heights = np.concatenate(
        (left_toe[left_static, 2], right_toe[right_static, 2])
    )
    static_indices = np.concatenate(
        (np.where(left_static)[0], np.where(right_static)[0])
    )
    if static_heights.size == 0:
        return 0.0, 0.0, False, np.empty((0,), dtype=np.int64)
    labels = DBSCAN(eps=0.005, min_samples=3).fit_predict(
        static_heights.reshape(-1, 1)
    ).astype(np.int64, copy=False)
    cluster_heights: dict[int, float] = {}
    cluster_roots: dict[int, float] = {}
    cluster_sizes: dict[int, int] = {}
    floor_height = math.inf
    floor_root_height = math.inf
    for label_value in np.unique(labels).tolist():
        label = int(label_value)
        member = labels == label
        toe_median = float(np.median(static_heights[member]))
        root_median = float(
            np.median(root[np.unique(static_indices[member]), 2])
        )
        cluster_heights[label] = toe_median
        cluster_roots[label] = root_median
        cluster_sizes[label] = int(np.count_nonzero(member))
        if toe_median < floor_height:
            floor_height = toe_median
            floor_root_height = root_median
    terrain = any(
        cluster_roots[label] > floor_root_height + 0.04
        and cluster_heights[label] > floor_height + 0.04
        and cluster_sizes[label] > int(0.25 * frame_rate_in_hz)
        for label in cluster_heights
    )
    return floor_height, floor_height - 0.01, terrain, labels


def stationary_tracks(
        dtype: type[np.float32] | type[np.float64],
        frame_count: int = 8,
    ) -> tuple[ARRAY_F, ARRAY_F, ARRAY_F]:
    root = np.zeros((frame_count, 3), dtype=dtype)
    left = np.zeros((frame_count, 3), dtype=dtype)
    right = np.zeros((frame_count, 3), dtype=dtype)
    root[:, 2] = 1.0
    left[:, 2] = 0.02
    right[:, 2] = 0.021
    return root, left, right


def compare_with_oracle(
        root: ARRAY_F,
        left: ARRAY_F,
        right: ARRAY_F,
        frame_rate: float,
    ) -> Static_Foot_HuMoR_Result:
    expected_toe, expected_floor, expected_terrain, labels = literal_upstream_oracle(
        root,
        left,
        right,
        frame_rate,
    )
    result = estimate_static_foot_humor_baseline(root, left, right, frame_rate)
    assert result.toe_joint_floor_height_in_meter == expected_toe
    assert result.upstream_floor_candidate_height_in_meter == expected_floor
    assert result.terrain_interaction is expected_terrain
    assert tuple(sample.dbscan_label for sample in result.samples) == tuple(
        int(label) for label in labels.tolist()
    )
    return result


def test_dense_cluster_oracle_dtype_order_and_immutability() -> None:
    for dtype in (np.float32, np.float64):
        root, left, right = stationary_tracks(dtype)
        root_before = root.copy()
        left_before = left.copy()
        right_before = right.copy()
        result = compare_with_oracle(root, left, right, 60.0)
        assert result.status == 'upstream_candidate'
        assert result.input_dtype == np.dtype(dtype).name
        assert result.selected_dbscan_label == 0
        assert result.accepted_candidate_height_in_meter == \
            result.upstream_floor_candidate_height_in_meter
        assert tuple(sample.side for sample in result.samples[:8]) == ('left',) * 8
        assert tuple(sample.side for sample in result.samples[8:]) == ('right',) * 8
        assert result.clusters[0].native_frame_indices == tuple(range(8))
        assert result.clusters[0].sample_count == 16
        assert np.array_equal(root, root_before)
        assert np.array_equal(left, left_before)
        assert np.array_equal(right, right_before)
        try:
            setattr(result, 'frame_count', 7)
        except FrozenInstanceError:
            pass
        else:
            raise AssertionError('result record must be frozen')


def test_displacement_is_only_a_permissive_gate() -> None:
    root, left_slow, right_slow = stationary_tracks(np.float64, 6)
    left_fast = left_slow.copy()
    right_fast = right_slow.copy()
    left_slow[:, 0] = np.arange(6) * 0.001
    right_slow[:, 0] = np.arange(6) * 0.001
    left_fast[:, 0] = np.arange(6) * 0.004
    right_fast[:, 0] = np.arange(6) * 0.004
    slow = compare_with_oracle(root, left_slow, right_slow, 60.0)
    fast = compare_with_oracle(root, left_fast, right_fast, 60.0)
    assert slow.samples == fast.samples
    assert slow.clusters == fast.clusters
    assert slow.toe_joint_floor_height_in_meter == fast.toe_joint_floor_height_in_meter
    assert slow.left_toe_displacement_in_meter != \
        fast.left_toe_displacement_in_meter


def test_strict_gate_equality_repeated_final_and_zero_fallback() -> None:
    root, left, right = stationary_tracks(np.float64, 2)
    left[:, 0] = (0.0, 0.005)
    right[:, 0] = (0.0, 0.005)
    result = compare_with_oracle(root, left, right, 30.0)
    assert result.left_toe_displacement_in_meter == (0.005, 0.005)
    assert result.right_toe_displacement_in_meter == (0.005, 0.005)
    assert result.status == 'upstream_zero_fallback'
    assert result.samples == ()
    assert result.clusters == ()
    assert result.selected_dbscan_label is None
    assert result.toe_joint_floor_height_in_meter == 0.0
    assert result.upstream_floor_candidate_height_in_meter == 0.0
    assert result.accepted_candidate_height_in_meter is None


def test_noise_is_pooled_and_lowest_label_median_wins() -> None:
    root, left, right = stationary_tracks(np.float64, 8)
    # Each two-frame plateau contributes its first frame; the large outgoing
    # jump rejects the second. All retained heights are isolated DBSCAN noise.
    left[:, 2] = (0.00, 0.00, 0.02, 0.02, 0.04, 0.04, 0.06, 0.08)
    right[:, 2] = (0.01, 0.01, 0.03, 0.03, 0.05, 0.05, 0.07, 0.09)
    result = compare_with_oracle(root, left, right, 30.0)
    assert set(sample.dbscan_label for sample in result.samples) == {-1}
    assert len(result.clusters) == 1
    assert result.clusters[0].dbscan_label == -1
    assert result.clusters[0].is_selected
    assert result.selected_dbscan_label == -1


def test_dbscan_epsilon_is_inclusive_and_root_frames_are_unique() -> None:
    root, left, right = stationary_tracks(np.float64, 5)
    left[:, 2] = 0.0
    right[:, 2] = 0.005
    epsilon_result = compare_with_oracle(root, left, right, 60.0)
    assert len(epsilon_result.clusters) == 1
    assert epsilon_result.clusters[0].sample_count == 10

    root[:, 2] = np.arange(5, dtype=np.float64)
    left[:, 2] = 0.0
    right[:, 2] = (0.0, 0.0, 0.10, 0.20, 0.30)
    unique_root_result = compare_with_oracle(root, left, right, 60.0)
    selected = next(
        cluster for cluster in unique_root_result.clusters if cluster.is_selected
    )
    assert selected.sample_count == 6
    assert selected.native_frame_indices == (0, 1, 2, 3, 4)
    assert selected.root_height_median_in_meter == 2.0


def test_mixed_noise_dense_cluster_and_equal_median_label_tie() -> None:
    root, left, right = stationary_tracks(np.float64, 6)
    left[:, 2] = 0.0
    right[:, 2] = (-0.10, -0.10, 0.0, 0.10, 0.10, 0.20)
    result = compare_with_oracle(root, left, right, 60.0)
    assert tuple(cluster.dbscan_label for cluster in result.clusters) == (-1, 0)
    noise, dense = result.clusters
    assert noise.sample_count == 2
    assert dense.sample_count == 6
    assert noise.toe_height_median_in_meter == 0.0
    assert dense.toe_height_median_in_meter == 0.0
    assert result.selected_dbscan_label == -1
    assert noise.is_selected
    assert not dense.is_selected


def terrain_tracks(
        high_root: float = 1.10,
        high_toe: float = 0.10,
    ) -> tuple[ARRAY_F, ARRAY_F, ARRAY_F]:
    root, left, right = stationary_tracks(np.float64, 12)
    root[:6, 2] = 1.0
    root[6:, 2] = high_root
    left[:6, 2] = 0.0
    right[:6, 2] = 0.001
    left[6:, 2] = high_toe
    right[6:, 2] = high_toe + 0.001
    return root, left, right


def test_terrain_three_way_conjunction_and_strict_boundaries() -> None:
    root, left, right = terrain_tracks()
    terrain = compare_with_oracle(root, left, right, 40.0)
    assert terrain.status == 'upstream_terrain_rejection'
    assert terrain.terrain_minimum_exclusive_sample_count == 10
    assert terrain.accepted_candidate_height_in_meter is None
    assert any(cluster.triggers_terrain_rejection for cluster in terrain.clusters)

    count_equality = compare_with_oracle(root, left, right, 48.0)
    assert count_equality.terrain_minimum_exclusive_sample_count == 12
    assert not count_equality.terrain_interaction
    root_equal, left_equal, right_equal = terrain_tracks(
        high_root=1.04,
        high_toe=0.04,
    )
    height_equality = compare_with_oracle(
        root_equal,
        left_equal,
        right_equal,
        40.0,
    )
    assert not height_equality.terrain_interaction
    root_low, left_high, right_high = terrain_tracks(high_root=1.03)
    missing_root = compare_with_oracle(root_low, left_high, right_high, 40.0)
    assert not missing_root.terrain_interaction


def test_boundary_safe_transform_invariants() -> None:
    root, left, right = stationary_tracks(np.float64, 8)
    baseline = compare_with_oracle(root, left, right, 60.0)
    vertical = np.array((0.0, 0.0, 0.5), dtype=np.float64)
    shifted = compare_with_oracle(
        root + vertical,
        left + vertical,
        right + vertical,
        60.0,
    )
    assert shifted.selected_dbscan_label == baseline.selected_dbscan_label
    assert shifted.toe_joint_floor_height_in_meter == \
        baseline.toe_joint_floor_height_in_meter + 0.5
    horizontal = np.array((4.0, -2.0, 0.0), dtype=np.float64)
    translated = compare_with_oracle(
        root + horizontal,
        left + horizontal,
        right + horizontal,
        60.0,
    )
    assert translated == baseline
    moving_left = left.copy()
    moving_right = right.copy()
    moving_left[:, 0] = np.arange(8, dtype=np.float64) * 0.001
    moving_right[:, 1] = np.arange(8, dtype=np.float64) * 0.001
    moving = compare_with_oracle(root, moving_left, moving_right, 60.0)
    rotation = np.array(
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    rotated = compare_with_oracle(
        root @ rotation.T,
        moving_left @ rotation.T,
        moving_right @ rotation.T,
        60.0,
    )
    assert rotated == moving


def test_validation_and_derived_overflow() -> None:
    root, left, right = stationary_tracks(np.float64, 4)
    invalid_tracks: tuple[object, ...] = (
        [[0.0, 0.0, 0.0]],
        np.zeros((4, 3), dtype=np.float16),
        np.zeros((4, 3), dtype=np.int64),
        np.zeros((4, 2), dtype=np.float64),
        np.zeros((1, 3), dtype=np.float64),
    )
    for invalid in invalid_tracks:
        expect_value_error(
            lambda invalid=invalid: estimate_static_foot_humor_baseline(
                cast(ARRAY_F, invalid),
                left,
                right,
                60.0,
            )
        )
    mismatched_dtype = left.astype(np.float32)
    expect_value_error(
        lambda: estimate_static_foot_humor_baseline(
            root,
            mismatched_dtype,
            right,
            60.0,
        )
    )
    nonfinite = root.copy()
    nonfinite[0, 0] = np.nan
    expect_value_error(
        lambda: estimate_static_foot_humor_baseline(
            nonfinite,
            left,
            right,
            60.0,
        )
    )
    huge_left = left.astype(np.float32)
    huge_right = right.astype(np.float32)
    huge_root = root.astype(np.float32)
    huge_left[:, 0] = (np.finfo(np.float32).max, -np.finfo(np.float32).max, 0.0, 0.0)
    expect_value_error(
        lambda: estimate_static_foot_humor_baseline(
            huge_root,
            huge_left,
            huge_right,
            60.0,
        )
    )
    for invalid_rate in (True, np.float64(60.0), 0, -1.0, math.inf, math.nan):
        expect_value_error(
            lambda invalid_rate=invalid_rate: estimate_static_foot_humor_baseline(
                root,
                left,
                right,
                cast(float, invalid_rate),
            )
        )


def test_public_reexports_and_frozen_config() -> None:
    names = (
        'Static_Foot_HuMoR_Cluster',
        'Static_Foot_HuMoR_Config',
        'Static_Foot_HuMoR_Result',
        'Static_Foot_HuMoR_Sample',
        'Static_Foot_HuMoR_Status',
        'estimate_static_foot_humor_baseline',
    )
    for name in names:
        assert getattr(package_root, name) is getattr(estimate_subpackage, name)
    assert STATIC_FOOT_HUMOR_CONFIG.upstream_commit == \
        'fc6ef84f0baa153be15427402e0147ed1a63a11a'
    assert STATIC_FOOT_HUMOR_CONFIG.displacement_threshold_in_meter_per_native_frame \
        == 0.005
    try:
        setattr(STATIC_FOOT_HUMOR_CONFIG, 'dbscan_epsilon_in_meter', 1.0)
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError('config record must be frozen')


def smoke_test_static_foot_humor() -> None:
    test_dense_cluster_oracle_dtype_order_and_immutability()
    test_displacement_is_only_a_permissive_gate()
    test_strict_gate_equality_repeated_final_and_zero_fallback()
    test_noise_is_pooled_and_lowest_label_median_wins()
    test_dbscan_epsilon_is_inclusive_and_root_frames_are_unique()
    test_mixed_noise_dense_cluster_and_equal_median_label_tie()
    test_terrain_three_way_conjunction_and_strict_boundaries()
    test_boundary_safe_transform_invariants()
    test_validation_and_derived_overflow()
    test_public_reexports_and_frozen_config()


if __name__ == '__main__':
    smoke_test_static_foot_humor()
    print('[OK] static-foot HuMoR baseline')
