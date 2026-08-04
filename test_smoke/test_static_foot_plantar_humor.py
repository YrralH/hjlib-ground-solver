'''Smoke tests for common-domain plantar HuMoR-style clustering.'''

from dataclasses import FrozenInstanceError
from typing import Callable, cast

import numpy as np

import hjlib_ground_solver
import hjlib_ground_solver.estimate_ground as estimate_ground
import hjlib_ground_solver.estimate_ground.by_static_foot_plantar_humor as implementation
from hjlib_ground_solver.estimate_ground.by_static_foot_plantar_humor import (
    Static_Foot_Plantar_HuMoR_Cluster,
    Static_Foot_Plantar_HuMoR_Config,
    Static_Foot_Plantar_HuMoR_Result,
    Static_Foot_Plantar_HuMoR_Sample,
    Static_Foot_Plantar_HuMoR_Status,
    estimate_static_foot_plantar_humor_baseline,
)


def config(speed: float = 0.5) -> Static_Foot_Plantar_HuMoR_Config:
    return Static_Foot_Plantar_HuMoR_Config(speed, 0.005, 3)


def expect_value_error(operation: Callable[[], object]) -> None:
    try:
        operation()
    except ValueError:
        return
    raise AssertionError('operation did not raise ValueError')


def estimate(
        left_height: np.ndarray,
        right_height: np.ndarray,
        left_speed: np.ndarray,
        right_speed: np.ndarray,
        used_config: Static_Foot_Plantar_HuMoR_Config | None = None,
    ) -> Static_Foot_Plantar_HuMoR_Result:
    return estimate_static_foot_plantar_humor_baseline(
        left_height,
        right_height,
        left_speed,
        right_speed,
        config() if used_config is None else used_config,
    )


def test_candidate_order_noise_and_evidence() -> None:
    left_height = np.asarray((0.000, 0.001, 0.002, 0.5, 0.7, 0.9), dtype=np.float64)
    right_height = np.asarray((0.020, 0.021, 0.022, 0.6, 0.8, 1.0), dtype=np.float64)
    left_speed = np.asarray((0.1, 0.1, 0.1, 1.0, 1.0), dtype=np.float64)
    right_speed = np.asarray((0.1, 0.1, 0.1, 0.1, 1.0), dtype=np.float64)
    result = estimate(left_height, right_height, left_speed, right_speed)
    assert result.status == 'candidate'
    assert result.left_eligible_sample_count == 3
    assert result.right_eligible_sample_count == 4
    assert tuple(sample.side for sample in result.samples) == (
        'left', 'left', 'left', 'right', 'right', 'right', 'right',
    )
    assert result.samples[-1].dbscan_label == -1
    assert all(cluster.dbscan_label >= 0 for cluster in result.clusters)
    assert result.selected_dbscan_label == 0
    assert result.candidate_height_in_meter == 0.001
    assert len(result.clusters) == 2
    assert result.clusters[0].is_selected
    assert result.clusters[0].height_span_in_meter == 0.002
    assert result.clusters[0].maximum_adjacent_height_gap_in_meter == 0.001


def test_equal_median_tie_uses_lower_label() -> None:
    heights = np.asarray((0.0, 1.0, 2.0, 0.0, 1.0, 2.0, -10.0), dtype=np.float64)
    frames = np.arange(7, dtype=np.int64)
    labels = np.asarray((2, 2, 2, 0, 0, 0, -1), dtype=np.int64)
    clusters, selected_label, candidate = (
        implementation.summarize_static_foot_plantar_humor_clusters(
            heights,
            frames,
            labels,
        )
    )
    assert tuple(cluster.dbscan_label for cluster in clusters) == (0, 2)
    assert selected_label == 0
    assert candidate == 1.0
    assert clusters[0].is_selected
    assert not clusters[1].is_selected


def test_strict_gate_terminal_repeat_and_empty_statuses() -> None:
    heights = np.asarray((0.0, 0.1, 0.2, 0.3), dtype=np.float64)
    equality = np.full((3,), 0.5, dtype=np.float64)
    no_samples = estimate(heights, heights, equality, equality)
    assert no_samples.status == 'no_contact_samples'
    assert no_samples.candidate_height_in_meter is None

    terminal = np.asarray((1.0, 1.0, 0.1), dtype=np.float64)
    repeated = estimate(heights, heights + 1.0, terminal, equality)
    assert repeated.left_eligible_sample_count == 2
    assert tuple(sample.native_frame_index for sample in repeated.samples) == (2, 3)
    assert repeated.status == 'noise_only'
    assert repeated.clusters == ()


def test_density_chain_and_vertical_equivariance() -> None:
    chained = np.asarray((0.000, 0.004, 0.008, 0.012), dtype=np.float64)
    moving = np.ones((3,), dtype=np.float64)
    static = np.zeros((3,), dtype=np.float64)
    result = estimate(chained, chained + 1.0, static, moving)
    assert result.status == 'candidate'
    assert len(result.clusters) == 1
    cluster = result.clusters[0]
    assert np.isclose(cluster.height_span_in_meter, 0.012)
    assert np.isclose(cluster.maximum_adjacent_height_gap_in_meter, 0.004)

    shift = 0.125
    shifted = estimate(chained + shift, chained + 1.0 + shift, static, moving)
    assert shifted.selected_dbscan_label == result.selected_dbscan_label
    assert np.isclose(
        cast(float, shifted.candidate_height_in_meter),
        cast(float, result.candidate_height_in_meter) + shift,
    )


def test_dtype_immutability_frozen_and_exports() -> None:
    for dtype in (np.float32, np.float64):
        heights = np.asarray((0.0, 0.001, 0.002, 0.003), dtype=dtype)
        speeds = np.zeros((3,), dtype=dtype)
        heights_before = heights.copy()
        speeds_before = speeds.copy()
        result = estimate(heights, heights + dtype(0.1), speeds, speeds)
        assert result.input_dtype == str(np.dtype(dtype))
        assert np.array_equal(heights, heights_before)
        assert np.array_equal(speeds, speeds_before)
    try:
        setattr(result, 'status', 'noise_only')
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError('result must be frozen')

    public_symbols = (
        ('Static_Foot_Plantar_HuMoR_Cluster', Static_Foot_Plantar_HuMoR_Cluster),
        ('Static_Foot_Plantar_HuMoR_Config', Static_Foot_Plantar_HuMoR_Config),
        ('Static_Foot_Plantar_HuMoR_Result', Static_Foot_Plantar_HuMoR_Result),
        ('Static_Foot_Plantar_HuMoR_Sample', Static_Foot_Plantar_HuMoR_Sample),
        ('Static_Foot_Plantar_HuMoR_Status', Static_Foot_Plantar_HuMoR_Status),
        (
            'estimate_static_foot_plantar_humor_baseline',
            estimate_static_foot_plantar_humor_baseline,
        ),
    )
    for name, symbol in public_symbols:
        assert getattr(implementation, name) is symbol
        assert getattr(estimate_ground, name) is symbol
        assert getattr(hjlib_ground_solver, name) is symbol


def test_validation() -> None:
    heights = np.asarray((0.0, 0.1, 0.2), dtype=np.float64)
    speeds = np.asarray((0.0, 0.0), dtype=np.float64)
    invalid_config_builders: tuple[Callable[[], object], ...] = (
        lambda: Static_Foot_Plantar_HuMoR_Config(-0.1, 0.005, 3),
        lambda: Static_Foot_Plantar_HuMoR_Config(0.0, 0.005, 3),
        lambda: Static_Foot_Plantar_HuMoR_Config(float('nan'), 0.005, 3),
        lambda: Static_Foot_Plantar_HuMoR_Config(0.5, 0.0, 3),
        lambda: Static_Foot_Plantar_HuMoR_Config(0.5, float('inf'), 3),
        lambda: Static_Foot_Plantar_HuMoR_Config(0.5, 0.005, 1),
        lambda: Static_Foot_Plantar_HuMoR_Config(0.5, 0.005, cast(int, True)),
    )
    for builder in invalid_config_builders:
        expect_value_error(builder)
    expect_value_error(lambda: estimate(heights[:1], heights[:1], speeds[:0], speeds[:0]))
    expect_value_error(lambda: estimate(heights[:, None], heights, speeds, speeds))
    expect_value_error(lambda: estimate(heights, heights[:2], speeds, speeds))
    expect_value_error(lambda: estimate(heights, heights, speeds[:1], speeds))
    expect_value_error(lambda: estimate(
        heights.astype(np.float32), heights, speeds, speeds,
    ))
    expect_value_error(lambda: estimate(
        heights.astype(np.int64), heights.astype(np.int64),
        speeds.astype(np.int64), speeds.astype(np.int64),
    ))
    nonfinite = heights.copy()
    nonfinite[0] = np.nan
    expect_value_error(lambda: estimate(nonfinite, heights, speeds, speeds))
    negative = speeds.copy()
    negative[0] = -0.1
    expect_value_error(lambda: estimate(heights, heights, negative, speeds))
    expect_value_error(lambda: estimate_static_foot_plantar_humor_baseline(
        heights,
        heights,
        speeds,
        speeds,
        cast(Static_Foot_Plantar_HuMoR_Config, object()),
    ))


def smoke_test_static_foot_plantar_humor() -> None:
    test_candidate_order_noise_and_evidence()
    test_equal_median_tie_uses_lower_label()
    test_strict_gate_terminal_repeat_and_empty_statuses()
    test_density_chain_and_vertical_equivariance()
    test_dtype_immutability_frozen_and_exports()
    test_validation()


if __name__ == '__main__':
    smoke_test_static_foot_plantar_humor()
    print('test_static_foot_plantar_humor: PASS')
