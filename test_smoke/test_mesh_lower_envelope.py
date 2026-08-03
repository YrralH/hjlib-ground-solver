'''Synthetic smoke tests for full-mesh lower-envelope statistics.'''

from dataclasses import FrozenInstanceError
from decimal import Decimal, ROUND_CEILING
from typing import Any, Callable, cast

import numpy as np
from numpy.typing import NDArray
import torch

import hjlib_ground_solver as package_root
import hjlib_ground_solver.estimate_ground as estimate_subpackage
import hjlib_ground_solver.estimate_ground.by_mesh_lower_envelope as implementation_module
from hjlib_ground_solver import (
    Mesh_Lower_Envelope_Candidate,
    Mesh_Lower_Envelope_Summary,
    compute_per_frame_mesh_minimum_height,
    summarize_mesh_lower_envelope,
)
from hjlib_ground_solver.estimate_ground import (
    compute_per_frame_mesh_minimum_height as compute_from_subpackage,
)


HEADLINE_COVERAGES = (1.0, 0.999, 0.995, 0.99)


def expect_value_error(operation: Callable[[], object]) -> None:
    try:
        operation()
    except ValueError:
        return
    raise AssertionError('operation must raise ValueError')


def candidate_for(
        summary: Mesh_Lower_Envelope_Summary,
        coverage: float,
    ) -> Mesh_Lower_Envelope_Candidate:
    return next(
        candidate
        for candidate in summary.candidates
        if candidate.retained_coverage == coverage
    )


def full_sort_oracle(values: NDArray[np.floating[Any]], coverage: float) -> float:
    frame_count = int(values.size)
    discarded = frame_count - int(
        (Decimal(str(coverage)) * Decimal(frame_count)).to_integral_value(
            rounding=ROUND_CEILING,
        )
    )
    return float(np.sort(values.astype(np.float64, copy=True))[discarded])


def test_mesh_reducer_contract() -> None:
    vertices = torch.tensor(
        [
            [[0.0, 2.0, 4.0], [1.0, -1.0, 3.0]],
            [[0.0, 3.0, -4.0], [1.0, 2.0, 5.0]],
        ],
        dtype=torch.float64,
    )
    before = vertices.clone()
    result = compute_per_frame_mesh_minimum_height(vertices, 2)
    assert result.dtype == vertices.dtype
    assert result.device == vertices.device
    assert torch.equal(result, torch.tensor([3.0, -4.0], dtype=torch.float64))
    assert torch.equal(vertices, before)
    assert compute_from_subpackage is compute_per_frame_mesh_minimum_height
    autograd_vertices = vertices.clone().requires_grad_(True)
    autograd_result = compute_per_frame_mesh_minimum_height(autograd_vertices, 2)
    assert autograd_result.requires_grad
    autograd_result.sum().backward()
    assert autograd_vertices.grad is not None
    assert float(autograd_vertices.grad[:, :, 2].sum().item()) == 2.0


def test_mesh_reducer_rejects_invalid_input() -> None:
    valid = torch.zeros((2, 3, 3), dtype=torch.float32)
    expect_value_error(lambda: compute_per_frame_mesh_minimum_height(valid[0], 2))
    expect_value_error(
        lambda: compute_per_frame_mesh_minimum_height(torch.zeros((0, 3, 3)), 2)
    )
    expect_value_error(
        lambda: compute_per_frame_mesh_minimum_height(torch.zeros((2, 0, 3)), 2)
    )
    expect_value_error(
        lambda: compute_per_frame_mesh_minimum_height(torch.zeros((2, 3, 2)), 1)
    )
    expect_value_error(
        lambda: compute_per_frame_mesh_minimum_height(
            torch.zeros((2, 3, 3), dtype=torch.int64),
            2,
        )
    )
    expect_value_error(lambda: compute_per_frame_mesh_minimum_height(valid, -1))
    expect_value_error(lambda: compute_per_frame_mesh_minimum_height(valid, 3))
    expect_value_error(lambda: compute_per_frame_mesh_minimum_height(valid, True))
    nonfinite = valid.clone()
    nonfinite[0, 0, 0] = torch.inf
    expect_value_error(lambda: compute_per_frame_mesh_minimum_height(nonfinite, 2))


def test_discrete_coverage_boundaries() -> None:
    expected = {
        1: (0, 0, 0),
        99: (0, 0, 0),
        100: (0, 0, 1),
        199: (0, 0, 1),
        200: (0, 1, 2),
        999: (0, 4, 9),
        1000: (1, 5, 10),
    }
    for frame_count, expected_counts in expected.items():
        values = np.arange(frame_count, dtype=np.float64)
        summary = summarize_mesh_lower_envelope(
            values,
            (0.999, 0.995, 0.99, 1.0),
        )
        actual_counts = tuple(
            candidate.discarded_frame_count
            for candidate in summary.candidates[:3]
        )
        assert actual_counts == expected_counts
        for candidate in summary.candidates:
            if candidate.discarded_frame_count == 0:
                assert candidate.height_in_meter == 0.0
                assert candidate.delta_from_absolute_minimum_in_meter == 0.0


def contamination_series(n_bad: int, spaced: bool) -> NDArray[np.float64]:
    values = np.concatenate(
        (
            np.zeros(200, dtype=np.float64),
            np.linspace(0.05, 0.30, 800, dtype=np.float64),
        )
    )
    if n_bad == 0:
        return values
    if spaced:
        indices = 200 + np.floor(
            np.arange(n_bad, dtype=np.float64) * 800.0 / n_bad
        ).astype(np.int64)
    else:
        indices = 200 + np.arange(n_bad, dtype=np.int64)
    values[indices] = -0.10
    return values


def test_contamination_oracle_and_temporal_run() -> None:
    for n_bad in (0, 1, 5, 10, 20, 50):
        consecutive = contamination_series(n_bad, spaced=False)
        spaced = contamination_series(n_bad, spaced=True)
        summary_consecutive = summarize_mesh_lower_envelope(
            consecutive,
            HEADLINE_COVERAGES,
            100.0,
        )
        summary_spaced = summarize_mesh_lower_envelope(
            spaced,
            HEADLINE_COVERAGES,
            100.0,
        )
        for left, right in zip(
                summary_consecutive.candidates,
                summary_spaced.candidates,
                strict=True,
            ):
            assert left.height_in_meter == right.height_in_meter
            assert left.height_in_meter == full_sort_oracle(
                consecutive,
                left.retained_coverage,
            )
            recovers_support = (
                n_bad <= left.discarded_frame_count < n_bad + 200
            )
            assert (left.height_in_meter == 0.0) is recovers_support
    consecutive_five = summarize_mesh_lower_envelope(
        contamination_series(5, spaced=False),
        HEADLINE_COVERAGES,
        100.0,
    )
    spaced_five = summarize_mesh_lower_envelope(
        contamination_series(5, spaced=True),
        HEADLINE_COVERAGES,
        100.0,
    )
    consecutive_candidate = candidate_for(consecutive_five, 0.995)
    spaced_candidate = candidate_for(spaced_five, 0.995)
    assert consecutive_candidate.longest_below_run_frame_count == 5
    assert spaced_candidate.longest_below_run_frame_count == 1
    assert consecutive_candidate.longest_below_run_duration_in_second == 0.05


def test_ties_order_dtype_and_immutability() -> None:
    values = np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    before = values.copy()
    summary = summarize_mesh_lower_envelope(values, (0.5, 1.0), 20)
    assert np.array_equal(values, before)
    assert tuple(candidate.retained_coverage for candidate in summary.candidates) == (
        0.5,
        1.0,
    )
    half = summary.candidates[0]
    assert half.height_in_meter == 0.0
    assert half.empirical_retained_fraction == 0.75
    assert half.longest_below_run_frame_count == 1
    assert half.retained_coverage_decimal == '0.5'
    binary_exact_32 = np.array([-0.5, 0.0, 0.25, 0.5], dtype=np.float32)
    binary_exact_64 = binary_exact_32.astype(np.float64)
    result_32 = summarize_mesh_lower_envelope(binary_exact_32, (0.5, 1.0))
    result_64 = summarize_mesh_lower_envelope(binary_exact_64, (0.5, 1.0))
    assert result_32 == result_64
    arbitrary_32 = np.array(
        [-0.31, 0.17, -0.07, 0.29, 0.03, -0.11, 0.23],
        dtype=np.float32,
    )
    arbitrary_64 = np.array(
        [-0.31, 0.17, -0.07, 0.29, 0.03, -0.11, 0.23],
        dtype=np.float64,
    )
    for arbitrary in (arbitrary_32, arbitrary_64):
        arbitrary_summary = summarize_mesh_lower_envelope(
            arbitrary,
            (0.6, 1.0),
        )
        assert arbitrary_summary.candidates[0].height_in_meter == full_sort_oracle(
            arbitrary,
            0.6,
        )
        assert arbitrary_summary.candidates[0].longest_below_run_duration_in_second is None
    try:
        setattr(summary, 'frame_count', 7)
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError('summary dataclass must be immutable')
    try:
        setattr(half, 'height_in_meter', 7.0)
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError('candidate dataclass must be immutable')


def test_summary_rejects_invalid_input() -> None:
    valid = np.array([0.0, 1.0], dtype=np.float64)
    invalid_series = (
        np.array([], dtype=np.float64),
        np.zeros((1, 2), dtype=np.float64),
        np.array([0.0, np.nan], dtype=np.float64),
        np.array([0.0, np.inf], dtype=np.float64),
        cast(
            NDArray[np.floating[Any]],
            np.array([0, 1], dtype=np.int64),
        ),
    )
    for values in invalid_series:
        expect_value_error(
            lambda values=values: summarize_mesh_lower_envelope(
                values,
                HEADLINE_COVERAGES,
            )
        )
    invalid_coverages = (
        (),
        (0.99,),
        (1.0, 1.0),
        (1.0, 0.0),
        (1.0, -0.5),
        (1.0, 1.1),
        (1.0, float('nan')),
        cast(tuple[float, ...], (1.0, 1)),
        cast(tuple[float, ...], (1.0, np.float64(0.99))),
    )
    for coverages in invalid_coverages:
        expect_value_error(
            lambda coverages=coverages: summarize_mesh_lower_envelope(
                valid,
                coverages,
            )
        )
    for frame_rate in (0.0, -1.0, float('inf'), float('nan')):
        expect_value_error(
            lambda frame_rate=frame_rate: summarize_mesh_lower_envelope(
                valid,
                HEADLINE_COVERAGES,
                frame_rate,
            )
        )


def test_three_level_public_reexports() -> None:
    symbol_names = (
        'Mesh_Lower_Envelope_Candidate',
        'Mesh_Lower_Envelope_Summary',
        'compute_per_frame_mesh_minimum_height',
        'summarize_mesh_lower_envelope',
    )
    for symbol_name in symbol_names:
        implementation_symbol = getattr(implementation_module, symbol_name)
        assert getattr(estimate_subpackage, symbol_name) is implementation_symbol
        assert getattr(package_root, symbol_name) is implementation_symbol


def test_chunk_boundary_invariance() -> None:
    frame_count = 37
    vertices = torch.arange(
        frame_count * 5 * 3,
        dtype=torch.float64,
    ).reshape(frame_count, 5, 3)
    vertices[:, :, 2] = vertices[:, :, 2] * 0.001 - 0.5
    expected = compute_per_frame_mesh_minimum_height(vertices, 2)
    for chunk_size in (1, 7, frame_count):
        chunks = tuple(
            compute_per_frame_mesh_minimum_height(
                vertices[start:start + chunk_size],
                2,
            )
            for start in range(0, frame_count, chunk_size)
        )
        actual = torch.cat(chunks)
        assert torch.equal(actual, expected)
        actual_summary = summarize_mesh_lower_envelope(
            actual.numpy(),
            HEADLINE_COVERAGES,
        )
        expected_summary = summarize_mesh_lower_envelope(
            expected.numpy(),
            HEADLINE_COVERAGES,
        )
        assert actual_summary == expected_summary


def smoke_test_mesh_lower_envelope() -> None:
    test_mesh_reducer_contract()
    test_mesh_reducer_rejects_invalid_input()
    test_discrete_coverage_boundaries()
    test_contamination_oracle_and_temporal_run()
    test_ties_order_dtype_and_immutability()
    test_summary_rejects_invalid_input()
    test_three_level_public_reexports()
    test_chunk_boundary_invariance()
    print('[OK] mesh_lower_envelope')


if __name__ == '__main__':
    smoke_test_mesh_lower_envelope()
