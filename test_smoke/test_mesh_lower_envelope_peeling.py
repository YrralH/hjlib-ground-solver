'''Synthetic smoke for iterative separated mesh lower-envelope peeling.'''

from dataclasses import FrozenInstanceError
from decimal import Decimal, ROUND_FLOOR
import math
import sys
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

import hjlib_ground_solver as package_root
import hjlib_ground_solver.estimate_ground as estimate_subpackage
import hjlib_ground_solver.estimate_ground.by_mesh_lower_envelope_peeling as implementation_module
from hjlib_ground_solver import (
    Mesh_Lower_Envelope_Peeling_Config,
    peel_separated_mesh_lower_envelope_prefixes,
)


def peeling_config(
        maximum_round_count: int = 3,
        maximum_total_removed_fraction_decimal: str = '0.8',
        maximum_total_removed_frame_count: int = 20,
        maximum_candidate_fraction_per_round_decimal: str = '0.5',
        maximum_candidate_frame_count_per_round: int = 10,
        minimum_retained_frame_count: int = 1,
        reference_gap_window_size: int = 2,
        minimum_boundary_gap_in_meter: float = 0.5,
        minimum_gap_ratio: float = 2.0,
    ) -> Mesh_Lower_Envelope_Peeling_Config:
    return Mesh_Lower_Envelope_Peeling_Config(
        maximum_candidate_fraction_per_round_decimal=(
            maximum_candidate_fraction_per_round_decimal
        ),
        maximum_candidate_frame_count_per_round=(
            maximum_candidate_frame_count_per_round
        ),
        minimum_retained_frame_count=minimum_retained_frame_count,
        reference_gap_window_size=reference_gap_window_size,
        minimum_boundary_gap_in_meter=minimum_boundary_gap_in_meter,
        minimum_gap_ratio=minimum_gap_ratio,
        maximum_round_count=maximum_round_count,
        maximum_total_removed_fraction_decimal=(
            maximum_total_removed_fraction_decimal
        ),
        maximum_total_removed_frame_count=maximum_total_removed_frame_count,
    )


def expect_value_error(function: Any) -> None:
    try:
        function()
    except ValueError:
        return
    raise AssertionError('expected ValueError')


def test_isolated_tied_and_two_round_collective_peels() -> None:
    isolated = peel_separated_mesh_lower_envelope_prefixes(
        np.array([-10.0, 0.0, 1.0, 2.0, 3.0]),
        peeling_config(),
        20.0,
    )
    assert isolated.status == 'stable_candidate'
    assert isolated.accepted_candidate_height_in_meter == 0.0
    assert isolated.applied_removed_frame_count == 1
    assert isolated.applied_peels[0].removed_original_frame_indices == (0,)
    assert isolated.applied_peels[0].longest_removed_run_duration_in_second == 0.05

    tied = peel_separated_mesh_lower_envelope_prefixes(
        np.array([-10.0, -10.0, 0.0, 1.0, 2.0, 3.0]),
        peeling_config(),
    )
    assert tied.applied_removed_frame_count == 2
    assert tied.applied_peels[0].removed_original_frame_indices == (0, 1)
    assert tied.current_candidate_height_in_meter == 0.0

    two_groups = peel_separated_mesh_lower_envelope_prefixes(
        np.array([0.0, 10.0, 14.0, 14.5, 15.0]),
        peeling_config(reference_gap_window_size=1, minimum_boundary_gap_in_meter=0.4),
    )
    assert two_groups.status == 'stable_candidate'
    assert tuple(peel.removed_frame_count for peel in two_groups.applied_peels) == (1, 1)
    assert tuple(
        peel.candidate_after_in_meter for peel in two_groups.applied_peels
    ) == (10.0, 14.0)


def test_round_and_removal_budget_stop_boundaries() -> None:
    values = np.array([0.0, 10.0, 14.0, 14.5, 15.0])
    stable_at_exact_rounds = peel_separated_mesh_lower_envelope_prefixes(
        values,
        peeling_config(
            maximum_round_count=2,
            reference_gap_window_size=1,
            minimum_boundary_gap_in_meter=0.4,
        ),
    )
    assert stable_at_exact_rounds.status == 'stable_candidate'
    assert len(stable_at_exact_rounds.applied_peels) == 2

    blocked_round = peel_separated_mesh_lower_envelope_prefixes(
        values,
        peeling_config(
            maximum_round_count=1,
            reference_gap_window_size=1,
            minimum_boundary_gap_in_meter=0.4,
        ),
    )
    assert blocked_round.status == 'unstable_maximum_round_count'
    assert blocked_round.accepted_candidate_height_in_meter is None
    assert blocked_round.current_candidate_height_in_meter == 10.0
    assert len(blocked_round.applied_peels) == 1
    assert blocked_round.blocked_peel is not None
    assert blocked_round.blocked_peel.round_index == 2

    exact_budget = peel_separated_mesh_lower_envelope_prefixes(
        np.array([-10.0, 0.0, 1.0, 2.0, 3.0]),
        peeling_config(
            maximum_total_removed_fraction_decimal='0.2',
            maximum_total_removed_frame_count=1,
        ),
    )
    assert exact_budget.status == 'stable_candidate'
    assert exact_budget.applied_removed_frame_count == 1
    assert exact_budget.effective_maximum_total_removed_frame_count == 1

    blocked_budget = peel_separated_mesh_lower_envelope_prefixes(
        np.array([-10.0, 0.0, 1.0, 2.0, 3.0]),
        peeling_config(
            maximum_total_removed_fraction_decimal='0.1',
            maximum_total_removed_frame_count=1,
        ),
    )
    assert blocked_budget.status == 'unstable_removal_budget'
    assert blocked_budget.applied_removed_frame_count == 0
    assert blocked_budget.current_candidate_height_in_meter == -10.0
    assert blocked_budget.blocked_peel is not None

    round_priority = peel_separated_mesh_lower_envelope_prefixes(
        np.array([-10.0, 0.0, 1.0, 2.0, 3.0]),
        peeling_config(
            maximum_round_count=0,
            maximum_total_removed_fraction_decimal='0.1',
            maximum_total_removed_frame_count=1,
        ),
    )
    assert round_priority.status == 'unstable_maximum_round_count'
    assert round_priority.applied_peels == ()


def test_fixed_reference_slots_ties_and_small_gaps() -> None:
    plateau = peel_separated_mesh_lower_envelope_prefixes(
        np.array([0.0, 10.0, 10.0, 10.0, 11.0]),
        peeling_config(reference_gap_window_size=2),
    )
    first = plateau.applied_peels[0]
    assert first.reference_gap_slot_count == 2
    assert first.reference_positive_gap_count == 0
    assert first.reference_median_gap_in_meter is None
    assert first.gap_ratio is None

    uniform = peel_separated_mesh_lower_envelope_prefixes(
        np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        peeling_config(),
    )
    assert uniform.status == 'stable_candidate'
    assert uniform.applied_peels == ()
    tiny = peel_separated_mesh_lower_envelope_prefixes(
        np.array([0.0, 0.001, 1.0, 2.0, 3.0]),
        peeling_config(minimum_boundary_gap_in_meter=0.01),
    )
    assert tiny.applied_peels == ()


def test_exact_decimal_floor_is_context_independent() -> None:
    near_tenth = '0.099999999999999999999999999999999999'
    round_floor_zero = peel_separated_mesh_lower_envelope_prefixes(
        np.array([-10.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        peeling_config(
            maximum_candidate_fraction_per_round_decimal=near_tenth,
        ),
    )
    assert round_floor_zero.status == 'stable_candidate'
    assert round_floor_zero.applied_peels == ()

    total_floor_zero = peel_separated_mesh_lower_envelope_prefixes(
        np.array([-10.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        peeling_config(
            maximum_total_removed_fraction_decimal=near_tenth,
        ),
    )
    assert total_floor_zero.status == 'unstable_removal_budget'
    assert total_floor_zero.maximum_total_removed_frame_count_by_fraction == 0
    assert total_floor_zero.applied_removed_frame_count == 0

    exact_fifty_eight = peel_separated_mesh_lower_envelope_prefixes(
        np.concatenate((np.zeros(58), np.full(42, 10.0))),
        peeling_config(
            maximum_candidate_fraction_per_round_decimal='0.58',
            maximum_candidate_frame_count_per_round=100,
            minimum_retained_frame_count=1,
            maximum_round_count=1,
            maximum_total_removed_fraction_decimal='0.8',
            maximum_total_removed_frame_count=100,
        ),
    )
    assert exact_fifty_eight.status == 'stable_candidate'
    assert exact_fifty_eight.applied_removed_frame_count == 58

    if hasattr(sys, 'set_int_max_str_digits'):
        original_limit = sys.get_int_max_str_digits()
        try:
            sys.set_int_max_str_digits(640)
            long_fraction = Decimal('0.' + '9' * 700)
            assert implementation_module.floor_exact_decimal_count(
                long_fraction,
                10,
            ) == 9
        finally:
            sys.set_int_max_str_digits(original_limit)


def test_time_permutation_changes_only_run_diagnostic() -> None:
    consecutive = np.array([-10.0, -10.0, 0.0, 1.0, 2.0, 3.0])
    spaced = np.array([-10.0, 0.0, 1.0, -10.0, 2.0, 3.0])
    left = peel_separated_mesh_lower_envelope_prefixes(
        consecutive,
        peeling_config(),
    )
    right = peel_separated_mesh_lower_envelope_prefixes(
        spaced,
        peeling_config(),
    )
    assert left.status == right.status
    assert left.current_candidate_height_in_meter == right.current_candidate_height_in_meter
    assert tuple(peel.removed_frame_count for peel in left.applied_peels) == tuple(
        peel.removed_frame_count for peel in right.applied_peels
    )
    assert left.applied_peels[0].longest_removed_run_frame_count == 2
    assert right.applied_peels[0].longest_removed_run_frame_count == 1


def reference_result(
        values: NDArray[np.float64],
        config: Mesh_Lower_Envelope_Peeling_Config,
    ) -> tuple[str, float, tuple[tuple[float, ...], ...]]:
    retained = tuple((float(value), index) for index, value in enumerate(values))
    applied: list[tuple[float, ...]] = []
    total_count = len(retained)
    total_budget = min(
        int(
            (
                Decimal(config.maximum_total_removed_fraction_decimal)
                * Decimal(total_count)
            ).to_integral_value(rounding=ROUND_FLOOR)
        ),
        config.maximum_total_removed_frame_count,
    )
    while True:
        ordered = tuple(sorted(retained))
        retained_count = len(ordered)
        search_limit = min(
            int(
                (
                    Decimal(config.maximum_candidate_fraction_per_round_decimal)
                    * Decimal(retained_count)
                ).to_integral_value(rounding=ROUND_FLOOR)
            ),
            config.maximum_candidate_frame_count_per_round,
            retained_count - config.minimum_retained_frame_count,
        )
        selected_count: int | None = None
        for removed_count in range(1, search_limit + 1):
            boundary_gap = ordered[removed_count][0] - ordered[removed_count - 1][0]
            if boundary_gap < config.minimum_boundary_gap_in_meter:
                continue
            reference_stop = min(
                removed_count + config.reference_gap_window_size,
                retained_count - 1,
            )
            reference = tuple(
                ordered[index + 1][0] - ordered[index][0]
                for index in range(removed_count, reference_stop)
                if ordered[index + 1][0] > ordered[index][0]
            )
            if reference:
                sorted_reference = sorted(reference)
                middle = len(sorted_reference) // 2
                if len(sorted_reference) % 2:
                    median = sorted_reference[middle]
                else:
                    median = sorted_reference[middle - 1] + (
                        sorted_reference[middle] - sorted_reference[middle - 1]
                    ) / 2.0
                if boundary_gap / median < config.minimum_gap_ratio:
                    continue
            selected_count = removed_count
            break
        if selected_count is None:
            return 'stable_candidate', ordered[0][0], tuple(applied)
        proposed_total = total_count - retained_count + selected_count
        if len(applied) == config.maximum_round_count:
            return 'unstable_maximum_round_count', ordered[0][0], tuple(applied)
        if proposed_total > total_budget:
            return 'unstable_removal_budget', ordered[0][0], tuple(applied)
        applied.append(tuple(item[0] for item in ordered[:selected_count]))
        removed_indices = {item[1] for item in ordered[:selected_count]}
        retained = tuple(item for item in retained if item[1] not in removed_indices)


def test_deterministic_randomized_repeated_sort_oracle() -> None:
    generator = np.random.default_rng(20260803)
    config = peeling_config(
        maximum_candidate_fraction_per_round_decimal='0.3',
        maximum_candidate_frame_count_per_round=6,
        minimum_retained_frame_count=3,
        reference_gap_window_size=4,
        minimum_boundary_gap_in_meter=0.2,
        minimum_gap_ratio=2.5,
        maximum_total_removed_fraction_decimal='0.4',
        maximum_total_removed_frame_count=8,
    )
    for _ in range(100):
        values = np.round(generator.normal(size=24), 2).astype(np.float64)
        expected_status, expected_candidate, expected_groups = reference_result(
            values,
            config,
        )
        result = peel_separated_mesh_lower_envelope_prefixes(values, config)
        assert result.status == expected_status
        assert result.current_candidate_height_in_meter == expected_candidate
        actual_groups = tuple(
            tuple(
                sorted(float(values[index]) for index in peel.removed_original_frame_indices)
            )
            for peel in result.applied_peels
        )
        assert actual_groups == expected_groups


def test_validation_overflow_immutability_and_reexports() -> None:
    valid = np.array([0.0, 1.0, 2.0])
    before = valid.copy()
    result = peel_separated_mesh_lower_envelope_prefixes(valid, peeling_config())
    assert np.array_equal(valid, before)
    try:
        setattr(result, 'frame_count', 9)
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError('result must be frozen')
    frozen_config = peeling_config()
    frozen_proposal = peel_separated_mesh_lower_envelope_prefixes(
        np.array([-10.0, 0.0, 1.0]),
        frozen_config,
    ).applied_peels[0]
    for frozen_object, attribute, value in (
            (frozen_config, 'maximum_round_count', 9),
            (frozen_proposal, 'removed_frame_count', 9),
        ):
        try:
            setattr(frozen_object, attribute, value)
        except FrozenInstanceError:
            pass
        else:
            raise AssertionError('config and proposal must be frozen')

    for invalid in (
            np.array([], dtype=np.float64),
            np.zeros((2, 2), dtype=np.float64),
            np.array([0.0, np.nan]),
            np.array([0.0, np.inf]),
            cast(NDArray[np.floating[Any]], np.array([0, 1])),
        ):
        expect_value_error(
            lambda invalid=invalid: peel_separated_mesh_lower_envelope_prefixes(
                invalid,
                peeling_config(),
            )
        )
    expect_value_error(
        lambda: peel_separated_mesh_lower_envelope_prefixes(
            valid,
            peeling_config(minimum_retained_frame_count=4),
        )
    )
    for frame_rate in (True, np.float64(30.0), 0.0, -1.0, math.inf, math.nan):
        expect_value_error(
            lambda frame_rate=frame_rate: peel_separated_mesh_lower_envelope_prefixes(
                valid,
                peeling_config(),
                cast(Any, frame_rate),
            )
        )
    expect_value_error(
        lambda: peel_separated_mesh_lower_envelope_prefixes(
            np.array([-10.0, 0.0, 1.0]),
            peeling_config(),
            float.fromhex('0x0.0000000000001p-1022'),
        )
    )
    expect_value_error(
        lambda: peel_separated_mesh_lower_envelope_prefixes(
            np.array([-1.7e308, 1.7e308]),
            peeling_config(
                maximum_candidate_fraction_per_round_decimal='1',
                minimum_retained_frame_count=1,
            ),
        )
    )
    expect_value_error(
        lambda: peel_separated_mesh_lower_envelope_prefixes(
            np.array([-1.0e308, 0.0, 5.0e-324, 1.0e-323]),
            peeling_config(
                maximum_candidate_fraction_per_round_decimal='0.5',
                reference_gap_window_size=1,
            ),
        )
    )
    large_median = peel_separated_mesh_lower_envelope_prefixes(
        np.array([-1.5e308, -0.5e308, 0.5e308, 1.5e308]),
        peeling_config(
            maximum_candidate_fraction_per_round_decimal='0.25',
            reference_gap_window_size=2,
            minimum_boundary_gap_in_meter=1.0,
        ),
    )
    assert large_median.status == 'stable_candidate'
    median = implementation_module.compute_overflow_safe_positive_median(
        np.array([1.0e308, 1.0e308]),
    )
    assert median == 1.0e308
    assert math.isfinite(median)

    invalid_configs = (
        dict(maximum_candidate_fraction_per_round_decimal='0'),
        dict(maximum_candidate_fraction_per_round_decimal='1.1'),
        dict(maximum_candidate_fraction_per_round_decimal=' 0.5'),
        dict(maximum_candidate_frame_count_per_round=0),
        dict(minimum_retained_frame_count=True),
        dict(reference_gap_window_size=0),
        dict(minimum_boundary_gap_in_meter=0.0),
        dict(minimum_boundary_gap_in_meter=cast(Any, 1)),
        dict(minimum_gap_ratio=1.0),
        dict(maximum_round_count=-1),
        dict(maximum_total_removed_fraction_decimal='1'),
        dict(maximum_total_removed_frame_count=0),
    )
    for overrides in invalid_configs:
        expect_value_error(
            lambda overrides=overrides: peeling_config(**cast(Any, overrides))
        )

    symbol_names = (
        'Mesh_Lower_Envelope_Peel_Proposal',
        'Mesh_Lower_Envelope_Peeling_Config',
        'Mesh_Lower_Envelope_Peeling_Result',
        'Mesh_Lower_Envelope_Peeling_Status',
        'peel_separated_mesh_lower_envelope_prefixes',
    )
    for symbol_name in symbol_names:
        symbol = getattr(implementation_module, symbol_name)
        assert getattr(estimate_subpackage, symbol_name) is symbol
        assert getattr(package_root, symbol_name) is symbol


def smoke_test_mesh_lower_envelope_peeling() -> None:
    test_isolated_tied_and_two_round_collective_peels()
    test_round_and_removal_budget_stop_boundaries()
    test_fixed_reference_slots_ties_and_small_gaps()
    test_exact_decimal_floor_is_context_independent()
    test_time_permutation_changes_only_run_diagnostic()
    test_deterministic_randomized_repeated_sort_oracle()
    test_validation_overflow_immutability_and_reexports()


if __name__ == '__main__':
    smoke_test_mesh_lower_envelope_peeling()
    print('mesh lower-envelope peeling smoke OK')
