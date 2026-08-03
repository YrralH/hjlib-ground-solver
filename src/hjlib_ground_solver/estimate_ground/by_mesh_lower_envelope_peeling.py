'''Iterative separated low-prefix diagnostics for mesh lower envelopes.'''

from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
import math
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray


type Mesh_Lower_Envelope_Peeling_Status = Literal[
    'stable_candidate',
    'unstable_maximum_round_count',
    'unstable_removal_budget',
]


@dataclass(frozen=True, slots=True)
class Mesh_Lower_Envelope_Peeling_Config:
    '''Explicit search, evidence, iteration, and removal bounds.'''

    maximum_candidate_fraction_per_round_decimal: str
    maximum_candidate_frame_count_per_round: int
    minimum_retained_frame_count: int
    reference_gap_window_size: int
    minimum_boundary_gap_in_meter: float
    minimum_gap_ratio: float
    maximum_round_count: int
    maximum_total_removed_fraction_decimal: str
    maximum_total_removed_frame_count: int

    def __post_init__(self) -> None:
        round_fraction = parse_peeling_fraction_decimal(
            self.maximum_candidate_fraction_per_round_decimal,
            'maximum_candidate_fraction_per_round_decimal',
        )
        if round_fraction <= 0 or round_fraction > 1:
            raise ValueError(
                'maximum candidate fraction per round must be within (0, 1]'
            )
        total_fraction = parse_peeling_fraction_decimal(
            self.maximum_total_removed_fraction_decimal,
            'maximum_total_removed_fraction_decimal',
        )
        if total_fraction <= 0 or total_fraction >= 1:
            raise ValueError(
                'maximum total removed fraction must be within (0, 1)'
            )
        positive_integer_fields = (
            (
                self.maximum_candidate_frame_count_per_round,
                'maximum_candidate_frame_count_per_round',
            ),
            (self.minimum_retained_frame_count, 'minimum_retained_frame_count'),
            (self.reference_gap_window_size, 'reference_gap_window_size'),
            (
                self.maximum_total_removed_frame_count,
                'maximum_total_removed_frame_count',
            ),
        )
        for value, label in positive_integer_fields:
            if type(value) is not int or value <= 0:
                raise ValueError('%s must be a positive Python int' % label)
        if type(self.maximum_round_count) is not int or self.maximum_round_count < 0:
            raise ValueError('maximum_round_count must be a nonnegative Python int')
        if (
                type(self.minimum_boundary_gap_in_meter) is not float
                or not math.isfinite(self.minimum_boundary_gap_in_meter)
                or self.minimum_boundary_gap_in_meter <= 0.0
            ):
            raise ValueError(
                'minimum_boundary_gap_in_meter must be a finite positive Python float'
            )
        if (
                type(self.minimum_gap_ratio) is not float
                or not math.isfinite(self.minimum_gap_ratio)
                or self.minimum_gap_ratio <= 1.0
            ):
            raise ValueError('minimum_gap_ratio must be a finite Python float above 1')


@dataclass(frozen=True, slots=True)
class Mesh_Lower_Envelope_Peel_Proposal:
    '''Complete evidence for one applied or blocked low-prefix peel.'''

    round_index: int
    retained_frame_count_before: int
    removed_frame_count: int
    cumulative_removed_frame_count: int
    removed_original_frame_indices: tuple[int, ...]
    removed_minimum_height_in_meter: float
    removed_maximum_height_in_meter: float
    boundary_gap_in_meter: float
    reference_gap_slot_count: int
    reference_positive_gap_count: int
    reference_median_gap_in_meter: float | None
    gap_ratio: float | None
    candidate_before_in_meter: float
    candidate_after_in_meter: float
    longest_removed_run_frame_count: int
    longest_removed_run_duration_in_second: float | None


@dataclass(frozen=True, slots=True)
class Mesh_Lower_Envelope_Peeling_Result:
    '''Immutable result whose accepted candidate exists only when stable.'''

    status: Mesh_Lower_Envelope_Peeling_Status
    config: Mesh_Lower_Envelope_Peeling_Config
    frame_count: int
    frame_rate_in_hz: float | None
    absolute_minimum_height_in_meter: float
    current_candidate_height_in_meter: float
    accepted_candidate_height_in_meter: float | None
    applied_removed_frame_count: int
    retained_frame_count: int
    maximum_total_removed_frame_count_by_fraction: int
    effective_maximum_total_removed_frame_count: int
    applied_peels: tuple[Mesh_Lower_Envelope_Peel_Proposal, ...]
    blocked_peel: Mesh_Lower_Envelope_Peel_Proposal | None


def parse_peeling_fraction_decimal(value: str, label: str) -> Decimal:
    '''Parse one whitespace-free finite decimal configuration field.'''
    if type(value) is not str or value.strip() != value:
        raise ValueError('%s must be a whitespace-free decimal string' % label)
    try:
        decimal_value = Decimal(value)
    except InvalidOperation as error:
        raise ValueError('%s must be a valid decimal string' % label) from error
    if not decimal_value.is_finite():
        raise ValueError('%s must be finite' % label)
    return decimal_value


def floor_exact_decimal_count(fraction: Decimal, count: int) -> int:
    '''Compute floor(fraction*count) by integer arithmetic, without context rounding.'''
    decimal_tuple = fraction.as_tuple()
    coefficient = 0
    for digit in decimal_tuple.digits:
        coefficient = coefficient * 10 + digit
    numerator = coefficient * count
    exponent = int(decimal_tuple.exponent)
    if exponent >= 0:
        return numerator * (10 ** exponent)
    scale = -exponent
    count_decimal_digit_count = 1
    remaining_count = count
    while remaining_count >= 10:
        remaining_count //= 10
        count_decimal_digit_count += 1
    if (
            numerator == 0
            or scale >= len(decimal_tuple.digits) + count_decimal_digit_count
        ):
        return 0
    return numerator // (10 ** scale)


def normalize_mesh_lower_envelope_peeling_input(
        per_frame_minimum_height_in_meter: NDArray[np.floating[Any]],
        config: Mesh_Lower_Envelope_Peeling_Config,
        frame_rate_in_hz: float | None,
    ) -> tuple[NDArray[np.float64], float | None, int, int]:
    '''Validate and copy input without sorting it.'''
    if type(config) is not Mesh_Lower_Envelope_Peeling_Config:
        raise ValueError('config must be Mesh_Lower_Envelope_Peeling_Config')
    if type(per_frame_minimum_height_in_meter) is not np.ndarray:
        raise ValueError('per-frame minima must be a numpy array')
    series = per_frame_minimum_height_in_meter
    if series.ndim != 1 or series.size == 0:
        raise ValueError('per-frame minima must have nonempty shape (T,)')
    if not np.issubdtype(series.dtype, np.floating):
        raise ValueError('per-frame minima must have a floating dtype')
    if not bool(np.isfinite(series).all()):
        raise ValueError('per-frame minima must contain only finite values')
    values = np.array(series, dtype=np.float64, copy=True)
    if not bool(np.isfinite(values).all()):
        raise ValueError('float64 conversion must preserve finite values')
    frame_count = int(values.size)
    if config.minimum_retained_frame_count > frame_count:
        raise ValueError('minimum retained frame count must not exceed T')

    normalized_frame_rate: float | None = None
    if frame_rate_in_hz is not None:
        if type(frame_rate_in_hz) not in (int, float):
            raise ValueError('frame_rate_in_hz must be a Python int or float')
        normalized_frame_rate = float(frame_rate_in_hz)
        if not math.isfinite(normalized_frame_rate) or normalized_frame_rate <= 0.0:
            raise ValueError('frame_rate_in_hz must be finite and positive')

    fraction = parse_peeling_fraction_decimal(
        config.maximum_total_removed_fraction_decimal,
        'maximum_total_removed_fraction_decimal',
    )
    maximum_by_fraction = floor_exact_decimal_count(fraction, frame_count)
    effective_maximum = min(
        maximum_by_fraction,
        config.maximum_total_removed_frame_count,
    )
    return values, normalized_frame_rate, maximum_by_fraction, effective_maximum


def compute_overflow_safe_positive_median(
        values: NDArray[np.float64],
    ) -> float:
    '''Return a finite median for a nonempty positive finite array.'''
    ordered = np.sort(values)
    count = int(ordered.size)
    middle = count // 2
    if count % 2 == 1:
        median = float(ordered[middle])
    else:
        lower = float(ordered[middle - 1])
        upper = float(ordered[middle])
        median = lower + (upper - lower) / 2.0
    if not math.isfinite(median) or median <= 0.0:
        raise ValueError('reference median gap must remain finite and positive')
    return median


def compute_longest_consecutive_frame_run(
        ordered_frame_indices: tuple[int, ...],
    ) -> int:
    '''Compute the longest consecutive run in an increasing index tuple.'''
    longest_run = 0
    current_run = 0
    previous_index: int | None = None
    for frame_index in ordered_frame_indices:
        if previous_index is not None and frame_index == previous_index + 1:
            current_run += 1
        else:
            current_run = 1
        longest_run = max(longest_run, current_run)
        previous_index = frame_index
    return longest_run


def find_first_mesh_lower_envelope_peel_proposal(
        sorted_height_in_meter: NDArray[np.float64],
        sorted_original_frame_index: NDArray[np.int64],
        adjacent_gap_in_meter: NDArray[np.float64],
        retained_start_index: int,
        config: Mesh_Lower_Envelope_Peeling_Config,
        frame_rate_in_hz: float | None,
    ) -> Mesh_Lower_Envelope_Peel_Proposal | None:
    '''Find the smallest eligible separated prefix at the retained start.'''
    frame_count = int(sorted_height_in_meter.size)
    retained_count = frame_count - retained_start_index
    round_fraction = parse_peeling_fraction_decimal(
        config.maximum_candidate_fraction_per_round_decimal,
        'maximum_candidate_fraction_per_round_decimal',
    )
    maximum_by_fraction = floor_exact_decimal_count(
        round_fraction,
        retained_count,
    )
    search_limit = min(
        maximum_by_fraction,
        config.maximum_candidate_frame_count_per_round,
        retained_count - config.minimum_retained_frame_count,
    )
    for removed_count in range(1, search_limit + 1):
        gap_index = retained_start_index + removed_count - 1
        boundary_gap = float(adjacent_gap_in_meter[gap_index])
        if (
                boundary_gap <= 0.0
                or boundary_gap < config.minimum_boundary_gap_in_meter
            ):
            continue
        reference_start = gap_index + 1
        reference_stop = min(
            reference_start + config.reference_gap_window_size,
            int(adjacent_gap_in_meter.size),
        )
        reference_slots = adjacent_gap_in_meter[reference_start:reference_stop]
        positive_reference = reference_slots[reference_slots > 0.0]
        reference_median: float | None = None
        gap_ratio: float | None = None
        if positive_reference.size > 0:
            reference_median = compute_overflow_safe_positive_median(
                positive_reference,
            )
            with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
                gap_ratio = float(np.divide(boundary_gap, reference_median))
            if not math.isfinite(gap_ratio):
                raise ValueError('boundary/reference gap ratio must remain finite')
            if gap_ratio < config.minimum_gap_ratio:
                continue

        cumulative_removed_count = retained_start_index + removed_count
        removed_original_indices = tuple(sorted(
            int(index)
            for index in sorted_original_frame_index[
                retained_start_index:cumulative_removed_count
            ].tolist()
        ))
        longest_run = compute_longest_consecutive_frame_run(
            removed_original_indices,
        )
        longest_duration: float | None = None
        if frame_rate_in_hz is not None:
            longest_duration = longest_run / frame_rate_in_hz
            if not math.isfinite(longest_duration):
                raise ValueError('longest removed-run duration must remain finite')
        return Mesh_Lower_Envelope_Peel_Proposal(
            round_index=0,
            retained_frame_count_before=retained_count,
            removed_frame_count=removed_count,
            cumulative_removed_frame_count=cumulative_removed_count,
            removed_original_frame_indices=removed_original_indices,
            removed_minimum_height_in_meter=float(
                sorted_height_in_meter[retained_start_index]
            ),
            removed_maximum_height_in_meter=float(
                sorted_height_in_meter[cumulative_removed_count - 1]
            ),
            boundary_gap_in_meter=boundary_gap,
            reference_gap_slot_count=int(reference_slots.size),
            reference_positive_gap_count=int(positive_reference.size),
            reference_median_gap_in_meter=reference_median,
            gap_ratio=gap_ratio,
            candidate_before_in_meter=float(
                sorted_height_in_meter[retained_start_index]
            ),
            candidate_after_in_meter=float(
                sorted_height_in_meter[cumulative_removed_count]
            ),
            longest_removed_run_frame_count=longest_run,
            longest_removed_run_duration_in_second=longest_duration,
        )
    return None


def peel_separated_mesh_lower_envelope_prefixes(
        per_frame_minimum_height_in_meter: NDArray[np.floating[Any]],
        config: Mesh_Lower_Envelope_Peeling_Config,
        frame_rate_in_hz: float | None = None,
    ) -> Mesh_Lower_Envelope_Peeling_Result:
    '''Iteratively peel the smallest eligible separated lower-envelope prefix.'''
    (
        values,
        normalized_frame_rate,
        maximum_by_fraction,
        effective_maximum,
    ) = normalize_mesh_lower_envelope_peeling_input(
        per_frame_minimum_height_in_meter,
        config,
        frame_rate_in_hz,
    )
    frame_count = int(values.size)
    original_indices = np.arange(frame_count, dtype=np.int64)
    order = np.lexsort((original_indices, values))
    sorted_values = values[order]
    sorted_original_indices = original_indices[order]
    with np.errstate(over='ignore', invalid='ignore'):
        adjacent_gaps = np.diff(sorted_values)
    if not bool(np.isfinite(adjacent_gaps).all()):
        raise ValueError('adjacent float64 gaps must remain finite')

    retained_start = 0
    applied_peels: list[Mesh_Lower_Envelope_Peel_Proposal] = []
    blocked_peel: Mesh_Lower_Envelope_Peel_Proposal | None = None
    status: Mesh_Lower_Envelope_Peeling_Status
    while True:
        proposal = find_first_mesh_lower_envelope_peel_proposal(
            sorted_values,
            sorted_original_indices,
            adjacent_gaps,
            retained_start,
            config,
            normalized_frame_rate,
        )
        if proposal is None:
            status = 'stable_candidate'
            break
        proposal = replace(proposal, round_index=len(applied_peels) + 1)
        if len(applied_peels) == config.maximum_round_count:
            status = 'unstable_maximum_round_count'
            blocked_peel = proposal
            break
        if proposal.cumulative_removed_frame_count > effective_maximum:
            status = 'unstable_removal_budget'
            blocked_peel = proposal
            break
        applied_peels.append(proposal)
        retained_start = proposal.cumulative_removed_frame_count

    current_candidate = float(sorted_values[retained_start])
    accepted_candidate = (
        current_candidate
        if status == 'stable_candidate'
        else None
    )
    return Mesh_Lower_Envelope_Peeling_Result(
        status=status,
        config=config,
        frame_count=frame_count,
        frame_rate_in_hz=normalized_frame_rate,
        absolute_minimum_height_in_meter=float(sorted_values[0]),
        current_candidate_height_in_meter=current_candidate,
        accepted_candidate_height_in_meter=accepted_candidate,
        applied_removed_frame_count=retained_start,
        retained_frame_count=frame_count - retained_start,
        maximum_total_removed_frame_count_by_fraction=maximum_by_fraction,
        effective_maximum_total_removed_frame_count=effective_maximum,
        applied_peels=tuple(applied_peels),
        blocked_peel=blocked_peel,
    )


__all__ = [
    'Mesh_Lower_Envelope_Peel_Proposal',
    'Mesh_Lower_Envelope_Peeling_Config',
    'Mesh_Lower_Envelope_Peeling_Result',
    'Mesh_Lower_Envelope_Peeling_Status',
    'peel_separated_mesh_lower_envelope_prefixes',
]
