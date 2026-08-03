'''Full-mesh lower-envelope statistics without semantic ground claims.'''

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray
import torch


@dataclass(frozen=True, slots=True)
class Mesh_Lower_Envelope_Candidate:
    '''One exact retained-coverage order statistic of per-frame mesh minima.'''

    retained_coverage: float
    retained_coverage_decimal: str
    discarded_frame_count: int
    height_in_meter: float
    delta_from_absolute_minimum_in_meter: float
    empirical_retained_fraction: float
    longest_below_run_frame_count: int
    longest_below_run_duration_in_second: float | None


@dataclass(frozen=True, slots=True)
class Mesh_Lower_Envelope_Summary:
    '''Immutable summary of a finite per-frame mesh lower envelope.'''

    frame_count: int
    frame_rate_in_hz: float | None
    absolute_minimum_height_in_meter: float
    candidates: tuple[Mesh_Lower_Envelope_Candidate, ...]


def compute_per_frame_mesh_minimum_height(
        vertices: torch.Tensor,
        up_axis_index: int,
    ) -> torch.Tensor:
    '''Reduce a finite ``(B,V,3)`` floating mesh chunk to ``(B,)`` minima.'''
    if vertices.ndim != 3 or vertices.shape[0] <= 0 or vertices.shape[1] <= 0:
        raise ValueError('vertices must have nonempty shape (B, V, 3)')
    if vertices.shape[2] != 3:
        raise ValueError('vertices must have final coordinate dimension 3')
    if not torch.is_floating_point(vertices):
        raise ValueError('vertices must have a floating dtype')
    if type(up_axis_index) is not int or up_axis_index not in (0, 1, 2):
        raise ValueError('up_axis_index must be one of 0, 1, or 2')
    finite_by_frame = torch.isfinite(vertices).all(dim=(1, 2))
    if not bool(finite_by_frame.all().item()):
        raise ValueError('vertices must contain only finite coordinates')
    return torch.amin(vertices[:, :, up_axis_index], dim=1)


def summarize_mesh_lower_envelope(
        per_frame_minimum_height_in_meter: NDArray[np.floating[Any]],
        retained_coverages: tuple[float, ...],
        frame_rate_in_hz: float | None = None,
    ) -> Mesh_Lower_Envelope_Summary:
    '''Summarize exact retained-coverage candidates of an ordered height series.'''
    if type(per_frame_minimum_height_in_meter) is not np.ndarray:
        raise ValueError('per-frame minima must be a numpy array')
    series = per_frame_minimum_height_in_meter
    if (
            series.ndim != 1
            or series.size == 0
        ):
        raise ValueError('per-frame minima must have nonempty shape (T,)')
    if not np.issubdtype(series.dtype, np.floating):
        raise ValueError('per-frame minima must have a floating dtype')
    if not bool(np.isfinite(series).all()):
        raise ValueError('per-frame minima must contain only finite values')
    if type(retained_coverages) is not tuple or not retained_coverages:
        raise ValueError('retained_coverages must be a nonempty tuple')
    if any(type(coverage) is not float for coverage in retained_coverages):
        raise ValueError('every retained coverage must be a Python float')
    if any(
            not math.isfinite(coverage) or coverage <= 0.0 or coverage > 1.0
            for coverage in retained_coverages
        ):
        raise ValueError('every retained coverage must be finite and within (0, 1]')
    if len(set(retained_coverages)) != len(retained_coverages):
        raise ValueError('retained coverages must be unique')
    if 1.0 not in retained_coverages:
        raise ValueError('retained coverages must include 1.0')

    normalized_frame_rate: float | None = None
    if frame_rate_in_hz is not None:
        if type(frame_rate_in_hz) not in (int, float):
            raise ValueError('frame_rate_in_hz must be numeric when supplied')
        normalized_frame_rate = float(frame_rate_in_hz)
        if not math.isfinite(normalized_frame_rate) or normalized_frame_rate <= 0.0:
            raise ValueError('frame_rate_in_hz must be finite and positive')

    values = np.array(
        series,
        dtype=np.float64,
        copy=True,
    )
    frame_count = int(values.size)
    coverage_decimals = tuple(str(coverage) for coverage in retained_coverages)
    discarded_counts = tuple(
        frame_count - int(
            (Decimal(coverage_decimal) * Decimal(frame_count)).to_integral_value(
                rounding=ROUND_CEILING,
            )
        )
        for coverage_decimal in coverage_decimals
    )
    partitioned = np.partition(values, tuple(sorted(set(discarded_counts))))
    absolute_minimum = float(partitioned[0])

    candidates: list[Mesh_Lower_Envelope_Candidate] = []
    for coverage, coverage_decimal, discarded_count in zip(
            retained_coverages,
            coverage_decimals,
            discarded_counts,
            strict=True,
        ):
        height = float(partitioned[discarded_count])
        below_mask = values < height
        longest_run = 0
        current_run = 0
        for is_below in below_mask.tolist():
            if bool(is_below):
                current_run += 1
                longest_run = max(longest_run, current_run)
            else:
                current_run = 0
        longest_duration = (
            None
            if normalized_frame_rate is None
            else longest_run / normalized_frame_rate
        )
        candidates.append(
            Mesh_Lower_Envelope_Candidate(
                retained_coverage=coverage,
                retained_coverage_decimal=coverage_decimal,
                discarded_frame_count=discarded_count,
                height_in_meter=height,
                delta_from_absolute_minimum_in_meter=height - absolute_minimum,
                empirical_retained_fraction=float(np.count_nonzero(values >= height))
                / frame_count,
                longest_below_run_frame_count=longest_run,
                longest_below_run_duration_in_second=longest_duration,
            )
        )

    return Mesh_Lower_Envelope_Summary(
        frame_count=frame_count,
        frame_rate_in_hz=normalized_frame_rate,
        absolute_minimum_height_in_meter=absolute_minimum,
        candidates=tuple(candidates),
    )


__all__ = [
    'Mesh_Lower_Envelope_Candidate',
    'Mesh_Lower_Envelope_Summary',
    'compute_per_frame_mesh_minimum_height',
    'summarize_mesh_lower_envelope',
]
