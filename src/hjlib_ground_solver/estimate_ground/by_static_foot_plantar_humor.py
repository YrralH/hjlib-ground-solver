'''HuMoR-style static-foot clustering over shared plantar observations.'''

from dataclasses import dataclass
import math
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import DBSCAN


type ARRAY_F = NDArray[np.floating[Any]]
type Static_Foot_Plantar_HuMoR_Status = Literal[
    'candidate',
    'no_contact_samples',
    'noise_only',
]
type Static_Foot_Plantar_HuMoR_Side = Literal['left', 'right']


@dataclass(frozen=True, slots=True)
class Static_Foot_Plantar_HuMoR_Config:
    '''Explicit physical-speed and one-dimensional DBSCAN parameters.'''

    maximum_contact_speed_in_meter_per_second: float
    dbscan_epsilon_in_meter: float
    dbscan_minimum_sample_count: int

    def __post_init__(self) -> None:
        positive_float_fields = (
            (
                self.maximum_contact_speed_in_meter_per_second,
                'maximum contact speed',
            ),
            (self.dbscan_epsilon_in_meter, 'DBSCAN epsilon'),
        )
        for value, label in positive_float_fields:
            if type(value) is not float or not math.isfinite(value) or value <= 0.0:
                raise ValueError('%s must be a finite positive float' % label)
        if (
                type(self.dbscan_minimum_sample_count) is not int
                or self.dbscan_minimum_sample_count < 2
            ):
            raise ValueError('DBSCAN minimum sample count must be an int >= 2')


@dataclass(frozen=True, slots=True)
class Static_Foot_Plantar_HuMoR_Sample:
    '''One speed-eligible plantar minimum-height sample.'''

    side: Static_Foot_Plantar_HuMoR_Side
    native_frame_index: int
    height_in_meter: float
    median_speed_in_meter_per_second: float
    dbscan_label: int


@dataclass(frozen=True, slots=True)
class Static_Foot_Plantar_HuMoR_Cluster:
    '''Auditable non-noise DBSCAN height cluster.'''

    dbscan_label: int
    native_frame_indices: tuple[int, ...]
    sample_count: int
    minimum_height_in_meter: float
    median_height_in_meter: float
    maximum_height_in_meter: float
    height_span_in_meter: float
    maximum_adjacent_height_gap_in_meter: float
    is_selected: bool


@dataclass(frozen=True, slots=True)
class Static_Foot_Plantar_HuMoR_Result:
    '''Immutable phase-1 candidate plus sample and cluster evidence.'''

    status: Static_Foot_Plantar_HuMoR_Status
    config: Static_Foot_Plantar_HuMoR_Config
    input_dtype: str
    frame_count: int
    left_eligible_sample_count: int
    right_eligible_sample_count: int
    samples: tuple[Static_Foot_Plantar_HuMoR_Sample, ...]
    clusters: tuple[Static_Foot_Plantar_HuMoR_Cluster, ...]
    selected_dbscan_label: int | None
    candidate_height_in_meter: float | None


def normalize_static_foot_plantar_humor_tracks(
        left_height_in_meter: ARRAY_F,
        right_height_in_meter: ARRAY_F,
        left_interval_median_speed_in_meter_per_second: ARRAY_F,
        right_interval_median_speed_in_meter_per_second: ARRAY_F,
        config: Static_Foot_Plantar_HuMoR_Config,
    ) -> tuple[ARRAY_F, ARRAY_F, ARRAY_F, ARRAY_F]:
    '''Validate and copy same-dtype finite height/speed tracks.'''
    tracks: tuple[object, object, object, object] = (
        left_height_in_meter,
        right_height_in_meter,
        left_interval_median_speed_in_meter_per_second,
        right_interval_median_speed_in_meter_per_second,
    )
    if any(type(track) is not np.ndarray for track in tracks):
        raise ValueError('all plantar tracks must be numpy arrays')
    if type(config) is not Static_Foot_Plantar_HuMoR_Config:
        raise ValueError('config must be Static_Foot_Plantar_HuMoR_Config')
    if left_height_in_meter.ndim != 1 or right_height_in_meter.ndim != 1:
        raise ValueError('plantar height tracks must be one-dimensional')
    frame_count = int(left_height_in_meter.size)
    if frame_count < 2 or right_height_in_meter.shape != (frame_count,):
        raise ValueError('plantar height tracks must share shape (T,) with T >= 2')
    interval_shape = (frame_count - 1,)
    if (
            left_interval_median_speed_in_meter_per_second.shape != interval_shape
            or right_interval_median_speed_in_meter_per_second.shape != interval_shape
        ):
        raise ValueError('plantar speed tracks must share shape (T-1,)')
    typed_tracks = (
        left_height_in_meter,
        right_height_in_meter,
        left_interval_median_speed_in_meter_per_second,
        right_interval_median_speed_in_meter_per_second,
    )
    supported_dtypes = (np.dtype(np.float32), np.dtype(np.float64))
    if any(track.dtype not in supported_dtypes for track in typed_tracks):
        raise ValueError('plantar tracks must have dtype float32 or float64')
    if any(track.dtype != left_height_in_meter.dtype for track in typed_tracks):
        raise ValueError('all plantar tracks must have the same dtype')
    if any(not bool(np.isfinite(track).all()) for track in typed_tracks):
        raise ValueError('all plantar tracks must contain only finite values')
    if (
            bool((left_interval_median_speed_in_meter_per_second < 0.0).any())
            or bool((right_interval_median_speed_in_meter_per_second < 0.0).any())
        ):
        raise ValueError('plantar median speeds must be nonnegative')
    left_height_copy = np.array(
        left_height_in_meter,
        dtype=left_height_in_meter.dtype,
        order='C',
        copy=True,
    )
    right_height_copy = np.array(
        right_height_in_meter,
        dtype=right_height_in_meter.dtype,
        order='C',
        copy=True,
    )
    left_speed_copy = np.array(
        left_interval_median_speed_in_meter_per_second,
        dtype=left_interval_median_speed_in_meter_per_second.dtype,
        order='C',
        copy=True,
    )
    right_speed_copy = np.array(
        right_interval_median_speed_in_meter_per_second,
        dtype=right_interval_median_speed_in_meter_per_second.dtype,
        order='C',
        copy=True,
    )
    return (
        left_height_copy,
        right_height_copy,
        left_speed_copy,
        right_speed_copy,
    )


def align_interval_speed_to_native_frames(interval_speed: ARRAY_F) -> ARRAY_F:
    '''Assign each forward interval to its first frame and repeat the last.'''
    aligned = np.concatenate((interval_speed, interval_speed[-1:]))
    if not bool(np.isfinite(aligned).all()):
        raise ValueError('derived aligned plantar speed must remain finite')
    return aligned


def build_static_foot_plantar_humor_samples(
        left_height_in_meter: ARRAY_F,
        right_height_in_meter: ARRAY_F,
        left_speed_in_meter_per_second: ARRAY_F,
        right_speed_in_meter_per_second: ARRAY_F,
        config: Static_Foot_Plantar_HuMoR_Config,
    ) -> tuple[
        tuple[Static_Foot_Plantar_HuMoR_Sample, ...],
        ARRAY_F,
        NDArray[np.int64],
        NDArray[np.int64],
        int,
    ]:
    '''Pool left-before-right eligible samples and compute DBSCAN labels.'''
    threshold = config.maximum_contact_speed_in_meter_per_second
    left_indices = np.flatnonzero(left_speed_in_meter_per_second < threshold).astype(
        np.int64,
        copy=False,
    )
    right_indices = np.flatnonzero(right_speed_in_meter_per_second < threshold).astype(
        np.int64,
        copy=False,
    )
    heights = np.concatenate((
        left_height_in_meter[left_indices],
        right_height_in_meter[right_indices],
    ))
    speeds = np.concatenate((
        left_speed_in_meter_per_second[left_indices],
        right_speed_in_meter_per_second[right_indices],
    ))
    native_frame_indices = np.concatenate((left_indices, right_indices))
    if heights.size == 0:
        labels = np.empty((0,), dtype=np.int64)
    else:
        labels = DBSCAN(
            eps=config.dbscan_epsilon_in_meter,
            min_samples=config.dbscan_minimum_sample_count,
        ).fit_predict(heights.reshape(-1, 1)).astype(np.int64, copy=False)
    left_count = int(left_indices.size)
    samples = tuple(
        Static_Foot_Plantar_HuMoR_Sample(
            side='left' if position < left_count else 'right',
            native_frame_index=int(native_frame_indices[position]),
            height_in_meter=float(heights[position]),
            median_speed_in_meter_per_second=float(speeds[position]),
            dbscan_label=int(labels[position]),
        )
        for position in range(int(heights.size))
    )
    return samples, heights, native_frame_indices, labels, left_count


def summarize_static_foot_plantar_humor_clusters(
        heights_in_meter: ARRAY_F,
        native_frame_indices: NDArray[np.int64],
        dbscan_labels: NDArray[np.int64],
    ) -> tuple[
        tuple[Static_Foot_Plantar_HuMoR_Cluster, ...],
        int | None,
        float | None,
    ]:
    '''Summarize non-noise clusters and select the lowest median.'''
    raw_clusters: list[
        tuple[int, tuple[int, ...], int, float, float, float, float, float]
    ] = []
    for label_value in sorted(
            int(value) for value in np.unique(dbscan_labels).tolist()
            if int(value) >= 0
        ):
        positions = np.flatnonzero(dbscan_labels == label_value)
        ordered_heights = np.sort(heights_in_meter[positions])
        minimum = float(ordered_heights[0])
        median = float(np.median(ordered_heights))
        maximum = float(ordered_heights[-1])
        with np.errstate(over='ignore', invalid='ignore'):
            span = float(maximum - minimum)
            maximum_gap = (
                float(np.max(np.diff(ordered_heights)))
                if ordered_heights.size > 1
                else 0.0
            )
        derived = (minimum, median, maximum, span, maximum_gap)
        if not all(math.isfinite(value) for value in derived):
            raise ValueError('derived plantar cluster evidence must remain finite')
        unique_frames = tuple(
            int(frame)
            for frame in np.unique(native_frame_indices[positions]).tolist()
        )
        raw_clusters.append((
            label_value,
            unique_frames,
            int(positions.size),
            minimum,
            median,
            maximum,
            span,
            maximum_gap,
        ))
    if not raw_clusters:
        return (), None, None
    selected = min(raw_clusters, key=lambda cluster: (cluster[4], cluster[0]))
    selected_label = selected[0]
    clusters = tuple(
        Static_Foot_Plantar_HuMoR_Cluster(
            dbscan_label=label,
            native_frame_indices=frames,
            sample_count=count,
            minimum_height_in_meter=minimum,
            median_height_in_meter=median,
            maximum_height_in_meter=maximum,
            height_span_in_meter=span,
            maximum_adjacent_height_gap_in_meter=maximum_gap,
            is_selected=label == selected_label,
        )
        for (
            label,
            frames,
            count,
            minimum,
            median,
            maximum,
            span,
            maximum_gap,
        ) in raw_clusters
    )
    return clusters, selected_label, selected[4]


def estimate_static_foot_plantar_humor_baseline(
        left_height_in_meter: ARRAY_F,
        right_height_in_meter: ARRAY_F,
        left_interval_median_speed_in_meter_per_second: ARRAY_F,
        right_interval_median_speed_in_meter_per_second: ARRAY_F,
        config: Static_Foot_Plantar_HuMoR_Config,
    ) -> Static_Foot_Plantar_HuMoR_Result:
    '''Estimate a common-domain plantar HuMoR-style phase-1 candidate.'''
    left_height, right_height, left_interval_speed, right_interval_speed = \
        normalize_static_foot_plantar_humor_tracks(
            left_height_in_meter,
            right_height_in_meter,
            left_interval_median_speed_in_meter_per_second,
            right_interval_median_speed_in_meter_per_second,
            config,
        )
    left_speed = align_interval_speed_to_native_frames(left_interval_speed)
    right_speed = align_interval_speed_to_native_frames(right_interval_speed)
    samples, heights, frames, labels, left_count = \
        build_static_foot_plantar_humor_samples(
            left_height,
            right_height,
            left_speed,
            right_speed,
            config,
        )
    if not samples:
        status: Static_Foot_Plantar_HuMoR_Status = 'no_contact_samples'
        clusters: tuple[Static_Foot_Plantar_HuMoR_Cluster, ...] = ()
        selected_label = None
        candidate = None
    else:
        clusters, selected_label, candidate = \
            summarize_static_foot_plantar_humor_clusters(
                heights,
                frames,
                labels,
            )
        status = 'candidate' if candidate is not None else 'noise_only'
    return Static_Foot_Plantar_HuMoR_Result(
        status=status,
        config=config,
        input_dtype=str(left_height.dtype),
        frame_count=int(left_height.size),
        left_eligible_sample_count=left_count,
        right_eligible_sample_count=len(samples) - left_count,
        samples=samples,
        clusters=clusters,
        selected_dbscan_label=selected_label,
        candidate_height_in_meter=candidate,
    )


__all__ = [
    'Static_Foot_Plantar_HuMoR_Cluster',
    'Static_Foot_Plantar_HuMoR_Config',
    'Static_Foot_Plantar_HuMoR_Result',
    'Static_Foot_Plantar_HuMoR_Sample',
    'Static_Foot_Plantar_HuMoR_Status',
    'estimate_static_foot_plantar_humor_baseline',
]
