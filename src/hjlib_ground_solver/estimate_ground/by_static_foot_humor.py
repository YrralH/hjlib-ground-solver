'''Exact HuMoR static-foot height-cluster baseline with explicit evidence.'''

from dataclasses import dataclass
import math
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import DBSCAN


type ARRAY_F = NDArray[np.floating[Any]]
type Static_Foot_HuMoR_Status = Literal[
    'upstream_candidate',
    'upstream_zero_fallback',
    'upstream_terrain_rejection',
]
type Static_Foot_HuMoR_Side = Literal['left', 'right']


@dataclass(frozen=True, slots=True)
class Static_Foot_HuMoR_Config:
    '''Frozen constants and upstream identity for the exact comparator.'''

    upstream_commit: str
    displacement_threshold_in_meter_per_native_frame: float
    dbscan_epsilon_in_meter: float
    dbscan_minimum_sample_count: int
    toe_joint_offset_in_meter: float
    terrain_toe_height_threshold_in_meter: float
    terrain_root_height_threshold_in_meter: float
    terrain_sample_count_fps_multiplier: float


@dataclass(frozen=True, slots=True)
class Static_Foot_HuMoR_Sample:
    '''One left-before-right static-toe sample after the motion gate.'''

    side: Static_Foot_HuMoR_Side
    native_frame_index: int
    height_in_meter: float
    dbscan_label: int


@dataclass(frozen=True, slots=True)
class Static_Foot_HuMoR_Cluster:
    '''One exact DBSCAN label summary, including pooled noise label ``-1``.'''

    dbscan_label: int
    native_frame_indices: tuple[int, ...]
    sample_count: int
    toe_height_median_in_meter: float
    root_height_median_in_meter: float
    is_selected: bool
    triggers_terrain_rejection: bool


@dataclass(frozen=True, slots=True)
class Static_Foot_HuMoR_Result:
    '''Immutable numerical result plus sufficient plotting/debug evidence.'''

    status: Static_Foot_HuMoR_Status
    config: Static_Foot_HuMoR_Config
    input_dtype: str
    frame_count: int
    frame_rate_in_hz: float
    terrain_minimum_exclusive_sample_count: int
    left_toe_displacement_in_meter: tuple[float, ...]
    right_toe_displacement_in_meter: tuple[float, ...]
    samples: tuple[Static_Foot_HuMoR_Sample, ...]
    clusters: tuple[Static_Foot_HuMoR_Cluster, ...]
    selected_dbscan_label: int | None
    toe_joint_floor_height_in_meter: float
    upstream_floor_candidate_height_in_meter: float
    accepted_candidate_height_in_meter: float | None
    terrain_interaction: bool


STATIC_FOOT_HUMOR_CONFIG = Static_Foot_HuMoR_Config(
    upstream_commit='fc6ef84f0baa153be15427402e0147ed1a63a11a',
    displacement_threshold_in_meter_per_native_frame=0.005,
    dbscan_epsilon_in_meter=0.005,
    dbscan_minimum_sample_count=3,
    toe_joint_offset_in_meter=0.01,
    terrain_toe_height_threshold_in_meter=0.04,
    terrain_root_height_threshold_in_meter=0.04,
    terrain_sample_count_fps_multiplier=0.25,
)


def normalize_static_foot_humor_tracks(
        root_position_in_meter: ARRAY_F,
        left_toe_position_in_meter: ARRAY_F,
        right_toe_position_in_meter: ARRAY_F,
        frame_rate_in_hz: int | float,
    ) -> tuple[ARRAY_F, ARRAY_F, ARRAY_F, float]:
    '''Validate/copy three same-dtype finite ``(T,3)`` native tracks.'''
    tracks: tuple[object, object, object] = (
        root_position_in_meter,
        left_toe_position_in_meter,
        right_toe_position_in_meter,
    )
    if any(type(track) is not np.ndarray for track in tracks):
        raise ValueError('all position tracks must be numpy arrays')
    typed_tracks = (
        root_position_in_meter,
        left_toe_position_in_meter,
        right_toe_position_in_meter,
    )
    if any(track.ndim != 2 or track.shape[1:] != (3,) for track in typed_tracks):
        raise ValueError('all position tracks must have shape (T, 3)')
    frame_count = int(root_position_in_meter.shape[0])
    if frame_count < 2:
        raise ValueError('position tracks must contain at least two frames')
    if any(int(track.shape[0]) != frame_count for track in typed_tracks):
        raise ValueError('all position tracks must have the same frame count')
    supported_dtypes = (np.dtype(np.float32), np.dtype(np.float64))
    if any(track.dtype not in supported_dtypes for track in typed_tracks):
        raise ValueError('position tracks must have dtype float32 or float64')
    if any(track.dtype != root_position_in_meter.dtype for track in typed_tracks):
        raise ValueError('all position tracks must have the same dtype')
    if any(not bool(np.isfinite(track).all()) for track in typed_tracks):
        raise ValueError('all position tracks must contain only finite coordinates')
    if type(frame_rate_in_hz) not in (int, float):
        raise ValueError('frame_rate_in_hz must be a Python int or float')
    normalized_frame_rate = float(frame_rate_in_hz)
    if not math.isfinite(normalized_frame_rate) or normalized_frame_rate <= 0.0:
        raise ValueError('frame_rate_in_hz must be finite and positive')
    copied_tracks = tuple(
        np.array(track, dtype=track.dtype, order='C', copy=True)
        for track in typed_tracks
    )
    root_copy, left_copy, right_copy = copied_tracks
    return root_copy, left_copy, right_copy, normalized_frame_rate


def compute_repeated_toe_displacement(
        toe_position_in_meter: ARRAY_F,
    ) -> ARRAY_F:
    '''Compute adjacent Euclidean displacement and repeat the final value.'''
    with np.errstate(over='ignore', invalid='ignore'):
        adjacent = np.linalg.norm(
            toe_position_in_meter[1:] - toe_position_in_meter[:-1],
            axis=1,
        )
        displacement = np.concatenate((adjacent, adjacent[-1:]))
    if not bool(np.isfinite(displacement).all()):
        raise ValueError('derived toe displacement must be finite')
    return displacement


def build_static_foot_humor_samples(
        left_toe_position_in_meter: ARRAY_F,
        right_toe_position_in_meter: ARRAY_F,
        left_toe_displacement_in_meter: ARRAY_F,
        right_toe_displacement_in_meter: ARRAY_F,
    ) -> tuple[
        tuple[Static_Foot_HuMoR_Sample, ...],
        ARRAY_F,
        NDArray[np.int64],
        NDArray[np.int64],
    ]:
    '''Build exact left-before-right pooled heights, indices, and labels.'''
    threshold = (
        STATIC_FOOT_HUMOR_CONFIG.
        displacement_threshold_in_meter_per_native_frame
    )
    left_indices = np.flatnonzero(
        left_toe_displacement_in_meter < threshold
    ).astype(np.int64, copy=False)
    right_indices = np.flatnonzero(
        right_toe_displacement_in_meter < threshold
    ).astype(np.int64, copy=False)
    heights = np.concatenate(
        (
            left_toe_position_in_meter[left_indices, 2],
            right_toe_position_in_meter[right_indices, 2],
        )
    )
    native_frame_indices = np.concatenate((left_indices, right_indices))
    if heights.size == 0:
        labels = np.empty((0,), dtype=np.int64)
    else:
        labels = DBSCAN(
            eps=STATIC_FOOT_HUMOR_CONFIG.dbscan_epsilon_in_meter,
            min_samples=STATIC_FOOT_HUMOR_CONFIG.dbscan_minimum_sample_count,
        ).fit_predict(heights.reshape(-1, 1)).astype(np.int64, copy=False)
    left_count = int(left_indices.size)
    samples = tuple(
        Static_Foot_HuMoR_Sample(
            side='left' if position < left_count else 'right',
            native_frame_index=int(native_frame_indices[position]),
            height_in_meter=float(heights[position]),
            dbscan_label=int(labels[position]),
        )
        for position in range(int(heights.size))
    )
    return samples, heights, native_frame_indices, labels


def summarize_static_foot_humor_clusters(
        root_position_in_meter: ARRAY_F,
        heights_in_meter: ARRAY_F,
        native_frame_indices: NDArray[np.int64],
        dbscan_labels: NDArray[np.int64],
        terrain_minimum_exclusive_sample_count: int,
    ) -> tuple[
        tuple[Static_Foot_HuMoR_Cluster, ...],
        int,
        float,
        float,
        bool,
    ]:
    '''Select the lowest label median and apply HuMoR terrain conjunction.'''
    raw_clusters: list[tuple[int, tuple[int, ...], int, float, float]] = []
    selected_label = 0
    selected_toe_median = math.inf
    selected_root_median = math.inf
    for label_value in np.unique(dbscan_labels).tolist():
        label = int(label_value)
        pooled_positions = np.flatnonzero(dbscan_labels == label)
        unique_frames = np.unique(native_frame_indices[pooled_positions])
        toe_median = float(np.median(heights_in_meter[pooled_positions]))
        root_median = float(
            np.median(root_position_in_meter[unique_frames, 2])
        )
        if not math.isfinite(toe_median) or not math.isfinite(root_median):
            raise ValueError('derived cluster medians must be finite')
        raw_clusters.append(
            (
                label,
                tuple(int(frame) for frame in unique_frames.tolist()),
                int(pooled_positions.size),
                toe_median,
                root_median,
            )
        )
        if toe_median < selected_toe_median:
            selected_label = label
            selected_toe_median = toe_median
            selected_root_median = root_median

    clusters: list[Static_Foot_HuMoR_Cluster] = []
    terrain_interaction = False
    for label, unique_frames, sample_count, toe_median, root_median in raw_clusters:
        triggers_terrain = (
            root_median
            > selected_root_median
            + STATIC_FOOT_HUMOR_CONFIG.terrain_root_height_threshold_in_meter
            and toe_median
            > selected_toe_median
            + STATIC_FOOT_HUMOR_CONFIG.terrain_toe_height_threshold_in_meter
            and sample_count > terrain_minimum_exclusive_sample_count
        )
        terrain_interaction = terrain_interaction or triggers_terrain
        clusters.append(
            Static_Foot_HuMoR_Cluster(
                dbscan_label=label,
                native_frame_indices=unique_frames,
                sample_count=sample_count,
                toe_height_median_in_meter=toe_median,
                root_height_median_in_meter=root_median,
                is_selected=label == selected_label,
                triggers_terrain_rejection=triggers_terrain,
            )
        )
    return (
        tuple(clusters),
        selected_label,
        selected_toe_median,
        selected_root_median,
        terrain_interaction,
    )


def estimate_static_foot_humor_baseline(
        root_position_in_meter: ARRAY_F,
        left_toe_position_in_meter: ARRAY_F,
        right_toe_position_in_meter: ARRAY_F,
        frame_rate_in_hz: int | float,
    ) -> Static_Foot_HuMoR_Result:
    '''Run the exact HuMoR floor/terrain subset on named native-rate tracks.'''
    root, left_toe, right_toe, frame_rate = normalize_static_foot_humor_tracks(
        root_position_in_meter,
        left_toe_position_in_meter,
        right_toe_position_in_meter,
        frame_rate_in_hz,
    )
    left_displacement = compute_repeated_toe_displacement(left_toe)
    right_displacement = compute_repeated_toe_displacement(right_toe)
    samples, heights, native_frames, labels = build_static_foot_humor_samples(
        left_toe,
        right_toe,
        left_displacement,
        right_displacement,
    )
    terrain_count_boundary = int(
        STATIC_FOOT_HUMOR_CONFIG.terrain_sample_count_fps_multiplier
        * frame_rate
    )
    if not samples:
        return Static_Foot_HuMoR_Result(
            status='upstream_zero_fallback',
            config=STATIC_FOOT_HUMOR_CONFIG,
            input_dtype=str(root.dtype),
            frame_count=int(root.shape[0]),
            frame_rate_in_hz=frame_rate,
            terrain_minimum_exclusive_sample_count=terrain_count_boundary,
            left_toe_displacement_in_meter=tuple(
                float(value) for value in left_displacement.tolist()
            ),
            right_toe_displacement_in_meter=tuple(
                float(value) for value in right_displacement.tolist()
            ),
            samples=(),
            clusters=(),
            selected_dbscan_label=None,
            toe_joint_floor_height_in_meter=0.0,
            upstream_floor_candidate_height_in_meter=0.0,
            accepted_candidate_height_in_meter=None,
            terrain_interaction=False,
        )

    clusters, selected_label, toe_floor, _root_floor, terrain = \
        summarize_static_foot_humor_clusters(
            root,
            heights,
            native_frames,
            labels,
            terrain_count_boundary,
        )
    upstream_candidate = (
        toe_floor - STATIC_FOOT_HUMOR_CONFIG.toe_joint_offset_in_meter
    )
    status: Static_Foot_HuMoR_Status = (
        'upstream_terrain_rejection' if terrain else 'upstream_candidate'
    )
    return Static_Foot_HuMoR_Result(
        status=status,
        config=STATIC_FOOT_HUMOR_CONFIG,
        input_dtype=str(root.dtype),
        frame_count=int(root.shape[0]),
        frame_rate_in_hz=frame_rate,
        terrain_minimum_exclusive_sample_count=terrain_count_boundary,
        left_toe_displacement_in_meter=tuple(
            float(value) for value in left_displacement.tolist()
        ),
        right_toe_displacement_in_meter=tuple(
            float(value) for value in right_displacement.tolist()
        ),
        samples=samples,
        clusters=clusters,
        selected_dbscan_label=selected_label,
        toe_joint_floor_height_in_meter=toe_floor,
        upstream_floor_candidate_height_in_meter=upstream_candidate,
        accepted_candidate_height_in_meter=(None if terrain else upstream_candidate),
        terrain_interaction=terrain,
    )


__all__ = [
    'Static_Foot_HuMoR_Cluster',
    'Static_Foot_HuMoR_Config',
    'Static_Foot_HuMoR_Result',
    'Static_Foot_HuMoR_Sample',
    'Static_Foot_HuMoR_Status',
    'estimate_static_foot_humor_baseline',
]
