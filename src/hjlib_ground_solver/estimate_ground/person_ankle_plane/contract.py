'''Contracts retained for the temporarily deprecated ankle-plane V1.'''

from dataclasses import dataclass
import math
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray


type ARRAY_F = NDArray[np.float64]
type ARRAY_I = NDArray[np.int64]
type Person_Ankle_Foot_Side = Literal['left', 'right']
type Person_Ankle_Height_Shape = Literal['band', 'transition', 'diffuse']
type Person_Ankle_Time_Shape = Literal['persistent', 'episodic']
type Person_Ankle_Space_Shape = Literal['compact', 'path', 'dispersed']
type Person_Ankle_Foot_Shape = Literal[
    'left', 'right', 'bilateral', 'alternating', 'mixed',
]
type Person_Ankle_Noise_Reason = Literal[
    'density_noise', 'ambiguous_border', 'incompatible_reference_planes',
]
type Person_Ankle_Transition_Outcome = Literal['valid', 'invalid', 'vetoed']
type Person_Ankle_Plane_Status = Literal[
    'single_support_plane',
    'plane_switch',
    'multi_layer_ambiguous',
    'local_episode_ambiguous',
    'no_ground_evidence',
    'incompatible_reference_planes',
]


def readonly_float_array(value: object, shape: tuple[int, ...], label: str) -> ARRAY_F:
    '''Validate one exact-float64 finite array and return a read-only copy.'''
    if type(value) is not np.ndarray:
        raise ValueError('%s must be a numpy array' % label)
    array = value
    if array.dtype != np.dtype(np.float64) or array.shape != shape:
        raise ValueError('%s must have shape %s and dtype float64' % (label, shape))
    if not bool(np.isfinite(array).all()):
        raise ValueError('%s must contain only finite values' % label)
    copied = np.array(array, dtype=np.float64, order='C', copy=True)
    copied.setflags(write=False)
    return copied


def readonly_int_array(value: object, shape: tuple[int, ...], label: str) -> ARRAY_I:
    '''Validate one exact-int64 array and return a read-only copy.'''
    if type(value) is not np.ndarray:
        raise ValueError('%s must be a numpy array' % label)
    array = value
    if array.dtype != np.dtype(np.int64) or array.shape != shape:
        raise ValueError('%s must have shape %s and dtype int64' % (label, shape))
    copied = np.array(array, dtype=np.int64, order='C', copy=True)
    copied.setflags(write=False)
    return copied


def validate_nonempty_string(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError('%s must be a non-empty string' % label)
    return value


def validate_positive_float(value: object, label: str) -> float:
    if type(value) is not float or not math.isfinite(value) or value <= 0.0:
        raise ValueError('%s must be a finite positive float' % label)
    return value


def validate_nonnegative_float(value: object, label: str) -> float:
    if type(value) is not float or not math.isfinite(value) or value < 0.0:
        raise ValueError('%s must be a finite nonnegative float' % label)
    return value


def validate_fraction(value: object, label: str, allow_zero: bool = True) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError('%s must be a finite float' % label)
    minimum = 0.0 if allow_zero else 0.0
    if value > 1.0 or value < minimum or (not allow_zero and value == 0.0):
        raise ValueError('%s must be in %s' % (label, '[0,1]' if allow_zero else '(0,1]'))
    return value


def validate_positive_int(value: object, label: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError('%s must be an int >= %d' % (label, minimum))
    return value


def validate_finite_float(value: object, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError('%s must be a finite float' % label)
    return value


def validate_tuple_of_nonnegative_ints(
        value: object,
        label: str,
        allow_empty: bool = False,
    ) -> tuple[int, ...]:
    if type(value) is not tuple or (not value and not allow_empty):
        raise ValueError('%s must be %sa tuple' % (
            label, 'an optionally empty ' if allow_empty else 'a non-empty '))
    if any(type(item) is not int or item < 0 for item in value):
        raise ValueError('%s must contain exact nonnegative ints' % label)
    if len(set(value)) != len(value):
        raise ValueError('%s must not contain duplicates' % label)
    return value


def validate_point_tuple(
        value: object,
        label: str,
    ) -> tuple[float, float, float]:
    if type(value) is not tuple or len(value) != 3:
        raise ValueError('%s must be a length-three tuple' % label)
    if any(type(item) is not float or not math.isfinite(item) for item in value):
        raise ValueError('%s must contain finite floats' % label)
    return value


def validate_nonzero_plane_normal(plane: ARRAY_F, label: str) -> None:
    if float(np.max(np.abs(plane[:3]))) == 0.0:
        raise ValueError('%s normal must be nonzero' % label)


@dataclass(frozen=True, slots=True)
class Person_Ankle_Source_Plane:
    plane_key: str
    ground_param_world: ARRAY_F
    body_side_sign: int

    def __post_init__(self) -> None:
        validate_nonempty_string(self.plane_key, 'plane key')
        plane = readonly_float_array(self.ground_param_world, (4,), 'ground plane')
        validate_nonzero_plane_normal(plane, 'ground plane')
        if type(self.body_side_sign) is not int or self.body_side_sign not in (-1, 1):
            raise ValueError('body side sign must be exact int -1 or +1')
        object.__setattr__(self, 'ground_param_world', plane)


@dataclass(frozen=True, slots=True)
class Person_Ankle_Run:
    run_key: str
    time_domain_key: str
    plane_key: str
    fps: float
    frame_indices_original: ARRAY_I
    ankle_world_in_meter: ARRAY_F
    observation_keys: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        validate_nonempty_string(self.run_key, 'run key')
        validate_nonempty_string(self.time_domain_key, 'time domain key')
        validate_nonempty_string(self.plane_key, 'run plane key')
        validate_positive_float(self.fps, 'fps')
        if type(self.frame_indices_original) is not np.ndarray:
            raise ValueError('frame indices must be a numpy array')
        frame_shape = cast(NDArray[Any], self.frame_indices_original).shape
        if len(frame_shape) != 1 or frame_shape[0] < 1:
            raise ValueError('frame indices must have shape (T,) with T >= 1')
        frame_count = int(frame_shape[0])
        frames = readonly_int_array(
            self.frame_indices_original, (frame_count,), 'frame indices')
        if frame_count > 1 and bool((np.diff(frames) <= 0).any()):
            raise ValueError('frame indices must be strictly increasing')
        ankles = readonly_float_array(
            self.ankle_world_in_meter, (frame_count, 2, 3), 'world ankles')
        if type(self.observation_keys) is not tuple or len(self.observation_keys) != frame_count:
            raise ValueError('observation keys must be a tuple with T rows')
        normalized_keys: list[tuple[str, str]] = []
        for row in self.observation_keys:
            if type(row) is not tuple or len(row) != 2:
                raise ValueError('each observation-key row must contain two strings')
            normalized_keys.append((
                validate_nonempty_string(row[0], 'left observation key'),
                validate_nonempty_string(row[1], 'right observation key'),
            ))
        object.__setattr__(self, 'frame_indices_original', frames)
        object.__setattr__(self, 'ankle_world_in_meter', ankles)
        object.__setattr__(self, 'observation_keys', tuple(normalized_keys))


@dataclass(frozen=True, slots=True)
class Person_Ankle_Plane_Input:
    person_key: str
    runs: tuple[Person_Ankle_Run, ...]
    source_planes: tuple[Person_Ankle_Source_Plane, ...]

    def __post_init__(self) -> None:
        validate_nonempty_string(self.person_key, 'person key')
        if type(self.runs) is not tuple or not self.runs:
            raise ValueError('runs must be a non-empty tuple')
        if type(self.source_planes) is not tuple or not self.source_planes:
            raise ValueError('source planes must be a non-empty tuple')
        if any(type(run) is not Person_Ankle_Run for run in self.runs):
            raise ValueError('runs must contain Person_Ankle_Run values')
        if any(type(plane) is not Person_Ankle_Source_Plane for plane in self.source_planes):
            raise ValueError('source planes must contain Person_Ankle_Source_Plane values')


@dataclass(frozen=True, slots=True)
class Person_Ankle_Run_Provenance:
    run_key: str
    time_domain_key: str
    plane_key: str
    fps: float
    frame_count: int
    first_frame_index_original: int
    last_frame_index_original: int

    def __post_init__(self) -> None:
        validate_nonempty_string(self.run_key, 'run provenance key')
        validate_nonempty_string(
            self.time_domain_key, 'run provenance time-domain key')
        validate_nonempty_string(self.plane_key, 'run provenance plane key')
        validate_positive_float(self.fps, 'run provenance fps')
        validate_positive_int(self.frame_count, 'run provenance frame count')
        if (
                type(self.first_frame_index_original) is not int
                or type(self.last_frame_index_original) is not int
                or self.first_frame_index_original > self.last_frame_index_original
            ):
            raise ValueError('run provenance frame interval is invalid')


@dataclass(frozen=True, slots=True)
class Person_Ankle_Local_Config:
    time_radius_in_second: float
    projected_radius_in_meter: float
    height_radius_in_meter: float
    minimum_core_neighbor_count: int
    band_quantile_width_max_in_meter: float
    band_full_span_max_in_meter: float
    transition_endpoint_fraction: float
    transition_height_min_in_meter: float
    transition_rank_min: float
    persistent_duration_min_in_second: float
    persistent_occupancy_min: float
    compact_radius_max_in_meter: float
    path_length_min_in_meter: float
    bilateral_frame_fraction_min: float
    alternating_change_fraction_min: float

    def __post_init__(self) -> None:
        for value, label in (
                (self.time_radius_in_second, 'local time radius'),
                (self.projected_radius_in_meter, 'local projected radius'),
                (self.height_radius_in_meter, 'local height radius'),
                (self.band_quantile_width_max_in_meter, 'band quantile width'),
                (self.band_full_span_max_in_meter, 'band full span'),
                (self.transition_height_min_in_meter, 'transition height'),
                (self.persistent_duration_min_in_second, 'persistent duration'),
                (self.compact_radius_max_in_meter, 'compact radius'),
                (self.path_length_min_in_meter, 'path length'),
            ):
            validate_positive_float(value, label)
        validate_positive_int(
            self.minimum_core_neighbor_count, 'minimum core neighbor count', 2)
        validate_fraction(
            self.transition_endpoint_fraction, 'transition endpoint fraction', False)
        if self.transition_endpoint_fraction > 0.5:
            raise ValueError('transition endpoint fraction must be <= 0.5')
        validate_fraction(self.transition_rank_min, 'transition rank minimum')
        validate_fraction(self.persistent_occupancy_min, 'persistent occupancy')
        validate_fraction(self.bilateral_frame_fraction_min, 'bilateral fraction')
        validate_fraction(
            self.alternating_change_fraction_min, 'alternating change fraction')


@dataclass(frozen=True, slots=True)
class Person_Ankle_Global_Config:
    sigma_floor_in_meter: float
    merge_z_max: float
    merge_height_gap_max_in_meter: float
    episode_stitch_time_max_in_second: float
    episode_stitch_distance_max_in_meter: float
    recurrent_episode_count_min: int
    recurrent_observed_duration_min_in_second: float
    recurrent_time_separation_min_in_second: float
    recurrent_distance_min_in_meter: float
    competitor_cluster_count_min: int
    competitor_observed_duration_min_in_second: float

    def __post_init__(self) -> None:
        for value, label in (
                (self.sigma_floor_in_meter, 'sigma floor'),
                (self.merge_z_max, 'merge z maximum'),
                (self.merge_height_gap_max_in_meter, 'merge height gap'),
                (self.episode_stitch_time_max_in_second, 'episode stitch time'),
                (self.episode_stitch_distance_max_in_meter, 'episode stitch distance'),
                (self.recurrent_observed_duration_min_in_second, 'recurrent duration'),
                (self.recurrent_time_separation_min_in_second, 'recurrent time separation'),
                (self.recurrent_distance_min_in_meter, 'recurrent distance'),
                (self.competitor_observed_duration_min_in_second, 'competitor duration'),
            ):
            validate_positive_float(value, label)
        validate_positive_int(
            self.recurrent_episode_count_min, 'recurrent episode count', 2)
        validate_positive_int(
            self.competitor_cluster_count_min, 'competitor cluster count')


@dataclass(frozen=True, slots=True)
class Person_Ankle_Transition_Config:
    segment_time_gap_max_in_second: float
    transition_overlap_duration_max_in_second: float
    transition_time_max_in_second: float
    transition_distance_max_in_meter: float
    switch_episode_duration_min_in_second: float
    switch_height_gap_min_in_meter: float
    switch_scale_multiplier: float
    transition_valid_fraction_min: float

    def __post_init__(self) -> None:
        for value, label in (
                (self.segment_time_gap_max_in_second, 'segment time gap'),
                (self.transition_overlap_duration_max_in_second, 'transition overlap duration'),
                (self.transition_time_max_in_second, 'transition time'),
                (self.transition_distance_max_in_meter, 'transition distance'),
                (self.switch_episode_duration_min_in_second, 'switch episode duration'),
                (self.switch_height_gap_min_in_meter, 'switch height gap'),
                (self.switch_scale_multiplier, 'switch scale multiplier'),
            ):
            validate_positive_float(value, label)
        validate_fraction(
            self.transition_valid_fraction_min, 'transition valid fraction')


@dataclass(frozen=True, slots=True)
class Person_Ankle_Plane_Config:
    reference_normal_angle_max_in_radian: float
    reference_residual_max_in_meter: float
    local: Person_Ankle_Local_Config
    global_config: Person_Ankle_Global_Config
    transition: Person_Ankle_Transition_Config

    def __post_init__(self) -> None:
        validate_nonnegative_float(
            self.reference_normal_angle_max_in_radian, 'reference normal angle')
        if self.reference_normal_angle_max_in_radian > math.pi:
            raise ValueError('reference normal angle must be <= pi')
        validate_nonnegative_float(
            self.reference_residual_max_in_meter, 'reference residual')
        if type(self.local) is not Person_Ankle_Local_Config:
            raise ValueError('local config has wrong type')
        if type(self.global_config) is not Person_Ankle_Global_Config:
            raise ValueError('global config has wrong type')
        if type(self.transition) is not Person_Ankle_Transition_Config:
            raise ValueError('transition config has wrong type')
        if (
                self.local.band_quantile_width_max_in_meter
                > self.local.band_full_span_max_in_meter
            ):
            raise ValueError('band quantile width maximum must not exceed full span maximum')
        if (
                self.transition.transition_overlap_duration_max_in_second
                > self.transition.transition_time_max_in_second
            ):
            raise ValueError('transition overlap duration must not exceed transition time maximum')


@dataclass(frozen=True, slots=True)
class Person_Ankle_Observation_Table:
    observation_keys: tuple[str, ...]
    run_keys: tuple[str, ...]
    time_domain_keys: tuple[str, ...]
    frame_indices_original: ARRAY_I
    foot_sides: tuple[Person_Ankle_Foot_Side, ...]
    times_in_second: ARRAY_F
    world_points_in_meter: ARRAY_F
    projected_points_in_meter: ARRAY_F
    heights_in_meter: ARRAY_F
    local_cluster_labels: ARRAY_I
    noise_reasons: tuple[Person_Ankle_Noise_Reason | None, ...]

    def __post_init__(self) -> None:
        if type(self.observation_keys) is not tuple or not self.observation_keys:
            raise ValueError('observation table must contain at least one observation')
        count = len(self.observation_keys)
        tuple_columns = (
            (self.run_keys, 'run keys'),
            (self.time_domain_keys, 'time-domain keys'),
            (self.foot_sides, 'foot sides'),
            (self.noise_reasons, 'noise reasons'),
        )
        if any(type(column) is not tuple or len(column) != count
               for column, _ in tuple_columns):
            raise ValueError('observation tuple columns must have a common length')
        if len(set(self.observation_keys)) != count:
            raise ValueError('observation table keys must be unique')
        if any(type(value) is not str or not value for value in self.observation_keys):
            raise ValueError('observation table keys must be non-empty strings')
        if any(type(value) is not str or not value for value in self.run_keys):
            raise ValueError('observation run keys must be non-empty strings')
        if any(type(value) is not str or not value for value in self.time_domain_keys):
            raise ValueError('observation time-domain keys must be non-empty strings')
        if any(value not in ('left', 'right') for value in self.foot_sides):
            raise ValueError('observation foot sides contain an invalid value')
        frames = readonly_int_array(
            self.frame_indices_original, (count,), 'observation frame indices')
        times = readonly_float_array(
            self.times_in_second, (count,), 'observation times')
        world = readonly_float_array(
            self.world_points_in_meter, (count, 3), 'observation world points')
        projected = readonly_float_array(
            self.projected_points_in_meter, (count, 3), 'observation projected points')
        heights = readonly_float_array(
            self.heights_in_meter, (count,), 'observation heights')
        labels = readonly_int_array(
            self.local_cluster_labels, (count,), 'observation local labels')
        if bool((labels < -1).any()):
            raise ValueError('observation local labels must be -1 or nonnegative')
        valid_noise_reasons = (
            None, 'density_noise', 'ambiguous_border',
            'incompatible_reference_planes',
        )
        if any(value not in valid_noise_reasons for value in self.noise_reasons):
            raise ValueError('observation noise reasons contain an invalid value')
        for label, reason in zip(labels, self.noise_reasons):
            if (int(label) < 0) == (reason is None):
                raise ValueError('observation labels and noise reasons are inconsistent')
        object.__setattr__(self, 'frame_indices_original', frames)
        object.__setattr__(self, 'times_in_second', times)
        object.__setattr__(self, 'world_points_in_meter', world)
        object.__setattr__(self, 'projected_points_in_meter', projected)
        object.__setattr__(self, 'heights_in_meter', heights)
        object.__setattr__(self, 'local_cluster_labels', labels)


@dataclass(frozen=True, slots=True)
class Person_Ankle_Local_Cluster:
    local_cluster_id: int
    run_key: str
    time_domain_key: str
    observation_indices: tuple[int, ...]
    sample_count: int
    unique_frame_count: int
    left_count: int
    right_count: int
    bilateral_frame_count: int
    minimum_height_in_meter: float
    median_height_in_meter: float
    maximum_height_in_meter: float
    percentile_10_height_in_meter: float
    percentile_90_height_in_meter: float
    bmad_in_meter: float
    maximum_adjacent_height_gap_in_meter: float
    first_time_in_second: float
    last_time_in_second: float
    observed_duration_in_second: float
    temporal_span_in_second: float
    temporal_occupancy: float
    contiguous_frame_run_count: int
    projected_centroid_in_meter: tuple[float, float, float]
    robust_radius_in_meter: float
    chronological_path_length_in_meter: float
    first_projected_by_foot: tuple[tuple[float, float, float] | None, ...]
    last_projected_by_foot: tuple[tuple[float, float, float] | None, ...]
    height_shape: Person_Ankle_Height_Shape
    time_shape: Person_Ankle_Time_Shape
    space_shape: Person_Ankle_Space_Shape
    foot_shape: Person_Ankle_Foot_Shape

    def __post_init__(self) -> None:
        validate_positive_int(
            self.local_cluster_id, 'local cluster id', 0)
        validate_nonempty_string(self.run_key, 'local cluster run key')
        validate_nonempty_string(
            self.time_domain_key, 'local cluster time-domain key')
        members = validate_tuple_of_nonnegative_ints(
            self.observation_indices, 'local cluster observation indices')
        if self.sample_count != len(members):
            raise ValueError('local cluster sample count must match membership')
        for value, label, minimum in (
                (self.sample_count, 'local cluster sample count', 1),
                (self.unique_frame_count, 'local cluster unique frame count', 1),
                (self.left_count, 'local cluster left count', 0),
                (self.right_count, 'local cluster right count', 0),
                (self.bilateral_frame_count, 'local cluster bilateral count', 0),
                (self.contiguous_frame_run_count, 'local cluster contiguous runs', 1),
            ):
            validate_positive_int(value, label, minimum)
        if self.left_count + self.right_count != self.sample_count:
            raise ValueError('local cluster foot counts must match sample count')
        if self.unique_frame_count > self.sample_count:
            raise ValueError('local cluster unique-frame count exceeds sample count')
        finite_values = (
            self.minimum_height_in_meter,
            self.median_height_in_meter,
            self.maximum_height_in_meter,
            self.percentile_10_height_in_meter,
            self.percentile_90_height_in_meter,
            self.bmad_in_meter,
            self.maximum_adjacent_height_gap_in_meter,
            self.first_time_in_second,
            self.last_time_in_second,
            self.observed_duration_in_second,
            self.temporal_span_in_second,
            self.temporal_occupancy,
            self.robust_radius_in_meter,
            self.chronological_path_length_in_meter,
        )
        if any(type(value) is not float or not math.isfinite(value)
               for value in finite_values):
            raise ValueError('local cluster statistics must be finite floats')
        if not (
                self.minimum_height_in_meter <= self.median_height_in_meter
                <= self.maximum_height_in_meter
            ):
            raise ValueError('local cluster height order is inconsistent')
        if self.first_time_in_second > self.last_time_in_second:
            raise ValueError('local cluster time order is inconsistent')
        if (
                self.bmad_in_meter < 0.0
                or self.maximum_adjacent_height_gap_in_meter < 0.0
                or self.observed_duration_in_second <= 0.0
                or self.temporal_span_in_second <= 0.0
                or not 0.0 <= self.temporal_occupancy <= 1.0
                or self.robust_radius_in_meter < 0.0
                or self.chronological_path_length_in_meter < 0.0
            ):
            raise ValueError('local cluster nonnegative statistics are invalid')
        validate_point_tuple(
            self.projected_centroid_in_meter, 'local cluster centroid')
        for endpoints, label in (
                (self.first_projected_by_foot, 'local cluster first endpoints'),
                (self.last_projected_by_foot, 'local cluster last endpoints'),
            ):
            if type(endpoints) is not tuple or len(endpoints) != 2:
                raise ValueError('%s must be a two-foot tuple' % label)
            for point in endpoints:
                if point is not None:
                    validate_point_tuple(point, label)
        if self.height_shape not in ('band', 'transition', 'diffuse'):
            raise ValueError('local cluster height shape is invalid')
        if self.time_shape not in ('persistent', 'episodic'):
            raise ValueError('local cluster time shape is invalid')
        if self.space_shape not in ('compact', 'path', 'dispersed'):
            raise ValueError('local cluster space shape is invalid')
        if self.foot_shape not in (
                'left', 'right', 'bilateral', 'alternating', 'mixed'):
            raise ValueError('local cluster foot shape is invalid')


@dataclass(frozen=True, slots=True)
class Person_Ankle_Recurrence_Episode:
    episode_id: int
    local_cluster_ids: tuple[int, ...]
    observed_duration_in_second: float
    midpoint_time_in_second: float
    projected_centroid_in_meter: tuple[float, float, float]

    def __post_init__(self) -> None:
        validate_positive_int(self.episode_id, 'episode id', 0)
        validate_tuple_of_nonnegative_ints(
            self.local_cluster_ids, 'episode local cluster ids')
        validate_positive_float(
            self.observed_duration_in_second, 'episode observed duration')
        validate_finite_float(self.midpoint_time_in_second, 'episode midpoint time')
        validate_point_tuple(self.projected_centroid_in_meter, 'episode centroid')


@dataclass(frozen=True, slots=True)
class Person_Ankle_Height_Hypothesis:
    hypothesis_id: int
    local_cluster_ids: tuple[int, ...]
    center_height_in_meter: float
    bmad_in_meter: float
    sample_weighted_center_in_meter: float
    observed_duration_in_second: float
    episodes: tuple[Person_Ankle_Recurrence_Episode, ...]
    temporal_separation_in_second: float
    projected_separation_in_meter: float
    recurrent: bool
    materially_supported_competitor: bool

    def __post_init__(self) -> None:
        validate_positive_int(self.hypothesis_id, 'hypothesis id', 0)
        validate_tuple_of_nonnegative_ints(
            self.local_cluster_ids, 'hypothesis local cluster ids')
        for value, label in (
                (self.center_height_in_meter, 'hypothesis center'),
                (self.bmad_in_meter, 'hypothesis BMAD'),
                (self.sample_weighted_center_in_meter, 'hypothesis weighted center'),
                (self.observed_duration_in_second, 'hypothesis observed duration'),
                (self.temporal_separation_in_second, 'hypothesis temporal separation'),
                (self.projected_separation_in_meter, 'hypothesis projected separation'),
            ):
            validate_finite_float(value, label)
        if (
                self.bmad_in_meter < 0.0
                or self.observed_duration_in_second <= 0.0
                or self.temporal_separation_in_second < 0.0
                or self.projected_separation_in_meter < 0.0
            ):
            raise ValueError('hypothesis nonnegative statistics are invalid')
        if type(self.episodes) is not tuple or not self.episodes:
            raise ValueError('hypothesis episodes must be a non-empty tuple')
        if any(type(value) is not Person_Ankle_Recurrence_Episode
               for value in self.episodes):
            raise ValueError('hypothesis episodes contain a wrong value type')
        if type(self.recurrent) is not bool:
            raise ValueError('hypothesis recurrent flag must be exact bool')
        if type(self.materially_supported_competitor) is not bool:
            raise ValueError('hypothesis competitor flag must be exact bool')
        if self.recurrent and self.materially_supported_competitor:
            raise ValueError('recurrent hypothesis cannot be a non-recurrent competitor')


@dataclass(frozen=True, slots=True)
class Person_Ankle_Hypothesis_Segment:
    segment_id: int
    run_key: str
    hypothesis_id: int
    first_frame_index_original: int
    last_frame_index_original: int
    first_time_in_second: float
    last_time_in_second: float
    observation_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        validate_positive_int(self.segment_id, 'segment id', 0)
        validate_nonempty_string(self.run_key, 'segment run key')
        validate_positive_int(self.hypothesis_id, 'segment hypothesis id', 0)
        if (
                type(self.first_frame_index_original) is not int
                or type(self.last_frame_index_original) is not int
                or self.first_frame_index_original > self.last_frame_index_original
            ):
            raise ValueError('segment frame interval is invalid')
        validate_finite_float(self.first_time_in_second, 'segment first time')
        validate_finite_float(self.last_time_in_second, 'segment last time')
        if self.first_time_in_second > self.last_time_in_second:
            raise ValueError('segment time interval is invalid')
        validate_tuple_of_nonnegative_ints(
            self.observation_indices, 'segment observation indices')


@dataclass(frozen=True, slots=True)
class Person_Ankle_Transition_Check:
    source_segment_id: int
    target_segment_id: int
    source_hypothesis_id: int
    target_hypothesis_id: int
    overlap_frame_indices_original: tuple[int, ...]
    endpoint_time_gap_in_second: float
    endpoint_distance_in_meter: float
    uses_cross_foot_distance: bool
    intervening_local_cluster_ids: tuple[int, ...]
    outcome: Person_Ankle_Transition_Outcome
    rejection_reason: str | None

    def __post_init__(self) -> None:
        for value, label in (
                (self.source_segment_id, 'transition source segment id'),
                (self.target_segment_id, 'transition target segment id'),
                (self.source_hypothesis_id, 'transition source hypothesis id'),
                (self.target_hypothesis_id, 'transition target hypothesis id'),
            ):
            validate_positive_int(value, label, 0)
        validate_tuple_of_nonnegative_ints(
            self.overlap_frame_indices_original,
            'transition overlap frames',
            allow_empty=True,
        )
        validate_nonnegative_float(
            self.endpoint_time_gap_in_second, 'transition endpoint time gap')
        validate_nonnegative_float(
            self.endpoint_distance_in_meter, 'transition endpoint distance')
        if type(self.uses_cross_foot_distance) is not bool:
            raise ValueError('transition cross-foot flag must be exact bool')
        validate_tuple_of_nonnegative_ints(
            self.intervening_local_cluster_ids,
            'transition intervening cluster ids',
            allow_empty=True,
        )
        if self.outcome not in ('valid', 'invalid', 'vetoed'):
            raise ValueError('transition outcome is invalid')
        if self.outcome == 'valid' and self.rejection_reason is not None:
            raise ValueError('valid transition must not have a rejection reason')
        if self.outcome != 'valid' and (
                type(self.rejection_reason) is not str or not self.rejection_reason
            ):
            raise ValueError('rejected transition requires one reason')


def validate_contiguous_ids(values: set[int], count: int, label: str) -> None:
    if values != set(range(count)):
        raise ValueError('%s must be contiguous from zero' % label)


def validate_result_evidence_graph(
        observations: Person_Ankle_Observation_Table,
        local_clusters: tuple[Person_Ankle_Local_Cluster, ...],
        hypotheses: tuple[Person_Ankle_Height_Hypothesis, ...],
        segments: tuple[Person_Ankle_Hypothesis_Segment, ...],
        transition_checks: tuple[Person_Ankle_Transition_Check, ...],
    ) -> None:
    observation_count = len(observations.observation_keys)
    cluster_by_id = {value.local_cluster_id: value for value in local_clusters}
    validate_contiguous_ids(
        set(cluster_by_id), len(local_clusters), 'result local cluster ids')
    membership = [-1] * observation_count
    for cluster in local_clusters:
        for observation_index in cluster.observation_indices:
            if observation_index >= observation_count:
                raise ValueError('local cluster observation index is out of range')
            if membership[observation_index] >= 0:
                raise ValueError('local cluster memberships must not overlap')
            if (
                    observations.run_keys[observation_index] != cluster.run_key
                    or observations.time_domain_keys[observation_index]
                    != cluster.time_domain_key
                ):
                raise ValueError('local cluster identity differs from its observations')
            membership[observation_index] = cluster.local_cluster_id
    if tuple(membership) != tuple(int(value) for value in observations.local_cluster_labels):
        raise ValueError('observation labels differ from local cluster memberships')

    hypothesis_by_id = {value.hypothesis_id: value for value in hypotheses}
    validate_contiguous_ids(
        set(hypothesis_by_id), len(hypotheses), 'result hypothesis ids')
    cluster_to_hypothesis: dict[int, int] = {}
    for hypothesis in hypotheses:
        for cluster_id in hypothesis.local_cluster_ids:
            if cluster_id not in cluster_by_id:
                raise ValueError('hypothesis references a missing local cluster')
            if cluster_by_id[cluster_id].height_shape != 'band':
                raise ValueError('hypothesis may reference only band clusters')
            if cluster_id in cluster_to_hypothesis:
                raise ValueError('band cluster belongs to multiple hypotheses')
            cluster_to_hypothesis[cluster_id] = hypothesis.hypothesis_id
        episode_cluster_ids: list[int] = []
        episode_ids = {value.episode_id for value in hypothesis.episodes}
        validate_contiguous_ids(
            episode_ids, len(hypothesis.episodes), 'hypothesis episode ids')
        for episode in hypothesis.episodes:
            episode_cluster_ids.extend(episode.local_cluster_ids)
            if any(cluster_id not in hypothesis.local_cluster_ids
                   for cluster_id in episode.local_cluster_ids):
                raise ValueError('episode references a cluster outside its hypothesis')
            time_domains = {
                cluster_by_id[cluster_id].time_domain_key
                for cluster_id in episode.local_cluster_ids
            }
            if len(time_domains) != 1:
                raise ValueError('episode must stay within one time domain')
        if sorted(episode_cluster_ids) != sorted(hypothesis.local_cluster_ids):
            raise ValueError('hypothesis episodes must partition its local clusters')
    expected_band_ids = {
        value.local_cluster_id for value in local_clusters
        if value.height_shape == 'band'
    }
    if set(cluster_to_hypothesis) != expected_band_ids:
        raise ValueError('result hypotheses must partition all band clusters')

    segment_by_id = {value.segment_id: value for value in segments}
    validate_contiguous_ids(
        set(segment_by_id), len(segments), 'result segment ids')
    for segment in segments:
        if segment.run_key not in set(observations.run_keys):
            raise ValueError('segment references a missing run')
        if segment.hypothesis_id not in hypothesis_by_id:
            raise ValueError('segment references a missing hypothesis')
        for observation_index in segment.observation_indices:
            if observation_index >= observation_count:
                raise ValueError('segment observation index is out of range')
            cluster_id = int(observations.local_cluster_labels[observation_index])
            if (
                    observations.run_keys[observation_index] != segment.run_key
                    or cluster_to_hypothesis.get(cluster_id) != segment.hypothesis_id
                ):
                raise ValueError('segment observation identity is inconsistent')

    seen_transition_pairs: set[tuple[int, int]] = set()
    for check in transition_checks:
        pair = (check.source_segment_id, check.target_segment_id)
        if pair in seen_transition_pairs:
            raise ValueError('transition segment pair must be unique')
        seen_transition_pairs.add(pair)
        if pair[0] not in segment_by_id or pair[1] not in segment_by_id:
            raise ValueError('transition references a missing segment')
        source = segment_by_id[pair[0]]
        target = segment_by_id[pair[1]]
        if (
                source.hypothesis_id != check.source_hypothesis_id
                or target.hypothesis_id != check.target_hypothesis_id
                or source.run_key != target.run_key
                or source.last_time_in_second > target.first_time_in_second
            ):
            raise ValueError('transition segment identity is inconsistent')
        if any(cluster_id not in cluster_by_id
               for cluster_id in check.intervening_local_cluster_ids):
            raise ValueError('transition references a missing local cluster')


@dataclass(frozen=True, slots=True)
class Person_Ankle_Plane_Result:
    person_key: str
    config: Person_Ankle_Plane_Config
    runs: tuple[Person_Ankle_Run_Provenance, ...]
    source_planes: tuple[Person_Ankle_Source_Plane, ...]
    reference_plane_key: str
    reference_ground_param_world: ARRAY_F
    reference_body_side_sign: int
    observations: Person_Ankle_Observation_Table
    local_clusters: tuple[Person_Ankle_Local_Cluster, ...]
    hypotheses: tuple[Person_Ankle_Height_Hypothesis, ...]
    segments: tuple[Person_Ankle_Hypothesis_Segment, ...]
    transition_checks: tuple[Person_Ankle_Transition_Check, ...]
    status: Person_Ankle_Plane_Status
    candidate_height_in_meter: float | None
    incompatible_plane_key: str | None
    incompatible_plane_angle_in_radian: float | None
    incompatible_plane_residual_in_meter: float | None

    @property
    def input_run_count(self) -> int:
        return len(self.runs)

    @property
    def input_source_plane_count(self) -> int:
        return len(self.source_planes)

    @property
    def input_frame_count(self) -> int:
        return sum(run.frame_count for run in self.runs)

    @property
    def input_observation_count(self) -> int:
        return len(self.observations.observation_keys)

    def __post_init__(self) -> None:
        validate_nonempty_string(self.person_key, 'result person key')
        if type(self.config) is not Person_Ankle_Plane_Config:
            raise ValueError('result config has wrong type')
        if type(self.runs) is not tuple or not self.runs or any(
                type(value) is not Person_Ankle_Run_Provenance for value in self.runs):
            raise ValueError('result runs must be a non-empty provenance tuple')
        if type(self.source_planes) is not tuple or not self.source_planes or any(
                type(value) is not Person_Ankle_Source_Plane
                for value in self.source_planes):
            raise ValueError('result source planes must be a non-empty source tuple')
        if len({value.run_key for value in self.runs}) != len(self.runs):
            raise ValueError('result run provenance keys must be unique')
        if len({value.plane_key for value in self.source_planes}) != len(self.source_planes):
            raise ValueError('result source plane keys must be unique')
        if {value.plane_key for value in self.runs} != {
                value.plane_key for value in self.source_planes}:
            raise ValueError('result run-to-plane provenance is incomplete')
        validate_nonempty_string(self.reference_plane_key, 'result reference plane key')
        reference = readonly_float_array(
            self.reference_ground_param_world, (4,), 'result reference plane')
        validate_nonzero_plane_normal(reference, 'result reference plane')
        if type(self.reference_body_side_sign) is not int or self.reference_body_side_sign not in (-1, 1):
            raise ValueError('result body side sign must be exact int -1 or +1')
        valid_statuses = (
            'single_support_plane', 'plane_switch', 'multi_layer_ambiguous',
            'local_episode_ambiguous', 'no_ground_evidence',
            'incompatible_reference_planes',
        )
        if self.status not in valid_statuses:
            raise ValueError('result status is invalid')
        if self.status == 'single_support_plane':
            if (
                    type(self.candidate_height_in_meter) is not float
                    or not math.isfinite(self.candidate_height_in_meter)
                ):
                raise ValueError('single-support result requires one finite float candidate')
        elif self.candidate_height_in_meter is not None:
            raise ValueError('only single-support result may carry a candidate')
        incompatibility_values = (
            self.incompatible_plane_key,
            self.incompatible_plane_angle_in_radian,
            self.incompatible_plane_residual_in_meter,
        )
        if self.status == 'incompatible_reference_planes':
            if (
                    type(self.incompatible_plane_key) is not str
                    or not self.incompatible_plane_key
                    or type(self.incompatible_plane_angle_in_radian) is not float
                    or not math.isfinite(self.incompatible_plane_angle_in_radian)
                    or self.incompatible_plane_angle_in_radian < 0.0
                    or type(self.incompatible_plane_residual_in_meter) is not float
                    or not math.isfinite(self.incompatible_plane_residual_in_meter)
                    or self.incompatible_plane_residual_in_meter < 0.0
                ):
                raise ValueError('incompatible result requires complete finite detail')
        elif any(value is not None for value in incompatibility_values):
            raise ValueError('compatible result must not carry incompatibility detail')
        if self.status == 'incompatible_reference_planes' and (
                self.local_clusters or self.hypotheses or self.segments
                or self.transition_checks
            ):
            raise ValueError('incompatible result must not carry inferred evidence')
        evidence_columns = (
            (self.local_clusters, Person_Ankle_Local_Cluster, 'local clusters'),
            (self.hypotheses, Person_Ankle_Height_Hypothesis, 'hypotheses'),
            (self.segments, Person_Ankle_Hypothesis_Segment, 'segments'),
            (self.transition_checks, Person_Ankle_Transition_Check, 'transition checks'),
        )
        for column, expected_type, label in evidence_columns:
            if type(column) is not tuple or any(
                    type(value) is not expected_type for value in column):
                raise ValueError('result %s must be a tuple of exact record types' % label)
        if type(self.observations) is not Person_Ankle_Observation_Table:
            raise ValueError('result observations have wrong type')
        if self.input_observation_count != 2 * self.input_frame_count:
            raise ValueError('result observation count must be twice the input frame count')
        canonical_source = self.source_planes[0]
        if (
                self.reference_plane_key != canonical_source.plane_key
                or self.reference_body_side_sign != canonical_source.body_side_sign
                or not bool(np.array_equal(
                    self.reference_ground_param_world,
                    canonical_source.ground_param_world,
                ))
            ):
            raise ValueError('result reference must equal the canonical source plane')
        run_by_key = {run.run_key: run for run in self.runs}
        if set(self.observations.run_keys) != set(run_by_key):
            raise ValueError('result observation run keys differ from provenance')
        for run_key, run in run_by_key.items():
            observation_indices = [
                index for index, value in enumerate(self.observations.run_keys)
                if value == run_key
            ]
            if len(observation_indices) != 2 * run.frame_count:
                raise ValueError('result per-run observation count differs from provenance')
            time_domains = {
                self.observations.time_domain_keys[index]
                for index in observation_indices
            }
            if time_domains != {run.time_domain_key}:
                raise ValueError('result per-run time domain differs from provenance')
            frame_indices = {
                int(self.observations.frame_indices_original[index])
                for index in observation_indices
            }
            if (
                    len(frame_indices) != run.frame_count
                    or min(frame_indices) != run.first_frame_index_original
                    or max(frame_indices) != run.last_frame_index_original
                ):
                raise ValueError('result per-run frame provenance is inconsistent')
            if any(
                    float(self.observations.times_in_second[index])
                    != int(self.observations.frame_indices_original[index]) / run.fps
                    for index in observation_indices
                ):
                raise ValueError('result per-run times differ from frame indices and FPS')
        validate_result_evidence_graph(
            self.observations,
            self.local_clusters,
            self.hypotheses,
            self.segments,
            self.transition_checks,
        )
        recurrent = tuple(
            hypothesis for hypothesis in self.hypotheses if hypothesis.recurrent)
        if self.status == 'single_support_plane':
            if len(recurrent) != 1:
                raise ValueError('single-support result requires one recurrent hypothesis')
            if any(hypothesis.materially_supported_competitor
                   for hypothesis in self.hypotheses
                   if hypothesis.hypothesis_id != recurrent[0].hypothesis_id):
                raise ValueError('single-support result cannot have a material competitor')
            if self.candidate_height_in_meter != recurrent[0].center_height_in_meter:
                raise ValueError('single-support candidate must equal recurrent center')
        if self.status == 'plane_switch' and len(recurrent) < 2:
            raise ValueError('plane-switch result requires multiple recurrent hypotheses')
        if self.status == 'plane_switch' and not any(
                value.outcome == 'valid' for value in self.transition_checks):
            raise ValueError('plane-switch result requires a valid transition')
        if self.status == 'multi_layer_ambiguous' and not (
                len(recurrent) >= 2
                or (
                    len(recurrent) == 1
                    and any(value.materially_supported_competitor
                            for value in self.hypotheses)
                )
            ):
            raise ValueError('multi-layer result requires recurrent competing evidence')
        if self.status == 'local_episode_ambiguous' and (
                not self.hypotheses or recurrent):
            raise ValueError('local-episode result requires only non-recurrent hypotheses')
        if self.status == 'no_ground_evidence' and self.hypotheses:
            raise ValueError('no-ground result must not carry hypotheses')
        object.__setattr__(self, 'reference_ground_param_world', reference)
