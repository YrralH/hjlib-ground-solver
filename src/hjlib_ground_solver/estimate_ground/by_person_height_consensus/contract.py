'''Contracts for identity-aware equivalent-height ground-offset solving.'''

from dataclasses import dataclass
import math
from typing import cast

import numpy as np
from numpy.typing import NDArray


type Float_Array = NDArray[np.float64]
type Int_Array = NDArray[np.int64]
type Bool_Array = NDArray[np.bool_]


def readonly_float_array(
        value: object,
        shape: tuple[int, ...],
        name: str,
        *,
        allow_nan: bool = False,
    ) -> Float_Array:
    if not isinstance(value, np.ndarray):
        raise TypeError('%s must be a numpy array' % name)
    array = np.asarray(cast(NDArray[np.generic], value))
    if array.dtype != np.dtype(np.float64) or array.shape != shape:
        raise ValueError('%s must have shape %r and dtype float64' % (name, shape))
    if allow_nan:
        if bool(np.any(np.isinf(array))):
            raise ValueError('%s must not contain infinity' % name)
    elif not bool(np.isfinite(array).all()):
        raise ValueError('%s must contain only finite values' % name)
    output = np.array(array, dtype=np.float64, order='C', copy=True)
    output.setflags(write=False)
    return output


def readonly_int_array(value: object, shape: tuple[int, ...], name: str) -> Int_Array:
    if not isinstance(value, np.ndarray):
        raise TypeError('%s must be a numpy array' % name)
    array = np.asarray(cast(NDArray[np.generic], value))
    if array.dtype != np.dtype(np.int64) or array.shape != shape:
        raise ValueError('%s must have shape %r and dtype int64' % (name, shape))
    output = np.array(array, dtype=np.int64, order='C', copy=True)
    output.setflags(write=False)
    return output


def readonly_bool_array(value: object, shape: tuple[int, ...], name: str) -> Bool_Array:
    if not isinstance(value, np.ndarray):
        raise TypeError('%s must be a numpy array' % name)
    array = np.asarray(cast(NDArray[np.generic], value))
    if array.dtype != np.dtype(np.bool_) or array.shape != shape:
        raise ValueError('%s must have shape %r and dtype bool' % (name, shape))
    output = np.array(array, dtype=np.bool_, order='C', copy=True)
    output.setflags(write=False)
    return output


def validated_intrinsics_matrix(value: object) -> Float_Array:
    matrix = readonly_float_array(value, (3, 3), 'camera_intrinsics')
    determinant = float(np.linalg.det(matrix))
    if not math.isfinite(determinant) or abs(determinant) <= 1e-12:
        raise ValueError('camera_intrinsics must be nonsingular')
    if not np.allclose(matrix[2], np.array([0.0, 0.0, 1.0]), rtol=0.0, atol=1e-12):
        raise ValueError('camera_intrinsics must use the standard pixel projection row')
    return matrix


def validated_camera_up_normal(value: object) -> Float_Array:
    normal = readonly_float_array(value, (3,), 'ground_normal_camera')
    if not math.isclose(float(np.linalg.norm(normal)), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError('ground_normal_camera must be unit length within 1e-12')
    if normal[1] >= 0.0:
        raise ValueError('ground_normal_camera must use camera-up orientation')
    return normal


@dataclass(frozen=True, slots=True)
class Person_Height_Consensus_Observations:
    '''Named bilateral joints on one canonical person/frame observation axis.'''

    person_ids: Int_Array
    frame_indices: Int_Array
    shoulder_xy_px: Float_Array
    hip_xy_px: Float_Array
    knee_xy_px: Float_Array
    ankle_xy_px: Float_Array

    def __post_init__(self) -> None:
        if self.person_ids.ndim != 1:
            raise ValueError('person_ids must have shape (N,)')
        count = int(self.person_ids.shape[0])
        if count < 1:
            raise ValueError('at least one observation is required')
        persons = readonly_int_array(self.person_ids, (count,), 'person_ids')
        frames = readonly_int_array(self.frame_indices, (count,), 'frame_indices')
        if bool(np.any(persons < 0)) or bool(np.any(frames < 0)):
            raise ValueError('person and frame identities must be nonnegative')
        order = np.lexsort((frames, persons))
        if not np.array_equal(order, np.arange(count, dtype=np.int64)):
            raise ValueError('observations must use ascending person/frame order')
        pairs = np.column_stack((persons, frames))
        if count > 1 and bool(np.any(np.all(pairs[1:] == pairs[:-1], axis=1))):
            raise ValueError('person/frame pairs must be unique')
        object.__setattr__(self, 'person_ids', persons)
        object.__setattr__(self, 'frame_indices', frames)
        for name in ('shoulder_xy_px', 'hip_xy_px', 'knee_xy_px', 'ankle_xy_px'):
            value = readonly_float_array(getattr(self, name), (count, 2, 2), name)
            object.__setattr__(self, name, value)

    @property
    def count(self) -> int:
        return int(self.person_ids.shape[0])


@dataclass(frozen=True, slots=True)
class Person_Height_Consensus_Config:
    '''Robust two-level aggregation policy.'''

    trim_fraction_each_tail: float = 0.05
    minimum_valid_frames_per_person: int = 1
    minimum_retained_person_count: int = 1
    geometry_epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if (
                type(self.trim_fraction_each_tail) is not float
                or not math.isfinite(self.trim_fraction_each_tail)
                or not 0.0 <= self.trim_fraction_each_tail < 0.5
            ):
            raise ValueError('trim_fraction_each_tail must be in [0, 0.5)')
        for value, name in (
                (self.minimum_valid_frames_per_person, 'minimum_valid_frames_per_person'),
                (self.minimum_retained_person_count, 'minimum_retained_person_count'),
            ):
            if type(value) is not int or value < 1:
                raise ValueError('%s must be a positive int' % name)
        if (
                type(self.geometry_epsilon) is not float
                or not math.isfinite(self.geometry_epsilon)
                or self.geometry_epsilon <= 0.0
            ):
            raise ValueError('geometry_epsilon must be a finite positive float')


@dataclass(frozen=True, slots=True)
class Power8_IPose_Measurements:
    '''Effective 2D height and corrected bottom for every input observation.'''

    shoulder_midpoint_xy_px: Float_Array
    vertical_direction_xy: Float_Array
    corrected_bottom_xy_px: Float_Array
    equivalent_height_px: Float_Array
    measurement_valid_mask: Bool_Array

    def __post_init__(self) -> None:
        if self.equivalent_height_px.ndim != 1:
            raise ValueError('equivalent_height_px must have shape (N,)')
        count = int(self.equivalent_height_px.shape[0])
        valid = readonly_bool_array(
            self.measurement_valid_mask, (count,), 'measurement_valid_mask')
        height = readonly_float_array(
            self.equivalent_height_px,
            (count,),
            'equivalent_height_px',
            allow_nan=True,
        )
        if (
                not bool(np.isfinite(height[valid]).all())
                or bool(np.any(height[valid] <= 0.0))
                or not bool(np.isnan(height[~valid]).all())
            ):
            raise ValueError('equivalent_height_px must be positive exactly where valid')
        object.__setattr__(self, 'equivalent_height_px', height)
        object.__setattr__(self, 'measurement_valid_mask', valid)
        shoulders = readonly_float_array(
            self.shoulder_midpoint_xy_px, (count, 2), 'shoulder_midpoint_xy_px')
        object.__setattr__(self, 'shoulder_midpoint_xy_px', shoulders)
        for name in ('vertical_direction_xy', 'corrected_bottom_xy_px'):
            value = readonly_float_array(
                getattr(self, name), (count, 2), name, allow_nan=True)
            if (
                    not bool(np.isfinite(value[valid]).all())
                    or not bool(np.isnan(value[~valid]).all())
                ):
                raise ValueError('%s must be finite exactly where valid' % name)
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class Person_Height_Consensus_Result:
    '''One scene's implied-ratio consensus and metric ground-offset result.'''

    observations: Person_Height_Consensus_Observations
    config: Person_Height_Consensus_Config
    measurements: Power8_IPose_Measurements
    observation_height_over_offset: Float_Array
    observation_valid_mask: Bool_Array
    person_ids: Int_Array
    person_valid_frame_counts: Int_Array
    person_eligible_mask: Bool_Array
    person_height_over_offset: Float_Array
    scene_person_retained_mask: Bool_Array
    scene_height_over_offset: float
    retained_person_trimmed_mean_equivalent_height_m: float
    person_equivalent_height_m: Float_Array
    plane_camera_abcd: Float_Array

    def __post_init__(self) -> None:
        observation_count = self.observations.count
        person_count = int(self.person_ids.shape[0])
        if self.measurements.equivalent_height_px.shape != (observation_count,):
            raise ValueError('measurements must share the observation axis')
        observation_ratios = readonly_float_array(
            self.observation_height_over_offset,
            (observation_count,),
            'observation_height_over_offset',
            allow_nan=True,
        )
        observation_valid = readonly_bool_array(
            self.observation_valid_mask, (observation_count,), 'observation_valid_mask')
        persons = readonly_int_array(self.person_ids, (person_count,), 'result_person_ids')
        valid_counts = readonly_int_array(
            self.person_valid_frame_counts,
            (person_count,),
            'person_valid_frame_counts',
        )
        eligible = readonly_bool_array(
            self.person_eligible_mask, (person_count,), 'person_eligible_mask')
        person_ratios = readonly_float_array(
            self.person_height_over_offset,
            (person_count,),
            'person_height_over_offset',
            allow_nan=True,
        )
        scene_retained = readonly_bool_array(
            self.scene_person_retained_mask,
            (person_count,),
            'scene_person_retained_mask',
        )
        person_heights = readonly_float_array(
            self.person_equivalent_height_m,
            (person_count,),
            'person_equivalent_height_m',
            allow_nan=True,
        )
        object.__setattr__(self, 'observation_height_over_offset', observation_ratios)
        object.__setattr__(self, 'observation_valid_mask', observation_valid)
        object.__setattr__(self, 'person_ids', persons)
        object.__setattr__(self, 'person_valid_frame_counts', valid_counts)
        object.__setattr__(self, 'person_eligible_mask', eligible)
        object.__setattr__(self, 'person_height_over_offset', person_ratios)
        object.__setattr__(self, 'scene_person_retained_mask', scene_retained)
        object.__setattr__(self, 'person_equivalent_height_m', person_heights)
        plane = readonly_float_array(self.plane_camera_abcd, (4,), 'plane_camera_abcd')
        object.__setattr__(self, 'plane_camera_abcd', plane)
        for value, name in (
                (self.scene_height_over_offset, 'scene_height_over_offset'),
                (
                    self.retained_person_trimmed_mean_equivalent_height_m,
                    'retained_person_trimmed_mean_equivalent_height_m',
                ),
            ):
            if type(value) is not float or not math.isfinite(value) or value <= 0.0:
                raise ValueError('%s must be a finite positive float' % name)
        expected_persons = np.unique(self.observations.person_ids)
        expected_counts = np.array([
            np.count_nonzero(
                (self.observations.person_ids == person_id) & observation_valid)
            for person_id in expected_persons
        ], dtype=np.int64)
        expected_eligible = expected_counts >= self.config.minimum_valid_frames_per_person
        if (
                not np.array_equal(persons, expected_persons)
                or not np.array_equal(valid_counts, expected_counts)
                or not np.array_equal(eligible, expected_eligible)
            ):
            raise ValueError('person identities, support, and eligibility are inconsistent')
        if (
                not bool(np.isfinite(observation_ratios[observation_valid]).all())
                or bool(np.any(observation_ratios[observation_valid] <= 0.0))
                or not bool(np.isnan(observation_ratios[~observation_valid]).all())
                or not bool(np.all(observation_valid <= self.measurements.measurement_valid_mask))
            ):
            raise ValueError('observation ratios and validity mask are inconsistent')
        if (
                not bool(np.isfinite(person_ratios[eligible]).all())
                or bool(np.any(person_ratios[eligible] <= 0.0))
                or not bool(np.isnan(person_ratios[~eligible]).all())
                or not bool(np.isfinite(person_heights[eligible]).all())
                or bool(np.any(person_heights[eligible] <= 0.0))
                or not bool(np.isnan(person_heights[~eligible]).all())
                or bool(np.any(scene_retained & ~eligible))
                or int(np.count_nonzero(scene_retained))
                < self.config.minimum_retained_person_count
            ):
            raise ValueError('person ratios, heights, and retained mask are inconsistent')
        expected_distance = (
            self.retained_person_trimmed_mean_equivalent_height_m
            / self.scene_height_over_offset
        )
        if (
                not math.isclose(
                    float(plane[3]), expected_distance, rel_tol=1e-12, abs_tol=1e-12)
                or not np.allclose(
                    person_heights[eligible],
                    plane[3] * person_ratios[eligible],
                    rtol=1e-12,
                    atol=1e-12,
                )
            ):
            raise ValueError('metric scale is inconsistent with ratios and plane offset')
