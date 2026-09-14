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

    def __post_init__(self) -> None:
        if self.equivalent_height_px.ndim != 1:
            raise ValueError('equivalent_height_px must have shape (N,)')
        count = int(self.equivalent_height_px.shape[0])
        height = readonly_float_array(
            self.equivalent_height_px, (count,), 'equivalent_height_px')
        if bool(np.any(height <= 0.0)):
            raise ValueError('equivalent_height_px must be positive')
        object.__setattr__(self, 'equivalent_height_px', height)
        for name in (
                'shoulder_midpoint_xy_px',
                'vertical_direction_xy',
                'corrected_bottom_xy_px',
            ):
            value = readonly_float_array(getattr(self, name), (count, 2), name)
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
        object.__setattr__(self, 'observation_height_over_offset', readonly_float_array(
            self.observation_height_over_offset,
            (observation_count,),
            'observation_height_over_offset',
            allow_nan=True,
        ))
        object.__setattr__(self, 'observation_valid_mask', readonly_bool_array(
            self.observation_valid_mask, (observation_count,), 'observation_valid_mask'))
        object.__setattr__(self, 'person_ids', readonly_int_array(
            self.person_ids, (person_count,), 'result_person_ids'))
        object.__setattr__(self, 'person_valid_frame_counts', readonly_int_array(
            self.person_valid_frame_counts,
            (person_count,),
            'person_valid_frame_counts',
        ))
        object.__setattr__(self, 'person_eligible_mask', readonly_bool_array(
            self.person_eligible_mask, (person_count,), 'person_eligible_mask'))
        object.__setattr__(self, 'person_height_over_offset', readonly_float_array(
            self.person_height_over_offset,
            (person_count,),
            'person_height_over_offset',
            allow_nan=True,
        ))
        object.__setattr__(self, 'scene_person_retained_mask', readonly_bool_array(
            self.scene_person_retained_mask,
            (person_count,),
            'scene_person_retained_mask',
        ))
        object.__setattr__(self, 'person_equivalent_height_m', readonly_float_array(
            self.person_equivalent_height_m,
            (person_count,),
            'person_equivalent_height_m',
            allow_nan=True,
        ))
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
