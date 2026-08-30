'''Input normalization retained for the temporarily deprecated ankle-plane V1.'''

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

from hjlib_geometry import orthogonally_project_points_to_plane

from hjlib_ground_solver.estimate_ground.person_ankle_plane.contract import (
    ARRAY_F,
    ARRAY_I,
    Person_Ankle_Foot_Side,
    Person_Ankle_Plane_Config,
    Person_Ankle_Plane_Input,
    Person_Ankle_Run,
    Person_Ankle_Source_Plane,
)


@dataclass(frozen=True, slots=True)
class Normalized_Ankle_Observation_Table:
    observation_keys: tuple[str, ...]
    run_keys: tuple[str, ...]
    time_domain_keys: tuple[str, ...]
    frame_indices_original: ARRAY_I
    foot_sides: tuple[Person_Ankle_Foot_Side, ...]
    times_in_second: ARRAY_F
    world_points_in_meter: ARRAY_F
    projected_points_in_meter: ARRAY_F
    heights_in_meter: ARRAY_F


@dataclass(frozen=True, slots=True)
class Normalized_Person_Ankle_Input:
    person_key: str
    runs: tuple[Person_Ankle_Run, ...]
    source_planes: tuple[Person_Ankle_Source_Plane, ...]
    reference_plane_key: str
    reference_ground_param_world: ARRAY_F
    reference_body_side_sign: int
    observations: Normalized_Ankle_Observation_Table
    incompatible_plane_key: str | None
    incompatible_plane_angle_in_radian: float | None
    incompatible_plane_residual_in_meter: float | None


def make_readonly_float_array(value: NDArray[np.float64]) -> ARRAY_F:
    output = np.array(value, dtype=np.float64, order='C', copy=True)
    output.setflags(write=False)
    return output


def make_readonly_int_array(value: NDArray[np.int64]) -> ARRAY_I:
    output = np.array(value, dtype=np.int64, order='C', copy=True)
    output.setflags(write=False)
    return output


def oriented_unit_plane(source: Person_Ankle_Source_Plane) -> ARRAY_F:
    normal = source.ground_param_world[:3]
    normal_scale = float(np.max(np.abs(normal)))
    scaled_normal = normal / normal_scale
    scaled_norm = float(np.linalg.norm(scaled_normal))
    unit = np.empty(4, dtype=np.float64)
    unit[:3] = source.body_side_sign * scaled_normal / scaled_norm
    unit[3] = (
        source.body_side_sign
        * (float(source.ground_param_world[3]) / normal_scale)
        / scaled_norm
    )
    if not bool(np.isfinite(unit).all()):
        raise ValueError('oriented ground plane normalization produced nonfinite values')
    unit.setflags(write=False)
    return unit


def validate_person_identities(
        person_input: Person_Ankle_Plane_Input,
    ) -> tuple[
        tuple[Person_Ankle_Run, ...],
        tuple[Person_Ankle_Source_Plane, ...],
    ]:
    runs = tuple(sorted(person_input.runs, key=lambda value: (
        value.time_domain_key,
        int(value.frame_indices_original[0]) / value.fps,
        int(value.frame_indices_original[-1]) / value.fps,
        value.run_key,
    )))
    planes = tuple(sorted(person_input.source_planes, key=lambda value: value.plane_key))
    run_keys = [run.run_key for run in runs]
    plane_keys = [plane.plane_key for plane in planes]
    if len(set(run_keys)) != len(run_keys):
        raise ValueError('run keys must be unique within one person')
    if len(set(plane_keys)) != len(plane_keys):
        raise ValueError('source plane keys must be unique within one person')
    plane_key_set = set(plane_keys)
    referenced_plane_keys = {run.plane_key for run in runs}
    missing = sorted(referenced_plane_keys - plane_key_set)
    if missing:
        raise ValueError('run references missing source plane key: %s' % missing[0])
    unreferenced = sorted(plane_key_set - referenced_plane_keys)
    if unreferenced:
        raise ValueError('source plane is not referenced by any run: %s' % unreferenced[0])
    observation_keys = [
        key
        for run in runs
        for row in run.observation_keys
        for key in row
    ]
    if len(set(observation_keys)) != len(observation_keys):
        raise ValueError('observation keys must be unique within one person')
    return runs, planes


def flatten_observations(
        runs: tuple[Person_Ankle_Run, ...],
        reference_plane: ARRAY_F,
        reference_body_side_sign: int,
    ) -> Normalized_Ankle_Observation_Table:
    observation_keys: list[str] = []
    run_keys: list[str] = []
    time_domain_keys: list[str] = []
    frame_indices: list[int] = []
    foot_sides: list[Person_Ankle_Foot_Side] = []
    times: list[float] = []
    world_rows: list[NDArray[np.float64]] = []
    for run in runs:
        for frame_offset, frame_index in enumerate(run.frame_indices_original):
            time_in_second = int(frame_index) / run.fps
            foot_rows: tuple[tuple[int, Person_Ankle_Foot_Side], ...] = (
                (0, 'left'), (1, 'right'))
            for foot_index, foot_side in foot_rows:
                observation_keys.append(run.observation_keys[frame_offset][foot_index])
                run_keys.append(run.run_key)
                time_domain_keys.append(run.time_domain_key)
                frame_indices.append(int(frame_index))
                foot_sides.append(foot_side)
                times.append(time_in_second)
                world_rows.append(run.ankle_world_in_meter[frame_offset, foot_index])
    world_points = np.asarray(world_rows, dtype=np.float64)
    projected_points = orthogonally_project_points_to_plane(
        world_points, reference_plane)
    reference_normal = reference_plane[:3]
    normal_scale = float(np.max(np.abs(reference_normal)))
    scaled_normal = reference_normal / normal_scale
    oriented_normal = (
        reference_body_side_sign
        * scaled_normal
        / np.linalg.norm(scaled_normal)
    )
    heights = np.sum(
        (world_points - projected_points) * oriented_normal[None, :], axis=1)
    return Normalized_Ankle_Observation_Table(
        observation_keys=tuple(observation_keys),
        run_keys=tuple(run_keys),
        time_domain_keys=tuple(time_domain_keys),
        frame_indices_original=make_readonly_int_array(
            np.asarray(frame_indices, dtype=np.int64)),
        foot_sides=tuple(foot_sides),
        times_in_second=make_readonly_float_array(
            np.asarray(times, dtype=np.float64)),
        world_points_in_meter=make_readonly_float_array(world_points),
        projected_points_in_meter=make_readonly_float_array(projected_points),
        heights_in_meter=make_readonly_float_array(heights),
    )


def normalize_person_ankle_input(
        person_input: Person_Ankle_Plane_Input,
        config: Person_Ankle_Plane_Config,
    ) -> Normalized_Person_Ankle_Input:
    '''Validate identities, choose one plane, and normalize every observation.'''
    if type(person_input) is not Person_Ankle_Plane_Input:
        raise ValueError('person input has wrong type')
    if type(config) is not Person_Ankle_Plane_Config:
        raise ValueError('person ankle-plane config has wrong type')
    runs, planes = validate_person_identities(person_input)
    reference_source = planes[0]
    reference_unit = oriented_unit_plane(reference_source)
    observations = flatten_observations(
        runs,
        reference_source.ground_param_world,
        reference_source.body_side_sign,
    )
    incompatible_key: str | None = None
    incompatible_angle: float | None = None
    incompatible_residual: float | None = None
    for source in planes[1:]:
        unit = oriented_unit_plane(source)
        angle = math.acos(float(np.clip(
            np.dot(reference_unit[:3], unit[:3]), -1.0, 1.0)))
        residuals = np.abs(
            observations.world_points_in_meter @ (unit[:3] - reference_unit[:3])
            + unit[3]
            - reference_unit[3]
        )
        residual = float(np.max(residuals))
        if (
                angle > config.reference_normal_angle_max_in_radian
                or residual > config.reference_residual_max_in_meter
            ):
            incompatible_key = source.plane_key
            incompatible_angle = angle
            incompatible_residual = residual
            break
    return Normalized_Person_Ankle_Input(
        person_key=person_input.person_key,
        runs=runs,
        source_planes=planes,
        reference_plane_key=reference_source.plane_key,
        reference_ground_param_world=reference_source.ground_param_world,
        reference_body_side_sign=reference_source.body_side_sign,
        observations=observations,
        incompatible_plane_key=incompatible_key,
        incompatible_plane_angle_in_radian=incompatible_angle,
        incompatible_plane_residual_in_meter=incompatible_residual,
    )


__all__ = [
    'Normalized_Ankle_Observation_Table',
    'Normalized_Person_Ankle_Input',
    'normalize_person_ankle_input',
]
