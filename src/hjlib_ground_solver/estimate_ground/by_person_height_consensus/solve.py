'''Two-level robust person-height consensus for a fixed Ground Normal.'''

import math

import numpy as np

from hjlib_ground_solver.estimate_ground.by_person_height_consensus.contract import (
    Bool_Array,
    Float_Array,
    Person_Height_Consensus_Config,
    Person_Height_Consensus_Observations,
    Person_Height_Consensus_Result,
    validated_camera_up_normal,
    validated_intrinsics_matrix,
)
from hjlib_ground_solver.estimate_ground.by_person_height_consensus.ipose import (
    compute_power8_ipose_measurements,
)


def trimmed_mean_with_mask(
        values: Float_Array,
        trim_fraction_each_tail: float,
    ) -> tuple[float, Bool_Array]:
    if values.ndim != 1 or values.size < 1 or not bool(np.isfinite(values).all()):
        raise ValueError('trimmed-mean values must be a nonempty finite vector')
    trim_count = int(math.floor(trim_fraction_each_tail * values.size))
    order = np.argsort(values, kind='stable')
    retained_order = (
        order[trim_count:values.size - trim_count]
        if trim_count > 0
        else order
    )
    mask = np.zeros(values.size, dtype=np.bool_)
    mask[retained_order] = True
    mask.setflags(write=False)
    return float(np.mean(values[mask])), mask


def implied_height_over_offset(
        bottoms_xy_px: Float_Array,
        equivalent_height_px: Float_Array,
        camera_intrinsics: Float_Array,
        ground_normal_camera: Float_Array,
        geometry_epsilon: float,
    ) -> tuple[Float_Array, Bool_Array]:
    count = int(equivalent_height_px.shape[0])
    homogeneous = np.column_stack((bottoms_xy_px, np.ones(count, dtype=np.float64)))
    rays = np.linalg.solve(camera_intrinsics, homogeneous.T).T
    ray_normal_dot = rays @ ground_normal_camera
    bottom_scale = np.divide(
        -1.0,
        ray_normal_dot,
        out=np.full(count, np.nan, dtype=np.float64),
        where=np.abs(ray_normal_dot) > geometry_epsilon,
    )
    vanishing = camera_intrinsics @ ground_normal_camera
    projected_vertical_length = np.linalg.norm(
        vanishing[:2][None, :] - bottoms_xy_px * vanishing[2],
        axis=1,
    )
    denominator = projected_vertical_length - equivalent_height_px * vanishing[2]
    ratios = np.divide(
        equivalent_height_px * bottom_scale,
        denominator,
        out=np.full(count, np.nan, dtype=np.float64),
        where=np.abs(denominator) > geometry_epsilon,
    )
    virtual_top_depth = bottom_scale * rays[:, 2] + ratios * ground_normal_camera[2]
    valid = (
        (ray_normal_dot < -geometry_epsilon)
        & (projected_vertical_length > geometry_epsilon)
        & (denominator > geometry_epsilon)
        & np.isfinite(ratios)
        & (ratios > 0.0)
        & np.isfinite(virtual_top_depth)
        & (virtual_top_depth > geometry_epsilon)
    )
    ratios[~valid] = np.nan
    ratios.setflags(write=False)
    valid.setflags(write=False)
    return ratios, valid


def solve_ground_offset_by_person_height_consensus(
        observations: Person_Height_Consensus_Observations,
        ground_normal_camera: Float_Array,
        camera_intrinsics: Float_Array,
        retained_person_trimmed_mean_equivalent_height_m: float,
        config: Person_Height_Consensus_Config = Person_Height_Consensus_Config(),
    ) -> Person_Height_Consensus_Result:
    '''Solve scene offset from per-person temporal effective-height consensus.'''
    if type(observations) is not Person_Height_Consensus_Observations:
        raise TypeError('observations must be Person_Height_Consensus_Observations')
    if type(config) is not Person_Height_Consensus_Config:
        raise TypeError('config must be Person_Height_Consensus_Config')
    mean_height = retained_person_trimmed_mean_equivalent_height_m
    if type(mean_height) is not float or not math.isfinite(mean_height) or mean_height <= 0.0:
        raise ValueError('retained-person mean equivalent height must be positive')
    matrix = validated_intrinsics_matrix(camera_intrinsics)
    normal = validated_camera_up_normal(ground_normal_camera)
    measurements = compute_power8_ipose_measurements(
        observations,
        matrix,
        normal,
        geometry_epsilon=config.geometry_epsilon,
    )
    observation_ratios, observation_valid = implied_height_over_offset(
        measurements.corrected_bottom_xy_px,
        measurements.equivalent_height_px,
        matrix,
        normal,
        config.geometry_epsilon,
    )
    observation_valid = np.asarray(
        observation_valid & measurements.measurement_valid_mask,
        dtype=np.bool_,
    )
    observation_ratios = np.array(observation_ratios, dtype=np.float64, copy=True)
    observation_ratios[~observation_valid] = np.nan
    observation_ratios.setflags(write=False)
    observation_valid.setflags(write=False)
    person_ids = np.unique(observations.person_ids)
    person_count = int(person_ids.size)
    valid_counts = np.zeros(person_count, dtype=np.int64)
    eligible = np.zeros(person_count, dtype=np.bool_)
    person_ratios = np.full(person_count, np.nan, dtype=np.float64)
    for person_index, person_id in enumerate(person_ids):
        rows = (observations.person_ids == person_id) & observation_valid
        valid_counts[person_index] = int(np.count_nonzero(rows))
        if valid_counts[person_index] < config.minimum_valid_frames_per_person:
            continue
        values = observation_ratios[rows]
        person_ratios[person_index], unused_mask = trimmed_mean_with_mask(
            values,
            config.trim_fraction_each_tail,
        )
        del unused_mask
        eligible[person_index] = True
    if int(np.count_nonzero(eligible)) < config.minimum_retained_person_count:
        raise ValueError('insufficient retained people for scene consensus')
    scene_ratio, eligible_scene_mask = trimmed_mean_with_mask(
        person_ratios[eligible],
        config.trim_fraction_each_tail,
    )
    scene_retained = np.zeros(person_count, dtype=np.bool_)
    scene_retained[np.flatnonzero(eligible)[eligible_scene_mask]] = True
    if int(np.count_nonzero(scene_retained)) < config.minimum_retained_person_count:
        raise ValueError('insufficient people after scene trimming')
    distance = mean_height / scene_ratio
    if not math.isfinite(distance) or distance <= 0.0:
        raise ValueError('solved ground offset must be finite and positive')
    person_heights = distance * person_ratios
    plane = np.concatenate((normal, np.array([distance], dtype=np.float64)))
    return Person_Height_Consensus_Result(
        observations=observations,
        config=config,
        measurements=measurements,
        observation_height_over_offset=observation_ratios,
        observation_valid_mask=observation_valid,
        person_ids=np.asarray(person_ids, dtype=np.int64),
        person_valid_frame_counts=valid_counts,
        person_eligible_mask=eligible,
        person_height_over_offset=person_ratios,
        scene_person_retained_mask=scene_retained,
        scene_height_over_offset=scene_ratio,
        retained_person_trimmed_mean_equivalent_height_m=mean_height,
        person_equivalent_height_m=person_heights,
        plane_camera_abcd=plane,
    )


__all__ = ['solve_ground_offset_by_person_height_consensus']
