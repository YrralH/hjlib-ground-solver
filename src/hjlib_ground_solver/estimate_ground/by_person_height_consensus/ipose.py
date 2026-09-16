'''Power-8 I-Pose measurements under a supplied projective vertical.'''

import math

import numpy as np

from hjlib_ground_solver.estimate_ground.by_person_height_consensus.contract import (
    Float_Array,
    Person_Height_Consensus_Observations,
    Power8_IPose_Measurements,
    validated_camera_up_normal,
    validated_intrinsics_matrix,
)


def stable_power8_mean(left: Float_Array, right: Float_Array) -> Float_Array:
    scale = np.maximum(left, right)
    normalized_left = np.divide(
        left,
        scale,
        out=np.zeros_like(left),
        where=scale > 0.0,
    )
    normalized_right = np.divide(
        right,
        scale,
        out=np.zeros_like(right),
        where=scale > 0.0,
    )
    return scale * np.power(
        0.5 * (np.power(normalized_left, 8) + np.power(normalized_right, 8)),
        0.125,
    )


def compute_power8_ipose_measurements(
        observations: Person_Height_Consensus_Observations,
        camera_intrinsics: Float_Array,
        ground_normal_camera: Float_Array,
        *,
        geometry_epsilon: float = 1e-12,
    ) -> Power8_IPose_Measurements:
    '''Measure virtual-upright image height without applying population filters.'''
    if type(observations) is not Person_Height_Consensus_Observations:
        raise TypeError('observations must be Person_Height_Consensus_Observations')
    if (
            type(geometry_epsilon) is not float
            or not math.isfinite(geometry_epsilon)
            or geometry_epsilon <= 0.0
        ):
        raise ValueError('geometry_epsilon must be a positive float')
    matrix = validated_intrinsics_matrix(camera_intrinsics)
    normal = validated_camera_up_normal(ground_normal_camera)
    shoulders = np.mean(observations.shoulder_xy_px, axis=1)
    hips = np.mean(observations.hip_xy_px, axis=1)
    torso = np.linalg.norm(shoulders - hips, axis=1)
    left_leg = (
        np.linalg.norm(
            observations.hip_xy_px[:, 0] - observations.knee_xy_px[:, 0],
            axis=1,
        )
        + np.linalg.norm(
            observations.knee_xy_px[:, 0] - observations.ankle_xy_px[:, 0],
            axis=1,
        )
    )
    right_leg = (
        np.linalg.norm(
            observations.hip_xy_px[:, 1] - observations.knee_xy_px[:, 1],
            axis=1,
        )
        + np.linalg.norm(
            observations.knee_xy_px[:, 1] - observations.ankle_xy_px[:, 1],
            axis=1,
        )
    )
    heights = torso + stable_power8_mean(left_leg, right_leg)

    vanishing = matrix @ normal
    directions = vanishing[:2][None, :] - shoulders * vanishing[2]
    direction_norms = np.linalg.norm(directions, axis=1)
    valid = (
        np.isfinite(heights)
        & (heights > geometry_epsilon)
        & np.isfinite(direction_norms)
        & (direction_norms > geometry_epsilon)
    )
    normalized_directions = np.full_like(directions, np.nan)
    np.divide(
        directions,
        direction_norms[:, None],
        out=normalized_directions,
        where=valid[:, None],
    )
    ankle_midpoints = np.mean(observations.ankle_xy_px, axis=1)
    orientation = np.sum(
        (ankle_midpoints - shoulders) * normalized_directions, axis=1)
    normalized_directions[orientation < 0.0] *= -1.0
    projection_scalars = np.sum(
        (observations.ankle_xy_px - shoulders[:, None, :])
        * normalized_directions[:, None, :],
        axis=2,
    )
    projected_ankles = (
        shoulders[:, None, :]
        + projection_scalars[:, :, None] * normalized_directions[:, None, :]
    )
    bottoms = np.mean(projected_ankles, axis=1)
    valid_heights = np.asarray(heights, dtype=np.float64)
    valid_heights[~valid] = np.nan
    return Power8_IPose_Measurements(
        np.asarray(shoulders, dtype=np.float64),
        np.asarray(normalized_directions, dtype=np.float64),
        np.asarray(bottoms, dtype=np.float64),
        valid_heights,
        np.asarray(valid, dtype=np.bool_),
    )


__all__ = ['compute_power8_ipose_measurements']
