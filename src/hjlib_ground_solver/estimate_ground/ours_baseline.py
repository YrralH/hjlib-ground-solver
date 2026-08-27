'''Registered Ours Ground normal, offset and centered-camera baselines.'''

from dataclasses import dataclass
from enum import StrEnum
import math
from typing import cast

import numpy as np
from numpy.typing import NDArray
import torch

from hjlib_camera import Camera_Intrinsics
from hjlib_camera_solver import (
    Centered_Focal_Vertical_VP_Config,
    Centered_Focal_Vertical_VP_Result,
    Simple_Vertical_VP_Config,
    Simple_Vertical_VP_Result,
    Vanishing_Direction_Source,
    select_and_refit_vertical_vp_by_simple_orthogonal_support,
    solve_centered_focal_and_vertical_vp_by_orthogonal_support,
)
from hjlib_ground_solver.estimate_ground.by_kp_rcr.solve_by_top_bot.search_D import (
    solve_D_search,
)


class Ground_Normal_Baseline(StrEnum):
    GROUND_NORMAL_BASELINE001 = 'ground_normal_baseline001'


class Ground_Offset_Baseline(StrEnum):
    GROUND_OFFSET_BASELINE001 = 'ground_offset_baseline001'


class Ground_Normal_And_Camera_Baseline(StrEnum):
    GROUND_NORMAL_AND_CAMERA_BASELINE001 = 'ground_normal_and_camera_baseline001'


@dataclass(frozen=True, slots=True, init=False)
class Ground_Normal_Config:
    baseline:Ground_Normal_Baseline
    camera_solver_config:Simple_Vertical_VP_Config

    def __init__(self) -> None:
        raise TypeError('Ground_Normal_Config is constructed by ground_normal_config')


@dataclass(frozen=True, slots=True, init=False)
class Ground_Normal_Result:
    config:Ground_Normal_Config
    direction_result:Simple_Vertical_VP_Result
    ground_normal_camera:NDArray[np.float64]

    def __init__(self) -> None:
        raise TypeError('Ground_Normal_Result is constructed by solve_ground_normal')


@dataclass(frozen=True, slots=True, init=False)
class Ground_Offset_Config:
    baseline:Ground_Offset_Baseline
    confidence_threshold_strict_gt:float
    ankle_ratio_threshold_strict_lt:float
    height_prior_m:float
    distance_min_m:float
    distance_max_m:float
    distance_step_m:float

    def __init__(self) -> None:
        raise TypeError('Ground_Offset_Config is constructed by ground_offset_config')


@dataclass(frozen=True, slots=True)
class Ground_Offset_Observations:
    top_xy_px:NDArray[np.float64]
    bottom_xy_px:NDArray[np.float64]
    confidence:NDArray[np.float64]
    ankle_ratio:NDArray[np.float64]

    def __post_init__(self) -> None:
        top = owned_float64_array('top_xy_px', self.top_xy_px)
        bottom = owned_float64_array('bottom_xy_px', self.bottom_xy_px)
        confidence = owned_float64_array('confidence', self.confidence)
        ankle = owned_float64_array('ankle_ratio', self.ankle_ratio)
        if top.ndim != 2 or top.shape[1] != 2:
            raise ValueError('top_xy_px must have shape (N, 2)')
        if bottom.shape != top.shape:
            raise ValueError('bottom_xy_px must have the same shape as top_xy_px')
        count = top.shape[0]
        if confidence.shape != (count,) or ankle.shape != (count,):
            raise ValueError('confidence and ankle_ratio must have shape (N,)')
        if bool(np.any(ankle < 0.0)):
            raise ValueError('ankle_ratio must be non-negative')
        object.__setattr__(self, 'top_xy_px', top)
        object.__setattr__(self, 'bottom_xy_px', bottom)
        object.__setattr__(self, 'confidence', confidence)
        object.__setattr__(self, 'ankle_ratio', ankle)


@dataclass(frozen=True, slots=True, init=False)
class Ground_Offset_Selection:
    observations:Ground_Offset_Observations
    config:Ground_Offset_Config
    retained_mask:NDArray[np.bool_]

    def __init__(self) -> None:
        raise TypeError(
            'Ground_Offset_Selection is constructed by select_ground_offset_observations',
        )


@dataclass(frozen=True, slots=True, init=False)
class Ground_Offset_Result:
    selection:Ground_Offset_Selection
    ground_normal_camera:NDArray[np.float64]
    plane_camera_abcd:NDArray[np.float64]
    objective:float

    def __init__(self) -> None:
        raise TypeError('Ground_Offset_Result is constructed by solve_ground_offset')


@dataclass(frozen=True, slots=True, init=False)
class Ground_Normal_And_Camera_Config:
    baseline:Ground_Normal_And_Camera_Baseline
    camera_solver_config:Centered_Focal_Vertical_VP_Config

    def __init__(self) -> None:
        raise TypeError(
            'Ground_Normal_And_Camera_Config is constructed by '
            'ground_normal_and_camera_config',
        )


@dataclass(frozen=True, slots=True, init=False)
class Ground_Normal_And_Camera_Result:
    config:Ground_Normal_And_Camera_Config
    camera_result:Centered_Focal_Vertical_VP_Result

    def __init__(self) -> None:
        raise TypeError(
            'Ground_Normal_And_Camera_Result is constructed by '
            'solve_ground_normal_and_camera',
        )

    @property
    def camera_intrinsics(self) -> Camera_Intrinsics:
        return self.camera_result.camera_intrinsics

    @property
    def ground_normal_camera(self) -> NDArray[np.float64]:
        return self.camera_result.direction_camera_up


def owned_float64_array(
        name:str,
        value:NDArray[np.generic],
    ) -> NDArray[np.float64]:
    array = np.array(value, dtype=np.float64, copy=True)
    if not bool(np.isfinite(array).all()):
        raise ValueError('%s must contain only finite values' % name)
    array.setflags(write=False)
    return array


def parse_ground_normal_baseline(
        baseline:Ground_Normal_Baseline | str,
    ) -> Ground_Normal_Baseline:
    try:
        return Ground_Normal_Baseline(baseline)
    except ValueError as error:
        legal = ', '.join(item.value for item in Ground_Normal_Baseline)
        raise ValueError('unknown Ground Normal baseline; legal values: %s' % legal) from error


def parse_ground_offset_baseline(
        baseline:Ground_Offset_Baseline | str,
    ) -> Ground_Offset_Baseline:
    try:
        return Ground_Offset_Baseline(baseline)
    except ValueError as error:
        legal = ', '.join(item.value for item in Ground_Offset_Baseline)
        raise ValueError('unknown ground-offset baseline; legal values: %s' % legal) from error


def parse_ground_normal_and_camera_baseline(
        baseline:Ground_Normal_And_Camera_Baseline | str,
    ) -> Ground_Normal_And_Camera_Baseline:
    try:
        return Ground_Normal_And_Camera_Baseline(baseline)
    except ValueError as error:
        legal = ', '.join(item.value for item in Ground_Normal_And_Camera_Baseline)
        raise ValueError(
            'unknown Ground Normal and camera baseline; legal values: %s' % legal,
        ) from error


def ground_normal_config(
        baseline:Ground_Normal_Baseline | str = (
            Ground_Normal_Baseline.GROUND_NORMAL_BASELINE001
        ),
    ) -> Ground_Normal_Config:
    parsed = parse_ground_normal_baseline(baseline)
    instance = object.__new__(Ground_Normal_Config)
    object.__setattr__(instance, 'baseline', parsed)
    object.__setattr__(
        instance,
        'camera_solver_config',
        Simple_Vertical_VP_Config(
            minimum_cluster_support=5,
            minimum_abs_camera_y=0.8,
            orthogonality_tolerance_deg=3.0,
            residual_gate_px=0.25,
            minimum_retained_support=5,
            maximum_refit_iterations=20,
        ),
    )
    return instance


def solve_ground_normal(
        source:Vanishing_Direction_Source,
        intrinsics:Camera_Intrinsics,
        baseline:Ground_Normal_Baseline | str = (
            Ground_Normal_Baseline.GROUND_NORMAL_BASELINE001
        ),
    ) -> Ground_Normal_Result:
    config = ground_normal_config(baseline)
    direction_result = select_and_refit_vertical_vp_by_simple_orthogonal_support(
        source,
        intrinsics,
        config.camera_solver_config,
    )
    normal = owned_float64_array(
        'ground_normal_camera',
        direction_result.direction_camera_up,
    )
    instance = object.__new__(Ground_Normal_Result)
    object.__setattr__(instance, 'config', config)
    object.__setattr__(instance, 'direction_result', direction_result)
    object.__setattr__(instance, 'ground_normal_camera', normal)
    return instance


def ground_offset_config(
        baseline:Ground_Offset_Baseline | str = (
            Ground_Offset_Baseline.GROUND_OFFSET_BASELINE001
        ),
    ) -> Ground_Offset_Config:
    parsed = parse_ground_offset_baseline(baseline)
    instance = object.__new__(Ground_Offset_Config)
    object.__setattr__(instance, 'baseline', parsed)
    object.__setattr__(instance, 'confidence_threshold_strict_gt', 4.3)
    object.__setattr__(instance, 'ankle_ratio_threshold_strict_lt', 0.20)
    object.__setattr__(instance, 'height_prior_m', 1.27)
    object.__setattr__(instance, 'distance_min_m', -5.0)
    object.__setattr__(instance, 'distance_max_m', 80.0)
    object.__setattr__(instance, 'distance_step_m', 0.1)
    return instance


def ground_normal_and_camera_config(
        baseline:Ground_Normal_And_Camera_Baseline | str = (
            Ground_Normal_And_Camera_Baseline.GROUND_NORMAL_AND_CAMERA_BASELINE001
        ),
    ) -> Ground_Normal_And_Camera_Config:
    parsed = parse_ground_normal_and_camera_baseline(baseline)
    instance = object.__new__(Ground_Normal_And_Camera_Config)
    object.__setattr__(instance, 'baseline', parsed)
    object.__setattr__(
        instance,
        'camera_solver_config',
        Centered_Focal_Vertical_VP_Config(
            vertical_config=ground_normal_config().camera_solver_config,
            minimum_orthogonal_neighbor_count=2,
            maximum_focal_refit_iterations=20,
        ),
    )
    return instance


def select_ground_offset_observations(
        observations:Ground_Offset_Observations,
        config:Ground_Offset_Config,
    ) -> Ground_Offset_Selection:
    if type(observations) is not Ground_Offset_Observations:
        raise ValueError('observations must be Ground_Offset_Observations')
    if type(config) is not Ground_Offset_Config:
        raise ValueError('config must be an exact registered Ground_Offset_Config')
    registered = ground_offset_config(config.baseline)
    if config != registered:
        raise ValueError('config must be an exact registered Ground_Offset_Config')
    retained = (
        (observations.confidence > config.confidence_threshold_strict_gt)
        & (observations.ankle_ratio < config.ankle_ratio_threshold_strict_lt)
    )
    if int(np.count_nonzero(retained)) < 3:
        raise ValueError('ground-offset baseline requires at least three retained observations')
    segment_lengths = np.linalg.norm(
        observations.top_xy_px[retained] - observations.bottom_xy_px[retained],
        axis=1,
    )
    if bool(np.any(segment_lengths <= 0.0)):
        raise ValueError('retained top/bottom segments must have positive pixel length')
    mask = np.array(retained, dtype=np.bool_, copy=True)
    mask.setflags(write=False)
    instance = object.__new__(Ground_Offset_Selection)
    object.__setattr__(instance, 'observations', observations)
    object.__setattr__(instance, 'config', config)
    object.__setattr__(instance, 'retained_mask', mask)
    return instance


def validated_camera_up_normal(
        value:NDArray[np.float64],
    ) -> NDArray[np.float64]:
    normal = owned_float64_array('ground_normal_camera', value)
    if normal.shape != (3,):
        raise ValueError('ground_normal_camera must have shape (3,)')
    norm = float(np.linalg.norm(normal))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError('ground_normal_camera must be unit length within 1e-12')
    if normal[1] > 0.0:
        raise ValueError('ground_normal_camera must use the camera-up orientation')
    if normal[1] == 0.0:
        first_nonzero_indices = np.flatnonzero(normal)
        if first_nonzero_indices.size == 0 or normal[int(first_nonzero_indices[0])] < 0.0:
            raise ValueError('ground_normal_camera violates the axial tie-break')
    return normal


def solve_ground_offset(
        observations:Ground_Offset_Observations,
        ground_normal_camera:NDArray[np.float64],
        intrinsics:Camera_Intrinsics,
        baseline:Ground_Offset_Baseline | str = (
            Ground_Offset_Baseline.GROUND_OFFSET_BASELINE001
        ),
        *,
        device:torch.device = torch.device('cpu'),
    ) -> Ground_Offset_Result:
    if type(intrinsics) is not Camera_Intrinsics:
        raise ValueError('intrinsics must be an undistorted Camera_Intrinsics')
    if type(device) is not torch.device:
        raise ValueError('device must be a torch.device')
    config = ground_offset_config(baseline)
    selection = select_ground_offset_observations(observations, config)
    normal = validated_camera_up_normal(ground_normal_camera)
    retained = selection.retained_mask
    top = observations.top_xy_px[retained]
    bottom = observations.bottom_xy_px[retained]
    ones = np.ones((top.shape[0], 1), dtype=np.float64)
    xt = np.concatenate((top, ones), axis=1).T
    xb = np.concatenate((bottom, ones), axis=1).T
    solver_result = solve_D_search(
        xb=xb,
        xt=xt,
        ground_normal=normal,
        cam_para=np.asarray(intrinsics.K, dtype=np.float64),
        H_prior=config.height_prior_m,
        device=device,
        distance_min=config.distance_min_m,
        distance_max=config.distance_max_m,
        distance_step=config.distance_step_m,
        preserve_ground_normal_orientation=True,
    )
    solved_plane_raw, objective_raw = cast(
        tuple[NDArray[np.generic], NDArray[np.generic]],
        solver_result,
    )
    solved_plane = owned_float64_array('plane_camera_abcd', solved_plane_raw)
    if solved_plane.shape != (4,) or not np.array_equal(solved_plane[:3], normal):
        raise ValueError('ground-offset solver did not preserve the supplied Ground Normal')
    if solved_plane[3] <= 0.0:
        raise ValueError('ground-offset baseline requires a positive solved D')
    objective_array = np.asarray(objective_raw, dtype=np.float64)
    if objective_array.size != 1:
        raise ValueError('ground-offset objective must be scalar')
    objective = float(objective_array.reshape(()))
    if not math.isfinite(objective):
        raise ValueError('ground-offset objective must be finite')
    instance = object.__new__(Ground_Offset_Result)
    object.__setattr__(instance, 'selection', selection)
    object.__setattr__(instance, 'ground_normal_camera', normal)
    object.__setattr__(instance, 'plane_camera_abcd', solved_plane)
    object.__setattr__(instance, 'objective', objective)
    return instance


def solve_ground_normal_and_camera(
        source:Vanishing_Direction_Source,
        baseline:Ground_Normal_And_Camera_Baseline | str = (
            Ground_Normal_And_Camera_Baseline.GROUND_NORMAL_AND_CAMERA_BASELINE001
        ),
    ) -> Ground_Normal_And_Camera_Result:
    if type(source) is not Vanishing_Direction_Source:
        raise ValueError('source must be a Vanishing_Direction_Source')
    config = ground_normal_and_camera_config(baseline)
    camera_result = solve_centered_focal_and_vertical_vp_by_orthogonal_support(
        source,
        config.camera_solver_config,
    )
    instance = object.__new__(Ground_Normal_And_Camera_Result)
    object.__setattr__(instance, 'config', config)
    object.__setattr__(instance, 'camera_result', camera_result)
    return instance


__all__:list[str] = [
    'Ground_Normal_And_Camera_Baseline',
    'Ground_Normal_And_Camera_Config',
    'Ground_Normal_And_Camera_Result',
    'Ground_Normal_Baseline',
    'Ground_Normal_Config',
    'Ground_Normal_Result',
    'Ground_Offset_Baseline',
    'Ground_Offset_Config',
    'Ground_Offset_Observations',
    'Ground_Offset_Result',
    'Ground_Offset_Selection',
    'ground_normal_config',
    'ground_offset_config',
    'ground_normal_and_camera_config',
    'select_ground_offset_observations',
    'solve_ground_normal',
    'solve_ground_offset',
    'solve_ground_normal_and_camera',
]
