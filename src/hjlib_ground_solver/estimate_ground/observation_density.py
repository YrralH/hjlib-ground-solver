'''Empirical spatial-density balancing for ground observations.'''

from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp
import scipy.spatial as scipy_spatial
from scipy.spatial.distance import cdist

from hjlib_geometry import intersect_rays_with_planes


class KDTree_Query_Protocol(Protocol):
    def query(
            self,
            coordinates: NDArray[np.float64],
            *,
            k: int,
            eps: float,
            p: float,
            workers: int,
        ) -> tuple[object, object]: ...


class KDTree_Factory_Protocol(Protocol):
    def __call__(
            self,
            coordinates: NDArray[np.float64],
        ) -> KDTree_Query_Protocol: ...


KDTree_Factory = cast(
    KDTree_Factory_Protocol,
    getattr(scipy_spatial, 'cKDTree'),
)


def owned_float64_array(
        value: object,
        name: str,
        shape_tail: tuple[int, ...],
    ) -> NDArray[np.float64]:
    if not isinstance(value, np.ndarray):
        raise TypeError('%s must be a numpy array' % name)
    array = cast(NDArray[np.generic], value)
    if array.dtype != np.dtype(np.float64):
        raise TypeError('%s must have dtype float64' % name)
    if array.ndim != len(shape_tail) + 1 or array.shape[1:] != shape_tail:
        raise ValueError('%s has invalid shape %r' % (name, array.shape))
    if not bool(np.isfinite(array).all()):
        raise ValueError('%s must be finite' % name)
    output = cast(NDArray[np.float64], array.copy(order='C'))
    output.setflags(write=False)
    return output


def owned_float64_exact_array(
        value: object,
        name: str,
        shape: tuple[int, ...],
    ) -> NDArray[np.float64]:
    if not isinstance(value, np.ndarray):
        raise TypeError('%s must be a float64 numpy array' % name)
    array = cast(NDArray[np.generic], value)
    if array.dtype != np.dtype(np.float64):
        raise TypeError('%s must be a float64 numpy array' % name)
    if array.shape != shape or not bool(np.isfinite(array).all()):
        raise ValueError('%s must be finite with shape %r' % (name, shape))
    return cast(NDArray[np.float64], array.copy(order='C'))


def require_python_float(value: object, name: str) -> float:
    if type(value) is not float:
        raise TypeError('%s must be a Python float' % name)
    output = value
    if not np.isfinite(output):
        raise ValueError('%s must be finite' % name)
    return output


def require_close(
        actual: NDArray[np.float64],
        expected: NDArray[np.float64],
        name: str,
    ) -> None:
    if not np.allclose(actual, expected, rtol=1e-12, atol=1e-15):
        raise ValueError('%s is inconsistent with density fields' % name)


def provisional_unit_plane_coordinates(
        bottom: NDArray[np.float64],
        K: NDArray[np.float64],
        normal_input: NDArray[np.float64],
    ) -> NDArray[np.float64]:
    normal_scale = float(np.max(np.abs(normal_input)))
    if not np.isfinite(normal_scale) or normal_scale <= 0.0:
        raise ValueError('provisional_normal_camera must be nonzero')
    normal_scaled = normal_input / normal_scale
    normal_norm = float(np.linalg.norm(normal_scaled))
    if not np.isfinite(normal_norm) or normal_norm <= 0.0:
        raise ValueError('provisional_normal_camera has invalid norm')
    normal = normal_scaled / normal_norm

    count = bottom.shape[0]
    pixels = np.column_stack([bottom, np.ones(count, dtype=np.float64)])
    try:
        rays = np.linalg.solve(K, pixels.T).T
    except np.linalg.LinAlgError as error:
        raise ValueError('camera_K must be nonsingular') from error
    if not bool(np.isfinite(rays).all()):
        raise ValueError('camera rays must be finite')
    ray_norms = np.linalg.norm(rays, axis=1)
    if not bool(np.isfinite(ray_norms).all()) or bool(np.any(ray_norms <= 0.0)):
        raise ValueError('camera rays must have finite positive norm')
    signed_denominators = rays @ normal
    normalized_cosines = np.abs(signed_denominators) / ray_norms
    if (
            not bool(np.isfinite(signed_denominators).all())
            or not bool(np.isfinite(normalized_cosines).all())
            or bool(np.any(normalized_cosines <= 1e-10))
        ):
        raise ValueError('camera ray is parallel or too close to provisional plane')
    all_positive = bool(np.all(signed_denominators > 0.0))
    all_negative = bool(np.all(signed_denominators < 0.0))
    if not all_positive and not all_negative:
        raise ValueError('camera rays have mixed signs against provisional normal')
    plane_offset = -1.0 if all_positive else 1.0
    plane = np.concatenate([normal, np.array([plane_offset], dtype=np.float64)])
    intersections, unused_distances = intersect_rays_with_planes(
        np.zeros((count, 3), dtype=np.float64),
        rays,
        np.broadcast_to(plane, (count, 4)).copy(),
        min_abs_cosine=1e-10,
    )
    del unused_distances

    axis_index = int(np.argmin(np.abs(normal)))
    axis = np.zeros(3, dtype=np.float64)
    axis[axis_index] = 1.0
    basis_first = np.cross(normal, axis)
    basis_first = basis_first / np.linalg.norm(basis_first)
    basis_second = np.cross(normal, basis_first)
    coordinates = intersections @ np.stack([basis_first, basis_second], axis=1)
    if not bool(np.isfinite(coordinates).all()):
        raise ValueError('provisional unit-plane coordinates must be finite')
    return coordinates


def density_radius_floor(
        coordinates: NDArray[np.float64],
        raw_radius: NDArray[np.float64],
    ) -> float:
    positive_radius = raw_radius[raw_radius > 0.0]
    if positive_radius.size > 0:
        reference_radius = float(np.median(positive_radius))
    else:
        unique_coordinates = np.unique(coordinates, axis=0)
        if unique_coordinates.shape[0] < 2:
            raise ValueError('density population is spatially collapsed')
        unique_tree = KDTree_Factory(unique_coordinates)
        distances_dynamic, unused_indices = unique_tree.query(
            unique_coordinates,
            k=2,
            eps=0.0,
            p=2.0,
            workers=1,
        )
        del unused_indices
        distances = np.asarray(distances_dynamic, dtype=np.float64)
        reference_radius = float(np.median(distances[:, 1]))
    radius_floor = reference_radius * 1e-6
    if not np.isfinite(radius_floor) or radius_floor <= 0.0:
        raise ValueError('density radius floor must be finite and positive')
    return radius_floor


@dataclass(frozen=True, slots=True)
class Ground_Observation_Density:
    '''One ordered empirical kNN-density and observation-weight record.'''

    provisional_unit_plane_xy: NDArray[np.float64]
    knn_radius_unit_plane: NDArray[np.float64]
    effective_knn_radius_unit_plane: NDArray[np.float64]
    empirical_knn_density_per_unit_area: NDArray[np.float64]
    relative_inverse_empirical_density: NDArray[np.float64]
    clipped_relative_inverse_empirical_density: NDArray[np.float64]
    normalized_observation_weights: NDArray[np.float64]
    neighbor_count: int
    radius_floor_unit_plane: float
    minimum_pre_normalization_weight: float
    maximum_pre_normalization_weight: float
    weight_normalization_factor: float
    effective_sample_size: float

    def __post_init__(self) -> None:
        coordinates = owned_float64_array(
            self.provisional_unit_plane_xy,
            'provisional_unit_plane_xy',
            (2,),
        )
        raw_radius = owned_float64_array(
            self.knn_radius_unit_plane,
            'knn_radius_unit_plane',
            (),
        )
        effective_radius = owned_float64_array(
            self.effective_knn_radius_unit_plane,
            'effective_knn_radius_unit_plane',
            (),
        )
        density = owned_float64_array(
            self.empirical_knn_density_per_unit_area,
            'empirical_knn_density_per_unit_area',
            (),
        )
        relative_inverse = owned_float64_array(
            self.relative_inverse_empirical_density,
            'relative_inverse_empirical_density',
            (),
        )
        clipped = owned_float64_array(
            self.clipped_relative_inverse_empirical_density,
            'clipped_relative_inverse_empirical_density',
            (),
        )
        weights = owned_float64_array(
            self.normalized_observation_weights,
            'normalized_observation_weights',
            (),
        )
        count = coordinates.shape[0]
        arrays = (
            raw_radius,
            effective_radius,
            density,
            relative_inverse,
            clipped,
            weights,
        )
        if any(array.shape[0] != count for array in arrays):
            raise ValueError('density arrays must share first-axis count')
        if type(self.neighbor_count) is not int:
            raise TypeError('neighbor_count must be a Python integer')
        neighbor_count = self.neighbor_count
        if neighbor_count < 1 or count < neighbor_count + 1:
            raise ValueError('density requires N >= neighbor_count + 1')
        radius_floor = require_python_float(
            self.radius_floor_unit_plane,
            'radius_floor_unit_plane',
        )
        minimum_weight = require_python_float(
            self.minimum_pre_normalization_weight,
            'minimum_pre_normalization_weight',
        )
        maximum_weight = require_python_float(
            self.maximum_pre_normalization_weight,
            'maximum_pre_normalization_weight',
        )
        normalization = require_python_float(
            self.weight_normalization_factor,
            'weight_normalization_factor',
        )
        effective_sample_size = require_python_float(
            self.effective_sample_size,
            'effective_sample_size',
        )
        if radius_floor <= 0.0:
            raise ValueError('radius_floor_unit_plane must be positive')
        if not 0.0 < minimum_weight <= 1.0 <= maximum_weight:
            raise ValueError('pre-normalization weight bounds are invalid')
        if normalization <= 0.0:
            raise ValueError('weight_normalization_factor must be positive')
        if bool(np.any(raw_radius < 0.0)) or bool(np.any(effective_radius <= 0.0)):
            raise ValueError('kNN radii must be nonnegative/positive')
        expected_floor = density_radius_floor(coordinates, raw_radius)
        if not np.isclose(radius_floor, expected_floor, rtol=1e-12, atol=0.0):
            raise ValueError('radius_floor_unit_plane is inconsistent')
        require_close(
            effective_radius,
            np.maximum(raw_radius, radius_floor),
            'effective_knn_radius_unit_plane',
        )
        expected_density = neighbor_count / (np.pi * effective_radius ** 2)
        require_close(density, expected_density, 'empirical_knn_density_per_unit_area')
        inverse_density = 1.0 / density
        expected_relative = inverse_density / float(np.median(inverse_density))
        require_close(
            relative_inverse,
            expected_relative,
            'relative_inverse_empirical_density',
        )
        expected_clipped = np.clip(relative_inverse, minimum_weight, maximum_weight)
        require_close(
            clipped,
            expected_clipped,
            'clipped_relative_inverse_empirical_density',
        )
        expected_normalization = float(np.mean(clipped))
        if not np.isclose(normalization, expected_normalization, rtol=1e-12, atol=0.0):
            raise ValueError('weight_normalization_factor is inconsistent')
        expected_weights = clipped / normalization
        require_close(weights, expected_weights, 'normalized_observation_weights')
        expected_effective_size = float(np.sum(weights) ** 2 / np.sum(weights ** 2))
        if not np.isclose(
                effective_sample_size,
                expected_effective_size,
                rtol=1e-12,
                atol=0.0,
            ):
            raise ValueError('effective_sample_size is inconsistent')
        if not 1.0 - 1e-12 <= effective_sample_size <= count + 1e-12:
            raise ValueError('effective_sample_size is outside [1,N]')

        object.__setattr__(self, 'provisional_unit_plane_xy', coordinates)
        object.__setattr__(self, 'knn_radius_unit_plane', raw_radius)
        object.__setattr__(self, 'effective_knn_radius_unit_plane', effective_radius)
        object.__setattr__(self, 'empirical_knn_density_per_unit_area', density)
        object.__setattr__(self, 'relative_inverse_empirical_density', relative_inverse)
        object.__setattr__(
            self,
            'clipped_relative_inverse_empirical_density',
            clipped,
        )
        object.__setattr__(self, 'normalized_observation_weights', weights)
        object.__setattr__(self, 'radius_floor_unit_plane', radius_floor)
        object.__setattr__(self, 'minimum_pre_normalization_weight', minimum_weight)
        object.__setattr__(self, 'maximum_pre_normalization_weight', maximum_weight)
        object.__setattr__(self, 'weight_normalization_factor', normalization)
        object.__setattr__(self, 'effective_sample_size', effective_sample_size)

    @property
    def count(self) -> int:
        return int(self.provisional_unit_plane_xy.shape[0])


@dataclass(frozen=True, slots=True)
class Ground_Observation_KDE_Density:
    '''One ordered exact LOO Gaussian-KDE observation-weight record.'''

    provisional_unit_plane_xy: NDArray[np.float64]
    kernel_covariance_unit_plane: NDArray[np.float64]
    loo_log_density_per_unit_area: NDArray[np.float64]
    log_relative_inverse_density: NDArray[np.float64]
    clipped_relative_inverse_density: NDArray[np.float64]
    normalized_observation_weights: NDArray[np.float64]
    scott_bandwidth_factor: float
    minimum_pre_normalization_weight: float
    maximum_pre_normalization_weight: float
    weight_normalization_factor: float
    effective_sample_size: float

    def __post_init__(self) -> None:
        coordinates = owned_float64_array(
            self.provisional_unit_plane_xy,
            'provisional_unit_plane_xy',
            (2,),
        )
        covariance = owned_float64_exact_array(
            self.kernel_covariance_unit_plane,
            'kernel_covariance_unit_plane',
            (2, 2),
        )
        log_density = owned_float64_array(
            self.loo_log_density_per_unit_area,
            'loo_log_density_per_unit_area',
            (),
        )
        log_relative = owned_float64_array(
            self.log_relative_inverse_density,
            'log_relative_inverse_density',
            (),
        )
        clipped = owned_float64_array(
            self.clipped_relative_inverse_density,
            'clipped_relative_inverse_density',
            (),
        )
        weights = owned_float64_array(
            self.normalized_observation_weights,
            'normalized_observation_weights',
            (),
        )
        count = coordinates.shape[0]
        if count < 3 or any(
                array.shape != (count,)
                for array in (log_density, log_relative, clipped, weights)
            ):
            raise ValueError('KDE density arrays have invalid shared count')
        if not np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-15):
            raise ValueError('kernel covariance must be symmetric')
        try:
            lower = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as error:
            raise ValueError('kernel covariance must be positive definite') from error
        if not bool(np.isfinite(lower).all()) or bool(np.any(np.diag(lower) <= 0.0)):
            raise ValueError('kernel covariance Cholesky factor is invalid')

        bandwidth = require_python_float(
            self.scott_bandwidth_factor,
            'scott_bandwidth_factor',
        )
        minimum_weight = require_python_float(
            self.minimum_pre_normalization_weight,
            'minimum_pre_normalization_weight',
        )
        maximum_weight = require_python_float(
            self.maximum_pre_normalization_weight,
            'maximum_pre_normalization_weight',
        )
        normalization = require_python_float(
            self.weight_normalization_factor,
            'weight_normalization_factor',
        )
        effective_sample_size = require_python_float(
            self.effective_sample_size,
            'effective_sample_size',
        )
        expected_bandwidth = float(count ** (-1.0 / 6.0))
        if not np.isclose(bandwidth, expected_bandwidth, rtol=1e-12, atol=0.0):
            raise ValueError('Scott bandwidth factor is inconsistent')
        expected_covariance = np.asarray(
            np.cov(coordinates, rowvar=False, ddof=1),
            dtype=np.float64,
        ) * expected_bandwidth ** 2
        require_close(
            covariance,
            expected_covariance,
            'kernel_covariance_unit_plane',
        )
        if not 0.0 < minimum_weight <= 1.0 <= maximum_weight:
            raise ValueError('pre-normalization weight bounds are invalid')
        if normalization <= 0.0:
            raise ValueError('weight_normalization_factor must be positive')

        expected_log_relative = np.median(log_density) - log_density
        require_close(
            log_relative,
            expected_log_relative,
            'log_relative_inverse_density',
        )
        expected_clipped = np.exp(np.clip(
            expected_log_relative,
            np.log(minimum_weight),
            np.log(maximum_weight),
        ))
        require_close(
            clipped,
            expected_clipped,
            'clipped_relative_inverse_density',
        )
        expected_normalization = float(np.mean(expected_clipped))
        if not np.isclose(normalization, expected_normalization, rtol=1e-12, atol=0.0):
            raise ValueError('weight_normalization_factor is inconsistent')
        expected_weights = expected_clipped / expected_normalization
        require_close(weights, expected_weights, 'normalized_observation_weights')
        expected_effective_size = float(count ** 2 / np.sum(weights ** 2))
        if not np.isclose(
                effective_sample_size,
                expected_effective_size,
                rtol=1e-12,
                atol=0.0,
            ):
            raise ValueError('effective_sample_size is inconsistent')
        if not 1.0 - 1e-12 <= effective_sample_size <= count + 1e-12:
            raise ValueError('effective_sample_size is outside [1,N]')

        covariance.setflags(write=False)
        object.__setattr__(self, 'provisional_unit_plane_xy', coordinates)
        object.__setattr__(self, 'kernel_covariance_unit_plane', covariance)
        object.__setattr__(self, 'loo_log_density_per_unit_area', log_density)
        object.__setattr__(self, 'log_relative_inverse_density', log_relative)
        object.__setattr__(self, 'clipped_relative_inverse_density', clipped)
        object.__setattr__(self, 'normalized_observation_weights', weights)
        object.__setattr__(self, 'scott_bandwidth_factor', bandwidth)
        object.__setattr__(self, 'minimum_pre_normalization_weight', minimum_weight)
        object.__setattr__(self, 'maximum_pre_normalization_weight', maximum_weight)
        object.__setattr__(self, 'weight_normalization_factor', normalization)
        object.__setattr__(self, 'effective_sample_size', effective_sample_size)

    @property
    def count(self) -> int:
        return int(self.provisional_unit_plane_xy.shape[0])


def compute_ground_observation_density(
        bottom_xy_px: NDArray[np.float64],
        camera_K: NDArray[np.float64],
        provisional_normal_camera: NDArray[np.float64],
        neighbor_count: int,
        minimum_pre_normalization_weight: float = 0.25,
        maximum_pre_normalization_weight: float = 4.0,
    ) -> Ground_Observation_Density:
    '''Compute one-pass empirical kNN inverse-density observation weights.'''
    bottom = owned_float64_array(bottom_xy_px, 'bottom_xy_px', (2,))
    K = owned_float64_exact_array(camera_K, 'camera_K', (3, 3))
    normal_input = owned_float64_exact_array(
        provisional_normal_camera,
        'provisional_normal_camera',
        (3,),
    )
    if type(neighbor_count) is not int:
        raise TypeError('neighbor_count must be a Python integer')
    count = bottom.shape[0]
    if neighbor_count < 1 or count < neighbor_count + 1:
        raise ValueError('density requires N >= neighbor_count + 1')
    minimum_weight = require_python_float(
        minimum_pre_normalization_weight,
        'minimum_pre_normalization_weight',
    )
    maximum_weight = require_python_float(
        maximum_pre_normalization_weight,
        'maximum_pre_normalization_weight',
    )
    if not 0.0 < minimum_weight <= 1.0 <= maximum_weight:
        raise ValueError('pre-normalization weight bounds are invalid')

    coordinates = provisional_unit_plane_coordinates(bottom, K, normal_input)

    tree = KDTree_Factory(coordinates)
    distances_dynamic, unused_indices = tree.query(
        coordinates,
        k=neighbor_count + 1,
        eps=0.0,
        p=2.0,
        workers=1,
    )
    del unused_indices
    distances = np.asarray(distances_dynamic, dtype=np.float64)
    if distances.shape != (count, neighbor_count + 1):
        raise ValueError('cKDTree returned an unexpected distance shape')
    raw_radius = distances[:, neighbor_count]
    radius_floor = density_radius_floor(coordinates, raw_radius)
    effective_radius = np.maximum(raw_radius, radius_floor)
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        density = neighbor_count / (np.pi * effective_radius ** 2)
        inverse_density = 1.0 / density
        relative_inverse = inverse_density / float(np.median(inverse_density))
    clipped = np.clip(relative_inverse, minimum_weight, maximum_weight)
    normalization = float(np.mean(clipped))
    weights = clipped / normalization
    effective_sample_size = float(np.sum(weights) ** 2 / np.sum(weights ** 2))
    numerical_arrays = (
        raw_radius,
        effective_radius,
        density,
        relative_inverse,
        clipped,
        weights,
    )
    if not all(bool(np.isfinite(array).all()) for array in numerical_arrays):
        raise ValueError('density calculation produced nonfinite values')

    return Ground_Observation_Density(
        provisional_unit_plane_xy=coordinates,
        knn_radius_unit_plane=raw_radius,
        effective_knn_radius_unit_plane=effective_radius,
        empirical_knn_density_per_unit_area=density,
        relative_inverse_empirical_density=relative_inverse,
        clipped_relative_inverse_empirical_density=clipped,
        normalized_observation_weights=weights,
        neighbor_count=neighbor_count,
        radius_floor_unit_plane=radius_floor,
        minimum_pre_normalization_weight=minimum_weight,
        maximum_pre_normalization_weight=maximum_weight,
        weight_normalization_factor=normalization,
        effective_sample_size=effective_sample_size,
    )


def compute_ground_observation_kde_density(
        bottom_xy_px: NDArray[np.float64],
        camera_K: NDArray[np.float64],
        provisional_normal_camera: NDArray[np.float64],
        *,
        minimum_pre_normalization_weight: float = 0.25,
        maximum_pre_normalization_weight: float = 4.0,
    ) -> Ground_Observation_KDE_Density:
    '''Compute stable exact LOO Gaussian-KDE inverse-density weights.'''
    bottom = owned_float64_array(bottom_xy_px, 'bottom_xy_px', (2,))
    K = owned_float64_exact_array(camera_K, 'camera_K', (3, 3))
    normal_input = owned_float64_exact_array(
        provisional_normal_camera,
        'provisional_normal_camera',
        (3,),
    )
    count = bottom.shape[0]
    if count < 3:
        raise ValueError('KDE density requires N >= 3')
    minimum_weight = require_python_float(
        minimum_pre_normalization_weight,
        'minimum_pre_normalization_weight',
    )
    maximum_weight = require_python_float(
        maximum_pre_normalization_weight,
        'maximum_pre_normalization_weight',
    )
    if not 0.0 < minimum_weight <= 1.0 <= maximum_weight:
        raise ValueError('pre-normalization weight bounds are invalid')

    coordinates = provisional_unit_plane_coordinates(bottom, K, normal_input)
    sample_covariance = np.asarray(
        np.cov(coordinates, rowvar=False, ddof=1),
        dtype=np.float64,
    )
    bandwidth = float(count ** (-1.0 / 6.0))
    kernel_covariance = sample_covariance * bandwidth ** 2
    if (
            kernel_covariance.shape != (2, 2)
            or not bool(np.isfinite(kernel_covariance).all())
            or not np.allclose(
                kernel_covariance,
                kernel_covariance.T,
                rtol=0.0,
                atol=1e-15,
            )
        ):
        raise ValueError('KDE kernel covariance is invalid')
    try:
        lower = np.linalg.cholesky(kernel_covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError('KDE population covariance must be positive definite') from error
    diagonal = np.diag(lower)
    if not bool(np.isfinite(lower).all()) or bool(np.any(diagonal <= 0.0)):
        raise ValueError('KDE Cholesky factor is invalid')

    centered = coordinates - np.mean(coordinates, axis=0)
    whitened = np.linalg.solve(lower, centered.T).T
    if not bool(np.isfinite(whitened).all()):
        raise ValueError('KDE whitened coordinates must be finite')
    log_det_covariance = 2.0 * float(np.sum(np.log(diagonal)))
    log_kernel_normalizer = -0.5 * (
        2.0 * np.log(2.0 * np.pi) + log_det_covariance
    )
    if not np.isfinite(log_det_covariance) or not np.isfinite(log_kernel_normalizer):
        raise ValueError('KDE log kernel normalizer must be finite')

    chunk_size = 256
    log_density = np.empty(count, dtype=np.float64)
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        squared_distances = cdist(
            whitened[start:stop],
            whitened,
            metric='sqeuclidean',
        )
        if (
                squared_distances.shape != (stop - start, count)
                or not bool(np.isfinite(squared_distances).all())
                or bool(np.any(squared_distances < 0.0))
            ):
            raise ValueError('KDE squared distances are invalid')
        local_indices = np.arange(stop - start)
        squared_distances[local_indices, start + local_indices] = np.inf
        log_density[start:stop] = (
            logsumexp(
                -0.5 * squared_distances + log_kernel_normalizer,
                axis=1,
            )
            - np.log(count - 1)
        )
    if not bool(np.isfinite(log_density).all()):
        raise ValueError('KDE leave-one-out log density must be finite')

    log_relative = np.median(log_density) - log_density
    clipped = np.exp(np.clip(
        log_relative,
        np.log(minimum_weight),
        np.log(maximum_weight),
    ))
    normalization = float(np.mean(clipped))
    weights = clipped / normalization
    effective_sample_size = float(count ** 2 / np.sum(weights ** 2))
    numerical_arrays = (log_relative, clipped, weights)
    if (
            not all(bool(np.isfinite(array).all()) for array in numerical_arrays)
            or not np.isfinite(normalization)
            or normalization <= 0.0
            or not np.isfinite(effective_sample_size)
        ):
        raise ValueError('KDE inverse-density calculation produced invalid values')

    return Ground_Observation_KDE_Density(
        provisional_unit_plane_xy=coordinates,
        kernel_covariance_unit_plane=kernel_covariance,
        loo_log_density_per_unit_area=log_density,
        log_relative_inverse_density=log_relative,
        clipped_relative_inverse_density=clipped,
        normalized_observation_weights=weights,
        scott_bandwidth_factor=bandwidth,
        minimum_pre_normalization_weight=minimum_weight,
        maximum_pre_normalization_weight=maximum_weight,
        weight_normalization_factor=normalization,
        effective_sample_size=effective_sample_size,
    )


__all__ = [
    'Ground_Observation_Density',
    'Ground_Observation_KDE_Density',
    'compute_ground_observation_density',
    'compute_ground_observation_kde_density',
]
