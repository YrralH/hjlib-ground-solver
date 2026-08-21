'''Portable smoke tests for empirical ground-observation density.'''

from dataclasses import replace
from typing import cast
from unittest.mock import patch

import numpy as np
from numpy.typing import NDArray
import pytest
from scipy.spatial import cKDTree  # pyright: ignore[reportAttributeAccessIssue]
from scipy.spatial.distance import cdist as scipy_cdist

from hjlib_ground_solver import (
    compute_ground_observation_density,
    compute_ground_observation_kde_density,
)


def identity_camera() -> NDArray[np.float64]:
    return np.eye(3, dtype=np.float64)


def forward_normal() -> NDArray[np.float64]:
    return np.array([0.0, 0.0, 1.0], dtype=np.float64)


def test_hand_solvable_coordinates_and_normal_invariance() -> None:
    bottom = np.array(
        [[0.0, 1.0], [1.0, 1.0], [0.0, 2.0], [2.0, 0.5], [-1.0, 1.5]],
        dtype=np.float64,
    )
    result = compute_ground_observation_density(
        bottom,
        identity_camera(),
        forward_normal(),
        2,
    )
    expected_xy = np.column_stack([bottom[:, 1], -bottom[:, 0]])
    np.testing.assert_allclose(result.provisional_unit_plane_xy, expected_xy)

    scaled = compute_ground_observation_density(
        bottom,
        identity_camera(),
        forward_normal() * 7.0,
        2,
    )
    flipped = compute_ground_observation_density(
        bottom,
        identity_camera(),
        -forward_normal(),
        2,
    )
    np.testing.assert_allclose(
        scaled.provisional_unit_plane_xy,
        result.provisional_unit_plane_xy,
    )
    distances = np.linalg.norm(
        result.provisional_unit_plane_xy[:, None, :]
        - result.provisional_unit_plane_xy[None, :, :],
        axis=2,
    )
    distances_flipped = np.linalg.norm(
        flipped.provisional_unit_plane_xy[:, None, :]
        - flipped.provisional_unit_plane_xy[None, :, :],
        axis=2,
    )
    np.testing.assert_allclose(distances_flipped, distances)
    np.testing.assert_allclose(
        flipped.empirical_knn_density_per_unit_area,
        result.empirical_knn_density_per_unit_area,
    )
    np.testing.assert_allclose(
        flipped.normalized_observation_weights,
        result.normalized_observation_weights,
    )


def test_scale_algebra_and_owned_readonly_fields() -> None:
    generator = np.random.default_rng(3)
    bottom = generator.uniform(-2.0, 2.0, size=(80, 2)).astype(np.float64)
    result = compute_ground_observation_density(
        bottom,
        identity_camera(),
        forward_normal(),
        16,
    )
    scaled_radius = result.effective_knn_radius_unit_plane * 11.0
    scaled_density = 16 / (np.pi * scaled_radius ** 2)
    scaled_inverse = 1.0 / scaled_density
    scaled_relative = scaled_inverse / np.median(scaled_inverse)
    scaled_clipped = np.clip(scaled_relative, 0.25, 4.0)
    scaled_weights = scaled_clipped / np.mean(scaled_clipped)
    np.testing.assert_allclose(
        scaled_weights,
        result.normalized_observation_weights,
    )
    assert np.isclose(np.mean(result.normalized_observation_weights), 1.0)
    assert 1.0 <= result.effective_sample_size <= result.count
    bottom[0] = 999.0
    assert not np.any(result.provisional_unit_plane_xy == 999.0)
    with pytest.raises(ValueError):
        result.normalized_observation_weights[0] = 2.0
    with pytest.raises(ValueError, match='inconsistent'):
        replace(
            result,
            normalized_observation_weights=np.ones(result.count, dtype=np.float64),
        )
def test_kde_chunked_cdist_and_global_diagonal() -> None:
    generator = np.random.default_rng(47)
    bottom = generator.normal(size=(257, 2)).astype(np.float64)
    call_shapes: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = []

    def recording_cdist(
            first: NDArray[np.float64],
            second: NDArray[np.float64],
            *,
            metric: str,
        ) -> NDArray[np.float64]:
        assert metric == 'sqeuclidean'
        output = cast(
            NDArray[np.float64],
            scipy_cdist(first, second, metric='sqeuclidean'),
        )
        call_shapes.append((first.shape, second.shape, output.shape))
        return output

    path = 'hjlib_ground_solver.estimate_ground.observation_density.cdist'
    with patch(path, recording_cdist):
        result = compute_ground_observation_kde_density(
            bottom,
            identity_camera(),
            forward_normal(),
        )
    assert call_shapes == [
        ((256, 2), (257, 2), (256, 257)),
        ((1, 2), (257, 2), (1, 257)),
    ]
    expected = brute_force_loo_log_density(
        result.provisional_unit_plane_xy,
        np.asarray(
            np.cov(result.provisional_unit_plane_xy, rowvar=False, ddof=1),
            dtype=np.float64,
        ) * 257 ** (-1.0 / 3.0),
    )
    np.testing.assert_allclose(
        result.loo_log_density_per_unit_area,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_density_balances_regions_and_documents_boundary_bias() -> None:
    generator = np.random.default_rng(11)
    dense = generator.uniform([0.0, 0.0], [1.0, 1.0], size=(800, 2))
    sparse = generator.uniform([2.0, 0.0], [3.0, 1.0], size=(200, 2))
    bottom = np.concatenate([dense, sparse]).astype(np.float64)
    result = compute_ground_observation_density(
        bottom,
        identity_camera(),
        forward_normal(),
        32,
        minimum_pre_normalization_weight=0.01,
        maximum_pre_normalization_weight=100.0,
    )
    dense_total = float(np.sum(result.normalized_observation_weights[:800]))
    sparse_total = float(np.sum(result.normalized_observation_weights[800:]))
    weighted_ratio = dense_total / sparse_total
    assert abs(weighted_ratio - 1.0) < abs(4.0 - 1.0)

    axis = np.linspace(-1.0, 1.0, 15, dtype=np.float64)
    x_grid, y_grid = np.meshgrid(axis, axis)
    lattice = np.column_stack([x_grid.ravel(), y_grid.ravel()])
    lattice_result = compute_ground_observation_density(
        lattice,
        identity_camera(),
        forward_normal(),
        8,
    )
    corner_index = 0
    center_index = 7 * 15 + 7
    assert (
        lattice_result.normalized_observation_weights[corner_index]
        > lattice_result.normalized_observation_weights[center_index]
    )
    np.testing.assert_allclose(
        lattice_result.normalized_observation_weights[center_index],
        lattice_result.normalized_observation_weights[center_index + 1],
    )


def test_duplicate_floor_and_neighbor_variants() -> None:
    few_duplicates = np.array(
        [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=np.float64,
    )
    few = compute_ground_observation_density(
        few_duplicates,
        identity_camera(),
        forward_normal(),
        3,
    )
    assert bool(np.all(few.knn_radius_unit_plane > 0.0))

    many_duplicates = np.concatenate([
        np.zeros((5, 2), dtype=np.float64),
        np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64),
    ])
    many = compute_ground_observation_density(
        many_duplicates,
        identity_camera(),
        forward_normal(),
        3,
    )
    assert bool(np.all(many.knn_radius_unit_plane[:5] == 0.0))
    assert bool(np.all(
        many.effective_knn_radius_unit_plane[:5]
        == many.radius_floor_unit_plane
    ))

    repeated_clusters = np.concatenate([
        np.zeros((3, 2), dtype=np.float64),
        np.tile(np.array([[2.0, 0.0]], dtype=np.float64), (3, 1)),
    ])
    clustered = compute_ground_observation_density(
        repeated_clusters,
        identity_camera(),
        forward_normal(),
        2,
    )
    assert bool(np.all(clustered.knn_radius_unit_plane == 0.0))
    assert clustered.radius_floor_unit_plane > 0.0
    np.testing.assert_allclose(clustered.normalized_observation_weights, 1.0)

    generator = np.random.default_rng(17)
    population = generator.uniform(-3.0, 3.0, size=(96, 2)).astype(np.float64)
    for neighbor_count in (16, 32, 64):
        output = compute_ground_observation_density(
            population,
            identity_camera(),
            forward_normal(),
            neighbor_count,
        )
        assert output.count == 96
        assert output.neighbor_count == neighbor_count


def test_invalid_inputs_fail_closed() -> None:
    bottom = np.array(
        [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0]],
        dtype=np.float64,
    )
    with pytest.raises(ValueError, match='N >='):
        compute_ground_observation_density(
            bottom,
            identity_camera(),
            forward_normal(),
            4,
        )
    with pytest.raises(TypeError, match='float64'):
        compute_ground_observation_density(
            cast(NDArray[np.float64], bottom.astype(np.float32)),
            identity_camera(),
            forward_normal(),
            2,
        )
    with pytest.raises(TypeError, match='camera_K'):
        compute_ground_observation_density(
            bottom,
            cast(NDArray[np.float64], [1.0, 2.0, 3.0]),
            forward_normal(),
            2,
        )
    with pytest.raises(TypeError, match='provisional_normal_camera'):
        compute_ground_observation_density(
            bottom,
            identity_camera(),
            cast(NDArray[np.float64], None),
            2,
        )
    with pytest.raises(ValueError, match='nonsingular'):
        compute_ground_observation_density(
            bottom,
            np.zeros((3, 3), dtype=np.float64),
            forward_normal(),
            2,
        )
    with pytest.raises(ValueError, match='parallel'):
        compute_ground_observation_density(
            bottom,
            identity_camera(),
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
            2,
        )
    mixed = bottom.copy()
    mixed[:, 0] = np.array([-2.0, -1.0, 1.0, 2.0])
    with pytest.raises(ValueError, match='mixed signs'):
        compute_ground_observation_density(
            mixed,
            identity_camera(),
            np.array([1.0, 0.0, 0.1], dtype=np.float64),
            2,
        )
    with pytest.raises(ValueError, match='spatially collapsed'):
        compute_ground_observation_density(
            np.ones((4, 2), dtype=np.float64),
            identity_camera(),
            forward_normal(),
            2,
        )
    with pytest.raises(ValueError, match='bounds'):
        compute_ground_observation_density(
            bottom,
            identity_camera(),
            forward_normal(),
            2,
            minimum_pre_normalization_weight=1.1,
        )
    nonfinite = bottom.copy()
    nonfinite[0, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        compute_ground_observation_density(
            nonfinite,
            identity_camera(),
            forward_normal(),
            2,
        )


class Counting_Tree:
    '''Count the one tree construction and query in the public operation.'''

    build_count = 0
    query_count = 0

    def __init__(self, coordinates: NDArray[np.float64]) -> None:
        type(self).build_count += 1
        self.tree = cKDTree(coordinates)

    def query(
            self,
            coordinates: NDArray[np.float64],
            k: int,
            eps: float,
            p: float,
            workers: int,
        ) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        type(self).query_count += 1
        distances, indices = self.tree.query(
            coordinates,
            k=k,
            eps=eps,
            p=p,
            workers=workers,
        )
        return (
            np.asarray(distances, dtype=np.float64),
            np.asarray(indices, dtype=np.int64),
        )


def test_one_tree_build_and_query() -> None:
    Counting_Tree.build_count = 0
    Counting_Tree.query_count = 0
    generator = np.random.default_rng(23)
    bottom = generator.uniform(-1.0, 1.0, size=(40, 2)).astype(np.float64)
    path = 'hjlib_ground_solver.estimate_ground.observation_density.KDTree_Factory'
    with patch(path, Counting_Tree):
        compute_ground_observation_density(
            bottom,
            identity_camera(),
            forward_normal(),
            8,
        )
    assert Counting_Tree.build_count == 1
    assert Counting_Tree.query_count == 1


def brute_force_loo_log_density(
        coordinates: NDArray[np.float64],
        covariance: NDArray[np.float64],
    ) -> NDArray[np.float64]:
    count = coordinates.shape[0]
    lower = np.linalg.cholesky(covariance)
    centered = coordinates - np.mean(coordinates, axis=0)
    whitened = np.linalg.solve(lower, centered.T).T
    log_determinant = 2.0 * float(np.sum(np.log(np.diag(lower))))
    normalizer = -0.5 * (2.0 * np.log(2.0 * np.pi) + log_determinant)
    output = np.empty(count, dtype=np.float64)
    for row in range(count):
        values: list[float] = []
        for other in range(count):
            if other == row:
                continue
            difference = whitened[row] - whitened[other]
            values.append(
                -0.5 * float(difference @ difference) + normalizer,
            )
        maximum = max(values)
        output[row] = (
            maximum
            + np.log(sum(np.exp(value - maximum) for value in values))
            - np.log(count - 1)
        )
    return output


def test_kde_exact_loo_oracle_and_invariances() -> None:
    bottom = np.array([
        [-1.0, -0.4],
        [-0.3, 0.8],
        [0.2, -0.7],
        [0.7, 0.4],
        [1.1, 1.3],
        [1.8, -0.2],
    ], dtype=np.float64)
    result = compute_ground_observation_kde_density(
        bottom,
        identity_camera(),
        forward_normal(),
    )
    expected = brute_force_loo_log_density(
        result.provisional_unit_plane_xy,
        result.kernel_covariance_unit_plane,
    )
    np.testing.assert_allclose(
        result.loo_log_density_per_unit_area,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )
    translated = compute_ground_observation_kde_density(
        bottom + np.array([1.0e6, -2.0e6], dtype=np.float64),
        identity_camera(),
        forward_normal(),
    )
    scaled = compute_ground_observation_kde_density(
        bottom * 17.0,
        identity_camera(),
        forward_normal(),
    )
    flipped = compute_ground_observation_kde_density(
        bottom,
        identity_camera(),
        -forward_normal(),
    )
    for alternative in (translated, scaled, flipped):
        np.testing.assert_allclose(
            alternative.normalized_observation_weights,
            result.normalized_observation_weights,
            rtol=1e-8,
            atol=1e-10,
        )
    assert np.isclose(np.mean(result.normalized_observation_weights), 1.0)
    assert 1.0 <= result.effective_sample_size <= result.count
    assert not result.normalized_observation_weights.flags.writeable
    assert not result.kernel_covariance_unit_plane.flags.writeable
    with pytest.raises(ValueError, match='kernel_covariance_unit_plane'):
        replace(
            result,
            kernel_covariance_unit_plane=result.kernel_covariance_unit_plane * 2.0,
        )
    with pytest.raises(ValueError, match='inconsistent'):
        replace(
            result,
            normalized_observation_weights=np.ones(result.count, dtype=np.float64),
        )


def test_kde_density_balancing_and_boundary_evidence() -> None:
    dense_axis = np.linspace(0.02, 0.98, 20, dtype=np.float64)
    sparse_axis = np.linspace(0.05, 0.95, 10, dtype=np.float64)
    dense_x, dense_y = np.meshgrid(dense_axis, dense_axis)
    sparse_x, sparse_y = np.meshgrid(sparse_axis, sparse_axis)
    dense = np.column_stack([dense_x.ravel(), dense_y.ravel()])
    sparse = np.column_stack([sparse_x.ravel() + 2.0, sparse_y.ravel()])
    regions = np.concatenate([dense, sparse])
    region_result = compute_ground_observation_kde_density(
        regions,
        identity_camera(),
        forward_normal(),
    )
    weighted_ratio = float(
        np.sum(region_result.normalized_observation_weights[:400])
        / np.sum(region_result.normalized_observation_weights[400:])
    )
    assert abs(np.log(weighted_ratio)) < abs(np.log(4.0))

    axis = np.linspace(0.0, 1.0, 21, dtype=np.float64)
    x_grid, y_grid = np.meshgrid(axis, axis)
    lattice = np.column_stack([x_grid.ravel(), y_grid.ravel()])
    lattice_result = compute_ground_observation_kde_density(
        lattice,
        identity_camera(),
        forward_normal(),
    )
    grid_weights = lattice_result.normalized_observation_weights.reshape(21, 21)
    boundary_mask = np.zeros((21, 21), dtype=np.bool_)
    boundary_mask[[0, -1], :] = True
    boundary_mask[:, [0, -1]] = True
    center_mask = np.zeros((21, 21), dtype=np.bool_)
    center_mask[6:15, 6:15] = True
    boundary_mean = float(np.mean(grid_weights[boundary_mask]))
    center_mean = float(np.mean(grid_weights[center_mask]))
    assert boundary_mean > center_mean


def test_kde_duplicates_isolation_and_invalid_covariance() -> None:
    generator = np.random.default_rng(71)
    bottom = np.concatenate([
        np.zeros((20, 2), dtype=np.float64),
        generator.normal(size=(80, 2)),
    ])
    duplicate_result = compute_ground_observation_kde_density(
        bottom,
        identity_camera(),
        forward_normal(),
    )
    assert bool(np.isfinite(duplicate_result.loo_log_density_per_unit_area).all())

    isolated = np.concatenate([
        generator.normal(scale=0.01, size=(99, 2)),
        np.array([[1.0e6, -1.0e6]], dtype=np.float64),
    ])
    isolated_result = compute_ground_observation_kde_density(
        isolated,
        identity_camera(),
        forward_normal(),
    )
    expected = brute_force_loo_log_density(
        isolated_result.provisional_unit_plane_xy,
        isolated_result.kernel_covariance_unit_plane,
    )
    np.testing.assert_allclose(
        isolated_result.loo_log_density_per_unit_area,
        expected,
        rtol=1e-11,
        atol=1e-11,
    )

    collinear = np.column_stack([
        np.linspace(-1.0, 1.0, 10, dtype=np.float64),
        np.zeros(10, dtype=np.float64),
    ])
    with pytest.raises(ValueError, match='positive definite'):
        compute_ground_observation_kde_density(
            collinear,
            identity_camera(),
            forward_normal(),
        )
    with pytest.raises(ValueError, match='N >= 3'):
        compute_ground_observation_kde_density(
            bottom[:2],
            identity_camera(),
            forward_normal(),
        )
    with pytest.raises(ValueError, match='bounds'):
        compute_ground_observation_kde_density(
            bottom,
            identity_camera(),
            forward_normal(),
            minimum_pre_normalization_weight=1.1,
        )

    near_collinear = np.column_stack([
        np.linspace(-1.0, 1.0, 30, dtype=np.float64),
        np.linspace(-1.0, 1.0, 30, dtype=np.float64) * 1.0e-5
        + np.where(np.arange(30) % 2 == 0, -1.0e-8, 1.0e-8),
    ])
    near_result = compute_ground_observation_kde_density(
        near_collinear,
        identity_camera(),
        forward_normal(),
    )
    assert bool(np.isfinite(near_result.normalized_observation_weights).all())


def smoke_test_observation_density() -> None:
    test_hand_solvable_coordinates_and_normal_invariance()
    test_scale_algebra_and_owned_readonly_fields()
    test_density_balances_regions_and_documents_boundary_bias()
    test_duplicate_floor_and_neighbor_variants()
    test_invalid_inputs_fail_closed()
    test_one_tree_build_and_query()
    test_kde_exact_loo_oracle_and_invariances()
    test_kde_chunked_cdist_and_global_diagonal()
    test_kde_density_balancing_and_boundary_evidence()
    test_kde_duplicates_isolation_and_invalid_covariance()
