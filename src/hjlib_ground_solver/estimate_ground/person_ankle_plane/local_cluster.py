'''Local clustering retained for the temporarily deprecated ankle-plane V1.'''

from dataclasses import dataclass
import math
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray
from scipy.spatial._ckdtree import cKDTree
from scipy.stats import rankdata

from hjlib_ground_solver.estimate_ground.person_ankle_plane.contract import (
    Person_Ankle_Foot_Shape,
    Person_Ankle_Height_Shape,
    Person_Ankle_Local_Cluster,
    Person_Ankle_Local_Config,
    Person_Ankle_Noise_Reason,
    Person_Ankle_Observation_Table,
    Person_Ankle_Space_Shape,
    Person_Ankle_Time_Shape,
)
from hjlib_ground_solver.estimate_ground.person_ankle_plane.normalize import (
    Normalized_Person_Ankle_Input,
    make_readonly_int_array,
)


@dataclass(frozen=True, slots=True)
class Local_Cluster_Stage_Result:
    observations: Person_Ankle_Observation_Table
    local_clusters: tuple[Person_Ankle_Local_Cluster, ...]


class Neighbor_Tree(Protocol):
    def query_ball_point(
            self,
            x: NDArray[np.float64],
            r: float,
            p: float,
        ) -> list[int]: ...


class Disjoint_Set:
    def __init__(self, size: int) -> None:
        self.parents = list(range(size))

    def find(self, index: int) -> int:
        parent = self.parents[index]
        while parent != self.parents[parent]:
            parent = self.parents[parent]
        while index != parent:
            next_index = self.parents[index]
            self.parents[index] = parent
            index = next_index
        return parent

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parents[max(left_root, right_root)] = min(left_root, right_root)


def exact_neighbors_for_run(
        indices: NDArray[np.int64],
        normalized: Normalized_Person_Ankle_Input,
        config: Person_Ankle_Local_Config,
    ) -> tuple[tuple[int, ...], ...]:
    coordinates, tree = make_scaled_tree(indices, normalized, config)
    return tuple(
        query_exact_neighbor_indices(
            local_index, indices, normalized, config, coordinates, tree)
        for local_index in range(len(indices))
    )


def make_scaled_tree(
        indices: NDArray[np.int64],
        normalized: Normalized_Person_Ankle_Input,
        config: Person_Ankle_Local_Config,
    ) -> tuple[NDArray[np.float64], Neighbor_Tree]:
    observations = normalized.observations
    coordinates = np.asarray(np.column_stack((
        observations.times_in_second[indices] / config.time_radius_in_second,
        observations.projected_points_in_meter[indices]
        / config.projected_radius_in_meter,
        observations.heights_in_meter[indices] / config.height_radius_in_meter,
    )), dtype=np.float64)
    return coordinates, cast(Neighbor_Tree, cKDTree(coordinates))


def query_exact_neighbor_indices(
        local_index: int,
        indices: NDArray[np.int64],
        normalized: Normalized_Person_Ankle_Input,
        config: Person_Ankle_Local_Config,
        coordinates: NDArray[np.float64],
        tree: Neighbor_Tree,
    ) -> tuple[int, ...]:
    observations = normalized.observations
    global_index = int(indices[local_index])
    candidate_local_indices = tree.query_ball_point(
        coordinates[local_index], r=1.0, p=np.inf)
    exact: list[int] = []
    for candidate_local_index_raw in candidate_local_indices:
        candidate_local_index = int(candidate_local_index_raw)
        candidate_global_index = int(indices[candidate_local_index])
        time_distance = abs(
            float(observations.times_in_second[global_index])
            - float(observations.times_in_second[candidate_global_index]))
        projected_distance = float(np.linalg.norm(
            observations.projected_points_in_meter[global_index]
            - observations.projected_points_in_meter[candidate_global_index]))
        height_distance = abs(
            float(observations.heights_in_meter[global_index])
            - float(observations.heights_in_meter[candidate_global_index]))
        if (
                time_distance <= config.time_radius_in_second
                and projected_distance <= config.projected_radius_in_meter
                and height_distance <= config.height_radius_in_meter
            ):
            exact.append(candidate_local_index)
    return tuple(sorted(exact))


def canonical_source_key(
        normalized: Normalized_Person_Ankle_Input,
        observation_index: int,
    ) -> tuple[str, int, str]:
    observations = normalized.observations
    return (
        observations.run_keys[observation_index],
        int(observations.frame_indices_original[observation_index]),
        observations.observation_keys[observation_index],
    )


def cluster_one_run(
        indices: NDArray[np.int64],
        normalized: Normalized_Person_Ankle_Input,
        config: Person_Ankle_Local_Config,
    ) -> tuple[list[tuple[int, ...]], dict[int, Person_Ankle_Noise_Reason]]:
    coordinates, tree = make_scaled_tree(indices, normalized, config)
    is_core = [False] * len(indices)
    for local_index in range(len(indices)):
        is_core[local_index] = (
            len(query_exact_neighbor_indices(
                local_index, indices, normalized, config, coordinates, tree))
            >= config.minimum_core_neighbor_count
        )
    disjoint_set = Disjoint_Set(len(indices))
    for local_index in range(len(indices)):
        if is_core[local_index]:
            row = query_exact_neighbor_indices(
                local_index, indices, normalized, config, coordinates, tree)
            for neighbor_local_index in row:
                if is_core[neighbor_local_index]:
                    disjoint_set.union(local_index, neighbor_local_index)
    core_components: dict[int, list[int]] = {}
    for local_index, core in enumerate(is_core):
        if core:
            core_components.setdefault(disjoint_set.find(local_index), []).append(local_index)
    component_members = {
        root: list(members)
        for root, members in core_components.items()
    }
    noise: dict[int, Person_Ankle_Noise_Reason] = {}
    for local_index, core in enumerate(is_core):
        if core:
            continue
        row = query_exact_neighbor_indices(
            local_index, indices, normalized, config, coordinates, tree)
        adjacent_roots = {
            disjoint_set.find(neighbor_local_index)
            for neighbor_local_index in row
            if is_core[neighbor_local_index]
        }
        if len(adjacent_roots) == 1:
            root = next(iter(adjacent_roots))
            component_members[root].append(local_index)
        else:
            global_index = int(indices[local_index])
            noise[global_index] = (
                'density_noise' if not adjacent_roots else 'ambiguous_border')
    components = [
        tuple(sorted(
            (int(indices[local_index]) for local_index in members),
            key=lambda value: canonical_source_key(normalized, value),
        ))
        for members in component_members.values()
    ]
    components.sort(key=lambda members: canonical_source_key(normalized, members[0]))
    return components, noise


def safe_spearman(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
    if left.size < 2 or right.size < 2:
        return 0.0
    if float(np.ptp(left)) == 0.0 or float(np.ptp(right)) == 0.0:
        return 0.0
    left_rank = rankdata(left, method='average')
    right_rank = rankdata(right, method='average')
    correlation = float(np.corrcoef(left_rank, right_rank)[0, 1])
    return correlation if math.isfinite(correlation) else 0.0


def foot_path_statistics(
        member_indices: tuple[int, ...],
        normalized: Normalized_Person_Ankle_Input,
    ) -> tuple[
        float,
        tuple[tuple[float, float, float] | None, ...],
        tuple[tuple[float, float, float] | None, ...],
        list[float],
    ]:
    observations = normalized.observations
    total_path_length = 0.0
    first_points: list[tuple[float, float, float] | None] = []
    last_points: list[tuple[float, float, float] | None] = []
    correlations: list[float] = []
    for foot_side in ('left', 'right'):
        foot_indices = [
            index for index in member_indices
            if observations.foot_sides[index] == foot_side
        ]
        foot_indices.sort(key=lambda index: (
            float(observations.times_in_second[index]),
            int(observations.frame_indices_original[index]),
            observations.observation_keys[index],
        ))
        if not foot_indices:
            first_points.append(None)
            last_points.append(None)
            correlations.append(0.0)
            continue
        points = observations.projected_points_in_meter[foot_indices]
        step_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        cumulative = np.concatenate((
            np.zeros(1, dtype=np.float64),
            np.cumsum(step_lengths, dtype=np.float64),
        ))
        total_path_length += float(cumulative[-1])
        first_points.append((
            float(points[0, 0]), float(points[0, 1]), float(points[0, 2])))
        last_points.append((
            float(points[-1, 0]), float(points[-1, 1]), float(points[-1, 2])))
        correlations.append(safe_spearman(
            cumulative,
            observations.heights_in_meter[foot_indices],
        ))
    return total_path_length, tuple(first_points), tuple(last_points), correlations


def classify_foot_shape(
        member_indices: tuple[int, ...],
        normalized: Normalized_Person_Ankle_Input,
        config: Person_Ankle_Local_Config,
    ) -> tuple[Person_Ankle_Foot_Shape, int, int, int]:
    observations = normalized.observations
    left_count = sum(observations.foot_sides[index] == 'left' for index in member_indices)
    right_count = len(member_indices) - left_count
    if right_count == 0:
        return 'left', left_count, right_count, 0
    if left_count == 0:
        return 'right', left_count, right_count, 0
    frame_feet: dict[int, set[str]] = {}
    for index in member_indices:
        frame_feet.setdefault(
            int(observations.frame_indices_original[index]), set()).add(
                observations.foot_sides[index])
    bilateral_count = sum(len(feet) == 2 for feet in frame_feet.values())
    if bilateral_count / len(frame_feet) >= config.bilateral_frame_fraction_min:
        return 'bilateral', left_count, right_count, bilateral_count
    single_feet = [
        next(iter(frame_feet[frame]))
        for frame in sorted(frame_feet)
        if len(frame_feet[frame]) == 1
    ]
    change_fraction = 0.0
    if len(single_feet) >= 2:
        change_fraction = sum(
            left != right for left, right in zip(single_feet[:-1], single_feet[1:])
        ) / (len(single_feet) - 1)
    foot_shape: Person_Ankle_Foot_Shape = (
        'alternating'
        if change_fraction >= config.alternating_change_fraction_min
        else 'mixed'
    )
    return foot_shape, left_count, right_count, bilateral_count


def make_local_cluster(
        local_cluster_id: int,
        member_indices: tuple[int, ...],
        normalized: Normalized_Person_Ankle_Input,
        config: Person_Ankle_Local_Config,
    ) -> Person_Ankle_Local_Cluster:
    observations = normalized.observations
    run_key = observations.run_keys[member_indices[0]]
    run = next(value for value in normalized.runs if value.run_key == run_key)
    heights = observations.heights_in_meter[list(member_indices)]
    sorted_heights = np.sort(heights)
    minimum = float(sorted_heights[0])
    median = float(np.median(sorted_heights))
    maximum = float(sorted_heights[-1])
    percentile_10, percentile_90 = (
        float(value) for value in np.percentile(sorted_heights, (10.0, 90.0)))
    bmad = float(np.median(np.abs(sorted_heights - median)))
    maximum_adjacent_gap = (
        float(np.max(np.diff(sorted_heights))) if len(sorted_heights) > 1 else 0.0)
    chronological = tuple(sorted(member_indices, key=lambda index: (
        float(observations.times_in_second[index]),
        int(observations.frame_indices_original[index]),
        observations.observation_keys[index],
    )))
    frame_indices = sorted({
        int(observations.frame_indices_original[index]) for index in member_indices
    })
    observed_duration = len(frame_indices) / run.fps
    temporal_span = (frame_indices[-1] - frame_indices[0] + 1) / run.fps
    occupancy = observed_duration / temporal_span
    contiguous_run_count = 1 + sum(
        right != left + 1 for left, right in zip(frame_indices[:-1], frame_indices[1:]))
    points = observations.projected_points_in_meter[list(member_indices)]
    centroid = np.mean(points, axis=0)
    radii = np.linalg.norm(points - centroid[None, :], axis=1)
    robust_radius = float(np.percentile(radii, 90.0))
    path_length, first_points, last_points, path_correlations = foot_path_statistics(
        member_indices, normalized)
    foot_shape, left_count, right_count, bilateral_count = classify_foot_shape(
        member_indices, normalized, config)
    if (
            percentile_90 - percentile_10 <= config.band_quantile_width_max_in_meter
            and maximum - minimum <= config.band_full_span_max_in_meter
        ):
        height_shape: Person_Ankle_Height_Shape = 'band'
    else:
        endpoint_count = max(
            1, math.ceil(len(chronological) * config.transition_endpoint_fraction))
        first_height = float(np.median(
            observations.heights_in_meter[list(chronological[:endpoint_count])]))
        last_height = float(np.median(
            observations.heights_in_meter[list(chronological[-endpoint_count:])]))
        time_correlation = safe_spearman(
            observations.times_in_second[list(chronological)],
            observations.heights_in_meter[list(chronological)],
        )
        association = max(
            [abs(time_correlation)] + [abs(value) for value in path_correlations])
        height_shape = (
            'transition'
            if (
                abs(last_height - first_height) >= config.transition_height_min_in_meter
                and association >= config.transition_rank_min
            )
            else 'diffuse'
        )
    time_shape: Person_Ankle_Time_Shape = (
        'persistent'
        if (
            observed_duration >= config.persistent_duration_min_in_second
            and occupancy >= config.persistent_occupancy_min
        )
        else 'episodic'
    )
    if robust_radius <= config.compact_radius_max_in_meter:
        space_shape: Person_Ankle_Space_Shape = 'compact'
    elif path_length >= config.path_length_min_in_meter:
        space_shape = 'path'
    else:
        space_shape = 'dispersed'
    return Person_Ankle_Local_Cluster(
        local_cluster_id=local_cluster_id,
        run_key=run_key,
        time_domain_key=observations.time_domain_keys[member_indices[0]],
        observation_indices=member_indices,
        sample_count=len(member_indices),
        unique_frame_count=len(frame_indices),
        left_count=left_count,
        right_count=right_count,
        bilateral_frame_count=bilateral_count,
        minimum_height_in_meter=minimum,
        median_height_in_meter=median,
        maximum_height_in_meter=maximum,
        percentile_10_height_in_meter=percentile_10,
        percentile_90_height_in_meter=percentile_90,
        bmad_in_meter=bmad,
        maximum_adjacent_height_gap_in_meter=maximum_adjacent_gap,
        first_time_in_second=float(observations.times_in_second[chronological[0]]),
        last_time_in_second=float(observations.times_in_second[chronological[-1]]),
        observed_duration_in_second=observed_duration,
        temporal_span_in_second=temporal_span,
        temporal_occupancy=occupancy,
        contiguous_frame_run_count=contiguous_run_count,
        projected_centroid_in_meter=(
            float(centroid[0]), float(centroid[1]), float(centroid[2])),
        robust_radius_in_meter=robust_radius,
        chronological_path_length_in_meter=path_length,
        first_projected_by_foot=first_points,
        last_projected_by_foot=last_points,
        height_shape=height_shape,
        time_shape=time_shape,
        space_shape=space_shape,
        foot_shape=foot_shape,
    )


def cluster_person_ankle_observations(
        normalized: Normalized_Person_Ankle_Input,
        config: Person_Ankle_Local_Config,
    ) -> Local_Cluster_Stage_Result:
    '''Cluster each run independently and retain every observation and reason.'''
    observation_count = len(normalized.observations.observation_keys)
    labels = np.full(observation_count, -1, dtype=np.int64)
    noise_reasons: list[Person_Ankle_Noise_Reason | None] = [None] * observation_count
    components: list[tuple[int, ...]] = []
    for run in normalized.runs:
        run_indices = np.asarray([
            index
            for index, run_key in enumerate(normalized.observations.run_keys)
            if run_key == run.run_key
        ], dtype=np.int64)
        run_components, run_noise = cluster_one_run(
            run_indices, normalized, config)
        components.extend(run_components)
        for index, reason in run_noise.items():
            noise_reasons[index] = reason
    components.sort(key=lambda members: canonical_source_key(normalized, members[0]))
    clusters: list[Person_Ankle_Local_Cluster] = []
    for cluster_id, members in enumerate(components):
        labels[list(members)] = cluster_id
        clusters.append(make_local_cluster(
            cluster_id, members, normalized, config))
    for index in range(observation_count):
        if labels[index] < 0 and noise_reasons[index] is None:
            noise_reasons[index] = 'density_noise'
    source = normalized.observations
    observations = Person_Ankle_Observation_Table(
        observation_keys=source.observation_keys,
        run_keys=source.run_keys,
        time_domain_keys=source.time_domain_keys,
        frame_indices_original=source.frame_indices_original,
        foot_sides=source.foot_sides,
        times_in_second=source.times_in_second,
        world_points_in_meter=source.world_points_in_meter,
        projected_points_in_meter=source.projected_points_in_meter,
        heights_in_meter=source.heights_in_meter,
        local_cluster_labels=make_readonly_int_array(labels),
        noise_reasons=tuple(noise_reasons),
    )
    return Local_Cluster_Stage_Result(observations, tuple(clusters))


def mark_incompatible_observations(
        normalized: Normalized_Person_Ankle_Input,
    ) -> Person_Ankle_Observation_Table:
    source = normalized.observations
    count = len(source.observation_keys)
    return Person_Ankle_Observation_Table(
        observation_keys=source.observation_keys,
        run_keys=source.run_keys,
        time_domain_keys=source.time_domain_keys,
        frame_indices_original=source.frame_indices_original,
        foot_sides=source.foot_sides,
        times_in_second=source.times_in_second,
        world_points_in_meter=source.world_points_in_meter,
        projected_points_in_meter=source.projected_points_in_meter,
        heights_in_meter=source.heights_in_meter,
        local_cluster_labels=make_readonly_int_array(
            np.full(count, -1, dtype=np.int64)),
        noise_reasons=tuple('incompatible_reference_planes' for _ in range(count)),
    )


__all__ = [
    'Local_Cluster_Stage_Result',
    'cluster_person_ankle_observations',
    'exact_neighbors_for_run',
    'mark_incompatible_observations',
]
