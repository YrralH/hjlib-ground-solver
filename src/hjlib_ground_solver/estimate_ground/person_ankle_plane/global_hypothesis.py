'''Person-global hypotheses retained for the temporarily deprecated V1.'''

from dataclasses import dataclass
import itertools
import math

import numpy as np

from hjlib_ground_solver.estimate_ground.person_ankle_plane.contract import (
    Person_Ankle_Global_Config,
    Person_Ankle_Height_Hypothesis,
    Person_Ankle_Local_Cluster,
    Person_Ankle_Observation_Table,
    Person_Ankle_Recurrence_Episode,
)
from hjlib_ground_solver.estimate_ground.person_ankle_plane.local_cluster import (
    Disjoint_Set,
)
from hjlib_ground_solver.estimate_ground.person_ankle_plane.normalize import (
    Normalized_Person_Ankle_Input,
)


@dataclass(frozen=True, slots=True)
class Global_Hypothesis_Stage_Result:
    hypotheses: tuple[Person_Ankle_Height_Hypothesis, ...]
    local_cluster_to_hypothesis: tuple[int, ...]


def pair_merge_distance(
        left: Person_Ankle_Local_Cluster,
        right: Person_Ankle_Local_Cluster,
        config: Person_Ankle_Global_Config,
    ) -> float:
    gap = abs(left.median_height_in_meter - right.median_height_in_meter)
    sigma = max(
        config.sigma_floor_in_meter,
        math.sqrt(left.bmad_in_meter ** 2 + right.bmad_in_meter ** 2),
    )
    return max(
        gap / sigma / config.merge_z_max,
        gap / config.merge_height_gap_max_in_meter,
    )


def group_merge_distance(
        left_ids: tuple[int, ...],
        right_ids: tuple[int, ...],
        clusters_by_id: dict[int, Person_Ankle_Local_Cluster],
        config: Person_Ankle_Global_Config,
    ) -> float:
    return max(
        pair_merge_distance(
            clusters_by_id[left_id], clusters_by_id[right_id], config)
        for left_id in left_ids
        for right_id in right_ids
    )


def complete_link_groups(
        band_clusters: tuple[Person_Ankle_Local_Cluster, ...],
        config: Person_Ankle_Global_Config,
    ) -> tuple[tuple[int, ...], ...]:
    clusters_by_id = {
        cluster.local_cluster_id: cluster for cluster in band_clusters
    }
    groups: list[tuple[int, ...]] = [
        (cluster.local_cluster_id,) for cluster in band_clusters
    ]
    groups.sort()
    while True:
        candidates: list[
            tuple[float, tuple[int, ...], tuple[int, ...], int, int]
        ] = []
        for left_index in range(len(groups)):
            for right_index in range(left_index + 1, len(groups)):
                left = groups[left_index]
                right = groups[right_index]
                distance = group_merge_distance(
                    left, right, clusters_by_id, config)
                if distance <= 1.0:
                    candidates.append((
                        distance, left, right, left_index, right_index))
        if not candidates:
            break
        _, left, right, left_index, right_index = min(candidates)
        merged = tuple(sorted(left + right))
        groups = [
            group for index, group in enumerate(groups)
            if index not in (left_index, right_index)
        ]
        groups.append(merged)
        groups.sort()
    return tuple(groups)


def endpoint_time_distance(
        left: Person_Ankle_Local_Cluster,
        right: Person_Ankle_Local_Cluster,
    ) -> tuple[float, float]:
    time_distance = min(
        abs(left_time - right_time)
        for left_time in (left.first_time_in_second, left.last_time_in_second)
        for right_time in (right.first_time_in_second, right.last_time_in_second)
    )
    left_points = [
        point for point in left.first_projected_by_foot + left.last_projected_by_foot
        if point is not None
    ]
    right_points = [
        point for point in right.first_projected_by_foot + right.last_projected_by_foot
        if point is not None
    ]
    projected_distance = min(
        math.dist(left_point, right_point)
        for left_point in left_points
        for right_point in right_points
    )
    return time_distance, projected_distance


def runs_are_episode_neighbors(
        left: Person_Ankle_Local_Cluster,
        right: Person_Ankle_Local_Cluster,
        normalized: Normalized_Person_Ankle_Input,
    ) -> bool:
    if left.run_key == right.run_key:
        return True
    if left.time_domain_key != right.time_domain_key:
        return False
    domain_runs = [
        run.run_key for run in normalized.runs
        if run.time_domain_key == left.time_domain_key
    ]
    return abs(domain_runs.index(left.run_key) - domain_runs.index(right.run_key)) == 1


def make_recurrence_episodes(
        cluster_ids: tuple[int, ...],
        clusters_by_id: dict[int, Person_Ankle_Local_Cluster],
        normalized: Normalized_Person_Ankle_Input,
        observations: Person_Ankle_Observation_Table,
        config: Person_Ankle_Global_Config,
    ) -> tuple[Person_Ankle_Recurrence_Episode, ...]:
    disjoint_set = Disjoint_Set(len(cluster_ids))
    for left_index, right_index in itertools.combinations(range(len(cluster_ids)), 2):
        left = clusters_by_id[cluster_ids[left_index]]
        right = clusters_by_id[cluster_ids[right_index]]
        if not runs_are_episode_neighbors(left, right, normalized):
            continue
        time_distance, projected_distance = endpoint_time_distance(left, right)
        if (
                time_distance <= config.episode_stitch_time_max_in_second
                and projected_distance <= config.episode_stitch_distance_max_in_meter
            ):
            disjoint_set.union(left_index, right_index)
    groups: dict[int, list[int]] = {}
    for index, cluster_id in enumerate(cluster_ids):
        groups.setdefault(disjoint_set.find(index), []).append(cluster_id)
    canonical_groups = sorted(tuple(sorted(value)) for value in groups.values())
    episodes: list[Person_Ankle_Recurrence_Episode] = []
    for episode_id, group in enumerate(canonical_groups):
        observed_duration = unique_observed_duration(
            group, clusters_by_id, observations, normalized)
        observation_indices = sorted({
            observation_index
            for cluster_id in group
            for observation_index in clusters_by_id[cluster_id].observation_indices
        })
        midpoint = float(np.mean(observations.times_in_second[observation_indices]))
        centroid = np.mean(
            observations.projected_points_in_meter[observation_indices], axis=0)
        episodes.append(Person_Ankle_Recurrence_Episode(
            episode_id=episode_id,
            local_cluster_ids=group,
            observed_duration_in_second=observed_duration,
            midpoint_time_in_second=midpoint,
            projected_centroid_in_meter=(
                float(centroid[0]), float(centroid[1]), float(centroid[2])),
        ))
    return tuple(episodes)


def maximum_episode_separations(
        episodes: tuple[Person_Ankle_Recurrence_Episode, ...],
        clusters_by_id: dict[int, Person_Ankle_Local_Cluster],
    ) -> tuple[float, float]:
    temporal = 0.0
    projected = 0.0
    for left, right in itertools.combinations(episodes, 2):
        left_domain = clusters_by_id[left.local_cluster_ids[0]].time_domain_key
        right_domain = clusters_by_id[right.local_cluster_ids[0]].time_domain_key
        if left_domain == right_domain:
            temporal = max(
                temporal,
                abs(left.midpoint_time_in_second - right.midpoint_time_in_second),
            )
        projected = max(
            projected,
            math.dist(
                left.projected_centroid_in_meter,
                right.projected_centroid_in_meter,
            ),
        )
    return temporal, projected


def unique_observed_duration(
        cluster_ids: tuple[int, ...],
        clusters_by_id: dict[int, Person_Ankle_Local_Cluster],
        observations: Person_Ankle_Observation_Table,
        normalized: Normalized_Person_Ankle_Input,
    ) -> float:
    frames_by_run: dict[str, set[int]] = {}
    for cluster_id in cluster_ids:
        cluster = clusters_by_id[cluster_id]
        for observation_index in cluster.observation_indices:
            frames_by_run.setdefault(cluster.run_key, set()).add(
                int(observations.frame_indices_original[observation_index]))
    fps_by_run = {run.run_key: run.fps for run in normalized.runs}
    return sum(
        len(frame_indices) / fps_by_run[run_key]
        for run_key, frame_indices in frames_by_run.items()
    )


def build_person_height_hypotheses(
        normalized: Normalized_Person_Ankle_Input,
        observations: Person_Ankle_Observation_Table,
        local_clusters: tuple[Person_Ankle_Local_Cluster, ...],
        config: Person_Ankle_Global_Config,
    ) -> Global_Hypothesis_Stage_Result:
    '''Complete-link local bands and retain recurrence and competitor evidence.'''
    band_clusters = tuple(
        cluster for cluster in local_clusters if cluster.height_shape == 'band')
    mapping = [-1] * len(local_clusters)
    if not band_clusters:
        return Global_Hypothesis_Stage_Result((), tuple(mapping))
    groups = complete_link_groups(band_clusters, config)
    clusters_by_id = {
        cluster.local_cluster_id: cluster for cluster in local_clusters
    }
    hypotheses: list[Person_Ankle_Height_Hypothesis] = []
    for hypothesis_id, group in enumerate(groups):
        for cluster_id in group:
            mapping[cluster_id] = hypothesis_id
        medians = np.asarray([
            clusters_by_id[cluster_id].median_height_in_meter
            for cluster_id in group
        ], dtype=np.float64)
        center = float(np.median(medians))
        bmad = float(np.median(np.abs(medians - center)))
        sample_counts = np.asarray([
            clusters_by_id[cluster_id].sample_count for cluster_id in group
        ], dtype=np.float64)
        sample_weighted_center = float(np.average(medians, weights=sample_counts))
        observed_duration = unique_observed_duration(
            group, clusters_by_id, observations, normalized)
        episodes = make_recurrence_episodes(
            group, clusters_by_id, normalized, observations, config)
        temporal_separation, projected_separation = maximum_episode_separations(
            episodes, clusters_by_id)
        recurrence_alternative = (
            temporal_separation >= config.recurrent_time_separation_min_in_second
            or projected_separation >= config.recurrent_distance_min_in_meter
            or any(
                clusters_by_id[cluster_id].foot_shape in ('bilateral', 'alternating')
                for cluster_id in group
            )
        )
        recurrent = (
            len(episodes) >= config.recurrent_episode_count_min
            and observed_duration >= config.recurrent_observed_duration_min_in_second
            and recurrence_alternative
        )
        materially_supported = (
            not recurrent
            and len(group) >= config.competitor_cluster_count_min
            and observed_duration >= config.competitor_observed_duration_min_in_second
        )
        hypotheses.append(Person_Ankle_Height_Hypothesis(
            hypothesis_id=hypothesis_id,
            local_cluster_ids=group,
            center_height_in_meter=center,
            bmad_in_meter=bmad,
            sample_weighted_center_in_meter=sample_weighted_center,
            observed_duration_in_second=observed_duration,
            episodes=episodes,
            temporal_separation_in_second=temporal_separation,
            projected_separation_in_meter=projected_separation,
            recurrent=recurrent,
            materially_supported_competitor=materially_supported,
        ))
    return Global_Hypothesis_Stage_Result(tuple(hypotheses), tuple(mapping))


__all__ = [
    'Global_Hypothesis_Stage_Result',
    'build_person_height_hypotheses',
    'complete_link_groups',
]
