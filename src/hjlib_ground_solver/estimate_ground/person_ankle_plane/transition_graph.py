'''Transition evidence retained for the temporarily deprecated ankle-plane V1.'''

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
import itertools
import math

from hjlib_ground_solver.estimate_ground.person_ankle_plane.contract import (
    Person_Ankle_Height_Hypothesis,
    Person_Ankle_Hypothesis_Segment,
    Person_Ankle_Local_Cluster,
    Person_Ankle_Observation_Table,
    Person_Ankle_Transition_Check,
    Person_Ankle_Transition_Config,
)


@dataclass(frozen=True, slots=True)
class Transition_Graph_Stage_Result:
    segments: tuple[Person_Ankle_Hypothesis_Segment, ...]
    transition_checks: tuple[Person_Ankle_Transition_Check, ...]
    coherent_plane_switch: bool


@dataclass(frozen=True, slots=True)
class Segment_Detail:
    segment: Person_Ankle_Hypothesis_Segment
    first_points_by_foot: tuple[tuple[float, float, float] | None, ...]
    last_points_by_foot: tuple[tuple[float, float, float] | None, ...]
    first_local_cluster_ids: tuple[int, ...]
    last_local_cluster_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Frame_Observation_Index:
    frames_by_run: dict[str, tuple[int, ...]]
    observation_indices_by_run_frame: dict[tuple[str, int], tuple[int, ...]]


def build_frame_observation_index(
        observations: Person_Ankle_Observation_Table,
    ) -> Frame_Observation_Index:
    mutable: dict[tuple[str, int], list[int]] = {}
    run_order: list[str] = []
    seen_runs: set[str] = set()
    for observation_index, run_key in enumerate(observations.run_keys):
        if run_key not in seen_runs:
            run_order.append(run_key)
            seen_runs.add(run_key)
        frame = int(observations.frame_indices_original[observation_index])
        mutable.setdefault((run_key, frame), []).append(observation_index)
    frames_by_run = {
        run_key: tuple(sorted(
            frame for candidate_run, frame in mutable
            if candidate_run == run_key
        ))
        for run_key in run_order
    }
    return Frame_Observation_Index(
        frames_by_run=frames_by_run,
        observation_indices_by_run_frame={
            key: tuple(value) for key, value in mutable.items()
        },
    )


def frame_hypothesis_sets(
        observations: Person_Ankle_Observation_Table,
        local_clusters: tuple[Person_Ankle_Local_Cluster, ...],
        local_cluster_to_hypothesis: tuple[int, ...],
    ) -> dict[str, dict[int, set[int]]]:
    output: dict[str, dict[int, set[int]]] = {}
    cluster_by_id = {
        cluster.local_cluster_id: cluster for cluster in local_clusters
    }
    for observation_index, local_cluster_id_raw in enumerate(
            observations.local_cluster_labels):
        local_cluster_id = int(local_cluster_id_raw)
        if local_cluster_id < 0:
            continue
        if cluster_by_id[local_cluster_id].height_shape != 'band':
            continue
        hypothesis_id = local_cluster_to_hypothesis[local_cluster_id]
        run_key = observations.run_keys[observation_index]
        frame = int(observations.frame_indices_original[observation_index])
        output.setdefault(run_key, {}).setdefault(frame, set()).add(hypothesis_id)
    return output


def endpoint_detail(
        observation_indices: tuple[int, ...],
        frame_index: int,
        observations: Person_Ankle_Observation_Table,
    ) -> tuple[
        tuple[tuple[float, float, float] | None, ...],
        tuple[int, ...],
    ]:
    points: list[tuple[float, float, float] | None] = []
    cluster_ids: set[int] = set()
    for foot_side in ('left', 'right'):
        candidates = [
            index for index in observation_indices
            if (
                int(observations.frame_indices_original[index]) == frame_index
                and observations.foot_sides[index] == foot_side
            )
        ]
        if candidates:
            point = observations.projected_points_in_meter[candidates[0]]
            points.append((float(point[0]), float(point[1]), float(point[2])))
            cluster_ids.update(
                int(observations.local_cluster_labels[index]) for index in candidates)
        else:
            points.append(None)
    return tuple(points), tuple(sorted(cluster_ids))


def build_segments(
        observations: Person_Ankle_Observation_Table,
        frame_sets: dict[str, dict[int, set[int]]],
        observation_index: Frame_Observation_Index,
        local_cluster_to_hypothesis: tuple[int, ...],
        config: Person_Ankle_Transition_Config,
    ) -> tuple[Segment_Detail, ...]:
    details: list[Segment_Detail] = []
    next_segment_id = 0
    for run_key, frames in observation_index.frames_by_run.items():
        active_frames: list[int] = []
        active_observation_indices: list[int] = []
        active_hypothesis: int | None = None
        previous_time: float | None = None

        def emit_active() -> None:
            nonlocal next_segment_id
            nonlocal active_frames, active_observation_indices, active_hypothesis
            if active_hypothesis is None or not active_frames:
                return
            indices = tuple(active_observation_indices)
            first_frame = active_frames[0]
            last_frame = active_frames[-1]
            first_indices = [
                index for index in indices
                if int(observations.frame_indices_original[index]) == first_frame
            ]
            last_indices = [
                index for index in indices
                if int(observations.frame_indices_original[index]) == last_frame
            ]
            segment = Person_Ankle_Hypothesis_Segment(
                segment_id=next_segment_id,
                run_key=run_key,
                hypothesis_id=active_hypothesis,
                first_frame_index_original=first_frame,
                last_frame_index_original=last_frame,
                first_time_in_second=min(
                    float(observations.times_in_second[index]) for index in first_indices),
                last_time_in_second=max(
                    float(observations.times_in_second[index]) for index in last_indices),
                observation_indices=indices,
            )
            first_points, first_cluster_ids = endpoint_detail(
                indices, first_frame, observations)
            last_points, last_cluster_ids = endpoint_detail(
                indices, last_frame, observations)
            details.append(Segment_Detail(
                segment=segment,
                first_points_by_foot=first_points,
                last_points_by_foot=last_points,
                first_local_cluster_ids=first_cluster_ids,
                last_local_cluster_ids=last_cluster_ids,
            ))
            next_segment_id += 1
            active_frames = []
            active_observation_indices = []
            active_hypothesis = None

        for frame in frames:
            hypotheses = frame_sets.get(run_key, {}).get(frame, set())
            frame_indices = observation_index.observation_indices_by_run_frame[
                (run_key, frame)]
            frame_time = min(
                float(observations.times_in_second[index]) for index in frame_indices)
            label = next(iter(hypotheses)) if len(hypotheses) == 1 else None
            hypothesis_indices = [
                index for index in frame_indices
                if (
                    int(observations.local_cluster_labels[index]) >= 0
                    and local_cluster_to_hypothesis[
                        int(observations.local_cluster_labels[index])]
                    == label
                )
            ]
            continues = (
                label is not None
                and label == active_hypothesis
                and previous_time is not None
                and frame_time - previous_time <= config.segment_time_gap_max_in_second
            )
            if label is None:
                emit_active()
            elif active_hypothesis is None:
                active_hypothesis = label
                active_frames = [frame]
                active_observation_indices = hypothesis_indices
            elif continues:
                active_frames.append(frame)
                active_observation_indices.extend(hypothesis_indices)
            else:
                emit_active()
                active_hypothesis = label
                active_frames = [frame]
                active_observation_indices = hypothesis_indices
            previous_time = frame_time
        emit_active()
    return tuple(details)


def endpoint_distance(
        source: Segment_Detail,
        target: Segment_Detail,
    ) -> tuple[float, bool]:
    same_foot_distances = [
        math.dist(left, right)
        for left, right in zip(
            source.last_points_by_foot, target.first_points_by_foot)
        if left is not None and right is not None
    ]
    if same_foot_distances:
        return min(same_foot_distances), False
    any_distances = [
        math.dist(left, right)
        for left in source.last_points_by_foot if left is not None
        for right in target.first_points_by_foot if right is not None
    ]
    return min(any_distances), True


def cluster_endpoint_is_adjacent(
        cluster: Person_Ankle_Local_Cluster,
        source: Segment_Detail,
        target: Segment_Detail,
        config: Person_Ankle_Transition_Config,
    ) -> bool:
    first_points = [
        point for point in cluster.first_projected_by_foot if point is not None
    ]
    last_points = [
        point for point in cluster.last_projected_by_foot if point is not None
    ]
    source_points = [
        point for point in source.last_points_by_foot if point is not None
    ]
    target_points = [
        point for point in target.first_points_by_foot if point is not None
    ]
    source_distance = min(
        math.dist(left, right) for left in source_points for right in first_points)
    target_distance = min(
        math.dist(left, right) for left in last_points for right in target_points)
    return (
        abs(cluster.first_time_in_second - source.segment.last_time_in_second)
        <= config.transition_time_max_in_second
        and abs(target.segment.first_time_in_second - cluster.last_time_in_second)
        <= config.transition_time_max_in_second
        and source_distance <= config.transition_distance_max_in_meter
        and target_distance <= config.transition_distance_max_in_meter
    )


def make_transition_check(
        source: Segment_Detail,
        target: Segment_Detail,
        frame_sets: dict[str, dict[int, set[int]]],
        observation_index: Frame_Observation_Index,
        observations: Person_Ankle_Observation_Table,
        local_clusters: tuple[Person_Ankle_Local_Cluster, ...],
        config: Person_Ankle_Transition_Config,
    ) -> Person_Ankle_Transition_Check:
    source_hypothesis = source.segment.hypothesis_id
    target_hypothesis = target.segment.hypothesis_id
    run_frames = observation_index.frames_by_run[source.segment.run_key]
    between_frames = run_frames[
        bisect_right(run_frames, source.segment.last_frame_index_original):
        bisect_left(run_frames, target.segment.first_frame_index_original)
    ]
    overlap_frames = [
        frame for frame in between_frames
        if len(frame_sets[source.segment.run_key].get(frame, set())) > 1
    ]
    required_pair = {source_hypothesis, target_hypothesis}
    invalid_overlap = [
        frame for frame in overlap_frames
        if frame_sets[source.segment.run_key][frame] != required_pair
    ]
    endpoint_time_gap = (
        target.segment.first_time_in_second - source.segment.last_time_in_second)
    distance, uses_cross_foot = endpoint_distance(source, target)
    if invalid_overlap:
        return Person_Ankle_Transition_Check(
            source.segment.segment_id, target.segment.segment_id,
            source_hypothesis, target_hypothesis, tuple(overlap_frames),
            endpoint_time_gap, distance, uses_cross_foot, (), 'vetoed',
            'overlap contains a third or mismatched hypothesis',
        )
    if overlap_frames:
        overlap_indices = [
            index
            for frame in overlap_frames
            for index in observation_index.observation_indices_by_run_frame[
                (source.segment.run_key, frame)]
        ]
        overlap_duration = (
            max(float(observations.times_in_second[index]) for index in overlap_indices)
            - min(float(observations.times_in_second[index]) for index in overlap_indices)
        )
        if overlap_duration > config.transition_overlap_duration_max_in_second:
            return Person_Ankle_Transition_Check(
                source.segment.segment_id, target.segment.segment_id,
                source_hypothesis, target_hypothesis, tuple(overlap_frames),
                endpoint_time_gap, distance, uses_cross_foot, (), 'vetoed',
                'two-hypothesis overlap duration exceeds the configured maximum',
            )
    cluster_by_id = {
        cluster.local_cluster_id: cluster for cluster in local_clusters
    }
    intervening_ids = sorted({
        int(observations.local_cluster_labels[index])
        for frame in between_frames
        for index in observation_index.observation_indices_by_run_frame[
            (source.segment.run_key, frame)]
        if (
            int(observations.local_cluster_labels[index]) >= 0
            and cluster_by_id[int(observations.local_cluster_labels[index])].height_shape
            != 'band'
        )
    })
    rejection: str | None = None
    if endpoint_time_gap > config.transition_time_max_in_second:
        rejection = 'segment endpoint time gap exceeds the configured maximum'
    elif distance > config.transition_distance_max_in_meter:
        rejection = 'segment endpoint distance exceeds the configured maximum'
    elif any(cluster_by_id[cluster_id].height_shape != 'transition'
             for cluster_id in intervening_ids):
        rejection = 'an intervening non-band cluster is not typed transition'
    elif any(not cluster_endpoint_is_adjacent(
            cluster_by_id[cluster_id], source, target, config)
            for cluster_id in intervening_ids):
        rejection = 'an intervening transition cluster does not join both endpoints'
    outcome = 'valid' if rejection is None else 'invalid'
    return Person_Ankle_Transition_Check(
        source.segment.segment_id, target.segment.segment_id,
        source_hypothesis, target_hypothesis, tuple(overlap_frames),
        endpoint_time_gap, distance, uses_cross_foot, tuple(intervening_ids),
        outcome, rejection,
    )


def hypothesis_has_persistent_incidence(
        hypothesis: Person_Ankle_Height_Hypothesis,
        valid_details: list[tuple[Segment_Detail, Segment_Detail]],
        config: Person_Ankle_Transition_Config,
    ) -> bool:
    persistent_episode_cluster_ids = {
        cluster_id
        for episode in hypothesis.episodes
        if episode.observed_duration_in_second
        >= config.switch_episode_duration_min_in_second
        for cluster_id in episode.local_cluster_ids
    }
    for source, target in valid_details:
        if (
                source.segment.hypothesis_id == hypothesis.hypothesis_id
                and persistent_episode_cluster_ids.intersection(
                    source.last_local_cluster_ids)
            ):
            return True
        if (
                target.segment.hypothesis_id == hypothesis.hypothesis_id
                and persistent_episode_cluster_ids.intersection(
                    target.first_local_cluster_ids)
            ):
            return True
    return False


def graph_is_connected(
        hypothesis_ids: set[int],
        valid_pairs: list[tuple[int, int]],
    ) -> bool:
    if not hypothesis_ids:
        return False
    reached = {min(hypothesis_ids)}
    while True:
        expanded = reached | {
            right for left, right in valid_pairs if left in reached
        } | {
            left for left, right in valid_pairs if right in reached
        }
        if expanded == reached:
            return reached == hypothesis_ids
        reached = expanded


def build_person_transition_graph(
        observations: Person_Ankle_Observation_Table,
        local_clusters: tuple[Person_Ankle_Local_Cluster, ...],
        hypotheses: tuple[Person_Ankle_Height_Hypothesis, ...],
        local_cluster_to_hypothesis: tuple[int, ...],
        config: Person_Ankle_Transition_Config,
        sigma_floor_in_meter: float,
    ) -> Transition_Graph_Stage_Result:
    '''Build all chronological checks and evaluate multi-level coherence.'''
    frame_sets = frame_hypothesis_sets(
        observations, local_clusters, local_cluster_to_hypothesis)
    observation_index = build_frame_observation_index(observations)
    details = build_segments(
        observations, frame_sets, observation_index,
        local_cluster_to_hypothesis, config)
    checks: list[Person_Ankle_Transition_Check] = []
    detail_pairs: list[tuple[Segment_Detail, Segment_Detail]] = []
    for source, target in zip(details[:-1], details[1:]):
        if source.segment.run_key != target.segment.run_key:
            continue
        if source.segment.hypothesis_id == target.segment.hypothesis_id:
            continue
        check = make_transition_check(
            source, target, frame_sets, observation_index,
            observations, local_clusters, config)
        checks.append(check)
        detail_pairs.append((source, target))
    recurrent = tuple(hypothesis for hypothesis in hypotheses if hypothesis.recurrent)
    if len(recurrent) < 2:
        return Transition_Graph_Stage_Result(
            tuple(detail.segment for detail in details), tuple(checks), False)
    separated = all(
        abs(left.center_height_in_meter - right.center_height_in_meter)
        >= config.switch_height_gap_min_in_meter
        and abs(left.center_height_in_meter - right.center_height_in_meter)
        >= config.switch_scale_multiplier * max(
            left.bmad_in_meter, right.bmad_in_meter, sigma_floor_in_meter)
        for left, right in itertools.combinations(recurrent, 2)
    )
    attempted = [check for check in checks if check.outcome != 'vetoed']
    valid = [check for check in checks if check.outcome == 'valid']
    valid_details = [
        pair for check, pair in zip(checks, detail_pairs) if check.outcome == 'valid'
    ]
    recurrent_ids = {hypothesis.hypothesis_id for hypothesis in recurrent}
    incidence = all(
        hypothesis_has_persistent_incidence(hypothesis, valid_details, config)
        for hypothesis in recurrent
    )
    valid_fraction = len(valid) / len(attempted) if attempted else 0.0
    valid_pairs = [
        (check.source_hypothesis_id, check.target_hypothesis_id)
        for check in valid
        if (
            check.source_hypothesis_id in recurrent_ids
            and check.target_hypothesis_id in recurrent_ids
        )
    ]
    coherent = (
        separated
        and incidence
        and bool(attempted)
        and valid_fraction >= config.transition_valid_fraction_min
        and graph_is_connected(recurrent_ids, valid_pairs)
    )
    return Transition_Graph_Stage_Result(
        tuple(detail.segment for detail in details), tuple(checks), coherent)


__all__ = [
    'Transition_Graph_Stage_Result',
    'build_person_transition_graph',
]
