'''Orchestration retained for the temporarily deprecated ankle-plane V1.'''

from hjlib_ground_solver.estimate_ground.person_ankle_plane.contract import (
    Person_Ankle_Plane_Config,
    Person_Ankle_Height_Hypothesis,
    Person_Ankle_Plane_Input,
    Person_Ankle_Plane_Result,
    Person_Ankle_Plane_Status,
    Person_Ankle_Run_Provenance,
)
from hjlib_ground_solver.estimate_ground.person_ankle_plane.global_hypothesis import (
    build_person_height_hypotheses,
)
from hjlib_ground_solver.estimate_ground.person_ankle_plane.local_cluster import (
    cluster_person_ankle_observations,
    mark_incompatible_observations,
)
from hjlib_ground_solver.estimate_ground.person_ankle_plane.normalize import (
    Normalized_Person_Ankle_Input,
    normalize_person_ankle_input,
)
from hjlib_ground_solver.estimate_ground.person_ankle_plane.transition_graph import (
    build_person_transition_graph,
)


def run_provenance(
        normalized: Normalized_Person_Ankle_Input,
    ) -> tuple[Person_Ankle_Run_Provenance, ...]:
    return tuple(Person_Ankle_Run_Provenance(
        run_key=run.run_key,
        time_domain_key=run.time_domain_key,
        plane_key=run.plane_key,
        fps=run.fps,
        frame_count=len(run.frame_indices_original),
        first_frame_index_original=int(run.frame_indices_original[0]),
        last_frame_index_original=int(run.frame_indices_original[-1]),
    ) for run in normalized.runs)


def infer_person_ankle_plane_distribution(
        person_input: Person_Ankle_Plane_Input,
        config: Person_Ankle_Plane_Config,
    ) -> Person_Ankle_Plane_Result:
    '''Infer typed ankle-height evidence without forcing a scalar answer.'''
    normalized = normalize_person_ankle_input(person_input, config)
    if normalized.incompatible_plane_key is not None:
        return Person_Ankle_Plane_Result(
            person_key=normalized.person_key,
            config=config,
            runs=run_provenance(normalized),
            source_planes=normalized.source_planes,
            reference_plane_key=normalized.reference_plane_key,
            reference_ground_param_world=normalized.reference_ground_param_world,
            reference_body_side_sign=normalized.reference_body_side_sign,
            observations=mark_incompatible_observations(normalized),
            local_clusters=(),
            hypotheses=(),
            segments=(),
            transition_checks=(),
            status='incompatible_reference_planes',
            candidate_height_in_meter=None,
            incompatible_plane_key=normalized.incompatible_plane_key,
            incompatible_plane_angle_in_radian=(
                normalized.incompatible_plane_angle_in_radian),
            incompatible_plane_residual_in_meter=(
                normalized.incompatible_plane_residual_in_meter),
        )
    local = cluster_person_ankle_observations(normalized, config.local)
    global_stage = build_person_height_hypotheses(
        normalized,
        local.observations,
        local.local_clusters,
        config.global_config,
    )
    transition = build_person_transition_graph(
        local.observations,
        local.local_clusters,
        global_stage.hypotheses,
        global_stage.local_cluster_to_hypothesis,
        config.transition,
        config.global_config.sigma_floor_in_meter,
    )
    recurrent: list[Person_Ankle_Height_Hypothesis] = [
        hypothesis for hypothesis in global_stage.hypotheses
        if hypothesis.recurrent
    ]
    candidate: float | None = None
    if not global_stage.hypotheses:
        status: Person_Ankle_Plane_Status = 'no_ground_evidence'
    elif not recurrent:
        status = 'local_episode_ambiguous'
    elif len(recurrent) >= 2:
        status = (
            'plane_switch'
            if transition.coherent_plane_switch
            else 'multi_layer_ambiguous'
        )
    else:
        has_competitor = any(
            hypothesis.materially_supported_competitor
            for hypothesis in global_stage.hypotheses
            if hypothesis.hypothesis_id != recurrent[0].hypothesis_id
        )
        if has_competitor:
            status = 'multi_layer_ambiguous'
        else:
            status = 'single_support_plane'
            candidate = recurrent[0].center_height_in_meter
    return Person_Ankle_Plane_Result(
        person_key=normalized.person_key,
        config=config,
        runs=run_provenance(normalized),
        source_planes=normalized.source_planes,
        reference_plane_key=normalized.reference_plane_key,
        reference_ground_param_world=normalized.reference_ground_param_world,
        reference_body_side_sign=normalized.reference_body_side_sign,
        observations=local.observations,
        local_clusters=local.local_clusters,
        hypotheses=global_stage.hypotheses,
        segments=transition.segments,
        transition_checks=transition.transition_checks,
        status=status,
        candidate_height_in_meter=candidate,
        incompatible_plane_key=None,
        incompatible_plane_angle_in_radian=None,
        incompatible_plane_residual_in_meter=None,
    )


__all__ = ['infer_person_ankle_plane_distribution']
