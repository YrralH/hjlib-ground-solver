'''Regression tests for the temporarily deprecated ankle-plane V1.'''

from dataclasses import FrozenInstanceError, replace
import math
from typing import Callable

import numpy as np

import hjlib_ground_solver as package_root
import hjlib_ground_solver.estimate_ground as estimate_subpackage
from hjlib_ground_solver import (
    Person_Ankle_Global_Config,
    Person_Ankle_Local_Config,
    Person_Ankle_Plane_Config,
    Person_Ankle_Plane_Input,
    Person_Ankle_Plane_Result,
    Person_Ankle_Plane_Status,
    Person_Ankle_Run,
    Person_Ankle_Source_Plane,
    Person_Ankle_Transition_Config,
    infer_person_ankle_plane_distribution,
)
from hjlib_ground_solver.estimate_ground.person_ankle_plane.local_cluster import (
    exact_neighbors_for_run,
)
from hjlib_ground_solver.estimate_ground.person_ankle_plane.normalize import (
    normalize_person_ankle_input,
)


def expect_value_error(operation: Callable[[], object]) -> None:
    try:
        operation()
    except ValueError:
        return
    raise AssertionError('operation must raise ValueError')


def common_config() -> Person_Ankle_Plane_Config:
    return Person_Ankle_Plane_Config(
        reference_normal_angle_max_in_radian=0.01,
        reference_residual_max_in_meter=0.01,
        local=Person_Ankle_Local_Config(
            time_radius_in_second=0.11,
            projected_radius_in_meter=0.25,
            height_radius_in_meter=0.031,
            minimum_core_neighbor_count=2,
            band_quantile_width_max_in_meter=0.012,
            band_full_span_max_in_meter=0.015,
            transition_endpoint_fraction=0.25,
            transition_height_min_in_meter=0.04,
            transition_rank_min=0.7,
            persistent_duration_min_in_second=0.19,
            persistent_occupancy_min=0.5,
            compact_radius_max_in_meter=0.15,
            path_length_min_in_meter=0.2,
            bilateral_frame_fraction_min=0.5,
            alternating_change_fraction_min=0.5,
        ),
        global_config=Person_Ankle_Global_Config(
            sigma_floor_in_meter=0.005,
            merge_z_max=3.0,
            merge_height_gap_max_in_meter=0.04,
            episode_stitch_time_max_in_second=0.11,
            episode_stitch_distance_max_in_meter=0.25,
            recurrent_episode_count_min=2,
            recurrent_observed_duration_min_in_second=0.39,
            recurrent_time_separation_min_in_second=0.25,
            recurrent_distance_min_in_meter=0.5,
            competitor_cluster_count_min=1,
            competitor_observed_duration_min_in_second=0.19,
        ),
        transition=Person_Ankle_Transition_Config(
            segment_time_gap_max_in_second=0.11,
            transition_overlap_duration_max_in_second=0.11,
            transition_time_max_in_second=0.3,
            transition_distance_max_in_meter=0.3,
            switch_episode_duration_min_in_second=0.19,
            switch_height_gap_min_in_meter=0.05,
            switch_scale_multiplier=2.0,
            transition_valid_fraction_min=0.5,
        ),
    )


def make_run(
        run_key: str,
        frames: tuple[int, ...],
        heights: tuple[float, ...],
        plane_key: str = 'plane',
        z_translation: float = 0.0,
        time_domain_key: str = 'time',
    ) -> Person_Ankle_Run:
    ankles = np.zeros((len(frames), 2, 3), dtype=np.float64)
    ankles[:, 0, 0] = -0.05
    ankles[:, 1, 0] = 0.05
    ankles[:, :, 2] = np.asarray(heights, dtype=np.float64)[:, None] + z_translation
    observation_keys = tuple(
        ('%s:%d:L' % (run_key, frame), '%s:%d:R' % (run_key, frame))
        for frame in frames
    )
    return Person_Ankle_Run(
        run_key=run_key,
        time_domain_key=time_domain_key,
        plane_key=plane_key,
        fps=10.0,
        frame_indices_original=np.asarray(frames, dtype=np.int64),
        ankle_world_in_meter=ankles,
        observation_keys=observation_keys,
    )


def make_person(
        runs: tuple[Person_Ankle_Run, ...],
        planes: tuple[Person_Ankle_Source_Plane, ...] | None = None,
    ) -> Person_Ankle_Plane_Input:
    if planes is None:
        planes = (Person_Ankle_Source_Plane(
            plane_key='plane',
            ground_param_world=np.asarray((0.0, 0.0, 1.0, 0.0), dtype=np.float64),
            body_side_sign=1,
        ),)
    return Person_Ankle_Plane_Input('person', runs, planes)


def result_for_heights(
        frames: tuple[int, ...],
        heights: tuple[float, ...],
    ) -> Person_Ankle_Plane_Result:
    return infer_person_ankle_plane_distribution(
        make_person((make_run('run', frames, heights),)), common_config())


def test_public_contract_validation_and_deep_immutability() -> None:
    for module in (package_root, estimate_subpackage):
        assert module.Person_Ankle_Plane_Config is Person_Ankle_Plane_Config
        assert module.Person_Ankle_Plane_Status is Person_Ankle_Plane_Status
        assert module.infer_person_ankle_plane_distribution is \
            infer_person_ankle_plane_distribution
    source_frames = np.asarray((0, 1), dtype=np.int64)
    source_ankles = np.zeros((2, 2, 3), dtype=np.float64)
    run = Person_Ankle_Run(
        'run', 'time', 'plane', 10.0, source_frames, source_ankles,
        (('0L', '0R'), ('1L', '1R')),
    )
    source_frames[0] = 9
    source_ankles[0, 0, 2] = 9.0
    assert tuple(run.frame_indices_original) == (0, 1)
    assert float(run.ankle_world_in_meter[0, 0, 2]) == 0.0
    assert not run.frame_indices_original.flags.writeable
    try:
        run.frame_indices_original[0] = 8
    except ValueError:
        pass
    else:
        raise AssertionError('contract arrays must be read-only')
    try:
        run.fps = 20.0  # type: ignore[reportAttributeAccessIssue]
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError('contract records must be frozen')
    expect_value_error(lambda: Person_Ankle_Run(
        'run', 'time', 'plane', 10.0,
        np.asarray((1, 1), dtype=np.int64),
        np.zeros((2, 2, 3), dtype=np.float64),
        (('a', 'b'), ('c', 'd')),
    ))
    expect_value_error(lambda: Person_Ankle_Run(
        'run', 'time', 'plane', 10.0,
        np.asarray((0,), dtype=np.int32),  # type: ignore[arg-type]
        np.zeros((1, 2, 3), dtype=np.float64),
        (('a', 'b'),),
    ))
    expect_value_error(lambda: Person_Ankle_Source_Plane(
        'plane', np.zeros(4, dtype=np.float64), 1))
    duplicate = make_run('run', (0, 1), (0.1, 0.1))
    expect_value_error(lambda: infer_person_ankle_plane_distribution(
        make_person((duplicate, duplicate)), common_config()))


def test_ckdtree_neighbors_match_brute_force_product_predicate() -> None:
    run = make_run('run', (0, 1, 2), (0.1, 0.1, 0.1))
    ankles = run.ankle_world_in_meter.copy()
    ankles[0, 0, :2] = (0.0, 0.0)
    ankles[0, 1, :2] = (0.20, 0.20)
    ankles[1, 0, :2] = (0.10, 0.0)
    ankles[1, 1, :2] = (0.25, 0.0)
    ankles[2, 0, :2] = (0.0, 0.0)
    ankles[2, 1, :2] = (0.0, 0.0)
    run = Person_Ankle_Run(
        run.run_key, run.time_domain_key, run.plane_key, run.fps,
        run.frame_indices_original, ankles, run.observation_keys,
    )
    normalized = normalize_person_ankle_input(make_person((run,)), common_config())
    indices = np.arange(6, dtype=np.int64)
    actual = exact_neighbors_for_run(indices, normalized, common_config().local)
    expected: list[tuple[int, ...]] = []
    observations = normalized.observations
    local = common_config().local
    for left in range(6):
        row = []
        for right in range(6):
            if (
                    abs(float(observations.times_in_second[left])
                        - float(observations.times_in_second[right]))
                    <= local.time_radius_in_second
                    and float(np.linalg.norm(
                        observations.projected_points_in_meter[left]
                        - observations.projected_points_in_meter[right]))
                    <= local.projected_radius_in_meter
                    and abs(float(observations.heights_in_meter[left])
                            - float(observations.heights_in_meter[right]))
                    <= local.height_radius_in_meter
                ):
                row.append(right)
        expected.append(tuple(row))
    assert actual == tuple(expected)
    assert 1 not in actual[0]


def test_ambiguous_border_is_retained_and_not_used_as_a_bridge() -> None:
    frames = tuple(range(5))
    run = make_run('run', frames, (0.1,) * 5)
    ankles = run.ankle_world_in_meter.copy()
    positions = (
        (-0.17, 0.0), (-0.16, 0.0),
        (-0.15, 0.0), (-0.09, 0.0),
        (0.09, 0.0), (0.15, 0.0),
        (0.16, 0.0), (0.17, 0.0),
        (0.0, 0.0), (0.5, 0.5),
    )
    ankles[:, :, :2] = np.asarray(positions, dtype=np.float64).reshape(5, 2, 2)
    run = Person_Ankle_Run(
        run.run_key, run.time_domain_key, run.plane_key, run.fps,
        run.frame_indices_original, ankles, run.observation_keys,
    )
    config = common_config()
    config = replace(config, local=replace(
        config.local,
        time_radius_in_second=1.0,
        projected_radius_in_meter=0.1,
        minimum_core_neighbor_count=4,
    ))
    result = infer_person_ankle_plane_distribution(make_person((run,)), config)
    reasons = dict(zip(
        result.observations.observation_keys,
        result.observations.noise_reasons,
    ))
    assert len(result.local_clusters) == 2
    assert reasons['run:4:L'] == 'ambiguous_border'
    assert reasons['run:4:R'] == 'density_noise'


def test_all_statuses_and_candidate_plane_equation() -> None:
    single = result_for_heights((0, 1, 10, 11), (0.10, 0.10, 0.10, 0.10))
    assert single.status == 'single_support_plane'
    assert single.candidate_height_in_meter == 0.10
    assert single.candidate_height_in_meter is not None
    candidate = single.candidate_height_in_meter
    reference = single.reference_ground_param_world
    new_offset = float(reference[3]) - (
        single.reference_body_side_sign
        * candidate
        * float(np.linalg.norm(reference[:3]))
    )
    assert new_offset == -0.10
    assert len(single.local_clusters) == 2
    assert len(single.hypotheses) == 1
    assert single.hypotheses[0].recurrent
    assert single.input_run_count == 1
    assert single.input_source_plane_count == 1
    assert single.input_frame_count == 4
    assert single.input_observation_count == 8
    assert single.runs[0].plane_key == 'plane'
    assert single.source_planes[0].plane_key == 'plane'

    local = result_for_heights((0, 1), (0.10, 0.10))
    assert local.status == 'local_episode_ambiguous'
    assert local.candidate_height_in_meter is None

    no_ground = result_for_heights(
        (0, 1, 2, 3, 4, 5), (0.00, 0.02, 0.04, 0.06, 0.08, 0.10))
    assert no_ground.status == 'no_ground_evidence'
    assert {cluster.height_shape for cluster in no_ground.local_clusters} == {'transition'}

    multi = result_for_heights(
        (0, 1, 10, 11, 20, 21, 30, 31),
        (0.10, 0.10, 0.40, 0.40, 0.10, 0.10, 0.40, 0.40),
    )
    assert multi.status == 'multi_layer_ambiguous'
    assert len([value for value in multi.hypotheses if value.recurrent]) == 2
    assert any(check.outcome == 'invalid' for check in multi.transition_checks)

    switching = result_for_heights(
        (0, 1, 2, 3, 4, 5, 6, 7),
        (0.10, 0.10, 0.40, 0.40, 0.10, 0.10, 0.40, 0.40),
    )
    assert switching.status == 'plane_switch'
    assert switching.candidate_height_in_meter is None
    assert all(check.outcome == 'valid' for check in switching.transition_checks)

    incompatible = infer_person_ankle_plane_distribution(
        make_person(
            (
                make_run('a', (0, 1), (0.10, 0.10), 'a'),
                make_run('b', (10, 11), (0.10, 0.10), 'b'),
            ),
            (
                Person_Ankle_Source_Plane(
                    'a', np.asarray((0.0, 0.0, 1.0, 0.0), dtype=np.float64), 1),
                Person_Ankle_Source_Plane(
                    'b', np.asarray((0.0, 0.0, 1.0, -0.2), dtype=np.float64), 1),
            ),
        ),
        common_config(),
    )
    assert incompatible.status == 'incompatible_reference_planes'
    assert incompatible.incompatible_plane_key == 'b'
    assert incompatible.local_clusters == ()
    assert set(incompatible.observations.noise_reasons) == {
        'incompatible_reference_planes'}


def test_single_recurrent_hypothesis_with_material_competitor_is_ambiguous() -> None:
    result = result_for_heights(
        (0, 1, 10, 11, 20, 21),
        (0.10, 0.10, 0.10, 0.10, 0.30, 0.30),
    )
    assert result.status == 'multi_layer_ambiguous'
    recurrent = [value for value in result.hypotheses if value.recurrent]
    competitors = [
        value for value in result.hypotheses
        if value.materially_supported_competitor
    ]
    assert len(recurrent) == 1
    assert len(competitors) == 1


def test_stitched_episode_duration_is_unique_over_run_frames() -> None:
    run = make_run('run', (0, 1), (0.10, 0.10))
    ankles = run.ankle_world_in_meter.copy()
    ankles[:, 0, 0] = -0.10
    ankles[:, 1, 0] = 0.10
    run = Person_Ankle_Run(
        run.run_key, run.time_domain_key, run.plane_key, run.fps,
        run.frame_indices_original, ankles, run.observation_keys,
    )
    config = common_config()
    config = replace(config, local=replace(
        config.local, projected_radius_in_meter=0.06))
    result = infer_person_ankle_plane_distribution(make_person((run,)), config)
    assert result.status == 'local_episode_ambiguous'
    assert len(result.local_clusters) == 2
    assert len(result.hypotheses) == 1
    assert len(result.hypotheses[0].episodes) == 1
    assert math.isclose(
        result.hypotheses[0].episodes[0].observed_duration_in_second, 0.2)
    assert math.isclose(result.hypotheses[0].observed_duration_in_second, 0.2)


def test_adjacent_run_split_stitches_one_episode_not_a_second_vote() -> None:
    result = infer_person_ankle_plane_distribution(
        make_person((
            make_run('a', (0, 1), (0.10, 0.10)),
            make_run('b', (2, 3), (0.10, 0.10)),
        )),
        common_config(),
    )
    assert result.status == 'local_episode_ambiguous'
    assert len(result.local_clusters) == 2
    assert len(result.hypotheses) == 1
    assert len(result.hypotheses[0].episodes) == 1
    whole = infer_person_ankle_plane_distribution(
        make_person((make_run('whole', (0, 1, 2, 3), (0.10,) * 4),)),
        common_config(),
    )
    split_episode = result.hypotheses[0].episodes[0]
    whole_episode = whole.hypotheses[0].episodes[0]
    assert split_episode.midpoint_time_in_second == \
        whole_episode.midpoint_time_in_second
    assert split_episode.projected_centroid_in_meter == \
        whole_episode.projected_centroid_in_meter


def test_run_fragment_stitching_uses_time_order_not_run_key_order() -> None:
    chronological = infer_person_ankle_plane_distribution(
        make_person((
            make_run('z_early', (0, 1), (0.10, 0.10)),
            make_run('a_adjacent', (2, 3), (0.10, 0.10)),
            make_run('m_far', (100, 101), (0.40, 0.40)),
        )),
        common_config(),
    )
    renamed = infer_person_ankle_plane_distribution(
        make_person((
            make_run('a_early', (0, 1), (0.10, 0.10)),
            make_run('m_adjacent', (2, 3), (0.10, 0.10)),
            make_run('z_far', (100, 101), (0.40, 0.40)),
        )),
        common_config(),
    )
    for result in (chronological, renamed):
        low = min(result.hypotheses, key=lambda value: value.center_height_in_meter)
        assert len(low.episodes) == 1
        assert not low.recurrent
        assert result.status == 'local_episode_ambiguous'


def test_independent_time_domain_origins_do_not_create_recurrence() -> None:
    def separated_feet(run: Person_Ankle_Run) -> Person_Ankle_Run:
        ankles = run.ankle_world_in_meter.copy()
        ankles[:, 0, 0] = -0.20
        ankles[:, 1, 0] = 0.20
        return Person_Ankle_Run(
            run.run_key, run.time_domain_key, run.plane_key, run.fps,
            run.frame_indices_original, ankles, run.observation_keys,
        )

    config = replace(common_config(), local=replace(
        common_config().local, projected_radius_in_meter=0.10))
    results = []
    for second_frames in ((100, 101), (10000, 10001)):
        results.append(infer_person_ankle_plane_distribution(
            make_person((
                separated_feet(make_run(
                    'first', (0, 1), (0.10, 0.10),
                    time_domain_key='clock_a')),
                separated_feet(make_run(
                    'second', second_frames, (0.10, 0.10),
                    time_domain_key='clock_b')),
            )),
            config,
        ))
    assert all(result.status == 'local_episode_ambiguous' for result in results)
    assert all(
        hypothesis.temporal_separation_in_second == 0.0
        for result in results for hypothesis in result.hypotheses
    )


def test_complete_link_does_not_chain_incompatible_height_endpoints() -> None:
    config = common_config()
    config = replace(config, global_config=replace(
        config.global_config,
        sigma_floor_in_meter=0.1,
        merge_z_max=2.0,
        merge_height_gap_max_in_meter=0.2,
    ))
    result = infer_person_ankle_plane_distribution(
        make_person((make_run(
            'run', (0, 1, 10, 11, 20, 21),
            (0.125, 0.125, 0.250, 0.250, 0.375, 0.375),
        ),)),
        config,
    )
    assert tuple(
        hypothesis.local_cluster_ids for hypothesis in result.hypotheses
    ) == ((0, 1), (2,))


def test_output_constructors_reject_mutable_or_incoherent_evidence() -> None:
    result = result_for_heights((0, 1, 10, 11), (0.10, 0.10, 0.10, 0.10))
    expect_value_error(lambda: replace(
        result,
        local_clusters=list(result.local_clusters),  # type: ignore[arg-type]
    ))
    expect_value_error(lambda: replace(
        result.local_clusters[0],
        observation_indices=list(  # type: ignore[arg-type]
            result.local_clusters[0].observation_indices),
    ))
    expect_value_error(lambda: replace(
        result, candidate_height_in_meter=0.20))
    expect_value_error(lambda: replace(
        result, status='plane_switch', candidate_height_in_meter=None))
    expect_value_error(lambda: replace(result, reference_plane_key='missing'))
    expect_value_error(lambda: replace(
        result,
        runs=(replace(result.runs[0], frame_count=3),),
    ))
    switching = result_for_heights(
        (0, 1, 2, 3, 4, 5, 6, 7),
        (0.10, 0.10, 0.40, 0.40, 0.10, 0.10, 0.40, 0.40),
    )
    expect_value_error(lambda: replace(switching, transition_checks=()))
    ambiguous = result_for_heights(
        (0, 1, 10, 11, 20, 21, 30, 31),
        (0.10, 0.10, 0.40, 0.40, 0.10, 0.10, 0.40, 0.40),
    )
    expect_value_error(lambda: replace(ambiguous, hypotheses=()))
    noise_only_observations = replace(
        result.observations,
        local_cluster_labels=np.full(
            result.input_observation_count, -1, dtype=np.int64),
        noise_reasons=('density_noise',) * result.input_observation_count,
    )
    expect_value_error(lambda: replace(
        result, observations=noise_only_observations))
    missing_cluster_hypothesis = replace(
        result.hypotheses[0], local_cluster_ids=(99,))
    expect_value_error(lambda: replace(
        result, hypotheses=(missing_cluster_hypothesis,)))
    expect_value_error(lambda: replace(switching, segments=()))


def test_missing_and_unreferenced_source_planes_fail_loudly() -> None:
    run = make_run('run', (0, 1), (0.1, 0.1), 'missing')
    expect_value_error(lambda: infer_person_ankle_plane_distribution(
        make_person((run,)), common_config()))
    valid_run = make_run('run', (0, 1), (0.1, 0.1))
    extra_plane = Person_Ankle_Source_Plane(
        'extra', np.asarray((0.0, 0.0, 1.0, 0.0), dtype=np.float64), 1)
    expect_value_error(lambda: infer_person_ankle_plane_distribution(
        make_person((valid_run,), (
            Person_Ankle_Source_Plane(
                'plane', np.asarray((0.0, 0.0, 1.0, 0.0), dtype=np.float64), 1),
            extra_plane,
        )),
        common_config(),
    ))


def test_extreme_nonzero_plane_scale_is_accepted() -> None:
    tiny_plane = Person_Ankle_Source_Plane(
        'plane', np.asarray((0.0, 0.0, 1e-200, 0.0), dtype=np.float64), 1)
    result = infer_person_ankle_plane_distribution(
        make_person(
            (make_run('run', (0, 1), (0.10, 0.10)),),
            (tiny_plane,),
        ),
        common_config(),
    )
    assert result.status == 'local_episode_ambiguous'


def result_memberships_by_key(
        result: Person_Ankle_Plane_Result,
    ) -> dict[str, int]:
    return {
        key: int(result.observations.local_cluster_labels[index])
        for index, key in enumerate(result.observations.observation_keys)
    }


def test_permutation_foot_relabel_and_rigid_translation_invariants() -> None:
    run_a = make_run('a', (0, 1), (0.10, 0.10))
    run_b = make_run('b', (10, 11), (0.10, 0.10))
    original = infer_person_ankle_plane_distribution(
        make_person((run_a, run_b)), common_config())
    permuted = infer_person_ankle_plane_distribution(
        make_person((run_b, run_a)), common_config())
    assert original.status == permuted.status == 'single_support_plane'
    assert result_memberships_by_key(original) == result_memberships_by_key(permuted)

    swapped_runs = []
    for run in (run_a, run_b):
        swapped_runs.append(Person_Ankle_Run(
            run.run_key, run.time_domain_key, run.plane_key, run.fps,
            run.frame_indices_original,
            run.ankle_world_in_meter[:, ::-1, :].copy(),
            tuple((right, left) for left, right in run.observation_keys),
        ))
    swapped = infer_person_ankle_plane_distribution(
        make_person(tuple(swapped_runs)), common_config())
    assert swapped.status == original.status
    assert swapped.candidate_height_in_meter == original.candidate_height_in_meter
    assert result_memberships_by_key(swapped) == result_memberships_by_key(original)

    translated_plane = Person_Ankle_Source_Plane(
        'plane', np.asarray((0.0, 0.0, 1.0, -2.0), dtype=np.float64), 1)
    translated_runs = tuple(
        make_run(run.run_key, tuple(int(value) for value in run.frame_indices_original),
                 (0.10, 0.10), z_translation=2.0)
        for run in (run_a, run_b)
    )
    translated = infer_person_ankle_plane_distribution(
        make_person(translated_runs, (translated_plane,)), common_config())
    assert translated.status == original.status
    assert math.isclose(
        translated.candidate_height_in_meter or math.nan,
        original.candidate_height_in_meter or math.nan,
        abs_tol=1e-12,
    )


def test_scaled_sign_flipped_equivalent_planes_and_weighting_diagnostic() -> None:
    runs = (
        make_run('a', (0, 1), (0.10, 0.10), 'a'),
        make_run('b', (10, 11, 12), (0.11, 0.11, 0.11), 'b'),
    )
    planes = (
        Person_Ankle_Source_Plane(
            'a', np.asarray((0.0, 0.0, 1.0, 0.0), dtype=np.float64), 1),
        Person_Ankle_Source_Plane(
            'b', np.asarray((0.0, 0.0, -2.0, 0.0), dtype=np.float64), -1),
    )
    result = infer_person_ankle_plane_distribution(
        make_person(runs, planes), common_config())
    assert result.status == 'single_support_plane'
    assert result.incompatible_plane_key is None
    assert not result.observations.heights_in_meter.flags.writeable
    assert math.isclose(result.hypotheses[0].center_height_in_meter, 0.105)
    assert math.isclose(
        result.hypotheses[0].sample_weighted_center_in_meter, 0.106)
    assert not math.isclose(
        result.hypotheses[0].center_height_in_meter,
        result.hypotheses[0].sample_weighted_center_in_meter,
    )


def test_bounded_two_hypothesis_overlap_bridges_are_explicit() -> None:
    frames = tuple(range(10))
    run = make_run('run', frames, (0.1,) * len(frames))
    ankles = run.ankle_world_in_meter.copy()
    ankle_heights = np.asarray((
        (0.1, 0.1), (0.1, 0.1), (0.1, 0.4),
        (0.4, 0.4), (0.4, 0.4),
        (0.1, 0.1), (0.1, 0.1), (0.1, 0.4),
        (0.4, 0.4), (0.4, 0.4),
    ), dtype=np.float64)
    ankles[:, :, 2] = ankle_heights
    run = Person_Ankle_Run(
        run.run_key, run.time_domain_key, run.plane_key, run.fps,
        run.frame_indices_original, ankles, run.observation_keys,
    )
    result = infer_person_ankle_plane_distribution(
        make_person((run,)), common_config())
    assert result.status == 'plane_switch'
    assert len(result.transition_checks) == 3
    assert result.transition_checks[0].overlap_frame_indices_original == (2,)
    assert result.transition_checks[2].overlap_frame_indices_original == (7,)
    assert all(check.outcome == 'valid' for check in result.transition_checks)


def test_excessive_two_hypothesis_overlap_is_vetoed() -> None:
    frames = tuple(range(7))
    run = make_run('a', frames, (0.1,) * len(frames))
    ankles = run.ankle_world_in_meter.copy()
    ankle_heights = np.asarray((
        (0.1, 0.1), (0.1, 0.1),
        (0.1, 0.4), (0.1, 0.4), (0.1, 0.4),
        (0.4, 0.4), (0.4, 0.4),
    ), dtype=np.float64)
    ankles[:, :, 2] = ankle_heights
    run_a = Person_Ankle_Run(
        run.run_key, 'time_a', run.plane_key, run.fps,
        run.frame_indices_original, ankles, run.observation_keys,
    )
    run_b_base = make_run('b', frames, (0.1,) * len(frames))
    ankles_b = run_b_base.ankle_world_in_meter.copy()
    ankles_b[:, :, 2] = ankle_heights
    run_b = Person_Ankle_Run(
        run_b_base.run_key, 'time_b', run_b_base.plane_key, run_b_base.fps,
        run_b_base.frame_indices_original, ankles_b, run_b_base.observation_keys,
    )
    config = common_config()
    config = replace(config, local=replace(
        config.local, bilateral_frame_fraction_min=0.3))
    result = infer_person_ankle_plane_distribution(
        make_person((run_a, run_b)), config)
    assert result.status == 'multi_layer_ambiguous'
    assert result.transition_checks
    assert all(check.outcome == 'vetoed' for check in result.transition_checks)


def test_reference_residual_threshold_is_inclusive_and_full_region_checked() -> None:
    runs = (
        make_run('a', (0, 1), (0.10, 0.10), 'a'),
        make_run('b', (10, 11), (0.10, 0.10), 'b'),
    )
    reference = Person_Ankle_Source_Plane(
        'a', np.asarray((0.0, 0.0, 1.0, 0.0), dtype=np.float64), 1)
    boundary = Person_Ankle_Source_Plane(
        'b', np.asarray((0.0, 0.0, 2.0, -0.02), dtype=np.float64), 1)
    accepted = infer_person_ankle_plane_distribution(
        make_person(runs, (reference, boundary)), common_config())
    assert accepted.status != 'incompatible_reference_planes'
    outside = Person_Ankle_Source_Plane(
        'b', np.asarray((0.0, 0.0, 2.0, -0.0202), dtype=np.float64), 1)
    rejected = infer_person_ankle_plane_distribution(
        make_person(runs, (reference, outside)), common_config())
    assert rejected.status == 'incompatible_reference_planes'


def test_config_cross_field_and_exact_count_validation() -> None:
    config = common_config()
    expect_value_error(lambda: replace(
        config,
        local=replace(
            config.local,
            band_quantile_width_max_in_meter=0.02,
            band_full_span_max_in_meter=0.01,
        ),
    ))
    expect_value_error(lambda: replace(
        config,
        transition=replace(
            config.transition,
            transition_overlap_duration_max_in_second=0.4,
            transition_time_max_in_second=0.3,
        ),
    ))


def test_moderate_dense_fixture_keeps_the_production_three_sweep_path() -> None:
    frame_count = 128
    frames = tuple(range(frame_count))
    run = make_run('dense', frames, (0.1,) * frame_count)
    config = common_config()
    config = replace(config, local=replace(
        config.local,
        time_radius_in_second=20.0,
        projected_radius_in_meter=1.0,
        minimum_core_neighbor_count=2,
    ))
    result = infer_person_ankle_plane_distribution(make_person((run,)), config)
    assert len(result.local_clusters) == 1
    assert result.local_clusters[0].sample_count == 2 * frame_count
    expect_value_error(lambda: replace(
        config,
        local=replace(config.local, minimum_core_neighbor_count=True),
    ))


def smoke_test_person_ankle_plane_distribution() -> None:
    test_public_contract_validation_and_deep_immutability()
    test_ckdtree_neighbors_match_brute_force_product_predicate()
    test_ambiguous_border_is_retained_and_not_used_as_a_bridge()
    test_all_statuses_and_candidate_plane_equation()
    test_single_recurrent_hypothesis_with_material_competitor_is_ambiguous()
    test_stitched_episode_duration_is_unique_over_run_frames()
    test_adjacent_run_split_stitches_one_episode_not_a_second_vote()
    test_run_fragment_stitching_uses_time_order_not_run_key_order()
    test_independent_time_domain_origins_do_not_create_recurrence()
    test_complete_link_does_not_chain_incompatible_height_endpoints()
    test_output_constructors_reject_mutable_or_incoherent_evidence()
    test_missing_and_unreferenced_source_planes_fail_loudly()
    test_extreme_nonzero_plane_scale_is_accepted()
    test_permutation_foot_relabel_and_rigid_translation_invariants()
    test_scaled_sign_flipped_equivalent_planes_and_weighting_diagnostic()
    test_bounded_two_hypothesis_overlap_bridges_are_explicit()
    test_excessive_two_hypothesis_overlap_is_vetoed()
    test_reference_residual_threshold_is_inclusive_and_full_region_checked()
    test_config_cross_field_and_exact_count_validation()
    test_moderate_dense_fixture_keeps_the_production_three_sweep_path()
