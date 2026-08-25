'''Smokes for upright-person vertical-direction evidence fitting.'''

from unittest.mock import patch

import numpy as np
import pytest

from hjlib_camera import Camera_Intrinsics
from hjlib_camera_solver import (
    Vanishing_Direction_Source,
    Vanishing_Point_Association,
)
from hjlib_ground_solver import (
    Person_Vertical_Direction_Evidence_Result,
    fit_person_vertical_direction_evidence,
)
import hjlib_ground_solver.estimate_ground.by_person_vertical_lines as evidence_module


def unit(value:np.ndarray) -> np.ndarray:
    return value / np.linalg.norm(value)


def make_intrinsics() -> Camera_Intrinsics:
    return Camera_Intrinsics(np.array([
        [600.0, 0.0, 320.0],
        [0.0, 600.0, 240.0],
        [0.0, 0.0, 1.0],
    ]), (640, 480))


def make_observations() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    direction = unit(np.array([0.0, -0.8, 0.6]))
    vp_h = make_intrinsics().K @ direction
    vp_xy = vp_h[:2] / vp_h[2]
    bottom = np.array([
        [80.0 + 40.0 * index, 380.0 - 5.0 * index]
        for index in range(10)
    ])
    top = bottom + 0.2 * (vp_xy[None] - bottom)
    weights = np.linspace(1.0, 2.0, 10)
    return top, bottom, weights, direction


def fit_default() -> Person_Vertical_Direction_Evidence_Result:
    top, bottom, weights, _ = make_observations()
    return fit_person_vertical_direction_evidence(
        top,
        bottom,
        weights,
        make_intrinsics(),
        'people',
        'people-all-frames',
        0.24,
        2,
    )


def test_fit_builds_checked_single_cluster_evidence_once() -> None:
    top, bottom, weights, expected_direction = make_observations()
    top[-1] = bottom[-1] + np.array([90.0, -10.0])
    weights[-1] = 0.125
    original = evidence_module.get_KN_with_filter
    bottom_h = np.concatenate(
        (bottom, np.ones((bottom.shape[0], 1), dtype=np.float64)),
        axis=1,
    ).T
    top_h = np.concatenate(
        (top, np.ones((top.shape[0], 1), dtype=np.float64)),
        axis=1,
    ).T
    expected_fit = original(
        bottom_h,
        top_h,
        prop_filter=0.24,
        times_filter=2,
        flag_ret_filtered_result=True,
        observation_weights=weights,
    )
    assert isinstance(expected_fit, tuple)
    expected_vp, expected_bottom, expected_top = expected_fit
    expected_endpoints = np.stack(
        (expected_bottom[:2].T, expected_top[:2].T),
        axis=1,
    )
    with patch.object(
            evidence_module,
            'get_KN_with_filter',
            wraps=original,
        ) as mocked:
        result = fit_person_vertical_direction_evidence(
            top,
            bottom,
            weights,
            make_intrinsics(),
            'people',
            'people-all-frames',
            0.24,
            2,
        )
    assert mocked.call_count == 1
    call_args, call_kwargs = mocked.call_args
    np.testing.assert_array_equal(call_args[0], bottom_h)
    np.testing.assert_array_equal(call_args[1], top_h)
    assert call_kwargs['prop_filter'] == 0.24
    assert call_kwargs['times_filter'] == 2
    assert call_kwargs['flag_ret_filtered_result'] is True
    np.testing.assert_array_equal(call_kwargs['observation_weights'], weights)
    assert result.source.source_id == 'people'
    assert result.source.line_segments.image_record_id == 'people-all-frames'
    assert result.retained_observation_count >= 3
    assert result.retained_observation_count < top.shape[0]
    assert result.source.association.pixel_vps_h.shape == (1, 3)
    assert set(result.source.association.labels.tolist()) == {0}
    assert result.direction_result.cluster_index == 0
    assert result.direction_result.support_count == result.retained_observation_count
    expected_association = Vanishing_Point_Association(
        result.source.association.image_record_id,
        result.source.association.line_segments_sha256,
        np.zeros((expected_endpoints.shape[0],), dtype=np.int64),
        expected_vp[None],
    )
    np.testing.assert_array_equal(
        result.source.association.pixel_vps_h[0],
        expected_association.pixel_vps_h[0],
    )
    np.testing.assert_array_equal(
        result.source.line_segments.endpoints_xy,
        expected_endpoints,
    )
    assert abs(float(np.dot(
        result.direction_result.direction_camera_up,
        expected_direction,
    ))) > 1 - 1e-12

    fitted_endpoints = np.array(result.source.line_segments.endpoints_xy, copy=True)
    top[:] = 0.0
    bottom[:] = 0.0
    weights[:] = 100.0
    np.testing.assert_array_equal(
        result.source.line_segments.endpoints_xy,
        fitted_endpoints,
    )


@pytest.mark.parametrize(
    ('field', 'value', 'message'),
    (
        ('top', np.ones((2, 2)), 'equal shape'),
        ('top', np.ones((3, 3)), 'equal shape'),
        ('top', np.ones((3, 2), dtype=np.bool_), 'real non-Boolean'),
        ('top', np.ones((3, 2), dtype=np.complex128), 'real non-Boolean'),
        ('weights', np.ones((10, 1)), 'shape'),
        ('weights', np.zeros((10,)), 'strictly positive'),
        ('weights', np.full((10,), np.nan), 'finite'),
    ),
)
def test_fit_rejects_malformed_observations(
        field:str,
        value:np.ndarray,
        message:str,
    ) -> None:
    top, bottom, weights, _ = make_observations()
    values = {'top':top, 'bottom':bottom, 'weights':weights}
    values[field] = value
    with pytest.raises(ValueError, match=message):
        fit_person_vertical_direction_evidence(
            values['top'],
            values['bottom'],
            values['weights'],
            make_intrinsics(),
            'people',
            'people-all-frames',
            0.24,
            2,
        )


def test_fit_requires_at_least_three_observations() -> None:
    top, bottom, weights, _ = make_observations()
    with pytest.raises(ValueError, match='at least three'):
        fit_person_vertical_direction_evidence(
            top[:2],
            bottom[:2],
            weights[:2],
            make_intrinsics(),
            'people',
            'people-all-frames',
            0.24,
            2,
        )


@pytest.mark.parametrize(
    ('prop_filter', 'times_filter', 'message'),
    (
        (0.0, 2, 'in'),
        (1.0, 2, 'in'),
        (float('nan'), 2, 'finite'),
        (0.24, True, 'Python integer'),
        (0.24, 0, 'Python integer'),
        (0.24, 10, 'Python integer'),
    ),
)
def test_fit_rejects_invalid_filter_domain(
        prop_filter:float,
        times_filter:int,
        message:str,
    ) -> None:
    top, bottom, weights, _ = make_observations()
    with pytest.raises(ValueError, match=message):
        fit_person_vertical_direction_evidence(
            top,
            bottom,
            weights,
            make_intrinsics(),
            'people',
            'people-all-frames',
            prop_filter,
            times_filter,
        )


def test_fit_rejects_zero_length_rank_degeneracy_and_vp_at_infinity() -> None:
    top, bottom, weights, _ = make_observations()
    top[0] = bottom[0]
    with pytest.raises(ValueError, match='non-zero length'):
        fit_person_vertical_direction_evidence(
            top, bottom, weights, make_intrinsics(),
            'people', 'people-all-frames', 0.24, 2,
        )

    bottom_repeated = np.repeat(np.array([[100.0, 300.0]]), 10, axis=0)
    top_repeated = np.repeat(np.array([[100.0, 200.0]]), 10, axis=0)
    with pytest.raises(ValueError, match='rank-degenerate'):
        fit_person_vertical_direction_evidence(
            top_repeated, bottom_repeated, weights, make_intrinsics(),
            'people', 'people-all-frames', 0.24, 2,
        )

    bottom_parallel = np.array([
        [50.0 + 40.0 * index, 350.0]
        for index in range(10)
    ])
    top_parallel = bottom_parallel + np.array([0.0, -100.0])
    with pytest.raises(ValueError, match='at infinity'):
        fit_person_vertical_direction_evidence(
            top_parallel, bottom_parallel, weights, make_intrinsics(),
            'people', 'people-all-frames', 0.24, 2,
        )


def test_checked_result_rejects_forged_source_structure() -> None:
    valid = fit_default()
    association = Vanishing_Point_Association(
        valid.source.association.image_record_id,
        valid.source.association.line_segments_sha256,
        valid.source.association.labels,
        np.repeat(valid.source.association.pixel_vps_h, 2, axis=0),
    )
    forged = Vanishing_Direction_Source(
        valid.source.source_id,
        association,
        valid.source.line_segments,
    )
    with pytest.raises(ValueError, match='exactly one VP'):
        Person_Vertical_Direction_Evidence_Result(forged, make_intrinsics())


def smoke_test_person_vertical_line_evidence() -> None:
    test_fit_builds_checked_single_cluster_evidence_once()
    for field, value, message in (
            ('top', np.ones((2, 2)), 'equal shape'),
            ('weights', np.zeros((10,)), 'strictly positive'),
        ):
        test_fit_rejects_malformed_observations(field, value, message)
    test_fit_requires_at_least_three_observations()
    test_fit_rejects_invalid_filter_domain(0.24, 0, 'Python integer')
    test_fit_rejects_zero_length_rank_degeneracy_and_vp_at_infinity()
    test_checked_result_rejects_forged_source_structure()


if __name__ == '__main__':
    smoke_test_person_vertical_line_evidence()
    print('smoke_test_person_vertical_line_evidence: OK')
