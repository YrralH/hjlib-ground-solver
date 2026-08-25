'''Smokes for discrete orthogonal-consensus Ground Normal wrappers.'''

import math
from unittest.mock import patch

import numpy as np
import pytest

from hjlib_camera import Camera_Intrinsics
from hjlib_camera_solver import (
    Image_Line_Segments,
    Orthogonal_Consensus_Config,
    Role_Aware_Vanishing_Direction_Source,
    Vanishing_Direction_Source,
    Vanishing_Point_Association,
)
from hjlib_ground_solver import (
    Orthogonal_Consensus_Ground_Normal_Result,
    solve_ground_normal_by_orthogonal_consensus,
    solve_ground_normal_by_role_aware_orthogonal_consensus,
)
import hjlib_ground_solver.estimate_ground.by_orthogonal_vanishing_direction as wrapper_module


def unit(value:np.ndarray) -> np.ndarray:
    return value / np.linalg.norm(value)


def make_intrinsics() -> Camera_Intrinsics:
    return Camera_Intrinsics(np.array([
        [600.0, 0.0, 320.0],
        [0.0, 600.0, 240.0],
        [0.0, 0.0, 1.0],
    ]), (640, 480))


def make_config() -> Orthogonal_Consensus_Config:
    return Orthogonal_Consensus_Config(5, 2, 0.8, 2.0, 3.0, 3.0)


def make_directions() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertical = unit(np.array([0.0, -0.8, 0.6]))
    horizontal_1 = np.array([1.0, 0.0, 0.0])
    tangent = np.cross(vertical, horizontal_1)
    horizontal_2 = unit(
        math.cos(math.radians(30.0)) * horizontal_1
        + math.sin(math.radians(30.0)) * tangent
    )
    return vertical, horizontal_1, horizontal_2


def make_source(
        source_id:str,
        directions:np.ndarray,
        supports:tuple[int, ...],
    ) -> Vanishing_Direction_Source:
    endpoints = np.array([
        [[float(index), 0.0], [float(index) + 0.25, 0.5]]
        for index in range(sum(supports))
    ])
    record_id = 'orthogonal-ground-normal'
    lines = Image_Line_Segments(record_id, (640, 480), endpoints)
    labels = np.repeat(np.arange(len(supports), dtype=np.int64), supports)
    association = Vanishing_Point_Association(
        record_id,
        lines.line_segments_sha256,
        labels,
        (make_intrinsics().K @ directions.T).T,
    )
    return Vanishing_Direction_Source(source_id, association, lines)


def test_full_source_wrapper_delegates_once_and_owns_winner() -> None:
    directions = np.stack(make_directions())
    source = make_source('full', directions, (10, 20, 20))
    original = wrapper_module.select_vertical_direction_by_orthogonal_consensus
    with patch.object(
            wrapper_module,
            'select_vertical_direction_by_orthogonal_consensus',
            wraps=original,
        ) as mocked:
        result = solve_ground_normal_by_orthogonal_consensus(
            (source,),
            make_intrinsics(),
            make_config(),
        )
    assert mocked.call_count == 1
    np.testing.assert_array_equal(
        result.ground_normal_camera,
        result.direction_consensus_result.winner.direction_camera_up,
    )
    assert result.ground_normal_camera is not (
        result.direction_consensus_result.winner.direction_camera_up
    )
    assert not result.ground_normal_camera.flags.writeable


def test_role_aware_wrapper_preserves_vertical_only_role() -> None:
    vertical, horizontal_1, horizontal_2 = make_directions()
    full = make_source(
        'full',
        np.stack((vertical, horizontal_1, horizontal_2)),
        (10, 20, 20),
    )
    person = make_source('person', vertical[None], (12,))
    result = solve_ground_normal_by_role_aware_orthogonal_consensus(
        (
            Role_Aware_Vanishing_Direction_Source(full, 'fixed.camera', True),
            Role_Aware_Vanishing_Direction_Source(person, 'fixed.camera', False),
        ),
        make_intrinsics(),
        make_config(),
    )
    evidence = {
        value.source_id:value
        for value in result.direction_consensus_result.winner.source_evidence
    }
    assert evidence['person'].contributes_orthogonal_consensus is False
    assert evidence['person'].orthogonal_group_cluster_indices == ()
    assert evidence['person'].orthogonal_count_fraction == 0.0


def test_vertical_only_source_cannot_complete_horizontal_consensus() -> None:
    vertical, horizontal, _ = make_directions()
    full = make_source('full', np.stack((vertical, horizontal)), (10, 20))
    person = make_source('person', vertical[None], (12,))
    original = (
        wrapper_module.select_vertical_direction_by_role_aware_orthogonal_consensus
    )
    with patch.object(
            wrapper_module,
            'select_vertical_direction_by_role_aware_orthogonal_consensus',
            wraps=original,
        ) as mocked:
        with pytest.raises(ValueError, match='no vertical hypothesis'):
            solve_ground_normal_by_role_aware_orthogonal_consensus(
                (
                    Role_Aware_Vanishing_Direction_Source(
                        full,
                        'fixed.camera',
                        True,
                    ),
                    Role_Aware_Vanishing_Direction_Source(
                        person,
                        'fixed.camera',
                        False,
                    ),
                ),
                make_intrinsics(),
                make_config(),
            )
    assert mocked.call_count == 1


def test_wrapper_propagates_failure_after_one_delegation() -> None:
    vertical, horizontal, _ = make_directions()
    source = make_source('insufficient', np.stack((vertical, horizontal)), (10, 20))
    original = wrapper_module.select_vertical_direction_by_orthogonal_consensus
    with patch.object(
            wrapper_module,
            'select_vertical_direction_by_orthogonal_consensus',
            wraps=original,
        ) as mocked:
        with pytest.raises(ValueError, match='no vertical hypothesis'):
            solve_ground_normal_by_orthogonal_consensus(
                (source,),
                make_intrinsics(),
                make_config(),
            )
    assert mocked.call_count == 1


def test_result_rejects_forged_ground_normal() -> None:
    directions = np.stack(make_directions())
    source = make_source('full', directions, (10, 20, 20))
    valid = solve_ground_normal_by_orthogonal_consensus(
        (source,),
        make_intrinsics(),
        make_config(),
    )
    with pytest.raises(ValueError, match='unit norm'):
        Orthogonal_Consensus_Ground_Normal_Result(
            valid.ground_normal_camera * 2.0,
            valid.direction_consensus_result,
        )
    with pytest.raises(ValueError, match='exactly equal'):
        Orthogonal_Consensus_Ground_Normal_Result(
            -valid.ground_normal_camera,
            valid.direction_consensus_result,
        )


def smoke_test_orthogonal_vanishing_direction_ground_normal() -> None:
    test_full_source_wrapper_delegates_once_and_owns_winner()
    test_role_aware_wrapper_preserves_vertical_only_role()
    test_vertical_only_source_cannot_complete_horizontal_consensus()
    test_wrapper_propagates_failure_after_one_delegation()
    test_result_rejects_forged_ground_normal()


if __name__ == '__main__':
    smoke_test_orthogonal_vanishing_direction_ground_normal()
    print('smoke_test_orthogonal_vanishing_direction_ground_normal: OK')
