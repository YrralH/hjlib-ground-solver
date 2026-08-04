'''Smoke tests for chunkable explicit vertex-subset observations.'''

from typing import Callable

import torch

import hjlib_ground_solver
import hjlib_ground_solver.estimate_ground as estimate_ground
from hjlib_ground_solver.estimate_ground.by_vertex_subset_observation import (
    Vertex_Subset_Observation_Chunk,
    compute_vertex_subset_observation_chunk,
)


def expect_value_error(operation: Callable[[], object]) -> None:
    try:
        operation()
    except ValueError:
        return
    raise AssertionError('operation did not raise ValueError')


def make_vertices(requires_grad: bool = False) -> torch.Tensor:
    vertices = torch.tensor(
        (
            ((0.0, 1.0, 0.0), (0.0, -10.0, 0.0), (0.0, 3.0, 0.0), (0.0, 5.0, 0.0)),
            ((1.0, 1.0, 0.0), (0.0, -10.0, 0.0), (2.0, 3.0, 0.0), (3.0, 5.0, 0.0)),
            ((5.0, 1.0, 0.0), (0.0, -10.0, 0.0), (7.0, 3.0, 0.0), (9.0, 5.0, 0.0)),
        ),
        dtype=torch.float64,
    )
    vertices.requires_grad_(requires_grad)
    return vertices


def test_observation_values_and_chunk_carry() -> None:
    vertices = make_vertices()
    indices = torch.tensor((0, 2, 3), dtype=torch.long)
    result = compute_vertex_subset_observation_chunk(
        vertices,
        indices,
        up_axis_index=1,
        frame_rate_in_hz=10.0,
    )
    assert type(result) is Vertex_Subset_Observation_Chunk
    assert torch.equal(
        result.minimum_height,
        torch.tensor((1.0, 1.0, 1.0), dtype=torch.float64),
    )
    assert torch.equal(
        result.interval_median_speed,
        torch.tensor((20.0, 50.0), dtype=torch.float64),
    )
    assert torch.equal(result.final_vertex_positions, vertices[-1, indices])
    assert result.final_vertex_positions.untyped_storage().data_ptr() \
        != vertices.untyped_storage().data_ptr()
    assert torch.equal(result.final_vertex_indices, indices)
    assert result.final_vertex_indices.untyped_storage().data_ptr() \
        != indices.untyped_storage().data_ptr()

    previous = torch.tensor(
        ((-7.0, 1.0, 0.0), (-8.0, 3.0, 0.0), (-9.0, 5.0, 0.0)),
        dtype=torch.float64,
    )
    with_previous = compute_vertex_subset_observation_chunk(
        vertices,
        indices,
        up_axis_index=1,
        frame_rate_in_hz=10.0,
        previous_vertex_positions=previous,
        previous_vertex_indices=indices,
    )
    assert torch.equal(
        with_previous.interval_median_speed,
        torch.tensor((80.0, 20.0, 50.0), dtype=torch.float64),
    )


def test_even_median_single_frame_and_autograd() -> None:
    vertices = make_vertices(requires_grad=True)
    even = compute_vertex_subset_observation_chunk(
        vertices[:2],
        torch.tensor((0, 3), dtype=torch.long),
        up_axis_index=1,
        frame_rate_in_hz=10.0,
    )
    assert torch.equal(
        even.interval_median_speed,
        torch.tensor((20.0,), dtype=torch.float64),
    )
    single = compute_vertex_subset_observation_chunk(
        vertices[:1],
        torch.tensor((0, 2, 3), dtype=torch.long),
        up_axis_index=1,
        frame_rate_in_hz=10.0,
    )
    assert single.interval_median_speed.shape == (0,)
    loss = even.minimum_height.sum() + even.interval_median_speed.sum() \
        + even.final_vertex_positions.sum()
    loss.backward()
    assert vertices.grad is not None
    assert bool(torch.isfinite(vertices.grad).all())


def test_chunk_union_dtype_repeat_and_previous_autograd() -> None:
    vertices = make_vertices()
    vertices_before = vertices.clone()
    indices = torch.tensor((0, 2, 3), dtype=torch.long)
    full = compute_vertex_subset_observation_chunk(vertices, indices, 1, 10.0)
    first = compute_vertex_subset_observation_chunk(vertices[:2], indices, 1, 10.0)
    second = compute_vertex_subset_observation_chunk(
        vertices[2:],
        indices,
        1,
        10.0,
        first.final_vertex_positions,
        first.final_vertex_indices,
    )
    assert torch.equal(
        torch.cat((first.minimum_height, second.minimum_height)),
        full.minimum_height,
    )
    assert torch.equal(
        torch.cat((first.interval_median_speed, second.interval_median_speed)),
        full.interval_median_speed,
    )
    assert torch.equal(second.final_vertex_positions, full.final_vertex_positions)

    left_indices = torch.tensor((0, 2), dtype=torch.long)
    right_indices = torch.tensor((3,), dtype=torch.long)
    union_indices = torch.tensor((0, 2, 3), dtype=torch.long)
    left = compute_vertex_subset_observation_chunk(vertices, left_indices, 1, 10.0)
    right = compute_vertex_subset_observation_chunk(vertices, right_indices, 1, 10.0)
    union = compute_vertex_subset_observation_chunk(vertices, union_indices, 1, 10.0)
    assert torch.equal(union.minimum_height, torch.minimum(
        left.minimum_height,
        right.minimum_height,
    ))
    repeated = compute_vertex_subset_observation_chunk(vertices, indices, 1, 10.0)
    assert torch.equal(repeated.minimum_height, full.minimum_height)
    assert torch.equal(repeated.interval_median_speed, full.interval_median_speed)
    assert torch.equal(vertices, vertices_before)

    float32_result = compute_vertex_subset_observation_chunk(
        vertices.to(torch.float32),
        indices,
        1,
        10.0,
    )
    assert float32_result.minimum_height.dtype == torch.float32
    assert float32_result.interval_median_speed.dtype == torch.float32

    current = vertices[:1].clone().requires_grad_(True)
    previous = torch.zeros((3, 3), dtype=torch.float64, requires_grad=True)
    with_previous = compute_vertex_subset_observation_chunk(
        current,
        indices,
        1,
        10.0,
        previous,
        indices,
    )
    previous_loss = with_previous.interval_median_speed.sum() \
        + with_previous.final_vertex_positions.sum()
    previous_loss.backward()
    assert current.grad is not None
    assert previous.grad is not None
    assert bool(torch.isfinite(current.grad).all())
    assert bool(torch.isfinite(previous.grad).all())


def test_cuda_when_available_and_exports() -> None:
    assert hjlib_ground_solver.Vertex_Subset_Observation_Chunk \
        is Vertex_Subset_Observation_Chunk
    assert estimate_ground.Vertex_Subset_Observation_Chunk \
        is Vertex_Subset_Observation_Chunk
    assert hjlib_ground_solver.compute_vertex_subset_observation_chunk \
        is compute_vertex_subset_observation_chunk
    assert estimate_ground.compute_vertex_subset_observation_chunk \
        is compute_vertex_subset_observation_chunk
    if not torch.cuda.is_available():
        return
    vertices = make_vertices().to('cuda')
    indices = torch.tensor((0, 2, 3), dtype=torch.long, device='cuda')
    result = compute_vertex_subset_observation_chunk(vertices, indices, 1, 10.0)
    assert result.minimum_height.device.type == 'cuda'
    assert torch.equal(
        result.interval_median_speed.cpu(),
        torch.tensor((20.0, 50.0), dtype=torch.float64),
    )


def test_invalid_contracts() -> None:
    vertices = make_vertices()
    valid_indices = torch.tensor((0, 2, 3), dtype=torch.long)
    expect_value_error(lambda: compute_vertex_subset_observation_chunk(
        vertices, torch.tensor((0, 0), dtype=torch.long), 1, 10.0,
    ))
    expect_value_error(lambda: compute_vertex_subset_observation_chunk(
        vertices, torch.tensor((0, 4), dtype=torch.long), 1, 10.0,
    ))
    expect_value_error(lambda: compute_vertex_subset_observation_chunk(
        vertices, valid_indices.to(torch.int32), 1, 10.0,
    ))
    expect_value_error(lambda: compute_vertex_subset_observation_chunk(
        vertices, valid_indices, 3, 10.0,
    ))
    expect_value_error(lambda: compute_vertex_subset_observation_chunk(
        vertices, valid_indices, 1, 0.0,
    ))
    for invalid_fps in (True, float('nan'), float('inf'), -float('inf')):
        expect_value_error(lambda invalid_fps=invalid_fps: (
            compute_vertex_subset_observation_chunk(
                vertices,
                valid_indices,
                1,
                invalid_fps,  # type: ignore[arg-type]
            )
        ))
    integer_fps = compute_vertex_subset_observation_chunk(
        vertices, valid_indices, 1, 10,
    )
    assert torch.equal(
        integer_fps.interval_median_speed,
        torch.tensor((20.0, 50.0), dtype=torch.float64),
    )
    expect_value_error(lambda: compute_vertex_subset_observation_chunk(
        vertices, valid_indices, True, 10.0,
    ))
    expect_value_error(lambda: compute_vertex_subset_observation_chunk(
        vertices,
        valid_indices,
        1,
        10.0,
        torch.zeros((2, 3), dtype=torch.float64),
        valid_indices,
    ))
    expect_value_error(lambda: compute_vertex_subset_observation_chunk(
        vertices,
        valid_indices,
        1,
        10.0,
        torch.zeros((3, 3), dtype=torch.float64),
        torch.tensor((3, 2, 0), dtype=torch.long),
    ))
    expect_value_error(lambda: compute_vertex_subset_observation_chunk(
        vertices,
        valid_indices,
        1,
        10.0,
        torch.zeros((3, 3), dtype=torch.float32),
        valid_indices,
    ))
    previous_nonfinite = torch.zeros((3, 3), dtype=torch.float64)
    previous_nonfinite[0, 0] = torch.inf
    expect_value_error(lambda: compute_vertex_subset_observation_chunk(
        vertices,
        valid_indices,
        1,
        10.0,
        previous_nonfinite,
        valid_indices,
    ))
    expect_value_error(lambda: compute_vertex_subset_observation_chunk(
        vertices,
        valid_indices,
        1,
        10.0,
        torch.zeros((3, 3), dtype=torch.float64),
    ))
    nonfinite = vertices.clone()
    nonfinite[0, 0, 0] = torch.nan
    expect_value_error(lambda: compute_vertex_subset_observation_chunk(
        nonfinite, valid_indices, 1, 10.0,
    ))
    overflow = torch.zeros((2, 3, 3), dtype=torch.float32)
    overflow[1] = 3e38
    expect_value_error(lambda: compute_vertex_subset_observation_chunk(
        overflow,
        torch.tensor((0, 1, 2), dtype=torch.long),
        1,
        1.0,
    ))
    if torch.cuda.is_available():
        expect_value_error(lambda: compute_vertex_subset_observation_chunk(
            vertices.to('cuda'),
            valid_indices,
            1,
            10.0,
        ))
        expect_value_error(lambda: compute_vertex_subset_observation_chunk(
            vertices.to('cuda'),
            valid_indices.to('cuda'),
            1,
            10.0,
            torch.zeros((3, 3), dtype=torch.float64),
            valid_indices.to('cuda'),
        ))


def smoke_test_vertex_subset_observation() -> None:
    test_observation_values_and_chunk_carry()
    test_even_median_single_frame_and_autograd()
    test_chunk_union_dtype_repeat_and_previous_autograd()
    test_cuda_when_available_and_exports()
    test_invalid_contracts()
    print('test_vertex_subset_observation: PASS')


if __name__ == '__main__':
    smoke_test_vertex_subset_observation()
