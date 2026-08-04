'''Chunkable height and motion observations over an explicit vertex subset.'''

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True, slots=True)
class Vertex_Subset_Observation_Chunk:
    '''Per-frame heights, interval speeds, and an owned next-chunk carry.'''

    minimum_height: torch.Tensor
    interval_median_speed: torch.Tensor
    final_vertex_positions: torch.Tensor
    final_vertex_indices: torch.Tensor


def validate_vertex_subset_observation_input(
        vertices: object,
        vertex_indices: object,
        up_axis_index: int,
        frame_rate_in_hz: float | int,
        previous_vertex_positions: object | None,
        previous_vertex_indices: object | None,
    ) -> None:
    '''Validate shapes, numeric contracts, and device-local subset identity.'''
    if not isinstance(vertices, torch.Tensor):
        raise ValueError('vertices must be a torch.Tensor')
    if (
            vertices.ndim != 3
            or vertices.shape[0] <= 0
            or vertices.shape[1] <= 0
            or vertices.shape[2] != 3
        ):
        raise ValueError('vertices must have nonempty shape (B,V,3)')
    if not torch.is_floating_point(vertices):
        raise ValueError('vertices must have a floating dtype')
    if not bool(torch.isfinite(vertices).all().item()):
        raise ValueError('vertices must contain only finite values')
    if not isinstance(vertex_indices, torch.Tensor):
        raise ValueError('vertex_indices must be a torch.Tensor')
    if (
            vertex_indices.ndim != 1
            or vertex_indices.numel() <= 0
            or vertex_indices.dtype != torch.long
        ):
        raise ValueError('vertex_indices must be a nonempty 1D torch.long tensor')
    if vertex_indices.device != vertices.device:
        raise ValueError('vertex_indices must be on the vertices device')
    if int(torch.unique(vertex_indices).numel()) != int(vertex_indices.numel()):
        raise ValueError('vertex_indices must be unique')
    if (
            int(torch.min(vertex_indices).item()) < 0
            or int(torch.max(vertex_indices).item()) >= vertices.shape[1]
        ):
        raise ValueError('vertex_indices must be in the vertex range')
    if type(up_axis_index) is not int or up_axis_index not in (0, 1, 2):
        raise ValueError('up_axis_index must be one of the Python ints 0, 1, 2')
    if (
            type(frame_rate_in_hz) not in (float, int)
            or not math.isfinite(frame_rate_in_hz)
            or frame_rate_in_hz <= 0.0
        ):
        raise ValueError('frame_rate_in_hz must be a finite positive Python int or float')
    if previous_vertex_positions is None:
        if previous_vertex_indices is not None:
            raise ValueError('previous vertex positions and indices must be provided together')
        return
    if not isinstance(previous_vertex_positions, torch.Tensor):
        raise ValueError('previous_vertex_positions must be a torch.Tensor or None')
    expected_shape = (int(vertex_indices.numel()), 3)
    if tuple(previous_vertex_positions.shape) != expected_shape:
        raise ValueError('previous_vertex_positions must have shape (N,3)')
    if previous_vertex_positions.dtype != vertices.dtype:
        raise ValueError('previous_vertex_positions must match vertices dtype')
    if previous_vertex_positions.device != vertices.device:
        raise ValueError('previous_vertex_positions must be on the vertices device')
    if not bool(torch.isfinite(previous_vertex_positions).all().item()):
        raise ValueError('previous_vertex_positions must contain only finite values')
    if not isinstance(previous_vertex_indices, torch.Tensor):
        raise ValueError('previous_vertex_indices must accompany previous positions')
    if (
            previous_vertex_indices.ndim != 1
            or previous_vertex_indices.dtype != torch.long
            or previous_vertex_indices.device != vertices.device
        ):
        raise ValueError('previous_vertex_indices must be device-local 1D torch.long')
    if not torch.equal(previous_vertex_indices, vertex_indices):
        raise ValueError('previous vertex subset identity or order does not match')


def compute_ordered_vertex_median(values: torch.Tensor) -> torch.Tensor:
    '''Compute the exact ordered median across each interval's vertices.'''
    ordered, _ = torch.sort(values, dim=1)
    vertex_count = int(ordered.shape[1])
    upper_index = vertex_count // 2
    if vertex_count % 2 == 1:
        return ordered[:, upper_index]
    lower = ordered[:, upper_index - 1]
    upper = ordered[:, upper_index]
    return lower + (upper - lower) * 0.5


def compute_vertex_subset_observation_chunk(
        vertices: torch.Tensor,
        vertex_indices: torch.Tensor,
        up_axis_index: int,
        frame_rate_in_hz: float | int,
        previous_vertex_positions: torch.Tensor | None = None,
        previous_vertex_indices: torch.Tensor | None = None,
    ) -> Vertex_Subset_Observation_Chunk:
    '''Observe subset minimum heights and median per-vertex interval speeds.'''
    validate_vertex_subset_observation_input(
        vertices,
        vertex_indices,
        up_axis_index,
        frame_rate_in_hz,
        previous_vertex_positions,
        previous_vertex_indices,
    )
    selected = torch.index_select(vertices, 1, vertex_indices)
    minimum_height = torch.amin(selected[:, :, up_axis_index], dim=1)
    if previous_vertex_positions is None:
        displacement = selected[1:] - selected[:-1]
    else:
        positions_with_previous = torch.cat(
            (previous_vertex_positions.unsqueeze(0), selected),
            dim=0,
        )
        displacement = positions_with_previous[1:] - positions_with_previous[:-1]
    distance = torch.linalg.vector_norm(displacement, dim=2)
    interval_median_speed = (
        compute_ordered_vertex_median(distance) * float(frame_rate_in_hz)
    )
    if not bool(torch.isfinite(minimum_height).all().item()):
        raise ValueError('derived minimum heights must remain finite')
    if not bool(torch.isfinite(interval_median_speed).all().item()):
        raise ValueError('derived interval median speeds must remain finite')
    final_vertex_positions = selected[-1].clone()
    final_vertex_indices = vertex_indices.clone()
    if not bool(torch.isfinite(final_vertex_positions).all().item()):
        raise ValueError('derived final vertex positions must remain finite')
    return Vertex_Subset_Observation_Chunk(
        minimum_height=minimum_height,
        interval_median_speed=interval_median_speed,
        final_vertex_positions=final_vertex_positions,
        final_vertex_indices=final_vertex_indices,
    )


__all__ = [
    'Vertex_Subset_Observation_Chunk',
    'compute_vertex_subset_observation_chunk',
]
