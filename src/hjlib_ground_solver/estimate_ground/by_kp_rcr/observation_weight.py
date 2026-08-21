'''Validation of optional RCR per-observation weights.'''

from typing import cast

import numpy as np
from numpy.typing import NDArray
import torch


def validated_numpy_observation_weights(
        value: object,
        count: int,
    ) -> NDArray[np.float64] | None:
    if value is None:
        return None
    if not isinstance(value, np.ndarray):
        raise TypeError('observation_weights must be a float64 numpy array')
    weights_dynamic = cast(NDArray[np.generic], value)
    if weights_dynamic.dtype != np.dtype(np.float64):
        raise TypeError('observation_weights must have dtype float64')
    if weights_dynamic.shape != (count,):
        raise ValueError('observation_weights must have shape (N,)')
    weights_float64 = cast(NDArray[np.float64], weights_dynamic)
    if (
            not bool(np.isfinite(weights_float64).all())
            or bool(np.any(weights_float64 <= 0.0))
        ):
        raise ValueError('observation_weights must be finite and positive')
    return weights_float64.copy()


def validated_torch_observation_weights(
        value: object,
        count: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor | None:
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        raise TypeError('observation_weights must be a torch tensor')
    if value.shape != torch.Size([count]):
        raise ValueError('observation_weights must have shape (N,)')
    if value.dtype != dtype:
        raise TypeError('observation_weights must match loss dtype')
    if value.device != device:
        raise ValueError('observation_weights must match loss device')
    if (
            not bool(torch.isfinite(value).all().item())
            or bool(torch.any(value <= 0.0).item())
        ):
        raise ValueError('observation_weights must be finite and positive')
    return value


__all__ = [
    'validated_numpy_observation_weights',
    'validated_torch_observation_weights',
]
