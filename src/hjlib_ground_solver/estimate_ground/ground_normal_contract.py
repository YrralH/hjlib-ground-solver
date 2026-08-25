'''Shared immutable Ground Normal value contract.'''

import math
from typing import cast

import numpy as np
from numpy.typing import NDArray


def checked_ground_normal(
        value:object,
        expected_direction:NDArray[np.float64],
    ) -> NDArray[np.float64]:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
            array.dtype,
            np.bool_,
        ) or np.issubdtype(array.dtype, np.complexfloating):
        raise ValueError('ground_normal_camera must have real numeric dtype')
    owned = np.asarray(array, dtype=np.float64)
    if owned.shape != (3,) or not np.all(np.isfinite(owned)):
        raise ValueError('ground_normal_camera must be a finite (3,) vector')
    if not math.isclose(
            float(np.linalg.norm(owned)),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
        raise ValueError('ground_normal_camera must have unit norm')
    immutable = cast(
        NDArray[np.float64],
        np.frombuffer(owned.tobytes(), dtype=np.float64).reshape((3,)),
    )
    if not np.array_equal(immutable, expected_direction):
        raise ValueError('ground_normal_camera must exactly equal winning direction')
    return immutable


__all__:list[str] = []
