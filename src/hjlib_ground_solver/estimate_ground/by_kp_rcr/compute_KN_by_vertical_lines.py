import math
from typing import List, Tuple, Union

import numpy as np

from hjlib_ground_solver.estimate_ground.by_kp_rcr.observation_weight import (
    validated_numpy_observation_weights,
)


def assert_zeros(array: np.ndarray, thres: float = 1e-6) -> None:
    # inlined from monolith utils_np.assert_zeros (+ equals_to_zeros body)
    assert isinstance(array, np.ndarray), type(array)
    assert np.max(np.abs(array)) < thres, np.max(np.abs(array))


def get_valid_filter_mask_by_max_value(list_values: List[float], max_value: float) -> List[bool]:
    # inlined from monolith utils_py.get_valid_filter_mask_by_max_value
    list_mask_valid = [value <= max_value for value in list_values]
    assert len(list_mask_valid) == len(list_values)
    return list_mask_valid


def filter_column_vectors_by_list_valid_mask(column_vectors: np.ndarray, list_valid_mask: List[bool]) -> np.ndarray:
    # inlined from monolith utils_np.filter_column_vectors_by_list_valid_mask
    NUM_VECTORS = column_vectors.shape[1]
    NUM_DIM = column_vectors.shape[0]
    assert column_vectors.shape == (NUM_DIM, NUM_VECTORS), column_vectors.shape
    assert len(list_valid_mask) == NUM_VECTORS, (len(list_valid_mask), NUM_VECTORS)

    column_vectors_filtered_as_list: List[np.ndarray] = []
    for idx in range(NUM_VECTORS):
        if list_valid_mask[idx]:
            column_vectors_filtered_as_list.append(column_vectors[:, idx])
    column_vectors_filtered = np.stack(column_vectors_filtered_as_list, axis=1)

    assert column_vectors_filtered.shape == (NUM_DIM, sum(list_valid_mask)), column_vectors_filtered.shape
    return column_vectors_filtered


def get_KN(
    xb: np.ndarray,
    xt: np.ndarray,
    flag_ret_A: bool = False,
    observation_weights: np.ndarray | None = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    assert xb.shape == xt.shape, (xb.shape, xt.shape)
    assert xb.shape[0] == 3, xb.shape

    num = xb.shape[1]
    if num < 3:
        raise ValueError('at least three vertical-line observations are required')
    if not bool(np.isfinite(xb).all()) or not bool(np.isfinite(xt).all()):
        raise ValueError('vertical-line observations must be finite')
    A = np.cross(xb.T, xt.T)
    weights = validated_numpy_observation_weights(observation_weights, num)
    if weights is None:
        A_fit = A
    else:
        A_fit = A * np.sqrt(weights)[:, None]
    if np.linalg.matrix_rank(A_fit) < 2:
        raise ValueError('vertical-line observations are rank-degenerate')
    unused_left_vectors, singular_values, right_vectors_h = np.linalg.svd(
        A_fit,
        full_matrices=False,
    )
    del unused_left_vectors, singular_values
    kn_raw = right_vectors_h[-1]
    if abs(float(kn_raw[2])) <= 1e-12:
        raise ValueError(
            'vertical vanishing point is at infinity; horizontal-camera RCR '
            'is not supported'
        )
    KN = kn_raw / kn_raw[2]
    if not bool(np.isfinite(KN).all()):
        raise ValueError('vertical vanishing point must be finite')
    if flag_ret_A:
        return KN, A
    else:
        return KN


def get_bias_from_2D_ground_normal(xt: np.ndarray, xb: np.ndarray, KN: np.ndarray) -> List[float]:
    '''
    KN is the vanishing point
    '''
    NUM_VECTOR = xt.shape[1]
    assert xt.shape == (3, NUM_VECTOR), xt.shape
    assert xb.shape == (3, NUM_VECTOR), xb.shape
    assert KN.shape == (3,), KN.shape

    assert_zeros(xt[2, :] - 1.0)
    assert_zeros(xb[2, :] - 1.0)
    assert abs(KN[2] - 1.0) < 1e-5, KN[2]

    vecs_tb = xt[0:2, :] - xb[0:2, :]
    vecs_tV = xt[0:2, :] - KN[0:2].reshape(2, 1)
    assert vecs_tb.shape == (2, NUM_VECTOR), vecs_tb.shape
    assert vecs_tV.shape == (2, NUM_VECTOR), vecs_tV.shape

    vecs_tb = vecs_tb / np.linalg.norm(vecs_tb, ord=2, axis=0, keepdims=True)
    vecs_tV = vecs_tV / np.linalg.norm(vecs_tV, ord=2, axis=0, keepdims=True)
    assert vecs_tb.shape == (2, NUM_VECTOR), vecs_tb.shape
    assert vecs_tV.shape == (2, NUM_VECTOR), vecs_tV.shape

    bias_cos = np.zeros((NUM_VECTOR,), dtype=np.float64)
    for i in range(NUM_VECTOR):
        bias_cos[i] = np.dot(vecs_tb[:, i], vecs_tV[:, i])
    bias_cos = np.clip(bias_cos, -1.0, 1.0)
    bias_arc = np.arccos(bias_cos)

    list_bias = bias_arc.tolist()

    return list_bias


def get_KN_with_filter(
        xb: np.ndarray,
        xt: np.ndarray,
        prop_filter: float = 0.2,
        times_filter: int = 2,
        flag_ret_filtered_result: bool = False,
        observation_weights: np.ndarray | None = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    assert xb.shape == xt.shape, (xb.shape, xt.shape)
    assert xb.shape[0] == 3, xb.shape
    assert prop_filter > 0 and prop_filter < 1, prop_filter
    assert times_filter > 0 and times_filter < 10, times_filter

    num_original = xb.shape[1]
    weights = validated_numpy_observation_weights(
        observation_weights,
        num_original,
    )

    def loop_filter(
            _xb: np.ndarray,
            _xt: np.ndarray,
            _prop_keep: float = 0.2,
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert _xb.shape == _xt.shape, (_xb.shape, _xt.shape)
        assert _xb.shape[0] == 3, _xb.shape
        _num = _xb.shape[1]
        _ret = get_KN(_xb, _xt, flag_ret_A=True)
        assert isinstance(_ret, tuple)
        _KN, _A = _ret

        _list_bias_to_KN = get_bias_from_2D_ground_normal(_xt, _xb, _KN)
        _list_bias_to_KN_sorted = sorted(_list_bias_to_KN)
        if _num < 3:
            raise ValueError('RCR filtering requires at least three observations')
        _num_keep = max(3, math.ceil(_num * _prop_keep))
        _max_bias = _list_bias_to_KN_sorted[_num_keep - 1]

        # the function is the same as a per-value filter, but the implementation is more general
        _list_valid_mask = get_valid_filter_mask_by_max_value(
            _list_bias_to_KN,
            _max_bias,
        )
        _valid_mask = np.asarray(_list_valid_mask, dtype=np.bool_)
        _xb2 = filter_column_vectors_by_list_valid_mask(_xb, _list_valid_mask)
        _xt2 = filter_column_vectors_by_list_valid_mask(_xt, _list_valid_mask)
        if _xb2.shape[1] < 3:
            raise ValueError('RCR filtering retained fewer than three observations')
        return _xb2, _xt2, _KN, _valid_mask

    xb_tmp = xb.copy()
    xt_tmp = xt.copy()
    weights_tmp = None if weights is None else weights.copy()

    for ____ in range(times_filter):
        xb_tmp, xt_tmp, ____, current_mask = loop_filter(
            xb_tmp,
            xt_tmp,
            _prop_keep=1 - prop_filter,
        )
        if weights_tmp is not None:
            weights_tmp = weights_tmp[current_mask]
    KN_RET = get_KN(
        xb_tmp,
        xt_tmp,
        observation_weights=weights_tmp,
    )
    assert isinstance(KN_RET, np.ndarray)

    if flag_ret_filtered_result:
        return KN_RET, xb_tmp, xt_tmp
    else:
        return KN_RET
