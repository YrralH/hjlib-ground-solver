from typing import List

import numpy as np
import cv2 as cv
import torch


def get_pixel_features_of_depth_map(depth_map: np.ndarray) -> np.ndarray:
    '''
    '''
    H, W = depth_map.shape
    assert depth_map.shape == (H, W), depth_map.shape

    filter_core_horizontal = np.array([[1, 3, 1], [0, 0, 0], [-1, -3, -1]], dtype=np.float32)
    filter_core_vertical = np.array([[1, 0, -1], [3, 0, -3], [1, 0, -1]], dtype=np.float32)

    SIZE_CORE = 11
    filter_core_horizontal = cv.resize(filter_core_horizontal, (SIZE_CORE, SIZE_CORE), interpolation=cv.INTER_LINEAR)
    filter_core_vertical = cv.resize(filter_core_vertical, (SIZE_CORE, SIZE_CORE), interpolation=cv.INTER_LINEAR)

    C = 2

    features = np.zeros((H, W, C), dtype=np.float32)

    for count_c, filter_core in enumerate([filter_core_horizontal, filter_core_vertical]):
        S_CORE = filter_core.shape[0]
        assert S_CORE % 2 == 1, S_CORE
        padding = S_CORE // 2
        assert filter_core.shape == (S_CORE, S_CORE), filter_core.shape

        filter_core_tensor = torch.from_numpy(filter_core).to(torch.float32)
        filter_core_tensor = filter_core_tensor.reshape(1, 1, S_CORE, S_CORE)
        depth_map_tensor = torch.from_numpy(depth_map).to(torch.float32)
        depth_map_tensor = depth_map_tensor.reshape(1, 1, H, W)

        features_tensor = torch.nn.functional.conv2d(depth_map_tensor, filter_core_tensor, padding=padding)
        features_tensor = features_tensor / (torch.sum(torch.abs(filter_core_tensor)) * 0.5)

        features[:, :, count_c] = features_tensor[0, 0].numpy()

    return features


def filter_vertical_and_horizontal_features(depth_map_features: np.ndarray) -> List[np.ndarray]:

    def filter_one_channel(
            depth_feature_c: np.ndarray,
            percentile: float = 30.0,
            ratio_smaller: float = 1.0,
            ratio_larger: float = 0.25
        ) -> np.ndarray:
        '''
        Filter the depth feature channel based on percentile values.
        '''
        value_percentile = np.percentile(depth_feature_c, percentile)
        print('Channel: percentile = %s' % value_percentile)

        mask_1 = (depth_feature_c > (value_percentile - abs(value_percentile * ratio_smaller)))
        mask_2 = (depth_feature_c < (value_percentile + abs(value_percentile * ratio_larger)))
        mask = mask_1 & mask_2

        return mask

    def filter_one_channel_with_value(
            depth_feature_c: np.ndarray,
            value_middle: float = 0.0,
            value_range: float = 1.0
        ) -> np.ndarray:
        '''
        Filter the depth feature channel based on a value range around a middle value.
        '''
        mask = (depth_feature_c > (value_middle - value_range)) & (depth_feature_c < (value_middle + value_range))

        return mask

    depth_map_horiz = depth_map_features[:, :, 0]
    depth_map_verti = depth_map_features[:, :, 1]

    mask_horiz = filter_one_channel(depth_map_horiz, 30, 0.35, 0.35)
    mask_verti = filter_one_channel_with_value(depth_map_verti, 0, 0.5)

    list_mask: List[np.ndarray] = []

    list_mask.append(mask_horiz)
    list_mask.append(mask_verti)

    return list_mask
