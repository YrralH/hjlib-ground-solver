from typing import List

import numpy as np
from scipy.cluster.vq import kmeans, vq


def cluster_pixel_features(features: np.ndarray, position_encoding: str = 'none') -> List[np.ndarray]:
    '''
    '''
    H, W, C = features.shape
    assert features.shape == (H, W, C), features.shape

    assert position_encoding in ['y', 'x', 'xy', 'none'], position_encoding

    if position_encoding == 'none':
        position_features = None
    elif position_encoding == 'y':
        position_features = np.zeros((H, W, 1), dtype=np.float32)
        position_features[:, :, 0] = np.arange(H).reshape((H, 1)) / H
    elif position_encoding == 'x':
        position_features = np.zeros((H, W, 1), dtype=np.float32)
        position_features[:, :, 0] = np.arange(W).reshape((1, W)) / W
    elif position_encoding == 'xy':
        position_features = np.zeros((H, W, 2), dtype=np.float32)
        position_features[:, :, 0] = np.arange(H).reshape((H, 1)) / H
        position_features[:, :, 1] = np.arange(W).reshape((1, W)) / W
    else:
        raise ValueError('Unknown position encoding: %s' % position_encoding)

    if position_features is not None:
        features = np.concatenate([features, position_features], axis=-1)

    C_CONCAT = features.shape[-1]
    assert features.shape == (H, W, C_CONCAT), features.shape

    # use k-means clustering to cluster the pixel features
    features_flat = features.reshape(H * W, C_CONCAT)
    num_clusters = 5

    centroids, _ = kmeans(features_flat.astype(np.float32), num_clusters, thresh=1e-6)
    clustered_features, _ = vq(features_flat.astype(np.float32), centroids)

    K = centroids.shape[0]
    list_mask: List[np.ndarray] = []
    for count_k in range(K):
        mask = (clustered_features == count_k).reshape(H, W)
        list_mask.append(mask)

    return list_mask
