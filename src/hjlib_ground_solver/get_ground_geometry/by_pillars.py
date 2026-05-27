from typing import Tuple

import numpy as np


def get_ground_by_pillars_on_the_ground(
        position: np.ndarray,
        direction: np.ndarray,
        ratio_border: float = 0.5,
        ratio_height: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    given some pillars with unit length,
        in details, the
        * start point and the
        * direction of each pillar is given
    return a ground mesh with to best fit these pillar according to the hyperparameters:
        * ratio_border: the ratio of the redundant border, range from 0 to inf
        * ratio_height: the ratio of the height of the ground plane, range from 0 to 1
        and the following constraints:

    Suppose the ground parameter is (A, B, C, D) of Ax + By + Cz + D = 0
    constraint 1 determine A, B, C while constraint 2 determine D
    we call [A, B, C] the ground normal and D the ground position

    1. the ground normal is the average of the direction of the pillars
    2. the ground position, e.g. first,
        if ratio_height is 0, the ground position should make sure the lowest start point of the pillar be on the ground
        if ratio_height is 1, the ground position should make sure the highest point of the pillar be on the ground
        i.e. each the start points of the pillars project to the ground normal can determine a point with a certain "height"
            all the "height"s are assume to be ranged from a to b.
            the ratio_height is to determine the "height" of the ground plane where 0 is a and 1 is b with linear interpolation

    3. Finally the area of the ground plane is first to make sure:
            all the starting points of the pillars can be projected to the ground plane
            and the ground area is rectangular and the "smallest" to some extent, i.e. some projected points are on the edge of the ground area
        then expand the area in all 4 sides of the rectangle with the ratio_border ranged from 0 to inf
    """

    N = position.shape[0]
    assert position.shape == (N, 3), position.shape
    assert direction.shape == (N, 3), direction.shape

    direction_mean = np.mean(direction, axis=0)

    # Only keep the top 75% similar directions
    cosine_similarities = np.dot(direction, direction_mean) / (
        np.linalg.norm(direction, axis=1) * np.linalg.norm(direction_mean)
    )
    sorted_indices = np.argsort(cosine_similarities)[::-1]
    top_75_percent_index = int(len(cosine_similarities) * 0.75)
    selected_indices = sorted_indices[:top_75_percent_index]

    # ground normal
    ground_normal = np.mean(direction[selected_indices, :], axis=0)
    ground_normal /= np.linalg.norm(ground_normal)  # normalize

    # ground position
    projections = np.dot(position, ground_normal)
    min_projection = projections.min()
    max_projection = projections.max()
    ground_height = min_projection + ratio_height * (max_projection - min_projection)

    point_on_ground = ground_normal * ground_height
    # this point_on_ground and ground_normal determine the ground plane
    ground_param = np.zeros(4)  # A, B, C, D where Ax + By + Cz + D = 0
    ground_param[:3] = ground_normal
    ground_param[3] = -np.dot(ground_normal, point_on_ground)

    # project all the position to the ground plane
    positions_on_ground = np.zeros((N, 3))
    for count_point in range(N):
        position_ori = position[count_point]
        # assert np.linalg.norm(ground_normal) == 1
        # we have
        distance_to_ground = np.dot(position_ori, ground_normal) + ground_param[3]
        position_on_ground = position_ori - distance_to_ground * ground_normal
        positions_on_ground[count_point] = position_on_ground

    # compute 3 basis vectors of the ground plane
    vector_z = np.cross(np.array([1, 0, 0]), ground_normal)
    vector_y = ground_normal
    vector_x = np.cross(vector_y, vector_z)

    # compute the ground area by new coordinates
    x_coordinates = np.dot(positions_on_ground, vector_x)
    y_coordinates = np.dot(positions_on_ground, vector_y)
    z_coordinates = np.dot(positions_on_ground, vector_z)

    # all the y_coordinates should be the same
    y_coordinates_mean = np.average(y_coordinates)
    assert np.average(np.abs(y_coordinates - y_coordinates_mean)) < 1e-6, y_coordinates
    '''
    from up to look down, the coor of the ground is like
            ^ z
            |
            |
    <-------+
    x
    '''

    max_x = np.max(x_coordinates)
    min_x = np.min(x_coordinates)
    max_z = np.max(z_coordinates)
    min_z = np.min(z_coordinates)

    max_x_add_border = max_x + ratio_border * (max_x - min_x)
    min_x_add_border = min_x - ratio_border * (max_x - min_x)
    max_z_add_border = max_z + ratio_border * (max_z - min_z)
    min_z_add_border = min_z - ratio_border * (max_z - min_z)

    '''
    from up to look down, the four corners of the ground is like
            ^ z
        P1  |   P2
            |
    x       |
    <-------+------->
            |
        P3  |   P4

    '''

    P1 = vector_x * max_x_add_border + vector_y * y_coordinates_mean + vector_z * max_z_add_border
    P2 = vector_x * min_x_add_border + vector_y * y_coordinates_mean + vector_z * max_z_add_border
    P3 = vector_x * max_x_add_border + vector_y * y_coordinates_mean + vector_z * min_z_add_border
    P4 = vector_x * min_x_add_border + vector_y * y_coordinates_mean + vector_z * min_z_add_border

    verts_ground = np.array([P1, P2, P3, P4]).reshape((4, 3))
    faces_ground = np.array([[1, 3, 2], [2, 3, 4]]) - 1

    return verts_ground, faces_ground
