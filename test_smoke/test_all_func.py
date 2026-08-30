'''Master smoke runner: imports each per-topic smoke_test_* and runs them.'''

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from test_segment_area import smoke_test_segment_area
from test_ground_geometry import smoke_test_ground_geometry
from test_ground_param_and_hvip import smoke_test_ground_param_and_hvip
from test_estimate_ground import smoke_test_estimate_ground
from test_mesh_lower_envelope import smoke_test_mesh_lower_envelope
from test_mesh_lower_envelope_peeling import smoke_test_mesh_lower_envelope_peeling
from test_static_foot_humor import smoke_test_static_foot_humor
from test_vertex_subset_observation import smoke_test_vertex_subset_observation
from test_static_foot_plantar_humor import smoke_test_static_foot_plantar_humor
from test_hj_derived_plantar_zmin import smoke_test_hj_derived_plantar_zmin
from test_observation_density import smoke_test_observation_density
from test_vanishing_direction_ground_normal import (
    smoke_test_vanishing_direction_ground_normal,
)
from test_orthogonal_vanishing_direction_ground_normal import (
    smoke_test_orthogonal_vanishing_direction_ground_normal,
)
from test_person_vertical_line_evidence import (
    smoke_test_person_vertical_line_evidence,
)
from test_equal_vertical_line_ground_normal import (
    smoke_test_equal_vertical_line_ground_normal,
)
from test_ours_ground_baselines import smoke_test_ours_ground_baselines
from test_person_ankle_plane_distribution import (
    smoke_test_person_ankle_plane_distribution,
)


def main() -> None:
    smoke_test_segment_area()
    smoke_test_ground_geometry()
    smoke_test_ground_param_and_hvip()
    smoke_test_estimate_ground()
    smoke_test_mesh_lower_envelope()
    smoke_test_mesh_lower_envelope_peeling()
    smoke_test_static_foot_humor()
    smoke_test_vertex_subset_observation()
    smoke_test_static_foot_plantar_humor()
    print('[OK] static_foot_plantar_humor')
    smoke_test_hj_derived_plantar_zmin()
    print('[OK] hj_derived_plantar_zmin')
    smoke_test_observation_density()
    print('[OK] observation_density')
    smoke_test_vanishing_direction_ground_normal()
    print('[OK] vanishing_direction_ground_normal')
    smoke_test_orthogonal_vanishing_direction_ground_normal()
    print('[OK] orthogonal_vanishing_direction_ground_normal')
    smoke_test_person_vertical_line_evidence()
    print('[OK] person_vertical_line_evidence')
    smoke_test_equal_vertical_line_ground_normal()
    print('[OK] equal_vertical_line_ground_normal')
    smoke_test_ours_ground_baselines()
    print('[OK] ours_ground_baselines')
    smoke_test_person_ankle_plane_distribution()
    print('[OK] person_ankle_plane_distribution')
    print('[ALL OK] hjlib-ground-solver smoke tests')


if __name__ == '__main__':
    main()
