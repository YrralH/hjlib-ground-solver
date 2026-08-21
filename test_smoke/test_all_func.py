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
    print('[ALL OK] hjlib-ground-solver smoke tests')


if __name__ == '__main__':
    main()
