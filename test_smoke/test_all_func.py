'''Master smoke runner: imports each per-topic smoke_test_* and runs them.'''

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from test_segment_area import smoke_test_segment_area
from test_ground_geometry import smoke_test_ground_geometry
from test_ground_param_and_hvip import smoke_test_ground_param_and_hvip
from test_estimate_ground import smoke_test_estimate_ground


def main() -> None:
    smoke_test_segment_area()
    smoke_test_ground_geometry()
    smoke_test_ground_param_and_hvip()
    smoke_test_estimate_ground()
    print('[ALL OK] hjlib-ground-solver smoke tests')


if __name__ == '__main__':
    main()
