'''Temporarily deprecated V1 per-person ankle-plane inference.

Retained only for historical provenance and regression. New ankle-ground work
must follow hjlib-dataset-std Campaign 02's temporal low-basin/getG path.
'''

from hjlib_ground_solver.estimate_ground.person_ankle_plane.contract import (
    Person_Ankle_Foot_Shape,
    Person_Ankle_Foot_Side,
    Person_Ankle_Global_Config,
    Person_Ankle_Height_Hypothesis,
    Person_Ankle_Height_Shape,
    Person_Ankle_Hypothesis_Segment,
    Person_Ankle_Local_Cluster,
    Person_Ankle_Local_Config,
    Person_Ankle_Noise_Reason,
    Person_Ankle_Observation_Table,
    Person_Ankle_Plane_Config,
    Person_Ankle_Plane_Input,
    Person_Ankle_Plane_Result,
    Person_Ankle_Plane_Status,
    Person_Ankle_Recurrence_Episode,
    Person_Ankle_Run,
    Person_Ankle_Run_Provenance,
    Person_Ankle_Source_Plane,
    Person_Ankle_Space_Shape,
    Person_Ankle_Time_Shape,
    Person_Ankle_Transition_Check,
    Person_Ankle_Transition_Config,
    Person_Ankle_Transition_Outcome,
)
from hjlib_ground_solver.estimate_ground.person_ankle_plane.infer import (
    infer_person_ankle_plane_distribution,
)


__all__ = [
    'Person_Ankle_Foot_Shape',
    'Person_Ankle_Foot_Side',
    'Person_Ankle_Global_Config',
    'Person_Ankle_Height_Hypothesis',
    'Person_Ankle_Height_Shape',
    'Person_Ankle_Hypothesis_Segment',
    'Person_Ankle_Local_Cluster',
    'Person_Ankle_Local_Config',
    'Person_Ankle_Noise_Reason',
    'Person_Ankle_Observation_Table',
    'Person_Ankle_Plane_Config',
    'Person_Ankle_Plane_Input',
    'Person_Ankle_Plane_Result',
    'Person_Ankle_Plane_Status',
    'Person_Ankle_Recurrence_Episode',
    'Person_Ankle_Run',
    'Person_Ankle_Run_Provenance',
    'Person_Ankle_Source_Plane',
    'Person_Ankle_Space_Shape',
    'Person_Ankle_Time_Shape',
    'Person_Ankle_Transition_Check',
    'Person_Ankle_Transition_Config',
    'Person_Ankle_Transition_Outcome',
    'infer_person_ankle_plane_distribution',
]
