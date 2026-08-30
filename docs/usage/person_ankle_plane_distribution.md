# Per-person ankle-plane distribution

> **Temporarily deprecated V1.** This API is retained only for historical
> provenance and regression. Do not extend or select it for new ankle-ground
> work. The sole active method is the temporal low-basin/getG path recorded by
> `hjlib-dataset-std` Campaign 02.

## When to use it

Use this API when the caller already owns one physical person's complete,
canonical-view left/right ankle tracks in world metres, original frame indices,
FPS, and every source old-ground plane. The function describes local height
structure and person-global recurrence; it does not read a dataset, choose a
camera, refit a normal, or force a scalar.

```python
import numpy as np

from hjlib_ground_solver import (
    Person_Ankle_Plane_Input,
    Person_Ankle_Run,
    Person_Ankle_Source_Plane,
    infer_person_ankle_plane_distribution,
)

run = Person_Ankle_Run(
    run_key='capture/run0',
    time_domain_key='capture',
    plane_key='capture/run0/plane',
    fps=30.0,
    frame_indices_original=np.arange(T, dtype=np.int64),
    ankle_world_in_meter=ankle_world.astype(np.float64),  # (T, 2, 3)
    observation_keys=tuple(
        ('frame:%d:left' % index, 'frame:%d:right' % index)
        for index in range(T)
    ),
)
person_input = Person_Ankle_Plane_Input(
    person_key='dataset/capture/person0',
    runs=(run,),
    source_planes=(Person_Ankle_Source_Plane(
        plane_key='capture/run0/plane',
        ground_param_world=np.asarray(ground, dtype=np.float64),
        body_side_sign=1,
    ),),
)

result = infer_person_ankle_plane_distribution(person_input, config)
if result.status == 'single_support_plane':
    candidate = result.candidate_height_in_meter
else:
    candidate = None
```

`config` is an explicit `Person_Ankle_Plane_Config` containing local physical
radii, band typing, global merge/recurrence, and transition thresholds. The lib
has no implicit default: experiment/config selection belongs to the caller.

## Input rules

- Each run has strictly increasing original frame indices and exactly one
  `plane_key`.
- Runs sharing a `time_domain_key` must use the same physical clock. Assembly
  fragments are ordered by numeric first/last time; `run_key` is identity only.
  Clock origins from different time domains are never compared for temporal
  recurrence.
- Observation keys are stable and unique within the person input.
- Duplicate camera views must be removed before calling; they are not independent
  evidence.
- Source planes must be geometrically compatible over the observed points after
  `body_side_sign` orientation. Incompatible inputs return
  `incompatible_reference_planes` rather than averaging labels.
- Arrays are exact `float64`/`int64`; public values copy them into read-only
  storage.

## Reading the result

The possible statuses are:

- `single_support_plane`: exactly one recurrent hypothesis without a material
  competitor; the only status that carries `candidate_height_in_meter`;
- `plane_switch`: multiple recurrent levels with coherent transitions;
- `multi_layer_ambiguous`: multiple supported levels without a coherent switch;
- `local_episode_ambiguous`: local height evidence exists but is not globally
  recurrent;
- `no_ground_evidence`: no eligible band hypothesis;
- `incompatible_reference_planes`: source old planes cannot share one reference.

All observations remain in `result.observations`, including density noise and
ambiguous borders. Local clusters keep orthogonal `height_shape`, `time_shape`,
`space_shape`, and `foot_shape`; hypotheses and transitions preserve why a
scalar was or was not returned.

## Identifiability boundary

A single static band is intentionally insufficient. With ankle-only
`(time, projected position, height, foot)` evidence, motionless standing and
motionless raised/seated ankles can be observationally identical. The API keeps
that case as `local_episode_ambiguous`; callers must not convert it to a scalar
using old-ground proximity, a pose name, or a dataset-specific fallback.

The frozen V1 design derivation and synthetic invariants now live in the
dataset-standard task record
[`hjlib-dataset-std/docs/design/tasks/person_ankle_plane_distribution/`](../../../hjlib-dataset-std/docs/design/tasks/person_ankle_plane_distribution/README.md).
This implementation remains available temporarily while the replacement task
is designed; the residence transfer does not validate V1's method.
