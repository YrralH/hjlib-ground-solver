# Person-Height Consensus Ground Offset

## Requirements

`hjlib-ground-solver` owns the dataset-independent geometry and aggregation.
The implementation lives in
`src/hjlib_ground_solver/estimate_ground/by_person_height_consensus/` and is
an additive public API. It must not change `solve_ground_offset`,
`ground_offset_baseline001`, or the temporarily deprecated
`person_ankle_plane` V1.

Inputs are finite, explicitly named left/right shoulder, hip, knee, and ankle
2D points in the same uncropped pixel frame as a supplied nonsingular camera
matrix, person/frame identities, a
supplied camera-up unit Ground Normal, and a positive retained-person
trimmed-mean equivalent-height prior. Dataset loading, track/window selection, GT
ground, metric evaluation, and hyperparameter calibration remain outside this
library.

## Mathematical Architecture

For each already-selected pose, define shoulder and hip midpoints. Let the
torso length be the shoulder-to-hip image distance and let each leg length be
the hip-to-knee plus knee-to-ankle chain. The power-8 I-Pose Height is

```text
h = torso + ((left_leg^8 + right_leg^8) / 2)^(1/8).
```

Let `v = K n` be the homogeneous vertical vanishing point/direction and `s` the
shoulder midpoint. Define the projective vertical image direction
`d = (v_xy - s v_z) / ||v_xy - s v_z||`; reject a vanishing direction at the
shoulder. Its sign is oriented toward the observed ankle midpoint, although
that sign does not change orthogonal projection. Project both ankles onto the
line `{s + alpha d}` and average the two projections to obtain the corrected
bottom `b`. This construction uses only 2D keypoints, `K`, and the supplied
Ground Normal.

The articulated-chain `h` is a heuristic effective pixel height under a
virtual-upright approximation. It is not claimed to equal the exact projection
of a physical straight 3D segment for a bent or walking pose. Correspondingly,
the recovered quantity is an implied effective `H/D`; algebraic closure of the
virtual model does not establish anthropometric accuracy.

For homogeneous bottom ray `r = K^-1 [b_x,b_y,1]`, plane
`n dot X + D = 0`, and equivalent vertical 3D height `H`, write `q = H/D`.
The bottom at unit offset is `X_b(1) = -r/(n dot r)`. Equating the projected
vertical segment length to `h` gives

```text
q = h * (-1 / (n dot r)) / (||v_xy - b v_z|| - h v_z).
```

Use a fixed `1e-12` geometry epsilon. Require `n dot r < -epsilon`, nonzero
projective vertical length, positive ratio denominator, positive finite ratio,
and positive virtual-top camera depth. Invalid rows are removed and their input
indices are reported. Persons below the caller-supplied minimum valid-frame
count are removed and reported; require a caller-supplied minimum retained
person count. For every retained person, sort its frame ratios and remove
`floor(0.05 N)` values from each tail, then take an equal-frame mean. Apply the
same middle-90% rule to the person ratios with equal-person weight and report
the person retained mask. Input `(person_id, frame_id)` pairs must be unique;
input observations and person outputs use ascending `(person, frame)` and
ascending person order respectively. `minimum_valid_frames_per_person` and
`minimum_retained_person_count` are explicit positive configuration fields.
When `floor(0.05 N)` is zero, no row is trimmed. Given retained-person
trimmed-mean equivalent height `H_mean`, recover

```text
D = H_mean / q_scene
H_person = D * q_person.
```

The result preserves the original observation axis: invalid implied ratios are
`NaN` and an immutable valid mask distinguishes them. It reports every input
person's post-geometry support, eligibility, ratio/height (`NaN` when
ineligible), the scene aggregation mask and ratio, and `[n, D]`. `H_mean` constrains the
trim-retained persons rather than the untrimmed reported population. Absolute scale is
unidentifiable without `H_mean`; this is an explicit input rather than an
implicit constant.

## Code Architecture

- `contract.py` owns immutable ndarray contracts for explicitly named joint
  pairs, person/frame identity, solver configuration, and result records. It
  does not own or repeat a COCO joint vocabulary.
- `ipose.py` computes the power-8 I-Pose Height with a max-normalized power mean
  and the corrected bottom for a caller-selected pose population. Confidence
  selection remains with the caller.
- `solve.py` validates camera geometry, computes `H/D`, performs the two-level
  middle-90% aggregation, and applies a supplied mean height.
- Package and repository `__init__.py` files re-export the additive API.

No module imports dataset, evaluation, experiment, visualization, or tracking
owners. Functions are deterministic CPU NumPy operations.

## Smoke-Test Standard

Synthetic virtual-segment tests must close the analytic inverse against known
`H/D`, without treating that closure as real-pose accuracy. Tests verify
equal-person rather than equal-frame weighting, the exact tail-count rule, and
rejection/removal for invalid geometry and insufficient per-person support. An
articulated COCO-17 pose whose shoulder-to-ankle line differs from `K n`
verifies the power-8 construction and corrected-bottom projection. The old
registered offset baseline receives a
regression test showing that its config and result are unchanged.

## Modification History

- 2026-09-14: Initial mathematical and code architecture for implementation.
- 2026-09-14: Mathematical review found no Critical issue and three Major
  ambiguities. Defined the projective vertical line, virtual-upright meaning,
  invalid/support masks, physical ray gates, retained-person prior semantics,
  and overflow-safe power mean before implementation.
- 2026-09-14: Code-architecture review identified the same corrected-bottom
  blocker plus identity/support and skeleton-owner gaps. The vertical-line
  correction is now explicit; the public API accepts named joints rather than
  COCO slots, preserves the observation axis with a valid mask, and freezes
  identity ordering and support configuration.
