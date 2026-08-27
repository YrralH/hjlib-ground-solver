# Registered Ours Ground baselines

## Public contract

The package owns two independently callable given-camera stages and one
registered composition:

- `solve_ground_normal`: a method-neutral line→VP source plus undistorted K
  produces a camera-up unit Ground Normal;
- `solve_ground_offset`: top/bottom person observations, that Ground Normal
  and K produce a camera-frame plane `(n_x, n_y, n_z, D)`.
- `solve_ground_normal_and_camera`: a line→VP source jointly produces centered
  square-pixel intrinsics and GN; it does not estimate plane offset.

Their stable IDs are `ground_normal_baseline001`,
`ground_offset_baseline001` and `ground_normal_and_camera_baseline001`. Config
constructors expose the frozen values;
unknown IDs fail and list legal values. `hjlib-experiments` directly re-exports
these owner objects and does not contain a second numeric registry.

## Baseline identities

GN composes the camera-solver exact simple vertical-VP probe with support `5`,
minimum absolute camera-y `0.8`, orthogonality tolerance `3°`, iterative native
pixel residual gate `0.25 px`, retained support `5` and at most `20` refits.
It deliberately has neither VP deduplication nor ground-direction diversity.

Offset applies strict `confidence > 4.3` and strict `ankle_ratio < 0.20`, then
uses equal observation weights, `H_prior=1.27 m`, and the existing D objective
on `[-5, 80) m` with `0.1 m` spacing. At least three positive-length retained
top/bottom segments are required.

Normal-and-camera baseline001 uses the camera-solver centered-focal vertical anchor with
the same `5 / 0.8 / 3° / 0.25 px / 5 / 20` vertical thresholds, at least two
informative focal neighbors and at most 20 focal-membership refits. Person
filters, height prior and D search remain exclusively in
`ground_offset_baseline001`, which callers invoke explicitly afterward.

`Ground_Offset_Selection` binds the immutable original observations, exact
registered config and a full-length immutable mask. This pure selection seam
is intentionally reusable by a later identity-aware experiment.

## Orientation and precision

GN and plane normal use a finite float64 camera-up unit vector. The registered
offset path asks `solve_D_search` to preserve that orientation and execute in
float64. It rejects a nonpositive full-grid winner rather than silently
searching a second positive-only grid. The low-level solver's default remains
the historical float32 plus `n_z` canonicalization for old callers.

## Extension

New frozen methods get new enum IDs and complete config records. Do not mutate
baseline001 or place detector/dataset provenance in the mathematical config.
Identity-aware offset remains a separate future method surface.

The empirical selection SSOT is the experiments Campaign 04
[hyperparameter-selection record](../../../hjlib-experiments/campaigns/04_virtualcrowd_vanishing_ground/task_ours_ground_baseline_hyperparameter_selection/README.md).
