# Ours Ground Baselines

## Status

- State: complete.
- Owner: `hjlib-ground-solver`.
- High-level consumer: `hjlib-experiments` Campaign 05.
- Empirical-selection owner: `hjlib-experiments` Campaign 04.

## Requirements

Freeze two independently callable, given-camera stages behind stable names:

1. `solve_ground_normal`: given camera intrinsics and one method-neutral
   line-to-VP source, estimate a camera-frame Ground Normal;
2. `solve_ground_offset`: given camera intrinsics, a Ground Normal and person
   top/bottom observations, estimate plane offset `D`.

The library owns the typed configs, validation, mathematical execution and
result records. It does not run ELSED or LIMAP and does not read VirtualCrowd.
Detector/LIMAP/seed provenance remains an experiment receipt because the
method-neutral input contract cannot verify those producer identities.

`solve_ground_normal_and_camera` and the identity-aware offset variant are not
part of this task.

## Public Identity And Sign Contract

The stable IDs are:

- `ground_normal_baseline001`;
- `ground_offset_baseline001`.

Both calls default to their sole registered baseline. Separate
`ground_normal_config` and `ground_offset_config` calls expose the frozen
configuration without executing a solver. Unknown names raise and list the
legal values.

A Ground Normal is a finite unit camera-up vector: `n_y <= 0`, with the
existing axial tie-break when `n_y == 0`. The offset result preserves that
exact input orientation. For the accepted camera convention, a valid solved
plane has `D > 0` in `n.x + D = 0`; zero or negative D fails. The wrapper does
not silently apply the historical `n_z` sign flip.

## Mathematical Architecture

### `ground_normal_baseline001`

The GN baseline delegates all line-to-VP selection and refit geometry to the
camera-solver simple-probe primitive. Its frozen values are:

- minimum VP support: `5` lines;
- minimum absolute camera-y: `0.8`;
- orthogonality tolerance: `3 degrees`;
- iterative native-pixel residual gate: `0.25 px`;
- minimum retained winner lines: `5`;
- maximum refit iterations: `20`.

The exact method has no VP deduplication and no ground-direction-diversity
check. It scores each eligible candidate by neighbor count, neighbor raw
support, absolute camera-y, candidate raw support and candidate index; only the
single winner label's lines are refitted. The ground layer interprets the
camera-up direction as a locally-horizontal Ground Normal without changing it.

### `ground_offset_baseline001`

`Ground_Offset_Observations` owns four aligned arrays:

- `top_xy_px`: finite `(N, 2)` shoulder-midpoint pixels;
- `bottom_xy_px`: finite `(N, 2)` ankle-midpoint pixels;
- `confidence`: finite `(N,)` values;
- `ankle_ratio`: finite non-negative `(N,)` ankle-pair-distance divided by
  bounding-box width.

The baseline retains observations with strict `confidence > 4.3` and strict
`ankle_ratio < 0.20`. Every retained observation has equal weight. At least
three observations must remain, and every retained top/bottom segment must
have strictly positive 2D Euclidean length.

Using the supplied camera-up unit GN, solve the existing distance objective
with:

- shoulder-to-ankle `H_prior = 1.27 m`;
- `D` candidates in `[-5, 80) m`;
- `0.1 m` step;
- no density/KDE weighting.

The returned float64 plane must retain the supplied float64 normal exactly and
have positive finite `D`. The opt-in low-level path executes all candidate and
final-objective arithmetic in float64; it neither flips nor round-trips the
normal through float32. The complete legacy grid still participates, so a
nonpositive winner causes wrapper failure rather than a second positive-only
argmin.

Observation selection is a named pure operation, not merely a mask emitted
after D search. `select_ground_offset_observations` returns an immutable
`Ground_Offset_Selection` binding the original immutable observations, exact
registered config and full-length retained mask. Both the baseline solver and
the later ID-aware experiment consume this same selection operation.

## Tuning Provenance

This baseline is development-set selected and is not described as GT-free.

- GN angular tolerance: `3°` and `5°` tied on `scene1` plus
  `scene1_view2`; the predeclared stricter-tolerance tie-break selected `3°`;
  `10°` was worse.
- GN residual gate: `0.25 px` minimized the equal-view GN error over
  `{none, 1.00, 0.75, 0.50, 0.25 px}` on the same two views.
- Offset search: unweighted profiles crossed strict confidence
  `2.0..5.0` at `0.1`, strict ankle ratio `{0.05, 0.10, 0.20}`, and an
  independently selected `H=1.20..1.35 m` at `0.01 m` for every profile.
- Nine profiles tied at the best two-view equal mean `0.409404 m`. The frozen
  tie-break selected the configuration retaining the most observations:
  `confidence>4.3`, `ankle_ratio<0.20`, `H_prior=1.27 m`.
- The other six released views were evaluated only after this selection and
  did not select any baseline value. Their eight-scene equal mean for the
  frozen profile was `1.899049 m` on the historical float32 path. The
  registered float64 preserved-GN replay keeps H and D winners and reports
  `1.899045 m`.

The previous `confidence>4.0, unweighted` profile was selected by an
eight-scene post-hoc comparison and is superseded as a registered baseline.

## Code Architecture

```text
src/hjlib_ground_solver/estimate_ground/ours_baseline.py
  Ground_Normal_Baseline / Ground_Normal_Config / Ground_Normal_Result
  Ground_Offset_Baseline / Ground_Offset_Config
  Ground_Offset_Observations / Ground_Offset_Selection / Ground_Offset_Result
  ground_normal_config / solve_ground_normal
  ground_offset_config / select_ground_offset_observations / solve_ground_offset

test_smoke/test_ours_ground_baselines.py
```

Exact call surface:

```python
ground_normal_config(
    baseline: Ground_Normal_Baseline | str = Ground_Normal_Baseline.GROUND_NORMAL_BASELINE001,
) -> Ground_Normal_Config

solve_ground_normal(
    source: Vanishing_Direction_Source,
    intrinsics: Camera_Intrinsics,
    baseline: Ground_Normal_Baseline | str = Ground_Normal_Baseline.GROUND_NORMAL_BASELINE001,
) -> Ground_Normal_Result

ground_offset_config(
    baseline: Ground_Offset_Baseline | str = Ground_Offset_Baseline.GROUND_OFFSET_BASELINE001,
) -> Ground_Offset_Config

select_ground_offset_observations(
    observations: Ground_Offset_Observations,
    config: Ground_Offset_Config,
) -> Ground_Offset_Selection

solve_ground_offset(
    observations: Ground_Offset_Observations,
    ground_normal_camera: NDArray[np.float64],
    intrinsics: Camera_Intrinsics,
    baseline: Ground_Offset_Baseline | str = Ground_Offset_Baseline.GROUND_OFFSET_BASELINE001,
    *,
    device: torch.device = torch.device('cpu'),
) -> Ground_Offset_Result
```

The exact records are:

```text
Ground_Normal_Config
  baseline: Ground_Normal_Baseline
  camera_solver_config: Simple_Vertical_VP_Config

Ground_Normal_Result
  config: Ground_Normal_Config
  direction_result: Simple_Vertical_VP_Result
  ground_normal_camera: read-only float64 (3,)

Ground_Offset_Config
  baseline: Ground_Offset_Baseline
  confidence_threshold_strict_gt: float
  ankle_ratio_threshold_strict_lt: float
  height_prior_m: float
  distance_min_m: float
  distance_max_m: float
  distance_step_m: float

Ground_Offset_Observations
  top_xy_px: read-only float64 (N, 2)
  bottom_xy_px: read-only float64 (N, 2)
  confidence: read-only float64 (N,)
  ankle_ratio: read-only float64 (N,)

Ground_Offset_Selection
  observations: Ground_Offset_Observations
  config: Ground_Offset_Config
  retained_mask: read-only bool (N,)

Ground_Offset_Result
  selection: Ground_Offset_Selection
  ground_normal_camera: read-only float64 (3,)
  plane_camera_abcd: read-only float64 (4,)
  objective: float
```

Config/result constructors validate exact owner types, registered-config
identity, shapes, finiteness, immutability and cross-field consistency. The GN
result must exactly equal its nested camera result. The offset result's plane
normal must exactly equal its recorded input GN.

The module composes the camera-solver simple VP primitive and
`solve_D_search`. `solve_D_search` gains the exact keyword
`preserve_ground_normal_orientation: bool = False`. Non-Boolean values fail.
The default retains historical float32 plus `n_z` canonicalization; `True`
uses float64 and preserves the supplied normal exactly. This baseline passes
`True`.

## Smoke-Test Standard

- Config calls return the exact frozen values and reject unknown IDs.
- A synthetic line/VP source proves one delegation and exact Ground Normal
  ownership.
- Synthetic person observations prove strict threshold semantics, aligned
  immutable filtering, exact-equality exclusion, zero-length rejection and
  minimum retained support. The reusable selector and solver share one
  selection result.
- An `n_z > 0` camera-up normal proves that the returned plane preserves the
  non-float32-exact supplied value rather than applying either the legacy sign
  flip or a float32 round trip.
- A synthetic nonpositive full-grid winner proves explicit failure instead of
  a positive-only second search.
- Result constructors reject forged masks, normals, planes and configs.
- Public exports and experiment high-level re-exports resolve to the same
  owner objects without a second numeric registry.
- Targeted smoke and strict pyright pass in every affected repository.

## Modification History

- 2026-08-27: Initial draft proposed a formal deduplicated/diversity-aware
  consensus and the old post-hoc `4.0 U` population.
- 2026-08-27: Mathematical/code reviews rejected both identities, required the
  exact simple probe, camera-solver ownership of line/VP refit, exact API
  signatures, retained-mask reuse and a preserved-normal sign contract.
- 2026-08-27: User selected the validation-only population-max tie-break
  `4.3 / 0.20 / 1.27`; design updated before implementation.
- 2026-08-27: Second architecture review found no remaining baseline-identity
  issue and one exact-normal blocker. Accepted findings specify the float64
  preserve path, complete record schemas, observation-bound reusable
  selection, retained zero-length rejection and nonpositive-D behavior.
- 2026-08-27: Registered `ground_normal_baseline001` and
  `ground_offset_baseline001`; the latter freezes strict `4.3 / 0.20`,
  unweighted observations and `H_prior=1.27 m`. Added a float64 preserved-GN
  low-level path, reusable immutable selection and synthetic contract smokes.
- 2026-08-28: Real-artifact replay matched all eight registered GN results and
  confirmed that the float64 offset path keeps the validation H/D selection;
  its separate registered metrics are documented by Campaign 04.
- 2026-08-28: Direct low-level compatibility, strict selection, nonpositive
  full-grid winner, package master smoke and strict pyright checks pass.
