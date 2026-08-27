# Solve Ground Normal And Camera

## Status

- State: implemented and verified.
- Owner: `hjlib-ground-solver`.
- Camera-geometry primitive owner: `hjlib-camera-solver`.
- High-level re-export: `hjlib-experiments.high_level_api.ours`.
- Empirical owner: `hjlib-experiments` Campaign 04 task
  `task_solve_ground_camera`.

## Requirements

Add one independently callable `solve_ground_normal_and_camera` baseline.
Given a method-neutral line-to-VP source, it jointly estimates:

1. centered square-pixel intrinsics with `fx = fy = focal_px`, zero skew and
   principal point `(width / 2, height / 2)`;
2. the camera-up Ground Normal from the selected vertical VP.

The camera result is camera-relative calibration, not a full world-camera
extrinsic. World yaw and translation remain unobservable from this input.
Image distortion must already have been removed.

It does not estimate ground-plane offset `D`. A caller explicitly feeds the
returned intrinsics and Ground Normal into the unchanged
`solve_ground_offset`. The identity-aware offset method is also outside this
task and remains the next separate Campaign 04 TODO.

## Mathematical Architecture

### Camera and vertical anchor

The registered identity is `ground_normal_and_camera_baseline001`. It reuses the frozen
Ground Normal thresholds: minimum cluster support `5`, minimum absolute
camera-y `0.8`, orthogonality tolerance `3 degrees`, native-pixel iterative
residual gate `0.25 px`, minimum retained vertical support `5`, and at most
`20` refit iterations.

Every VP cluster with at least five lines is independently refitted from only
its own labeled line endpoints. A candidate that does not converge under the
strict `0.25 px` gate is not allowed to anchor camera estimation. This avoids
selecting a high-consensus raw cluster which cannot support a valid vertical
VP after robust refit.

Association VPs use the existing canonical max-absolute homogeneous scale;
each refitted anchor uses the existing canonical Euclidean-unit homogeneous
scale. For refined anchor `v` and raw neighbor `h_i`, define centered
homogeneous vectors:

```text
p_v = (v_x - cx v_w, v_y - cy v_w, v_w)
p_i = (h_ix - cx h_iw, h_iy - cy h_iw, h_iw)
a_i = p_v[:2] dot p_i[:2]
b_i = p_v[2] p_i[2]
```

Only a neighbor with `abs(b_i) > 64 * float64_eps` is focal-equation eligible.
An infinite or near-infinite VP may be valid line/VP evidence elsewhere but
does not vote in this focal selector. Every positive finite
`s_i = -a_i / b_i` proposes `focal_px^2`. At each proposal, the refined anchor
and equation-eligible raw neighbors are calibrated with `focal_px = sqrt(s)`.
If the resulting centered square-pixel `K` violates the existing camera
intrinsics conditioning contract, that initial proposal is skipped while
other proposals remain eligible. If no proposal is calibratable, the selector
hard-fails with a stage-specific `ValueError`.
The anchor passes when absolute camera-y is greater than or equal to `0.8`;
neighbors whose acute orthogonality deviation is less than or equal to
`3 degrees` form its focal consensus.

Proposals rank lexicographically by:

1. orthogonal-neighbor count;
2. neighbor raw line support;
3. anchor raw line support;
4. refined-anchor absolute camera-y;
5. anchor cluster index;
6. smaller proposed focal squared as the final deterministic tie-break.

The larger cluster index wins the fifth level. There is no fallback to a
lower-ranked anchor after selection.

Anchor support precedes camera-y here, unlike the given-camera GN baseline.
Without given intrinsics, camera-y itself changes with each focal proposal;
using it before direct anchor evidence selected a low-support false vertical
in the development probe.

The winning anchor is fixed. For current consensus `I`, focal squared is
refitted by the unweighted algebraic least-squares objective
`sum_i (a_i + s b_i)^2`, whose exact solution is
`s = -sum_i(a_i b_i) / sum_i(b_i^2)`. The denominator must be finite and
strictly positive, and the resulting `s` must be positive and finite.
The final refitted `K` must also satisfy the camera intrinsics conditioning
contract; failure here is a hard failure with no anchor fallback.
Membership is then recomputed. At least two distinct equation-eligible
neighbor VP clusters must remain. Stable membership on iteration 20 succeeds;
membership that changes on iteration 20 fails. There is no anchor fallback.
No VP deduplication and no ground-direction-diversity test are added in this
baseline. The final focal equations, camera-up gate, neighbor membership and
Ground Normal all use the same refined vertical anchor.

No refit-valid anchor, no positive focal proposal, no camera-up proposal,
fewer than two informative neighbors, invalid least-squares denominator or
solution, or non-convergence is a hard `ValueError` with stage-specific text.

The Ground Normal is the calibrated refined-anchor direction after the
existing deterministic camera-up orientation rule.

### Explicit downstream offset

The camera/GN baseline ends after returning centered intrinsics and the
camera-up unit normal. Downstream ground-plane experiments call the separately
registered `ground_offset_baseline001` explicitly. Its person filtering,
height prior and D grid are not part of this baseline's config or result.

### Evaluation contract

Camera quality is reported separately as focal relative error and axial acute
GN angular error `acos(abs(n_pred dot n_gt))`. Final ground quality compares
the predicted-camera ray/plane
intersection with the GT camera-frame 3D intersection. With given GT
intrinsics, this reduces to the previous same-ray ground-intersection metric.
An evaluation ray parallel to the plane, a non-finite intersection, or a
non-positive ray scale is a scene failure rather than a clipped numeric
error. The equal-scene and pooled eight-scene aggregates are reported only
when all eight scenes succeed.

The frozen eight-scene probe obtained:

- mean focal relative error: `0.9953%`;
- mean GN angular error: `0.0950 degrees`;
- equal-scene ground error: `1.7203 m`;
- pooled ground error: `2.1692 m`.

The exact population, artifacts and per-scene receipt live in the empirical
Campaign 04 task and are not a second numeric registry in the library.

The downstream ground error may improve through compensation among focal, GN
and offset errors; it must not be interpreted as evidence that estimated
camera intrinsics are more accurate than GT intrinsics.

## Code Architecture

`hjlib-camera-solver` owns a generic centered-square-pixel camera primitive in
`simple_vertical_vp.py` because the focal/VP equations do not depend on a
ground or person model:

```text
Centered_Focal_Vertical_VP_Config
Centered_Focal_Vertical_VP_Score
Centered_Focal_Vertical_VP_Result
solve_centered_focal_and_vertical_vp_by_orthogonal_support(source, config)
```

The exact primitive signature is:

```text
solve_centered_focal_and_vertical_vp_by_orthogonal_support(
    source: Vanishing_Direction_Source,
    config: Centered_Focal_Vertical_VP_Config,
) -> Centered_Focal_Vertical_VP_Result
```

`Centered_Focal_Vertical_VP_Config` owns a
`Simple_Vertical_VP_Config vertical_config`,
`minimum_orthogonal_neighbor_count=2`, and
`maximum_focal_refit_iterations=20`. It reuses the existing source binding
validation, pixel-line equations, canonical VP fit, iterative refit and
camera-up orientation primitives. The new selector has independent ranking
and focal-consensus logic; it does not call or alter the given-K selector.
Both new scalar fields require exact built-in `int`: neighbor count must be at
least `2`, and maximum focal-refit iterations must be at least `1`.

`Centered_Focal_Vertical_VP_Score` stores the five ranking evidence fields and
`proposed_focal_squared_px2`; its ranking key adds negative proposed focal
squared only as the last tie-break.

Exact camera schemas are:

```text
Centered_Focal_Vertical_VP_Config
  vertical_config: Simple_Vertical_VP_Config
  minimum_orthogonal_neighbor_count: int
  maximum_focal_refit_iterations: int

Centered_Focal_Vertical_VP_Score
  orthogonal_neighbor_count: int
  orthogonal_neighbor_raw_support: int
  candidate_raw_support: int
  abs_camera_y: float
  candidate_index: int
  proposed_focal_squared_px2: float

Centered_Focal_Vertical_VP_Result
  source: Vanishing_Direction_Source
  config: Centered_Focal_Vertical_VP_Config
  winner_score: Centered_Focal_Vertical_VP_Score
  orthogonal_neighbor_indices: tuple[int, ...]
  initial_winner_line_count: int
  retained_winner_line_count: int
  retained_line_mask: read-only bool (N,)
  refined_pixel_vp_h: read-only float64 (3,)
  camera_intrinsics: Camera_Intrinsics
  direction_camera_up: read-only float64 (3,)
  vertical_iteration_count: int
  focal_iteration_count: int
  winner_cluster_index: derived int property
  focal_px: derived float property
```

`Centered_Focal_Vertical_VP_Result` is constructor-closed and stores exact
`source`, exact `config`, `winner_score`, final neighbor indices, full-length
read-only retained-line mask, refined read-only VP, initial/retained winner
line counts, vertical/focal iteration counts, final
`Camera_Intrinsics`, and read-only camera-up direction. `focal_px` is a
derived property reading `camera_intrinsics.K[0, 0]`; it is not duplicated
state. The initial proposal exists only as
`winner_score.proposed_focal_squared_px2`; the result has no duplicate
proposal field. Cross-field validation binds the intrinsics image size and centered
square-pixel matrix to the source and binds all winner evidence to the same
cluster. The primitive does not estimate `D`.

`hjlib-ground-solver/estimate_ground/ours_baseline.py` owns the registered
composition:

```text
Ground_Normal_And_Camera_Baseline
Ground_Normal_And_Camera_Config
Ground_Normal_And_Camera_Result
ground_normal_and_camera_config(
    baseline=ground_normal_and_camera_baseline001,
)
solve_ground_normal_and_camera(source, baseline=...)
```

Exact public signatures are:

```text
ground_normal_and_camera_config(
    baseline: Ground_Normal_And_Camera_Baseline | str =
        Ground_Normal_And_Camera_Baseline.GROUND_NORMAL_AND_CAMERA_BASELINE001,
) -> Ground_Normal_And_Camera_Config

solve_ground_normal_and_camera(
    source: Vanishing_Direction_Source,
    baseline: Ground_Normal_And_Camera_Baseline | str =
        Ground_Normal_And_Camera_Baseline.GROUND_NORMAL_AND_CAMERA_BASELINE001,
) -> Ground_Normal_And_Camera_Result
```

`Ground_Normal_And_Camera_Config` is constructor-closed and stores the baseline
enum plus the exact camera-solver config. It contains no person filter, height
prior or D-search fields. Unknown IDs raise with all legal values.

Exact ground schemas are:

```text
Ground_Normal_And_Camera_Config
  baseline: Ground_Normal_And_Camera_Baseline
  camera_solver_config: Centered_Focal_Vertical_VP_Config

Ground_Normal_And_Camera_Result
  config: Ground_Normal_And_Camera_Config
  camera_result: Centered_Focal_Vertical_VP_Result
  camera_intrinsics: derived Camera_Intrinsics property
  ground_normal_camera: derived read-only float64 (3,) property
```

`Ground_Normal_And_Camera_Result` is constructor-closed and stores the exact
registered config plus the nested `Centered_Focal_Vertical_VP_Result`. Its
`camera_intrinsics` and `ground_normal_camera` are derived properties reading
the nested owner result; arrays are never duplicated. Offset composition is
an explicit caller operation rather than hidden behavior in this call.

`hjlib-experiments.high_level_api.ours` directly re-exports the ground-solver
types and calls. It contains no second implementation and no dataset logic.
The superseded `Ground_Camera_*`, `ground_camera_config` and
`solve_ground_camera` names are absent at both public layers; no compatibility
aliases are retained because the surface was corrected before commit.

## Smoke-Test Standard

- Camera-solver synthetic VP/line evidence recovers a known centered focal
  and vertical direction, uses the refitted anchor consistently and rejects
  unsupported or non-convergent inputs.
- Deterministic scoring proves anchor support precedes camera-y and the focal
  tie-break is stable.
- Exact `3 degrees` and camera-y `0.8` boundaries are inclusive; iteration-20
  stable/changed cases, infinite/near-infinite VP exclusion and invalid focal
  least-squares cases are covered.
- Ill-conditioned initial `K` proposals are skipped, all-invalid proposals
  fail, and an ill-conditioned final least-squares `K` fails without fallback.
- Generic config construction rejects non-built-in integers and values below
  the two frozen lower bounds.
- Ground-solver wrapper proves exact config/unknown-ID behavior, nested result
  ownership, immutability and absence of offset execution.
- Public exports exist in camera-solver and ground-solver. Experiments tests
  prove direct owner-object identity and absence of a wrapper implementation.
- Ground-solver and experiments negative export smokes prove all superseded
  all-in-one camera names are absent.
- An explicit downstream `solve_ground_offset` composition smoke proves the
  returned K and GN can reproduce the recorded ground evaluation.
- Existing given-camera Ground Normal and offset baseline tests remain passing.

## Migration Plan

No legacy caller is replaced. This is an additive baseline. The one-off
Campaign 04 probe remains empirical evidence; production callers use only the
new library APIs.

## Modification History

- 2026-08-28: initial requirements, mathematical architecture, code
  architecture and smoke-test standard recorded from the eight-scene probe.
- 2026-08-28: initial math/code reviews found missing exact focal algebra,
  infinite-VP semantics, same-image binding and public schemas. Accepted all
  findings and revised the design before implementation.
- 2026-08-28: code re-review found an impossible result-level intrinsics check
  and two schema gaps. Kept the existing offset result unchanged, moved
  intrinsics equality to an exact-call invariant, removed duplicate proposal
  state, and froze new integer domains.
- 2026-08-28: final mathematical and code-architecture re-reviews accepted the
  revised exact algebra, invalid-camera semantics, evaluation receipt,
  same-image seam and public schemas with no remaining findings. Marked both
  layers implementation-ready.
- 2026-08-28: implemented the camera primitive, registered ground composition
  and experiments direct re-export. Synthetic smokes, strict pyright and the
  eight-scene production-API replay passed without numeric drift.
- 2026-08-28: user corrected the public decomposition before commit. Replaced
  the all-in-one `solve_ground_camera` contract with
  `solve_ground_normal_and_camera`, removed observations/D from its input and
  result, and kept `solve_ground_offset` as an explicit downstream stage.
- 2026-08-28: final mathematical and code re-reviews found no remaining
  findings. Full ground-solver smoke, both repository pyright checks, direct
  re-export smokes and the decomposed eight-scene production replay passed.
