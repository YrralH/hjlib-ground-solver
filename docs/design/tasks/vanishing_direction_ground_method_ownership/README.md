# Vanishing-Direction Ground-Method Ownership

## Status

- State: implementation verified; clean commit/run pending
- Owner: `hjlib-ground-solver`
- Consumer: Campaign 04 in `hjlib-experiments`
- Camera primitive dependency: `hjlib-camera-solver`

## Requirements

`hjlib-ground-solver` owns reusable methods whose declared output is a
camera-space Ground Normal. `hjlib-camera-solver` continues to own calibrated
line/VP direction primitives and generic scene-vertical selection. The
experiment layer owns dataset identity, configured profile matrices, GT
evaluation and publication only.

This task adds three missing ground-method boundaries:

1. interpret the existing single-source VP baseline winner as a locally-horizontal
   Ground Normal;
2. interpret the discrete orthogonal-consensus winner as a locally-horizontal
   Ground Normal, including the role-aware full/vertical-only source mode;
3. turn person top/bottom image observations into one directly supported
   vertical-direction evidence source using the existing weighted RCR fit.

The task also removes Campaign 04-local solve/fitting helpers in favor of these
public ground APIs. It does not migrate the Stage 1 or Stage 2 selector kernels
out of `hjlib-camera-solver`.

The following remain outside the library:

- VirtualCrowd and `baseline001` paths or NPZ schema;
- O/E/P profile names and the seven-profile matrix;
- frozen-result parity checks and artifact SHA binding;
- GT Ground Normal access, metrics, summaries and result publication.

## Mathematical Architecture

### Locally-horizontal interpretation

Both discrete orthogonal consensus and formal robust fusion estimate one
camera-space axial scene-vertical direction from calibrated vanishing
directions. Under the explicit locally-horizontal assumption, this direction
is the Ground Normal up to sign. The ground-facing result uses the existing
camera-up orientation and must exactly equal the selector winner by value.

No plane offset `D`, camera height, slope or world extrinsic is inferred. The
ground wrapper does not rescore, refine or otherwise change the camera-solver
winner. GT is unavailable at this layer.

The existing single-source VP baseline wrapper accepts one
`Vanishing_Direction_Source`, delegates once to the public
`select_vertical_vanishing_direction`, and returns an immutable Ground Normal
plus the complete `Vertical_Vanishing_Direction_Result`. Its selection failures
also propagate unchanged without a partial result. The source may contain
multiple VP clusters: the camera primitive retains its exact support
eligibility and `max |camera y|` winner semantics. `source`, `intrinsics`, and
`min_support_count` are forwarded unchanged; the ground wrapper adds no
single-cluster restriction or threshold reinterpretation.

For discrete consensus, the public ground result contains:

- the immutable unit `ground_normal_camera`;
- the complete nested `Orthogonal_Consensus_Result` ledger.

The legacy full-source solve and the role-aware solve have separate entry
points but share this result contract. Role-aware semantics remain those of the
camera primitive: full sources may supply vertical and ground-parallel
directions, while a vertical-only source may propose or confirm the vertical
candidate but may not contribute orthogonal consensus.

The discrete selector has no all-rejected result state. If no hypothesis meets
its contract, or if any input is invalid, its `ValueError` propagates unchanged
through the ground entry point and no partial ground result is constructed. The
wrapper delegates exactly once on both success and failure paths.

### Person vertical-line evidence

Given `N` person observations in one fixed calibrated camera:

- `top_xy_px`: shoulder-midpoint pixels, shape `(N, 2)`;
- `bottom_xy_px`: ankle-midpoint pixels, shape `(N, 2)`;
- positive finite observation weights, shape `(N,)`;
- camera intrinsics and explicit fit thresholds;

the method calls the existing local weighted RCR primitive
`get_KN_with_filter` exactly once with retained-line output enabled. The
retained bottom-to-top segments form one VP cluster whose homogeneous pixel VP
is the fitted `KN`. The existing public camera primitive
`select_vertical_vanishing_direction(..., min_support_count=1)` calibrates and
camera-up orients that unique VP; ground-solver does not import an internal
camera symbol or reproduce the orientation convention.

This inference assumes every shoulder-midpoint to ankle-midpoint segment is a
standing or near-upright body-axis observation of gravity/scene vertical.
Leaning people, sitting/running/action poses, systematic pose-estimation bias,
or a population not aligned with gravity violate the assumption. Every
observation must already be expressed in the same uncropped pixel frame with
one fixed `K`; mixed crops, resizes, changing intrinsics or camera motion are
out of contract even if array shapes match.

The numeric boundary is exact:

- `N >= 3`; top and bottom are NumPy arrays with real, non-Boolean, finite
  numeric values and shape `(N, 2)`; accepted arrays are defensively converted
  to owned float64;
- every paired top/bottom image segment has non-zero Euclidean length;
- weights are a NumPy array with real, non-Boolean, finite, strictly positive
  values and shape `(N,)`, defensively converted to owned float64;
- `prop_filter` is a finite Python `float` in `(0, 1)` and is the fraction
  discarded per pass;
- `times_filter` is a Python `int` in `[1, 9]`; Boolean values are rejected;
- unweighted angular-residual ranking determines trimming membership,
  threshold ties are retained, and weights enter only the final retained SVD
  fit, matching the frozen RCR implementation;
- every pass and the final fit retain at least three observations with
  sufficient line-system rank;
- the fitted VP must be finite with non-zero third homogeneous coordinate;
  a VP at infinity is an explicit unsupported-case failure.

The returned evidence is a normal camera-solver `Vanishing_Direction_Source`:
all retained labels are zero, its association is hash-bound to its retained
line payload, and its record/source IDs are supplied by the caller. The result
also carries the checked camera direction result plus retained observation
count. It does not load a `baseline001` file or compare against a frozen
prediction; those are experiment provenance duties.

The result is a checked receipt rather than a loose dataclass. Its custom
constructor receives only the built source and fit-time intrinsics; it derives
the retained count and nested direction result itself. The following therefore
hold by construction:

- `retained_observation_count >= 3`;
- retained count equals the source endpoint and label population;
- the association has exactly one VP, all labels are zero, record IDs and line
  hash agree through the camera contracts;
- the nested direction result selects cluster zero with support equal to the
  retained count and its immutable direction is the public camera primitive's
  calibrated, camera-up-canonicalized value.

There is no constructor argument for a direction, count or nested direction
receipt, so a caller cannot forge those redundant fields. The original input
count `N` is intentionally absent because a filtered source cannot prove it;
the fit caller records `top.shape[0]` in its input provenance. Mutable input
arrays and retained endpoints are defensively owned.

This function is an evidence builder, not a second fusion algorithm. Its
output becomes vertical-only only when the caller wraps it in
`Role_Aware_Vanishing_Direction_Source`.

## Code Architecture

New residence:

```text
src/hjlib_ground_solver/estimate_ground/
  by_vanishing_direction.py
  by_orthogonal_vanishing_direction.py
  by_person_vertical_lines.py
  ground_normal_contract.py
```

The single-source VP wrapper joins the existing formal robust interpretation in
`by_vanishing_direction.py`; both are thin interpretations of camera-owned
direction results. The other two method modules are new. Every result
constructor reuses `checked_ground_normal` from `ground_normal_contract.py`;
that module alone owns real/finite/unit validation, immutable copying and
exact-winner equality.

Public contracts:

```python
@dataclass(frozen=True, slots=True)
class Vertical_VP_Selection_Ground_Normal_Result:
    ground_normal_camera: NDArray[np.float64]
    direction_result: Vertical_Vanishing_Direction_Result

solve_ground_normal_by_vertical_vp_selection(
    source: Vanishing_Direction_Source,
    intrinsics: Camera_Intrinsics,
    min_support_count: int = 5,
) -> Vertical_VP_Selection_Ground_Normal_Result

@dataclass(frozen=True, slots=True)
class Orthogonal_Consensus_Ground_Normal_Result:
    ground_normal_camera: NDArray[np.float64]
    direction_consensus_result: Orthogonal_Consensus_Result

solve_ground_normal_by_orthogonal_consensus(
    sources: Sequence[Vanishing_Direction_Source],
    intrinsics: Camera_Intrinsics,
    config: Orthogonal_Consensus_Config,
) -> Orthogonal_Consensus_Ground_Normal_Result

solve_ground_normal_by_role_aware_orthogonal_consensus(
    sources: Sequence[Role_Aware_Vanishing_Direction_Source],
    intrinsics: Camera_Intrinsics,
    config: Orthogonal_Consensus_Config,
) -> Orthogonal_Consensus_Ground_Normal_Result

@dataclass(frozen=True, slots=True, init=False)
class Person_Vertical_Direction_Evidence_Result:
    source: Vanishing_Direction_Source
    direction_result: Vertical_Vanishing_Direction_Result
    retained_observation_count: int

    def __init__(
        self,
        source: Vanishing_Direction_Source,
        intrinsics: Camera_Intrinsics,
    ) -> None: ...

fit_person_vertical_direction_evidence(
    top_xy_px: NDArray[np.generic],
    bottom_xy_px: NDArray[np.generic],
    observation_weights: NDArray[np.generic],
    intrinsics: Camera_Intrinsics,
    source_id: str,
    image_record_id: str,
    prop_filter: float,
    times_filter: int,
) -> Person_Vertical_Direction_Evidence_Result
```

Both modules are re-exported through `estimate_ground/__init__.py` and the
package root. The existing formal robust wrapper remains in
`by_vanishing_direction.py`; Campaign 04 calls it directly instead of keeping
a local `solve_formal_sources` alias.

`hjlib-experiments/run.py` retains only a baseline artifact loader that:

1. validates and loads the frozen NPZ schema;
2. calls `fit_person_vertical_direction_evidence`;
3. checks sign-invariant parity with the frozen baseline normal;
4. binds input paths, hashes and experiment receipts.

Across its single-source, discrete and role-aware profiles, the Campaign calls
the three ground-normal solve APIs directly. It does not import camera-solver
selector functions.

Dependency direction remains acyclic:

```text
hjlib-camera -> hjlib-camera-solver -> hjlib-ground-solver -> experiments
```

`hjlib-ground-solver` already directly depends on both camera packages; its
camera-solver pin must advance to the committed role-aware API revision before
the new ground API is committed.

## Smoke-Test Standard

Ground-solver smokes must establish:

1. full-source and role-aware wrappers delegate exactly once to the matching
   camera selector and preserve the complete nested ledger;
2. the single-source VP baseline wrapper delegates exactly once, retains its
   support/margin receipt, and preserves multi-cluster winner selection plus
   support-threshold boundaries;
3. every Ground Normal is immutable, unit length and exactly equal to the winner;
4. vertical-only evidence cannot satisfy horizontal consensus through the
   ground entry point;
5. person evidence reconstructs a hand-generated VP, retains the exact filtered
   segments, labels one cluster and binds association hash to line payload;
6. malformed shapes, non-finite/non-positive weights, invalid filter values and
   record/source IDs fail before a valid result exists;
7. zero-length segments, rank degeneracy and a VP at infinity are explicit
   failures;
8. exact count/direction/source invariants cannot be forged through redundant
   constructor fields, and mutable inputs cannot alter returned evidence/results;
9. selector failure propagates unchanged after exactly one delegation and
   produces no partial Ground Normal result.

Campaign smokes must establish:

1. no local selector or RCR fit helper remains;
2. the seven-profile call matrix reaches ground-solver public entry points with
   the same O/E/P objects and roles;
3. existing O, E and O+E numerical outputs remain equal to the reviewed Stage 1
   result for identical artifacts;
4. frozen baseline parity, GT ordering, provenance and atomic publication remain
   unchanged.
5. the legacy OpenCV/DeepLSD/ELSED single-source evaluation routes all three
   methods through the new ground wrapper while preserving `results.json`
   success/failure, direction, support, winning camera-y score and margin; a
   valid winner without a runner-up serializes its undefined margin as JSON
   `null` rather than non-standard `NaN`.

## Migration Plan

1. Land and review this residence.
2. Implement and verify the three ground-method surfaces, shared result
   contract and public exports.
3. Advance the ground-solver camera-solver dependency pin leaf-first.
4. Update the adjacent camera role-aware residence, ground README/design/usage/
   test indexes and public exports so owner statements have one meaning.
5. Replace Campaign 04 local fitting/solve helpers with ground API calls.
6. Run ground-solver full smoke/pyright, Campaign 04 targeted smoke/pyright and
   cross-repository architecture review.
7. Commit and push ground-solver, then advance its exact pin through direct
   consumers `hjlib-evaluation` and `hjlib-dataset-std`; smoke/commit each and
   continue through any further consumers reported by `hjlibm`.
8. Synchronize the camera/ground owner summaries in `hjlibm` and its generated
   family index.
9. Commit the experiment wiring only after the leaf-first cascade is clean,
   then execute the commit-bound eight-scene evaluation.

## Modification History

- 2026-08-25: Initial design after user clarified that reusable ground
  estimation methods and their method-level callers belong in
  `hjlib-ground-solver`, while camera direction-selection kernels do not move.
- 2026-08-25: Mathematical review concerns were accepted: the person geometry
  assumption, exact numeric/filter/degeneracy domains, checked receipt
  invariants and unchanged selector failure semantics are now frozen. Code
  review concerns were accepted: the design reuses the public camera selector
  for canonicalization, removes forgeable redundant constructor inputs, and
  includes all adjacent owner documents in migration.
- 2026-08-25: Re-review found that a filtered source cannot prove its original
  observation count. The field was removed from the checked ground result;
  input `N` remains explicit experiment provenance rather than a forgeable
  library receipt claim.
- 2026-08-25: Campaign inventory added the older single-source VP baseline ground
  interpretation to the same owner boundary; the camera selection primitive
  remains unchanged in camera-solver.
- 2026-08-25: Narrow review clarified that the old baseline is one source over
  multiple eligible VP clusters, placed its wrapper in the existing
  `by_vanishing_direction.py`, and added three-method Campaign parity coverage.
- 2026-08-25: Implementation landed with three ground interpretation entry
  points, a checked weighted person-line evidence builder, public exports,
  usage/design/test docs and Campaign wiring. Review closure centralized the
  Ground Normal value contract, kept ground-facing results through experiment
  metrics/serialization, completed method-owner provenance, and strengthened
  failure/exact-oracle smokes. Ground smoke is `105 passed`; Campaign smoke is
  `24 passed`; changed-file strict pyright is zero in both.
