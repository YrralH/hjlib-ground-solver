# AMASS HJ-Derived Plantar ZMin Ground

## Requirements

The user selected sequence-level plantar `zmin` as the practical AMASS ground
height for the current raw-data workflow. The public name and result must make
clear that this is an HJ-derived estimate, not a native or official AMASS
annotation.

The method consumes already-realized left/right plantar minimum-height tracks.
It does not own AMASS reading, SMPL forward, plantar topology, contact labels,
rendering, or scene geometry. It must not add a percentile, outlier peel,
static-foot gate, offset, or zero fallback under the `zmin` name.

The result is a single horizontal height proxy. It does not claim that AMASS
contains a ground plane, that every foot contact occurs at that height, or that
stairs/platforms are absent. A later contact-label task must treat it as a
derived reference and preserve AMASS fitting/coordinate uncertainty.

## Mathematical Architecture

For finite left/right plantar minimum tracks of equal length `T >= 1`, in
metres along the caller's chosen up axis,

```text
hL, hR in R^T
g_HJ = min({hL[t]} union {hR[t]})
```

The method returns `g_HJ` exactly in Python float precision after validation.
It also reports the per-side minima, the selected side and track-local frame index,
and the exact count of pooled samples equal to the global minimum.

Selection ties are deterministic: lower height, then left before right, then
earlier track-local frame. The tie rule affects evidence identity only, never the
height. Both tracks must be NumPy arrays with the same `(T,)` shape and the same
`float32` or `float64` dtype; all values must be finite. Inputs are never
mutated.

The returned provenance literal is frozen to:

```text
hj_derived_nonofficial
```

This is a semantic safety invariant, not configurable metadata. In exact real
arithmetic the method is vertically equivariant. For floating inputs, strict
numeric equivariance is required only when every shifted value and the selected
minimum plus offset are exactly representable in the input dtype. Evidence
identity additionally requires that ordering/equality relations are preserved;
ordinary rounding may otherwise change the height arithmetic or merge distinct
values. Smoke uses exactly representable values for the positive oracle and
separately records that a rounding collision carries no height-identity or
tie-count promise. The method is invariant to
frame permutation except for the reported track-frame identity under equal
minima.

### Numerical review contract

Review must be independent in three senses:

1. synthetic oracle review of selection, ties, validation, dtype, immutability,
   and vertical-shift behavior;
2. configured AMASS review over each frozen motion's complete native sequence,
   proving the new function reproduces direct plantar `zmin` for every tau
   candidate;
3. evidence review of minimum support concentration: exact-min multiplicity,
   samples within 0.5/1/2/5/10 mm, next-distinct gap, side/frame identity, and
   sensitivity to the accepted 3/5 mm plantar masks.

For this configured evidence review, pool both tracks into `x` with denominator
`2T`, promote values and thresholds to float64, and define:

```text
N_delta = count(x <= g_HJ + delta)
F_delta = N_delta / (2T)
delta in {0.0005, 0.001, 0.002, 0.005, 0.010} m
next_distinct_gap = min{x - g_HJ | x > g_HJ}
```

Each threshold reports count and fraction. If every pooled sample equals the
minimum, `next_distinct_gap` is `None`. Exact-min multiplicity is computed in
the input dtype by equality to the selected minimum; the promoted threshold
statistics are review evidence and do not alter the estimator.

The configured evidence additionally reports, for every delta:

- pooled side-frame count/fraction with denominator `2T`;
- left and right side-frame counts separately;
- unique-frame count/fraction from `min(hL, hR)` with denominator `T`;
- longest consecutive near-minimum run in frames and seconds.

Specifically, define

```text
b_delta[t] = (min(hL[t], hR[t]) <= g_HJ + delta)
```

and take the longest consecutive `True` run in `b_delta`. Duration is
`run_frame_count / frame_rate_in_hz`, including the one-frame case as one frame
period; it is not `(run_frame_count - 1) / frame_rate_in_hz`.

Temporal support alone cannot detect a one-vertex mesh spike. During the same
numerical forward, the raw operation retains only the selected foot's plantar
vertex heights at the current best minimum frame. It reports exact and
0.5/1/2/5/10 mm vertex count/fraction plus the next-distinct vertex-height gap.
No connectivity claim is made because the current plantar selector does not
own a mesh-adjacency contract.

For selected-foot/frame plantar heights `y`, with denominator `|y|`, promote
threshold arithmetic to float64 and define:

```text
V_delta = count(y <= g_HJ + delta)
spatial_fraction_delta = V_delta / |y|
next_distinct_vertex_gap = min{y - g_HJ | y > g_HJ}
```

The exact spatial count uses input-dtype equality to `g_HJ`. If every value in
`y` equals the minimum, `next_distinct_vertex_gap` is `None`.

The review may reject a semantic-ground claim while still confirming exact
implementation. Any visual escalation is bounded to at most 20 images viewed
by the implementing agent. Images must label the plane `HJ-derived
nonofficial`, not official AMASS ground.

## Code Architecture

Add
`src/hjlib_ground_solver/estimate_ground/by_hj_derived_plantar_zmin.py` with:

```python
estimate_hj_derived_plantar_zmin_ground(
    left_per_frame_minimum_height_in_meter,
    right_per_frame_minimum_height_in_meter,
) -> HJ_Derived_Plantar_ZMin_Ground_Result
```

The frozen/slots result schema is:

```text
input_dtype: str
frame_count: int
left_minimum_height_in_meter: float
right_minimum_height_in_meter: float
ground_height_in_meter: float
selected_side: Literal['left', 'right']
selected_frame_index_within_input_track: int
tied_global_minimum_sample_count: int
provenance: Literal['hj_derived_nonofficial']
```

`provenance` uses
`field(default='hj_derived_nonofficial', init=False)` so callers cannot
construct a differently labelled result. Export the result, provenance/side
literals, and function from the estimate subpackage and repository root.

The solver remains topology- and dataset-name-free. `hjlib-smpl` continues to
own shaped plantar indices; `hjlib-dataset-raw` continues to own AMASS
realization and configured evidence.

Extend the existing raw one-forward foot-sole operation. `Candidate_Tracks`
adds exactly one
`hj_derived_plantar_zmin_ground_result: HJ_Derived_Plantar_ZMin_Ground_Result`
for every tau. It also owns one compact
`selected_frame_plantar_vertex_height_in_meter: NDArray[np.float64]` captured
from the selected side/frame during the same chunk traversal. Compute the
public result once after scalar tracks are complete and validate the compact
spatial evidence against its side/frame/minimum. JSON,
comparison panels, profiles, diagnostic-frame construction, and side lines
must read height/provenance from that field. `union_height` remains available
for peeling and distribution evidence, but no downstream `np.min` may derive
the headline ground. Do not add a raw-reader `ground` field or a second mesh
forward. Rename visual labels to `HJ-derived zmin (nonofficial)` while
preserving the other diagnostic candidates.

Make the configured operation's `analyze_motion` receive its chunk
`forward_vertices` callable explicitly, alongside its existing injected
reducers. Production passes `forward_smplh_vertices`; smoke passes a counting
fake. This is testable dependency injection inside the configured operation,
not a public library abstraction.

Streaming spatial evidence uses exactly the estimator tie key:

```text
(float64 height, side priority left=0/right=1,
 frame_index_within_input_track)
```

It may replace an earlier-encountered right minimum when a later-encountered
left sample has the same height; equal minima on one side retain the earlier
track frame. Raw smoke covers both cases before result/evidence validation.

The post-review configured run loads all frames of each frozen native motion:
`frame_start_inclusive=0`, `frame_stop_exclusive=full_frame_count`, policy
`full_native_sequence`. Historical middle-80% v5/v7 artifacts remain immutable
and are not relabelled as full-sequence evidence.

## Smoke-Test Standard

Solver smoke covers left/right/earliest ties, exact tied-sample count,
float32/float64, negative and signed-zero heights, vertical shifts, input
immutability, a documented rounding-collision non-guarantee, frozen records,
public export identity, and rejection of wrong
type/shape/dtype/non-finite/mismatched inputs.

Raw smoke proves the configured operation calls the public result as its
single absolute-height source, serializes the frozen provenance, and never
labels the candidate as official. A fake `analyze_motion` run with `T` frames
and chunk size `B` asserts `ceil(T/B)` forward calls independent of tau count
and exactly one HJ-derived estimator call per tau. Existing bounded rendering,
publication, and source-provenance checks remain green.

The same smoke validates the compact selected-frame vertex vector and exact
formulas for pooled side-frame, per-side, unique-frame, longest-run, and
spatial-vertex support. Configured acceptance records direct-oracle error,
3/5 mm result/identity stability, selected side/local/native frame, and
per-source aggregates without extrapolating nine motions to all AMASS.

## Migration Plan

1. Land and independently review Mathematical and Code Architecture.
2. Implement the solver leaf, exports, and synthetic smoke.
3. Route the configured raw operation through the public result.
4. Publish a new bounded numerical/visual package and conduct multi-party
   numerical review; inspect no more than 20 images if visual escalation is
   useful.
5. Land usage/design/campaign docs and complete HJ review/checks.

No new dependency direction is required. Project descriptions are checked at
closure and should remain unchanged unless the implementation changes an owner
boundary.

## Modification History

- 2026-08-04: Registered after the user selected plantar zmin as the practical
  AMASS ground height and required the function name to distinguish HJ-derived
  output from official data. Implementation remains gated on independent
  Mathematical and Code Architecture review.
- 2026-08-04: Mathematical review found no Critical and three Concerns. Changed
  native-frame evidence to track-local indexing, qualified floating vertical
  equivariance, and froze the float64 near-minimum count/fraction and
  next-distinct-gap review formulas. Re-review is pending.
- 2026-08-04: Mathematical re-review left one floating-height Concern. Limited
  strict numerical equivariance to exactly representable input-dtype shifts;
  final narrow review is pending.
- 2026-08-04: Code Architecture review found no Critical and three Concerns.
  Froze every result field and non-configurable provenance, made the result an
  explicit per-tau `Candidate_Tracks` SSOT, and required a counting fake smoke
  for chunk-forward and estimator call counts. Re-review is pending.
- 2026-08-04: Independent numerical pre-review found two Critical evidence
  gaps and three Concerns. Changed the new acceptance run from middle-80% to
  full native sequences; added same-pass selected-frame plantar-vertex support,
  unique-frame/per-side support and longest runs. Historical artifacts remain
  frozen. Mathematical and Code Architecture narrow re-review is required.
- 2026-08-04: Mathematical review of the expanded evidence left two formula
  Concerns. Froze the unique-frame longest-run boolean/duration definition and
  selected-frame vertex count/fraction/next-gap formula. Re-review is pending.
- 2026-08-04: Code Architecture review of same-pass spatial evidence left one
  tie-order Concern. Froze the streaming key to height, left-before-right, then
  track frame, with two raw smoke fixtures. Re-review is pending.
- 2026-08-04: All design and numerical-evidence reviews closed at Critical 0 /
  Concern 0. Implemented the public result/function, exports, synthetic smoke,
  raw SSOT, counting fake, full-native-sequence v9 run, same-pass spatial
  evidence, and explicitly nonofficial visuals.
- 2026-08-04: Final Mathematical Architecture, Code Architecture, and numerical
  implementation reviews closed at Critical 0 / Concern 0. The evidence accepts the
  implementation as an HJ-derived nonofficial height proxy while preserving
  the isolated-minimum warning for future contact-label work. Targeted and
  repository-wide smoke, strict solver pyright, and diff checks passed.
