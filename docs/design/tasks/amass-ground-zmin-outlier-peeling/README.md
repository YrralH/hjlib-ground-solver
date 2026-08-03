# AMASS Ground ZMin Outlier Peeling

## Status

- Layered Design state: complete / independently reviewed
- Campaign ledger:
  [`hjlib-dataset-raw/campaigns/01_amass_raw_support/task_ground_zmin_outlier_peeling/`](../../../../../hjlib-dataset-raw/campaigns/01_amass_raw_support/task_ground_zmin_outlier_peeling/)
- Implementation state: public core, synthetic smoke, raw operation, and
  authoritative bounded v3 evidence complete; full gates pass

## Requirements

Given the reviewed per-frame full-mesh lower envelope `m[t]`, determine whether
the currently lowest value or collectively lowest value subset is separated
enough to treat as a lower-tail outlier group. Remove the whole group and repeat
until no eligible group remains. Excessive repetition or removal must produce
an explicit unstable result rather than an accepted candidate.

The implementation must be deterministic, bounded, immutable, finite-input
only, metre-valued, and efficient for long motions. It must preserve temporal
indices for diagnostics while making the detection decision only in sorted
height space. It must not claim that the final scalar is ground truth, infer a
supporting body part, split equal-valued observations, or normalize AMASS.

Generalized ESD is not selected: the
[NIST definition](https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h3.htm)
requires an approximately normal univariate distribution, which conflicts with
the strongly non-normal and often multimodal lower-envelope series. Tukey, MAD,
and the [Hubert--Vandervieren adjusted boxplot](https://wis.kuleuven.be/statdatascience/robust/papers/2008/adjboxplot-revision.pdf)
remain comparison references, but boxplot fences target potential point
outliers in continuous unimodal distributions and do not directly encode the
requested separated collective prefix and repeated-removal contract. This task
therefore names its method plainly as a configured sorted-gap peeling heuristic
rather than a universal outlier algorithm.

## Ownership And Dependency Direction

The public mathematical core belongs in `hjlib-ground-solver` beside the
existing mesh-lower-envelope summary. `hjlib-dataset-raw` owns only the
configured AMASS operation and evidence. The operation composes
`AMASS_Raw_Reader`, `hjlib-smpl`, and `hjlib-ground-solver`; no production
dependency is added between the two owning libraries.

The accepted fixed-coverage artifact
`task_ground_zmin_family/artifacts/bounded_reduced10_v2` and its generator are
frozen. The new operation may reuse their motion-loading and mesh-realization
functions through an explicit operation-level boundary, but it must have a new
source file, artifact directory, receipt, and source hashes.

## Mathematical Architecture

### 1. Input and ordering

The input is a nonempty finite floating array

```text
m = (m[0], ..., m[T-1]), T > 0,
```

measured in metres along the already-supported up axis. Copy it to float64 and
reject if conversion changes any value to non-finite. Sort pairs
`(height, original_frame_index)` lexicographically. The frame index
is only a deterministic tie-break and a temporal-diagnostic key. Because each
peel removes a sorted prefix, sorting occurs once. Let `s` be the first retained
sorted position; initially `s=0`.

Equal values cannot be split. A candidate boundary after retained-prefix count
`k` is eligible for examination only when

```text
y[s+k-1] < y[s+k].
```

### 2. Search bound and retained safety

For current retained count `N=T-s`, examine only

```text
1 <= k <= L,
L = min(
    floor(q_round * N),
    n_round,
    N - n_retain,
).
```

`q_round` is an exact decimal fraction in `(0,1]`, `n_round > 0`, and
`n_retain > 0`. Per-call validation requires `n_retain <= T`; a configuration
that cannot retain its required minimum is invalid input, not a stable result.
`floor(q_round*N)` parses the decimal string, then evaluates its coefficient
and exponent through integer arithmetic. It does not use binary float or the
ambient Decimal context. If `L < 1` after this validation, the result is stable because no
configured low-prefix boundary is searchable. These are search and safety
bounds, not percentile estimators.

### 3. Gap evidence

For each searchable boundary after `k`, define

```text
g_boundary = y[s+k] - y[s+k-1] > 0.
```

Its local reference uses exactly the next at most `w` adjacent gap slots above
the boundary and then filters zero gaps from those fixed slots:

```text
G_ref = positive values from
        y[j+1]-y[j], j=s+k,...,min(s+k+w-1,T-2)
r = median(G_ref), when G_ref is nonempty.
```

The boundary is eligible exactly when:

```text
g_boundary >= delta_min
and (G_ref is empty or g_boundary / r >= rho_min).
```

The implementation must not scan farther upward to collect `w` positive gaps.
Because `G_ref` contains only positive gaps, its median is positive whenever it
is present; the empty branch is retained when every gap in the fixed slot
window is zero or when no slot remains above the boundary.
`delta_min > 0`
metres prevents tiny numerical spacing from becoming scale-free evidence.
`rho_min > 1` requires the boundary to dominate nearby within-distribution
spacing. `w > 0` is explicit.

Finite inputs can still overflow derived float64 arithmetic. After sorting, the
implementation computes adjacent gaps and rejects the call if any gap is
non-finite. For an even reference count, its median uses the overflow-safe
formula `a + (b-a)/2` for the two ordered middle positive gaps, not `(a+b)/2`.
The median is explicitly checked finite. Every finite-reference ratio is then
computed once and must be finite; otherwise the call raises `ValueError`.
Eligibility and the recorded trace use that same validated ratio, so comparison
and diagnostics cannot silently disagree.

Examine boundaries in increasing `k` and choose the first eligible boundary.
This is the smallest separated prefix containing the current minimum. It avoids
merging two already-separable low groups merely because a later boundary has a
larger ratio; the later group is reconsidered only after the first peel. The
rule is deterministic and deliberately local. It is not calibrated as a
probability or p-value.

### 4. Iteration and stop states

The immutable configuration domain is:

```text
q_round_decimal: decimal string with Decimal value in (0,1]
n_round: Python int > 0, excluding bool
n_retain: Python int > 0, excluding bool
w: Python int > 0, excluding bool
delta_min: finite Python float > 0, metres
rho_min: finite Python float > 1
R_max: Python int >= 0, excluding bool
q_total_decimal: decimal string with Decimal value in (0,1)
n_total: Python int > 0, excluding bool
```

The decimal strings must contain no leading or trailing whitespace and must
parse through `Decimal` to finite values in their stated intervals. Their
verbatim form is retained for the result contract; NaN, infinity, and
non-string fraction inputs are invalid. Per-call validation additionally
requires `n_retain <= T`.
A total fractional budget whose exact `floor(q_total*T)` is zero is valid: it
allows analysis but blocks every proposed peel as
`unstable_removal_budget`. Configured positive bounds and their effective
integer counts are therefore distinct.

The optional `frame_rate_in_hz` input is either `None` or a Python `int`/`float`
excluding `bool`; it is normalized to float and must be finite and positive.
NumPy scalar numerics are rejected. It affects only run duration, never peel
eligibility. Duration is `longest_run_frame_count / normalized_frame_rate`; a
non-finite derived duration rejects the call with `ValueError`.

For a chosen prefix, compute proposed cumulative removal `s+k` before mutating
state. Three outcomes exist:

```text
stable_candidate
unstable_maximum_round_count
unstable_removal_budget
```

If no eligible boundary exists, return `stable_candidate`. Otherwise, if the
number of already-applied rounds equals `R_max`, return
`unstable_maximum_round_count` and record the proposed boundary without
applying it. `R_max=0` is valid and permits a detection-only run in which the
first eligible peel is blocked. Reaching exactly `R_max` applied rounds is
therefore stable when a fresh scan finds no further eligible group.

If proposed cumulative removal exceeds either

```text
floor(q_total*T) or n_total,
```

return `unstable_removal_budget` and again do not apply the proposed peel. The
exact fractional count uses the same Decimal coefficient/exponent integer floor
for `q_total*T`; the frame-count budget uses `n_total` directly. A proposal
exactly equal to both bounds is allowed.
Retained safety remains an independent search bound. Otherwise apply the peel
(`s <- s+k`), append its trace, and scan again.

The returned current candidate is always `y[s]`. Only
`status='stable_candidate'` permits downstream code to treat it as this
method's accepted lower-envelope candidate. An unstable result exposes the
current scalar for diagnosis but must not be silently promoted to a ground.

### 5. Trace and temporal diagnostic

Each applied round records:

- one-based round index;
- retained count before the peel;
- removed count and cumulative removed count;
- removed minimum and maximum heights;
- boundary gap and reference median gap, with an explicit empty-reference flag;
- finite gap ratio, or `None` when the reference is empty;
- candidate before and after the peel;
- longest consecutive run among removed original frame indices, in frames and
  optional seconds.

An unstable result additionally records the same proposed-boundary evidence as
a blocked peel. Temporal order cannot make a boundary eligible, ineligible, or
change which boundary wins. It only helps the analyst distinguish isolated
low frames from a sustained low episode.

### 6. Invariants

For every applied round:

```text
removed_count > 0
cumulative_removed_count strictly increases
candidate_after > candidate_before
boundary_gap = candidate_after - removed_maximum
cumulative_removed_count <= floor(q_total*T)
cumulative_removed_count <= n_total
retained_count >= n_retain
```

No tied height is divided across removed and retained sets. Concatenating the
applied removed frame-index groups and retained frame indices is a permutation
of `range(T)`. The input array is unchanged. Reordering the same height
multiset produces the same status, candidate, round counts, gaps, and removed
height groups; only temporal-run diagnostics may differ.

### 7. Complexity

Sorting once costs `O(T log T)` time and `O(T)` memory. Each applied round and
the final stable or blocked scan examine
at most `L` candidate boundaries and at most `w` following gaps per boundary.
The direct implementation sorts each reference window and each removed native
index group, so its auditable upper bound is
`O((R_max+1)*L*w*log(w) + sum(k_round*log(k_round)))`. Configured `w`, `L`,
round, and removal caps keep this small relative to mesh realization. No mesh,
histogram, ECDF, or derived motion is persisted by the core.

### 8. Interpretation limit

A separated low group can be contamination, penetration, another genuine
support mode, or valid human-scene contact. This scalar method cannot decide
which. Its final output remains an outlier-peeled lower-envelope candidate.
The later ground-method analysis must retain `unknown` and
`multilevel_or_piecewise` outcomes and compare against static-foot methods and
human-reviewed sequences.

## Initial AMASS Profile

The first bounded operation must pass configuration explicitly and record it in
JSON. Values are not frozen until a small read-only probe and synthetic tests
verify scale and boundary behavior. The profile will specify:

- per-round fraction and frame-count search caps;
- minimum retained frame count;
- local reference-window gap count;
- minimum absolute boundary gap in metres;
- minimum gap ratio;
- maximum applied rounds;
- total removal fraction and frame-count caps.

The profile may be adjusted before the first authoritative artifact, but not
silently tuned per sequence. Any later profile is a separately named artifact.

## Synthetic Acceptance Matrix

1. One isolated low value and one dense tied low cluster each peel once.
2. Two separated low clusters peel in two rounds.
3. Exactly `R_max` necessary rounds may be stable; an indicated next round is
   blocked as `unstable_maximum_round_count`. `R_max=0` blocks the first
   eligible proposal.
4. A proposal exactly equal to both total-removal bounds is allowed. Exceeding
   either bound is blocked without changing the candidate or applied-round
   trace. If round and removal budgets both block it, round status wins.
5. Equal heights are never split; a plateau above a positive boundary supports
   the explicit empty-reference branch. A separate fixed-slot fixture places
   `w` zero gaps followed immediately by a positive gap and verifies that the
   implementation does not scan past the window.
6. Uniform spacing and sub-`delta_min` numerical gaps remain stable.
7. Time permutations preserve the value-space result while changing allowed
   run diagnostics.
8. Shape, dtype, empty, non-finite, `n_retain>T`, frame-rate, and every invalid
   configuration field raise `ValueError`. Frame rate accepts only `None` or a
   finite positive Python numeric excluding bool. An extremely small positive
   rate that makes a nonzero run duration overflow is rejected. Extreme finite
   values whose derived gap, median, or ratio overflows are rejected; large
   positive gaps also verify the overflow-safe even median.
9. Inputs are not mutated; result/config/trace records are frozen; public
   re-exports are identical.
10. A straightforward repeated-sort reference oracle agrees with the optimized
    sort-once implementation across deterministic randomized cases.
11. A frozen two-low-cluster fixture makes the first and second separating
    boundaries eligible and verifies that increasing-`k` selection peels them
    in two rounds rather than merging them.
12. Long-decimal q-round and q-total values straddling an integral count prove
    that floor arithmetic is independent of the ambient Decimal context; a
    separate `0.58*100=58` boundary confirms exact integral selection.

## Code Architecture

### 1. Public solver module

Add
`src/hjlib_ground_solver/estimate_ground/by_mesh_lower_envelope_peeling.py`.
It owns one explicit public entry:

```python
peel_separated_mesh_lower_envelope_prefixes(
    per_frame_minimum_height_in_meter,
    config,
    frame_rate_in_hz=None,
) -> Mesh_Lower_Envelope_Peeling_Result
```

The public immutable records and exact fields are:

```python
type Mesh_Lower_Envelope_Peeling_Status = Literal[
    'stable_candidate',
    'unstable_maximum_round_count',
    'unstable_removal_budget',
]

@dataclass(frozen=True, slots=True)
class Mesh_Lower_Envelope_Peeling_Config:
    maximum_candidate_fraction_per_round_decimal: str  # q_round
    maximum_candidate_frame_count_per_round: int       # n_round
    minimum_retained_frame_count: int                   # n_retain
    reference_gap_window_size: int                      # w fixed slots
    minimum_boundary_gap_in_meter: float                # delta_min
    minimum_gap_ratio: float                            # rho_min
    maximum_round_count: int                            # R_max
    maximum_total_removed_fraction_decimal: str         # q_total
    maximum_total_removed_frame_count: int              # n_total

@dataclass(frozen=True, slots=True)
class Mesh_Lower_Envelope_Peel_Proposal:
    round_index: int
    retained_frame_count_before: int
    removed_frame_count: int
    cumulative_removed_frame_count: int
    removed_original_frame_indices: tuple[int, ...]
    removed_minimum_height_in_meter: float
    removed_maximum_height_in_meter: float
    boundary_gap_in_meter: float
    reference_gap_slot_count: int
    reference_positive_gap_count: int
    reference_median_gap_in_meter: float | None
    gap_ratio: float | None
    candidate_before_in_meter: float
    candidate_after_in_meter: float
    longest_removed_run_frame_count: int
    longest_removed_run_duration_in_second: float | None

@dataclass(frozen=True, slots=True)
class Mesh_Lower_Envelope_Peeling_Result:
    status: Mesh_Lower_Envelope_Peeling_Status
    config: Mesh_Lower_Envelope_Peeling_Config
    frame_count: int
    frame_rate_in_hz: float | None
    absolute_minimum_height_in_meter: float
    current_candidate_height_in_meter: float
    accepted_candidate_height_in_meter: float | None
    applied_removed_frame_count: int
    retained_frame_count: int
    maximum_total_removed_frame_count_by_fraction: int
    effective_maximum_total_removed_frame_count: int
    applied_peels: tuple[Mesh_Lower_Envelope_Peel_Proposal, ...]
    blocked_peel: Mesh_Lower_Envelope_Peel_Proposal | None
```

`accepted_candidate_height_in_meter` equals the current candidate only for
`stable_candidate`; it is `None` for both unstable states. This makes accidental
consumption of an unstable diagnostic harder.

Each proposal also carries the removed original-frame indices sorted in native
time. Its size is bounded by the per-round search caps, makes visualization and
run diagnostics reproducible, and prevents the raw operation from reimplementing
the selection algorithm. `reference_median_gap_in_meter` and `gap_ratio` are
`None` exactly when `reference_positive_gap_count == 0`; the slot count
distinguishes a plateau window from the top-of-series boundary.

The module is re-exported from `estimate_ground/__init__.py` and the package
root. No dependency or `pyproject.toml` change is required; the current package
description already covers mesh lower-envelope candidates.

### 2. Core function decomposition

Keep state-free work in ordinary module functions with managed-noun names:

1. the frozen config validates its static fields in `__post_init__`;
2. `floor_exact_decimal_count` evaluates fraction/count floors by Decimal
   coefficient/exponent integer arithmetic without ambient context rounding;
3. `normalize_mesh_lower_envelope_peeling_input` validates/copies float data,
   optional FPS, per-call retained count, and exact total fractional budget; it
   returns the float64 copy, normalized FPS, fractional count, and effective
   count without sorting;
4. `find_first_mesh_lower_envelope_peel_proposal` scans increasing `k` for one
   eligible boundary and builds the complete proposal, including overflow-safe
   median/ratio and native-time run diagnostics;
5. `peel_separated_mesh_lower_envelope_prefixes` owns only the loop, stop-state
   priority, immutable result assembly, and applied-versus-blocked state.

The public entry lexsorts the normalized copy once, computes all adjacent gaps,
and rejects any non-finite gap before entering the loop. The finder receives
that sorted copy, its original-index permutation, precomputed validated gaps,
and current `s`; it does not sort or mutate. The core does not import AMASS,
SMPL, OpenCV, Typer, or filesystem code.

### 3. Solver smoke residence

Add `test_smoke/test_mesh_lower_envelope_peeling.py` and register its single
master entry in `test_smoke/test_all_func.py`. Tests call the public import and
also verify subpackage/root symbol identity. A deliberately simple repeated-sort
oracle remains test-only. The smoke matrix is the one frozen above, including
manual stop-boundary oracles; randomized oracle equality cannot replace those
semantic cases.

### 4. Configured AMASS operation

Add `hjlib-dataset-raw/test/amass_ground_zmin_outlier_peeling.py`. It is an
acceptance operation, not a public raw reader API. Its top-level import remains
data-free with respect to every `hjlib-*` module. `main` first sets
`HJLIB_SMPL_MODELS_PATH`; `execute_operation` then dynamically imports solver,
config, SMPL, raw, and the frozen `test/amass_ground_zmin_family.py` sibling
operation. It reuses only:

- the nine motion identities and reduced-10 realization constants;
- native-window loading, model loading, and chunked mesh analysis;
- model/repository/path/manifests/hash validation and package-version helpers.

This reuse intentionally performs the already-reviewed fixed-coverage summary
as a negligible side result while obtaining the per-frame minima. Refactoring
those helpers into production raw code is rejected because they compose three
owners and because modifying the accepted v2 generator would invalidate its
recorded source hash. The new operation never calls the old `run_operation` or
publisher.

The operation resolves `helper.__file__` and requires exact equality with
`REPOSITORY_ROOT/test/amass_ground_zmin_family.py`. Before any model/data work,
it requires the frozen v2 receipt at
`task_ground_zmin_family/artifacts/bounded_reduced10_v2/receipt.json`, verifies
its hard-coded SHA-256
`c7d615d34bb47fe558fa8fe27c6426550ad5b8658882249359a1c8cbfd0d5d60`,
and checks both current frozen sources against the receipt values:

```text
helper generator = 61ea7b344809c6c5ac2c9c8ce25b52ebe203e9bcb32e7c94de44f6b188078183
lower-envelope core = f759cb40e09ee02ddf24fd63661fee20a2fbc27dbd66ea89ba88bd340b86dfcb
```

Any mismatch aborts the operation. The new receipt still records before/after
hashes for all four executed sources. Thus it proves both frozen-baseline
identity and no mid-run change.

Because the old model-root equality and repository/source before-after checks
live inside the old `run_operation`, the new `execute_operation` explicitly
reimplements them. It calls reusable low-level validators where available but
does not assume importing helpers activates an old high-level gate.

The new Typer CLI exposes every peeling configuration field as a sequence-wide
option and constructs one config shared by all nine motions. It retains the
existing explicit model root, sibling repository roots, output path, device,
and chunk size. Imports of `hjlib-*` remain delayed until after the CLI sets the
SMPL model environment. Data-free import/help smoke verifies that boundary.
Only the reused `Motion_Numerical_Input_Error` becomes a per-motion
`invalid_input` record and panel while remaining motions continue. Residence,
frozen-source, configuration, model, provenance, and publication failures abort
the complete operation. A peeling-core `ValueError` is wrapped as an operation
failure and aborts; it is never downgraded to one motion's invalid record.

The operation implementation is decomposed as follows:

- `build_peeling_config`: static sequence-wide config construction;
- `analyze_configured_motions`: bounded model cache, forward, core call, and the
  narrow mesh-input invalid policy;
- `verify_operation_invariants`: data/repository/source before-after gates;
- `build_operation_payloads`: JSON-only result/receipt construction;
- `execute_operation`: delayed imports, residence/model/source preparation,
  device setup, and orchestration of those bounded functions;
- `run_operation`: frozen-baseline error translation only.

### 5. Artifact and visualization

The operation atomically publishes a new no-clobber directory containing:

- `analysis.json`: exact config plus per-motion result and realization record;
- `confirmation.png`: a 3x3 review surface;
- `receipt.json`: completion marker, input/model/source/repository hashes,
  package versions, timing/memory, unchanged AMASS manifests, and file hashes.

Each panel plots native-time minima, colors points removed by applied round,
draws the current candidate, and marks a blocked proposal distinctly. A compact
sorted-low-tail inset shows the selected gap boundaries because decisions occur
in sorted value space rather than time. Text includes status, removed count,
round count, and candidate. The figure is diagnostic only and draws no ground
plane.

The receipt hashes the new generator, frozen helper generator, new peeling
core, and existing lower-envelope core before and after execution. Publication
retains the accepted exclusive-single-writer assumption and verifies JSON/PNG
reopen before atomic rename.

### 6. Raw smoke and documentation

Add `test_smoke/test_amass_ground_zmin_outlier_peeling.py` and register it in the
raw master. It checks data-free import/help, the shared nine identities, all CLI
config switches, exact helper/receipt/source hash gates, and
no-clobber/outside-data output rules. Config construction and validation belong
to solver smoke. Any raw smoke that needs solver imports runs in an isolated
subprocess with an explicit model environment; the raw master process never
pre-imports the solver and cannot freeze `hjlib-smpl` model state ahead of a CLI
run.

The configured real-data command and artifact contract live in
`docs/design/test.md`; reusable public use belongs in the solver usage doc.

### 7. Implementation order

1. implement solver records/core and exhaustive synthetic smoke;
2. pass targeted smoke and strict pyright;
3. implement data-free AMASS operation and raw smoke;
4. run a small read-only minima probe to choose one sequence-wide profile;
5. publish the bounded nine-motion artifact and inspect/synchronize the PNG;
6. land docs, independent implementation/HJ reviews, and full repository gates.

## Modification History

- 2026-08-03: Landed Requirements and Mathematical Architecture for independent
  review. No implementation has started.
- 2026-08-03: Mathematical Architecture passed independent review after
  resolving retained-count validity, fixed-slot references, first-eligible
  selection, stop equality/priority, and all derived finite arithmetic. Landed
  Code Architecture for independent review.
- 2026-08-03: Code Architecture passed independent review with zero Critical
  and zero Concern findings. Started solver implementation and synthetic smoke.
- 2026-08-03: Implemented the immutable public core and raw acceptance
  operation, closed exact-floor/error-boundary/structure findings, and passed
  31 solver smoke cases, master smoke, and strict pyright. Published the
  authoritative nine-motion v2 package; its analysis and PNG are byte-identical
  to historical pre-review v1. Closure review remains.
- 2026-08-03: Closure review removed one remaining dependency on Python's
  integer-to-string digit limit, added a reduced-limit exact-floor regression,
  and published v3. Its analysis and PNG remain byte-identical to v1/v2; v3 is
  the authoritative current-source receipt.
- 2026-08-03: Logic-boundary, implementation, and cross-document consistency
  closure reviews each returned zero Critical and zero Concern findings. Final
  31-case solver smoke/master/strict-pyright and 35-case raw smoke/master plus
  task-targeted strict pyright passed; closed the Layered Design task.
