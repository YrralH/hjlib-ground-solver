# AMASS Ground Static-Foot Robust

## Requirements

This task consumes the accepted left/right plantar observation tracks and must
never return to full-body minima or toe-joint-plus-offset evidence. It has two
phases:

1. implement a common-domain `plantar_humor_baseline` so foot-only zmin and a
   HuMoR-style static-foot cluster can be compared on the same SMPL-H mesh
   realization;
2. later decide whether temporal support, multilevel classification, or an
   explicit unknown result justifies a stronger robust estimator.

Only phase 1 is authorized now. Its output is a reproducible candidate and
sensitivity surface, not semantic ground truth and not the final robust method.
The exact toe-joint HuMoR comparator remains unchanged as historical
provenance evidence.

The configured comparison must reuse the frozen nine-motion middle-80-percent
SMPL-H cohort, native 16 betas, no PCA, and the shaped plantar masks. Every
candidate for one motion must share one mesh forward. Both `tau=3 mm` and
`tau=5 mm` are reported; neither is selected by this task.

## Mathematical Architecture

### Inputs and alignment

For `T >= 2`, the solver receives four finite, same-dtype one-dimensional
tracks:

```text
hL, hR in R^T       left/right plantar minimum height, metres
sL, sR in R^(T-1)   left/right interval median plantar speed, metres/second
```

The interval speed between frames `i` and `i+1` is assigned to frame `i`; the
last interval value is repeated for frame `T-1`, matching the historical
HuMoR gate's terminal alignment without converting back to per-frame
displacement:

```text
v[k] = s[k]       for 0 <= k < T-1
v[T-1] = s[T-2]
```

At speed threshold `v_max`, side `q` contributes frame `k` iff
`v_q[k] < v_max`. Equality is rejected. Left samples precede right samples.
The sample height is the side's plantar minimum; contact eligibility uses the
median motion of all selected plantar vertices. This intentionally prevents a
single minimum-identity switch from defining the motion gate.

### Height clustering and candidate

Eligible heights are pooled and clustered in one dimension with DBSCAN. The
phase-1 family freezes:

```text
eps = 0.005 m
min_samples = 3
speed sweep = 0.1, 0.2, 0.5, 1.0 m/s
visual/reference speed = 0.5 m/s
```

Unlike exact HuMoR parity, DBSCAN label `-1` is explicit noise and is never a
candidate cluster. Every non-noise label records its sample count, sorted
unique native-frame indices, height minimum/median/maximum/span, and maximum
adjacent gap in sorted height order. This does not prohibit DBSCAN density
chaining, but makes an overly broad chained cluster directly auditable. The
selected candidate is the lowest cluster median; exact median ties use
ascending integer label. No
`0.01 m` toe-joint offset is applied because `hL/hR` already observe the
plantar surface.

Statuses are:

- `candidate`: at least one non-noise cluster exists;
- `no_contact_samples`: the speed gate retains no sample;
- `noise_only`: samples exist but DBSCAN finds no non-noise cluster.

Only `candidate` has a numerical candidate height. No zero fallback is
allowed. A candidate remains a method output, not an accepted ground plane.

For exactly representable shifts away from DBSCAN and selection boundaries,
the estimator is equivariant to adding the same finite constant to both height
tracks; finite-precision boundary exceptions remain explicit. It is invariant
to horizontal transforms because no horizontal
coordinate enters after the owner observation layer. Time order affects only
the already-derived interval speeds; clustering pools heights and has no
temporal persistence rule. That limitation is phase-1 evidence for the later
robust design.

### Fair comparison metrics

For each motion, `tau in {3,5} mm`, and speed threshold, the configured
operation records status, eligible sample counts, cluster summaries, and the
static-foot candidate. At the `0.5 m/s` reference threshold it reports:

```text
plantar_humor - absolute plantar zmin
plantar_humor - current 1 mm / 5x peeling
plantar_humor - mild 0.75 mm / 4x peeling
```

It also reports the `tau=5 mm - tau=3 mm` candidate delta. Aggregation is
descriptive only: median/range and per-motion values. A missing candidate stays
missing and is never coerced to zero. Aggregation uses only available
candidates and always reports available and missing motion counts beside each
median/range.

### Visual acceptance

The bounded artifact contains:

1. the canonical SMPL-H/SMPL-X mask grid;
2. nine native-time profiles with absolute zmin, current peeling, mild
   peeling, and the reference plantar-HuMoR candidate;
3. a `3 x 3` numerical comparison plot showing method deltas for both `tau`
   values and the speed sweep;
4. at most 15 side-orthographic frames with absolute/current/mild labeled
   lines, the shared plantar points, and a fourth plantar-HuMoR line only when
   its candidate exists. A missing candidate is labeled `unavailable` and is
   never drawn at zero.

Diagnostic frames prioritize the absolute minimum, a selected static-support
sample nearest the chosen cluster median, and a mild-only peeled frame or
ordinary context. Equal nearest-sample distances resolve by pooled order,
which is left side first and then ascending native frame. The images must call every line a candidate, never official
or semantic ground.

## Code Architecture

### Solver owner

Add
`src/hjlib_ground_solver/estimate_ground/by_static_foot_plantar_humor.py`.
It owns frozen/slots records and one public function:

```python
estimate_static_foot_plantar_humor_baseline(
    left_height_in_meter,
    right_height_in_meter,
    left_interval_median_speed_in_meter_per_second,
    right_interval_median_speed_in_meter_per_second,
    config,
) -> Static_Foot_Plantar_HuMoR_Result
```

`Static_Foot_Plantar_HuMoR_Config` contains positive finite speed/epsilon and
an integer DBSCAN minimum sample count. Result evidence contains aligned
eligible samples and non-noise clusters, but not AMASS names, topology, mesh,
or rendering state. Export the records/status/function from the estimate
subpackage and root. Keep the exact toe-joint HuMoR module untouched.

The module decomposes validation/copying, terminal speed alignment, pooled
sample construction/DBSCAN, cluster reduction, and public orchestration into
ordinary stateless functions. Complexity is `O(T)` outside scikit-learn's
DBSCAN implementation.

### Raw configured operation

Extend the existing
`test/amass_ground_foot_sole_domain.py` rather than create another mesh
realization. Its chunk loop already owns all required scalar tracks. After the
five mask tracks are complete, call the new solver for the four speed
thresholds. `Candidate_Tracks` owns one typed four-element result tuple in the
exact `(0.1, 0.2, 0.5, 1.0)` order. JSON, profile, comparison, diagnostic-frame
selection, and side rendering must consume this same immutable tuple; no
parallel result dict or renderer-side recomputation is allowed. Add the
results to `analysis.json`, add the dedicated comparison PNG, and extend
profile/side renders. The operation remains test-only; the raw reader gains no
solver API or derived field.

The receipt adds the new solver source hash and continues to gate all previous
source/data/model/repository manifests. Publication remains exclusive,
atomic, no-clobber, and bounded.

### Smoke-Test Standard

Solver smoke covers strict speed equality, repeated terminal alignment,
left-before-right pooling, nonnegative speed validation, DBSCAN noise
exclusion, lowest median selection, label tie order, both empty statuses,
float32/float64, vertical equivariance, input immutability, frozen records,
an explicit density-chained fixture with audited span/gap evidence, public
export identity, and every shape/dtype/config/non-finite rejection.

Raw smoke freezes the four thresholds and reference value, confirms the new
solver source is in provenance, exercises comparison-payload missing/candidate
branches, and spies on the side renderer to prove all four candidate lines are
drawn when available and only the three zmin-family lines are drawn when the
static candidate is missing. The configured v6 run is the real-data
numerical/visual gate.

## Migration Plan

1. review this Mathematical and Code Architecture;
2. implement/check the solver leaf and its public exports;
3. extend/check the raw configured operation without a second mesh forward;
4. publish and synchronize v6, then review the per-motion numbers and images;
5. land usage/design/campaign docs and run the HJ closed-loop checks.

No dependency pin or project description changes are needed: the existing
dependency direction and owner summaries already cover this surface.

## Modification History

- 2026-08-04: Activated phase 1 after the user asked whether the new zmin was
  implemented and requested a fresh numerical/visual comparison. The design
  refuses the old unfair toe-joint comparison and defines a common plantar
  HuMoR-style baseline over the existing height/speed tracks.
- 2026-08-04: Independent Mathematical Architecture review returned no
  Critical and closed two Concerns by adding auditable cluster span/gap
  evidence plus explicit missing-candidate rendering and aggregation. It also
  qualified finite-precision equivariance and fixed diagnostic tie order.
- 2026-08-04: Independent Code Architecture review returned no Critical and
  closed its one Concern by making the per-tau four-result tuple the raw-side
  SSOT for every serializer and renderer. Implementation is authorized.
