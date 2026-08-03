# AMASS Ground ZMin Family

## Status

- Layered Design state: Requirements, Mathematical Architecture, Code
  Architecture, implementation, and independent HJ review complete
- Campaign ledger:
  [`hjlib-dataset-raw/campaigns/01_amass_raw_support/task_ground_zmin_family/`](../../../../../hjlib-dataset-raw/campaigns/01_amass_raw_support/task_ground_zmin_family/)
- Implementation state: public core and configured bounded evidence complete

## Requirements

Implement the simplest reproducible AMASS ground-candidate family without
claiming more semantics than its input contains. The observation is the lowest
full-body mesh height in every frame. Sequence-level candidates are the
absolute minimum and exact retained-coverage order statistics of that time
series.

The task must:

- preserve native AMASS coordinates;
- use supported `+Z` as the explicit analysis axis, not as an official AMASS
  zero-plane claim;
- realize meshes in bounded chunks and retain only one scalar per frame;
- use exact finite-sample order statistics rather than interpolated quantiles;
- expose how far each derivative rises above absolute zmin and how many native
  frames it discards;
- reject invalid numerical input rather than silently filtering it;
- keep method selection and semantic-ground acceptance outside this task.

The task must not:

- add `ground` to `AMASS_Raw_Reader`;
- shift vertices, rewrite archives, or create a converted dataset;
- call a lower-envelope candidate native ground, official ground, or ground
  truth;
- infer contact, supporting body part, stairs, slopes, chairs, platforms, or
  scene geometry from a scalar height series;
- choose a universal percentile before the later method-analysis task has
  comparable methods and human-reviewed targets.

## Ownership And Dependency Direction

The durable campaign ledger and configured AMASS identities live in
`hjlib-dataset-raw`. Reusable lower-envelope mathematics lives here in
`hjlib-ground-solver`. Body-model construction and forward remain owned by
`hjlib-smpl`.

The permitted composition is:

```text
AMASS_Raw_Reader
    -> task-owned configured operation
        -> hjlib-smpl chunked forward
        -> per-frame mesh minimum on the forward device
        -> hjlib-ground-solver lower-envelope summary
```

No production module adds a `hjlib-dataset-raw <-> hjlib-ground-solver`
dependency. The configured operation may compose installed siblings but is not
a raw public method.

The old `get_ground_geometry/by_smpl.py` is not reused as the core: it extracts
rough torso pillars and returns a display mesh under different geometric
assumptions. The new managed noun is a mesh lower envelope, not ground geometry
from presumed on-ground people.

## Mathematical Architecture

### 1. Per-frame observation

For a realized body mesh with frames `t`, vertices `v`, and supported up vector
`u = e_z`, define

```text
m[t] = min_v dot(u, V[t, v]) = min_v V[t, v, 2].
```

The implementation first reduces
`isfinite(V_chunk).all(dim=(vertex, coordinate))` per frame, then reduces the
height minimum on the current device. A chunk of `B` frames and `V` vertices
produces only `B` validity flags and `B` scalar minima for host transfer. Any
invalid frame rejects the motion result; a finite minimum cannot hide an
infinite or NaN non-minimum vertex. Full-motion vertices are neither
concatenated nor persisted.

The summary input is the ordered one-dimensional array
`m = (m[0], ..., m[T-1])`. `T` must be positive and every value must be finite.
An empty series, NaN, or infinity is invalid input. The configured operation
maps such a failure to an explicit `invalid_input` record; the mathematical
core never drops frames silently.

### 2. Exact retained coverage

Let `m_sorted` be `m` in ascending order. For requested retained coverage
`c in (0, 1]`, define

```text
d(c, T) = T - ceil(c*T)
E_c     = m_sorted[d(c, T)].
```

The API receives a Python float but evaluates the integer count through the
exact decimal value `Decimal(str(c))`; binary floating-point multiplication is
not allowed to move a mathematically integral `c*T` just above its boundary.
The stored float and its decimal string are both recorded by the configured
operation.

`d` is the exact number of lowest frames ignored by the candidate. Therefore
at least `ceil(c*T)` frames satisfy `m[t] >= E_c`. Ties can make the empirical
retained fraction larger than requested; they do not change `d`.

The absolute candidate is

```text
E_100 = E_1 = m_sorted[0].
```

The sensitivity of a derivative to the absolute lower envelope is

```text
Delta_c = E_c - E_100 >= 0.
```

`Delta_c` answers only how far the candidate rises after ignoring `d` lowest
frames. It does not say whether those frames are penetration, valid hand/body
contact, flight, or another support level.

The reusable core accepts an explicit ordered tuple of unique coverages and
requires `1.0` to be present. Output preserves caller order; the tuple need not
be sorted. The first AMASS operation uses the compact headline tuple

```text
(100%, 99.9%, 99.5%, 99%).
```

The older bounded artifact's 95% and 90% values may be retained only as
compatibility fields when that artifact is read. They are not headline
candidates and do not become defaults in the new implementation.

### 3. Short sequences and discrete resolution

When `d(c,T)=0`, the requested coverage cannot discard even one native frame.
The result must still contain the requested row, with:

```text
E_c = E_100
Delta_c = 0
discarded_frame_count = 0.
```

This includes E99 for sufficiently short sequences and E99.9 for every
sequence shorter than 1000 frames. No interpolated quantile is used to create
a distinction that finite frames do not support. `T=1` is valid but every
candidate necessarily degenerates to the same value.

### 4. Compact temporal-tail diagnostic

For each candidate, define the ignored-tail mask

```text
b_c[t] = (m[t] < E_c).
```

The summary reports the longest consecutive `True` run in native frames. If a
finite positive frame rate is supplied, it also reports seconds as
`run_frames / fps`.

It also reports the empirical retained fraction
`count(m >= E_c) / T`, which can exceed `c` when values tie at the candidate.

This separates an isolated extreme from a long low interval at negligible
cost. It does not classify either case as good or bad ground evidence. Ties at
`E_c` are not below the candidate and therefore are not part of the run.

### 5. Result invariants

For every requested coverage row:

```text
0 <= d(c,T) < T
E_c >= E_100
Delta_c = E_c - E_100
count(m >= E_c) / T >= c
0 <= longest_below_run_frames <= d(c,T)
```

The last inequality holds because every strictly-below frame must lie among
the `d` discarded order positions, while ties can make the strict-below count
smaller than `d`.

The result is a `mesh_lower_envelope_summary`, not a ground plane or a
`Ground_Resolution_Result`. It cannot emit `single_horizontal_plane` on its
own. That classification belongs to later reviewed analysis.

### 6. Complexity and numerical contract

With `K` requested coverages:

- mesh reduction costs `O(T*V)` arithmetic, peak device memory
  `O(B*V)` for chunk size `B`, and host storage `O(T)`;
- all required order positions are computed by one `numpy.partition` call on a
  float64 copy of the `T` minima, with expected `O(T)` work and `O(T)` working
  storage;
- candidate/run construction costs `O(K*T)`, with the task grid fixing `K=4`;
- no full sort, full-motion mesh, histogram, ECDF, or per-vertex artifact is
  required.

Both float32 and float64 input series are accepted and summarized in float64.
Coverage values are finite Python floats, unique, within `(0,1]`, and must
include `1.0`. Coverage-count arithmetic uses exact decimal values as defined
above; mesh heights and deltas use float64.

Each selected height must equal the full-sort oracle applied to the float64
copy of that same input array. There is no general exact-equality claim between
float32 and float64 arrays that originated from different rounded values. The
cross-dtype smoke uses binary-exact heights and therefore expects exact equality
after conversion; other callers compare against their own dtype's selected
order statistic.

### 7. Core and operation result boundary

The reusable core receives the one-dimensional finite series, an ordered
coverage tuple, and optional finite positive frame rate. Its immutable summary
contains:

- `frame_count` and optional `frame_rate_in_hz`;
- `absolute_minimum_height_in_meter`;
- one row per requested coverage, preserving caller order;
- per row: coverage float and decimal string, discarded-frame count, height in
  metres, delta from absolute minimum in metres, empirical retained fraction,
  longest strict-below run in frames, and optional seconds.

There is no separate short-sequence status string: the exact
`discarded_frame_count == 0` is the unambiguous signal that a requested tail
level could not remove one frame.

The core raises `ValueError` for shape, empty, non-finite, coverage, or frame-rate
violations. The configured operation catches that failure and emits
`status='invalid_input'` plus a sanitized reason. It does not fabricate a
candidate and does not map a numerical failure to the semantic
`Ground_Resolution_Result.unknown`; that later result type belongs to the
method-analysis layer.

## Initial Analysis Surface

The zmin task intentionally avoids a percentile winner. Its compact output is:

- per sequence: `E100`, `E99.9`, `E99.5`, `E99`, each `Delta_c`, each exact
  discarded-frame count, empirical retained fraction, and each longest
  below-tail run in frames/seconds;
- per source prefix: P50/P90/P99/max of each `Delta_c` plus sequence count;
- globally: top-K sequences by `Delta_c`, with time-series plots or visual
  frames generated only for that review set.

The first bounded implementation may emit only the per-sequence record. The
source aggregation and top-K review belong to the deferred method-analysis
task when a cohort is frozen. This keeps the candidate implementation simple
and prevents exploratory plots from becoming an accidental ground rule.

The bounded operation evaluates every native frame when a motion has at most
1200 frames and otherwise the first 1200 consecutive native frames. Every
candidate and run diagnostic uses that identical window and records the full
frame count plus inclusive start/exclusive stop. This is an acceptance window,
not part of the reusable estimator definition and not a claim about the full
motion when truncation occurs.

## Realization Contract And Model Limitation

The configured AMASS baseline starts with SMPL-X only because `hjlib-smpl`
already owns a typed full-pose SMPL-X constructor and forward. It uses:

- the native flat 165D axis-angle pose and translation;
- native gender;
- the locally available `smplx/SMPLX_<GENDER>.npz` model and its supported
  first 10 AMASS betas;
- no DMPL and no coordinate/floor transform;
- model path and file SHA-256 recorded in every run manifest.

After delayed imports, the operation also requires the model root frozen by
`hjlib_smpl.local_setting` to resolve exactly to the explicit CLI model root.
This closes the provenance gap when a caller invokes `main` in a process that
had already imported `hjlib-smpl` before the environment override.

The raw reader preserves all native 16 betas. A read-only model-file probe found
`shapedirs.shape == (10475,3,20)` in all three local `smplx` files and no
`smplx_v1_1` asset directory on this machine, so a truthful native-16 forward is
currently unavailable. The operation must not call the reduced model
“native-16” or silently claim all native shape coefficients were realized.

The earlier nine-motion artifact used the same reduced model and 10 realized
betas but described it loosely as SMPL-X v1.1. The new run is a reproducibility
baseline for that exact model contract and records the model hash plus the
10-of-16 truncation explicitly. A full native-16 model, if later acquired and
registered, is a separate model-contract ablation; its absolute or
millimetre-level zmin values must not be mixed with the reduced-10 result. The
old JSON/PNG remain immutable historical evidence.

SMPL+H lower-envelope mathematics is identical once `m[t]` exists, but
`hjlib-smpl` currently lacks the matching typed SMPL+H flat-pose constructor and
forward adapter. The first operation must not bypass that owner by calling the
third-party model directly. Cross-representation execution is deferred to a
separate `hjlib-smpl` increment if the later comparison needs it.

## Synthetic Acceptance Matrix

The deterministic contamination oracle is:

```text
T = 1000
known support height = 0 m
clean series = 200 frames at 0 m, followed by 800 frames linearly spaced
               from 0.05 m through 0.30 m inclusive
n_bad = (0, 1, 5, 10, 20, 50)
bad height = -0.10 m
replacement targets = non-support frames only, so all 200 support frames remain
placement A = indices 200, ..., 200+n_bad-1, consecutive
placement B = indices 200+floor(i*800/n_bad), i=0,...,n_bad-1, when n_bad>0
both placements are empty when n_bad=0
```

For each coverage, the full-sort oracle supplies the expected selected value;
the analytic support-recovery condition is
`n_bad <= d(c,T) < n_bad + 200`. Placements A and B must have identical order
statistics while their longest low-tail runs differ.

The implementation smoke must cover:

1. discrete boundary lengths `T=(1, 99, 100, 199, 200, 999, 1000)` for E99,
   E99.5, and E99.9 discard counts;
2. the frozen `T=1000` contamination oracle with every `n_bad` value;
3. isolated and consecutive bad-frame placement with identical statistics but
   different longest-run results;
4. ties at the candidate height and empirical retained coverage;
5. exact full-sort-oracle agreement per input dtype, plus exact cross-dtype
   agreement only for binary-exact test heights;
6. one multi-kth partition result against a full-sort oracle;
7. empty, NaN, infinity, duplicate/missing/invalid coverage rejection;
8. identical results for different mesh-forward chunk boundaries.

## Code Architecture

### Reusable solver core

Create one managed-noun module:

```text
src/hjlib_ground_solver/estimate_ground/by_mesh_lower_envelope.py
```

It contains two frozen slotted dataclasses and two stateless functions:

```python
@dataclass(frozen=True, slots=True)
class Mesh_Lower_Envelope_Candidate:
    retained_coverage: float
    retained_coverage_decimal: str
    discarded_frame_count: int
    height_in_meter: float
    delta_from_absolute_minimum_in_meter: float
    empirical_retained_fraction: float
    longest_below_run_frame_count: int
    longest_below_run_duration_in_second: float | None


@dataclass(frozen=True, slots=True)
class Mesh_Lower_Envelope_Summary:
    frame_count: int
    frame_rate_in_hz: float | None
    absolute_minimum_height_in_meter: float
    candidates: tuple[Mesh_Lower_Envelope_Candidate, ...]


def compute_per_frame_mesh_minimum_height(
    vertices: torch.Tensor,
    up_axis_index: int,
) -> torch.Tensor:
    ...


def summarize_mesh_lower_envelope(
    per_frame_minimum_height_in_meter: NDArray[np.floating[Any]],
    retained_coverages: tuple[float, ...],
    frame_rate_in_hz: float | None = None,
) -> Mesh_Lower_Envelope_Summary:
    ...
```

`compute_per_frame_mesh_minimum_height` accepts one `(B,V,3)` floating tensor,
requires positive `B/V`, validates `up_axis_index in {0,1,2}`, rejects any
non-finite coordinate, and returns a `(B,)` tensor on the same device and dtype.
It performs no detach, CPU copy, model forward, or state mutation.

`summarize_mesh_lower_envelope` owns shape/value/coverage/frame-rate
validation, decimal retained-frame counts, one multi-kth partition, row
construction, and temporal runs. It never mutates caller input. Its only state
is local immutable output; an accumulator class would add lifecycle complexity
without reducing the required `O(T)` series storage.

The two classes and two functions are public reusable lower-envelope symbols.
They are re-exported by `hjlib_ground_solver.estimate_ground` and the package
root, matching the family top-level public-API rule. Their names make the
geometric limitation explicit; no API is named `estimate_ground_from_zmin`.

### Configured AMASS operation

Create:

```text
hjlib-dataset-raw/test/amass_ground_zmin_family.py
```

This is an explicit campaign operation, not a pytest file, data-master entry,
or raw public API. It uses a flat Typer command with required `--model-root`,
`--ground-solver-repository`, `--smpl-repository`, and `--output-directory`,
plus `--device` and positive `--chunk-size` controls. The output directory is
one no-clobber publication unit rather than two independently visible files.

The operation has these responsibilities:

1. load the reviewed raw AMASS test configuration and the frozen nine motion
   identities;
2. let `main` validate repository/model paths, set `HJLIB_SMPL_MODELS_PATH`,
   and only then call `run_operation`, whose body dynamically imports
   `reader_amass`, `hjlib-smpl`, and `hjlib-ground-solver`; this preserves
   data-free module import and `--help` even though `reader_amass` imports the
   full raw package;
3. load native fields, freeze the bounded window, and hash native/model files;
4. cache one reduced-10-beta `smplx` model per gender on the requested device;
5. build typed `SMPLX_Param_Seq` chunks with `hjlib-smpl`, move their tensor
   fields to the device, forward under `torch.inference_mode`, reduce on device,
   and retain only the concatenated float64 `(T,)` minima;
6. call the top-level solver summary and serialize dataclasses into one compact
   versioned JSON artifact;
7. draw one bounded nine-panel time-series confirmation PNG with existing
   OpenCV: `m[t]` plus the four candidate lines, no mesh or plane semantics;
8. stage JSON, PNG, and receipt in one sibling temporary directory, reopen and
   validate them, then publish the complete directory with one atomic rename;
9. refuse overwrite or output inside either AMASS root, and verify both raw
   root metadata manifests are unchanged.

The configured campaign run has one writer for its new output path. Atomic
rename prevents a partial package from becoming visible; the pre-rename
existence check is a single-writer no-clobber contract, not a general
multi-process lock or Linux-specific `renameat2(RENAME_NOREPLACE)` API.

The explicit `--model-root` is intentional. The raw test runtime passes its
managed checker, while `hjlib-smpl` currently has no managed descriptor on this
machine. The task therefore does not read or modify an unverified shared
`local_setting.py`; it supplies the existing environment override before the
first `hjlib-smpl` import.

The operation may compose installed sibling packages without adding a
production `hjlib-dataset-raw -> hjlib-ground-solver` dependency. This is a
documented campaign-only workspace exception: it validates that imported
module files reside under the two explicitly supplied repository roots and
records each repository HEAD, dirty-state digest, source-file hashes, and
installed package versions. It fails closed on a path mismatch. If this
composition becomes routine rather than campaign acceptance, it moves to
`hjlib-integration-tests` instead of becoming a reader method.

### Function boundaries in the operation

Keep orchestration explicit and each function below roughly 100 lines:

- `load_native_motion_window`: raw projection and native-contract validation;
- `move_smplx_param_seq_to_device`: return a new typed parameter with moved
  tensor fields; never mutate the input object;
- `analyze_motion`: chunked forward/reduction and summary construction;
- `summary_to_json`: dataclass-to-JSON projection with identity/provenance;
- `render_confirmation_figure`: OpenCV drawing from `(T,)` and summary only;
- `publish_artifact_directory`: staged, reopened, no-overwrite package
  publication;
- `run_operation`: dynamic imports, model cache, nine-motion loop, manifests,
  timing, and receipt construction;
- `main`: Typer parsing, path resolution, pre-import environment setup, and
  sanitized operation-error mapping only.

These are stateless ordinary functions. Model cache and accumulated records are
operation-local variables in `run_operation`; no new manager class or hidden
module cache is introduced.

## Smoke-Test Standard

Create `test_smoke/test_mesh_lower_envelope.py`. It exposes pytest `test_*`
cases and `smoke_test_mesh_lower_envelope()`, then joins
`test_smoke/test_all_func.py`.

The smoke implements the Mathematical Architecture matrix with synthetic NumPy
and torch inputs. In addition to numeric oracles, it asserts:

- the reducer preserves device/dtype and does not modify vertices;
- summary does not modify the input series;
- all output dataclasses are immutable;
- candidate order matches caller coverage order;
- invalid tensor shape/dtype/axis/non-finite coordinate and invalid series/
  coverage/frame rate raise `ValueError`;
- chunk sizes `(1, 7, T)` reproduce the same ordered minima and summary.

The configured AMASS operation is not collected by pytest or included in the
data master because nine body-model forwards are an explicit campaign cost. A
data-free raw smoke imports the operation, runs `--help` in a subprocess, and
asserts that no `hjlib_*` package was imported before model configuration. A
successful configured invocation must publish JSON/PNG/receipt, reopen them,
record peak device memory and elapsed time, and prove both AMASS metadata
manifests unchanged.

## Implementation And Acceptance Evidence

The reviewed implementation is:

- `estimate_ground/by_mesh_lower_envelope.py`: immutable candidate/summary
  results, device-preserving mesh reduction, and exact Decimal-count coverage
  summary;
- `test_smoke/test_mesh_lower_envelope.py`: 8 synthetic public-contract tests;
- `hjlib-dataset-raw/test/amass_ground_zmin_family.py`: campaign-only delayed-
  import orchestration and atomic artifact publication;
- `hjlib-dataset-raw/test_smoke/test_amass_ground_zmin_family.py`: data-free
  import/help and output-boundary smoke.

The authoritative configured package is
[`bounded_reduced10_v2`](../../../../../hjlib-dataset-raw/campaigns/01_amass_raw_support/task_ground_zmin_family/artifacts/bounded_reduced10_v2/).
It observed all nine motions, used no more
than 307,012,608 peak allocated GPU bytes, preserved both AMASS metadata
manifests, and verified source/repository state before and after analysis. Its
3x3 time-series PNG was inspected; it is a lower-envelope confirmation, not a
ground-plane visualization.

Final checks:

- `hjlib-ground-solver`: 24 smoke tests, master smoke, and strict pyright pass;
- `hjlib-dataset-raw`: 32 smoke tests, master smoke, and strict pyright pass;
- independent implementation review and two-dimension HJ review found no
  remaining Critical issue. The campaign-only single-writer publication
  assumption is documented rather than generalized into a cross-platform lock.

## Documentation And Public Boundary

Implementation updates:

- `docs/usage/README.md`: add the two lower-envelope entries, shapes, exact
  coverage meaning, and the non-ground caveat;
- `docs/design/README.md`: update layout/state and link this frozen residence;
- `docs/design/test.md`: add the synthetic smoke and configured-operation
  boundary;
- package `__init__.py` files: public re-exports;
- `hjlib-dataset-raw/docs/design/test.md` and zmin task ledger: operation and
  artifact receipt.

`pyproject.toml` descriptions are checked in both repositories. The
ground-solver description should mention mesh lower-envelope candidates because
the behavior is public; raw's description remains unchanged because no reader
surface is added. No dependency pin changes are expected.

## Modification History

- 2026-08-03: Landed Requirements and Mathematical Architecture. Defined the
  exact lower-envelope/retained-coverage family, compact delta/run output,
  chunked memory bound, and explicit SMPL-X realization contract. Awaiting
  independent mathematical review before code architecture or implementation.
- 2026-08-03: Independent Mathematical Architecture review found two Critical
  and four Concern items. Froze decimal coverage-count arithmetic and the
  bounded 1200-frame window, added pre-minimum vertex finiteness, made the core
  versus operation failure/result boundary explicit, and completed the
  deterministic and discrete-boundary smoke oracle. No mathematical findings
  remain open.
- 2026-08-03: Landed Code Architecture for the public stateless solver core,
  synthetic smoke, and raw task-owned configured operation. Awaiting
  independent code-architecture review before implementation.
- 2026-08-03: A read-only local model probe found no full `smplx_v1_1` asset and
  confirmed the available files support the existing reduced-10 realization.
  Corrected the first run to a hash-pinned 10-of-native-16 reproducibility
  baseline and deferred native-16 effects to a separate asset-backed ablation.
- 2026-08-03: Independent Code Architecture review found no Critical items and
  seven Concerns. Addressed import ordering, thin-CLI orchestration, immutable
  device transfer, atomic directory publication, campaign-only cross-repo
  provenance, data-free operation smoke coverage, and cost/field naming.
- 2026-08-03: Implemented and verified the core and bounded operation.
  Independent implementation/HJ review found no Critical owner or algorithm
  issue. Closed model-root caching, before/after source-state, dependency-pin,
  public-field, metadata, alias-style, and ledger findings; accepted the v2
  nine-motion artifact without choosing a percentile.
