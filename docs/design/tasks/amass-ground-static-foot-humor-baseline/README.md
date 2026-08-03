# AMASS Ground Static-Foot HuMoR Baseline

## Status

- Layered Design state: Requirements, Mathematical Architecture, and Code
  Architecture reviewed; solver implementation active
- Campaign ledger:
  [`hjlib-dataset-raw/campaigns/01_amass_raw_support/task_ground_static_foot_humor_baseline/`](../../../../../hjlib-dataset-raw/campaigns/01_amass_raw_support/task_ground_static_foot_humor_baseline/)
- Implementation state: `hjlib-smpl` leaf committed; reusable solver core and
  synthetic oracle smoke implemented; raw configured operation pending

## Requirements

### 1. Purpose and owner boundary

Implement a deliberately faithful HuMoR-style flat-floor baseline for
comparison with the completed full-mesh zmin family. The reusable estimator
belongs to `hjlib-ground-solver`. `hjlib-dataset-raw` later owns only frozen
AMASS motion identities, faithful body/joint realization, bounded execution,
visual evidence, and provenance receipts.

The baseline is not AMASS publisher ground truth. It may return a usable flat
floor candidate or flag terrain interaction, but it cannot establish scene
geometry, slopes, stairs, platforms, or arbitrary supporting objects.

### 2. Frozen upstream source

Behavior is pinned to HuMoR commit
[`fc6ef84f0baa153be15427402e0147ed1a63a11a`](https://github.com/davrempe/humor/blob/fc6ef84f0baa153be15427402e0147ed1a63a11a/humor/scripts/process_amass_data.py),
specifically `determine_floor_height_and_contacts`. The floor/terrain subset,
not HuMoR's contact-label output or complete AMASS preprocessing pipeline, is
the comparator owned by this task.

The exact upstream constants are:

| Name | Value | Upstream meaning |
| --- | ---: | --- |
| `FLOOR_VEL_THRESH` | `0.005` | permissive candidate gate with a strict `<` comparison, in metres per native frame |
| DBSCAN `eps` | `0.005` | one-dimensional static-toe height radius in metres |
| DBSCAN `min_samples` | `3` | upstream density threshold |
| `FLOOR_HEIGHT_OFFSET` | `0.01 m` | subtract from the selected toe-joint height |
| `TERRAIN_HEIGHT_THRESH` | `0.04 m` | higher static-toe mode gate |
| `ROOT_HEIGHT_THRESH` | `0.04 m` | corresponding higher root-mode gate |
| `CLUSTER_SIZE_THRESH` | `0.25` | multiplier of FPS, not sequence fraction |

### 3. Exact baseline behavior

The reusable core consumes finite `+Z`-up root, left-toe, and right-toe joint
tracks in metres at the sequence's native temporal resolution plus a positive
FPS. It requires at least two frames because the upstream displacement
construction appends the last displacement.

This is a height-cluster estimator, not a per-frame motion estimator. Adjacent
toe displacement is only a deliberately permissive eligibility gate that
removes grossly moving samples. Whether a stable support-height mode exists,
and which height wins, is decided by clustering the eligible toe heights. The
exact comparator must preserve that ordering of responsibility rather than
describing the displacement test as contact inference.

It must preserve this behavior:

1. toe velocity is Euclidean displacement between adjacent frames, with the
   last displacement repeated; it is not divided by frame duration;
2. a toe sample is static only when displacement is strictly below `0.005`;
3. left static heights and indices are concatenated before right static values;
4. when no static toe sample exists, both upstream floor values are `0.0` and
   terrain rejection is false;
5. otherwise all static heights are clustered with one-dimensional
   `sklearn.cluster.DBSCAN(eps=0.005, min_samples=3)`; only those two arguments
   are explicit, matching upstream, while remaining constructor behavior comes
   from the recorded scikit-learn version;
6. every label returned by `np.unique`, including DBSCAN noise label `-1`, is
   treated as one cluster; this known defect is preserved;
7. labels are visited in ascending `np.unique` order; the selected cluster is
   updated only for a strictly lower toe-height median, so equal medians retain
   the first label, and its root median uses unique native frame indices;
8. the toe-joint floor is that median; the returned HuMoR floor candidate is
   `toe_joint_floor - 0.01 m`;
9. terrain interaction is true when any cluster simultaneously has root median
   strictly above the selected root median by `0.04 m`, toe median strictly
   above the selected toe median by `0.04 m`, and sample count strictly greater
   than `int(0.25 * fps)`.

The result must expose the unoffset toe-joint floor, offset HuMoR candidate,
terrain flag, cluster medians/sizes/labels, static sample/frame evidence, and
the exact pinned configuration. Extra evidence fields are allowed only when
they do not change baseline selection or rejection.

### 4. Deliberately preserved limitations

- The velocity threshold is frame-rate dependent because it is metres per
  native frame, despite the variable being named as velocity. Upstream calls
  the estimator before its later 30 FPS output downsampling, so exact parity
  must not silently resample first or reinterpret the threshold as metres per
  second.
- DBSCAN noise is pooled into one ordinary candidate cluster.
- The lowest height mode wins even when it is a small or semantically invalid
  support mode.
- No-static evidence silently falls back to zero in upstream behavior.
- Terrain cluster size is compared with `int(0.25 * fps)`, not motion length.
- A terrain flag discards a sequence in HuMoR preprocessing; it does not
  describe multiple planes or recover terrain geometry.
- The fixed one-centimetre toe-joint offset is body-model-specific heuristic
  compensation, not measured sole thickness.

These limitations belong to this exact comparator. Repairs, robust statistics,
explicit unknown states, contact probability, adaptive thresholds, or corrected
noise semantics belong to the separate `static_foot_robust` task.

### 5. Validation and visual acceptance

Synthetic smoke must provide an independent literal reference oracle and cover
the zero-static fallback, strict threshold equalities, noise-label pooling,
lowest-mode selection, duplicate left/right frame indices, terrain conjunction,
FPS-dependent size boundary, input validation, immutability, and public
re-exports.

The later AMASS operation must publish compact JSON plus a bounded visual review
surface. Each panel must show native-time left/right toe heights, static sample
selection, one-dimensional cluster membership, selected toe-joint and offset
floor lines, and terrain-rejection evidence. It must not draw or label the
candidate as publisher ground truth. Cohort identity and body-model realization
must be frozen before execution; no per-motion parameter tuning is allowed.
The receipt must record the scikit-learn version as part of exact-comparator
provenance.

### 6. Questions delegated to Mathematical Architecture

The Mathematical Architecture below freezes the named-track input, immutable
evidence semantics, numerical gates and wrapper statuses, faithful SMPL-H
realization, bounded cohort, and visual artifact. Code placement and function
decomposition remain for Code Architecture after mathematical review.

## Mathematical Architecture

### 1. Input and estimator priority

The reusable estimator receives three explicitly named NumPy tracks:

```text
root_position_in_meter       r in R^(T x 3)
left_toe_position_in_meter   l in R^(T x 3)
right_toe_position_in_meter  q in R^(T x 3)
frame_rate_in_hz             f > 0
```

All tracks have the same `T >= 2`, dtype exactly `numpy.float32` or
`numpy.float64`, and shape `(T, 3)`. Every coordinate and every derived
displacement must be finite. Coordinates use metres and index `2` is the
supported `+Z` axis. The solver accepts named tracks rather than a joint tensor
or HuMoR joint indices; body-model vocabulary and extraction remain outside
the reusable core.

For toe side `s` and native frame `t`, the baseline first forms an eligibility
mask

```text
d_s[t] = ||p_s[t + 1] - p_s[t]||_2
eligible_s[t] = d_s[t] < 0.005 m/native-frame
```

with the final displacement repeated from the preceding pair. This mask is a
wide motion prefilter, not the ground decision and not a contact label. The
estimator then pools `z_s[t]` only where `eligible_s[t]` is true and applies the
frozen one-dimensional DBSCAN. The selected ground evidence is the lowest
cluster median under the exact upstream label-order rule. Cluster membership,
cluster median, and cluster-level terrain evidence therefore have semantic
priority over the magnitude of per-frame displacement once a sample passes the
gate.

The exact `humor_baseline` keeps `0.005 m/native-frame` for parity. A later
robust sibling may make the gate explicitly loose and FPS-normalized, but its
support-height decision must still be cluster-led; that change is outside this
task.

### 2. Pooled height evidence and exact DBSCAN semantics

Let `E_l` and `E_q` be native frame indices passing the two eligibility masks.
Construct the pooled sample arrays in this exact order:

```text
x = concat(l[E_l, 2], q[E_q, 2])
i = concat(E_l, E_q)
s = concat('left' repeated |E_l|, 'right' repeated |E_q|)
```

If `x` is nonempty, fit
`DBSCAN(eps=0.005, min_samples=3)` to `x.reshape(-1, 1)`, specifying no other
constructor arguments. The installed scikit-learn version is execution
provenance, not an implicit method change.

Let `a` be the returned label per pooled sample and let `K=unique(a)` in
ascending NumPy order. Exact HuMoR behavior treats every `k in K`, including
noise label `-1`, as a candidate cluster. For each such label define:

```text
P_k = {j : a[j] = k}                 pooled sample positions
F_k = unique(i[P_k])                 native frame indices
h_k = median(x[P_k])                 toe-height median
r_k = median(r[F_k, 2])              root-height median
n_k = |P_k|                          left/right samples, not unique frames
```

The immutable evidence preserves pooled order, side, native frame index,
height, DBSCAN label, and per-label `(h_k, r_k, n_k, F_k)`. Thus a frame where
both toes pass contributes two height samples but only one root sample to that
label's root median, exactly as upstream.

“Cluster-led” here describes the upstream computation, not a claim that all
its candidates are valid density clusters. Pooling all DBSCAN noise under
label `-1` is precisely a defect of the exact comparator. The robust sibling
must require genuine non-noise cluster evidence rather than inherit that bug.

### 3. Candidate selection, offset, and terrain rejection

Visit `K` in ascending order and update the selected label only when its
`h_k` is strictly lower than the current minimum. Equivalently, the selected
label `k*` is the first ascending label attaining the smallest median. Define:

```text
toe_joint_floor = h_k*
humor_floor_candidate = h_k* - 0.01 m
terrain_k =
    (r_k > r_k* + 0.04 m)
    and (h_k > h_k* + 0.04 m)
    and (n_k > floor(0.25 f))
terrain_interaction = any_k terrain_k
```

The FPS affects only the terrain sample-count boundary. It never normalizes
toe displacement. All three terrain comparisons are strict; equality does not
trigger rejection.

If `x` is empty, exact upstream numerical behavior is:

```text
toe_joint_floor = 0.0
humor_floor_candidate = 0.0
terrain_interaction = false
```

No cluster is fabricated and the offset is not subtracted in this branch.

### 4. Wrapper status and acceptance boundary

The exact numeric outputs above are always retained. The safer wrapper status
is one of:

```text
upstream_candidate
upstream_zero_fallback
upstream_terrain_rejection
```

Status precedence is `upstream_zero_fallback` when `x` is empty, otherwise
`upstream_terrain_rejection` when the upstream discard flag is true, otherwise
`upstream_candidate`. Only `upstream_candidate` exposes a non-`None`
`accepted_candidate_height_in_meter`; the exact upstream candidate remains
available separately in every branch. This does not repair or reinterpret the
baseline. It prevents a zero fallback or a candidate that upstream would
discard from being consumed accidentally as accepted evidence.

`upstream_candidate` intentionally avoids the stronger word `estimated`: pure
DBSCAN noise can win because the pinned code pools label `-1`. The status means
only that upstream produced a non-rejected comparison candidate.

These statuses are method-local diagnostics. They do not emit the campaign's
later `single_horizontal_plane`, `multilevel_or_piecewise`, or `unknown`
semantic result.

### 5. Numerical contract and invariants

Validation copies inputs without changing their floating dtype. All three
tracks must share that dtype so implicit promotion cannot alter strict
boundaries. Float32 and float64 are supported and compared with a literal
oracle operating on the same dtype; no cross-dtype bit-equality claim is made.
Shape, dtype, empty/one-frame input, non-finite input, bool/non-numeric FPS,
non-positive/non-finite FPS, or non-finite derived displacement/median raises
`ValueError`. Input arrays are never mutated.

For nonempty evidence:

```text
sum_k n_k = |x|
union_k P_k = {0, ..., |x|-1}
toe_joint_floor = min_k h_k
humor_floor_candidate = toe_joint_floor - 0.01 m
```

In exact real arithmetic, a common translation of all three tracks leaves
eligibility, labels, cluster sizes, and terrain rejection unchanged. For a
`+Z` translation `c`, all nonempty cluster/root medians and both floor heights
shift by `c`. Horizontal rigid translation or rotation around `+Z` likewise
leaves the result invariant. Under float32/float64 these properties are tested
only with exactly representable, boundary-safe fixtures: rounding after a
transform may legitimately move a strict `0.005` or DBSCAN `eps=0.005`
comparison across its boundary. Transformed arrays and every derived value
must remain valid and finite. The upstream zero-fallback branch deliberately
violates vertical equivariance because it remains fixed at zero.

Time order matters only through adjacent-displacement eligibility and the
repeated final displacement. After eligibility, clustering itself is over the
pooled one-dimensional height samples and has no temporal-persistence rule.
This limitation is exposed rather than repaired.

### 6. Faithful AMASS realization

The authoritative acceptance surface uses the downloaded native SMPL+H tree,
not the existing reduced-10 SMPL-X zmin realization. For each motion it
reproduces the HuMoR-side model inputs:

- native 156-value pose layout: root `[:3]`, body `[3:66]`, and two hands
  `[66:156]`;
- gendered official SMPL-H model, first 16 native betas, no DMPL;
- `use_pca=False`, flat hand mean, and float32 forward inputs;
- the first 22 SMPL joints, with root index `0`, left toe index `10`, and right
  toe index `11`;
- no coordinate transform and `+Z` as the analysis axis.

The local official SMPL-H archives contain exactly 16 shape directions. The
current generic `hjlib-smpl.get_official_smplh` exposes only 10 betas and
interprets 45 hand values as PCA coefficients, so it is not faithful for this
operation. Code Architecture must place the already-verified HuMoR-compatible
16-beta/no-PCA construction in the `hjlib-smpl` owner rather than disguise a
reduced model as parity. HuMoR's construction pads unused shape directions
only to bypass the upstream `smplx` ten-beta guard; the first 16 model
directions remain the actual calculation.

For parity with HuMoR AMASS preprocessing, the operation analyzes the middle
80 percent of native frames:

```text
start = floor(0.1 T_native)
stop  = floor(0.9 T_native)
frames = [start, stop)
```

The estimator runs on this native-rate window before any 30 FPS output
downsampling. The operation does not generate HuMoR's contacts, velocities,
aligned coordinates, or normalized dataset files.

The frozen cohort includes GRAB, which is not in this HuMoR commit's declared
`ALL_DATASETS` list and uses a newer native field spelling. Its panel is an
explicit extension of the pinned floor/terrain estimator to another AMASS
motion, not evidence that the complete upstream preprocessing pipeline handled
GRAB. The receipt distinguishes source-method parity from corpus coverage.

SMPL-X is deliberately excluded from this task's numeric acceptance artifact.
The paired local trees agree on frame count and gender for the frozen cohort,
but several paired archives disagree on FPS metadata. Mixing them here would
confound method behavior, realization, and timing metadata. The later method
analysis must rerun zmin/static-foot candidates on one common frozen
realization rather than compare these SMPL-H numbers directly against the old
reduced-10 SMPL-X acceptance artifact.

### 7. Bounded cohort and visual evidence

The bounded acceptance cohort reuses the nine semantic motions selected for
zmin review, but freezes their SMPL-H identities explicitly:

```text
ACCAD/Male2MartialArtsStances_c3d/D1 - stand to ready_poses
ACCAD/Female1General_c3d/A10 - lie to crouch_poses
DFaust_67/50021/50021_running_on_spot_poses
DFaust_67/50022/50022_knees_poses
CMU/61/61_01_poses
KIT/572/stomp_left01_poses
EKUT/125/SLP101_poses
SFU/0005/0005_2FeetJump001_poses
GRAB/s1/airplane_fly_1_stageii
```

All nine pairs were verified locally to match the existing SMPL-X cohort in
frame count and gender; exact relative names, native file hashes, model hashes,
frame window, native FPS, dtype, beta policy, scikit-learn version, and source
hashes belong in the receipt.

The confirmation image is a bounded `3 x 3` diagnostic. Each panel overlays
native-time left/right toe heights, distinguishes samples rejected or retained
by the permissive displacement gate, colors the retained samples by exact
DBSCAN label, draws the selected toe-joint and offset heights, and names every
terrain-triggering cluster. Noise label `-1` is visibly distinct even though
the exact selector treats it as a cluster. Text includes FPS, window/counts,
status, selected label, medians, and terrain flag. The image labels its line
as a HuMoR candidate, never publisher ground truth.

### 8. Complexity and smoke-test standard

For `T` frames, joint-track storage and all non-DBSCAN work are `O(T)`.
One-dimensional DBSCAN owns its library-dependent time/memory behavior; the
operation keeps only three joint tracks and compact evidence, never a
full-motion vertex tensor. SMPL-H forward is chunked, while the three joint
tracks may be concatenated because they require only `9T` floating values.

Synthetic smoke uses an independent literal implementation of the pinned
source and covers:

1. strict-below gate behavior, equality at `0.005`, and repeated final
   displacement;
2. unchanged cluster selection under different sub-threshold displacement
   magnitudes, proving the gate does not rank accepted samples;
3. left-before-right pooling, duplicate frame/root handling, ascending label
   order, equal-median ties, and lowest-median selection;
4. a real dense cluster, pure noise pooled under `-1`, and mixed noise/cluster
   evidence;
5. the exact empty-evidence double-zero fallback and safe wrapper status;
6. every strict terrain equality boundary, the three-way conjunction, and
   `floor(0.25 f)` sample-count behavior at multiple native FPS values;
7. float32/float64 same-dtype oracle parity, input immutability, derived
   overflow, and all validation failures;
8. exact-representable, boundary-safe vertical/horizontal transformation
   invariants and the documented zero-fallback/finite-precision exceptions;
9. public/subpackage re-export identity and frozen immutable result evidence.

## Code Architecture

### 1. Cross-repository residence and dependency direction

The task has one design residence here but three code owners:

```text
hjlib-smpl
    official SMPL-H loader options + 156-pose builder + typed joint forward
        -> hjlib-dataset-raw configured operation
            -> hjlib-ground-solver named-track estimator
```

`hjlib-smpl` owns model-file interpretation and body forward.
`hjlib-ground-solver` owns only the reusable estimator after the three joint
tracks exist. `hjlib-dataset-raw/test/` owns the configured AMASS operation,
frozen identities, visual evidence, and receipt. Production raw code does not
import solver or add a derived `ground` field, and solver does not import AMASS
or a body-model joint vocabulary.

Implementation and commits proceed leaf first: land/check/commit `hjlib-smpl`,
then update both direct consumers' `hjlib-smpl` commit pins. The solver pin
lands with the solver core; the raw pin lands with the configured operation.
The raw test operation uses delayed solver imports and does not add a new
production `hjlib-ground-solver` dependency.

### 2. `hjlib-smpl` public extensions

Extend the existing `get_official_smplh` entry in
`src/hjlib_smpl/smpl_lib.py` with keyword options while preserving current
defaults:

```python
get_official_smplh(
    gender='neutral',
    num_betas=10,
    use_pca=True,
    num_pca_comps=45,
    flat_hand_mean=True,
) -> smplx.SMPLH
```

When `num_betas` exceeds the third-party ten-beta guard but does not exceed the
actual model-file shape dimension, the loader passes a trusted `data_struct`
whose unused tail is zero-padded to `SMPL.SHAPE_SPACE_DIM`. It rejects requests
beyond the native shape dimension; padded zeros can never be selected as real
betas. The existing no-argument/default behavior remains 10-beta PCA-45.

Add the missing SMPL-H array/typed-forward layer beside the current SMPL and
SMPL-X entries:

```python
split_pose_smplh(poses_flat) -> dict[str, ARRAY_F]
build_smplh_param_seq(...components...) -> SMPLH_Param_Seq
build_smplh_param_seq_from_flat(...) -> SMPLH_Param_Seq
forward_smplh_joints_from_param(model, param) -> torch.Tensor
```

The flat builder freezes the 156 layout and converts native NumPy arrays to
contiguous float32, matching existing builders. The forward entry supports the
existing Single/Batch/Seq variants, verifies gender/beta/hand dimensions,
expands time-invariant sequence betas without mutating the parameter object,
and returns the model's complete joint tensor on the current device. The raw
operation slices indices `0`, `10`, and `11`; the SMPL library does not import
HuMoR names or hard-code this task's three-track selection.

Update top-level re-exports, pure builder smoke, loader/forward usage docs, and
a real-model data test for 16-beta/no-PCA joint forward. The real-model gate
verifies model `num_betas == 16`, hand dimension `45`, finite `(T,J,3)` joints,
and no input mutation. It is not placed in data-free smoke.

### 3. Solver module and immutable records

Add
`src/hjlib_ground_solver/estimate_ground/by_static_foot_humor.py` with one
public estimator:

```python
estimate_static_foot_humor_baseline(
    root_position_in_meter,
    left_toe_position_in_meter,
    right_toe_position_in_meter,
    frame_rate_in_hz,
) -> Static_Foot_HuMoR_Result
```

The exact config is internal and non-tunable at the call site. The public
frozen/slots records are:

```python
type Static_Foot_HuMoR_Status = Literal[
    'upstream_candidate',
    'upstream_zero_fallback',
    'upstream_terrain_rejection',
]

@dataclass(frozen=True, slots=True)
class Static_Foot_HuMoR_Config:
    upstream_commit: str
    displacement_threshold_in_meter_per_native_frame: float
    dbscan_epsilon_in_meter: float
    dbscan_minimum_sample_count: int
    toe_joint_offset_in_meter: float
    terrain_toe_height_threshold_in_meter: float
    terrain_root_height_threshold_in_meter: float
    terrain_sample_count_fps_multiplier: float

@dataclass(frozen=True, slots=True)
class Static_Foot_HuMoR_Sample:
    side: Literal['left', 'right']
    native_frame_index: int
    height_in_meter: float
    dbscan_label: int

@dataclass(frozen=True, slots=True)
class Static_Foot_HuMoR_Cluster:
    dbscan_label: int
    native_frame_indices: tuple[int, ...]
    sample_count: int
    toe_height_median_in_meter: float
    root_height_median_in_meter: float
    is_selected: bool
    triggers_terrain_rejection: bool

@dataclass(frozen=True, slots=True)
class Static_Foot_HuMoR_Result:
    status: Static_Foot_HuMoR_Status
    config: Static_Foot_HuMoR_Config
    input_dtype: str
    frame_count: int
    frame_rate_in_hz: float
    terrain_minimum_exclusive_sample_count: int
    left_toe_displacement_in_meter: tuple[float, ...]
    right_toe_displacement_in_meter: tuple[float, ...]
    samples: tuple[Static_Foot_HuMoR_Sample, ...]
    clusters: tuple[Static_Foot_HuMoR_Cluster, ...]
    selected_dbscan_label: int | None
    toe_joint_floor_height_in_meter: float
    upstream_floor_candidate_height_in_meter: float
    accepted_candidate_height_in_meter: float | None
    terrain_interaction: bool
```

`HuMoR` is kept as the official proper-name spelling in class names; it is one
token, not a newly introduced family acronym or a split `Hu_Mo_R` hierarchy.

Result tuples preserve exact pooled/label order, remain `O(T)`, serialize with
`dataclasses.asdict`, and contain everything the raw plot needs. Eligible
left/right indices are reconstructed from labeled `samples`; cluster pooled
positions are reconstructed from sample labels, so neither duplicate tuple is
stored. Cluster `native_frame_indices` remains explicit because it is the
deduplicated root-median evidence. The terrain count field is `floor(0.25*f)`;
“exclusive” records that a cluster must have strictly more samples.

### 4. Solver function decomposition and smoke

Keep stateless work in ordinary module functions named for the managed noun:

1. `normalize_static_foot_humor_tracks` validates and copies the three same-
   dtype tracks plus FPS;
2. `compute_repeated_toe_displacement` implements adjacent Euclidean
   displacement and final repetition without time normalization;
3. `build_static_foot_humor_samples` creates the exact left-then-right pooled
   evidence and DBSCAN labels;
4. `summarize_static_foot_humor_clusters` computes ordered medians, unique
   root-frame evidence, selection, and terrain conjunction;
5. `estimate_static_foot_humor_baseline` owns orchestration, zero fallback,
   wrapper status, and immutable result assembly.

Only the records/status/public estimator are re-exported from
`estimate_ground/__init__.py` and the package root. Helpers are module-local by
export policy but do not use leading underscores, which in this family denote
state-risky external calls rather than generic internal helpers.

Add `test_smoke/test_static_foot_humor.py` and register it in the solver master.
The independent literal oracle and case matrix are exactly Mathematical
Architecture section 8. Tests also compare public/subpackage symbol identity
and use the installed scikit-learn version on both oracle and implementation.

### 5. Configured AMASS operation

Add
`hjlib-dataset-raw/test/amass_ground_static_foot_humor_baseline.py`. Its module
top level imports no `hjlib-*` package. The Typer CLI exposes only operation
parameters: explicit model root, sibling repository roots, new output
directory, device, and chunk size. It exposes no estimator threshold. The
execution procedure runs the managed local-setting checker before invoking the
CLI; `main` then sets `HJLIB_SMPL_MODELS_PATH` before delayed imports and loads
the already-validated configured SMPL-H root. The operation does not shell out
to `hjlib-codex` or own local-setting policy.

Add an operation-scoped SMPL-H config loader beside `test/reader_amass.py` that
validates only `PATH_AMASS_SMPLH_ROOT` and returns a concrete frozen path
record. Do not reuse `load_amass_test_config`, because that older aggregate
also validates the inactive SMPL-X root and four unrelated representative
motions. Data-free smoke uses an isolated placeholder runtime and proves that
this SMPL-H-only branch never resolves or validates SMPL-X settings.

The operation defines the nine SMPL-H names as one frozen tuple and uses these
functions:

- `load_native_smplh_window`: selective raw fields, schema/gender/FPS/shape
  validation, exact middle-80-percent slicing, and native file hash;
- `move_smplh_param_seq_to_device`: immutable `dataclasses.replace` transfer;
- `load_humor_compatible_smplh_model`: gender cache plus 16-beta/no-PCA model
  and file validation;
- `realize_humor_joint_tracks`: chunked flat builder/typed joint forward and
  CPU transfer of only root/left-toe/right-toe tracks;
- `analyze_configured_motions`: per-motion estimator call and bounded invalid-
  numerical-input handling;
- `verify_operation_invariants`: repository/source/data/model state before and
  after execution;
- `build_operation_payloads` and `render_confirmation_figure`: JSON-safe
  records and the frozen `3 x 3` review surface;
- `execute_operation`: delayed imports, orchestration, and atomic publication;
- `run_operation`: sanitized CLI error boundary only.

Model/repository/source/configuration/publication failures abort the operation.
A native motion with invalid numerical estimator input becomes an explicit
panel/record and does not fabricate a candidate; remaining motions continue.
No full-motion vertices or joints are persisted, and native AMASS arrays are
never changed.

### 6. Artifact and provenance contract

Publish by exclusive no-clobber temporary-directory rename:

```text
analysis.json
confirmation.png
receipt.json
```

Before rename, JSON must round-trip, PNG must reopen with the expected shape,
and receipt hashes must match both files. The receipt records the operation,
solver core, SMPL-H loader/builder/forward, and raw reader source hashes before
and after; all three repository states/commits; scikit-learn/NumPy/Torch/smplx
versions; native/model hashes; model construction options; execution device,
chunk size, timing and peak memory; and the unchanged active SMPL-H root
manifest. The artifact cannot be published inside the active AMASS SMPL-H root
or model root; the inactive SMPL-X tree is neither scanned nor validated.

Add `test_smoke/test_amass_ground_static_foot_humor_baseline.py` and register it
in the raw master. Smoke covers data-free import/help, exact frozen names,
absence of threshold CLI options, delayed model-root configuration, output
no-clobber/outside-data/model rules, SMPL-H-only runtime loading,
source-residence gates, and JSON/PNG/receipt publication structure. Exact
clustering mathematics stays in solver smoke.

### 7. Implementation order

1. extend `hjlib-smpl`, pass builder smoke, real-model data test, its documented
   standard-mode pyright fallback gate, docs/review, and commit;
2. update the solver's direct SMPL pin, implement the exact core/records and
   exhaustive solver smoke, pass strict pyright, docs/review, and commit;
3. update raw's direct SMPL pin, implement the raw configured operation and raw
   smoke;
4. run the frozen nine-motion operation once, inspect the `3 x 3` result, and
   synchronize the PNG to the Windows visual slot;
5. land usage/design/test docs, run repository masters and each documented
   pyright gate, then perform logic/implementation/cross-document closure
   review before commit.

## Smoke-Test Standard

Frozen in Mathematical Architecture section 8. The solver implementation now
executes that matrix against an independent literal oracle; the raw operation
keeps only configured-data and artifact tests.

## Migration Plan

Requirements, Mathematical Architecture, and Code Architecture reviews are
closed. The SMPL leaf is committed and the solver core has passed smoke and
strict pyright; raw operation and visual evidence follow after solver commit.

## Modification History

- 2026-08-04: Activated after the two zmin tasks were documented and committed.
  Landed Requirements from the pinned official HuMoR source; no implementation
  has started.
- 2026-08-04: Requirements contract/boundary review found no Critical issues
  and closed two Concerns by freezing equal-median label order and
  scikit-learn/default-parameter provenance. Mathematical Architecture remains
  pending.
- 2026-08-04: Clarified that native-frame toe displacement is only a permissive
  eligibility gate. Height-cluster evidence is the estimator's primary
  decision surface; upstream performs this step before 30 FPS downsampling.
- 2026-08-04: Landed Mathematical Architecture. It resolves named-track input,
  exact pooling/cluster/terrain equations, safe wrapper status, same-dtype
  numerical behavior, faithful SMPL-H 16-beta realization, middle-80-percent
  native-rate analysis, and the bounded nine-motion visual surface. Independent
  review is pending; implementation remains gated.
- 2026-08-04: Two bounded built-in review sessions failed to return after
  scope reduction, so the required mathematical review fell back to the main
  thread. It found no Critical issue and closed three Concerns: renamed the
  normal status to avoid overstating pooled noise as an estimate, bounded
  transformation invariants under finite precision, and labeled GRAB as
  estimator extension rather than complete upstream-corpus parity. Activated
  Code Architecture; implementation remains gated.
- 2026-08-04: Landed Code Architecture across the SMPL owner, pure solver, and
  raw configured-operation layers. A third bounded review session failed to
  return, so the main-thread fallback closed no Critical and four Concerns:
  both direct-consumer SMPL pins, the documented SMPL pyright fallback,
  SMPL-H-only runtime loading, and removal of duplicate linear evidence tuples.
  Implementation is ready and must start leaf-first in `hjlib-smpl`.
- 2026-08-04: Committed the `hjlib-smpl` native 16-beta/no-PCA loader, 156-pose
  builder, and typed joint forward as `64f4f49`. Implemented the solver's exact
  cluster-led comparator, immutable evidence records, public exports, literal
  oracle smoke, and updated its direct SMPL pin. Raw operation is next.
