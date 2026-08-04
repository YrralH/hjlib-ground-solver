# AMASS Ground Foot-Sole Domain And Lower Envelope

## Status

- Layered Design state: complete; Mathematical/Code Architecture and
  post-implementation findings closed; v5 acceptance published.
- Campaign owner: `hjlib-dataset-raw/campaigns/01_amass_raw_support`.
- Reusable owners: `hjlib-smpl` for body-model plantar topology;
  `hjlib-ground-solver` for height reduction and peeling statistics.
- Reusable implementation is public in both owner repositories; the
  authoritative configured domain artifact is raw's post-review v5 package.

## Requirements

### Purpose

Put geometric lower-envelope and static-foot methods on one anatomical
observation domain: the left and right plantar foot surface, including heel,
mid-foot, big toe, and small toe. Hands, knees, shins, and torso must not lower
a candidate merely because an action crouches, lies down, or touches the floor.

This task replaces neither historical implementation:

- the completed full-mesh lower-envelope and peeling artifacts remain frozen
  diagnostic comparators;
- the exact HuMoR implementation remains a provenance-pinned toe-joint
  comparator.

The new foot-only outputs are separately named candidates. They do not rewrite
old artifacts or silently change public full-mesh function semantics.

### Boundary

`hjlib-smpl` owns topology-aware foot and plantar vertex selection because it
owns SMPL-family models, linear-blend-skinning weights, native vertex topology,
and official heel/toe landmark interpretation.

`hjlib-ground-solver` owns stateless reduction of an explicit vertex subset to
per-frame heights and the existing sorted-low-prefix peeling algorithm. It
must not infer SMPL topology from vertex count or import AMASS paths.

`hjlib-dataset-raw` owns the configured AMASS model realization, frozen motion
identities, run profile, bounded evidence, and visualization. The raw reader
still does not expose a fabricated native `ground` field.

The task handles SMPL-H first because the fair HuMoR comparison uses native
16-beta SMPL-H. The topology contract must also admit SMPL-X without assuming
that SMPL-H and SMPL-X vertex indices coincide.

### Required outputs

For one shaped body-model instance, produce immutable left, right, and union
plantar vertex indices with enough provenance to reproduce the selection. For
one sequence, produce:

- left plantar minimum height per native frame;
- right plantar minimum height per native frame;
- union plantar minimum height per native frame;
- left/right inter-frame plantar motion evidence in metres per second;
- absolute and retained-coverage lower-envelope summaries;
- peeling results under an explicit run profile;
- mask and sequence-level visual confirmations.

The left/right tracks are required for the later robust static-foot task. That
task may add physical velocity, contact support, height clustering, and
`unknown`/`multilevel` classification, but must consume the same plantar
definition rather than returning to toe-joint height plus a fixed offset.

### Interpretation and failure policy

A foot-only lower envelope is still not semantic ground. A pure-flight crop,
feet-on-bed motion, handstand, or lying sequence without credible foot contact
may have a numerically valid foot envelope but no supported ground estimate.
The later method-analysis/static-foot layers must preserve an explicit unknown
outcome.

Invalid topology, non-finite geometry, degenerate sole landmarks, overlapping
left/right sets, missing landmark coverage, or a too-small plantar set fail
explicitly. There is no fallback to full mesh, six landmarks alone, toe joints,
or `z=0`.

### Visual acceptance

Every material layer has a bounded visible check:

1. plantar-domain acceptance renders canonical shaped feet from side and
   bottom views, distinguishing the dense foot pre-domain, selected plantar
   vertices, and six heel/toe landmarks;
2. sequence acceptance uses side orthographic views in which every candidate
   horizontal plane is a line, highlights plantar vertices below each line,
   and names peeled native frames;
3. at least one standing/walking case, one running/jump crossing, and one
   crouch/lying case are included so exclusion of low hands/body is visible.

Large meshes are not persisted. JSON, PNG, and a receipt are sufficient.

## Mathematical Architecture

### 1. Inputs and notation

For one body-model topology let:

```text
W in R^(V x J_W)     finite LBS weights
X0 in R^(V x 3)      finite shaped zero-pose vertices, metres
Q0 in R^(J_Q x 3)    finite shaped zero-pose joints, metres
X in R^(T x V x 3)   finite posed sequence vertices, metres
u in {0,1,2}         world up-axis index; AMASS acceptance uses u=2
fps                   native frame rate in hertz
```

The zero-pose vertices and joints must come from the same model variant,
gender, betas, dtype family, and topology as the sequence realization. The
selection is computed once per shaped subject and reused across sequence
chunks. It does not depend on world translation or motion frames.

Require positive `T`, `V`, `J_W`, and `J_Q`; exact final coordinate dimension
three; valid `u`; finite arrays; and distinct in-range topology indices. Every
LBS weight is nonnegative and each row sum must be within configured tolerance
of one. These are model-contract checks, not silent normalization.
`fps` is a Python `int` or `float` excluding `bool`, normalized to float, and
must be finite and positive.

For each side `s` in `{L,R}`, the topology contract supplies distinct indices:

```text
aW_s, fW_s       ankle/foot columns of W
aQ_s             corresponding ankle row of Q0
r_s, b_s, c_s    heel/big-toe/small-toe vertices of X0 and X
```

The current official SMPL-H/SMPL-X body-joint contract gives equal numeric
values for `aW_s` and `aQ_s`, but the mathematics does not infer or require that
equality. SMPL-H and SMPL-X mappings are distinct named records backed by the
installed official `smplx` model tables. Every mapping is validated against
`J_W`, `J_Q`, and `V` before array indexing.

### 2. Dense foot pre-domain

With minimum combined skinning weight `tau_w`, define:

```text
D_s = {v | W[v,aW_s] + W[v,fW_s] >= tau_w}.
```

The two sets must be nonempty, disjoint, and contain their three official
landmark vertices. This pre-domain deliberately includes more than the sole;
it excludes unrelated body regions before the geometric plantar test.

### 3. Shaped plantar reference plane

For each side form landmark points:

```text
p_r = X0[r_s]
p_b = X0[b_s]
p_c = X0[c_s]
n_raw = cross(p_b - p_r, p_c - p_r).
```

If `||n_raw||` is non-finite or below the configured squared-metre degeneracy
tolerance, the selection is invalid. Normalize `n_raw`, then compute the ankle
orientation evidence:

```text
p_bar = (p_r + p_b + p_c) / 3
n = n_raw / ||n_raw||
o = dot(n, Q0[aQ_s] - p_bar)
if abs(o) <= tau_o: invalid ambiguous orientation
if o < -tau_o: n = -n and o = -o.
```

`tau_o` is a positive orientation margin in metres. A valid oriented result has
`o > tau_o`; landmark ordering can never decide a near-zero case silently.
Thus positive signed distance points toward the dorsal foot/ankle. This plane
is a foot-local selector, not a claim about the sequence's world ground.

For `v in D_s`, signed distance is:

```text
d_s(v) = dot(n, X0[v] - p_bar).
```

With maximum plantar distance above the landmark plane `tau_p`, define:

```text
F_s = {v in D_s | d_s(v) <= tau_p}.
```

The generic configuration domain is:

```text
tau_w in (0,1]
tau_p finite and >= 0 m
tau_deg finite and > 0 m^2
tau_o finite and > 0 m
tau_sum finite and >= 0
n_foot integer >= 3
```

The initial AMASS mask probe freezes `tau_w=0.5`, `tau_deg=1e-8 m^2`,
`tau_o=1e-4 m`, `tau_sum=1e-5`, and `n_foot=96` vertices per side. It compares
the exact `tau_p` candidates `(0.0, 0.003, 0.005, 0.008, 0.010) m`, renders
every candidate for both SMPL-H and SMPL-X, and records counts, selected signed-
distance ranges, landmark membership, and cross-shape index stability. One
value is frozen only after user-visible mask acceptance; until then configured
AMASS artifacts remain gated. There is no gender- or motion-specific tuning.

Negative distance is retained because it is on the plantar-facing side of the
landmark plane. The construction cannot prove that every far-negative
posterior/lateral foot vertex is anatomically plantar, so the signed-distance
range and bottom/side visualization are acceptance evidence rather than a
theorem of anatomical segmentation.

Each final side must contain all three landmarks and at least a configured
minimum vertex count. `F_L` and `F_R` must be disjoint. Their union is `F`.

### 4. Rigid-frame invariance

For any common rigid transform `(R,t)` applied to `X0` and `Q0`, oriented
signed distances are unchanged, up to finite-precision tolerance. Therefore
the selected indices are invariant to a common canonical coordinate rotation
or translation. They are not invariant to shape, topology, or model variant;
those are intentional inputs and recorded provenance.

### 5. Per-frame plantar heights

For sequence vertices in a `+u`-up world frame:

```text
h_L(t) = min_{v in F_L} X[t,v,u]
h_R(t) = min_{v in F_R} X[t,v,u]
m(t)   = min(h_L(t), h_R(t))
       = min_{v in F} X[t,v,u].
```

This reduction is chunk-local on the device. Only the three `T`-length scalar
tracks move to CPU. No full sequence mesh or second dataset representation is
persisted.

High positive-distance dorsal vertices are excluded by `F_s`, and unrelated
body contact is excluded by `D_s`. The geometric rule alone does not prove that
every retained far-negative sidewall vertex is plantar; quantitative mask
receipts and visible acceptance close that model-specific boundary.

### 6. Inter-frame plantar motion evidence

The later static-foot task cannot derive physical motion from changing scalar
argmin heights. During the same chunked forward pass, this task also computes
for side `s` and native interval `t-1 -> t`:

```text
q_s(t) = fps * median_{v in F_s} ||X[t,v,:] - X[t-1,v,:]||_2,
         t = 1,...,T-1.
```

For an odd vertex count the median is the ordered middle value. For an even
count it is the overflow-safer arithmetic midpoint
`lower + (upper-lower)/2` of the two ordered middle values, not
`torch.median`'s lower-middle convention. The midpoint and final speed must
remain finite.

`q_s` has shape `(T-1,)` and units metres per second. It has no fabricated
first/final sample. Chunked execution carries only the preceding chunk's final
selected plantar positions, not a full mesh history. The robust static-foot
task may accept or replace this contact evidence in its own reviewed math, but
it must explicitly choose; `h_s(t)` alone is insufficient.

Finite `X` can still overflow subtraction, norm, multiplication by `fps`, or
median. Every derived displacement and speed is checked finite; any overflow
or non-finite result rejects the operation rather than entering clustering.

### 7. Lower-envelope summaries

The existing exact retained-coverage and peeling mathematics consume `m(t)`
unchanged. Their generic scalar contracts do not know whether inputs came from
full mesh or a vertex subset. Output names and artifact metadata must say
`foot_sole_lower_envelope`, not `mesh_lower_envelope` or semantic ground.

The first candidate peeling profile is:

```text
maximum candidate fraction per round = '0.02'
maximum candidate frames per round    = 24
minimum retained frames               = 32
reference gap window                  = 32 slots
minimum absolute boundary gap         = 0.00075 m
minimum boundary/reference gap ratio  = 4.0
maximum applied rounds                = 3
maximum total removed fraction        = '0.05'
maximum total removed frames          = 60
```

Only the absolute gap and ratio are mildly more sensitive than the historical
full-mesh profile (`0.001 m`, `5.0`). Search and deletion budgets stay fixed.
A read-only nine-motion SMPL-H **dense pre-domain** probe, before the geometric
`F_s` selector existed, found seven unchanged results, a stable five-frame EKUT
peel, and a stable six-frame SFU peel. The more aggressive
`0.0005 m`, `3.0` profile made SFU hit the maximum-round instability state and
is rejected as the initial profile.

This evidence only motivates a candidate; it does not freeze the profile. The
complete profile and probe must rerun after `tau_p` is visibly accepted, and
the exact mask parameters are recorded with it. It remains one sequence-wide
profile, never silently tuned per motion.

### 8. Static-foot handoff

The robust static-foot sibling receives `h_L(t)`, `h_R(t)`, `q_L(t)`, and
`q_R(t)` from the same pass, or explicitly replaces `q_s` under its own reviewed
contract. Its support-height samples must
be plantar surface heights. It must not subtract HuMoR's fixed `0.01 m`
toe-joint offset, pool DBSCAN noise as a cluster, or force a candidate when no
credible foot support exists.

The exact details of velocity normalization, contact eligibility, clustering,
dominance, and multilevel classification remain owned by that task's reviewed
Mathematical Architecture. This task freezes only the shared anatomical height
domain.

### 9. Invariants

For every valid result:

```text
F_L subset D_L; F_R subset D_R
F_L intersection F_R is empty
all six heel/toe landmarks belong to their side's F_s
min(|F_L|, |F_R|) >= configured minimum
m(t) = min(h_L(t), h_R(t)) for every t
all output heights are finite and preserve input length T
q_L and q_R are finite, nonnegative, and have length T-1
chunking does not change indices or scalar tracks
inputs are not mutated
```

Reordering vertices together with weights, landmarks, vertices, and sequence
indices produces the correspondingly permuted sets and identical heights.

### 10. Edge cases

- No valid dense or plantar set: fail, never use full mesh.
- Degenerate/collinear sole landmarks: fail.
- Landmark absent from its LBS pre-domain: fail and expose model-contract drift.
- Very short sequence: subset heights remain valid; peeling retains its
  existing explicit retained-count and zero-budget behavior.
- One foot never approaches support: numerical union minima remain a
  diagnostic; later contact classification may use the other foot or return
  unknown.
- Feet at multiple persistent levels: do not flatten here; later analysis may
  classify multilevel/piecewise.
- Pure flight or lying with raised feet: numeric result is not promoted to
  ground.
- `T=1`: height tracks are length one and motion tracks are valid empty
  `(0,)` arrays; peeling retains its existing length/budget validation.

## Code Architecture

### 1. `hjlib-smpl`: topology and plantar selection

Add `src/hjlib_smpl/foot_support.py`. It is a NumPy metadata/geometry module;
it does not run a body model, read model files, depend on AMASS, or retain
mutable model state.

Public records:

```python
type SMPL_Foot_Topology_Name = Literal['smplh', 'smplx']

@dataclass(frozen=True, slots=True)
class Foot_Side_Topology:
    ankle_weight_index: int
    foot_weight_index: int
    ankle_joint_index: int
    heel_vertex_index: int
    big_toe_vertex_index: int
    small_toe_vertex_index: int

@dataclass(frozen=True, slots=True)
class SMPL_Foot_Topology:
    name: SMPL_Foot_Topology_Name
    left: Foot_Side_Topology
    right: Foot_Side_Topology

@dataclass(frozen=True, slots=True)
class Plantar_Vertex_Selection_Config:
    minimum_combined_skinning_weight: float
    maximum_distance_above_landmark_plane_in_meter: float
    minimum_landmark_cross_norm_in_square_meter: float
    minimum_ankle_orientation_margin_in_meter: float
    maximum_lbs_weight_sum_error: float
    minimum_vertex_count_per_side: int

@dataclass(frozen=True, slots=True)
class Foot_Side_Plantar_Selection:
    dense_vertex_indices: tuple[int, ...]
    plantar_vertex_indices: tuple[int, ...]
    landmark_vertex_indices: tuple[int, int, int]
    plane_center_in_meter: tuple[float, float, float]
    dorsal_normal: tuple[float, float, float]
    ankle_orientation_distance_in_meter: float
    dense_signed_distance_range_in_meter: tuple[float, float]
    plantar_signed_distance_range_in_meter: tuple[float, float]

@dataclass(frozen=True, slots=True)
class Plantar_Vertex_Selection:
    topology: SMPL_Foot_Topology
    config: Plantar_Vertex_Selection_Config
    left: Foot_Side_Plantar_Selection
    right: Foot_Side_Plantar_Selection
    union_vertex_indices: tuple[int, ...]

@dataclass(frozen=True, slots=True)
class Shaped_Foot_Reference:
    topology: SMPL_Foot_Topology
    model_class_name: str
    beta_count: int
    lbs_weights: NDArray[np.floating[Any]]
    reference_vertices_in_meter: NDArray[np.floating[Any]]
    reference_joints_in_meter: NDArray[np.floating[Any]]
```

Public functions:

```python
get_smpl_foot_topology(name) -> SMPL_Foot_Topology

build_shaped_foot_reference_from_model(
    model,
    betas,
    topology,
) -> Shaped_Foot_Reference

resolve_plantar_vertex_selection(
    lbs_weights,
    reference_vertices_in_meter,
    reference_joints_in_meter,
    topology,
    config,
) -> Plantar_Vertex_Selection
```

`get_smpl_foot_topology` is a cheap lookup over immutable module constants. It
uses the installed official `smplx.vertex_ids` heel/toe table at import time
and separately records body ankle/foot indices. It never infers topology from
vertex count.

`build_shaped_foot_reference_from_model` is the public owner-side adapter that
prevents raw code from reading model internals or issuing an unreviewed direct
zero-pose call. It accepts an official `smplx` SMPL-H or SMPL-X module plus one
one-dimensional floating beta tensor on the model device. Under inference mode
it performs exactly one zero-pose/no-translation forward, copies `lbs_weights`,
the single shaped vertex array, and the model's complete native joint output to
finite C-contiguous CPU NumPy arrays, marks those owned copies read-only, and
validates them against the explicit topology. It
does not touch `SMPLX_Driver._model`; configured raw code obtains the public
model from existing `hjlib-smpl` loaders and passes it to this adapter. The
result records model class and beta count; the raw receipt remains responsible
for model-file and beta provenance.

Compatibility is strict: an official `smplx.SMPLH` instance accepts only the
`'smplh'` topology and an official `smplx.SMPLX` instance only `'smplx'`;
layers/other model classes are rejected. The beta count must equal the model's
configured `num_betas`, and beta dtype/device must match the model's shapedirs
contract. Any configured `joint_mapper` is rejected because it would change
the native `Q0` row namespace. The complete unremapped `output.joints[0]` is
stored; no ambiguous leading-row truncation is allowed.

`resolve_plantar_vertex_selection` accepts floating NumPy arrays, copies no
large sequence, validates the shared contract, and delegates each side to one
stateless side-selection function. Input normalization, one-side plane math,
and final cross-side invariants remain separate functions, each intended to fit
within roughly 100 lines. Only the public records/functions are re-exported
from `hjlib_smpl`; non-exported helpers do not use a leading underscore because
HJ naming reserves that prefix for state-risky external calls.

Add synthetic `test_smoke/test_foot_support.py`, wire it into
`test_smoke/test_all_func.py`, and add model-backed
`test/test_foot_support_with_data.py` for one official SMPL-H and one official
SMPL-X model. Existing top-level-import tests gain the public records/functions.

### 2. `hjlib-ground-solver`: explicit subset observations

Add
`src/hjlib_ground_solver/estimate_ground/by_vertex_subset_observation.py`.
It owns no SMPL topology and accepts an explicit device-local `torch.long`
index tensor.

```python
@dataclass(frozen=True, slots=True)
class Vertex_Subset_Observation_Chunk:
    minimum_height: torch.Tensor
    interval_median_speed: torch.Tensor
    final_vertex_positions: torch.Tensor
    final_vertex_indices: torch.Tensor

compute_vertex_subset_observation_chunk(
    vertices,
    vertex_indices,
    up_axis_index,
    frame_rate_in_hz,
    previous_vertex_positions=None,
    previous_vertex_indices=None,
) -> Vertex_Subset_Observation_Chunk
```

The function validates `(B,V,3)` floating finite vertices, a nonempty unique
in-range one-dimensional long index tensor on the same device, the up axis,
finite-positive Python FPS, and optional previous positions of shape `(N,3)`
with matching dtype/device plus the exact same ordered subset identity. Positions
and indices must be provided together. It uses one `torch.index_select`, computes `(B,)`
heights and the mathematically specified interval median speeds, checks every
derived tensor finite, and returns the last selected positions for the next
chunk as a storage-independent `clone()`, never as a view retaining the whole
chunk allocation. `clone()` preserves autograd; callers performing inference
decide whether to detach the carry.

No existing full-mesh reducer, summary, peeling function, or exact HuMoR API is
renamed or behaviorally changed. Re-export the new record/function from the
estimate subpackage and package root. Add
`test_smoke/test_vertex_subset_observation.py` and wire the master runner.

### 3. `hjlib-dataset-raw`: configured probe and acceptance

Add `test/amass_ground_foot_sole_domain.py` as a Typer CLI, not reader API. It
imports package APIs only after explicit repository/model paths are validated,
uses the existing AMASS reader, native-16-beta/no-PCA SMPL-H builder, frozen
nine-motion identities, middle-80-percent window, atomic output publication,
repository/source hashes, and data-root manifest checks.

The operation is explicitly a **mask/profile probe** until one `tau_p` is
visually accepted. It generates all five configured `tau_p` candidates in one
model-reference pass per motion and records candidate identities rather than
silently selecting one. For each candidate it:

1. obtains `W/X0/Q0` only through
   `build_shaped_foot_reference_from_model` and resolves left/right masks;
2. realizes sequence chunks once, accumulating left/right height and speed
   tracks for all small candidate masks without moving full meshes to CPU;
3. computes union lower-envelope summaries and peeling under current plus mild
   profiles;
4. emits mask/sequence panels and compact JSON evidence;
5. publishes a completion receipt only after hashes, counts, repository state,
   and unchanged AMASS manifests pass.

Split orchestration into named functions for frozen input loading, shaped
reference resolution, chunk observation, candidate analysis, mask rendering,
sequence rendering, payload construction, and atomic publication. Model
instances and device/chunk state remain operation-local; mathematical reducers
remain stateless library functions.

Sequence rendering never retains full-motion meshes. After scalar analysis
selects at most three diagnostic native frames for each of at most five named
motions (maximum 15 total), a bounded second model forward realizes only those
frames and transfers their mesh/plantar points to CPU for side-orthographic
rendering. Canonical mask panels use the already bounded zero-pose reference.

Add `test_smoke/test_amass_ground_foot_sole_domain.py` for frozen identities,
profile construction, delayed-import isolation, CLI help, outside-data/output
validation, no-clobber publication, failed-publication cleanup, absence of a
completion receipt on failure, JSON/PNG reopening, receipt/hash consistency,
schema helpers, and data-free render fixtures, then wire the existing raw smoke
master. Real model/data execution is invoked through the CLI and recorded in
the campaign task, not run during ordinary smoke.

### 4. Dependency and metadata decisions

- `hjlib-smpl` remains the L2 body-model owner and uses only its existing
  `numpy`/`smplx` dependencies.
- `hjlib-ground-solver` remains L3 and already directly depends on
  `hjlib-smpl`; the numerical module itself stays topology-free.
- `hjlib-dataset-raw` composes both only in its configured test operation.
- No new dependency or direction is introduced.
- Existing `pyproject.toml` descriptions still match each repository owner
  boundary; no description change is planned unless implementation review
  finds a broader public-surface routing need.
- Because `hjlib-smpl` has unrelated in-progress changes, edits are limited to
  one new module plus surgical re-export/test/doc additions. Existing diffs are
  preserved and are not committed by this task. Before touching every existing
  overlapping file, make a fresh snapshot. If the user later authorizes a
  commit, use selected-hunk staging plus the HJ commit gate; never whole-file
  stage these overlaps. Alternatively finish/commit the pre-existing SMPL work
  first under its own scope.

## Smoke-Test Standard

### `hjlib-smpl` synthetic matrix

1. A small symmetric fixture selects disjoint left/right dense and plantar
   sets, includes all landmarks, orients both normals toward their ankles, and
   records exact ranges/counts.
2. Reversing landmark order preserves the selected indices and flips then
   reorients the normal consistently.
3. Common translations and proper rotations preserve indices and signed
   distances within tolerance; coordinated vertex permutation preserves the
   result under inverse permutation.
4. Degenerate landmarks, near-plane ankle orientation, non-finite arrays,
   wrong dimensions, negative or mis-summed LBS rows, invalid/out-of-range or
   colliding topology indices, missing landmarks, overlap, too-small sets, and
   every invalid config field raise `ValueError`.
5. Inputs are unchanged and records/configs are frozen.
6. Official topology lookup returns distinct SMPL-H/SMPL-X landmark indices
   and rejects unknown names.
7. Top-level and canonical submodule exports resolve to identical objects.
8. The public model adapter rejects malformed betas/model outputs and returns
   topology-consistent finite CPU copies without exposing private driver state.
9. Wrong SMPL-H/SMPL-X topology labels, unsupported model/layer classes,
   beta count/dtype/device mismatch, and a configured joint mapper are rejected;
   the accepted reference preserves the complete native output-joint row count.

### `hjlib-smpl` model-backed matrix

For one SMPL-H and one SMPL-X official model, resolve zero-pose selections and
verify finite nondegenerate planes, positive orientation margins, six landmark
memberships, disjoint sides, minimum counts, in-range indices, and repeated-call
equality. No fixed count is asserted across topologies or shapes.

### `hjlib-ground-solver` synthetic matrix

1. Hand-computable heights and median speeds match an independent oracle.
2. Single-frame/no-previous produces `(1,)` heights and `(0,)` speeds; a
   previous carry produces one interval speed.
3. Two chunks plus carry equal one full-chunk call exactly for heights, speeds,
   and final positions.
4. Left/right calls satisfy union height `min(h_L,h_R)` against direct union
   indexing.
5. Float32/float64, CPU and CUDA when available, autograd, input immutability,
   and deterministic repeated calls are covered.
6. Invalid shapes/dtypes/devices/indices/up axis/FPS/previous state and
   non-finite or derived-overflow values raise `ValueError`.
7. Odd and even vertex-count fixtures verify the ordered-middle and
   average-of-two median rules independently of `torch.median`.
8. The returned final carry has independent minimal storage and preserves
   gradients through current vertices and a supplied previous carry.
9. Existing full-mesh lower-envelope and peeling smoke remains byte-for-byte
   behaviorally green; new three-level exports are identical.

### Configured AMASS acceptance

The run must observe all frozen motions or fail the complete operation. It
records per candidate/model/motion mask counts/ranges, height/speed shapes,
absolute/coverage/peeling results, peeled native frames, hashes, timings,
memory, and unchanged roots. At least one SMPL-X model-backed mask panel is
generated even though the fair sequence comparison is SMPL-H.

Visual review must confirm heel/mid-foot/toe coverage and reject visible ankle,
dorsal, fragmented, or sidewall-dominated masks before selecting `tau_p`. A
second side-orthographic panel confirms that low hands/body no longer affect
the new envelope and that current-versus-mild profile changes occur only at
recorded peeled foot frames.

## Migration Plan

1. Preserve historical full-mesh and exact-HuMoR code/artifacts unchanged.
2. Land and review Mathematical Architecture.
3. Land and review Code Architecture plus Smoke-Test Standard.
4. Implement the leaf body-model topology contract in `hjlib-smpl`.
5. Extend generic subset reduction in `hjlib-ground-solver` without changing
   existing full-mesh behavior.
6. Implement the configured AMASS operation and visual artifact in
   `hjlib-dataset-raw`.
7. Land docs, review all affected repositories and their cross-repo contract,
   and run final gates. Do not commit unless separately authorized.

## Implemented Evidence (2026-08-04)

The reviewed leaf APIs are implemented in `hjlib-smpl/foot_support.py` and
`hjlib-ground-solver/estimate_ground/by_vertex_subset_observation.py`. Synthetic
smoke covers transform/permutation invariance, exact chunk composition, union
height, ordered median, autograd/storage, dtype/device, and derived overflow;
official SMPL-H and SMPL-X model-backed selection tests pass.

The strict configured raw operation is
`hjlib-dataset-raw/test/amass_ground_foot_sole_domain.py`. Its post-review v5
package is:

```text
/home/hj/Data_Process/sample_vis/Code_as_Libs/
amass_ground_foot_sole_domain_20260804_probe_v5/
```

The v5 `receipt.json` SHA-256 is
`0fbf140af5c58367b5b037d99299f4b7900cdd8004336b4824c15b2728788c55`.

The v5 machine-readable schema distinguishes canonical shaped-reference
`+Y up` from AMASS sequence `+Z up`; smoke freezes that schema, provenance, and
the two-line/two-highlight side-render semantics.

The receipt records 9/9 SMPL-H motions, five tau candidates each, 15 bounded
diagnostic rerenders, unchanged AMASS metadata/repositories/source hashes, and
348.8 MiB peak allocated CUDA memory. It also records selected-index hashes,
cross-shape stability, complete executed-source provenance, and typed owner-side
SMPL-H forwards. All artifact hashes reopen and match.

Visual and numeric conclusions:

- `tau_p = 0` already forms a continuous heel–mid-foot–toe footprint in the
  inspected SMPL-H and SMPL-X shapes; `3/5 mm` add a modest boundary band;
  `8/10 mm` increasingly add sidewall thickness.
- Across all nine motions, all five tau values produce identical union minimum
  height and identical current/mild peeling. Tau therefore remains unselected
  here and should be chosen for the future contact-speed stability contract,
  not to tune zmin.
- Masks are shape-dependent rather than one global fixed index list: all five
  candidates have nine distinct union hashes across the nine shaped bodies.
  Against the first shape, maximum symmetric difference is `18/18/13/11/13`
  vertices for `0/3/5/8/10 mm`; minimum Jaccard is
  `0.9388/0.9459/0.9632/0.9708/0.9667`. Callers must resolve per shaped body
  and preserve the resulting index identity in chunk carry.
- The mild `0.75 mm / 4x` profile differs from `1 mm / 5x` only for EKUT
  (five frames, candidate +2.1 mm) and SFU jump (six rather than two frames,
  candidate +2.4 mm). All other motions are unchanged.
- Side-orthographic rerenders confirm the envelope observes only the selected
  feet. In lie/crouch frames, hands may appear below that candidate; this is
  visible evidence for later contact/terrain semantics, not a reason to let
  hands redefine the plantar observation domain.

The operation intentionally writes `selected_tau: null`; `3 mm` and `5 mm`
are the remaining candidates for the next robust static-foot/contact task.

## Modification History

- 2026-08-04: Registered after side-orthographic review showed that full-mesh
  zmin can follow hands/body while exact HuMoR uses an offset toe joint. User
  required both method families to use only the plantar foot surface including
  toes. A read-only SMPL-H probe supported the mildly sensitive `0.75 mm` / `4x`
  peeling candidate while rejecting `0.5 mm` / `3x` as initially too aggressive.
- 2026-08-04: Independent Mathematical Architecture review found one Critical
  orientation ambiguity and five contract Concerns. Added an orientation
  margin/fail state, separated LBS/joint namespaces, bounded every selector
  parameter and the `tau_p` probe grid, weakened the anatomical overclaim,
  defined chunkable median plantar speed, required SMPL-H plus SMPL-X tests,
  and marked the earlier peeling probe as dense-pre-domain evidence only.
- 2026-08-04: Mathematical re-review returned zero Critical and one remaining
  Concern. Added the finite-positive native-FPS input/type contract and explicit
  overflow rejection for all derived plantar displacements and speeds.
- 2026-08-04: Final narrow mathematical verification returned zero Critical
  and zero Concern. Activated Code Architecture; implementation remains gated.
- 2026-08-04: Landed multi-repository Code Architecture: pure NumPy plantar
  topology in `hjlib-smpl`, topology-free chunk observations in
  `hjlib-ground-solver`, and a Typer probe/visual acceptance operation in raw.
  Public APIs, state carry, tests, provenance, and dirty-worktree preservation
  are explicit; independent code-architecture review is the active gate.
- 2026-08-04: Independent Code Architecture review found one Critical missing
  owner-side model/reference adapter and five Concerns. Added the public shaped
  reference adapter, storage-independent carry, exact even median, publication
  safety smoke, selected-hunk landing rule, and bounded 15-frame rerender path.
  Implementation remains gated on re-review.
- 2026-08-04: Code Architecture re-review returned zero Critical and one
  adapter-compatibility Concern. Froze exact model/topology matching, beta
  count/dtype/device parity, joint-mapper rejection, complete native joint
  output preservation, and matching negative tests.
- 2026-08-04: Final code-architecture verification returned zero Critical and
  zero Concern. Activated leaf-first implementation in `hjlib-smpl`.
