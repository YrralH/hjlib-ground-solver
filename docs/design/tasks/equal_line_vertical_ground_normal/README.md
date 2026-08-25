# Equal-Line Vertical Ground Normal

## Status

- State: implemented and implementation-reviewed
- Ground method owner: `hjlib-ground-solver`
- Camera-direction primitive owner: `hjlib-camera-solver`
- Experiment consumer: `hjlib-experiments` Campaign 04
- Initial profile: ELSED + LIMAP vertical lines joined with filtered
  `baseline001` person vertical lines

## Requirements

Expose a Ground Normal method that interprets one equal-weight image-line axial
direction as the camera-space Ground Normal under the explicit assumption that
local ground is horizontal. The Ground solver must delegate calibrated 2D-line
TLS exactly once to `hjlib-camera-solver`; it must not duplicate camera
geometry, orientation or numerical-degeneracy logic.

For Campaign 04, the independent `E+P equal-line TLS` profile is fixed as:

1. load frozen ELSED lines and LIMAP association;
2. run the existing E-only Ground Normal orthogonal-consensus wrapper with the
   exact Stage 1 config (`min_support=5`, `min_horizontal=2`,
   `min_abs_camera_y=0.8`, duplicate `2 degrees`, vertical match `3 degrees`,
   orthogonal gate `3 degrees`);
3. in its winner, require exactly one `elsed_limap` source-evidence receipt and
   use that receipt's `vertical_group_cluster_indices` field, not the
   hypothesis seed's `hypothesis_cluster_indices`;
4. select exactly E lines whose association label is in those indices;
   unassigned label `-1` and every other cluster are excluded;
5. take every P line retained by the frozen baseline001 person fit;
6. fit one new axial direction from the selected E and P segments, with every
   segment equally weighted, and interpret it as Ground Normal;
7. finish predictions for all eight scenes before reading any GT Ground Normal
   or extrinsics-derived value; allowed reviewed camera intrinsics are explicit
   solver inputs;
8. read GT Ground Normal only to compute the final sign-invariant angular
   error.

The full profile inherits the E-only selector's hard camera-y eligibility
gate. Only the joint TLS objective itself has no camera-up prior, source
balancing, line-length weight, candidate vote, robust loss or GT access.

## Mathematical Architecture

The generic equal-line objective, calibration requirements, scatter
eigendecomposition, exact eigengap gate and camera-up sign convention are owned
by [`hjlib-camera-solver`'s equal-weight image-line direction task](../../../../../hjlib-camera-solver/docs/design/tasks/equal_weight_image_line_direction/README.md).

This layer adds one mathematical assumption only: local ground normal is
parallel to the fitted scene vertical. Consequently the Ground Normal is the
camera owner's unit, canonically oriented axial direction without numerical
modification. The wrapper fails whenever the camera primitive fails.

Its result contains the checked Ground Normal and the complete nested camera
direction result. Therefore total line count, canonical per-source
`source_id/direction_frame_id/image_record_id/hash/count` receipts, eigenvalues,
eigengap and incidence residual have one source of truth and cannot drift from
a separately assembled Ground ledger.

## Code Architecture

Reusable Ground wrapper:

```text
src/hjlib_ground_solver/estimate_ground/
  by_equal_vertical_lines.py
test_smoke/
  test_equal_vertical_line_ground_normal.py
```

Public result and function:

```python
class Equal_Weight_Vertical_Line_Ground_Normal_Result:
    ground_normal_camera: NDArray[np.float64]
    direction_result: Equal_Weight_Axial_Direction_Result

solve_ground_normal_by_equal_weight_vertical_lines(
    sources: Sequence[Equal_Weight_Image_Line_Source],
    intrinsics: Camera_Intrinsics,
) -> Equal_Weight_Vertical_Line_Ground_Normal_Result
```

The function calls `fit_axial_direction_by_equal_weight_image_lines` exactly
once and passes its direction to the existing `checked_ground_normal` contract.
The result is frozen, uses immutable backing, and verifies exact agreement with
the nested owner result. Exports enter `estimate_ground/__init__.py` and the
package root. The topic smoke enters the master smoke runner and export oracle;
README, usage and design/test indexes are updated.

Experiment composition/publication is owned by the independent Campaign task:

```text
hjlib-experiments/campaigns/04_virtualcrowd_vanishing_ground/
  task_equal_line_elsed_people_fusion/README.md
```

That residence owns the staged prepare-all/evaluate-after-prepare API, E-only
preselection and label extraction, CLI/schema/path, selected-E/P artifact
identity, no-overwrite transaction, interruption and failure semantics. This
Ground residence owns only the reusable wrapper contract and its consumer
requirements.

## Smoke-Test Standard

1. Ground result equals the nested camera direction exactly and owner failures
   propagate without a fallback or second solve.
2. Result construction rejects non-owner nested results, mutable/corrupt Ground
   normals and disagreement with the nested direction.
3. Ground topic smoke, master runner, three export layers and strict pyright
   pass.
4. The independent Campaign residence owns and tests exact E provenance,
   selected label membership, all-eight prediction staging, delayed GT access,
   artifact publication and real-data result semantics.

## Migration Plan

1. Close revised Mathematical and Code Architecture reviews in both owners.
2. Implement and commit camera primitive, tests, exports, README/usage and
   design/test index.
3. Update the ground exact pin, implement and commit its exactly-once wrapper,
   tests, exports and documentation.
4. Propagate exact pins leaf-first through the full chain:
   `ground-solver -> dataset-std -> dataset-assembly -> evaluation`; run smoke
   and commit at every level, then update evaluation's ground/std/assembly pins
   together.
5. Add the independent Campaign command/profile and contract tests; run full
   experiment gates and commit.
6. Run the frozen eight-scene evaluation, record result/failure analysis and
   push the result ledger.

## Modification History

- 2026-08-25: Initial design proposed placing equal-line TLS directly in the
  Ground solver.
- 2026-08-25: Mathematical review accepted the TLS objective but required an
  exact E provenance field, honest camera-prior scope, focal-calibration
  semantics, deterministic eigengap/sign rules, explicit GT-camera allowance
  and complete receipt invariants. Code review moved generic TLS to
  `hjlib-camera-solver`, retained an exactly-once Ground wrapper, and required
  hash-bound nested receipts and full smoke/export/doc registration. The
  revised two-owner design incorporates all findings.
- 2026-08-25: The exactly-once Ground wrapper, nested result contract, exports
  and synthetic smokes were implemented after the camera primitive passed its
  implementation review.
- 2026-08-25: Mathematical and code implementation reviews closed clean after
  adding explicit owner-failure propagation, full three-level function export
  identity and unambiguous master-smoke reporting.
