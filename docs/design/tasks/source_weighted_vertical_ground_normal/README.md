# Source-Weighted Vertical-Line Ground Normal

## Requirements

Under an explicit locally-horizontal scene assumption, expose the camera-solver
source-total weighted axial-direction fit as a camera-space unit Ground Normal.
The method solves only the normal: plane offset, camera height, and slope are
out of scope.

## Mathematical Architecture

Every input source must already have been independently selected as evidence of
the same physical scene-vertical/gravity axis. This caller precondition cannot
be inferred by the numerical validator: generic parallel building lines or a
mixture of directions are non-applicable inputs even if the eigensolve is
numerically well-conditioned.

Under that precondition, the camera vertical returned by the source-weighted
image-line objective is the locally-horizontal Ground Normal in the same camera
frame. The shared source `direction_frame_id` is exactly the output GN frame;
the wrapper performs no coordinate conversion. It also performs no additional
fitting, clustering, weighting, or sign choice. It inherits the camera-owner
camera-up sign, so `ground_normal_camera` must exactly equal, rather than merely
be sign-equivalent to, the nested direction. The output is an immutable,
non-aliased float64 unit vector with shape `(3,)`.

All source fractions and line semantics are owned by the camera-solver task
residence. A failure in its validation/eigensolve propagates as `ValueError`;
there is no fallback or GT-dependent branch.

## Code Architecture

Extend `estimate_ground/by_equal_vertical_lines.py` with an additive sibling
result and function:

```python
class Source_Weighted_Vertical_Line_Ground_Normal_Result:
    ground_normal_camera: NDArray[np.float64]
    direction_result: Source_Weighted_Axial_Direction_Result

def solve_ground_normal_by_source_weighted_vertical_lines(
    sources: Sequence[Source_Weighted_Image_Line_Source],
    intrinsics: Camera_Intrinsics,
) -> Source_Weighted_Vertical_Line_Ground_Normal_Result: ...
```

The module continues to own only the locally-horizontal interpretation and
delegates the numerical direction fit to `hjlib-camera-solver`. Both symbols are
re-exported from the package root. The wrapper calls the camera owner exactly
once with the unchanged `sources` and `intrinsics`, and preserves the returned
`Source_Weighted_Axial_Direction_Result` object by identity. The result
constructor requires that exact owner type, then uses `checked_ground_normal`
to create an immutable, non-aliased normal exactly equal to its nested
direction.

The module `__all__`, `estimate_ground.__init__`, and package root re-export the
same result/function objects. The legacy equal-line entry remains a direct call
to the original equal-line camera primitive; it is not simulated through the
new source-weighted path.

## Smoke-Test Standard

- exact synthetic weighted direction and exact-equal checked unit GN;
- complete nested owner receipts and configured fractions preserved;
- shared-frame mismatch and camera-solver validation/eigengap failures
  propagate without fallback;
- success and sentinel-failure spies prove exact-once unchanged forwarding,
  nested-result object identity, and propagation of the same `ValueError`
  object;
- forged negated/nonunit/wrong-shape nested results are rejected, and returned
  arrays are immutable/non-aliased;
- non-vertical line families are documented as non-applicable semantic input;
- module-to-estimate-to-root exports are object-identical, and the existing
  equal-line wrapper remains directly routed to its original camera primitive.

## Migration Plan

After camera-solver commits, update the exact dependency pin leaf-first, then
run Ground smoke and strict pyright.

## Modification History

- 2026-08-25: Requirements, mathematical architecture, code architecture, and
  smoke standard drafted before implementation.
- 2026-08-25: Mathematical review had no Critical finding. Accepted concerns
  made the shared physical-vertical precondition and exact frame/sign/shape/unit
  invariants explicit and expanded negative smoke requirements.
- 2026-08-25: Code Architecture review had no Critical finding. Accepted
  concerns froze the full public signature, exact-once/identity delegation,
  checked-result construction, three-level exports, sentinel failure semantics,
  and direct legacy routing.
- 2026-08-25: Implemented the checked wrapper/result, three-level public export,
  exact owner-result propagation, and legacy-route/negative smokes. Full owner
  smoke is 114 passed and strict pyright analyzes 52 files with 0 errors.
- 2026-08-25: Mathematical implementation review found the solve contract
  clean and no Critical issue. Its test-readiness concern added real mixed-frame
  and non-unique-eigendirection owner failures through the Ground wrapper.
- 2026-08-25: Code implementation review found no Critical or Concern. Its one
  accepted Note updated the module description to cover both objective families.
