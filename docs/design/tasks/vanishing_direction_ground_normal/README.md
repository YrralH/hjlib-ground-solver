# Ground Normal From Robust Vanishing Direction

## Requirements And Owner Boundary

`hjlib-ground-solver` owns the public solving interpretation under an explicit
scene assumption: the target ground is locally horizontal, so its normal is
parallel to the gravity/scene vertical evidenced by the input VP directions.
In a calibrated camera that robust axial vertical is the camera-space Ground
Normal up to sign. Sloped ground is not solved by this method. Architectural
verticals that do not represent gravity violate the input assumption because
the camera core deliberately performs no semantic classification.

Direction/VP clustering, scoring and refinement remain in
`hjlib-camera-solver`; generic line rasterization remains in `hjlib-geometry`.

This API solves only the unit normal `(A,B,C)`. It does not invent plane offset
`D`, camera height or world extrinsics. GT Ground Normal and dataset IO are
forbidden.

## Public API

Add the direct family dependency `hjlib-camera-solver` and expose:

```python
@dataclass(frozen=True, slots=True)
class Vanishing_Direction_Ground_Normal_Result:
    ground_normal_camera: NDArray[np.float64] | None
    direction_fusion_result: Robust_Vertical_Direction_Result

solve_ground_normal_from_vanishing_directions(
    sources: Sequence[Vanishing_Direction_Source],
    intrinsics: Camera_Intrinsics,
    config: Robust_Vertical_Direction_Config,
) -> Vanishing_Direction_Ground_Normal_Result
```

The function delegates exactly once to
`select_vertical_direction_by_robust_fusion`. The result normal is a defensive,
non-writeable float64 copy of the winning refined direction when the nested
result status is `success`, oriented by the camera-solver camera-up convention.
For `no_accepted_candidate` it is exactly `None`. Constructor validation accepts
mutable array input but makes a defensive copy; later input mutation cannot
change the result. A present normal requires shape `(3,)`, finite unit norm, no
aliasing, and exact array equality with the winner referenced by
`direction_fusion_result`. All direction candidate/rejection and score
diagnostics remain accessible through the nested owner result.

Residence is
`src/hjlib_ground_solver/estimate_ground/by_vanishing_direction.py`, re-exported
at package root. The wrapper contains no second score, threshold, optimizer or
normal copy of the direction diagnostics.

## Smoke-Test Standard

Inject synthetic method-neutral sources through the real camera-solver entry;
assert the wrapper calls it once, returns the exact winning direction by value
without aliasing, preserves nested diagnostics, defensively copies mutable
constructor input, rejects forged/nonunit mismatch constructors, and maps an
all-rejected direction result to `None` without losing its ledger. Invalid
camera-solver input failures propagate unchanged. Include an explicit sloped
ground non-applicability contract test/documentation assertion.

## Migration And Dependency Order

Land leaf-first: geometry raster primitive, camera-solver direction fusion,
then ground-solver wrapper and dependency pin, then experiments composition.
Existing ground APIs and outputs remain unchanged.

## Modification History

- 2026-08-25: Initial owner-boundary design after formal fusion review found
  that a Ground Normal public noun cannot reside in camera-solver.
- 2026-08-25: Implemented the exact one-call interpretation wrapper with
  immutable winner equality, all-rejected ledger preservation and explicit
  sloped-ground non-applicability coverage.
