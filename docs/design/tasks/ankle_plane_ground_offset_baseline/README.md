# Ankle Plane Ground Offset Baseline

## Requirements

Register the accepted VirtualCrowd Ankle Plane A offset as
`ground_offset_baseline002` without changing `ground_offset_baseline001`.
Baseline002 freezes strict confidence `>2.1`, strict ankle-pair/bbox-width ratio
`<0.10`, `H=1.22 m`, equal observation weights and
`np.arange(-5,80,0.05)` for D.

The baseline remains camera- and dataset-neutral. It consumes the existing
`Ground_Offset_Observations`, a supplied camera-up Ground Normal and supplied K.
It owns no tracked-scene adapter, GT support, VirtualCrowd path or result value.

## Mathematical Architecture

Both offset baselines use the same objective and `solve_D_search` path. The
only difference is their registered selection/search configuration:

| Baseline | Confidence | Ankle ratio | H | D step |
| --- | ---: | ---: | ---: | ---: |
| `ground_offset_baseline001` | `>4.3` | `<0.20` | 1.27 m | 0.10 m |
| `ground_offset_baseline002` | `>2.1` | `<0.10` | 1.22 m | 0.05 m |

`select_ground_offset_observations` continues to apply strict inequalities.
`solve_ground_offset` continues to preserve the supplied float64 camera-up
normal, use unweighted rows and require a positive finite winning D.

## Code Architecture

Extend the existing `Ground_Offset_Baseline` enum and the closed switch in
`ground_offset_config`. No new solver or wrapper is introduced. The existing
`solve_ground_offset(..., baseline=...)` and result types serve both baselines.
The package exports do not change because the enum/config/solve objects are
already public.

## Smoke-Test Standard

- both baseline configs resolve their exact frozen values;
- an unknown ID still fails with both legal IDs listed;
- baseline001 behavior remains unchanged;
- baseline002 selects strict boundary cases and solves the synthetic plane;
- strict pyright and focused smoke tests pass.

## Migration Plan

This is additive. Existing callers retain baseline001 through the current
default. The experiments application switches its accepted A execution from a
private copy of the constants to baseline002 and verifies exact ALL8 parity.

## Modification History

- 2026-09-15: design accepted for implementation after the ALL8 A reproduction
  and numerical review.
- 2026-09-15: implemented the additive enum/config row; focused smoke tests and
  strict pyright passed, and the experiments High-Level API reproduced the
  frozen ALL8 values.
