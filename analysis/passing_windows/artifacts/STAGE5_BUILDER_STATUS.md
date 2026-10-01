# Stage 5 builder status

**Date:** 2026-09-30  
**Upstream:** `stage4_sklearn_logo_v3`  
**Audit brief:** `artifacts/STAGE5_AUDIT_BRIEF_FOR_BUILDER.md`  
**Live audit:** **SAFE TO ACCEPT** (`artifacts/STAGE5_ADVERSARIAL_AUDIT.md`)

## Shipped

| Deliverable | Status |
|---|---|
| `src/passing_windows/windows/` | Done |
| `pipelines/05_segment_windows.py` | Done |
| `pipelines/_stage5_adversarial_live_audit.py` | Done |
| `configs/stage5_windows.yaml` | Done |
| `tests/test_stage5.py` | **37 passed** |
| Full 10-game LOGO run | series=**176004**, windows=**12827**, grid=**50** |
| Threshold grid table | `tables/stage5_threshold_grid_windows.parquet` |
| Window report | `artifacts/stage5_window_report.md` |
| Verification / handoff / regression lock / manifest | Done |

## Binding rules locked

- 5 Hz series only; causal EWMA; open = `q≥thr ∧ NOV>0 ∧ ¬rejected`
- Persistence ≥0.20s; hysteresis `close≤open`; LOGO train-only thresholds
- `used` = known-receiver only; existence prob deferred Stage 7
- Robustness grid (±0.10 open deltas) emitted as episode-count summaries

## Deferred (honest Stage 7)

- Tracking-perturbation existence probability
- Label-stability kill adjudication (>15% flip)

## Stage 6

Consume tables listed in `STAGE5_HANDOFF_STAGE6.md`.
