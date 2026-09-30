# Stage 3 dual sampling

Stage 3 candidate features use **two overlapping sampling regimes**:

| Flag / role | Purpose |
|---|---|
| `is_model_5hz_frame` / `sampling_role` contains `model_5hz` | Primary-eligible stride-5 (5 Hz) frames for **window time series** (Stage 5+) |
| `is_forced_release_frame` / `sampling_role` contains `pass_release` | Every pass `startFrame` with usable tracking, **force-inserted** even when off the 5 Hz lattice |

Overlaps (release lands on a 5 Hz frame) are dual-flagged (`sampling_role = model_5hz+pass_release`).

## Coverage (current run)

- Narrow-path completion labels: **4530**
- Labels with a Stage 3 release feature row: **4397** (~97.1%, above the 90% Stage 4 gate)
- True-receiver rows: **4397**
- Residual gaps: touches that cannot form exactly four teammate candidates, or release frames without tracked players

## Why

Stage 4 passability is defined at **release-time state**:

`q(j,t) = P(pass reaches j | release-time state)`

Only ~22% of completion-eligible passes had `startFrame` on the primary 5 Hz grid. Forced release rows recover ≥90% of the 4530 narrow-path completion labels for training while leaving the 5 Hz series intact for option-curve / window detection.

## Stage 4 usage

- **Passability / choice labels:** filter `is_pass_release_frame` (or `is_forced_release_frame`) and `completion_label_eligible`.
- **Window series:** filter `is_model_5hz_frame`.
- Do not require `frameIdx % 5 == 0` on release rows.
