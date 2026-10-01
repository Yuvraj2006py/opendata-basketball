# Stage 5 → Stage 6 handoff

**Status:** Stage 5 window objects ready for Stage 6 (`SAFE TO ACCEPT` pending live audit).

## Primary tables Stage 6 must consume

| Table | Grain | Notes |
|---|---|---|
| `tables/stage5_windows.parquet` | window episode (+ never_open stubs) | labels + continuous fields |
| `tables/stage5_candidate_series.parquet` | 5 Hz candidate-frame | smoothed q/V_catch/Q/NOV |
| `tables/stage5_option_set_frames.parquet` | touch × 5 Hz frame | option-set features |
| `tables/stage5_threshold_selections.parquet` | fold | LOGO open/close/late |
| `tables/stage5_threshold_grid_windows.parquet` | fold × grid delta | robustness episode counts |

## Dual sampling reminder

- Window series: `is_model_5hz_frame` only
- Use/late timing: release frames (forced OK; off-lattice OK)

- Series rows this run: **176004**
- Window rows this run: **12827**
- Option-set frames: **44001**
- 5 Hz support rejection rate: **0.1172**

## Deferred to Stage 7

- `window_existence_probability_under_tracking_perturbation` (status=`deferred_stage7`)
- Full geometry Monte Carlo / label instability adjudication (kill >15% flip)

## Do not

- Refit Stage 4 component models
- Treat interceptor as intended target
- Invent causal blame language for unused or used options

## Claim language

Use model-preferred / available under the model / unused.
Do not imply player error or counterfactual scoring from unused windows.
