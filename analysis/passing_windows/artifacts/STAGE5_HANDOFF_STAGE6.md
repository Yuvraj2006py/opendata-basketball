# Stage 5 → Stage 6 handoff

**Status:** Stage 5 MVP window objects available.

## Primary tables Stage 6 must consume

| Table | Grain | Notes |
|---|---|---|
| `tables/stage5_windows.parquet` | window episode (+ never_open stubs) | labels + continuous fields |
| `tables/stage5_candidate_series.parquet` | 5 Hz candidate-frame | smoothed q/V_catch/Q/NOV |
| `tables/stage5_option_set_frames.parquet` | touch × 5 Hz frame | option-set features |
| `tables/stage5_threshold_selections.parquet` | fold | LOGO open/close/late |

## Dual sampling reminder

- Window series: `is_model_5hz_frame` only
- Use/late timing: release frames (forced OK; off-lattice OK)

- Series rows this run: **18968**
- Window rows this run: **1386**

## Deferred to Stage 7

- `window_existence_probability_under_tracking_perturbation` (status=`deferred_stage7`)
- Full geometry Monte Carlo / label instability adjudication

## Do not

- Refit Stage 4 component models
- Treat interceptor as intended target
- Use forbidden claim language (correct/bad decision / points left on the table)
