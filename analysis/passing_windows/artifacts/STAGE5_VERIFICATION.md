# Stage 5 verification

**Freeze:** `passing_windows_stage0_20250924`
**Upstream model:** `stage4_sklearn_logo_v3`
**Games processed:** 10
**Series rows (5 Hz):** 176004
**Window rows (incl. never_open stubs):** 12827

## Binding checks (MVP)

- Window series filtered to `is_model_5hz_frame`
- Causal EWMA only (no future frames)
- Open requires q ≥ open_thr AND NOV > 0 AND not rejected
- Persistence ≥ 0.20s (≥2 consecutive 5 Hz samples)
- Hysteresis: close_threshold ≤ open_threshold
- Thresholds selected on train games only (nested LOGO)
- `used` requires known-receiver narrow-path gates
- Existence probability: `deferred_stage7`

## Per-game summaries

| gameId | fold_id | n_series | n_windows | open_thr | close_thr |
|---|---|---:|---:|---:|---:|
| 114243 | fold_holdout_114243 | 16832 | 1312 | 0.08352719789767807 | 0.033527197897678065 |
| 114234 | fold_holdout_114234 | 16940 | 1278 | 0.08319977482877075 | 0.03319977482877075 |
| 114169 | fold_holdout_114169 | 16632 | 1221 | 0.08438739109184806 | 0.03438739109184806 |
| 114099 | fold_holdout_114099 | 18212 | 1307 | 0.08121811265986789 | 0.03121811265986789 |
| 114086 | fold_holdout_114086 | 18968 | 1386 | 0.08471637716969888 | 0.03471637716969887 |
| 178442 | fold_holdout_178442 | 18128 | 1190 | 0.08456519726433204 | 0.034565197264332034 |
| 179612 | fold_holdout_179612 | 17392 | 1378 | 0.08374404896358349 | 0.03374404896358349 |
| 184439 | fold_holdout_184439 | 15940 | 1154 | 0.08373005673628262 | 0.03373005673628261 |
| 188630 | fold_holdout_188630 | 19736 | 1379 | 0.08241651241239599 | 0.03241651241239599 |
| 191313 | fold_holdout_191313 | 17224 | 1222 | 0.08088058453219571 | 0.030880584532195707 |

## Status

MVP ship — see `STAGE5_BUILDER_STATUS.md` for remaining audit gaps.
