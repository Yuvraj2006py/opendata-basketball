# Stage 5 verification

**Freeze:** `passing_windows_stage0_20250924`
**Upstream model:** `stage4_sklearn_logo_v3`
**Games processed:** 1
**Series rows (5 Hz):** 18968
**Window rows (incl. never_open stubs):** 1386

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
| 114086 | fold_holdout_114086 | 18968 | 1386 | 0.08471637716969888 | 0.03471637716969887 |

## Status

MVP ship — see `STAGE5_BUILDER_STATUS.md` for remaining audit gaps.
