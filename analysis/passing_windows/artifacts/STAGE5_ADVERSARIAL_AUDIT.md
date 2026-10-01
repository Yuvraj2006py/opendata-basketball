# Stage 5 adversarial audit (Passing Windows)

**Auditor role:** live adversarial checks (`pipelines/_stage5_adversarial_live_audit.py`)
**Freeze:** `passing_windows_stage0_20250924`
**Narrow-path lock:** `passing_windows_stage2_narrow_path_20260924`
**Upstream model:** `stage4_sklearn_logo_v3`
**Brief:** `artifacts/STAGE5_AUDIT_BRIEF_FOR_BUILDER.md`

## Verdict: **SAFE TO ACCEPT** (live audit complete)

### Live checks

| Check | Status | Detail |
|---|---|---|
| `tables_present` | **PASS** | all present |
| `exists:05_segment_windows.py` | **PASS** |  |
| `exists:windows` | **PASS** |  |
| `exists:test_stage5.py` | **PASS** |  |
| `exists:STAGE5_REGRESSION_LOCK.md` | **PASS** |  |
| `no_random_splits` | **PASS** | [] |
| `no_season_aggregates` | **PASS** | [] |
| `no_component_refit` | **PASS** | [] |
| `forbidden_claim_language` | **PASS** | [] |
| `series_5hz_only` | **PASS** |  |
| `n_series_176004` | **PASS** | 176004 |
| `ten_folds` | **PASS** | 10 |
| `grid_rows_ge_50` | **PASS** | 50 |
| `gate_columns` | **PASS** | [] |
| `col_q_smooth` | **PASS** |  |
| `col_V_catch_smooth` | **PASS** |  |
| `col_Q_smooth` | **PASS** |  |
| `col_NOV_smooth` | **PASS** |  |
| `rejection_rate_finite` | **PASS** | 0.1172 |
| `labels_subset` | **PASS** | ['late', 'never_open', 'unused', 'used'] |
| `regression_lock_blockers` | **PASS** | [] |
| `manifest_n_files` | **PASS** | 40 |

### Residual / deferred

- Existence probability under tracking perturbation → Stage 7
- Label-stability kill (>15% flip) adjudication → Stage 7
- Full geometry Monte Carlo → Stage 7

### Go / no-go language

Stage 5 may proceed to Stage 6 under the narrow-path lock. Do not claim unused windows are mistakes; keep q and V_catch visible.
