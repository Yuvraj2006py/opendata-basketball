# Stage 2 pre–Stage 3 readiness audit

**Verdict:** `PASS`  
**Date:** 2026-09-29  
**Lock:** `passing_windows_stage2_narrow_path_20260924`  
**Freeze:** `passing_windows_stage0_20250924`

Fresh independent audit of on-disk Stage 2 tables against
`configs/stage2_narrow_path.yaml` and Stage 0 kill/fallback rules.

## What was executed

1. `pytest tests/test_stage1.py tests/test_stage2.py` → **114 passed**
2. Stage 1 + Stage 2 `--verify-manifest` → **OK** (33 + 15 files, 0 mismatches)
3. Independent live-table gate audit (`pipelines/_stage2_pre_stage3_audit.py`) → **PASS**
4. Packaging remediation: probabilities table previously had **7/10** lock gate columns;
   `_attach_gates_to_candidates` now broadcasts the full `GATE_COLUMNS` set;
   Stage 2 rebuilt with `--force`; tests now assert gate columns on probabilities too

## Checks

| Check | Result |
|---|---|
| `kill_rates` | **PASS** |
| `narrow_path_triggers` | **PASS** |
| `gate_columns` (all four written_to tables, full 10 cols) | **PASS** |
| `assert_narrow_path_gates` | **PASS** |
| `interceptor_rule` | **PASS** |
| `label_policy` | **PASS** |
| `reviewer_blank` | **PASS** |
| `simplex` (`target_probability`) | **PASS** |
| `logo_folds` | **PASS** |
| `eligibility_unique` | **PASS** |
| `report_phrasing` (no failed-pass precision proxy abuse) | **PASS** |

## Counts

- Failed passes: **89** (auto=60, ambiguous=27, unresolved=2)
- Pre-audit unresolved: **32.58%** (freeze kill line 10%; breach → narrow path)
- Strict unresolved: **2.25%**
- Stage 3 default training labels (`receiver_specific_eligible`): **4530**
- Probability rows: **18476** across **4619** passes

## Known non-blocking residual (by design)

| Residual | Status |
|---|---|
| Human failed-pass audit | `deferred_waived` — intentional under narrow path; reviewer columns blank |
| `failed_residuals_within_completed_support` | Diagnostic **FAIL** — evidence that completed-pass proxy ≠ failed-pass precision; already locked as diagnostic-only |
| Held-out completed-pass top-1 / auto-accept precision (0.819 / 0.968) | Diagnostic only; must not be cited as failed-pass precision |

## Stage 3 go / no-go

**CONDITIONAL GO** under `target_inference_mode: narrow_path` only.

Mandatory constraints:

1. Join `tables/stage2_label_eligibility.parquet`.
2. Completion labels only via `receiver_completion_training_labels` / `receiver_specific_eligible`.
3. Failed passes = touch-level turnover evidence by default.
4. Auto-accepted failures opt-in only with exclusion sensitivity published in the same artifact.
5. Never fill reviewer columns; never cite completed-pass proxy as failed-pass precision.
6. Never describe Stage 2 failed-pass targets as audited or validated.

**NO-GO** if Stage 3 would train on inferred failed targets without those gates.

## Artifacts

- `artifacts/STAGE2_PRE_STAGE3_AUDIT.json` — machine-readable twin of this verdict
- `pipelines/_stage2_pre_stage3_audit.py` — reproducible audit entry point
