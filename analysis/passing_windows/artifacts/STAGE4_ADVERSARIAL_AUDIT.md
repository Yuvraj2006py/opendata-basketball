# Stage 4 adversarial audit (Passing Windows)

**Auditor role:** independent of builder (audit scripts + regression lock only)  
**Freeze:** `passing_windows_stage0_20250924`  
**Narrow-path lock:** `passing_windows_stage2_narrow_path_20260924`  
**Plan:** Stage 4 — Fit cross-fitted component models  
**Handoff in:** `artifacts/STAGE3_HANDOFF_STAGE4.md`  
**Handoff out:** `artifacts/STAGE4_HANDOFF_STAGE5.md`  
**Dual sampling:** `artifacts/STAGE3_DUAL_SAMPLING.md`

## Verdict: **SAFE TO ACCEPT** (live audit complete)

Live adversarial checks PASS. `pytest tests/test_stage4.py` → **69 passed**.  
Stage 4 may proceed to Stage 5 under the narrow-path lock. Builder resume **not** required for blockers.

### Residual MAJOR (documented, non-blocking)

| ID | Issue | Evidence | Required follow-up | Regression test |
|---|---|---|---|---|
| S4-M13 | Freeze names regularized logistic **GAM**; ship uses sklearn `SplineTransformer` + L2 logistic/Ridge | `artifacts/STAGE4_VERIFICATION.md:24-30` | Keep substitution explicit in article methods; do not claim GAM smooths | `test_stage4_gam_substitution_documented` |
| S4-M07/M08 | Secondary catch targets / baselines use linear probability Ridge, not multinomial GAM | `STAGE4_VERIFICATION.md:41-43` | Label as sensitivity / documented substitute | `test_stage4_secondary_outcomes_documented` (present) |

---

## Severity template (every finding)

| Field | Required content |
|---|---|
| **Finding ID** | `S4-B##` / `S4-M##` / `S4-N##` |
| **Severity** | BLOCKER / MAJOR / MINOR |
| **Rule** | Freeze / narrow-path / handoff citation |
| **Exact failure** | What is wrong (observable) |
| **Evidence** | `file:line` and/or table/artifact proof |
| **Required fix** | Concrete remediation for builder |
| **Regression test** | Proposed `test_*` name that must fail if the defect recurs |

---

## Freeze-derived adversarial checklist

### A. Validation design / leakage (LOGO)

| ID | Severity | Rule | Adversarial check | Status |
|---|---|---|---|---|
| S4-B01 | BLOCKER | Freeze nested LOGO; forbid random frame/touch/chance splits | No random sklearn splitters in fit path | **PASS** |
| S4-B02 | BLOCKER | Fit/tune/calibrate on nine; predict holdout | Holdout isolation in fold_id / train_gameIds | **PASS** (pytest) |
| S4-B03 | BLOCKER | Fold seeds `2025092401`–`10`; master `20250924` | Seeds match freeze | **PASS** (pytest) |
| S4-M01 | MAJOR | Inner CV never peeks at outer holdout | No GridSearchCV/RandomizedSearchCV on holdout | **PASS** (pytest) |

### B. Dual sampling + release-time passability

| ID | Severity | Rule | Adversarial check | Status |
|---|---|---|---|---|
| S4-B04 | BLOCKER | `q` at release-time state | Completion-eligible ∩ release frames | **PASS** (pytest) |
| S4-B05 | BLOCKER | ≥90% of 4530 labels have release rows | Live yield 4397/4530 | **PASS** |
| S4-M02 | MAJOR | Window series uses `is_model_5hz_frame` | Dual flags retained on Stage 4 table | **PASS** |
| S4-N01 | MINOR | `sampling_role` provenance | Column present | **PASS** |

### C. Narrow-path label leakage

| ID | Severity | Rule | Adversarial check | Status |
|---|---|---|---|---|
| S4-B06 | BLOCKER | Completion via narrow-path only | Zero failed cohort among eligible (joined Stage 3) | **PASS** |
| S4-B07 | BLOCKER | Never interceptor as target | pytest interceptor check | **PASS** |
| S4-B08 | BLOCKER | `GATE_COLUMNS` on Stage 4 outputs | All 10 present on predictions | **PASS** |
| S4-B09 | BLOCKER | No fabricated audit claims | Reviewer blank; no audited-language hits | **PASS** |
| S4-B10 | BLOCKER | Completed-pass proxy diagnostic-only | No failed-precision proxy phrasing | **PASS** |
| S4-M03 | MAJOR | Named-failed sensitivity if used | Default path excludes named failures | **PASS** (pytest) |

### D–F. Component models

| ID | Severity | Rule | Status |
|---|---|---|---|
| S4-B11 | BLOCKER | OOS `q_passability_model` filled | **PASS** (190348 finite) |
| S4-M04 | MAJOR | Ablation ladder | **PASS** |
| S4-B12 | BLOCKER | Imbalance weights fold-internal | **PASS** (code + verification) |
| S4-M05 | MAJOR | Held-out metrics | **PASS** (`stage4_fold_metrics.json`) |
| S4-B13 | BLOCKER | Calibration slope ~0.75–1.25 | **PASS** (overall slope ≈ 1.000; per-game in band) |
| S4-M06 | MAJOR | Fold-internal recalibration | **PASS** |
| S4-B14–B19 | BLOCKER | V_catch causal; pts from shots+FT; V_fail=0; components exposed | **PASS** |
| S4-M09 | MAJOR | Choice model metrics | **PASS** (rank acc ≈ 0.756) |
| S4-B20 | BLOCKER | No optimality language | **PASS** |

### G–I. Geometry / data / outputs

| ID | Severity | Rule | Status |
|---|---|---|---|
| S4-B21 | BLOCKER | Trajectory `lead_if_velocity_ok_else_direct` | **PASS** |
| S4-B22 | BLOCKER | Mask `rejected_outside_support` on Q/NOV | **PASS** (22865 rejected; Q/NOV NaN rate 1.0) |
| S4-B23 | BLOCKER | No season aggregates | **PASS** |
| S4-B24 | BLOCKER | Pipeline + tests exist | **PASS** |
| S4-B25 | BLOCKER | Manifest SHA-256 | **PASS** (27 files, 0 mismatch) |
| S4-B26 | BLOCKER | Regression lock | **PASS** (tests named + passing) |
| S4-B27 | BLOCKER | All 10 LOGO folds | **PASS** |
| S4-B28 | BLOCKER | Handoff coverage honest | **PASS** (190348 rows = Stage 3) |
| S4-M12 | MAJOR | Model placeholders filled | **PASS** |

---

## Concrete blockers (live)

_No open live blockers._

## Live verification

**Status:** PASS
**Verdict:** SAFE TO ACCEPT
**Narrow-path lock:** `passing_windows_stage2_narrow_path_20260924`

| Check | Result | Detail |
|---|---|---|
| `source_scan` | PASS | {"n_files": 11, "files": ["pipelines/04_fit_component_models.py", "src/passing_windows/models/__init__.py", "src/passing_windows/models/catch_value.py", "src/passing_windows/models/choice.py", "src/passing_windows/models/features.py", "src/passing_windows/models/keep_state.py", "src/passing_windo... |
| `no_random_group_split` | PASS | {"detail": "flag raw ShuffleSplit/train_test_split if used without LOGO wrapper"} |
| `no_season_aggregates` | PASS | {} |
| `outcome_not_ptsScored_as_feature` | PASS | {"detail": "ptsScored may appear only as prohibition text"} |
| `shotQuality_not_feature` | PASS | {"detail": "shotQuality may appear only in forbidden lists / comments"} |
| `trajectory_rule_preserved` | PASS | {"detail": "require lead_if_velocity_ok_else_direct in code; prohibition phrasing OK"} |
| `stage4_tables` | PASS | {"files": ["stage4_candidate_predictions.parquet", "stage4_passability_ablations_long.parquet"]} |
| `primary_table` | PASS | {"path": "stage4_candidate_predictions.parquet", "n_rows": 190348, "columns": ["gameId", "touchId", "frameIdx", "candidateId", "fold_id", "chanceId", "possessionId", "passId", "is_pass_release_frame", "is_model_5hz_frame", "is_forced_release_frame", "completion_label_eligible", "is_true_receiver"... |
| `gate_columns` | PASS | {"present": ["narrow_path_active", "target_inference_mode", "target_reliability_status", "human_audit_status", "usable_as_receiver_completion_label", "usable_as_named_failed_target", "named_failed_target_requires_sensitivity_check", "usable_for_touch_level_turnover", "receiver_specific_eligible",... |
| `no_failed_completion_labels` | PASS | {"n_bad": 0} |
| `q_oos_predictions` | PASS | {"column": "q_passability_model", "n_finite": 190348} |
| `components_not_nov_only` | PASS | {"columns": ["V_catch_model", "NOV_model"]} |
| `trajectory_rule` | PASS | {} |
| `support_rejection` | PASS | {"n_rejected": 22865, "q_nan_rate": 1.0, "nov_nan_rate": 1.0} |
| `release_label_yield` | PASS | {"n_labels": 4530, "n_hit": 4397} |
| `manifest` | PASS | {"n_files": 27, "n_missing": 0, "n_mismatched": 0, "n_duplicate_paths": 0, "missing": [], "mismatched": [], "duplicate_paths": []} |
| `report_phrasing` | PASS | {"hits": []} |
| `logo_ten_fold_coverage` | PASS | {"pred_games": [114086, 114099, 114169, 114234, 114243, 178442, 179612, 184439, 188630, 191313], "n_markers": 10, "n_freeze_games": 10, "missing_games": []} |
| `handoff_coverage_honesty` | PASS | {"detail": "ok"} |


## Go / no-go language

| Verdict | Meaning |
|---|---|
| **SAFE TO ACCEPT** | All BLOCKERs clear live + pytest; majors documented or cleared |
| **CONDITIONAL GO** | Blockers clear; residual majors require documented follow-ups before Stage 5 headlines |
| **NO-GO** | Any open BLOCKER, or missing Stage 4 artifacts |

**Current:** **SAFE TO ACCEPT** — Stage 5 may begin; keep S4-M13 GAM-substitution caveat in methods text.
