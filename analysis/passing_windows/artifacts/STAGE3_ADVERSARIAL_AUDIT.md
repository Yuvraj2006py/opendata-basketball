# Stage 3 adversarial audit (Passing Windows)

**Auditor role:** independent of builder  
**Freeze:** `passing_windows_stage0_20250924`  
**Narrow-path lock:** `passing_windows_stage2_narrow_path_20260924`  
**Live clear:** 2026-09-29 (forced release-frame rebuild)

## Verdict: **SAFE TO ACCEPT** (Stage 3 complete → Stage 4 may begin)

Geometry + dual sampling + narrow-path label join all clear live checks.

---

## Live verification

**Status:** PASS
**Rows:** 190348
**Narrow-path lock:** `passing_windows_stage2_narrow_path_20260924`

| Check | Result | Detail |
|---|---|---|
| `four_candidates_per_frame` | PASS | {"n_bad": 0, "mode": 4} |
| `model_5hz_stride` | PASS | {"n_diffs": 40843} |
| `null_velocity_is_missing` | PASS | {} |
| `narrow_path_labels` | PASS | {"n_labels": 4530, "failed_eligible": 0} |
| `stage3_gate_columns` | PASS | {"present": ["narrow_path_active", "target_inference_mode", "target_reliability_status", "human_audit_status", "usable_as_receiver_completion_label", "usable_as_named_failed_target", "named_failed_target_requires_sensitivity_check", "usable_for_touch_level_turnover", "receiver_specific_eligible", "label_gate_reason"], "missing": []} |
| `release_label_yield` | PASS | {"n_labels": 4530, "n_labels_with_release_row": 4397, "n_true_receiver_rows": 4397, "n_completion_label_eligible_rows": 17588, "threshold": ">=90% of 4530 labels must have a release feature row"} |
| `no_failed_completion_labels` | PASS | {"n_bad": 0} |
| `trajectory_rule` | PASS | {} |
| `fold_safe_flight_time` | PASS | {"n_folds": 10} |
| `manifest` | PASS | {"n_files": 18, "n_missing": 0, "n_mismatched": 0, "n_duplicate_paths": 0, "missing": [], "mismatched": [], "duplicate_paths": []} |
| `markers` | PASS | {"n_done": 10, "n_games": 10} |

