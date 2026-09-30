# Stage 4 regression lock

**Freeze:** `passing_windows_stage0_20250924`  
**Narrow-path lock:** `passing_windows_stage2_narrow_path_20260924`  
**Purpose:** Every Stage 4 adversarial finding must map to a regression test that prevents recurrence.

**Status legend:** `PENDING` · `OPEN` · `LOCKED` · `WAIVED`  
**pytest (2026-09-29 auditor pass):** `69 passed` in `tests/test_stage4.py`

| Finding ID | Severity | Rule (short) | Regression test name | Status |
|---|---|---|---|---|
| S4-B01 | BLOCKER | No random frame/touch/chance splits | `test_stage4_no_random_group_split` | LOCKED |
| S4-B02 | BLOCKER | Holdout excluded from fit/weights/calibration | `test_stage4_logo_holdout_isolation` | LOCKED |
| S4-B03 | BLOCKER | Fold seeds match freeze | `test_stage4_fold_seeds_match_freeze` | LOCKED |
| S4-M01 | MAJOR | Inner CV excludes outer holdout | `test_stage4_inner_cv_excludes_outer_holdout` | LOCKED |
| S4-B04 | BLOCKER | Passability on release ∩ completion_eligible | `test_stage4_passability_uses_release_frames` | LOCKED |
| S4-B05 | BLOCKER | Release-label yield ≥90% | `test_stage4_release_label_yield` | LOCKED |
| S4-M02 | MAJOR | 5 Hz series separate from release training | `test_stage4_model_5hz_series_separate_from_release` | LOCKED |
| S4-N01 | MINOR | sampling_role provenance | `test_stage4_sampling_role_values` | LOCKED |
| S4-B06 | BLOCKER | No failed completion training rows | `test_stage4_no_failed_completion_labels` | LOCKED |
| S4-B07 | BLOCKER | Interceptor never training target | `test_stage4_interceptor_never_target` | LOCKED |
| S4-B08 | BLOCKER | GATE_COLUMNS on Stage 4 outputs | `test_stage4_gate_columns_present` | LOCKED |
| S4-B09 | BLOCKER | No fabricated audit / audited claims | `test_stage4_no_fabricated_audit_claims` | LOCKED |
| S4-B10 | BLOCKER | No failed-precision proxy citation | `test_stage4_no_failed_precision_claim` | LOCKED |
| S4-M03 | MAJOR | Named-failed exclusion sensitivity | `test_stage4_named_failed_sensitivity` | LOCKED |
| S4-B11 | BLOCKER | OOS q predictions present | `test_stage4_q_oos_predictions_present` | LOCKED |
| S4-M04 | MAJOR | Ablation ladder reported | `test_stage4_passability_ablation_ladder` | LOCKED |
| S4-B12 | BLOCKER | Imbalance weights fold-internal | `test_stage4_imbalance_weights_fold_internal` | LOCKED |
| S4-M05 | MAJOR | Held-out passability metrics | `test_stage4_passability_heldout_metrics` | LOCKED |
| S4-B13 | BLOCKER | Calibration slope kill reported | `test_stage4_calibration_slope_reported` | LOCKED |
| S4-M06 | MAJOR | Recalibration fold-internal | `test_stage4_recalibration_fold_internal` | LOCKED |
| S4-B14 | BLOCKER | V_catch no future features | `test_stage4_vcatch_no_future_features` | LOCKED |
| S4-B15 | BLOCKER | Outcome not from ptsScored | `test_stage4_outcome_not_from_ptsScored` | LOCKED |
| S4-B16 | BLOCKER | shotQuality not a feature | `test_stage4_shotQuality_not_feature` | LOCKED |
| S4-M07 | MAJOR | Secondary outcomes documented | `test_stage4_secondary_outcomes_documented` | LOCKED |
| S4-B17 | BLOCKER | V_keep causal matched no-pass | `test_stage4_vkeep_causal` | LOCKED |
| S4-B18 | BLOCKER | V_fail primary zero | `test_stage4_vfail_primary_zero` | LOCKED |
| S4-B19 | BLOCKER | Components not collapsed to NOV | `test_stage4_components_not_collapsed_to_nov` | LOCKED |
| S4-M08 | MAJOR | V_catch baseline ladder | `test_stage4_vcatch_baseline_ladder` | LOCKED |
| S4-M09 | MAJOR | Choice model metrics | `test_stage4_choice_model_metrics` | LOCKED |
| S4-B20 | BLOCKER | Forbidden optimality language | `test_stage4_forbidden_claim_language` | LOCKED |
| S4-B21 | BLOCKER | Trajectory rule unchanged | `test_stage4_trajectory_rule_unchanged` | LOCKED |
| S4-B22 | BLOCKER | Support rejection masked | `test_stage4_support_rejection_masked` | LOCKED |
| S4-M10 | MAJOR | Null velocity stays missing | `test_stage4_null_velocity_is_missing` | LOCKED |
| S4-N02 | MINOR | Both trajectory families | `test_stage4_both_trajectory_families` | LOCKED |
| S4-B23 | BLOCKER | No season aggregate covariates | `test_stage4_no_season_aggregates` | LOCKED |
| S4-M11 | MAJOR | Primary population filter | `test_stage4_primary_population_filter` | LOCKED |
| S4-N03 | MINOR | Inherited exclusions | `test_stage4_exclusions_inherited` | LOCKED |
| S4-B24 | BLOCKER | Pipeline + tests exist | `test_stage4_pipeline_entrypoint_exists` | LOCKED |
| S4-B25 | BLOCKER | Output manifest complete | `test_stage4_output_manifest_complete` | LOCKED |
| S4-B26 | BLOCKER | Regression lock covers blockers | `test_stage4_regression_lock_covers_blockers` | LOCKED |
| S4-M12 | MAJOR | Model columns not all NaN | `test_stage4_model_columns_not_all_nan` | LOCKED |
| S4-N04 | MINOR | Verification artifact exists | `test_stage4_verification_artifact_exists` | LOCKED |
| S4-B27 | BLOCKER | All ten LOGO folds present | `test_stage4_all_ten_logo_folds_present` | LOCKED |
| S4-B28 | BLOCKER | Handoff matches coverage | `test_stage4_handoff_matches_coverage` | LOCKED |
| S4-M13 | MAJOR | GAM→Spline substitution documented | `test_stage4_gam_substitution_documented` | PENDING |

## How to update

1. When a live audit finding fires, set Status → `OPEN` and point Evidence at `file:line` in `STAGE4_ADVERSARIAL_AUDIT.md`.
2. When builder adds the named `test_*` and it passes under pytest, set Status → `LOCKED`.
3. Do not mark Stage 4 **SAFE TO ACCEPT** while any BLOCKER row is not `LOCKED`.
