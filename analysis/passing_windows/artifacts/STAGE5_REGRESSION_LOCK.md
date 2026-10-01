# Stage 5 regression lock

| Finding ID | Summary | Regression test |
|---|---|---|
| S5-B00 | Pipeline / package exist | `test_stage5_pipeline_entrypoint_exists` |
| S5-B01 | Consumes Stage 4 1:1 keys / counts | `test_stage5_consumes_stage4_predictions_1to1` |
| S5-B02 | No Stage 4 component refit | `test_stage5_no_component_model_refit` |
| S5-B03 | 5 Hz series only | `test_stage5_windows_use_model_5hz_series_only` |
| S5-B04 | Off-lattice release use labels | `test_stage5_use_labels_allow_off_lattice_release` |
| S5-B05 | Causal smoothing | `test_stage5_smoothing_is_causal` |
| S5-B06 | LOGO threshold isolation | `test_stage5_thresholds_logo_holdout_isolation` |
| S5-B07 | Hysteresis open/close | `test_stage5_hysteresis_separate_open_close` |
| S5-B08 | Persistence ≥0.20s | `test_stage5_persistence_ge_0_20s_wallclock` |
| S5-B09 | Late delta fold-internal | `test_stage5_late_delta_fold_internal` |
| S5-B10 | Rejected cannot open | `test_stage5_rejected_cannot_open_windows` |
| S5-B11 | Used = known-receiver | `test_stage5_used_requires_known_receiver` |
| S5-B12 | GATE columns present | `test_stage5_gate_columns_present` |
| S5-B13 | Interceptor never use target | `test_stage5_interceptor_never_use_target` |
| S5-B14 | Forbidden claim language | `test_stage5_forbidden_claim_language` |
| S5-B15 | Open = q ∧ NOV | `test_stage5_open_requires_q_and_nov` |
| S5-B16 | Labels mutually exclusive | `test_stage5_labels_mutually_exclusive` |
| S5-B17 | Continuous window fields | `test_stage5_continuous_window_fields_present` |
| S5-B18 | Option-set features complete | `test_stage5_option_set_features_complete` |
| S5-B19 | No random group splits | `test_stage5_no_random_group_split` |
| S5-B20 | No season-level aggregate covariates | `test_stage5_no_season_aggregates` |
| S5-B21 | All ten LOGO folds | `test_stage5_all_ten_logo_folds_present` |
| S5-B22 | Output manifest | `test_stage5_output_manifest_complete` |
| S5-B23 | Regression lock covers blockers | `test_stage5_regression_lock_covers_blockers` |
| S5-M01 | sampling_role retained | `test_stage5_sampling_role_retained` |
| S5-M04 | Threshold robustness grid | `test_stage5_threshold_robustness_grid_emitted` |
| S5-M06 | Support rejection rate reported | `test_stage5_support_rejection_rate_reported` |
| S5-M08 | Use timing fields | `test_stage5_use_timing_fields` |
| S5-M09 | Existence prob deferral | `test_stage5_existence_prob_column_or_explicit_deferral` |
| S5-M11 | Option-set excludes rejected | `test_stage5_option_set_excludes_rejected` |
