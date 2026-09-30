# Stage 5 regression lock

| Finding ID | Summary | Regression test |
|---|---|---|
| S5-B03 | 5 Hz series only | `test_stage5_windows_use_model_5hz_series_only` |
| S5-B05 | Causal smoothing | `test_stage5_smoothing_is_causal` |
| S5-B06 | LOGO threshold isolation | `test_stage5_thresholds_logo_holdout_isolation` |
| S5-B07 | Hysteresis open/close | `test_stage5_hysteresis_separate_open_close` |
| S5-B08 | Persistence ≥0.20s | `test_stage5_persistence_ge_0_20s_wallclock` |
| S5-B10 | Rejected cannot open | `test_stage5_rejected_cannot_open_windows` |
| S5-B11 | Used = known-receiver | `test_stage5_used_requires_known_receiver` |
| S5-B15 | Open = q ∧ NOV | `test_stage5_open_requires_q_and_nov` |
