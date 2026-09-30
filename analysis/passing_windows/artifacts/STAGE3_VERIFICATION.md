# Stage 3 verification checklist

**Freeze ID:** `passing_windows_stage0_20250924`  
**Status:** PASS

## Counts

| Metric | Value |
|---|---:|
| Candidate rows | 190348 |
| Decision frames | 47587 |
| Candidates/frame (mode) | 4 |
| % rejected (support) | 12.01 |
| % missing matchup | 6.26 |
| Completion-label-eligible rows | 17588 |

## Automated checks

| Check | Result |
|---|---|
| `four_candidates_per_frame` | PASS mode=4; n_bad=0 |
| `explicit_trajectory_choice` | PASS rule_ok=True; both_families=True |
| `null_velocity_is_missing` | PASS receiver_vx NaN wherever velocity_ok is False |
| `missing_matchup_is_missing` | PASS target_defender_recovery_s NaN when matchup unmatched |
| `fold_safe_flight_time` | PASS n_folds=10 |
| `narrow_path_label_join` | PASS completion_labels=4530; failed_eligible=0 |
| `no_interceptor_as_target` | PASS clash_rows=0 |
| `support_rejection_flag` | PASS pct=12.01 |
| `model_5hz_stride` | PASS n_diffs_checked=40843 |
| `dual_sampling_flags` | PASS n_5hz_frames=44001; n_release_frames=4615 |
| `release_label_coverage` | PASS 4397/4530 completion labels (in reconstructed games) have release feature rows |
| `full_gate_columns` | PASS present=10/10; missing=[] |

## Human sign-off

| Item | Status |
|---|---|
| Four candidates per primary-eligible frame | ☐ |
| Causal velocity only (no future frames) | ☐ |
| Explicit trajectory choice (not max-score) | ☐ |
| Fold-safe flight time (no held-out leakage) | ☐ |
| Narrow-path eligibility join | ☐ |
| No interceptor-as-target | ☐ |
