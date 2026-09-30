# Stage 3 geometry self-check

**Freeze ID:** `passing_windows_stage0_20250924`  
**Overall:** PASS

These checks mirror the Stage 2 geometry self-check: machine-verified
invariants of the candidate reconstruction, not Stage 4 model fit quality.

## Checks

| Check | Result | Detail |
|---|---|---|
| `four_candidates_per_frame` | PASS | mode=4; n_bad=0 |
| `explicit_trajectory_choice` | PASS | rule_ok=True; both_families=True |
| `null_velocity_is_missing` | PASS | receiver_vx NaN wherever velocity_ok is False |
| `missing_matchup_is_missing` | PASS | target_defender_recovery_s NaN when matchup unmatched |
| `fold_safe_flight_time` | PASS | n_folds=10 |
| `narrow_path_label_join` | PASS | completion_labels=4530; failed_eligible=0 |
| `no_interceptor_as_target` | PASS | clash_rows=0 |
| `support_rejection_flag` | PASS | pct=12.01 |
| `model_5hz_stride` | PASS | n_diffs_checked=40843 |
| `dual_sampling_flags` | PASS | n_5hz_frames=44001; n_release_frames=4615 |
| `release_label_coverage` | PASS | 4397/4530 completion labels (in reconstructed games) have release feature rows |
| `full_gate_columns` | PASS | present=10/10; missing=[] |

## Population snapshot

- Candidate rows: **190348**
- Decision frames: **47587**
- Candidates/frame mode: **4**
- Rejected outside support: **12.01%**
- Missing matchup: **6.26%**
- Trajectory mix: `{'lead': 189215, 'direct': 1133}`

## Semantics locked here

- Trajectory rule: `lead_if_velocity_ok_else_direct` (explicit; never score-maximizing).
- Both `direct_*` and `lead_*` feature families are retained.
- Missing velocities stay NaN (not zero); unmatched matchups leave recovery NaN.
- Fold flight-time models never train on the held-out game.
- Completion labels only via narrow-path `receiver_specific_eligible`.
