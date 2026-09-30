# Stage 2 failed-pass target inference report

**Freeze ID:** `passing_windows_stage0_20250924`
**Stage:** 2 — resolve failed-pass targets with explicit uncertainty
**Design:** leave-one-game-out; every fold fits flight time, likelihood scales,
and decision thresholds on nine games and scores only the tenth.
**Acceptance mode:** `narrow_path` (lock `passing_windows_stage2_narrow_path_20260924`, `configs/stage2_narrow_path.yaml`)
**Human audit:** `deferred_waived` — no blinded human review of
failed-pass targets exists. The reviewer columns of the audit sheet are blank and
stay blank; nothing in this report is a human judgement.

## 1. Cohorts

`complete` is tri-state in the raw feed and the three classes carry different evidence.

| Pass class | N | Enters target inference |
|---|---:|---|
| `complete` (usable for validation) | 4530 | as a known-receiver control |
| `incomplete_turnover` (failed) | 89 | **yes — the Stage 2 population** |
| `unknown_outcome` (`complete` null) | 68 | no: no end frame, receiver, or interceptor |
| completed passes excluded from control | 8 | receiver outside the four-candidate set |

`toReceiverId` is the interceptor. It is never a candidate and never a target:
candidates come from the lineup containing the passer, so an opponent cannot appear.

## 2. Headline metrics

| Metric | Value |
|---|---:|
| Failed passes scored | 89 |
| Auto-accepted | 60 (67.4%) |
| Ambiguous (audit sheet) | 27 (30.3%) |
| Unresolved, no usable evidence | 2 (2.2%) |
| Unresolved before human audit (ambiguous + no evidence) | 29 (32.6%) |
| Held-out completed-pass top-1 agreement (diagnostic) | 0.819 (n=4530) |
| Held-out completed-pass auto-accept precision (diagnostic) | 0.968 |
| Held-out completed-pass auto-accept rate (diagnostic) | 0.726 |

The last three rows are measured on **completed** passes, where the receiver is known.
They are diagnostics of the scoring machinery. They are **not** failed-pass precision
and may not be cited as evidence that any inferred failed target is correct; see
section 3.1.

## 3. Kill criterion: unresolved failed-pass targets > 10%

- Frozen budget: at most **8** of 89 failed passes.
- Strict unresolved (no candidate set, no ball flight, or no tracking): **2** = 2.2% — within budget.
- Pre-audit unresolved (everything without an accepted label): **29** = 32.6% — BREACH.

Both numbers are reported because they answer different questions. The strict figure is
the share where the method has nothing to work with. The pre-audit figure is the honest
share while the audit sheet is still blank, and it is the one to read against the kill line
until a reviewer fills it in.

### Narrow path — ACTIVE and binding

Triggers recomputed from this run: `pre_audit_unresolved_rate_exceeds_frozen_threshold`, `human_audit_status_is_not_complete`.

Stage 0's `on_inadequate_reliability` clause has fired, so the narrow path is the
Stage 2 acceptance mode rather than a caveat. It is locked in
`configs/stage2_narrow_path.yaml` and enforced in code by
`src/passing_windows/narrow_path.py`:

1. Failed passes are retained for **touch-level turnover and pass-attempt
   features only**. No failed pass is a named-receiver completion label.
2. Receiver-specific completion labels come from **completed passes with a known
   receiver, and nothing else**. That is the default Stage 3+ training set.
3. Auto-accepted inferred failures remain available as an *optional* named target
   (`usable_as_named_failed_target`) but are excluded from
   `receiver_specific_eligible`. A downstream stage may opt in only if it also
   publishes the same result with them excluded.
4. Ambiguous and unresolved failures may never be attributed to a named receiver.
5. The interceptor is never substituted for the target under any path.

Machine-readable gate columns are written to every Stage 2 pass table and
consolidated in `tables/stage2_label_eligibility.parquet` (4619 passes):

| Gate | Passes |
|---|---:|
| `receiver_specific_eligible` (Stage 3+ default) | 4530 |
| `usable_as_receiver_completion_label` | 4530 |
| `usable_as_named_failed_target` (opt-in, sensitivity required) | 60 |
| failed passes restricted to touch level | 89 |

Reliability mix: `{'known_receiver': 4530, 'inferred_provisional_unaudited': 60, 'ambiguous_unreviewed': 27, 'no_evidence': 2}`.

### 3.1 Why the 0.968 figure does not close this gate

The auto-accept precision above is measured on held-out **completed** passes. A
completed pass reaches its receiver and is geometrically easier than an intercepted
one, so that number is an optimistic upper bound on failed-pass precision, not an
estimate of it. The project's own diagnostic
`failed_residuals_within_completed_support` fails: a material share of accepted
failures sit beyond the completed-correct residual support, which is direct evidence
that the two populations differ. The completed-pass proxy is therefore recorded as
**diagnostic only** in the lock, and citing it as failed-pass precision is a
forbidden use.

### 3.2 Human audit: DEFERRED / WAIVED

The frozen protocol required a blinded human review of every ambiguous failure, a
random auto-accepted sample, and completed controls, with the auto-accept margin
re-selected against those labels. That review has **not** been performed. The
reviewer columns in `artifacts/stage2_ambiguous_audit.csv` are blank and the
pipeline refuses to run if anything fills them while the audit is deferred, because
an invented reviewer decision would be indistinguishable downstream from a real one.
Stage 2 is accepted under the narrow path *instead of* that gate, not as if it had
passed it.

## 4. Per-fold thresholds and held-out agreement

Thresholds are **provisional and selected under a substituted protocol**. Stage 0 froze
the procedure as `failed_pass_auto_accept_geometric_margin: TBD_TRAINING_FOLD`, chosen
inside training folds to maximise *audit* agreement. No audit labels exist, so training
folds' completed-pass receiver labels were used instead at a target precision of
0.95. That is a documented protocol substitution, not the
frozen rule, and it is the second reason the narrow-path lock is binding: the thresholds
below are not qualified to license receiver-specific failed-pass claims.

| fold_id             |   p_min |   prob_margin |   geom_margin_ft |   train_precision | met_target   |   scale_sigma_ball_ft |   scale_sigma_pos_ft |   scale_sigma_flight_s |   n_failed_heldout |   heldout_completed_top1_accuracy |   heldout_completed_auto_accept_precision |   failed_auto_accept |   failed_ambiguous |   failed_unresolved |
|:--------------------|--------:|--------------:|-----------------:|------------------:|:-------------|----------------------:|---------------------:|-----------------------:|-------------------:|----------------------------------:|------------------------------------------:|---------------------:|-------------------:|--------------------:|
| fold_holdout_114243 |     0.5 |           0.1 |                1 |            0.9705 | True         |                1.5095 |                  0.5 |                 0.0654 |                  8 |                            0.7949 |                                    0.9446 |                    6 |                  2 |                   0 |
| fold_holdout_114234 |     0.5 |           0.1 |                1 |            0.9656 | True         |                1.5157 |                  0.5 |                 0.0651 |                 14 |                            0.8194 |                                    0.9832 |                   10 |                  4 |                   0 |
| fold_holdout_114169 |     0.5 |           0.1 |                1 |            0.9669 | True         |                1.5086 |                  0.5 |                 0.0673 |                  6 |                            0.8588 |                                    0.9731 |                    5 |                  0 |                   1 |
| fold_holdout_114099 |     0.5 |           0.1 |                1 |            0.9679 | True         |                1.514  |                  0.5 |                 0.0672 |                 12 |                            0.842  |                                    0.9641 |                    8 |                  3 |                   1 |
| fold_holdout_114086 |     0.5 |           0.1 |                1 |            0.9671 | True         |                1.4779 |                  0.5 |                 0.0665 |                 12 |                            0.8471 |                                    0.9832 |                    5 |                  7 |                   0 |
| fold_holdout_178442 |     0.5 |           0.1 |                1 |            0.9675 | True         |                1.507  |                  0.5 |                 0.0649 |                  9 |                            0.8186 |                                    0.9614 |                    6 |                  3 |                   0 |
| fold_holdout_179612 |     0.5 |           0.1 |                1 |            0.9665 | True         |                1.5069 |                  0.5 |                 0.0659 |                  6 |                            0.8166 |                                    0.9755 |                    5 |                  1 |                   0 |
| fold_holdout_184439 |     0.5 |           0.1 |                1 |            0.9684 | True         |                1.5224 |                  0.5 |                 0.0649 |                  5 |                            0.7705 |                                    0.955  |                    3 |                  2 |                   0 |
| fold_holdout_188630 |     0.5 |           0.1 |                1 |            0.9663 | True         |                1.5166 |                  0.5 |                 0.066  |                  8 |                            0.7946 |                                    0.9784 |                    7 |                  1 |                   0 |
| fold_holdout_191313 |     0.5 |           0.1 |                1 |            0.9683 | True         |                1.4843 |                  0.5 |                 0.0661 |                  9 |                            0.8305 |                                    0.9596 |                    5 |                  4 |                   0 |

## 5. Evidence channels

| Channel | Residual | Availability on failed passes |
|---|---|---:|
| Lane alignment | perpendicular miss from the observed ball bearing to the projected catch point, scaled by a tolerance that grows with distance and shrinks with the ball chord | 87/89 |
| Flight-time compatibility | clean flight duration vs ball travel time to the candidate | 77/89 |
| Direction sign | candidates behind the release bearing are penalised | 89/89 |

Both residuals use a Student-t likelihood (nu = 3). A Gaussian fitted to the sharp peak of
known-receiver residuals treats the genuine tail — a pass tipped at release, a receiver who
cut after the ball left — as impossible, and then returns certainty on the passes it gets
wrong. The heavy tail is what makes section 6's calibration hold.

The flight-time channel is asymmetric for failures: an intercepted ball stops short, so a
candidate farther away than the ball reached is expected and only weakly penalised, while a
candidate the ball had already flown past is evidence against.

## 6. Is `p_top` a probability?

Held-out completed passes only, so the receiver is known and the fold never saw the game.

| p_top_bin     |    n |   mean_p_top |   empirical_accuracy |   calibration_gap |
|:--------------|-----:|-------------:|---------------------:|------------------:|
| (-0.001, 0.4] |  835 |        0.251 |                0.261 |            -0.01  |
| (0.4, 0.6]    |   69 |        0.534 |                0.391 |             0.142 |
| (0.6, 0.8]    |  160 |        0.725 |                0.7   |             0.025 |
| (0.8, 0.9]    |  184 |        0.856 |                0.837 |             0.02  |
| (0.9, 0.95]   |  226 |        0.929 |                0.912 |             0.018 |
| (0.95, 0.99]  |  699 |        0.976 |                0.951 |             0.024 |
| (0.99, 1.0]   | 2357 |        0.998 |                0.988 |             0.01  |

## 7. Automated geometry self-consistency

Invariants must hold for the output to be valid; diagnostics describe the data and are
reported rather than enforced.

| Check | Severity | Result | Detail |
|---|---|---|---|
| `probability_simplex` | invariant | **PASS** | `{'n_passes': 4619, 'max_abs_sum_error': 4.440892098500626e-16, 'n_negative': 0, 'n_nan': 0}` |
| `interceptor_never_target` | invariant | **PASS** | `{'n_interceptor_in_candidate_set': 0, 'n_interceptor_assigned_as_target': 0}` |
| `target_not_passer` | invariant | **PASS** | `{'n_passer_as_candidate': 0}` |
| `assignment_in_candidate_set` | invariant | **PASS** | `{'n_checked': 60, 'n_outside_candidate_set': 0}` |
| `unresolved_has_no_label` | invariant | **PASS** | `{'n_labelled_non_accepted': 0}` |
| `mirror_invariance` | invariant | **PASS** | `{'max_abs_probability_difference': 0.0, 'probabilities': [0.472463, 9e-06, 0.527517, 1.1e-05]}` |
| `flight_time_monotonic` | invariant | **PASS** | `{'n_bins': 18, 'n_train': 3364, 'distance_support_max_ft': 37.95235558678171, 't_at_5ft': 0.2, 't_at_15ft': 0.4, 't_at_30ft': 0.72, 'min_diff': 0.0}` |
| `release_point_matches_passer` | diagnostic | **PASS** | `{'n': 4507, 'median_ft': 3.7769880385514862, 'p95_ft': 7.3613429159982555, 'share_within_6ft': 0.8850676725094297}` |
| `failed_residuals_within_completed_support` | diagnostic | **FAIL** | `{'n_accepted': 60, 'n_reference': 3711, 'top_angle_deg_reference_p99': 47.554562400609235, 'top_angle_deg_share_beyond_reference_p99': 0.15, 'top_lateral_ft_reference_p99': 8.09566763766684, 'top_lateral_ft_share_beyond_reference_p99': 0.11666666666666667}` |

## 8. Sensitivity: excluding all inferred failures

Stub for Stage 4. Dropping every inferred failure removes the only failure evidence
attributable to a receiver, which forces receiver-specific completion rates to 1.0 by
construction — the reason the frozen narrow path restricts that calibration rather than
silently accepting it.

| Quantity | Value |
|---|---:|
| Receivers with at least one completed pass | 200 |
| Receivers gaining an inferred failure | 51 |
| Attempts added by inferred failures | 60 |
| Max single-receiver share of attempts that is inferred | 0.250 |

Full table: `tables/stage2_failure_sensitivity.parquet`.

## 9. Manual audit export

- `artifacts/stage2_ambiguous_audit.csv`: 596 candidate rows across 149 passes.
- Contains every failed pass (auto-accepted, ambiguous, and unresolved) plus a seeded
  sample of 6 completed passes per game as a blind control.
- One row per candidate, carrying release coordinates, projected catch point, ball release
  point and bearing, and the per-channel residuals — the tabular court-frame sync that
  stands in until the Stage 9 interactive viewer exists.
- Reviewer columns (`reviewer_is_intended_target`, `reviewer_decision`, `reviewer_confidence`, `reviewer_reason`) are **blank and must stay blank**
  while `human_audit_status = deferred_waived`. The pipeline asserts this
  and exits non-zero if any value appears, so a fabricated reviewer decision cannot enter
  the study by accident. Mark `reviewer_is_intended_target` on the candidate row believed
  to be the target, then record `reviewer_decision`, `reviewer_confidence`, and
  `reviewer_reason` once per pass — only if a real blinded review is actually performed.
- The sheet also carries tabular release geometry rather than synchronized court-frame
  playback. That is another reason the audit is recorded as deferred rather than merely
  outstanding: the review material the frozen protocol asked for does not exist yet.

## 10. Known limitations

1. Completed passes are a **proxy** for audit labels and are recorded in the lock as
   diagnostic only. A completed pass reaches its receiver, so its geometry is easier than
   an intercepted pass; the held-out precision reported here is an optimistic bound on
   failed-pass precision and may not be cited as failed-pass precision.
2. Candidate motion is projected from release-time velocity only. Cuts that start after
   release are invisible to the projection by design (no future leakage).
3. A deflection ends the clean flight segment, so heavily deflected passes contribute a
   short bearing and a weak flight-time channel.
4. `matchup_matched == False` states and null velocities are carried as missing, never as
   zero; a candidate with no tracking receives no likelihood rather than a default share.
5. **No failed-pass target in this study has been verified by a human.** Every
   `inferred_targetId` is an unaudited geometric inference, which is exactly what the
   narrow-path gate columns encode.

