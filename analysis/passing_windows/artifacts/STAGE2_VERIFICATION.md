# Stage 2 USER VERIFICATION CHECKLIST

**Freeze ID:** `passing_windows_stage0_20250924`
**Narrow-path lock:** `passing_windows_stage2_narrow_path_20260924` (`configs/stage2_narrow_path.yaml`)
**Acceptance mode:** `narrow_path`
**Human audit:** `deferred_waived` — NOT complete, NOT simulated
**Purpose:** approve failed-pass target inference *under the narrow path* before Stage 3.

This checklist is **not** a claim that the frozen human-audit gate was passed. It was
not. Stage 2 is offered for acceptance under Stage 0's `on_inadequate_reliability`
fallback, which the run below triggers and which the code now enforces. See
[`STAGE2_NARROW_PATH_LOCK.md`](STAGE2_NARROW_PATH_LOCK.md) for the decision and
[`STAGE2_AUDIT_REPORT.md`](STAGE2_AUDIT_REPORT.md) for the adversarial audit that
forced it.

Mark each item **APPROVE** / **REJECT** / **CHANGE REQUEST**.

## A. Sample sizes

| Metric | Value |
|---|---:|
| Failed passes (`complete == False`) | 89 |
| `unknown_outcome` passes held out of inference | 68 |
| Completed passes used as known-receiver controls | 4530 |
| Auto-accepted failed targets | 60 |
| Ambiguous failed targets | 27 |
| Unresolved (no evidence) failed targets | 2 |

## B. Frozen rules

| # | Rule | Pipeline result | Verify |
|---|---|---|---|
| B1 | `toReceiverId` never used as intended target | PASS | ☐ |
| B2 | Probabilities form a simplex per pass | PASS | ☐ |
| B3 | Passer never a candidate for their own pass | PASS | ☐ |
| B4 | Accepted labels lie inside the candidate set | PASS | ☐ |
| B5 | Non-accepted passes carry no hard label | PASS | ☐ |
| B6 | Mirroring does not change probabilities | PASS | ☐ |
| B7 | Flight-time model monotonic in distance | PASS | ☐ |
| B8 | Ball release anchored on the passer | PASS | ☐ |
| B9 | Tri-state `complete`: unknown-outcome passes excluded from the failed cohort | 68 held out | ☐ |
| B10 | Thresholds selected inside training folds only (leave-one-game-out) | PASS | ☐ |

## C. Kill criteria

| # | Criterion | Value | Verdict | Verify |
|---|---|---:|---|---|
| C1 | Unresolved (strict) > 10% | 2.2% | OK | ☐ |
| C2 | Unresolved pre-audit > 10% | 32.6% | BREACH — narrow path | ☐ |
| C3 | Held-out completed-pass agreement (**diagnostic only**) | 0.819 | — | ☐ |
| C4 | Held-out auto-accept precision (**diagnostic only, NOT failed-pass precision**) | 0.968 | — | ☐ |

C3 and C4 are measured on completed passes. They describe the scoring machinery and
carry no information about whether any inferred failed target is correct.

## D. Narrow-path acceptance criteria (replaces the human-audit gate)

Stage 2 is accepted only if every row below holds. These are machine-checked; the
pipeline exits non-zero on any violation.

| # | Criterion | Pipeline result | Verify |
|---|---|---|---|
| D1 | `target_inference_mode` locked to `narrow_path` | PASS | ☐ |
| D2 | Narrow path active, triggers recomputed from this run | PASS (`pre_audit_unresolved_rate_exceeds_frozen_threshold`, `human_audit_status_is_not_complete`) | ☐ |
| D3 | Gate columns present on every Stage 2 pass table | PASS (10 columns) | ☐ |
| D4 | Receiver-specific eligible = completed passes with known receivers only | 4530 of 4619 | ☐ |
| D5 | No failed pass carries `usable_as_receiver_completion_label` | True | ☐ |
| D6 | Failed passes restricted to touch-level turnover use | 89 | ☐ |
| D7 | Auto-accepted failures opt-in only, sensitivity check required | 60 flagged | ☐ |
| D8 | Interceptor never a target — asserted, not merely reported | PASS | ☐ |
| D9 | Reviewer columns blank; no fabricated reviewer decisions | PASS (`human_audit_status = deferred_waived`) | ☐ |

## E. Human audit status: DEFERRED / WAIVED

| # | Item | Status |
|---|---|---|
| E1 | Blinded human review of ambiguous failures | **NOT PERFORMED** |
| E2 | Blinded review of random auto-accepted sample | **NOT PERFORMED** |
| E3 | Completed-pass control review | **NOT PERFORMED** |
| E4 | Margin re-selected against audit labels | **NOT PERFORMED** (completed-pass proxy used instead) |
| E5 | Synchronized court-frame review material | **NOT BUILT** (tabular sheet only) |
| E6 | Reviewer columns in `stage2_ambiguous_audit.csv` | **BLANK BY DESIGN — do not fill with model output** |

The user waived human checklist sign-off and directed autonomous execution. Waiving the
sign-off does not create the audit, so the audit is recorded as deferred and the narrow
path carries the reliability burden instead. Reopening the lock requires the real review
listed above, a threshold re-selection against those labels, and a new lock revision.

## F. Artifacts present

| # | Artifact | Verify |
|---|---|---|
| F1 | `artifacts/stage2_target_inference_report.md` | ☐ |
| F2 | `artifacts/stage2_ambiguous_audit.csv` (reviewer columns blank) | ☐ |
| F3 | `artifacts/stage2_geometry_selfcheck.md` | ☐ |
| F4 | `artifacts/stage2_fold_thresholds.json` | ☐ |
| F5 | `artifacts/stage2_narrow_path_state.json` | ☐ |
| F6 | `artifacts/STAGE2_NARROW_PATH_LOCK.md` | ☐ |
| F7 | `configs/stage2_narrow_path.yaml` | ☐ |
| F8 | `tables/stage2_label_eligibility.parquet` | ☐ |
| F9 | `artifacts/stage2_output_manifest.json` verifies clean | ☐ |

## Sign-off

| Role | Name | Date | Overall |
|---|---|---|---|
| User / analyst | _(waived: autonomous execution directed by user)_ | | APPROVE / REJECT |
| Human failed-pass target reviewer | _(none — audit deferred)_ | | N/A |

**If APPROVE:** Stage 3 may begin **under the narrow path only**. It must join
`tables/stage2_label_eligibility.parquet`, train completion labels on
`receiver_specific_eligible == True` only, and treat every failed pass as touch-level
turnover evidence. No Stage 3+ output may attribute a failed pass to a named receiver
without opting in via `usable_as_named_failed_target` and publishing the exclusion
sensitivity alongside it.

## G. Fold thresholds actually applied

| fold_id             |   p_min |   prob_margin |   geom_margin_ft |   train_precision | met_target   |
|:--------------------|--------:|--------------:|-----------------:|------------------:|:-------------|
| fold_holdout_114243 |     0.5 |           0.1 |                1 |            0.9705 | True         |
| fold_holdout_114234 |     0.5 |           0.1 |                1 |            0.9656 | True         |
| fold_holdout_114169 |     0.5 |           0.1 |                1 |            0.9669 | True         |
| fold_holdout_114099 |     0.5 |           0.1 |                1 |            0.9679 | True         |
| fold_holdout_114086 |     0.5 |           0.1 |                1 |            0.9671 | True         |
| fold_holdout_178442 |     0.5 |           0.1 |                1 |            0.9675 | True         |
| fold_holdout_179612 |     0.5 |           0.1 |                1 |            0.9665 | True         |
| fold_holdout_184439 |     0.5 |           0.1 |                1 |            0.9684 | True         |
| fold_holdout_188630 |     0.5 |           0.1 |                1 |            0.9663 | True         |
| fold_holdout_191313 |     0.5 |           0.1 |                1 |            0.9683 | True         |

