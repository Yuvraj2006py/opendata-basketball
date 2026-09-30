# Stage 2 independent narrow-path re-audit

**Re-audit date:** 2026-09-24  
**Freeze ID:** `passing_windows_stage0_20250924`  
**Lock ID:** `passing_windows_stage2_narrow_path_20260924`  
**Verdict:** **CLEAR** (under narrow path only)  
**Stage 3:** **CONDITIONAL GO** — narrow-path constraints mandatory (see below)

## Scope and method

Independent re-auditor. Did not trust the prior `STAGE2_AUDIT_REPORT.md`
remediation narrative. Did not invent human reviewer labels. Did not start
Stage 3.

Empirically:

- Loaded and queried `tables/stage2_label_eligibility.parquet`,
  `stage2_failed_pass_assignments.parquet`,
  `stage2_completed_pass_validation.parquet`,
  `stage2_target_probabilities.parquet`
- Recomputed unresolved rates from live assignment decisions
- Re-evaluated `evaluate_narrow_path` triggers from live unresolved figures
- Ran `assert_narrow_path_gates`, `assert_interceptor_never_target`,
  `assert_reviewer_columns_blank`, `assert_no_failed_precision_claim`,
  `receiver_completion_training_labels` against disk artifacts
- Independently SHA-256 / byte-size verified all **15** strict manifest entries
- Ran `pytest tests/test_stage2.py`: **73 passed**

## CLEAR checklist (empirically verified)

| # | Criterion | Result | Evidence |
|---|---|---|---|
| 1 | `narrow_path_active` frozen / present in outputs | **CLEAR** | Lock YAML `target_inference_mode: narrow_path`; state JSON `narrow_path_active: true`; all rows on eligibility / failed / completed / probs carry `narrow_path_active=True` and `target_inference_mode=narrow_path`. Triggers recomputed: `pre_audit_unresolved_rate_exceeds_frozen_threshold`, `human_audit_status_is_not_complete`. |
| 2 | Gates + eligibility table | **CLEAR** | Eligibility has all 10 `GATE_COLUMNS`. Failed and completed pass tables have all 10. Counts: 4619 passes (4530 completed + 89 failed); `usable_as_receiver_completion_label` = 4530; `usable_as_named_failed_target` = 60; `receiver_specific_eligible` = 4530; equals completion flag under narrow path. Reliability mix: known_receiver 4530 / inferred_provisional_unaudited 60 / ambiguous_unreviewed 27 / no_evidence 2. |
| 3 | 0 failed passes usable as receiver completion labels by default | **CLEAR** | Failed cohort: `usable_as_receiver_completion_label` = 0, `receiver_specific_eligible` = 0 (eligibility and assignments). Same zeros on probability rows for failed `passId`s. |
| 4 | Completed-pass proxy = diagnostic only | **CLEAR** | Lock + state `completed_pass_proxy_role: diagnostic_only`. Report and verification label 0.819 / 0.968 as diagnostic. `assert_no_failed_precision_claim` passes on both generated reports. |
| 5 | Interceptor never intended target | **CLEAR** | 0 candidate rows with `candidateId == interceptorId`; 0 assignments with `inferred_targetId == interceptorId`. Fatal assertion path exercised by tests and live `assert_narrow_path_gates`. |
| 6 | Human audit columns blank + deferred/waived documented | **CLEAR** | Audit CSV 596 rows; all four reviewer columns have **0** nonblank values. `human_audit_status` unique value on gated tables = `deferred_waived`. Lock documents DEFERRED/WAIVED; no fabricated labels invented by this re-audit. |
| 7 | Narrow-path tests pass when run | **CLEAR** | `pytest tests/test_stage2.py` → **73 passed** (includes lock, gate, promotion-fatal, blank-reviewer, real-table, and proxy-phrasing suites). |
| 8 | Manifest / state consistent | **CLEAR** | 15/15 strict files match recorded SHA-256 and bytes. Manifest meta matches lock: `narrow_path_lock_id`, `target_inference_mode=narrow_path`, `human_audit_status=deferred_waived`. State counts match eligibility (4530 / 89 / 60). |
| 9 | Stage 3 constraints enforceable from tables | **CLEAR** | `receiver_completion_training_labels(eligibility)` returns exactly 4530 completed known-receiver rows; failed-only eligibility returns 0. Named failed targets (60) require sensitivity check and stay out of `receiver_specific_eligible`. Live `assert_narrow_path_gates` passes. |

## Bypass probe (Stage 3 silent training)

**Question:** Could Stage 3 silently train on inferred failed targets?

| Path | Result |
|---|---|
| Default filter `receiver_specific_eligible == True` | Excludes all 89 failures (0 eligible) |
| Helper `receiver_completion_training_labels` | Returns 0 rows from failed-only eligibility; re-asserts known-receiver |
| Reading `stage2_target_probabilities.parquet` alone | Failed passes still have `usable_as_receiver_completion_label=False` and `receiver_specific_eligible=False` |
| Opt-in `usable_as_named_failed_target` | 60 auto-accept rows flagged; `named_failed_target_requires_sensitivity_check=True`; **not** default training |

Gates are **not** bypassable by default. Silent training on inferred failed targets
requires an explicit consumer that ignores both eligibility columns and the
documented Stage 3 helper — that is a Stage 3 protocol violation, not a Stage 2
gate failure.

## Non-blocking observation

`configs/stage2_narrow_path.yaml` lists
`tables/stage2_target_probabilities.parquet` under `gate_columns.written_to`
(full 10-column set). The pipeline intentionally broadcasts a **candidate subset**
via `CANDIDATE_GATE_COLUMNS` plus mode/audit flags (7 of 10). Missing on probs
only: `named_failed_target_requires_sensitivity_check`,
`usable_for_touch_level_turnover`, `label_gate_reason`.

The Stage 3 join target is `stage2_label_eligibility.parquet` (complete). Critical
default filters are present and correct on the probability table. Tests and
`assert_narrow_path_gates` require full gate columns on pass-level tables, not
probs. This is a packaging / yaml-doc mismatch, **not** a Stage 3 silent-training
blocker. Optional cleanup: align yaml `written_to` with the intentional subset,
or broadcast the three missing columns — not required to CLEAR this re-audit.

## Kill criterion status (unchanged)

| Trigger | Measured | Status |
|---|---|---|
| Strict unresolved | 2/89 = 2.25% | within 10% budget |
| Pre-audit unresolved | 29/89 = 32.6% | **BREACH** → narrow path |
| Audit reliability | unmeasured (no human review) | **BREACH** → narrow path |

Human audit remains **DEFERRED / WAIVED**. The frozen human-audit gate was not
passed and is not claimed here.

## Stage 3 go / no-go

**CONDITIONAL GO under `target_inference_mode: narrow_path` only.**

### Mandatory constraints (enforceable from tables)

1. Join `tables/stage2_label_eligibility.parquet`.
2. Obtain completion labels only via
   `narrow_path.receiver_completion_training_labels` (or equivalent filter on
   `receiver_specific_eligible == True`).
3. Train receiver-specific completion labels on **completed known-receiver**
   passes only (n=4530 under this freeze).
4. Treat all 89 failed passes as **touch-level turnover / pass-attempt** evidence
   only — never as default named-receiver completion labels.
5. Never use ambiguous (27) or unresolved (2) failures as named-receiver labels
   under any weighting.
6. Carry `target_reliability_status` through any receiver-level output.
7. If opting into the 60 auto-accepted inferred failures via
   `usable_as_named_failed_target`, publish the exclusion sensitivity in the same
   artifact and label them provisional/unaudited everywhere.
8. Never cite the completed-pass proxy (0.968 / 0.819) as failed-pass precision,
   and never describe Stage 2 failed targets as audited or validated.
9. Do not fill reviewer columns; do not invent human labels.

### NO-GO conditions

- Training completion labels on failed-pass `inferred_targetId` without the
  eligibility gate / sensitivity protocol above.
- Treating blank audit CSV reviewer fields as completed review.
- Proceeding as if `human_audit_status == complete`.

## Sign-off

**Verdict: CLEAR** under the locked narrow path.  
**Stage 3: CONDITIONAL GO** with the mandatory constraints above.  
Human failed-pass target audit: still **NOT PERFORMED** (deferred/waived).
