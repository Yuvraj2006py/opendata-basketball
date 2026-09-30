# Stage 2 narrow-path lock

**Lock ID:** `passing_windows_stage2_narrow_path_20260924`
**Freeze ID:** `passing_windows_stage0_20250924`
**Locked:** 2026-09-24
**Machine-readable form:** [`../configs/stage2_narrow_path.yaml`](../configs/stage2_narrow_path.yaml)
**Enforced in:** `src/passing_windows/narrow_path.py`
**Forced by:** [`STAGE2_AUDIT_REPORT.md`](STAGE2_AUDIT_REPORT.md) (BLOCKERS, Stage 3 NO-GO)

## Decision

Stage 2's frozen acceptance gate was a blinded human audit of failed-pass
targets. That audit does not exist and will not exist on this pass. Rather than
manufacture it, Stage 2 is accepted under **Stage 0's own fallback clause**:

> `failed_pass_target_inference.on_inadequate_reliability`: Retain failed passes
> only for touch-level turnover modeling; restrict receiver-specific completion
> calibration to outcomes with reliable targets.

`target_inference_mode` is locked to `narrow_path`. This is no longer a caveat
in a report that a later stage can read past — it is a set of boolean columns on
every Stage 2 pass table, a consolidated gate table, and four fatal assertions
in the pipeline.

## Why the gate fired

Two independent triggers, both recomputed from the live run rather than
hard-coded, so the lock cannot go stale:

| Trigger | Frozen line | Measured | Status |
|---|---|---|---|
| Pre-audit unresolved failed targets (ambiguous + no evidence) | > 10% kills | 29/89 = **32.6%** | **BREACH** |
| Strict unresolved (no usable evidence at all) | > 10% kills | 2/89 = 2.2% | within budget |
| Audit reliability | must be measured and adequate | **not measured** | **BREACH** |

The Stage 0 kill criterion reads `unresolved failed-pass targets exceed 10% OR
audit reliability is poor`. Unmeasured reliability is treated the same as poor
reliability: an unaudited label carries no reliability evidence, so it cannot
clear a reliability gate.

## What we are *not* claiming

**The human audit is DEFERRED / WAIVED, not complete.** The user waived human
checklist sign-off and directed autonomous execution. Waiving a sign-off does
not perform the review, so:

- The reviewer columns of `stage2_ambiguous_audit.csv` are blank and stay blank.
  `assert_reviewer_columns_blank` makes the pipeline exit non-zero if anything
  fills them while `human_audit_status != complete`. A fabricated
  `reviewer_decision` is the single most damaging artifact this stage could
  produce, because a later stage cannot tell it apart from a real one.
- No geometry-versus-review agreement is reported, because none exists.
- The per-fold auto-accept margins were selected against **completed-pass**
  receiver labels, not audit labels. That is a documented substitution of the
  frozen protocol, and it is recorded as such in
  `stage2_fold_thresholds.json` and section 4 of the inference report.
- The audit sheet ships tabular release geometry, not the synchronized
  court-frame playback the frozen protocol asked for. The review material does
  not exist yet either.

**The 0.968 figure is a diagnostic, not failed-pass precision.** It is held-out
auto-accept precision on *completed* passes, where the receiver is known. A
completed pass reaches its receiver and is geometrically easier than an
intercepted one, so the number bounds failed-pass precision from above instead
of estimating it. The project's own diagnostic
`failed_residuals_within_completed_support` already fails — a material share of
accepted failures sit beyond the completed-correct residual support — which is
direct evidence that the two populations are not interchangeable. The lock
records `completed_pass_proxy.role: diagnostic_only` and lists the forbidden
uses; `assert_no_failed_precision_claim` scans generated reports for phrasing
that would smuggle the claim back in.

## What the lock does

### Machine-readable gate columns

Written to `tables/stage2_label_eligibility.parquet` (the join target for every
later stage) and mirrored onto `stage2_failed_pass_assignments.parquet`,
`stage2_completed_pass_validation.parquet`, and
`stage2_target_probabilities.parquet`:

| Column | Meaning |
|---|---|
| `narrow_path_active` | The gate is in force for this run |
| `target_inference_mode` | `narrow_path` |
| `target_reliability_status` | `known_receiver` / `inferred_provisional_unaudited` / `ambiguous_unreviewed` / `no_evidence` |
| `human_audit_status` | `deferred_waived` |
| `usable_as_receiver_completion_label` | True only for completed passes with a known receiver |
| `usable_as_named_failed_target` | Opt-in flag, True only for auto-accepted failures |
| `named_failed_target_requires_sensitivity_check` | True wherever the opt-in flag is True |
| `usable_for_touch_level_turnover` | True for every pass attempt |
| `receiver_specific_eligible` | **The default Stage 3+ filter.** Equals `usable_as_receiver_completion_label` while the narrow path is active |
| `label_gate_reason` | Why this row landed where it did |

### Fatal assertions

The pipeline returns exit code 2 on any of:

1. `interceptor_never_target` — no candidate row and no `inferred_targetId` may
   equal `interceptorId`. This was previously only *reported* by the
   self-consistency suite; it now raises.
2. `completion_labels_have_known_receiver` —
   `usable_as_receiver_completion_label` implies
   `target_reliability_status == known_receiver`, a non-null `true_receiverId`,
   and a row from the completed cohort.
3. `named_failed_targets_are_accepted` — `usable_as_named_failed_target`
   implies `decision == auto_accept` and a non-null `inferred_targetId`.
4. `gate_columns_present` — every gated table carries all ten columns.
5. `no_fabricated_reviewer_values` — reviewer columns blank while the audit is
   deferred.

### Row-level effect on this run

| Cohort | N | Default Stage 3 use |
|---|---:|---|
| Completed, known receiver | 4530 | receiver-specific completion labels |
| Failed, auto-accepted | 60 | touch-level turnover; named target opt-in with sensitivity check |
| Failed, ambiguous | 27 | touch-level turnover only |
| Failed, no evidence | 2 | touch-level turnover only |
| `unknown_outcome` | 68 | not a Stage 2 population at all |

## Stage 3 constraints

Stage 3 has **not** been started. When it does start, under this lock it must:

1. Join `tables/stage2_label_eligibility.parquet` and obtain completion labels
   through `narrow_path.receiver_completion_training_labels`, which filters on
   `receiver_specific_eligible` and re-asserts the gate before returning.
2. Train passability completion labels on **completed passes with known
   receivers only**. That is the default and needs no justification; anything
   else does.
3. Treat all 89 failed passes as touch-level turnover and pass-attempt evidence,
   never as evidence about a named receiver.
4. Never use ambiguous or unresolved failures as named-receiver completion
   labels, under any weighting scheme.
5. Carry `target_reliability_status` through to any receiver-level output so the
   provenance of every label survives aggregation.
6. If it opts into auto-accepted failures via `usable_as_named_failed_target`,
   publish the same analysis with them excluded, in the same artifact, and label
   them provisional/unaudited in every table and figure.
7. Never describe Stage 2 failed-pass targets as audited or validated, and never
   cite the completed-pass proxy as failed-pass precision.

## How to retire this lock

Narrow mode is not retired by re-running the pipeline or by a better unresolved
rate. It requires, in order:

1. A blinded human review of every ambiguous failure, the required random
   auto-accepted sample, and the completed controls, recording decision,
   confidence, and reason.
2. Review material sufficient to perform that review — synchronized court-frame
   or trajectory playback, not only the tabular snapshot.
3. Per-fold decision rules re-selected against **training-fold audit labels**,
   replacing the completed-pass proxy.
4. Failed-pass geometry-versus-review agreement and auto-accept precision
   reported with uncertainty.
5. Strict and pre-audit unresolved rates recomputed. If the >10% line still
   breaches, or reliability is poor, failed passes stay touch-level regardless.
6. A new Stage 2 lock revision in `configs/stage2_narrow_path.yaml`.

Until all six are done, `human_audit_status` stays `deferred_waived`,
`narrow_path_active` stays True, and no failed pass in this study supports a
receiver-specific claim.
