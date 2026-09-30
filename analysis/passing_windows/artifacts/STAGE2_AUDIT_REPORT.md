# Stage 2 independent adversarial audit

**Audit date:** 2026-09-24  
**Freeze ID:** `passing_windows_stage0_20250924`  
**Original verdict:** **BLOCKERS** (human-audit gate not met)  
**Remediation (2026-09-24):** **NARROW PATH ACTIVATED**  
**Stage 3:** **CONDITIONAL GO under narrow path only** — see
[`STAGE2_NARROW_PATH_LOCK.md`](STAGE2_NARROW_PATH_LOCK.md) and
`configs/stage2_narrow_path.yaml`. Human audit remains **DEFERRED / WAIVED**;
reviewer columns stay blank (not fabricated).

## Remediation status (post-audit)

The blockers below still correctly describe the *frozen human-audit gate*: it
was not passed, and inventing `reviewer_decision` values would not pass it.
Stage 0 already specified the fallback when reliability is inadequate
(`on_inadequate_reliability`). That clause has been machine-enforced:

| Required fix from this audit | Status |
|---|---|
| Complete blinded human review | **DEFERRED / WAIVED** — not performed; columns blank by design |
| Synchronized court-frame review material | **NOT BUILT** — still tabular only |
| Failed-pass geometry-vs-review agreement | **NOT AVAILABLE** — no review labels exist |
| Re-select fold thresholds on audit labels | **NOT DONE** — completed-pass proxy remains; documented as provisional |
| Machine-readable gate fields | **DONE** — `narrow_path_active`, `usable_as_receiver_completion_label`, `usable_as_named_failed_target`, `receiver_specific_eligible`, et al. on every Stage 2 pass table + `tables/stage2_label_eligibility.parquet` |
| Tests that fail on blank review / narrow gate / unaudited labels | **DONE** — `tests/test_stage2.py` narrow-path suite |
| Completed-pass proxy treated as diagnostic only | **DONE** — lock + report wording + `assert_no_failed_precision_claim` |

Under the narrow path, Stage 3 may begin **only** if it trains receiver-specific
completion labels on `receiver_specific_eligible == True` (completed passes with
known receivers) and treats all failed passes as touch-level turnover evidence.
Receiver-specific failed-pass claims remain forbidden without a later lock
revision backed by a real audit.

## Executive finding (original adversarial audit)

The claimed counts, held-out completed-pass metrics, test count, strict manifest
hashes, tri-state handling, and interceptor exclusion reproduce. However, Stage
2 has not completed the frozen manual-audit gate. All reviewer fields are blank,
the pre-audit unresolved rate is **29/89 = 32.6%**, and the claimed **0.968**
auto-accept precision is measured on completed passes, not failed-pass audit
labels. The code itself describes that number as an optimistic bound.

The frozen threshold procedure required audit agreement. Instead, thresholds
were selected from completed-pass receiver labels and remain provisional. This
is not leakage across games, but it is a substantive protocol substitution.
Stage 3 must not begin on the *full* receiver-specific path while failed-pass
targets remain unaudited; it may begin only under the narrow-path lock above.

## Independent empirical checks

- Raw/consolidated pass classes: **4,629 complete**, **89
  incomplete_turnover**, **68 unknown_outcome**.
- Failed decisions: **60 auto_accept**, **27 ambiguous**, **2 unresolved**.
- Every one of the 89 failed passes has four candidate-probability rows;
  maximum probability-sum error was `3.33e-16`, with no negative probabilities.
- Interceptor occurrences: **0** in candidate rows and **0** in accepted target
  labels. Non-accepted hard labels: **0**.
- Completed controls: `n=4,530`; top-1 agreement
  `0.8192052980`; auto-accept precision `0.9677909450`.
- The audit CSV contains 596 rows across 149 passes and covers all 27 ambiguous
  failures, but all four reviewer columns contain **zero nonblank values**.
- All ten games are written only by their matching holdout fold; every fold
  records nine training games.
- Stage 2 candidate release coordinates and velocities were cross-joined back
  to each game's Stage 1 `tracking_players.parquet`: maximum x/y difference
  `0.0`, maximum velocity difference `0.0`, and zero velocity-flag mismatches.
- Independent SHA-256 and byte-size recomputation matched all **12** strict
  manifest entries.
- Test run: **88 passed in 5.13s**.
- The pipeline manifest verifier also returned 12 files, zero missing, zero
  mismatched, and zero duplicate paths.

## Checklist verdicts

### 1. Plan compliance — FAIL

The implementation does form four teammates, emits a probability simplex,
requires probability and geometric margins, exports candidate diagnostics, and
structurally excludes the interceptor. Those mechanical requirements pass.

The overall Stage 2 plan does not pass because the required human audit was not
performed. The audit export has blank reviewer decisions, confidence, and
reasons. No geometry-versus-audit agreement exists, and thresholds were not
selected against audit agreement as frozen. The CSV also supplies tabular
release geometry rather than the requested synchronized court-frame review
material; that is insufficient evidence that the intended manual review was
possible or completed.

### 2. Leave-one-game-out / leakage — PASS

Source inspection shows that flight-time support, likelihood scales, and
decision thresholds use only `train_gameIds`; only the held-out game is scored.
Output inspection found exactly one matching holdout fold per game and nine
training games per fold. No held-out receiver labels feed fitting or threshold
selection.

The completed-pass proxy is a validity problem, not cross-game leakage.

### 3. Tri-state `complete` handling — PASS

The 89 `incomplete_turnover` IDs exactly equal the failed assignment IDs. The
68 `unknown_outcome` IDs have zero overlap and are separately counted. The
4,629 complete records are not conflated with either class.

### 4. Kill criteria and narrow path — FAIL (original) → REMEDIATED (machine gate)

The report honestly gives both strict unresolved (**2/89 = 2.2%**) and
pre-audit unresolved (**29/89 = 32.6%**) and marks the latter as a breach.

**Original gap:** Enforcement was incomplete. All 89 assignment rows were marked
provisional with no machine-readable `narrow_path_active` or reliability status,
so receiver-specific consumers could silently use provisional labels.

**Remediation:** `configs/stage2_narrow_path.yaml` locks
`target_inference_mode: narrow_path`. `src/passing_windows/narrow_path.py`
writes gate columns onto every Stage 2 pass table and
`tables/stage2_label_eligibility.parquet`, and fatal assertions reject leaked
completion labels, missing gates, interceptor-as-target, and fabricated
reviewer values. Auto-accepted failures remain opt-in only via
`usable_as_named_failed_target` with a required sensitivity check — they are
still **not** claimed as audited.

### 5. Coordinate-frame consistency — PASS

Stage 2 uses the Stage 1 `*_event` coordinates directly. Independent row-level
joins at `(gameId, startFrame, candidateId)` produced zero coordinate,
velocity, or velocity-validity differences. Mirror-invariance tests also pass.

### 6. Tests and claimed invariants — PASS WITH GAP (original) → GAP CLOSED for narrow path

The full suite passes. Tests cover candidate construction, probability simplex,
geometric margins, interceptor exclusion, tri-state cohorts, mirror invariance,
assignment membership, and holdout fold mapping.

**Original gap:** Tests accepted blank reviewer columns and did not require an
active narrow-path flag.

**Remediation:** `tests/test_stage2.py` now asserts the lock, triggers,
gate-column presence, zero failed completion labels, fatal promotion of
ambiguous/unresolved/auto-accepted failures into completion labels, blank
reviewer columns under `deferred_waived`, and that generated reports do not
phrase the completed-pass proxy as failed-pass precision. Passing tests now
establish Stage 2 completion **under the narrow path**, not under the frozen
human-audit gate.

### 7. Manifest hashes — PASS

All 12 strict entries independently match their recorded SHA-256 and byte size.
The official verifier also passes. `STAGE2_VERIFICATION.md` is a mutable
sign-off file outside the strict list by design.

### 8. Fallthroughs, joins, and proxy validity — FAIL

No wrong join or interceptor fallthrough was found. The blocking issue is
optimistic proxy abuse at the release gate:

- Completed passes reveal their receiver and are geometrically easier than
  intercepted passes.
- The reported `0.968` is completed-pass precision, not failed-pass precision.
- The project self-check already fails
  `failed_residuals_within_completed_support`: among 60 accepted failures,
  **15.0%** exceed the completed-correct top-angle p99 and **11.7%** exceed the
  completed-correct lateral-residual p99.
- Nevertheless, those 60 rows receive receiver IDs under thresholds selected
  from completed controls.

This evidence does not validate receiver-specific failed-pass claims for Stage
3 or later.

## Required fixes

1. Complete a blinded human review of every ambiguous failed pass, the required
   random auto-accepted sample, and completed controls; record decision,
   confidence, and reason.
2. Provide synchronized court-frame/trajectory review material sufficient to
   perform that audit, not only a tabular release snapshot.
3. Report failed-pass geometry-versus-review agreement and auto-accept
   precision with uncertainty.
4. Re-select each fold's decision rule using only training-fold audit labels,
   or formally revise the frozen protocol before proceeding.
5. Recompute strict and pre-audit unresolved rates after review. If the >10%
   criterion still breaches or reliability is poor, keep failed passes
   touch-level only.
6. Add machine-readable gate fields such as `narrow_path_active`,
   `target_reliability_status`, and `receiver_specific_eligible`; downstream
   stages must reject provisional/unreviewed labels by default.
7. Add regression tests that fail when Stage 3 is attempted with blank review
   fields, an active narrow gate, or unaudited receiver-specific failures.

## Sign-off

**Original:** REJECT against the frozen human-audit gate.

**After narrow-path remediation:** Stage 3 may proceed **only under**
`target_inference_mode: narrow_path` (`configs/stage2_narrow_path.yaml`).
Completed passes retain known-receiver labels (`receiver_specific_eligible`).
Failed passes remain touch-level turnover evidence only; no failed pass supports
a default receiver-specific Stage 3+ claim. Human audit is DEFERRED/WAIVED —
reviewer fields must stay blank until a real review retires the lock.
