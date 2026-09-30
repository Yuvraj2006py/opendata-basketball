# Stage 1 independent adversarial re-audit

**Re-audit date:** 2026-09-24  
**Freeze ID:** `passing_windows_stage0_20250924`  
**Verdict:** **CLEAR**  
**Stage 2:** **MAY PROCEED**, subject to the design constraints below.

## Scope and method

This re-audit did not trust the fix notes. It inspected the Stage 1 source and
tests, reran the 41-test suite, directly queried the consolidated parquet
tables, parsed all ten raw dynamic-event files, independently reconstructed the
matchup overlap winner, and independently recomputed every output-manifest hash.
No Stage 2 work was performed.

The repository contains no study-local virtual environment. Tests therefore ran
with `C:\Python313\python.exe`, the same Python 3.13.7 interpreter and package
versions recorded in `stage1_environment.json`.

## Required fix verification

### 1. Four candidates per primary touch — CLEAR

- 6,662 touches; 3,636 primary-eligible and 3,026 excluded. The exclusion
  reasons partition the non-primary population exactly.
- `ballhandler_not_in_offense` exists and excludes 11 touches. No primary touch
  has a ball-handler outside `offPlayerIds`.
- `touch_candidates.parquet` has 14,544 rows in 3,636 touch groups.
- Candidate count and unique-candidate count both have min = max = 4.
- No primary touch is missing candidates; no candidate equals the ball-handler.
- Strict candidate construction rejects a touch that cannot form exactly four
  teammates.

The prior primary count of 3,645 falling to 3,636 is explained by the nine
previously eligible lineup/ball-handler mismatches; it is not an unexplained
sample loss.

### 2. Matchup interval join — CLEAR

- `candidate_states.parquet` has 280,468 rows, exactly
  `70,117 model frames × 4 candidates`.
- There are zero duplicate `(gameId, touchId, frameIdx, candidateId)` keys and
  zero frame groups with fanout other than four.
- Assignment coverage is 61.4391% overall and **96.7762%** over the 176,004
  primary-eligible candidate states.
- 108,151 states are unmatched; 3,486 require overlap resolution; the maximum
  number of covering intervals is four.
- An independent interval reconstruction found 172,317 states with a covering
  interval. For every one, `n_matchup_intervals` matched the independently
  counted intervals and the selected `matchupId` matched the documented rule:
  latest start, shortest interval, then smallest ID. There were zero selected
  winners without a covering interval.

### 3. Causal velocities — CLEAR

- 2,972,460 tracking-player rows; 2,943,095 finite causal speeds and 29,365
  intentional nulls.
- Maximum retained causal speed is **36.0212 ft/s**; zero retained rows exceed
  40 ft/s.
- Eighteen raw derivatives exceeded the plausibility cap and are explicitly
  flagged and nulled; none survives as a usable speed.
- Zero orientation-boundary rows and zero gaps over 25 frames carry a velocity.
- Invalid velocity rows do not encode missing `vx`, `vy`, or candidate speed as
  zero.

Source inspection confirms that derivatives are taken on broadcast `x/y`,
segmented on orientation, gap-limited, then rotated into event coordinates.

### 4. Output manifest — CLEAR

Both the pipeline verifier and a separate SHA-256 implementation found:

- 33 declared files and 33 entries;
- zero missing files;
- zero hash or byte-size mismatches;
- zero duplicate paths;
- the manifest is not included in its own strict file list.

### 5. Scout addenda — CLEAR

Direct parsing of all 4,786 raw pass records reproduced:

- complete: 4,629;
- incomplete turnover: **89**;
- unknown outcome (`complete = null`): 68.

All 89 incomplete turnovers have an end frame and interceptor, but no
`receiverId`, `receiverLoc`, `distance`, or `receiverRegion`. All 68 unknown
outcomes also lack those receiver fields and additionally lack an end frame and
interceptor. Parquet outcome-class counts match raw counts exactly, and
`failed_pass_inference_eligible` contains **89**, not 157, rows.

Retention checks found the full first second after release for all 157
non-complete/unknown passes. Across all requested complete-pass windows, 19
frame numbers were absent from the retained table; direct inspection of the
compressed tracking feeds confirmed that those 19 frame numbers do not exist in
the raw tracking data. This is not a retention loss.

## Regression checks

- Primary eligibility and exclusion reasons are internally exhaustive.
- Candidate-state fanout remains exactly four and join keys remain unique.
- Eligible candidate-state count equals `4 × 44,001 eligible model frames`.
- Null velocities remain null rather than being converted to stationary zeros.
- All 41 tests pass.
- No new Stage 1 blocker was found.

## Residual constraints for Stage 2

1. The failed-pass receiver-inference cohort is **N = 89**, not 157. Under the
   frozen 10% unresolved kill threshold, the practical unresolved budget is
   about nine cases.
2. Failed passes have no raw receiver location or receiver identity. Target
   inference must use retained ball flight, release direction, candidate
   geometry, and completed-pass flight-time support; the interceptor must never
   be used as the target.
3. The 68 unknown-outcome passes cannot support receiver-specific target
   inference because they lack an end frame, receiver, and interceptor.
4. About 3.22% of primary-eligible candidate states have no matchup assignment;
   Stage 2 must model this as missing, not assume a defender.
5. Matchup identity required deterministic tie-breaking for 1.24% of states;
   later headline analyses should include an alternative-rule sensitivity check.
6. Velocity nulls are intentional boundary/gap/missing-data values and must not
   be imputed as zero.

## Sign-off

**APPROVE. Stage 2 may proceed.**
