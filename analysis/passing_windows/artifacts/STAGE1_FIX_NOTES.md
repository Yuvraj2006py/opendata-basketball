# Stage 1 audit fix notes

**Fix date:** 2026-09-24
**Freeze ID:** `passing_windows_stage0_20250924`
**Responds to:** [`STAGE1_AUDIT_REPORT.md`](STAGE1_AUDIT_REPORT.md) (verdict: BLOCKERS)
**Rebuild:** `python pipelines/01_build_canonical_data.py --force` — completed, exit 0, 331.7 s
**Tests:** `python -m pytest tests/test_stage1.py -q` — **41 passed** (was 12)

All four audit findings are fixed at the root cause, plus the three Stage 2 scout
findings. Primary-eligible touches move from 3645 to **3636**.

---

## Blocker 1 — primary candidate sets are not always four teammates

### Root cause

Not an arithmetic bug in `build_touch_candidates`. Nine primary touches name a
ball-handler who is **not a member of the chance's offensive lineup**:

| touchId | ball-handler | in `offPlayerIds` | in `defPlayerIds` | `touch.offTeamId` vs `chance.offTeamId` |
|---|---:|---|---|---|
| `touch-114243-2-194` | 59181 | no | yes | 1330 vs 1319 |
| `touch-114243-3-514` | 28247 | no | yes | 1319 vs 1330 |
| `touch-114243-3-515` | 59221 | no | yes | 1319 vs 1330 |
| `touch-114234-2-239` | 58474 | no | yes | 1322 vs 1331 |
| `touch-114099-3-494` | 59189 | no | yes | 1315 vs 1316 |
| `touch-114099-4-569` | 59188 | no | **no** | 1315 vs 1315 |
| `touch-114086-3-462` | 19426 | no | **no** | 1328 vs 1328 |
| `touch-184439-3-353` | 59174 | no | yes | 1338 vs 1333 |
| `touch-191313-2-164` | 59207 | no | yes | 1331 vs 1322 |

Seven are **defender-credited touches inside an offensive chance** (a deflection
or steal): the touch is attributed to the opposing team, so the chance's five
offensive players all survive the "exclude the ball-handler" filter. Two name a
player absent from both recorded lineups. Player-ID aliasing was ruled out — the
canonical IDs do not resolve any of the nine.

### Fix

- `joins.ballhandler_in_offense()` is the single predicate for "this touch can
  form four teammates".
- `normalize_touches` now emits a `ballhandler_in_offense` column.
- `quality.EXCLUSION_ORDER` gains `ballhandler_not_in_offense`, placed
  immediately after `missing_ballhandler` (a ball-handler must exist before its
  team membership can be checked), and `primary_exclusion_reason` enforces it.
- `build_touch_candidates(..., strict=True)` raises rather than emitting a wrong
  candidate count; the pipeline calls it in strict mode, so this class of defect
  now fails the build instead of reaching Stage 2.
- `candidate_lineup_from_chance` also de-duplicates repeated lineup IDs.

### Evidence

- 11 touches carry `exclusion_reason = ballhandler_not_in_offense` (the 9 that
  were primary-eligible plus 2 that were previously excluded by a later rule).
- `touch_candidates.parquet`: 14,544 rows; **min = max = 4 candidates per
  touch**; 3636 × 4 = 14,544 exactly.
- `candidate_sets_exactly_four` audit **PASS** — all seven checks true, including
  no ball-handler as its own candidate and no candidate outside the chance lineup.
- Tests: `test_exclusion_ballhandler_not_in_offense`,
  `test_build_touch_candidates_rejects_five_candidate_touch` (uses the real
  `touch-114243-2-194` lineup), `test_build_touch_candidates_exactly_four_per_touch`,
  `test_build_touch_candidates_drops_duplicate_lineup_ids`,
  `test_audit_candidate_sets_detects_five_candidates`,
  `test_ballhandler_rule_is_in_consort_order`.

---

## Blocker 2 — required matchup interval join is not part of Stage 1

### Root cause

`interval_join_matchups` existed but was never called by the pipeline, and its
overlap handling was `drop_duplicates(keep="first")` — dependent on row order,
so not reproducible. No table carried `assigned_defenderId` or `matchupId`.

Overlap is not an edge case in this feed: 2,739 of 4,968 adjacent interval pairs
overlap, and up to 8 intervals (5 distinct defenders) exist for one
`(gameId, chanceId, offPlayerId)`.

### Fix

- `interval_join_matchups` rewritten to preserve input row count and column
  order, append prefixed matchup columns, and resolve overlaps **deterministically**:
  latest `startFrame` (most recent assignment) → shortest interval → smallest
  `matchupId`. It reports `n_matchup_intervals`, `matchup_matched`, and
  `matchup_ambiguous` per row so ambiguity is measured, not hidden.
- New `joins.build_candidate_states()` materializes the decision-state layer:
  one row per (touch, 5 Hz frame, candidate receiver), carrying the touch and
  candidate keys, the resolved defender, and the candidate's event-frame
  position/velocity/`predError`.
- New table **`tables/candidate_states.parquet`** (280,468 rows, 27 columns).
- `touch_model_frames` also gains the ball-handler's own assignment via the same
  join (`ballhandler_*` prefix).
- New `audit_matchup_interval_join` gates the build.

### Evidence

| Metric | Value |
|---|---:|
| Candidate states | 280,468 |
| Coverage, primary-eligible states | **96.78%** |
| Coverage, all states (incl. dead-clock/backcourt) | 61.44% |
| Unmatched states | 108,151 |
| States needing overlap tie-breaking | 3,486 (**1.24%**) |
| Max intervals covering one frame | 4 |
| Duplicate `(gameId, touchId, frameIdx, candidateId)` keys | **0** |
| Ball-handler assignment coverage, eligible frames | 97.27% |

Audit checks all **PASS**: the chosen interval always contains the frame, keys
are unique, no defender is assigned to itself, and every assigned defender comes
from the matchup feed. The 96.8% figure reproduces the auditor's manual number,
and the 15 duplicate keys they found are gone because resolution is now a single
deterministic choice per key rather than an order-dependent dedupe.

Tests: `test_matchup_interval_join_boundaries_are_inclusive` (frames 99/100/120/121
against interval [100,120]), `test_matchup_interval_join_resolves_overlaps_deterministically`
(three overlapping intervals; also asserts a reversed input feed gives the same
answer), `test_matchup_interval_join_preserves_keys_and_row_count`,
`test_matchup_join_handles_empty_matchups`, `test_build_candidate_states_materializes_join`,
`test_audit_matchup_join_detects_interval_violation`.

---

## Blocker 3 — causal velocities corrupted at orientation changes

### Root cause

The pipeline differenced **event-aligned** coordinates (`x_event`, `y_event`).
Those are produced by `broadcast_to_event_xy`, which applies a 180° rotation
`(x, y) → (-x, -y)` when `leftHoop` is False. When possession flips the attacking
hoop, a stationary player's event coordinates jump by up to ~80 ft between
consecutive retained samples. All 4,050 rows above 40 ft/s coincided exactly with
a `leftHoop` change; the worst was 1,169.1 ft/s.

There was also no reset across wide gaps in the retained-frame timeline (the
largest player-to-player frame gap was 108,131 frames).

### Fix

Differentiate the **continuous broadcast coordinates**, then rotate the resulting
vector into the event frame. Both coordinate maps are linear, so a velocity
rotates by the same scalar (`velocities.orientation_sign`), and the artifact
disappears at the source rather than being clamped after the fact. On top of that:

- `causal_velocity_from_positions` accepts `segment_id` and `max_frame_gap` and
  returns NaN rather than differencing across either boundary.
- `orientation_segment_ids` segments each player's track at every `leftHoop`
  change (and isolates unknown-orientation frames).
- `MAX_CAUSAL_FRAME_GAP = 25` frames (1.0 s): a backward difference over a longer
  interval is an average, not an instantaneous velocity.
- `MAX_PLAUSIBLE_PLAYER_SPEED_FPS = 40.0` (elite sprint ≈ 32 ft/s): anything
  above is flagged in `speed_implausible` and nulled, so no implausible value can
  silently reach a feature.
- New columns: `velocity_segment`, `velocity_frame_gap`, `speed_implausible`,
  `velocity_ok`.
- Retention of ±1 s of tracking around every pass (see Scout 2) also removes most
  of the wide-gap differencing.

### Evidence

| Metric | Before | After |
|---|---:|---:|
| Max `speed_causal` | 1,169.1 ft/s | **36.02 ft/s** |
| Rows > 40 ft/s | 4,050 | **0** |
| Rows with velocity at an orientation boundary | all of them | **0** |
| Rows with velocity across a > 25-frame gap | unbounded | **0** |
| Max eligible ball-handler speed | 459.7 ft/s | **24.56 ft/s** |
| Max eligible candidate speed | 472.7 ft/s | **23.73 ft/s** |
| 99.9th percentile speed | — | 21.24 ft/s |
| Null velocities (segment starts, wide gaps, missing coords) | — | 29,365 / 2,972,460 (0.99%) |

`causal_velocity_boundaries` audit **PASS**. `audit_causal_velocity` now also
proves the synthetic segment/gap resets and asserts that the *unsegmented*
version of the same series still produces an impossible speed, so the regression
cannot be reintroduced by reverting to the old code path.

Tests: `test_velocity_resets_at_orientation_boundary` (constructs the real
failure mode, asserts the naive event-coordinate difference exceeds the cap while
the fixed path returns the true 2.5 ft/s), `test_velocity_rotates_into_event_frame`,
`test_velocity_resets_across_wide_frame_gap`,
`test_implausible_speed_is_flagged_and_nulled`,
`test_causal_velocity_segment_argument_blocks_differencing`,
`test_orientation_segment_ids_split_on_change`,
`test_audit_causal_velocity_boundaries_catches_corrupt_track`.

---

## Major 1 — output manifest stale and self-invalidating

### Root cause

Two distinct bugs:

1. The manifest hashed `artifacts/stage1_output_manifest.json` and was then
   written to that same path, so that entry could never verify.
2. `ARTIFACTS.glob("stage1_*")` is **case-insensitive on Windows**, so it also
   matched the hand-written `STAGE1_*` documents. `STAGE1_VERIFICATION.md` was
   picked up by the glob *and* appended explicitly, producing the duplicate entry
   and reporting the same mismatch twice. The same glob would have swallowed
   `STAGE1_AUDIT_REPORT.md`.

The `stage1_calculation_audit.md` mismatch followed from generation order.

### Fix

- `build_output_manifest` takes `manifest_path` and excludes it from its own file
  list, and `_hash_entries` de-duplicates paths.
- Generated artifacts are now named **explicitly** instead of globbed.
- The manifest is written **last**, after every table and artifact is final.
- `STAGE1_VERIFICATION.md` moves to a separate `sign_off_files` section: it is
  edited by a human after the build, so including it in the strictly verified set
  guarantees a mismatch. `verify_output_manifest` validates only `files`.
- New `verify_output_manifest` / `refresh_sign_off_hashes`, surfaced as
  `--verify-manifest` and `--refresh-sign-off`. The pipeline self-verifies at the
  end of every build and returns exit code 2 on mismatch.

### Evidence

```
$ python pipelines/01_build_canonical_data.py --verify-manifest
{ "n_files": 33, "n_missing": 0, "n_mismatched": 0, "n_duplicate_paths": 0, "ok": true }
```

Tests: `test_manifest_excludes_itself_and_dedupes`,
`test_manifest_verification_detects_tampering`,
`test_manifest_verification_flags_duplicate_entries`,
`test_manifest_sign_off_files_are_separate`.

---

## Minor 1 — tests did not exercise the blocking paths

Test count 12 → **41**. Every finding above has at least one test that fails if
the defect is reintroduced, and the fixtures use the real offending lineups and
the real coordinate failure mode rather than abstract stand-ins.

---

# Stage 2 scout findings

## Scout 1 — `receiverLoc` / `distance` / `receiverRegion` null for all incomplete passes

**Verified against the raw feed; this is a data limitation, not a Stage 1 loss.**
Parsing `{gameId}_dynamic_events.json` directly shows that every non-complete
pass record has `receiverLoc: null`, `distance: null`, `receiverRegion: null`,
**and** `receiverId: null`. Stage 1 drops nothing.

A related defect *was* found in Stage 1: `complete` is **tri-state** in the raw
feed, and `~passes["complete"].astype(bool)` silently merged the nulls into the
failed-pass cohort. The 157 "incomplete" passes are two different populations:

| `pass_outcome_class` | N | `endFrame` | `receiverId` | `receiverLoc` | interceptor | `turnover` |
|---|---:|---:|---:|---:|---:|---:|
| `complete` | 4629 | 4629 | 4629 | 4541 | 0 | 0 |
| `incomplete_turnover` (`complete = False`) | 89 | 89 | 0 | 0 | 89 | 89 |
| `unknown_outcome` (`complete = null`) | 68 | **0** | 0 | 0 | 0 | 0 |

`normalize_passes` now emits `complete_flag` (nullable boolean),
`pass_outcome_class`, `has_endFrame`, `has_receiver_evidence`, `is_failed_pass`,
and `failed_pass_inference_eligible`. New `audit_pass_outcome_classes` asserts
the evidence gaps. Note 88 *completed* passes also lack `receiverLoc`.

### Stage 2 design constraints

1. **Do not plan to compare candidates against `receiverLoc` for failed passes.**
   That channel does not exist. Use ball trajectory from the retained flight
   window (Scout 2), release-time pass direction, and empirical flight-time
   support fitted on completed passes.
2. **The failed-pass target-inference population is 89 passes**, not 157. Use
   `failed_pass_inference_eligible`. Stage 0's `unresolved_kill_threshold_pct: 10`
   therefore allows at most ~9 unresolved cases — a tight budget worth flagging
   before Stage 2 begins.
3. The 68 `unknown_outcome` passes have no end frame, no receiver, and no
   interceptor. They can support release-time touch-level turnover/pressure
   analysis but cannot enter receiver-specific target inference.
4. `toReceiverId` remains the interceptor and is never a target (unchanged).

## Scout 2 — sparse tracking retention around pass flight

**Fixed in Stage 1** (not deferred). `_needed_frames_for_game` now retains every
25 Hz frame within `PASS_WINDOW_FRAMES = 25` (±1.0 s) of each pass `startFrame`
and `endFrame`. Non-complete passes have no `endFrame` at all, so their release
window widens to `FAILED_PASS_WINDOW_FRAMES = 50` (±2.0 s) to capture the flight
without an end anchor. `tracking_frames` gains `is_pass_window_frame` and
`is_model_sample_frame` so the retention reason for any frame is explicit.

| Metric | Before | After |
|---|---:|---:|
| Retained 25 Hz frames | 71,988 | **307,175** |
| — of which pass-window | 0 | 263,440 |
| — of which 5 Hz model samples | 71,988 | 71,928 |
| Tracking player rows | 719,880 | 2,972,460 |
| `tracking_players.parquet` | 57 MB | 248 MB |
| Build wall time | 315.9 s | 331.7 s |

All **157/157** failed and unknown-outcome passes have the full +1.0 s
post-release window retained, and 97.7% of pass-window frames carry ball `xyz`.
Stage 2 needs no re-stream hook or ad-hoc extraction.

## Scout 3 — null `endFrame` and primary-eligibility honesty

Covered by Scout 1's classification. `stage1_quality_report.md` now carries the
per-class evidence table, the failed-pass eligible count (89), and the retention
and velocity-plausibility sections. `STAGE1_VERIFICATION.md` reports the three
pass-outcome counts as first-class sample sizes.

---

# Not fixed / residual risks

1. **35% of all candidate states have no defensive assignment.** This is
   concentrated in dead-clock and backcourt frames that primary eligibility
   already removes; coverage among primary-eligible states is 96.8%. The
   remaining 3.2% is a genuine gap in the matchup feed. Stage 2 must handle
   `matchup_matched == False` explicitly rather than assuming a defender exists.
2. **1.24% of states needed overlap tie-breaking.** The rule is deterministic and
   documented, but it is a *choice*. If defensive assignment identity matters to
   a headline result, Stage 3+ should run a sensitivity check under an
   alternative rule (for example longest-interval-wins).
3. **0.99% of player rows have null velocity** by design (segment starts, wide
   gaps, missing coordinates). Stage 3 must treat `velocity_ok == False` as
   missing data, not as zero velocity.
4. **The 40 ft/s cap and 25-frame gap limit are fixed engineering constants,**
   not fold-calibrated. They are documented in `stage1_environment.json` and the
   calculation audit. They screen physical impossibility only; the
   `predError`-based tracking caps remain `TBD_TRAINING_FOLD` as frozen.
5. **Only 89 failed passes are eligible for target inference** (Scout 1 item 2).
   Stage 0's 10% unresolved kill threshold is a small absolute budget.
6. **The 0–2 s shot-clock exclusion rate remains elevated at ~61%**, carried
   forward unchanged from the original audit as a sensitivity warning.
7. **`hoop_orientation_ok` and `leftHoop` are used for rotation only.** The
   velocity fix depends on `leftHoop` being correct per possession; possessions
   with unknown orientation produce null velocities rather than wrong ones.
8. Stage 2 and Stage 3 remain **not started**, as instructed.
