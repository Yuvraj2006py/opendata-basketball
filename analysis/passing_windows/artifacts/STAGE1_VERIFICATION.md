# Stage 1 USER VERIFICATION CHECKLIST

**Freeze ID:** `passing_windows_stage0_20250924`
**Purpose:** Approve the canonical data layer before Stage 2 (failed-pass target inference).

Mark each item **APPROVE** / **REJECT** / **CHANGE REQUEST**.

## A. Sample sizes

| Metric | Value |
|---|---:|
| All touches | 6662 |
| Primary-eligible touches | 3636 |
| Model frames (5 Hz, primary touches) | 70117 |
| Primary-eligible model frames | 44001 |
| Candidate states (frame x candidate) | 280468 |
| Retained 25 Hz tracking frames | 307175 |
| Passes (all) | 4786 |
| Passes: complete | 4629 |
| Passes: incomplete/turnover | 89 |
| Passes: unknown outcome (no endFrame) | 68 |

## B. Joins and geometry tests

| # | Check | Pipeline result | Verify |
|---|---|---|---|
| B1 | `event_tracking_clocks_passes` | PASS | ☐ |
| B2 | `pass_release_vs_touch_end` | PASS | ☐ |
| B3 | `receiver_loc_vs_tracking` | PASS | ☐ |
| B4 | `mirroring_symmetry` | PASS | ☐ |
| B5 | `causal_velocity` | PASS | ☐ |
| B6 | `causal_velocity_boundaries` | PASS | ☐ |
| B7 | `synthetic_lane_geometry` | PASS | ☐ |
| B8 | `candidate_sets_exactly_four` | PASS | ☐ |
| B9 | `matchup_interval_join` | PASS | ☐ |
| B10 | `pass_outcome_classes` | PASS | ☐ |
| B11 | `primary_invariants` | PASS | ☐ |

## C. Exclusions and frozen rules

| # | Rule | Verify |
|---|---|---|
| C1 | Self-passes excluded from primary | ☐ |
| C2 | Malformed / zero-elapsed chances excluded | ☐ |
| C3 | Multi-pass touches excluded (audited table retained) | ☐ |
| C4 | Inbounds touches excluded | ☐ |
| C5 | Substitution-boundary possessions excluded | ☐ |
| C6 | `toReceiverId` not used as intended target | ☐ |
| C7 | Points reconstructed from shots+FT only | ☐ |
| C8 | Season aggregates not used | ☐ |
| C9 | Player aliases added; original IDs preserved | ☐ |
| C10 | `predError` caps still TBD_TRAINING_FOLD | ☐ |
| C11 | Ball-handler outside `chance.offPlayerIds` excluded; every primary touch forms exactly 4 candidates | ☐ |
| C12 | Causal velocities differenced on continuous coords; reset at orientation/gap boundaries | ☐ |
| C13 | Matchup interval join materialized with deterministic overlap resolution | ☐ |
| C14 | Failed-pass evidence limits documented (`receiverLoc` absent for all non-complete passes) | ☐ |

## D. Artifacts present

| # | Artifact | Verify |
|---|---|---|
| D1 | `artifacts/stage1_sample_flow.md` | ☐ |
| D2 | `artifacts/stage1_join_audit.md` | ☐ |
| D3 | `artifacts/stage1_calculation_audit.md` | ☐ |
| D4 | `artifacts/stage1_quality_report.md` | ☐ |
| D5 | `artifacts/stage1_output_manifest.json` (verify with `--verify-manifest`) | ☐ |
| D6 | `tables/*.parquet` consolidated outputs | ☐ |
| D7 | `tables/candidate_states.parquet` defensive-assignment layer | ☐ |

## Sign-off

The user waived the manual checklist, so sign-off rests on the automated audit
suite. Every pipeline check in section B is **PASS** and
`python -m pytest tests/test_stage1.py -q` reports **41 passed**.

| Role | Name | Date | Overall |
|---|---|---|---|
| Adversarial audit (human checklist waived) | Automated Stage 1 auditor | 2026-09-24 | **REJECT** (superseded) |
| Audit fix implementation | Stage 1 fix implementer | 2026-09-24 | **APPROVE** |
| Independent adversarial re-audit | Stage 1 re-auditor | 2026-09-24 | **APPROVE** |

**Stage 2 may begin.** The independent empirical re-audit is **CLEAR**; see
[`STAGE1_REAUDIT_REPORT.md`](STAGE1_REAUDIT_REPORT.md).

## Resolution of the 2026-09-24 audit rejection

The four blocking findings in [`STAGE1_AUDIT_REPORT.md`](STAGE1_AUDIT_REPORT.md)
are fixed at the root cause and the canonical tables were rebuilt with
`--force`. Full detail, evidence, and residual risks are in
[`STAGE1_FIX_NOTES.md`](STAGE1_FIX_NOTES.md).

| Audit finding | Status | Headline evidence |
|---|---|---|
| 1. Nine primary touches produced five candidates | **Fixed** | Ball-handler was outside `chance.offPlayerIds` (defender-credited touches). New `ballhandler_not_in_offense` exclusion; candidates per primary touch now min = max = **4**; 3636 × 4 = 14,544 rows exactly |
| 2. Matchup interval join absent | **Fixed** | `tables/candidate_states.parquet` materialized (280,468 rows); **96.78%** coverage on primary-eligible states; overlaps resolved deterministically; **0** duplicate keys |
| 3. Impossible causal velocities | **Fixed** | Differentiation moved to continuous broadcast coordinates with orientation/gap resets; max speed **1169.1 → 36.02 ft/s**; rows above 40 ft/s **4050 → 0** |
| 4. Manifest stale hashes / duplicate entry | **Fixed** | Manifest excluded from its own hash list, paths de-duplicated, explicit file list replaces the case-insensitive `stage1_*` glob, generated last; `--verify-manifest` reports **33 files, 0 mismatched, 0 duplicates** |
| Minor. Tests did not cover the blockers | **Fixed** | Test count **12 → 41**; each defect has a regression test using the real offending data |

Stage 2 scout findings are also addressed: the failed-pass `receiverLoc` gap is
confirmed as a raw-feed limitation (not a Stage 1 loss) and documented as a
Stage 2 design constraint; tri-state `pass_outcome_class` now separates the 89
turnover passes from the 68 unknown-outcome passes; and tracking retention was
widened to ±1 s around every pass (±2 s around non-complete releases) so ball
flight is available without re-streaming.

**Open items for Stage 2 to respect:** 3.2% of primary-eligible candidate states
have no defensive assignment, 1.24% required overlap tie-breaking, 0.99% of
player rows have null velocity by design, and only **89** failed passes are
eligible for target inference against Stage 0's 10% unresolved kill threshold.

