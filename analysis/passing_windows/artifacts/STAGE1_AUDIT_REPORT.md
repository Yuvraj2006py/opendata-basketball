# Stage 1 adversarial audit report

**Audit date:** 2026-09-24  
**Freeze ID:** `passing_windows_stage0_20250924`  
**Overall verdict:** **BLOCKERS**  
**Stage 2:** **MAY NOT PROCEED**

## Executive finding

Stage 1 is not ready for Stage 2. The event tables and most frozen touch-level
exclusions are coherent, and all 12 existing tests pass, but three defects can
directly corrupt Stage 2 candidate inference: some primary touches have five
candidate receivers, the required matchup interval join is absent from the
canonical outputs, and causal velocities contain possession-boundary coordinate
discontinuities. The Stage 1 output manifest is also not internally valid.

## Severity-ranked findings

### Blocker 1 — primary candidate sets are not always four teammates

`build_touch_candidates` removes the ball-handler from `chance.offPlayerIds`
but does not require that the ball-handler is in that list
(`src/passing_windows/joins.py:40-69`). Touch eligibility only checks that a
ball-handler ID exists, not that it belongs to the offensive lineup
(`src/passing_windows/quality.py:102-109`). The pipeline then builds candidates
from every touch marked primary (`pipelines/01_build_canonical_data.py:374-377`).

Parquet spot-check:

- 3,636 primary touches have four candidates.
- **9 primary touches have five candidates**, because `playerId` is absent from
  the five-player `offPlayerIds` list.
- Examples include `touch-114243-2-194`, `touch-114086-3-462`, and
  `touch-191313-2-164`.

This violates the frozen Stage 2 contract to form four teammate candidates and
can admit an opponent or otherwise wrong player into target inference.

**Required before approval:** resolve the lineup/touch mismatch or exclude and
flag these touches; assert exactly four unique teammates for every primary
touch and test it.

### Blocker 2 — required matchup interval join is not part of Stage 1

An interval-join helper exists (`src/passing_windows/joins.py:73-145`), but the
pipeline imports and calls only pass/touch, candidate, and post-catch joins
(`pipelines/01_build_canonical_data.py:52-56,374-379`). No consolidated table
contains `assigned_defenderId`, `matchupId`, or interval boundaries, and no
matchup interval audit appears in `stage1_join_audit.md:5-12`.

A manual audit join found 96.8% candidate-state coverage, but also 15 duplicated
`(gameId, chanceId, frameIdx, offPlayerId)` keys. Those ambiguities are neither
resolved nor documented by Stage 1.

**Required before approval:** materialize the candidate-frame matchup join,
preserve touch/candidate keys, deterministically resolve overlapping intervals,
report unmatched/ambiguous rates, and add boundary/overlap tests.

### Blocker 3 — causal velocities are numerically corrupted at orientation changes

Tracking coordinates are rotated according to `leftHoop`, then velocities are
computed continuously by player across all retained frames
(`pipelines/01_build_canonical_data.py:250-269`). The velocity implementation
blindly differences adjacent retained samples and has no possession/orientation
boundary reset (`src/passing_windows/velocities.py:12-44`).

Parquet audit:

- **4,050** player rows have `speed_causal > 40 ft/s`.
- Every one of those rows coincides with a `leftHoop` change.
- Maximum reported speed is **1,169.1 ft/s**.
- Even among primary-eligible decision states, 23 ball-handler rows and 94
  candidate rows exceed 40 ft/s (maxima 459.7 and 472.7 ft/s).

The synthetic no-future test proves causality but not correctness across
coordinate discontinuities (`src/passing_windows/audit.py:163-188`). Stage 2
explicitly depends on projected movement at release, so these values are unsafe.

**Required before approval:** reset derivatives at possession/orientation and
large-gap boundaries (or compute in a continuous frame before normalization),
define plausibility checks, and test real boundary cases.

### Major 1 — Stage 1 output SHA manifest is stale and self-invalidating

The pipeline includes the existing `stage1_output_manifest.json` in the paths
to hash and then overwrites that same file
(`pipelines/01_build_canonical_data.py:817-828`). The manifest also lists
`STAGE1_VERIFICATION.md` twice.

Before this required audit/sign-off edit, recomputed SHA-256 values already
found mismatches for:

- `artifacts/stage1_output_manifest.json`
- `artifacts/stage1_calculation_audit.md`

The first mismatch is guaranteed by the generation order; the second shows an
artifact changed after manifest creation. Updating `STAGE1_VERIFICATION.md` to
record this rejection necessarily adds that file as another current mismatch;
its duplicate manifest entry reports the same mismatch twice. This fails the
reproducibility gate.

**Required before approval:** exclude the manifest from its own file list (or
use a separately defined digest), de-duplicate paths, regenerate after all
other artifacts are final, and add a manifest verification command/test.

### Minor 1 — existing tests do not exercise the blocking paths

`tests/test_stage1.py:27-150` covers aliases, synthetic coordinates/velocity,
basic pass/touch joining, geometry, and one malformed exclusion. It has no
test for exactly four candidate teammates, real orientation-boundary velocity,
matchup interval joining, output-manifest verification, or complete primary
invariants. This explains why **12/12 tests pass** despite the blockers.

## Required checklist

| Audit item | Result | Evidence |
|---|---|---|
| Primary population / exclusions match freeze | **FAIL** | Usable, non-transition, zero-clock, frame order, 5v5-list length, substitutions, self-pass, multipass, and inbounds flags are clean for all 3,645 primary touches; however 9 primary touches have a ball-handler outside `offPlayerIds`, yielding five candidates. Frame-level live/frontcourt/10-player/BH filtering produces 44,045 eligible states from 70,685 sampled states (`stage1_quality_report.md:20-27`). |
| Joins: clocks, release/touch, receiver/tracking, matchups | **FAIL** | Clock sample 300/300, ordinary release/touch, receiver/tracking, and primary pass joins pass (`stage1_join_audit.md:5-12`; primary-pass spot-check: 99.77% within touch, 99.80% near end, 100% passer match). Matchup interval join is absent and ambiguity is unaudited. |
| Coordinates and causal velocities | **FAIL** | Event alignment and mirroring tests pass (`stage1_calculation_audit.md:70-79`), but 4,050 impossible velocities occur exactly at `leftHoop` changes. |
| `toReceiverId` never treated as intended target | **PASS** | Only copied to `interceptorId`; code search found no target use (`src/passing_windows/normalize.py:150-153`). All 89 non-null values occur on incomplete passes. |
| Season aggregates not used as covariates | **PASS** | No Python load/reference to aggregate files; only prohibition text appears. |
| Player aliases preserve original IDs | **PASS** | Canonical columns are added while originals remain (`src/passing_windows/aliases.py:44-56`, `src/passing_windows/ingest.py:184-203`); parquet checks confirmed both forms in touches, passes, and matchups. |
| Manifests / SHA hashes present and valid | **FAIL** | Input and output manifests are present, but the Stage 1 output manifest already had two SHA mismatches before sign-off, has one duplicate path, and now also cannot validate the required updated verification file. |
| Sample flow coherent / concentration plausible | **PASS** | 6,662 total minus 3,017 first-match exclusions equals 3,645 touch-level primary rows (`stage1_sample_flow.md:7-37`). Per-game primary counts range 341–395. The 0–2 second shot-clock exclusion rate is elevated at 61.5% (`stage1_sample_flow.md:99-106`) but is not a single-game/team collapse; retain as a sensitivity warning. |
| Tests actually run and pass | **PASS** | `python -m pytest tests/test_stage1.py -q` on 2026-09-24: **12 passed in 3.21s**. Coverage is insufficient for the defects above. |

## Additional spot-checks

- `passes.parquet`: 4,786 rows; 89 non-null interceptor IDs; none on completed
  passes.
- `passes_joined.parquet`: 33 rows lack a touch join overall; among passes from
  primary touches, pass/touch and passer-ID agreement are high as reported
  above.
- `touch_candidates.parquet`: 14,589 rows for 3,645 primary touches; expected
  count would be 14,580 if every touch had exactly four candidates.
- `post_catch_chain.parquet`: 4,511 completed non-self pass rows; reported next
  receiver-touch coverage is 99.62% (`stage1_calculation_audit.md:35-40`).
- Tracking normalized-coordinate null rate is 0.77% in retained player rows;
  primary state eligibility already filters missing players, but Stage 2 should
  continue to handle coordinate nulls explicitly.

## Sign-off

**REJECT. Stage 2 may not proceed** until all four blocker/major findings are
fixed, the canonical tables and audits are regenerated, and this adversarial
audit is rerun.
