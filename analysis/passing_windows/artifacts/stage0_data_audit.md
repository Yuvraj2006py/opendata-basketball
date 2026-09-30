# Stage 0 data audit — Passing Windows

**Freeze ID:** `passing_windows_stage0_20250924`  
**Audit date:** 2026-09-24  
**Companion manifest:** [`input_manifest.json`](input_manifest.json)

This audit inventories what exists on disk and documents known-issue spot checks. It does **not** produce basketball findings.

---

## 1. Verdict

| Check | Result |
|---|---|
| All 10 games present | **PASS** — each has `game_data.json`, `dynamic_events.json`, `tracking_data.jsonl.gz` |
| Git LFS pointer stubs | **PASS** — none detected; all tracking files are real gzip payloads |
| SHA-256 inventory | **PASS** — 35 input files hashed in `input_manifest.json` |
| Hash mismatches vs published digest list | **N/A** — no vendor checksum file in repo; Stage 0 establishes the baseline digest |
| Aggregates present but forbidden as covariates | **NOTED** — all 3 CSVs present and inventoried |

**Blockers for Stage 1 data layer:** none from missing files. Environment still needs `pygam` / PyYAML pins when modeling starts.

---

## 2. Expected inputs vs presence

### Global

| File | Present | Size (bytes) | Role |
|---|---|---|---|
| `data/matches.json` | yes | 4,526 | Match index |
| `data/player_id_aliases.csv` | yes | 1,081 | Duplicate player-id map (18 rows / 9 persons) |
| `data/aggregates/acb_shotsaggregates_20252026.csv` | yes | 601,645 | **FORBIDDEN covariate** |
| `data/aggregates/acb_drivesaggregates_20252026.csv` | yes | 162,374 | **FORBIDDEN covariate** |
| `data/aggregates/acb_picksaggregates_20252026.csv` | yes | 840,190 | **FORBIDDEN covariate** |

### Per game (all 10)

| gameId | game_data | dynamic_events | tracking jsonl.gz | LFS pointer? |
|---|---|---|---|---|
| 114243 | yes (2,253 B) | yes (3.5 MB) | yes (33.7 MB) | no |
| 114234 | yes (2,130 B) | yes (3.5 MB) | yes (31.7 MB) | no |
| 114169 | yes (2,098 B) | yes (3.5 MB) | yes (45.6 MB) | no |
| 114099 | yes (2,225 B) | yes (3.4 MB) | yes (33.1 MB) | no |
| 114086 | yes (2,168 B) | yes (3.3 MB) | yes (39.8 MB) | no |
| 178442 | yes (2,271 B) | yes (3.6 MB) | yes (39.1 MB) | no |
| 179612 | yes (2,221 B) | yes (3.4 MB) | yes (36.2 MB) | no |
| 184439 | yes (2,260 B) | yes (3.3 MB) | yes (33.8 MB) | no |
| 188630 | yes (2,260 B) | yes (3.2 MB) | yes (31.9 MB) | no |
| 191313 | yes (2,294 B) | yes (3.3 MB) | yes (35.9 MB) | no |

Tracking spot-checks: first JSONL line of each gzip decompresses and parses. Full frame counts sampled for two games (114243: 146,870 frames; 191313: 143,045 frames). Roughly half of frames are dead-time empties, as documented in the primer.

Exact SHA-256 digests: see `input_manifest.json`.

---

## 3. Dynamic-events sample sizes (10-game totals)

| Family | Count |
|---|---|
| chances | 2,055 |
| chances with `usable==true` (`qualityIndex≥3`) | 1,911 |
| possessions | 1,564 |
| touches | 6,662 |
| passes | 4,786 |
| shots | 1,459 |
| free_throws | 430 |
| turnovers | 248 |
| matchups | 14,416 |
| drives | 660 |
| picks | 1,520 |

### `qualityIndex` histogram (chances)

| QI | Count |
|---|---|
| 0 | 39 |
| 1 | 45 |
| 2 | 60 |
| 3 | 170 |
| 4 | 574 |
| 5 | 1,167 |

### Approximate primary-population touch pool (pre–Stage 1 frame filters)

Touches whose parent chance is `usable`, non-transition, has 5+5 player ids, and has a non-null `frontcourtFrame`:

| gameId | usable chances | primary-ish touches | passes | self-passes | toReceiverId≠null |
|---|---|---|---|---|---|
| 114243 | 196 | 577 | 525 | 6 | 8 |
| 114234 | 190 | 495 | 461 | 8 | 14 |
| 114169 | 200 | 573 | 460 | 6 | 6 |
| 114099 | 195 | 527 | 469 | 8 | 12 |
| 114086 | 184 | 549 | 510 | 9 | 12 |
| 178442 | 202 | 559 | 475 | 13 | 9 |
| 179612 | 188 | 628 | 533 | 14 | 6 |
| 184439 | 192 | 543 | 448 | 6 | 5 |
| 188630 | 172 | 561 | 472 | 11 | 8 |
| 191313 | 192 | 493 | 433 | 10 | 9 |
| **Total** | **1,911** | **~5,505** | **4,786** | **91** | **89** |

**Caveat:** “primary-ish” is a chance-level filter only. Stage 1 must still apply live-clock, frontcourt-at-frame, ten-players-at-frame, inbound, substitution, and tracking-error filters. Final CONSORT counts will be lower.

---

## 4. Known-issues inventory (frozen into handling rules)

| Known issue | Spot-check in 10 games | Stage 0 handling lock |
|---|---|---|
| `toReceiverId` = interceptor | 89 non-null; 0 on complete passes; equals incomplete+turnover count | Never use as intended target; infer targets with uncertainty |
| Self-passes | 91 | Exclude / flag |
| `ptsScored` team-score unreliability | Docs: season-wide issue; 10 games close exactly but still **do not** use for outcomes | Reconstruct from `shots` + `free_throws` |
| Zero elapsed game clock | 21 chances | Flag / exclude |
| Bad frame order | 3 chances | Exclude |
| Assisted shot without `assistOpp` | 38 | Noise flag; not a Stage 0 sample filter |
| Multi-pass touches | 6 | Audit; split or exclude deterministically |
| Player ID duplicates | 18 alias rows / 9 persons | Alias for person merges; keep raw IDs for joins |
| Season aggregates leakage | 3 CSVs on disk | **Forbidden** as contemporaneous covariates |

Additional pass structure notes:

- Incomplete + turnover passes: 89 (matches `toReceiverId` non-null)
- Incomplete non-turnover: 68 (need Stage 1 classification; not treated as interceptor targets)
- Inbounds passes: 807 (excluded from primary)

---

## 5. Exclusion inventory (design → audit trail)

Stage 1 must produce a CONSORT-style flow. Reasons already frozen:

1. `usable == false`
2. `transition == true` (primary)
3. Dead clock / stopped clock frames
4. Pre-frontcourt / backcourt frames
5. Fewer than ten tracked players or missing ball-handler
6. Inbounds
7. Substitution boundaries
8. Self-passes
9. Malformed chance/touch frame order
10. Zero-elapsed chances
11. Tracking error above fold-declared caps
12. Multi-pass touches (split or drop after audit)
13. Counterfactuals outside training distance support (model stage)

Risk: exclusions concentrated by period, score, team, or shot clock — must be checked in Stage 1 quality report.

---

## 6. Risks and unknowns (honest)

| Risk | Status |
|---|---|
| Final primary N after frame-level filters | Unknown until Stage 1 |
| Hoop orientation / leftHoop consistency | Must verify before mirroring |
| Failed-pass target resolvability | Unknown; kill at >10% unresolved |
| `predError` interpretation for Monte Carlo | Multiple radial interpretations in plan; sensitivity required |
| Package lockfile | Stage 0 snapshot only; `pygam` / PyYAML not installed on audit host |
| No vendor checksum file | Manifest digests are the baseline going forward |

---

## 7. What Stage 0 did not do

- No feature extraction beyond counts needed for inventory
- No model fitting
- No window segmentation
- No basketball claims or headline selection

Proceed to Stage 1 only after user verification of [`STAGE0_VERIFICATION.md`](STAGE0_VERIFICATION.md).
