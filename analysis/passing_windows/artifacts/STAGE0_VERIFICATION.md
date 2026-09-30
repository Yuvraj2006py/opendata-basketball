# Stage 0 USER VERIFICATION CHECKLIST

**Freeze ID:** `passing_windows_stage0_20250924`  
**Purpose:** Approve or reject every locked Stage 0 decision before Stage 1 begins.  
**Artifacts to review:**

- [`../configs/stage0_analysis_spec.md`](../configs/stage0_analysis_spec.md)
- [`../configs/stage0_freeze.yaml`](../configs/stage0_freeze.yaml)
- [`input_manifest.json`](input_manifest.json)
- [`stage0_data_audit.md`](stage0_data_audit.md)

Mark each item **APPROVE** / **REJECT** / **CHANGE REQUEST** (with note).

---

## A. Thesis and claims

| # | Locked decision | Verify |
|---|---|---|
| A1 | Thesis: possessions are sequences of disappearing passing windows with measurable risk/value/lifetime/predictive content | ☐ |
| A2 | Allowed claims: within-touch variation; q ≠ value; finite lifetimes/usage patterns; option sets improve held-out near-term prediction | ☐ |
| A3 | Forbidden claims: causal “would have scored more”; unused ⇒ mistake; player/team rankings; universal “correct pass” | ☐ |
| A4 | Required language: model-preferred / available under the model / unused — never correct pass / bad decision / points left on the table | ☐ |

---

## B. Populations

| # | Locked decision | Verify |
|---|---|---|
| B1 | Primary: live, frontcourt, non-transition touches in `usable==true` chances with all ten players present | ☐ |
| B2 | Primary unit is **chance**-nested touches, not full possession | ☐ |
| B3 | Secondary: transition; lower-quality tracking; inbounds; full-possession unit | ☐ |
| B4 | Primary excludes inbounds and substitution boundaries | ☐ |

---

## C. Temporal sampling

| # | Locked decision | Verify |
|---|---|---|
| C1 | Model at **5 Hz**; render at **25 Hz** | ☐ |
| C2 | Velocities/smoothing causal (no future frames) | ☐ |

---

## D. Outcomes and baselines

| # | Locked decision | Verify |
|---|---|---|
| D1 | Primary outcome: next-3s offensive points in same chance from `shots` + `free_throws` | ☐ |
| D2 | Never use `chances.ptsScored` as outcome/score source | ☐ |
| D3 | Secondary outcomes as listed (open shot / any shot / paint / terminal points; `shotQuality` validation-only) | ☐ |
| D4 | Predictive ladder and passability ablation baselines as frozen | ☐ |

---

## E. Scores and formulas

| # | Locked decision | Verify |
|---|---|---|
| E1 | \(q(j,t)=P(\text{reach }j\mid\text{release-time state})\) | ☐ |
| E2 | \(Q=q\,V_{\text{catch}}+(1-q)\,V_{\text{fail}}\) with primary \(V_{\text{fail}}=0\) | ☐ |
| E3 | \(\mathrm{NOV}=Q-V_{\text{keep}}\) | ☐ |
| E4 | Present q and \(V_{\text{catch}}\) before collapsing to NOV | ☐ |

---

## F. Windows

| # | Locked decision | Verify |
|---|---|---|
| F1 | Min persistence ≈ **0.20 s** | ☐ |
| F2 | Separate open/close thresholds (hysteresis) | ☐ |
| F3 | Open requires passability > open threshold **and** NOV > 0 | ☐ |
| F4 | Threshold values are `TBD_TRAINING_FOLD` with fold-internal selection + robustness grid | ☐ |

---

## G. Exclusions and target inference

| # | Locked decision | Verify |
|---|---|---|
| G1 | `toReceiverId` is interceptor — never intended target | ☐ |
| G2 | Self-passes excluded/flagged | ☐ |
| G3 | Malformed / zero-elapsed chances excluded/flagged | ☐ |
| G4 | Multi-pass touches audited then split or excluded | ☐ |
| G5 | Failed-pass targets inferred with probabilities; auto-accept margin `TBD_TRAINING_FOLD` | ☐ |
| G6 | Unresolved > 10% or poor audit ⇒ kill/narrow | ☐ |
| G7 | Season aggregates **forbidden** as contemporaneous covariates | ☐ |
| G8 | Player aliases for person merges; preserve original IDs for joins | ☐ |

---

## H. Validation and seeds

| # | Locked decision | Verify |
|---|---|---|
| H1 | Nested leave-one-game-out only; no random frame/touch/chance splits | ☐ |
| H2 | Ten outer folds = one per held-out gameId as listed in freeze YAML | ☐ |
| H3 | Master seed `20250924`; fold seeds `2025092401`…`2025092410` | ☐ |
| H4 | MC perturbation seed `2025092499`; example-selection seed `2025092488` | ☐ |
| H5 | Analyses 1–7 confirmatory; Analysis 8 exploratory | ☐ |

---

## I. Kill criteria

| # | Criterion | Verify |
|---|---|---|
| I1 | Unresolved failed targets > 10% / poor audit | ☐ |
| I2 | > 20% eligible states fail tracking | ☐ |
| I3 | Calibration slope outside ~0.75–1.25 / poor reliability | ☐ |
| I4 | > 10% counterfactuals outside training support | ☐ |
| I5 | Window labels change > 15% under tracking perturbation | ☐ |
| I6 | Predictive improvement fails in ≥ 7/10 games (need ≥ 7/10 wins + beat nulls) | ☐ |
| I7 | Result comparable to nulls | ☐ |
| I8 | Single-game driver or tracking restriction reverses headline | ☐ |

---

## J. Data inventory acceptance

| # | Finding | Verify |
|---|---|---|
| J1 | All 10 games have game_data + dynamic_events + tracking gzip | ☐ |
| J2 | No LFS pointer stubs | ☐ |
| J3 | SHA-256 manifest in `input_manifest.json` accepted as baseline | ☐ |
| J4 | Approximate pool ~1,911 usable chances / ~5,505 primary-ish touches acknowledged as pre-frame-filter | ☐ |
| J5 | Spot-checks: 91 self-passes, 89 `toReceiverId`, 807 inbounds, 21 zero-elapsed, 3 bad frame order | ☐ |
| J6 | Host package snapshot accepted; full lockfile deferred to Stage 1; pygam/PyYAML currently missing | ☐ |

---

## K. Coordinates and games

| # | Locked decision | Verify |
|---|---|---|
| K1 | Feet; origin court center; event hoop on −x; attack left = +y | ☐ |
| K2 | Normalize attack direction after hoop verification; mirroring tests required | ☐ |
| K3 | Game IDs: 114243, 114234, 114169, 114099, 114086, 178442, 179612, 184439, 188630, 191313 | ☐ |

---

## Sign-off

| Role | Name | Date | Overall |
|---|---|---|---|
| User / analyst | | | APPROVE / REJECT |

**If REJECT or any CHANGE REQUEST:** do not start Stage 1; revise freeze artifacts and re-issue Stage 0.

**If APPROVE:** Stage 1 may begin (canonical data layer, manifests, joins, coordinate normalization, quality audit).
