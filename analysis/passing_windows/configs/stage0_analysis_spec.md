# Stage 0 Analysis Specification — Passing Windows

**Freeze ID:** `passing_windows_stage0_20250924`  
**Freeze date:** 2026-09-24  
**Status:** FROZEN (design lock only; no basketball findings)  
**Machine-readable twin:** [`stage0_freeze.yaml`](stage0_freeze.yaml)  
**Plan:** [`.cursor/plans/passing-windows-study_aff10b97.plan.md`](../../../.cursor/plans/passing-windows-study_aff10b97.plan.md)

This document locks study design before headline analysis. It does not claim empirical basketball results.

---

## 1. Thesis (locked)

Basketball possessions are sequences of disappearing passing windows. At each moment, the ball-handler has a changing set of receiver options with distinct delivery risk and post-catch value; these options can be measured from tracking, have finite lifetimes, and contain predictive information beyond ordinary ball, clock, and spacing features.

---

## 2. Claims allowed vs forbidden (locked)

### May support
- Passing opportunities vary materially within a touch rather than being static properties of court location.
- Passability and post-catch value are distinct dimensions: an easy pass is not necessarily valuable, and an ambitious pass is not necessarily available.
- High-value windows have measurable lifetimes and are sometimes used before peak, near peak, after peak, or not used.
- The quality and breadth of the live option set improve held-out prediction of near-term offensive outcomes.

### Will NOT claim
- A modeled alternative pass would causally have produced more points.
- An unused modeled option proves a player made a mistake.
- Ten games support stable player or team rankings.
- A single scalar represents the universally “correct” pass.

**Required article language:** “model-preferred,” “available under the model,” and “unused” — never “correct pass,” “bad decision,” or “points left on the table.”

---

## 3. Primary population (locked)

**Primary unit:** touches nested in **chances** (not full possessions).

A decision state / touch is eligible for the primary sample only if all of the following hold:

| Criterion | Rule |
|---|---|
| Chance quality | `chances.usable == true` (`qualityIndex >= 3`) |
| Transition | `chances.transition == false` |
| Clock | Live clock at modeled frames (`gameClockStopped == false`) |
| Court | Frontcourt: `frameIdx >= chance.frontcourtFrame` and ball-handler in frontcourt |
| Lineup / tracking | All ten players present and an identifiable ball-handler |
| Inbounds | Not an inbound throw-in context |
| Substitutions | No substitution boundary spanning the modeled interval |
| Frame integrity | Valid touch and chance frame ordering |

Secondary populations (sensitivity / context only):

1. **Transition** — same rules with `transition == true`
2. **Lower-quality tracking** — include `usable == false` and/or higher `predError` strata
3. **Inbounds** — `passes.inbounds == true`
4. **Full possession** — analysis unit = possession rather than chance

---

## 4. Temporal sampling (locked)

| Role | Rate | Notes |
|---|---|---|
| Model candidate states | **5 Hz** | Stride 5 on the native 25 fps timeline |
| Render examples / article | **25 Hz** | Native tracking rate |

Velocities and smoothing use **current and past frames only** (no future leakage).

---

## 5. Primary outcomes and baselines (locked)

### Primary outcome
Expected **offensive points in the next 3 seconds of the same chance**, reconstructed from `shots` and `free_throws` only.

### Secondary outcomes
- Open / lightly contested shot within 3s
- Any shot within 3s
- Paint touch within 3s
- Terminal chance points (longer-horizon sensitivity)
- Observed `shotQuality` **only** as a validation target when a shot occurs — **never** as a contemporaneous feature

### Baselines
**Predictive ladder (Analysis 6):** state baseline → + realized actions → + static candidate geometry → + passability summaries → + full option-set/window summaries.

**Passability ablation:** distance only → nearest-defender distance → static lane geometry → kinematic geometry without uncertainty → full model.

---

## 6. Core scores (locked formulas)

**Passability**

\[
q(j,t) = P(\text{pass reaches candidate } j \mid \text{release-time state})
\]

**Option value**

\[
Q(j,t) = q(j,t)\, V_{\text{catch}}(j,t) + \bigl[1 - q(j,t)\bigr]\, V_{\text{fail}}(t)
\]

- Primary \(V_{\text{fail}} = 0\)
- Optional possession-cost sensitivity later; do **not** invent opponent transition value from this 10-game sample

**Keep-state baseline** \(V_{\text{keep}}(t)\): predictive value of continuing the current touch from current/past information only (matched no-pass states).

**Net option value**

\[
\text{NOV}(j,t) = Q(j,t) - V_{\text{keep}}(t)
\]

**Presentation rule:** show \(q\) and \(V_{\text{catch}}\) separately before combining into NOV.

---

## 7. Window persistence and hysteresis (locked rules; thresholds TBD)

- Minimum persistence: **~0.20 seconds** (5 frames at 25 Hz)
- Separate **open** and **close** thresholds (hysteresis) to prevent one-frame flicker
- **Open** label (secondary thresholded label): passability exceeds fold-calibrated open threshold **and** `NOV > 0`, with persistence ≥ 0.20s
- Threshold values: `TBD_TRAINING_FOLD` — selected from calibration/decision considerations **inside each training fold only**, then all headline findings repeated on a prespecified threshold grid

Continuous window outputs (primary): opening/peak/closing time, duration, peak and integrated NOV, delay to actual release, value at use vs peak, window-existence probability under tracking perturbation.

Labels: open / used / late / unused / never-open.

---

## 8. Exclusion rules and failed-pass target inference (locked)

### Hard handling rules (from KNOWN_ISSUES + plan)

| Issue | Frozen handling |
|---|---|
| `passes.toReceiverId` | **Interceptor**, never intended receiver. Never substitute as target. |
| `chances.ptsScored` | Unreliable for team scoring / outcomes outside these games’ special cases. Reconstruct points from `shots` + `free_throws`. |
| Self-passes (`passerId == receiverId`) | Remove or separately flag (~91 in 10 games). |
| Malformed chances (`endFrame < startFrame`) | Exclude (~3). |
| Zero elapsed game clock | Flag / exclude from primary (~21). |
| Multi-pass touches | Audit; deterministically split or exclude (~6). |
| Season aggregate CSVs | **Inventoried only.** Forbidden as contemporaneous covariates. |
| Player ID aliases | Use `player_id_aliases.csv` for person-level merges; preserve original IDs for within-game joins. |

### Failed-pass intended-target inference
1. Form four teammate candidates at release.
2. Score with position / projected movement vs `receiverLoc`, ball trajectory when usable, pass direction, flight-time support.
3. Output a **probability distribution**, not only a hard label.
4. Auto-accept only high-confidence assignments with geometric margin over runner-up (`TBD_TRAINING_FOLD`).
5. Ambiguous cases → audit sheet; manual audit of all ambiguous + sample of auto-accepted/completed.
6. If reliability inadequate: failed passes only for touch-level turnover modeling; receiver-specific completion calibration restricted to reliable targets.

Kill line: unresolved failed-pass targets **> 10%** or poor audit reliability → narrow claims.

Tracking error caps for passer / receiver / lane defenders: `TBD_TRAINING_FOLD`, chosen so otherwise-eligible decision-state tracking failure stays under the 20% kill line.

---

## 9. Primary validation (locked)

**Nested leave-one-game-out.** Never random frame / touch / chance splits.

For each outer fold: fit, tune, threshold, and calibrate on the other nine games; write predictions only for the held-out game.

| Fold | Held-out `gameId` | Seed |
|---|---|---|
| fold_holdout_114243 | 114243 | 2025092401 |
| fold_holdout_114234 | 114234 | 2025092402 |
| fold_holdout_114169 | 114169 | 2025092403 |
| fold_holdout_114099 | 114099 | 2025092404 |
| fold_holdout_114086 | 114086 | 2025092405 |
| fold_holdout_178442 | 178442 | 2025092406 |
| fold_holdout_179612 | 179612 | 2025092407 |
| fold_holdout_184439 | 184439 | 2025092408 |
| fold_holdout_188630 | 188630 | 2025092409 |
| fold_holdout_191313 | 191313 | 2025092410 |

**Master RNG seed:** `20250924`  
Also frozen: Monte Carlo tracking perturbation seed `2025092499`; example-selection seed `2025092488`.

**Confirmatory:** Analyses 1–7. **Exploratory:** Analysis 8 context chapters (promote to headline only if stable across games and robustness specs).

---

## 10. Kill / narrow criteria (locked)

Narrow or stop claims if any of the following hold:

1. Unresolved failed-pass targets exceed **10%** or audit reliability is poor.
2. More than **20%** of otherwise eligible decision states fail tracking criteria.
3. Completion calibration slope outside roughly **0.75–1.25** or reliability is visibly poor.
4. Over **10%** of evaluated counterfactuals lie outside training support.
5. Window labels change by more than **15%** under plausible tracking perturbations.
6. Predictive improvement fails in at least **7 of 10** held-out games (central claim requires beating state baseline in ≥ 7/10 **and** exceeding matched nulls).
7. Result is comparable to time-shifted / permuted nulls.
8. One game drives the headline, or tracking-quality restrictions reverse it.

---

## 11. Game IDs, coordinates, packages (locked)

### Games (10)
`114243`, `114234`, `114169`, `114099`, `114086`, `178442`, `179612`, `184439`, `188630`, `191313`

### Coordinates
- Units: **feet**; origin at **court center**
- Event convention: offensive hoop on **negative x**; when attacking, left = **+y**, right = **−y**
- Analysis: normalize each possession to one attacking direction after hoop-orientation checks; mirroring tests required
- Tracking: 25 fps; `frameIdx` aligns with event `frame` / `startFrame` / `endFrame`

### Packages (Stage 0 host snapshot; full lockfile deferred to Stage 1)
| Package | Version seen |
|---|---|
| Python | 3.13.7 |
| numpy | 2.1.3 |
| pandas | 2.3.3 |
| scipy | 1.15.3 |
| scikit-learn | 1.8.0 |
| statsmodels | 0.14.5 |
| matplotlib | 3.10.0 |
| pygam | **missing** (to pin when installed) |
| PyYAML | **missing** (to pin when installed) |

---

## 12. Explicit data prohibition (locked)

**Do NOT use season aggregate CSVs as contemporaneous covariates.**

Files inventoried but forbidden as features:

- `data/aggregates/acb_shotsaggregates_20252026.csv`
- `data/aggregates/acb_drivesaggregates_20252026.csv`
- `data/aggregates/acb_picksaggregates_20252026.csv`

Reason: they contain information from outside and after the ten published games (293-game season rollups).

---

## 13. Confirmatory vs exploratory (locked)

| Class | Content |
|---|---|
| Confirmatory | Reliability/calibration; q vs V distinctness; option counts; lifetimes; usage timing; predictive falsification; used/late/unused descriptive comparisons |
| Exploratory | Context chapters (drives vs picks, paint windows, skips, closeouts, late clock) |

---

## 14. TBD_TRAINING_FOLD registry

Numeric values intentionally **not** frozen at Stage 0, but **selection procedures are**:

| Item | Selection procedure |
|---|---|
| Window open / close thresholds | Fold-internal calibration/decision curves; hysteresis; then fixed robustness grid |
| Failed-pass auto-accept geometric margin | Fold-internal audit agreement vs unresolved-rate tradeoff |
| `predError` caps | Fold-internal, keep tracking failure < 20% kill line |
| Pass distance support rejection | Training-fold empirical attempted-pass support |
| Class imbalance loss weights | Training-fold only |
| Late-use material NOV loss | Training-fold NOV quantile, predeclared before held-out labeling |

---

## 15. Artifact index for Stage 0

| Artifact | Path |
|---|---|
| This spec | `analysis/passing_windows/configs/stage0_analysis_spec.md` |
| Freeze YAML | `analysis/passing_windows/configs/stage0_freeze.yaml` |
| Input manifest (SHA-256) | `analysis/passing_windows/artifacts/input_manifest.json` |
| Data audit | `analysis/passing_windows/artifacts/stage0_data_audit.md` |
| User verification checklist | `analysis/passing_windows/artifacts/STAGE0_VERIFICATION.md` |

Stage 0 intentionally does **not** extract features, fit models, or assert basketball findings.
