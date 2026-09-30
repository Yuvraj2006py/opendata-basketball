# Stage 4 V_catch observed-catch training — design

**Date:** 2026-09-30  
**Status:** approved for planning (pending user review of this file)  
**Model version target:** `stage4_sklearn_logo_v2`  
**Scope:** improve Stage 4 post-catch value (`V_catch` / NOV) by fixing the training population; no Stage 5, passability, or choice changes.

## Problem

`stage4_sklearn_logo_v1` reports held-out `V_catch` R² ≈ 0.033 (RMSE ≈ 0.69). Passability is already strong (log loss ≈ 0.35; full vs kinematic nearly identical). The value bottleneck is largely **label noise**: v1 trains on (almost) all candidate rows with next-3s points at the projected catch frame, even when that catch never occurs. Most labels are “what the possession did anyway,” not “value if this receiver caught here.”

## Decisions locked

| Decision | Choice |
|---|---|
| Priority component | `V_catch` / NOV |
| Training philosophy | Observed catches only, then score all candidates |
| Nearby expansion | True-receiver trajectory within **±0.4s** of release (5 Hz + release) |
| First approach | **Minimal:** change training mask only; keep Ridge + B-splines + current features |
| Escalate if accept fails | Approach 2 (shot / open-shot decomposition) before stronger learners |

## §1 Training population and labels

### Train mask (per outer LOGO fold)

Include a candidate row if **all** of the following hold:

1. The candidate is the **true receiver** for that pass (`is_true_receiver` / `true_receiverId` join as already used by Stage 4).
2. The row is either:
   - a **pass-release** row with `completion_label_eligible == True`, or
   - an `is_model_5hz_frame` row on the same `(gameId, touchId, true_receiverId)` with  
     `|frameIdx - release_frameIdx| / 25 ≤ 0.4`.

Exclude other candidates and frames outside the window.

### Target

Unchanged from freeze:

- Primary: `catch_points_next_3s` reconstructed from `shots` + `free_throws` only (never `chances.ptsScored`).
- Outcome window starts at projected catch: `frameIdx + round(flight_time_s * 25)`.

### Inference / scoring

Predict `V_catch_model` for **all** candidate rows (5 Hz + release), same column contract as Stage 4 → Stage 5 handoff. This remains a spatial / predictive surface under projected catch geometry; article language stays non-causal for unused options.

### Ablation

Primary ship: observed-catch training mask.  
Retain (or one-shot recompute) the v1-style all-candidate Ridge fit as a sensitivity comparator (`V_catch_ablation_all_candidates` or documented side table). Location-only and location+defender baselines from `CATCH_VALUE_BASELINES` remain.

## §2 Evaluation and accept criteria

### Primary metrics

Computed on the **observed-catch test mask** (same definition as train, on holdout game), LOGO-pooled and per-game:

- RMSE, MAE, R² vs `catch_points_next_3s`
- Reliability: binned mean predicted vs mean actual
- Within-game Spearman of `V_catch` vs realized points

### Comparators

- Current all-candidate Ridge evaluated on the **same** observed-catch test rows
- `location_only` and `location_defender` baselines

### Accept (stay on approach 1; do not escalate yet)

Any of the following, with improvement in **≥ 7 / 10** games for the primary metric used:

- Held-out R² on observed-catch rises by **≥ +0.05** absolute vs v1-style all-candidate model on that mask, **or**
- RMSE drops by **≥ 10%** relative vs that comparator

### Escalate to approach 2

If accept fails **and** location baselines still show R² > 0 on observed-catch (signal exists; linear full state underfits / target still hard).

### Unchanged reporting rules

- Nested leave-one-game-out only (no random frame/touch/chance splits)
- Report `q` and `V_catch` separately before NOV
- Mask counterfactual `Q` / `NOV` where `rejected_outside_support`
- Do not claim unused options would have produced the observed points

## §3 Implementation shape

### Versioning

Bump model version string to `stage4_sklearn_logo_v2`. Document v1 → v2 as **training-population change only** (not a GAM claim change).

### Code touchpoints

| Area | Change |
|---|---|
| `src/passing_windows/models/catch_value.py` | Add `observed_catch_training_mask(...)`; train `fit_catch_value` on that mask by default |
| `pipelines/04_fit_component_models.py` | Emit §2 metrics; wire version `v2`; optional all-candidate ablation column/table |
| `tests/test_stage4.py` | Mask unit tests; leakage unchanged; fold-metrics keys for observed-catch R²/RMSE |
| Artifacts / verification | Update model report + `STAGE4_VERIFICATION.md` notes on population |

### Explicitly unchanged

- `CATCH_VALUE_FEATURES` and Ridge + `SplineTransformer` pipeline
- Passability, choice, `V_keep`, Q/NOV formulas (`V_fail = 0`)
- Stage 5 smoothing / windowing
- Forbidden feature / leakage guards

### Tests (minimum)

1. Mask includes release true receiver when `completion_label_eligible`.
2. Mask includes true-receiver 5 Hz rows inside ±0.4s and excludes outside.
3. Mask never includes non-true-receiver candidates.
4. Features still exclude forbidden columns (`shotQuality` as covariate, etc.).
5. Fold metrics JSON exposes observed-catch held-out R²/RMSE keys.

## Out of scope

- Approach 2 (shot / open-shot decomposition) and approach 3 (HistGBRT / real GAM)
- Rewriting `V_keep` population
- Expanding beyond 10-game LOGO or adding season aggregates
- Causal claims for counterfactual catches

## Success definition for this design

A landed `stage4_sklearn_logo_v2` with documented observed-catch training, regression tests for the mask, and a model report that either (a) meets §2 accept criteria or (b) clearly fails them and triggers the pre-agreed escalate path to approach 2.
