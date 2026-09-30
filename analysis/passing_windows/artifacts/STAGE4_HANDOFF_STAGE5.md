# Stage 4 → Stage 5 handoff

**Status:** Stage 4 component models complete (`stage4_sklearn_logo_v3`).

## Primary table Stage 5 must consume

`tables/stage4_candidate_predictions.parquet`

Join keys: `(gameId, touchId, frameIdx, candidateId)` — aligns 1:1 with
`tables/stage3_candidate_features.parquet`.

### Required score columns

| Column | Meaning |
|---|---|
| `q_passability_model` | Calibrated P(complete to j \| state) |
| `q_ablation_*` | Ablation probabilities |
| `V_catch_model` | Stacked E[next-3s points \| projected catch]; observed-catch train |
| `V_catch_ablation_ridge_splines` / `_hist_gbrt` / `_two_stage` | Family components |
| `V_catch_ablation_all_candidates` | All-candidate ridge sensitivity |
| `V_keep_model` | E[next-3s points \| keep / no-pass state] |
| `V_fail` | Primary 0 |
| `Q_option_model` | q·V_catch (NaN if rejected_outside_support) |
| `NOV_model` | Q − V_keep (NaN if rejected) |
| `choice_prob_model` | Softmax choice prob at release (else NaN) |
| `fold_id` | Outer LOGO fold |
| `model_version` | Model version flag |

- Rows: **190348**
- Held-out passability log loss: **0.3475089980885733**
- Choice rank accuracy: **0.7566522629065272**

## Dual sampling (unchanged from Stage 3)

- Window series: `is_model_5hz_frame`
- Passability / choice labels: `completion_label_eligible` / release frames

## Stage 5 scope

Smooth cross-fitted `q` / `V_catch` / `NOV` series and segment windows.
Do not refit component models. Open threshold remains TBD_TRAINING_FOLD.

## Binding constraints carried forward

1. Narrow-path completion labels only.
2. Mask `rejected_outside_support` for counterfactual claims.
3. Report q and V_catch separately before collapsing to NOV.
4. Nested LOGO only.
