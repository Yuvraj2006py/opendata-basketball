# Stage 4 verification

**Model version:** `stage4_sklearn_logo_v2`

## Freeze faithfulness

- [x] Nested leave-one-game-out only (no random frame/touch/chance splits)
- [x] Passability trained on `completion_label_eligible`; label = `is_true_receiver`
- [x] Predictions written for all candidate rows (5 Hz + release)
- [x] Ablations: distance_only, nearest_defender_distance,
      static_lane_geometry_no_velocity, kinematic_geometry_no_uncertainty, full_model
- [x] Imbalance loss weights inside train folds only
- [x] Fold-internal logistic recalibration
- [x] V_catch from shots+free_throws next-3s points (not chances.ptsScored)
- [x] V_catch features at projected catch only
- [x] V_catch trained on observed-catch mask (true receiver ±0.4s); scored on all candidates
- [x] V_keep from no-pass / keep states, causal features only
- [x] Q = q*V_catch; V_fail = 0; NOV = Q - V_keep
- [x] Choice model at pass-release over four candidates (descriptive)
- [x] `rejected_outside_support` masked for counterfactual Q/NOV
- [x] Never use interceptorId as intended target
- [x] Never use season aggregates as covariates
- [x] Trajectory choice honored (not score-max)

## GAM → sklearn substitution

The freeze specifies a regularized logistic GAM. This Stage 4 ship uses
`sklearn.preprocessing.SplineTransformer` (B-splines, 5 knots, degree 3)
on distance, clearance, temporal margin, separation, pressure, and
`predError`, followed by L2 `LogisticRegression` / `Ridge`. This is a
documented pragmatic substitute; coefficients are not GAM smooths.

## V_catch training population (v2)

`stage4_sklearn_logo_v2` trains `V_catch` on observed true-receiver states
(completion-eligible release ∪ `is_model_5hz_frame` within ±0.4s on the
true-receiver trajectory). Predictions remain on all candidate rows.
`V_catch_ablation_all_candidates` retains the prior all-candidate training
recipe for sensitivity comparison on the observed-catch evaluation mask.

## Key metrics snapshot

- Rows: 190348
- Passability log loss: 0.3475089980885733
- Passability Brier: 0.10967549806288553
- V_catch observed-catch R²: 0.10995125650392956
- V_catch observed-catch RMSE: 0.6490332835124624
- Choice rank accuracy: 0.7557425517398226

## Residual risks

- Secondary catch outcomes use linear probability Ridge, not multinomial GAM.
- Choice model is a conditional-softmax over independent utilities (descriptive).
- Catch/keep training may subsample large training folds for runtime.
- Observed-catch labels remain associative (actual possession path), not causal
  counterfactuals for unused options.
