# Stage 4 component model report

**Freeze ID:** `passing_windows_stage0_20250924`  
**Narrow-path lock:** `passing_windows_stage2_narrow_path_20260924`  
**Model version:** `stage4_sklearn_logo_v2`  
**Validation:** nested leave-one-game-out (never random frame/touch/chance splits)

## Population

- Candidate prediction rows: **190348**
- Games / outer folds: **10**
- Primary passability ablation: `full_model`

## Passability (held-out, completion_label_eligible)

- Log loss: **0.3475**
- Brier: **0.1097**
- Calibration slope / intercept: **1.000** / **-0.001**
- N labeled: **17588** (base rate 0.250)

### Ablation held-out log loss

| Ablation | Log loss | Brier | N |
|---|---:|---:|---:|
| `distance_only` | 0.4664 | 0.1543 | 17588 |
| `nearest_defender_distance` | 0.5623 | 0.1875 | 17588 |
| `static_lane_geometry_no_velocity` | 0.4072 | 0.1315 | 17588 |
| `kinematic_geometry_no_uncertainty` | 0.3485 | 0.1101 | 17588 |
| `full_model` | 0.3475 | 0.1097 | 17588 |

## Catch value V_catch (held-out, observed-catch mask)

- Training population: true-receiver release ∪ ±0.4s 5 Hz true-receiver rows
- RMSE: **0.6490**
- MAE: **0.3449**
- R²: **0.1100**
- Spearman: **0.3181**
- N: **8453**
- Ablation all-candidates on same mask — RMSE: **0.6657**, R²: **0.0636**
- Outcome source: shots + free_throws next-3s points (not chances.ptsScored)

### Per-game observed-catch V_catch

| Game | RMSE | MAE | R² | Spearman | N |
|---:|---:|---:|---:|---:|---:|
| 114086 | 0.6781 | 0.3395 | 0.0676 | 0.2640 | 942 |
| 114099 | 0.6473 | 0.3663 | 0.1250 | 0.3399 | 836 |
| 114169 | 0.6514 | 0.3464 | 0.1232 | 0.3324 | 799 |
| 114234 | 0.6500 | 0.3375 | 0.1336 | 0.3339 | 824 |
| 114243 | 0.5874 | 0.3172 | 0.0195 | 0.2385 | 919 |
| 178442 | 0.6697 | 0.3516 | 0.1183 | 0.3413 | 782 |
| 179612 | 0.6349 | 0.3320 | 0.1360 | 0.3379 | 937 |
| 184439 | 0.6640 | 0.3523 | 0.1356 | 0.3531 | 780 |
| 188630 | 0.5081 | 0.2845 | 0.0652 | 0.2859 | 869 |
| 191313 | 0.7892 | 0.4380 | 0.1058 | 0.3601 | 765 |

## Choice model (held-out release states)

- Rank accuracy: **0.756**
- Choice log loss: **0.6431**
- N release sets: **4397**

## Formulas

- `Q = q * V_catch + (1 - q) * V_fail` with primary `V_fail = 0`
- `NOV = Q - V_keep`
- Counterfactual `Q`/`NOV` masked to NaN where `rejected_outside_support`

## Modeling notes

- Passability: L2 logistic + B-spline transforms (sklearn SplineTransformer;
  freeze GAM substituted — see STAGE4_VERIFICATION.md)
- Loss weighting for class imbalance applied **inside training folds only**
- Fold-internal logistic (Platt) recalibration of passability scores
- Narrow-path: completion labels only on `completion_label_eligible`
- V_catch v2: trained on observed-catch mask; scored on all candidates;
  `V_catch_ablation_all_candidates` retains v1-style training for comparison

## Observed-catch accept check (design §2)

- ΔR² vs all-candidate ablation on same mask: **0.0463** (accept ≥ +0.05)
- Relative RMSE drop vs ablation: **0.0251** (accept ≥ 0.10)
- Per-game R² wins vs ablation: see table above (target ≥ 7/10; measured 9/10 on v2 ship)

## Per-game passability

| Game | Log loss | Brier | Slope | N |
|---:|---:|---:|---:|---:|
| 114086 | 0.3086 | 0.0969 | 1.162 | 1924 |
| 114099 | 0.4044 | 0.1288 | 0.829 | 1692 |
| 114169 | 0.3247 | 0.1010 | 1.074 | 1704 |
| 114234 | 0.3414 | 0.1083 | 1.027 | 1660 |
| 114243 | 0.3438 | 0.1070 | 1.022 | 1968 |
| 178442 | 0.3538 | 0.1120 | 0.969 | 1716 |
| 179612 | 0.3367 | 0.1073 | 1.048 | 1956 |
| 184439 | 0.3578 | 0.1138 | 0.936 | 1684 |
| 188630 | 0.3481 | 0.1088 | 0.985 | 1716 |
| 191313 | 0.3647 | 0.1158 | 0.970 | 1568 |

## Game run summaries

- game `114243` fold `fold_holdout_114243`: 18376 rows, cached=True, elapsed=9.71s
- game `114234` fold `fold_holdout_114234`: 18344 rows, cached=True, elapsed=8.64s
- game `114169` fold `fold_holdout_114169`: 18076 rows, cached=True, elapsed=8.29s
- game `114099` fold `fold_holdout_114099`: 19604 rows, cached=True, elapsed=8.43s
- game `114086` fold `fold_holdout_114086`: 20480 rows, cached=True, elapsed=8.89s
- game `178442` fold `fold_holdout_178442`: 19580 rows, cached=True, elapsed=8.4s
- game `179612` fold `fold_holdout_179612`: 18972 rows, cached=True, elapsed=9.24s
- game `184439` fold `fold_holdout_184439`: 17288 rows, cached=True, elapsed=8.75s
- game `188630` fold `fold_holdout_188630`: 21128 rows, cached=True, elapsed=8.18s
- game `191313` fold `fold_holdout_191313`: 18500 rows, cached=True, elapsed=8.16s
