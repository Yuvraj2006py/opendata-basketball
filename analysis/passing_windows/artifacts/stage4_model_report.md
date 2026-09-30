# Stage 4 component model report

**Freeze ID:** `passing_windows_stage0_20250924`  
**Narrow-path lock:** `passing_windows_stage2_narrow_path_20260924`  
**Model version:** `stage4_sklearn_logo_v3`  
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
- Primary predictor: stacked ridge + HistGBRT + two-stage shot composition
- RMSE: **0.6458**
- MAE: **0.3346**
- R²: **0.1188**
- Spearman: **0.3230**
- N: **8453**
- Ablation all-candidates ridge on same mask — RMSE: **0.6657**, R²: **0.0636**
- Outcome source: shots + free_throws next-3s points (not chances.ptsScored)

### Family ablations on observed-catch mask

| Family | RMSE | MAE | R² | Spearman | N |
|---|---:|---:|---:|---:|---:|
| `ridge_splines` | 0.6490 | 0.3449 | 0.1100 | 0.3181 | 8453 |
| `hist_gbrt` | 0.6516 | 0.3235 | 0.1029 | 0.3061 | 8453 |
| `two_stage` | 0.6517 | 0.3215 | 0.1027 | 0.3119 | 8453 |

### Per-game observed-catch V_catch

| Game | RMSE | MAE | R² | Spearman | N |
|---:|---:|---:|---:|---:|---:|
| 114086 | 0.6695 | 0.3270 | 0.0909 | 0.2778 | 942 |
| 114099 | 0.6379 | 0.3571 | 0.1502 | 0.3609 | 836 |
| 114169 | 0.6455 | 0.3325 | 0.1391 | 0.3449 | 799 |
| 114234 | 0.6553 | 0.3359 | 0.1193 | 0.3275 | 824 |
| 114243 | 0.5898 | 0.3060 | 0.0113 | 0.2448 | 919 |
| 178442 | 0.6651 | 0.3398 | 0.1305 | 0.3534 | 782 |
| 179612 | 0.6331 | 0.3195 | 0.1408 | 0.3421 | 937 |
| 184439 | 0.6661 | 0.3423 | 0.1302 | 0.3401 | 780 |
| 188630 | 0.4988 | 0.2720 | 0.0993 | 0.2887 | 869 |
| 191313 | 0.7861 | 0.4312 | 0.1127 | 0.3600 | 765 |

## Choice model (held-out release states)

- Rank accuracy: **0.757**
- Choice log loss: **0.6424**
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
- V_catch v3: observed-catch training + stacked ridge/HistGBRT/two-stage
  (inner-LOGO non-negative blend); `V_catch_ablation_*` retain components

## Observed-catch accept check (design §2)

- ΔR² vs all-candidate ablation on same mask: **0.0552** (accept ≥ +0.05)
- Relative RMSE drop vs ablation: **0.0299** (accept ≥ 0.10)
- Per-game R² wins vs ablation: see table above (target ≥ 7/10)

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

- game `114243` fold `fold_holdout_114243`: 18376 rows, cached=True, elapsed=29.22s
- game `114234` fold `fold_holdout_114234`: 18344 rows, cached=True, elapsed=25.67s
- game `114169` fold `fold_holdout_114169`: 18076 rows, cached=True, elapsed=24.24s
- game `114099` fold `fold_holdout_114099`: 19604 rows, cached=True, elapsed=24.28s
- game `114086` fold `fold_holdout_114086`: 20480 rows, cached=True, elapsed=23.52s
- game `178442` fold `fold_holdout_178442`: 19580 rows, cached=True, elapsed=23.76s
- game `179612` fold `fold_holdout_179612`: 18972 rows, cached=True, elapsed=23.26s
- game `184439` fold `fold_holdout_184439`: 17288 rows, cached=True, elapsed=24.18s
- game `188630` fold `fold_holdout_188630`: 21128 rows, cached=True, elapsed=33.31s
- game `191313` fold `fold_holdout_191313`: 18500 rows, cached=True, elapsed=43.05s
