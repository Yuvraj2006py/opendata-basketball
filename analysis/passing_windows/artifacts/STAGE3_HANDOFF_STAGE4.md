# Stage 3 → Stage 4 handoff

**Status:** Stage 3 candidate reconstruction complete under the Stage 2
narrow path, with **forced release-frame rows** for passability labeling.

**Freeze:** `passing_windows_stage0_20250924`  
**Narrow-path lock:** `passing_windows_stage2_narrow_path_20260924`  
**Trajectory rule:** `lead_if_velocity_ok_else_direct` (explicit; never score-max)  
**Dual sampling:** see `artifacts/STAGE3_DUAL_SAMPLING.md`

---

## Primary table Stage 4 must consume

`tables/stage3_candidate_features.parquet`  
(one row per decision frame × four teammate candidates)

### Dual sampling (critical)

| Column | Use |
|---|---|
| `is_model_5hz_frame` | Window / option-curve series (primary-eligible stride 5) |
| `is_forced_release_frame` | Force-inserted pass `startFrame` (may be off-lattice) |
| `sampling_role` | `model_5hz`, `pass_release`, or `model_5hz+pass_release` |
| `is_pass_release_frame` | True when a Stage 1 pass joins this `(touch, frame)` |

**Passability training:** rows with `completion_label_eligible` (implies release
join + narrow-path known receiver). Expect ≈ one true-receiver row per labeled
pass among the four candidates.

### Keys / identity

| Column | Notes |
|---|---|
| `gameId`, `touchId`, `chanceId`, `possessionId` | Possession context |
| `frameIdx` | Decision frame (5 Hz **or** release) |
| `period`, `gameClock`, `shotClock`, `touch_age_s` | Timing |
| `ballhandlerId`, `candidateId`, `candidate_rank` | Passer + receiver option |
| `fold_id` | Outer LOGO fold for this game |
| `model_hz` (=5), `frame_stride` (=5) | Nominal series cadence |

### Chosen-trajectory geometry (primary surface)

| Column | Family |
|---|---|
| `trajectory_choice`, `trajectory_choice_rule` | Explicit choice (not max-score) |
| `catch_x_event`, `catch_y_event`, `catch_x_norm`, `catch_y_norm` | Projected catch |
| `pass_distance_ft`, `pass_angle_deg`, `trajectory_length_ft`, `flight_time_s`, `pass_speed_fps` | Geometry / timing (`pass_speed = dist / flight_time`) |
| `min_lane_clearance_ft`, `min_temporal_margin_s`, `min_temporal_margin_const35_s` | Defensive lane (primary margin uses fitted flight time; const35 is ablation) |
| `receiver_sep_ft`, `passer_pressure_ft`, `target_defender_recovery_s`, `help_density_12ft` | Pressure / help |
| `sideline_prox_ft`, `baseline_prox_ft` | Boundary |
| `dist_to_rim_ft`, `angle_to_rim_deg`, `court_region`, `receiver_toward_rim_fps` | Offensive context |
| `geometry_passability_score` | Transparent geometry-only baseline |
| `rejected_outside_support` | Fold **attempted-pass** distance-support reject |
| `intercepting_defenderId`, `intercepting_defender_vx/vy`, `intercepting_defender_velocity_ok` | Geometric interceptor (≠ pass `interceptorId`) |
| `lane_predError_mean` | Reliability |
| `distance_support_min_ft`, `distance_support_max_ft` | Fold support bounds |

### Both trajectory families (retained)

Every primary metric also exists as `direct_*` and `lead_*`. Stage 4 must **not**
silently switch to the higher-scoring trajectory.

### Motion / matchup (causal; missing stays missing)

| Column | Notes |
|---|---|
| `passer_vx/vy`, `passer_velocity_ok` | NaN when not ok — never zeroed |
| `receiver_vx/vy`, `receiver_velocity_ok` | Same |
| `assigned_defenderId`, `matchup_matched`, `matchupId` | Interval join |
| `assigned_defender_vx/vy`, `assigned_defender_velocity_ok` | NaN if unmatched |

### Stage 4 model placeholders (do not treat as fitted)

`q_passability_model`, `V_catch_model`, `NOV_model` — NaN until Stage 4 fits.

### Narrow-path gates (full `GATE_COLUMNS` on every row)

| Column | Rule |
|---|---|
| `passId`, `is_pass_release_frame` | Release-time join |
| `completion_label_eligible`, `is_true_receiver`, `true_receiverId` | Training labels |
| All of `GATE_COLUMNS` | Broadcast from `stage2_label_eligibility` (defaults False/mode off-release) |
| `cohort`, `label_gate_reason`, `pass_outcome_class`, `interceptorId` | Provenance |

**Default Stage 4 completion-training set:** `completion_label_eligible == True`;
positives where `is_true_receiver == True`. Helper:
`narrow_path.receiver_completion_training_labels`.

---

## Supporting tables

| Path | Use |
|---|---|
| `tables/stage2_label_eligibility.parquet` | Authoritative gate |
| `tables/stage3_support_rejection.parquet` | Compact reject audit |
| `artifacts/stage3_fold_flight_models.json` | LOGO flight-time + **attempted** distance support |
| `artifacts/STAGE3_DUAL_SAMPLING.md` | Sampling contract |
| `artifacts/stage3_candidate_report.md` | Population QC |
| `artifacts/stage3_geometry_selfcheck.md` | Automated invariants |
| `artifacts/STAGE3_VERIFICATION.md` | Sign-off checklist |
| `artifacts/stage3_output_manifest.json` | SHA-256 outputs |

---

## Narrow-path constraints (binding)

1. Completion labels only via `receiver_specific_eligible` / `receiver_completion_training_labels`.
2. Failed passes = touch-level turnover evidence by default.
3. Auto-accepted inferred failures opt-in only + exclusion sensitivity.
4. Never use `interceptorId` as intended target (distinct from geometric `intercepting_defenderId`).
5. Never fill Stage 2 reviewer columns; never claim failed targets audited.
6. Never cite completed-pass proxy as failed-pass precision.
7. Never use season aggregate CSVs as covariates.
8. LOGO nested folds only; flight fitting always sees all nine train games (even under `--games`).
9. Causal velocities; NaN stays NaN.
10. Mask `rejected_outside_support` for counterfactual scoring.

## Flight / cache notes for rebuild

- `--games` limits **reconstruction only**; fold flight + support always load all freeze games.
- Empty train completed / empty support → hard fail.
- Per-game `_DONE_STAGE3` stores `hash=<flight_bundle_cache_key>`; mismatch rebuilds.
