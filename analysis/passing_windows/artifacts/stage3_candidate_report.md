# Stage 3 candidate reconstruction report

**Freeze ID:** `passing_windows_stage0_20250924`  
**Narrow-path lock:** `passing_windows_stage2_narrow_path_20260924`  
**Trajectory choice rule:** `lead_if_velocity_ok_else_direct` (explicit; not score-maximizing)

## Population

- Candidate rows (5 Hz primary + forced release × 4 receivers): **190348**
- Games: **10**
- Touches: **5650**
- Decision frames: **47587**
- Model 5 Hz frames: **44001**
- Forced release frames: **4615**
- Candidates per frame (mode): **4**

## Dual sampling

- `model_5hz` / `is_model_5hz_frame`: primary-eligible stride-5 window series.
- `pass_release` / `is_forced_release_frame`: every pass `startFrame` with tracking,
  force-inserted even when off the 5 Hz lattice so Stage 4 can train passability
  at release-time state. Overlaps are dual-flagged.

## Narrow-path label join

- Completion-label-eligible candidate rows (release × known receiver pass): **17588**
- True-receiver rows among those: **4397**
- Failed passes remain touch-level only; inferred targets are **not** used as
  named receivers for completion calibration.

## Support rejection and missingness

- Rejected outside fold distance support: **12.01%**
- Missing matchup (`matchup_matched == False`): **6.26%**
- Receiver velocity missing/invalid: **0.60%**

## Trajectory mix (primary choice)

- `direct`: 1133
- `lead`: 189215

## Court region mix (chosen trajectory catch)

- `wing`: 88895
- `corner`: 42066
- `slot`: 35706
- `paint`: 18052
- `backcourt`: 4824
- `unknown`: 805

## Fold flight-time models (leave-one-game-out)

| Fold | Held-out | Train completed | Bins | Support max ft |
|---|---:|---:|---:|---:|
| `fold_holdout_114086` | 114086 | 4046 | 20 | 40.75 |
| `fold_holdout_114099` | 114099 | 4087 | 20 | 40.58 |
| `fold_holdout_114169` | 114169 | 4091 | 20 | 40.72 |
| `fold_holdout_114234` | 114234 | 4098 | 20 | 40.69 |
| `fold_holdout_114243` | 114243 | 4023 | 20 | 40.76 |
| `fold_holdout_178442` | 178442 | 4089 | 20 | 40.55 |
| `fold_holdout_179612` | 179612 | 4023 | 19 | 40.54 |
| `fold_holdout_184439` | 184439 | 4103 | 20 | 40.66 |
| `fold_holdout_188630` | 188630 | 4087 | 20 | 40.60 |
| `fold_holdout_191313` | 191313 | 4123 | 20 | 40.54 |

## Per-game summaries

| Game | Rows | % rejected | % missing matchup | Cached | Seconds |
|---:|---:|---:|---:|---|---:|
| 114243 | 18376 | 10.72 | 5.30 | False | 130.15 |
| 114234 | 18344 | 11.33 | 5.64 | False | 128.38 |
| 114169 | 18076 | 11.24 | 7.52 | False | 126.9 |
| 114099 | 19604 | 12.94 | 5.86 | False | 171.48 |
| 114086 | 20480 | 10.91 | 5.56 | False | 253.19 |
| 178442 | 19580 | 13.94 | 6.04 | False | 243.27 |
| 179612 | 18972 | 14.13 | 7.95 | False | 223.2 |
| 184439 | 17288 | 11.64 | 5.66 | False | 205.02 |
| 188630 | 21128 | 11.21 | 6.23 | False | 244.78 |
| 191313 | 18500 | 12.02 | 6.88 | False | 216.41 |

## Feature families retained

- Pass geometry: distance, angle, lane clearance, trajectory length, sideline/baseline proximity
- Motion: passer / receiver / assigned-defender velocity components (NaN when missing)
- Timing: flight time, shot clock, touch age, period
- Defensive context: passer pressure, receiver separation, help density, temporal margin, recovery time
- Offensive context: rim distance/angle, paint/corner/wing region, toward-rim speed
- Reliability: lane `predError` mean; detection via tracked flags
- Transparent `geometry_passability_score` plus Stage-4 placeholders (`q_passability_model`, etc.)

## Constraints obeyed

1. Joined `tables/stage2_label_eligibility.parquet`.
2. Completion labels via `receiver_completion_training_labels` only.
3. All failed passes touch-level by default (no interceptor-as-target).
4. Missing matchups / null velocities stay missing, not zero.
5. Primary sampling at 5 Hz (stride 5) per Stage 0 freeze.
