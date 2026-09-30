# Stage 1 tracking quality report

**Freeze ID:** `passing_windows_stage0_20250924`

## Per-game tracking summary

|   gameId |   n_frames_total |   n_live |   n_dead |   live_rate |   n_ten_players |   ten_player_rate_of_live |   predError_mean |   ball_predError_mean |   max_speed_causal_fps |   velocity_ok_rate |
|---------:|-----------------:|---------:|---------:|------------:|----------------:|--------------------------:|-----------------:|----------------------:|-----------------------:|-------------------:|
|   114243 |           146870 |    77023 |    69847 |    0.52443  |           77023 |                         1 |          1.55755 |               1.84586 |                36.0212 |           0.990208 |
|   114234 |           158923 |    72132 |    86791 |    0.45388  |           72132 |                         1 |          1.62581 |               1.75898 |                29.0925 |           0.990071 |
|   114169 |           172257 |   106176 |    66081 |    0.616381 |          106176 |                         1 |          1.85936 |               1.94233 |                26.3283 |           0.989243 |
|   114099 |           147892 |    75655 |    72237 |    0.511556 |           75655 |                         1 |          1.58079 |               1.8203  |                26.6131 |           0.990531 |
|   114086 |           124796 |    92188 |    32608 |    0.73871  |           92188 |                         1 |          1.52012 |               1.83089 |                23.2036 |           0.990311 |
|   178442 |           162479 |    89642 |    72837 |    0.551714 |           89642 |                         1 |          1.60346 |               1.87809 |                24.5703 |           0.989496 |
|   179612 |           140856 |    83437 |    57419 |    0.592357 |           83437 |                         1 |          1.72282 |               1.8343  |                35.0263 |           0.99063  |
|   184439 |           154021 |    77428 |    76593 |    0.502711 |           77428 |                         1 |          1.61009 |               1.86802 |                28.9915 |           0.989791 |
|   188630 |           121793 |    73138 |    48655 |    0.600511 |           73138 |                         1 |          1.5673  |               1.80009 |                27.2309 |           0.99093  |
|   191313 |           143045 |    82885 |    60160 |    0.579433 |           82885 |                         1 |          1.75089 |               1.90456 |                27.817  |           0.989906 |

## Tracking retention (Stage 2 ball-flight windows)

- Retained 25 Hz frames: **307175**
- Pass-window half-width: **±25 frames** (±1.0 s) around every pass release and recorded end frame
- Non-complete passes have no `endFrame`, so their release window is widened to **±50 frames** (±2.0 s)
- Frames retained for pass windows: **263440**
- Frames retained as 5 Hz model samples: **71928**
- Stage 2 can therefore read ball trajectory around any pass directly from `tracking_frames.parquet` without re-streaming the raw feed.

## Causal velocity plausibility

- Velocity frame-gap reset: **> 25 frames** (1.0 s)
- Plausibility cap: **40.0 ft/s** (elite sprint is ~32 ft/s)
- Max retained `speed_causal`: **36.02123842795578** ft/s
- 99.9th percentile `speed_causal`: **21.242543338307588** ft/s
- Rows above cap after fix: **0**
- Rows with null velocity (segment starts, wide gaps, missing coords): **29365** of 2972460

Velocities are differentiated on continuous broadcast coordinates and then
rotated into the event frame. Differencing `x_event`/`y_event` directly made
every attacking-hoop change look like a teleport (previously up to 1169 ft/s).

## Matchup interval join coverage

- Candidate states: **280468**
- Defensive assignment coverage (all states): **0.6143909465607484**
- Coverage among primary-eligible states: **0.9677620963159929**
- Unmatched states: **108151** (0.38560905343925156)
- States where overlapping intervals required tie-breaking: **3486** (0.012429225437483064)
- Max intervals covering a single frame: **4**
- Deterministic resolution rule: latest startFrame, then shortest interval, then smallest matchupId

## Pass outcome classes and failed-pass evidence

| Class | N | with `endFrame` | with `receiverId` | with `receiverLoc` | with interceptor |
|---|---:|---:|---:|---:|---:|
| `complete` | 4629 | 4629 | 4629 | 4541 | 0 |
| `incomplete_turnover` | 89 | 89 | 0 | 0 | 89 |
| `unknown_outcome` | 68 | 0 | 0 | 0 | 0 |

`receiverLoc` / `distance` / `receiverRegion` / `receiverId` are absent from the
raw feed for every non-complete pass, so Stage 2 cannot compare candidates to a
recorded receiver location for the failed-pass population. `unknown_outcome`
passes also lack an `endFrame` entirely and are excluded from failed-pass target
inference via `failed_pass_inference_eligible`.

- Failed passes eligible for target inference: **89**

## Model-frame eligibility (5 Hz primary touches)

- Model frames built: **70117**
- Primary-frame eligible: **44001** (62.8%)
- `live_clock` true: 63962 / 70117
- `frontcourt` true: 45438 / 70117
- `ten_players` true: 65708 / 70117
- `ballhandler_present` true: 65708 / 70117

## TBD_TRAINING_FOLD tracking caps

Stage 0 freezes **selection procedure** only for `predError` caps.
Stage 1 reports distributions but does **not** apply numeric caps yet.
Caps will be chosen inside each outer-fold training set so tracking-failure
among otherwise-eligible decision states stays under the 20% kill line.

Primary-eligible touches: 3636

