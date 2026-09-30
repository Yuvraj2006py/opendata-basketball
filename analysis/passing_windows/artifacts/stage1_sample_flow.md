# Stage 1 sample flow (CONSORT-style)

**Freeze ID:** `passing_windows_stage0_20250924`

## Overall cascade (first-match exclusion)

| Stage | Remaining | Excluded here |
|---|---:|---:|
| `all_touches` | 6662 | 0 |
| `after_exclude_missing_chance` | 6662 | 0 |
| `after_exclude_malformed_chance_frame_order` | 6662 | 0 |
| `after_exclude_malformed_touch_frame_order` | 6662 | 0 |
| `after_exclude_usable_false` | 6345 | 317 |
| `after_exclude_transition_true` | 5638 | 707 |
| `after_exclude_zero_elapsed_clock` | 5628 | 10 |
| `after_exclude_missing_frontcourt_frame` | 5495 | 133 |
| `after_exclude_lineup_not_5v5` | 5495 | 0 |
| `after_exclude_substitution_boundary` | 4220 | 1275 |
| `after_exclude_missing_ballhandler` | 4220 | 0 |
| `after_exclude_ballhandler_not_in_offense` | 4209 | 11 |
| `after_exclude_self_pass_touch` | 4161 | 48 |
| `after_exclude_multipass_touch` | 4161 | 0 |
| `after_exclude_inbounds_touch` | 3636 | 525 |
| `primary_touch_eligible` | 3636 | 0 |

**Primary-eligible touches:** 3636 / 6662

## Exclusions by reason

| Reason | N |
|---|---:|
| `ballhandler_not_in_offense` | 11 |
| `inbounds_touch` | 525 |
| `missing_frontcourt_frame` | 133 |
| `self_pass_touch` | 48 |
| `substitution_boundary` | 1275 |
| `transition_true` | 707 |
| `usable_false` | 317 |
| `zero_elapsed_clock` | 10 |
| `primary_eligible` | 3636 |

## Per-game primary eligible

| gameId | all touches | primary |
|---|---:|---:|
| 114086 | 690 | 387 |
| 114099 | 652 | 369 |
| 114169 | 653 | 341 |
| 114234 | 655 | 359 |
| 114243 | 728 | 365 |
| 178442 | 682 | 347 |
| 179612 | 714 | 391 |
| 184439 | 635 | 340 |
| 188630 | 635 | 395 |
| 191313 | 618 | 342 |

## Exclusion concentration checks

### by_period

|   period |   n_all |   n_excluded |   excl_rate |
|---------:|--------:|-------------:|------------:|
|        1 |    1690 |          723 |    0.427811 |
|        2 |    1667 |          767 |    0.460108 |
|        3 |    1603 |          691 |    0.431067 |
|        4 |    1634 |          817 |    0.5      |
|        5 |      68 |           28 |    0.411765 |

### by_off_team

|   offTeamId |   n_all |   n_excluded |   excl_rate |
|------------:|--------:|-------------:|------------:|
|        1315 |     325 |          159 |    0.489231 |
|        1316 |     327 |          124 |    0.379205 |
|        1319 |     370 |          204 |    0.551351 |
|        1322 |     586 |          251 |    0.428328 |
|        1325 |     321 |          155 |    0.482866 |
|        1326 |     351 |          183 |    0.521368 |
|        1328 |     686 |          242 |    0.35277  |
|        1330 |     358 |          159 |    0.444134 |
|        1331 |     687 |          321 |    0.467249 |
|        1332 |     325 |          157 |    0.483077 |
|        1333 |     335 |          159 |    0.474627 |
|        1334 |     318 |          178 |    0.559748 |
|        1335 |     357 |          178 |    0.498599 |
|        1336 |     335 |          134 |    0.4      |
|        1338 |     300 |          136 |    0.453333 |
|        1339 |     393 |          168 |    0.427481 |
|        1340 |     288 |          118 |    0.409722 |

### by_score_diff

| score_diff_bin   |   n_all |   n_excluded |   excl_rate |
|:-----------------|--------:|-------------:|------------:|
| <=-15            |     337 |          150 |    0.445104 |
| -15:-5           |    1479 |          706 |    0.47735  |
| -5:5             |    3301 |         1440 |    0.436231 |
| 5:15             |     954 |          437 |    0.458071 |
| >=15             |     591 |          293 |    0.49577  |

### by_shot_clock

| shot_clock_bin   |   n_all |   n_excluded |   excl_rate |
|:-----------------|--------:|-------------:|------------:|
| 0-2              |     218 |          134 |    0.614679 |
| 2-7              |     321 |          115 |    0.358255 |
| 7-14             |    1552 |          495 |    0.318943 |
| 14-24            |    3220 |         1257 |    0.390373 |

**Interpretation note:** large rate swings by bin warrant Stage 2+ sensitivity; they do not invalidate the freeze but must be reported.
