# Stage 1 calculation audit

Every derived Stage 1 field is documented with formula, units, spot-check, and pitfalls.

## Registry

| Field | Formula | Units | Spot-check | Pitfalls |
|---|---|---|---|---|
| `elapsed_frames` | `endFrame - startFrame` | frames (25 fps) | one-game identity + 10-game totals | Negative => malformed; excluded from primary |
| `duration_s_from_frames` | `elapsed_frames / 25` | seconds | corr vs `touchTime` | Clock stoppage can diverge slightly from wall time |
| `elapsed_game_clock` | `startGameClock - endGameClock` | seconds | zero-elapsed flagged | Game clock pauses; not wall duration |
| `distance_recomputed` | `hypot(passerLoc - receiverLoc)` | feet | max abs diff vs `distance` ~0 | Null locs => NaN |
| `speed_causal` | `(p[t]-p[t-1]) / (Δframe/25)` on **broadcast** coords, then rotated to the event frame | ft/s | future-mutate invariant + real-data boundary audit | First frame NaN; no future frames; **never difference `x_event`** (it flips sign at attacking-hoop changes); NaN across orientation segments, frame gaps > 25, and speeds > 40 ft/s |
| `velocity_segment` | cumulative count of `leftHoop` changes per player | index | boundary audit | Segment change forces a derivative reset |
| `speed_implausible` | `speed_causal > 40` ft/s before nulling | bool | 0 retained | Cap is a physical plausibility screen, not a model choice |
| `assigned_defenderId` | matchup interval covering the frame for that offensive player | player id | interval-containment audit | Intervals overlap; ties broken by latest `startFrame`, then shortest, then smallest `matchupId` |
| `pass_outcome_class` | `complete` / `incomplete_turnover` / `unknown_outcome` from tri-state `complete` | category | class counts vs raw feed | `complete` is nullable; `astype(bool)` silently merges nulls into failures |
| `x_norm,y_norm` | `x_norm = attack_sign * x` (`attack_sign=-1`) | feet | hoop side frac | Events already attack −x |
| `usable_flag` | `qualityIndex >= 3` | bool | equals vendor `usable` | Do not use QI as continuous covariate without care |
| `points_recon` | made FG 2/3 + made FT 1 | points | manual sum == recon sum | **Never** use `chances.ptsScored` for outcomes |
| `interceptorId` | copy of `toReceiverId` | player id | complete passes null | Never intended target |
| `primary_touch_eligible` | first-match exclusion cascade | bool | invariants audit | Order matters for CONSORT |
| `ballhandler_in_offense` | `playerId ∈ chance.offPlayerIds` | bool | exactly-4-candidates audit | Defender-credited touches inside an offensive chance otherwise yield five candidates |
| `primary_frame_eligible` | live ∧ frontcourt ∧ 10 players ∧ BH present | bool | model-frame rates | `predError` caps TBD_TRAINING_FOLD |

## Spot-check results

### `points_reconstruction` — PASS

```
{'shot_pts_manual': 1390, 'ft_pts_manual': 329, 'points_recon_sum': 1719, 'manual_equals_recon': True, 'ptsScored_sum_for_reference_only': 1719, 'ok': True, 'note': 'ptsScored must not be used as outcome source even if totals match in these 10 games'}
```

### `spot_check_game_114243` — PASS

```
{'gameId': 114243, 'elapsed_frames_formula_ok': True, 'touchTime_vs_frames_corr': 0.9999999999999999, 'pass_distance_recompute_ok': True, 'usable_equals_qi_ge_3': True, 'ok': True}
```

### `totals_10_games` — PASS

```
{'n_touches': 6662, 'n_primary_touches': 3636, 'n_passes': 4786, 'n_self_passes': 91, 'n_multipass_touches': 6, 'n_chances': 2055, 'n_usable_chances': 1911, 'n_model_frames': 70117, 'n_model_frames_eligible': 44001, 'post_catch_with_receiver_touch_rate': 0.9962314342717801, 'hoop_orientation_ok_rate': 1.0, 'ok': True}
```

### `mirroring` — PASS

```
{'n_cases': 2, 'cases': [{'distance_symmetric': True, 'y_delta_flips': True, 'clearance_symmetric': True}, {'distance_symmetric': True, 'y_delta_flips': True, 'clearance_symmetric': True}], 'ok': True}
```

### `causal_velocity` — PASS

```
{'v_at_3': 25.0, 'v_at_3_after_future_mutate': 25.0, 'speed_at_unsegmented_boundary': 1950.0, 'checks': {'future_mutate_invariant': True, 'leakage_detector_works': True, 'segment_boundary_resets': True, 'unsegmented_boundary_is_impossible': True, 'wide_gap_resets': True}, 'future_mutate_invariant': True, 'leakage_detector_works': True, 'segment_boundary_resets': True, 'unsegmented_boundary_is_impossible': True, 'wide_gap_resets': True, 'ok': True}
```

### `causal_velocity_boundaries` — PASS

```
{'n_player_rows': 2972460, 'n_speed_finite': 2943095, 'max_speed_fps': 36.02123842795578, 'p99_9_speed_fps': 21.242543338307588, 'n_speed_over_cap': 0, 'speed_cap_fps': 40.0, 'max_frame_gap_frames': 25, 'n_rows_at_orientation_boundary_with_speed': 0, 'max_speed_at_orientation_boundary': nan, 'n_velocity_across_wide_gap': 0, 'n_speed_null': 29365, 'checks': {'no_speed_over_cap': True, 'no_implausible_flagged_retained': True, 'no_velocity_at_orientation_boundary': True, 'no_velocity_across_wide_gap': True}, 'ok': True}
```

### `candidate_sets_exactly_four` — PASS

```
{'n_primary_touches': 3636, 'n_candidate_rows': 14544, 'expected_rows': 14544, 'candidates_per_touch_counts': {4: 3636}, 'max_candidates_per_touch': 4, 'min_candidates_per_touch': 4, 'n_touches_wrong_count': 0, 'n_touches_with_duplicates': 0, 'n_primary_touches_missing_candidates': 0, 'n_ballhandler_as_candidate': 0, 'n_candidates_outside_lineup': 0, 'checks': {'all_touches_exactly_4': True, 'no_duplicate_candidates': True, 'every_primary_touch_covered': True, 'no_candidates_outside_primary': True, 'no_ballhandler_as_candidate': True, 'no_candidate_outside_chance_lineup': True, 'no_primary_ballhandler_outside_offense': True}, 'ok': True}
```

### `matchup_interval_join` — PASS

```
{'n_candidate_states': 280468, 'n_matched': 172317, 'coverage_all_states': 0.6143909465607484, 'coverage_primary_eligible_states': 0.9677620963159929, 'n_unmatched': 108151, 'unmatched_rate': 0.38560905343925156, 'n_ambiguous_overlapping_intervals': 3486, 'ambiguous_rate': 0.012429225437483064, 'max_intervals_covering_one_frame': 4, 'n_duplicate_keys': 0, 'n_self_assignments': 0, 'resolution_rule': 'latest startFrame, then shortest interval, then smallest matchupId', 'checks': {'chosen_interval_contains_frame': True, 'keys_unique': True, 'no_self_assignment': True, 'defender_from_matchup_feed': True, 'coverage_above_min': True}, 'ok': True}
```

### `pass_outcome_classes` — PASS

```
{'by_class': {'complete': {'n': 4629, 'n_with_endFrame': 4629, 'n_with_receiverId': 4629, 'n_with_receiverLoc': 4541, 'n_with_distance': 4541, 'n_with_interceptor': 0, 'n_turnover': 0}, 'incomplete_turnover': {'n': 89, 'n_with_endFrame': 89, 'n_with_receiverId': 0, 'n_with_receiverLoc': 0, 'n_with_distance': 0, 'n_with_interceptor': 89, 'n_turnover': 89}, 'unknown_outcome': {'n': 68, 'n_with_endFrame': 0, 'n_with_receiverId': 0, 'n_with_receiverLoc': 0, 'n_with_distance': 0, 'n_with_interceptor': 0, 'n_turnover': 0}}, 'n_failed_or_unknown': 157, 'note': 'receiverLoc / distance / receiverRegion / receiverId are absent from the raw feed for every non-complete pass; Stage 1 drops nothing. Stage 2 must infer targets from ball trajectory, pass direction, and flight-time support.', 'checks': {'no_receiverLoc_on_failed_passes': True, 'unknown_outcome_has_no_endFrame': True, 'no_receiverId_on_failed_passes': True}, 'ok': True}
```

### `lane_geometry` — PASS

```
{'checks': {'midpoint_dist_zero': True, 'offset_dist_3': True, 'clearance_mid_negative': True, 'clearance_off_positive': True, 'far_margin_gt_on_lane': True}, 'ok': True, 'values': {'d_mid': 0.0, 'd_off': 3.0, 'clear_mid': -1.0, 'clear_off': 2.0, 'margin_on_lane': -0.5, 'margin_far': 0.475}}
```

### `tracking_retention` — PASS

```
{'pass_window_frames_half_width': 25, 'failed_pass_window_frames_half_width': 50, 'n_retained_tracking_frames': 307175, 'n_pass_window_frames': 263440, 'n_model_sample_frames': 71928, 'ok': True}
```

## KNOWN_ISSUES handled in Stage 1

- Self-passes excluded from primary
- Malformed chance/touch frame order excluded
- Zero-elapsed chances excluded
- `toReceiverId` renamed semantically to interceptor; never used as target
- Multi-pass touches excluded from primary (audit table retained)
- Season aggregates not loaded
- `ptsScored` not used for `points_recon`
- Touches whose ball-handler is absent from `chance.offPlayerIds` excluded
  (`ballhandler_not_in_offense`), so every primary touch forms exactly four
  teammate candidates
- Causal velocities differentiated on continuous broadcast coordinates and
  reset at orientation/gap boundaries, replacing event-coordinate differencing

## Known raw-data limitations recorded for Stage 2

- `receiverLoc`, `distance`, `receiverRegion`, and `receiverId` are null for
  **every** non-complete pass in the raw feed. Stage 1 drops nothing; the
  channel does not exist. Stage 2 must infer targets from ball trajectory,
  pass direction, and empirical flight-time support.
- `complete` is tri-state. `unknown_outcome` passes additionally have no
  `endFrame`, no receiver, and no interceptor, so they cannot enter
  failed-pass target inference; use `failed_pass_inference_eligible`.

