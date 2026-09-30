# Stage 2 geometry self-consistency report

**Freeze ID:** `passing_windows_stage0_20250924`

Automated checks only. These prove the inference machinery is internally coherent;
they do not substitute for the human audit that `stage2_ambiguous_audit.csv` collects.
An invariant failure fails the build; a diagnostic failure is a statement about the data.

| Check | Severity | Result |
|---|---|---|
| `probability_simplex` | invariant | PASS |
| `interceptor_never_target` | invariant | PASS |
| `target_not_passer` | invariant | PASS |
| `assignment_in_candidate_set` | invariant | PASS |
| `unresolved_has_no_label` | invariant | PASS |
| `mirror_invariance` | invariant | PASS |
| `flight_time_monotonic` | invariant | PASS |
| `release_point_matches_passer` | diagnostic | PASS |
| `failed_residuals_within_completed_support` | diagnostic | FAIL |

## Detail

### `probability_simplex`

```
{'ok': True, 'n_passes': 4619, 'max_abs_sum_error': 4.440892098500626e-16, 'n_negative': 0, 'n_nan': 0, 'severity': 'invariant'}
```

### `interceptor_never_target`

```
{'ok': True, 'n_interceptor_in_candidate_set': 0, 'n_interceptor_assigned_as_target': 0, 'severity': 'invariant'}
```

### `target_not_passer`

```
{'ok': True, 'n_passer_as_candidate': 0, 'severity': 'invariant'}
```

### `assignment_in_candidate_set`

```
{'ok': True, 'n_checked': 60, 'n_outside_candidate_set': 0, 'severity': 'invariant'}
```

### `unresolved_has_no_label`

```
{'ok': True, 'n_labelled_non_accepted': 0, 'severity': 'invariant'}
```

### `mirror_invariance`

```
{'ok': True, 'max_abs_probability_difference': 0.0, 'probabilities': [0.472463, 9e-06, 0.527517, 1.1e-05], 'severity': 'invariant'}
```

### `flight_time_monotonic`

```
{'ok': True, 'n_bins': 18, 'n_train': 3364, 'distance_support_max_ft': 37.95235558678171, 't_at_5ft': 0.2, 't_at_15ft': 0.4, 't_at_30ft': 0.72, 'min_diff': 0.0, 'severity': 'invariant'}
```

### `release_point_matches_passer`

```
{'ok': True, 'n': 4507, 'median_ft': 3.7769880385514862, 'p95_ft': 7.3613429159982555, 'share_within_6ft': 0.8850676725094297, 'severity': 'diagnostic'}
```

### `failed_residuals_within_completed_support`

```
{'n_accepted': 60, 'n_reference': 3711, 'top_angle_deg_reference_p99': 47.554562400609235, 'top_angle_deg_share_beyond_reference_p99': 0.15, 'top_lateral_ft_reference_p99': 8.09566763766684, 'top_lateral_ft_share_beyond_reference_p99': 0.11666666666666667, 'ok': False, 'severity': 'diagnostic'}
```

## Per-fold likelihood scales

| fold_id             |   scale_sigma_ball_ft |   scale_sigma_pos_ft |   scale_flight_offset_s |   scale_sigma_flight_s |   flight_time_bins |   distance_support_max_ft |
|:--------------------|----------------------:|---------------------:|------------------------:|-----------------------:|-------------------:|--------------------------:|
| fold_holdout_114243 |                1.5095 |                  0.5 |                 -0.122  |                 0.0654 |                 18 |                   38.0425 |
| fold_holdout_114234 |                1.5157 |                  0.5 |                 -0.1218 |                 0.0651 |                 18 |                   38.395  |
| fold_holdout_114169 |                1.5086 |                  0.5 |                 -0.122  |                 0.0673 |                 18 |                   38.9847 |
| fold_holdout_114099 |                1.514  |                  0.5 |                 -0.1225 |                 0.0672 |                 18 |                   37.9857 |
| fold_holdout_114086 |                1.4779 |                  0.5 |                 -0.1223 |                 0.0665 |                 18 |                   39.2525 |
| fold_holdout_178442 |                1.507  |                  0.5 |                 -0.1216 |                 0.0649 |                 18 |                   37.9601 |
| fold_holdout_179612 |                1.5069 |                  0.5 |                 -0.1226 |                 0.0659 |                 18 |                   37.9493 |
| fold_holdout_184439 |                1.5224 |                  0.5 |                 -0.1217 |                 0.0649 |                 18 |                   37.9703 |
| fold_holdout_188630 |                1.5166 |                  0.5 |                 -0.1218 |                 0.066  |                 19 |                   37.9813 |
| fold_holdout_191313 |                1.4843 |                  0.5 |                 -0.1233 |                 0.0661 |                 18 |                   37.9524 |

Scales are robust (1.4826 x MAD) residual widths of the **known** receiver in the nine
training games of each fold, so the likelihood width is data-driven rather than assumed.
`sigma_ball_ft` sets the bearing error (divided by the observed ball chord) and
`sigma_pos_ft` the receiver position/projection error; the lateral tolerance at a
candidate combines them and therefore widens with distance and with short ball chords.

Checks that need a fitted model (`mirror_invariance`, `flight_time_monotonic`) use the
last fold's model; every fold's scales are listed above so the choice is inspectable.

