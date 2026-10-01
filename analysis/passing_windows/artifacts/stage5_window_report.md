# Stage 5 window report

**Upstream:** `stage4_sklearn_logo_v3`
**Mode:** descriptive window objects under nested LOGO thresholds

## Population

- 5 Hz series rows: **176004**
- Window rows (incl. never_open stubs): **12827**
- Games: **10**

## Support rejection (5 Hz series)

- Fraction `rejected_outside_support`: **0.1172**
- Rejected rows cannot open windows (NOV non-viable)

## Label counts

| Label | N |
|---|---:|
| `never_open` | 10437 |
| `unused` | 1928 |
| `late` | 315 |
| `used` | 147 |

## Duration / NOV (non–never_open)

- Median duration_s: **0.800**
- Mean peak_NOV: **0.0805**
- Mean integrated_NOV: **0.0569**

## Per-game label mix

| gameId | used | late | unused | never_open | n_windows |
|---:|---:|---:|---:|---:|---:|
| 114086 | 23 | 35 | 176 | 1152 | 1386 |
| 114099 | 14 | 41 | 241 | 1011 | 1307 |
| 114169 | 9 | 20 | 190 | 1002 | 1221 |
| 114234 | 14 | 36 | 183 | 1045 | 1278 |
| 114243 | 20 | 21 | 159 | 1112 | 1312 |
| 178442 | 10 | 29 | 207 | 944 | 1190 |
| 179612 | 13 | 39 | 199 | 1127 | 1378 |
| 184439 | 18 | 22 | 198 | 916 | 1154 |
| 188630 | 10 | 37 | 192 | 1140 | 1379 |
| 191313 | 16 | 35 | 183 | 988 | 1222 |

## Language note

Windows are **model-available** intervals under fold-calibrated thresholds.
Unused does not mean a mistake; used does not mean optimal.
