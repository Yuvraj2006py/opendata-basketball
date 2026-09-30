# Stage 5 builder status (MVP)

**Date:** 2026-09-30  
**Upstream:** `stage4_sklearn_logo_v3`  
**Audit brief:** `artifacts/STAGE5_AUDIT_BRIEF_FOR_BUILDER.md`

## Shipped this session

| Deliverable | Status |
|---|---|
| `src/passing_windows/windows/` package | Done (`smooth`, `thresholds`, `segment`, `labels`, `option_set`, `leakage`) |
| `pipelines/05_segment_windows.py` | Done (restartable, `--games` / `--force`, LOGO thresholds) |
| `configs/stage5_windows.yaml` | Done (procedure locked) |
| `tests/test_stage5.py` | **19 passed** (S5-B03,B05–B08,B10,B11,B15 + extras) |
| 1-game smoke (`--games 114086`) | Done: 18968 series rows, 1386 window rows |
| **Full 10-game `--force` run** | **Done:** series=**176004**, windows=**12827** |
| `STAGE5_VERIFICATION.md` / `STAGE5_HANDOFF_STAGE6.md` / manifest | Updated from full 10-game run |
| README Stage 5 status | Updated to MVP |

### Smoke label counts (game 114086)

`never_open` 1152 · `unused` 176 · `late` 35 · `used` 23

### Full 10-game command (completed)

```powershell
cd analysis\passing_windows
python pipelines\05_segment_windows.py --force
python -m pytest tests\test_stage5.py -q
```
## Binding rules locked in code

- Series: `is_model_5hz_frame` only; persistence ≥ 0.20s = **≥2 consecutive 5 Hz samples**
- Causal EWMA only; open = `q ≥ open_thr ∧ NOV > 0 ∧ ¬rejected`
- Hysteresis `close ≤ open`; thresholds via nested LOGO train-only median-q procedure
- `used` = known-receiver ∩ true-receiver ∩ eligibility gates; off-lattice release OK
- Existence prob column present with `existence_prob_status=deferred_stage7`

## Remaining gaps vs audit brief §D (not cleared this MVP)

| ID | Severity | Gap |
|---|---|---|
| S5-B01 | BLOCKER | Full-table 190348 assert / 1:1 Stage3 join not integration-tested on all games |
| S5-B04 | BLOCKER | Covered by unit test; live audit script not yet run on full release joins |
| S5-B09 | BLOCKER | Late delta fold-internal unit covered via selector; dedicated `test_stage5_late_delta_fold_internal` thin |
| S5-B12–B14 | BLOCKER | GATE carry in pipeline; interceptor/forbidden-language live tests incomplete |
| S5-B16–B18 | BLOCKER | Mutual exclusivity + continuous fields partially tested; option-set §H completeness not fully locked |
| S5-B19–B23 | BLOCKER | No-random-split / no-season-agg / all-10-folds / full regression lock / live audit pending full run |
| S5-M01–M13 | MAJOR | sampling_role retention, robustness grid table write, support-rejection rate report, named-failed default, use-timing fields, label-stability hook docs, primary population filter, GAM caveat |
| S5-N01–N04 | MINOR | Mostly partially addressed; handoff coverage honesty after full run |
| Live audit | — | `pipelines/_stage5_adversarial_live_audit.py` **not created** |
| Artifacts | — | `STAGE5_ADVERSARIAL_AUDIT.md`, `stage5_window_report.md`, `stage5_threshold_grid_windows.parquet` deferred |
| Full LOGO | — | Only fold `fold_holdout_114086` smoked; other 9 folds not written |

## Next builder steps

1. Run full 10-game `05_segment_windows.py --force`
2. Add `_stage5_adversarial_live_audit.py` + expand regression lock to all BLOCKERs
3. Emit threshold grid windows table + window report
4. Fill remaining MAJOR/MINOR tests from audit brief §D
