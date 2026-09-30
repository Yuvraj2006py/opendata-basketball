# Stage 5 audit brief for builder — Passing Windows

**Auditor role:** STAGE 5 AUDIT AGENT (adversarial / freeze-faithful; brief only — do not implement Stage 5 in this pass)  
**Freeze:** `passing_windows_stage0_20250924` (`configs/stage0_freeze.yaml`)  
**Narrow-path lock:** `passing_windows_stage2_narrow_path_20260924` (`configs/stage2_narrow_path.yaml`)  
**Plan:** `.cursor/plans/passing-windows-study_aff10b97.plan.md` § Stage 5  
**Handoff in:** `artifacts/STAGE4_HANDOFF_STAGE5.md`  
**Dual sampling:** `artifacts/STAGE3_DUAL_SAMPLING.md`  
**Prior audit style:** `artifacts/STAGE4_ADVERSARIAL_AUDIT.md`, `artifacts/STAGE4_REGRESSION_LOCK.md`  
**Upstream model:** `stage4_sklearn_logo_v3`  
**Primary input (verified present):** `tables/stage4_candidate_predictions.parquet` — **190348** rows; **176004** with `is_model_5hz_frame==True`

---

## A. Verdict on current Stage 5 state

### **NOT STARTED**

| Probe | Result |
|---|---|
| `pipelines/05*` | **Absent** (only `01`–`04` + Stage 2–4 audit scripts) |
| `src/passing_windows/windows*` or `src/**/windows*` | **Absent** (no window segmentation package) |
| `tests/test_stage5.py` | **Absent** |
| `artifacts/STAGE5_*` (verification / handoff / regression lock) | **Absent** (this brief is the first Stage 5 artifact) |
| `tables/stage5_*` | **Absent** |
| README Stage 5+ | Explicitly **“Not started”** |

**Upstream readiness:** Stage 4 is **SAFE TO ACCEPT** (`artifacts/STAGE4_ADVERSARIAL_AUDIT.md`). Stage 5 **may begin** under the narrow-path lock. Do **not** refit Stage 4 component models.

**Interpretation:** Inputs exist; Stage 5 implementation surface does not. Builder starts from greenfield modules + TDD, consuming frozen Stage 4 predictions.

---

## B. Binding constraints (checklist)

Carry every item forward. Any violation is a Stage 5 BLOCKER unless waived by a new freeze/lock revision.

### B1. Stage 0 freeze — windows & validation

- [ ] `windows.min_persistence_seconds == 0.20` (`configs/stage0_freeze.yaml` § `windows`)
- [ ] `windows.min_persistence_frames_at_25hz == 5` (wall-clock equivalent of 0.20s at native 25 Hz)
- [ ] Separate **open** and **close** thresholds (hysteresis); both `TBD_TRAINING_FOLD`
- [ ] Open rule: `q` exceeds fold open threshold **AND** `NOV > 0`, with persistence ≥ 0.20s
- [ ] Labels exactly: `open` / `used` / `late` / `unused` / `never_open` (freeze uses `never_open`)
- [ ] Continuous outputs: opening/peak/closing time, duration, peak_NOV, integrated_NOV, delay_opening_or_peak_to_release, value_at_use_vs_peak, window_existence_probability_under_tracking_perturbation
- [ ] Threshold selection **inside each outer-fold training set only**; apply to held-out game only
- [ ] After primary selection, document a **fixed robustness grid** and re-run headline-ready window summaries on that grid (grid values still TBD until Stage 5 freeze artifact)
- [ ] Nested LOGO only; forbidden: random frame/touch/chance splits
- [ ] Smoothing causality: **no future frames** (`temporal_sampling.smoothing_causality`)
- [ ] Presentation rule: expose `q` and `V_catch` separately before collapsing to NOV (do not ship NOV-only tables as the sole Stage 5 product)
- [ ] Forbidden claim language: no “correct pass”, “bad decision”, “points left on the table”; use model-preferred / available under the model / unused
- [ ] Kill criterion `window_label_instability`: labels change >15% under plausible tracking perturbations → narrow/stop claims (Stage 5 must produce the machinery or an honest deferred hook + Stage 7 handoff)

### B2. Stage 4 handoff (`artifacts/STAGE4_HANDOFF_STAGE5.md`)

- [ ] Consume `tables/stage4_candidate_predictions.parquet` joined 1:1 on `(gameId, touchId, frameIdx, candidateId)` with `tables/stage3_candidate_features.parquet` when geometry context is needed
- [ ] Required score columns: `q_passability_model`, `q_ablation_*`, `V_catch_model`, `V_catch_ablation_*`, `V_keep_model`, `V_fail`, `Q_option_model`, `NOV_model`, `choice_prob_model`, `fold_id`, `model_version`
- [ ] **Do not refit** passability / V_catch / V_keep / choice models
- [ ] Open threshold remains `TBD_TRAINING_FOLD` (select; do not invent a global constant outside folds)
- [ ] Mask `rejected_outside_support` for counterfactual window claims (`Q`/`NOV` already NaN on rejected rows in Stage 4)
- [ ] Report `q` and `V_catch` separately before NOV collapse
- [ ] Nested LOGO only

### B3. Dual sampling (`artifacts/STAGE3_DUAL_SAMPLING.md`)

- [ ] **Window series / segmentation:** filter `is_model_5hz_frame == True` (176004 of 190348 rows)
- [ ] **Release / use labels:** use `is_pass_release_frame` / `is_forced_release_frame` + `completion_label_eligible` / `is_true_receiver` — do **not** require `frameIdx % 5 == 0` on release rows
- [ ] Overlaps (`model_5hz+pass_release`) may appear in both roles; segmentation series still keys off `is_model_5hz_frame`
- [ ] Preserve `sampling_role` provenance on Stage 5 outputs

### B4. Narrow path (`configs/stage2_narrow_path.yaml`)

- [ ] Completion / used-window attribution uses **known receivers only** (`target_reliability_status == known_receiver`, `receiver_specific_eligible`)
- [ ] Never treat interceptor / `toReceiverId` as intended target
- [ ] Carry all `GATE_COLUMNS` from `src/passing_windows/narrow_path.py` onto Stage 5 tables
- [ ] No fabricated reviewer values; no “audited failed-pass” language
- [ ] Named failed targets excluded from receiver-specific window **use** labels by default; if ever included, require exclusion sensitivity (prefer exclude)

### B5. Model / schema identity

- [ ] Assert `model_version == stage4_sklearn_logo_v3` on input
- [ ] Do not overwrite Stage 4 prediction columns with newly fitted values
- [ ] Preserve `rejected_outside_support`, `trajectory_choice_rule == lead_if_velocity_ok_else_direct`

---

## C. Required deliverables for Stage 5 DONE

### C1. Code

| Path | Role |
|---|---|
| `src/passing_windows/windows/__init__.py` | Package export |
| `src/passing_windows/windows/smoothing.py` | Causal smooth of cross-fitted `q` / `V_catch` / `NOV` (and optionally `Q`) on 5 Hz series |
| `src/passing_windows/windows/thresholds.py` | Fold-internal open/close + late-loss delta selection; robustness grid freeze |
| `src/passing_windows/windows/segment.py` | Hysteresis + persistence → contiguous window intervals per `(gameId, touchId, candidateId)` |
| `src/passing_windows/windows/labels.py` | open / used / late / unused / never_open assignment + continuous metrics |
| `src/passing_windows/windows/option_set.py` | Per-frame option-set features (touch × frame) |
| `src/passing_windows/windows/leakage.py` | Guards: no future smooth; threshold LOGO isolation; no Stage 4 refit |
| `pipelines/05_segment_windows.py` | Restartable Stage 5 entrypoint (read Stage 4 → write tables/artifacts) |
| `pipelines/_stage5_adversarial_live_audit.py` | Live checklist runner (mirror Stage 4 auditor) |
| `tests/test_stage5.py` | Full regression suite mapping every S5 finding |
| `configs/stage5_windows.yaml` (recommended) | Frozen post-selection thresholds per fold + robustness grid + late delta procedure (values written after fold-internal selection; procedure locked a priori) |

### C2. Tables

| Path | Grain | Required content |
|---|---|---|
| `tables/stage5_candidate_series.parquet` | `(gameId, touchId, frameIdx, candidateId)` on **5 Hz** rows | Smoothed `q_smooth`, `V_catch_smooth`, `Q_smooth`, `NOV_smooth`; raw Stage 4 scores retained; open-state flags; fold_id; model_version; GATE_COLUMNS; `rejected_outside_support`; dual-sampling flags |
| `tables/stage5_windows.parquet` | One row per detected window interval (and never-open stubs per candidate-touch if required by labels) | opening/peak/closing frame+time, duration_s, peak_NOV, integrated_NOV, label, delays, value_at_use_vs_peak, fold thresholds used, existence_prob (+ status) |
| `tables/stage5_option_set_frames.parquet` | `(gameId, touchId, frameIdx)` on 5 Hz | best_NOV, n_viable, logsumexp_Q or NOV, best–second gap, open/close event rates, accumulated area, entropy/concentration |
| `tables/stage5_threshold_selections.parquet` | `(fold_id, …)` | open_threshold, close_threshold, late_use_material_loss_delta, selection diagnostics (train games only), robustness grid id |
| `tables/stage5_threshold_grid_windows.parquet` (or long summary) | Windows / counts under each grid point | Required for freeze `threshold_grid_policy` |

### C3. Artifacts

| Path | Role |
|---|---|
| `artifacts/STAGE5_VERIFICATION.md` | Checklist + live metrics |
| `artifacts/STAGE5_ADVERSARIAL_AUDIT.md` | Findings with PASS/FAIL (Stage 4 style) |
| `artifacts/STAGE5_REGRESSION_LOCK.md` | Finding → `test_*` map |
| `artifacts/STAGE5_HANDOFF_STAGE6.md` | Inputs Stage 6 must consume; dual sampling reminder; what is deferred to Stage 7 |
| `artifacts/stage5_output_manifest.json` | SHA-256 of all Stage 5 outputs |
| `artifacts/stage5_fold_thresholds.json` | Machine-readable per-fold open/close/late delta + grid |
| `artifacts/stage5_window_report.md` | Counts by label, duration summaries, per-game breakdowns (descriptive only) |

### C4. Tests (minimum set — expand per §D)

- Unit: hysteresis open/close, persistence wall-clock, never-open, used/late/unused edge cases
- Unit: causal smoother rejects future peek
- Unit: option-set feature formulas on synthetic 4-candidate frames
- Integration: LOGO threshold isolation (thresholds fit on 9 games; applied to 10th)
- Integration: 5 Hz filter; release-frame use labels join without requiring lattice alignment
- Integration: rejected rows never count as open windows for counterfactual claims
- Lock: no Stage 4 model fit imports / no `fit(` on passability/catch/keep/choice in Stage 5 pipeline
- Lock: GATE_COLUMNS present; narrow-path known-receiver for `used`
- Lock: forbidden language scan on reports

---

## D. Adversarial checklist (BLOCKER / MAJOR / MINOR)

Severity template matches Stage 4: Finding ID · Severity · Rule · Exact failure mode · Required fix · Proposed regression test.

### D0. Pre-existing gaps (current NOT STARTED state)

These are **expected open BLOCKERs** until the builder lands code + tests. Clearing them is the definition of implementation progress.

| Finding ID | Severity | Rule citation | Exact failure mode | Required fix | Proposed regression test |
|---|---|---|---|---|---|
| S5-B00 | BLOCKER | Plan Stage 5; README Stage 5+ | No `pipelines/05_segment_windows.py`, no `src/passing_windows/windows/`, no `tests/test_stage5.py` | Implement package + pipeline + tests per §C/E | `test_stage5_pipeline_entrypoint_exists` |
| S5-B01 | BLOCKER | Handoff primary table | Stage 5 cannot run without consuming `tables/stage4_candidate_predictions.parquet` 1:1 with Stage 3 keys | Pipeline loads Stage 4; asserts row count 190348 and join keys unique | `test_stage5_consumes_stage4_predictions_1to1` |
| S5-B02 | BLOCKER | Handoff: do not refit | Any Stage 5 path that re-trains `q`/`V_catch`/`V_keep`/choice | Read-only scores; forbid fit entrypoints in Stage 5 modules | `test_stage5_no_component_model_refit` |

### D1. Dual sampling & series construction

| Finding ID | Severity | Rule citation | Exact failure mode | Required fix | Proposed regression test |
|---|---|---|---|---|---|
| S5-B03 | BLOCKER | `STAGE3_DUAL_SAMPLING.md`; freeze `model_hz: 5` | Window segmentation run on all rows including forced release-only frames | Segment only `is_model_5hz_frame`; document release rows used solely for use/late timing joins | `test_stage5_windows_use_model_5hz_series_only` |
| S5-B04 | BLOCKER | Dual sampling release role | `used`/`late` require release on 5 Hz lattice (`frameIdx % 5 == 0`) | Join actual release frames (forced OK) to candidate windows by `(gameId, touchId, candidateId)` + time | `test_stage5_use_labels_allow_off_lattice_release` |
| S5-M01 | MAJOR | Dual sampling provenance | Dropping `sampling_role` / dual flags on Stage 5 tables | Retain flags from Stage 4 | `test_stage5_sampling_role_retained` |
| S5-N01 | MINOR | Freeze render_hz 25 | Times exported only as frameIdx without seconds | Emit `*_time_s` using 25 Hz clock (`frameIdx / 25`) consistently | `test_stage5_times_use_25hz_clock` |

### D2. Causal smoothing

| Finding ID | Severity | Rule citation | Exact failure mode | Required fix | Proposed regression test |
|---|---|---|---|---|---|
| S5-B05 | BLOCKER | Freeze `smoothing_causality: no_future_frames` | Any smoother using future samples (centered rolling mean, filtfilt, etc.) | Causal smoother only (e.g. backward EWMA / causal spline / past window); unit test with implanted future spike | `test_stage5_smoothing_is_causal` |
| S5-M02 | MAJOR | Handoff: smooth cross-fitted series | Smoothing raw geometry instead of Stage 4 OOS scores | Smooth `q_passability_model`, `V_catch_model`, `NOV_model` (and `Q_option_model`) written by Stage 4 | `test_stage5_smooths_crossfitted_scores_not_geometry` |
| S5-M03 | MAJOR | Freeze presentation_rule | Smooth only NOV; drop q / V_catch series | Keep smoothed q, V_catch, Q, NOV columns | `test_stage5_components_not_collapsed_to_nov` |

### D3. Threshold protocol (LOGO / TBD_TRAINING_FOLD)

| Finding ID | Severity | Rule citation | Exact failure mode | Required fix | Proposed regression test |
|---|---|---|---|---|---|
| S5-B06 | BLOCKER | Freeze `validation.rule`; `tbd_training_fold.window_open_threshold` | Global open/close thresholds fit on all 10 games | Per `fold_id`: select on `train_gameIds` only; apply to held-out `gameId` | `test_stage5_thresholds_logo_holdout_isolation` |
| S5-B07 | BLOCKER | Freeze hysteresis | Single threshold for open and close | Distinct `open_threshold` and `close_threshold` with `close ≤ open` (or documented hysteresis direction on the open-signal) | `test_stage5_hysteresis_separate_open_close` |
| S5-B08 | BLOCKER | Freeze `min_persistence_seconds: 0.20` | Counting 5 consecutive **5 Hz** frames as “25 Hz persistence” (→ 1.0s) or treating 1 sample as 0 duration forever | Define persistence in **wall-clock seconds** on the 5 Hz timeline: Δt = 0.20s per step; require open-signal hold ≥ 0.20s before committing open (see §F) | `test_stage5_persistence_ge_0_20s_wallclock` |
| S5-B09 | BLOCKER | Freeze `late_use_material_loss_delta` TBD | Late labels using a constant invented from held-out NOV | Predeclare late delta from **training-fold** NOV quantiles before labeling holdout | `test_stage5_late_delta_fold_internal` |
| S5-M04 | MAJOR | Freeze `threshold_grid_policy` | Only one threshold pair; no grid artifact | After primary selection, freeze grid in `stage5_fold_thresholds.json` and emit grid windows/summaries | `test_stage5_threshold_robustness_grid_emitted` |
| S5-M05 | MAJOR | Freeze fold seeds | Threshold RNG / bootstrap uses wall time | Use freeze fold seeds `2025092401`–`10` / master `20250924` if stochastic | `test_stage5_fold_seeds_match_freeze` |

### D4. Support rejection & counterfactuals

| Finding ID | Severity | Rule citation | Exact failure mode | Required fix | Proposed regression test |
|---|---|---|---|---|---|
| S5-B10 | BLOCKER | Handoff mask rejected; Stage4 S4-B22 | Windows opened from `rejected_outside_support` rows using imputed Q/NOV | Treat rejected as non-viable: NaN NOV cannot satisfy `NOV > 0`; do not fill | `test_stage5_rejected_cannot_open_windows` |
| S5-M06 | MAJOR | Kill `counterfactual_support` | Stage 5 claims ignore >10% support rejection context | Report % rejected among 5 Hz series; handoff note for Stage 6 | `test_stage5_support_rejection_rate_reported` |

### D5. Narrow path / labels

| Finding ID | Severity | Rule citation | Exact failure mode | Required fix | Proposed regression test |
|---|---|---|---|---|---|
| S5-B11 | BLOCKER | Narrow path completion labels | `used` attributed via interceptor or failed inferred target | `used` only if pass to candidate with known-receiver completion eligibility (`is_true_receiver` at release ∩ gates) | `test_stage5_used_requires_known_receiver` |
| S5-B12 | BLOCKER | Narrow path GATE_COLUMNS | Stage 5 tables missing gate columns | Propagate all 10 GATE_COLUMNS | `test_stage5_gate_columns_present` |
| S5-B13 | BLOCKER | Forbidden interceptor | `used`/`late` when `candidateId == interceptorId` | Assert never | `test_stage5_interceptor_never_use_target` |
| S5-B14 | BLOCKER | Claims language | Reports say correct/bad decision / points left on table | Scan Stage 5 markdown artifacts | `test_stage5_forbidden_claim_language` |
| S5-M07 | MAJOR | Narrow path sensitivity | Optional named-failed uses without exclusion sensitivity | Default exclude; if included, dual report | `test_stage5_named_failed_excluded_by_default` |

### D6. Window object semantics

| Finding ID | Severity | Rule citation | Exact failure mode | Required fix | Proposed regression test |
|---|---|---|---|---|---|
| S5-B15 | BLOCKER | Plan/freeze open rule | Open if only `q > thr` OR only `NOV > 0` | Require **both** + persistence | `test_stage5_open_requires_q_and_nov` |
| S5-B16 | BLOCKER | Plan labels used/late/unused/never-open | Ambiguous mutually exclusive labels (e.g. row both used and unused) | Enforce disjoint label priority (§G) | `test_stage5_labels_mutually_exclusive` |
| S5-B17 | BLOCKER | Plan continuous outputs | Missing opening/peak/closing/duration/peak_NOV/integrated_NOV | Populate all primary continuous fields on `stage5_windows` | `test_stage5_continuous_window_fields_present` |
| S5-M08 | MAJOR | Plan delay / value_at_use | No delay_opening_or_peak_to_release or value_at_use_vs_peak for used/late | Compute when release exists; NaN otherwise | `test_stage5_use_timing_fields` |
| S5-M09 | MAJOR | Freeze existence probability | Column missing with no deferral status | Either thin Stage 5 score-perturbation existence prob **or** column + `existence_prob_status=deferred_stage7` documented in handoff (full geometry MC remains Stage 7) | `test_stage5_existence_prob_column_or_explicit_deferral` |
| S5-M10 | MAJOR | Kill `window_label_instability` | No measurement path for >15% label flip | Provide Stage 5 hook + Stage 6/7 handoff contract; do not claim stability without numbers | `test_stage5_label_stability_hook_documented` |
| S5-N02 | MINOR | Touch right-censor | Closing at touch end unlabeled | Flag `censored_at_touch_end` for survival handoff to Stage 6 Analysis 4 | `test_stage5_censor_flag_at_touch_end` |

### D7. Option-set features

| Finding ID | Severity | Rule citation | Exact failure mode | Required fix | Proposed regression test |
|---|---|---|---|---|---|
| S5-B18 | BLOCKER | Plan Stage 5 option-set list | Missing best value, n_viable, log-sum-exp, gap, open/close rates, accumulated area, entropy | Implement all seven feature families (§H) on 5 Hz frames | `test_stage5_option_set_features_complete` |
| S5-M11 | MAJOR | Rejection mask | Viable count includes rejected candidates | Exclude `rejected_outside_support` from viability | `test_stage5_option_set_excludes_rejected` |
| S5-N03 | MINOR | Soft total definition | Ambiguous log-sum-exp on Q vs NOV | Freeze formula in `stage5_windows.yaml` / verification (recommend `logsumexp(Q_smooth)` among non-rejected) | `test_stage5_logsumexp_definition_documented` |

### D8. Validation design / data policy

| Finding ID | Severity | Rule citation | Exact failure mode | Required fix | Proposed regression test |
|---|---|---|---|---|---|
| S5-B19 | BLOCKER | Freeze forbidden_splits | Threshold tuning via ShuffleSplit on frames | LOGO only | `test_stage5_no_random_group_split` |
| S5-B20 | BLOCKER | Freeze data_policy | Season aggregates as covariates for thresholds/features | Ban aggregate CSV reads in Stage 5 | `test_stage5_no_season_aggregates` |
| S5-B21 | BLOCKER | All 10 folds | Thresholds missing for any freeze game | Emit selections for all 10 `fold_holdout_*` | `test_stage5_all_ten_logo_folds_present` |
| S5-B22 | BLOCKER | Manifest | Outputs unhashed | Write `stage5_output_manifest.json` | `test_stage5_output_manifest_complete` |
| S5-B23 | BLOCKER | Regression lock | Findings without tests | Maintain `STAGE5_REGRESSION_LOCK.md` covering all BLOCKERs | `test_stage5_regression_lock_covers_blockers` |
| S5-M12 | MAJOR | Primary population | Window stats include transition/inbounds/etc. without flag | Inherit Stage 3/4 eligibility; document population filter | `test_stage5_primary_population_filter` |
| S5-M13 | MAJOR | Stage4 S4-M13 GAM caveat | Stage 5 methods claim GAM windows | Keep sklearn logo_v3 / spline substitution language | `test_stage5_gam_substitution_caveat_retained` |
| S5-N04 | MINOR | Handoff honesty | Handoff omits 5 Hz vs release counts | Document series row counts and label yields | `test_stage5_handoff_coverage_honesty` |

---

## E. Implementation order for the builder (TDD-first)

Do **not** skip to full parquet generation before unit tests exist.

1. **Scaffold + red tests**  
   Create `tests/test_stage5.py` with failing stubs for S5-B00–B08, B15–B18. Create empty `src/passing_windows/windows/` package.

2. **Causal smoother (unit)**  
   Implement `smoothing.py`; lock `test_stage5_smoothing_is_causal`. Synthetic series with future spike must not move past estimates.

3. **Open-signal + hysteresis + persistence (unit)**  
   Implement core state machine in `segment.py` on toy 5 Hz series (Δt=0.20s). Cover flicker (1-frame blip fails persistence), hysteresis close below open, `NOV>0` ∧ `q>open_thr`.

4. **Threshold selectors (unit + LOGO isolation)**  
   Implement `thresholds.py`: train-fold calibration/decision procedure for open/close; late delta from train NOV quantiles; write selections structure. Red/green `test_stage5_thresholds_logo_holdout_isolation`.

5. **Label grammar (unit)**  
   Implement `labels.py` per §G with exhaustive edge-case table. Lock mutual exclusivity + used known-receiver.

6. **Option-set features (unit)**  
   Implement `option_set.py` on synthetic 4-candidate frames; lock §H formulas.

7. **Leakage / refit guards**  
   Static scans + `leakage.py` helpers mirroring Stage 4 patterns.

8. **Pipeline integration**  
   `pipelines/05_segment_windows.py`: load Stage 4 → filter 5 Hz → smooth → per-fold thresholds → segment → labels → option-set → write tables + JSON + markdown. Assert `model_version`.

9. **Robustness grid pass**  
   Re-segment under frozen grid; write grid table; document in `stage5_fold_thresholds.json`.

10. **Existence-prob policy**  
    Either minimal score perturbation **or** explicit Stage 7 deferral column + handoff language (S5-M09).

11. **Artifacts + live audit**  
    Verification, adversarial audit, regression lock, Stage 6 handoff, output manifest. Run `_stage5_adversarial_live_audit.py`.

12. **Acceptance**  
    `pytest tests/test_stage5.py` all green; live audit SAFE TO ACCEPT / CONDITIONAL GO only if majors documented.

---

## F. Threshold protocol (open/close = TBD_TRAINING_FOLD)

### F1. What is selected inside training folds only

For each outer fold `fold_holdout_<held_out_gameId>` with `train_gameIds` from freeze:

| Parameter | Freeze name | Selection locus | Apply to |
|---|---|---|---|
| Open threshold on `q` | `window_open_threshold` | Train games’ 5 Hz (or release-calibration) scores | Held-out game series |
| Close threshold on `q` | `window_close_threshold` | Same, separately | Held-out |
| Late material loss | `late_use_material_loss_delta` | Train NOV distribution quantiles | Held-out late labels |
| Robustness grid | `window_threshold_robustness_grid` | Prespecify after seeing **train** calibration support; freeze identically for all folds | All folds’ specification curve |

**Forbidden:** pooling all 10 games to pick a single global threshold; peeking at held-out calibration curves; fitting thresholds on frames from the held-out `gameId`.

### F2. Recommended selection procedure (builder must document exact choice in `stage5_fold_thresholds.json`)

Primary suggestion (freeze allows “calibration/decision considerations”; lock the chosen procedure before labeling holdout):

1. On training-fold **release ∩ completion_eligible** rows (passability semantics), build reliability / decision curve for `q_passability_model`.
2. Choose `open_threshold` as the smallest q at which empirical completion rate ≥ a predeclared target band **or** Youden/utility on train only — **state the rule in code + JSON**.
3. Choose `close_threshold = open_threshold - δ` with `δ ≥ 0` selected on train to minimize one-step flicker rate among train 5 Hz series while keeping open prevalence in a predeclared band.
4. Require open-signal: `(q_smooth ≥ open_threshold) ∧ (NOV_smooth > 0)` to **enter** opening; remain open while `(q_smooth ≥ close_threshold) ∧ (NOV_smooth > 0)` (or NOV rule stays strict `>0` — document).
5. Persistence: signal must hold for **≥ 0.20 s wall-clock** before an opening is committed; after close signal, require ≥ 0.20 s before committing closure (symmetric anti-flicker), unless touch ends (censor).

### F3. Persistence on the 5 Hz lattice (binding interpretation)

- Native gap between consecutive `is_model_5hz_frame` samples: **0.20 s** (`model_frame_stride: 5` at 25 Hz).
- Freeze `min_persistence_frames_at_25hz: 5` ≡ **0.20 s**, not “5 model frames.”
- **Operational rule:** a qualifying open episode requires the open-signal to be true on a span of length ≥ 0.20 s. On exact 5 Hz data, that means **at least two consecutive 5 Hz samples** (interval 0.20 s) **or** one sample if the builder explicitly treats each sample as representing a 0.20 s hold — **pick one and lock it in tests**.  
  **Auditor recommendation (preferred):** require **≥ 2 consecutive True samples** (duration ≥ 0.20 s between first and last True in the committing run) so single-frame spikes never open. Document in verification.

### F4. Hysteresis diagram (logical)

```
closed --(q≥open_thr ∧ NOV>0 for ≥0.20s)--> open
open   --(q<close_thr ∨ NOV≤0 for ≥0.20s)--> closed
```

with `close_thr ≤ open_thr`.

### F5. Robustness grid

After fold-internal primary selection, freeze a small grid (example shape only — values from train support):

- open ∈ {primary, primary±Δ₁, primary±Δ₂}
- close ∈ {primary_close, …} with close≤open constraints

Re-run window counts / label shares / median duration per game; store in `tables/stage5_threshold_grid_windows.parquet`. Stage 6 headlines must be repeatable on this grid; Stage 5 only produces the objects.

---

## G. Window labels — exact definitions and edge cases

Labels are **secondary** to continuous outputs but must be unambiguous.

### G1. Primitives

- **Open-signal** at 5 Hz frame t for candidate j:  
  `q_smooth(j,t) ≥ open_thr(fold)` **and** `NOV_smooth(j,t) > 0` **and** not `rejected_outside_support`.
- **Qualifying window:** maximal contiguous episode after hysteresis+persistence commit, within one `(gameId, touchId, candidateId)`.
- **Peak:** frame of max `NOV_smooth` inside the window (ties → earliest).
- **Release-to-j:** pass release on this touch with `is_true_receiver` for j and narrow-path known-receiver gates (off-lattice OK).

### G2. Label definitions

| Label | Definition |
|---|---|
| **open** | Frame-level or episode-level state: currently inside a qualifying window (use episode table for lifetimes; frame flag for option-set). |
| **used** | A qualifying window for j exists (or is open) and a release-to-j occurs **while the window is open** (release time ∈ [opening_time, closing_time]). |
| **late** | Release-to-j occurs with material loss relative to peak, **or** shortly after closure. Material loss: `peak_NOV - NOV_at_release ≥ late_use_material_loss_delta(fold)` with release after peak; **or** release in `(closing_time, closing_time + ε]` with ε predeclared from train (recommend ε = 0.20 s). |
| **unused** | ≥1 qualifying window for j on the touch closes with **no** release-to-j during open (and not classified late). |
| **never_open** | No qualifying window for j on the touch (open-signal never commits). |

### G3. Priority / exclusivity (per candidate-touch)

Assign exactly one **touch-level summary label** per `(gameId, touchId, candidateId)`:

1. If any `used` → **used** (even if other unused windows exist on same candidate — document; optional multi-window rows still store episodes).
2. Else if any `late` → **late**.
3. Else if any qualifying window → **unused**.
4. Else → **never_open**.

Episode-level rows in `stage5_windows.parquet` keep granular labels; summary column must follow priority above.

### G4. Edge cases (must have tests)

| Case | Expected |
|---|---|
| Single 5 Hz spike open-signal | Not open (fails persistence) |
| q high but NOV ≤ 0 | Not open |
| NOV > 0 but q below open_thr | Not open |
| Rejected support, finite q filled somehow | Still not open |
| Release on forced non-5Hz frame during open interval | **used** if candidate is true receiver |
| Release after peak with loss ≥ delta while still open | **late** (not used-at-peak); still record value_at_use_vs_peak |
| Release just after close within ε | **late** |
| Release to j with never_open | Not used; descriptive “pass without modeled open window” — do not call mistake |
| Touch ends while open | Close at touch end; `censored_at_touch_end=True`; still a window for duration/survival |
| Multiple disjoint windows, pass in second | Episode rows separate; summary **used** |
| Ambiguous failed pass | Cannot create receiver-specific used/late |
| `NOV` NaN | Cannot open |

### G5. Continuous fields (per window episode)

- `opening_time` / `peak_time` / `closing_time` (frameIdx + seconds)
- `duration_s`
- `peak_NOV`, `integrated_NOV` (trapezoid on 5 Hz; document)
- `delay_opening_to_release_s`, `delay_peak_to_release_s` (NaN if no release-to-j)
- `value_at_use_vs_peak` = `NOV_at_release / peak_NOV` (NaN if unused/never_open)
- `window_existence_probability_under_tracking_perturbation` (+ status)

---

## H. Option-set frame features required

Grain: `(gameId, touchId, frameIdx)` for `is_model_5hz_frame` rows. Candidates = up to 4 teammates; **exclude** `rejected_outside_support` from viability.

| Feature | Definition (lock in code) |
|---|---|
| `best_option_value` | `max_j NOV_smooth(j,t)` among non-rejected; else NaN |
| `n_viable_options` | Count j with open-signal True at t (post-threshold; persistence may be frame-instant for counting — **document**: recommend count raw open-signal, plus `n_viable_persistent` using committed open state) |
| `soft_total_option_value` | `logsumexp(Q_smooth(j,t))` over non-rejected j (temperature=1 unless freeze later) |
| `best_second_gap` | best NOV − second-best NOV (NaN if <2 non-rejected) |
| `opening_rate` / `closing_rate` | Indicators or EWMA of open-commit / close-commit events at t within touch |
| `accumulated_option_value_area` | Running integral of soft total or best NOV from touch start through t (causal) |
| `option_concentration` / `entropy` | Shannon entropy of softmax(NOV) or softmax(Q) over non-rejected j; also report Herfindahl optional |

Also carry: `fold_id`, clocks, `model_version`, GATE_COLUMNS as touch-constant fields.

---

## I. Explicit OUT OF SCOPE (Stage 5)

Do **not** implement in Stage 5:

- **Stage 6 analyses** 1–8 (calibration headlines, KM survival write-ups, predictive ladder, matching used/late/unused outcomes, context chapters)
- **Stage 7** full Monte Carlo geometry perturbation / null models / specification-curve article figures (Stage 5 may only stub existence prob or thin score perturbation)
- **Stage 8–10** findings matrix, article, hero explorer, site
- **Refitting** Stage 4 passability / catch / keep / choice models or changing `stage4_sklearn_logo_v3` predictions
- **Broadening narrow path** (adding unaudited failed targets as completion/use labels)
- **Causal / normative language** in reports
- **Player or team rankings**
- Re-opening Stage 0 freeze values without a new freeze revision
- Using season aggregate CSVs as features
- Using `chances.ptsScored` or `shotQuality` as features
- Substituting interceptor as target

Stage 5 **does** produce the window objects and option-set features Stage 6 will analyze.

---

## J. Definition of Done + kill criteria for Stage 5 acceptance

### J1. Definition of Done (all required)

1. All §C deliverables exist on disk with manifest hashes.
2. `pytest tests/test_stage5.py` passes; every BLOCKER in §D mapped to LOCKED in `STAGE5_REGRESSION_LOCK.md`.
3. Live audit script reports **SAFE TO ACCEPT** (or **CONDITIONAL GO** only for documented residuals analogous to S4-M13 — no open BLOCKERs).
4. Windows segmented only on 5 Hz series; use/late join works for off-lattice releases.
5. Per-fold open/close/late-delta selected on train games only; holdout applies stored thresholds.
6. Persistence ≥ 0.20 s wall-clock with hysteresis; open requires q thr ∧ NOV>0.
7. Rejected candidates cannot open; GATE_COLUMNS present; known-receiver for used.
8. Option-set features complete (§H).
9. No Stage 4 refit; `model_version` unchanged.
10. `STAGE5_HANDOFF_STAGE6.md` lists tables, dual sampling, threshold grid path, and explicit Stage 7 deferrals.
11. Forbidden claim language absent from Stage 5 artifacts.

### J2. Stage 5–relevant kill / narrow criteria (from freeze)

Monitor and report; **fail Stage 5 acceptance** if Stage 5 itself introduces the breach; **handoff as CONDITIONAL** if breach is upstream/deferred:

| Kill ID | Stage 5 responsibility |
|---|---|
| `window_label_instability` (>15% label flip under perturbation) | Must expose measurement hook; if thin Stage 5 perturbation already shows >15% flip on primary labels → **NO-GO** for probabilistic window claims until thresholds/smoothing revised or claims narrowed |
| `counterfactual_support` | Report rejection share; if Stage 5 “opens” rejected rows → **NO-GO** |
| `unresolved_failed_targets` / narrow path | Do not reopen; keep narrow path → used labels known-receiver only |
| `tracking_eligibility_loss` | Do not silently expand population |
| `completion_calibration` / `predictive_improvement` / `null_comparable` / `single_game_driver` | **Out of scope to adjudicate in Stage 5**; do not claim they passed |

### J3. Go / no-go language (mirror Stage 4)

| Verdict | Meaning |
|---|---|
| **SAFE TO ACCEPT** | All BLOCKERs clear live + pytest; majors documented or cleared; Stage 6 may consume windows |
| **CONDITIONAL GO** | Blockers clear; residual majors require documented follow-ups before Stage 6 headlines |
| **NO-GO** | Any open BLOCKER, missing artifacts, LOGO leakage, refit detected, or persistence/threshold protocol violated |

### J4. Minimal-run-and-audit command set (builder target)

```powershell
cd analysis\passing_windows
.\.venv\Scripts\python.exe -m pytest tests\test_stage5.py -q
.\.venv\Scripts\python.exe pipelines\05_segment_windows.py
.\.venv\Scripts\python.exe pipelines\_stage5_adversarial_live_audit.py
```

Expected: pytest all green; live audit `stage5_may_begin_stage6: true` with zero open BLOCKERs.

---

## Appendix — Input column inventory (Stage 4 primary table)

Verified on `tables/stage4_candidate_predictions.parquet` (190348 rows, `model_version=stage4_sklearn_logo_v3`):

Keys / sampling: `gameId`, `touchId`, `frameIdx`, `candidateId`, `fold_id`, `is_pass_release_frame`, `is_model_5hz_frame`, `is_forced_release_frame`, `completion_label_eligible`, `is_true_receiver`, `sampling_role`, …

Scores: `q_passability_model`, `q_ablation_*`, `V_catch_model`, `V_catch_ablation_*`, `V_keep_model`, `V_fail`, `Q_option_model`, `NOV_model`, `choice_prob_model`

Support / gates: `rejected_outside_support`, full narrow-path `GATE_COLUMNS`, `interceptorId`, `true_receiverId`

---

**End of Stage 5 audit brief.** Builder implements; auditor re-runs live adversarial checklist after artifacts land. Do not mark Stage 5 complete from this brief alone.
