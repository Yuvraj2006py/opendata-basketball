# Passing Windows analysis project

Self-contained study workspace for the SkillCorner ACB open-data **Passing Windows** research plan.

## Stage status

| Stage | Status |
|---|---|
| 0 — Freeze specification | **Complete** (approved) |
| 1 — Canonical data layer | **Complete** (re-audit CLEAR) |
| 2 — Failed-pass target inference | **Accepted under the narrow path** (human audit deferred; labels provisional) |
| 3 — Candidate pass reconstruction | **Complete** (narrow-path handoff ready for Stage 4) |
| 4 — Cross-fitted component models | **Complete** (LOGO q / V_catch / V_keep / Q / NOV; handoff ready for Stage 5) |
| 5 — Window segmentation | **MVP** (unit tests green; 1-game smoke; full 10-game run pending) |
| 6+ | Not started |

## Setup

```powershell
cd analysis\passing_windows
python -m venv .venv
.\.venv\Scripts\pip.exe install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest tests\test_stage1.py tests\test_stage2.py tests\test_stage3.py tests\test_stage4.py tests\test_stage5.py -q
.\.venv\Scripts\python.exe pipelines\01_build_canonical_data.py
.\.venv\Scripts\python.exe pipelines\02_infer_failed_pass_targets.py
.\.venv\Scripts\python.exe pipelines\03_build_candidate_states.py
.\.venv\Scripts\python.exe pipelines\04_fit_component_models.py
.\.venv\Scripts\python.exe pipelines\05_segment_windows.py
.\.venv\Scripts\python.exe pipelines\04_fit_component_models.py
```

Full Stage 0–3 rebuild commands (with `--force` / `--verify-manifest`): see
[`artifacts/STAGE3_REBUILD.md`](artifacts/STAGE3_REBUILD.md).

Both Stage 1–2 pipelines are restartable per game and take the same flags: `--force` rebuilds
cached per-game tables, `--games <id> [<id> ...]` limits the run, and
`--verify-manifest` re-checks every output hash. Stage 1 also takes
`--refresh-sign-off` to re-hash the verification checklist after a human records
approval. Stage 2 exits 2 if any geometry *invariant* or any narrow-path gate fails
(including a non-blank reviewer column in the audit sheet); data *diagnostics* are
printed and written to the self-check report without failing the run. Stage 3 exits 2
on self-check or manifest failure; fold flight-time models always re-fit so held-out
games never leak into their own flight curves.

## Stage 1 artifacts

| Artifact | Path |
|---|---|
| User verification checklist | [`artifacts/STAGE1_VERIFICATION.md`](artifacts/STAGE1_VERIFICATION.md) |
| Adversarial audit report | [`artifacts/STAGE1_AUDIT_REPORT.md`](artifacts/STAGE1_AUDIT_REPORT.md) |
| Audit fix notes | [`artifacts/STAGE1_FIX_NOTES.md`](artifacts/STAGE1_FIX_NOTES.md) |
| CONSORT sample flow | [`artifacts/stage1_sample_flow.md`](artifacts/stage1_sample_flow.md) |
| Join audit | [`artifacts/stage1_join_audit.md`](artifacts/stage1_join_audit.md) |
| Calculation audit | [`artifacts/stage1_calculation_audit.md`](artifacts/stage1_calculation_audit.md) |
| Tracking quality | [`artifacts/stage1_quality_report.md`](artifacts/stage1_quality_report.md) |
| Output manifest (SHA-256) | [`artifacts/stage1_output_manifest.json`](artifacts/stage1_output_manifest.json) |
| Canonical tables | [`tables/*.parquet`](tables/) |

## Key Stage 1 finding (coordinates)

Broadcast tracking and event locations agree when `possessions.leftHoop=True`. When `leftHoop=False`, tracking must be rotated 180° `(x,y)→(-x,-y)` to match the event frame (offensive hoop at negative x). Stage 1 stores raw `x,y` plus `x_event,y_event` and attack-normalized `x_norm,y_norm`.

**Never differentiate `x_event`/`y_event`.** They flip sign whenever the attacking
hoop changes, which turns a possession change into an apparent teleport. Causal
velocities are taken on the continuous broadcast coordinates and only then rotated
into the event frame, with resets at orientation boundaries and frame gaps wider
than 1 s and a 40 ft/s plausibility screen.

## Stage 2 artifacts

| Artifact | Path |
|---|---|
| Narrow-path lock (decision) | [`artifacts/STAGE2_NARROW_PATH_LOCK.md`](artifacts/STAGE2_NARROW_PATH_LOCK.md) |
| Narrow-path lock (machine-readable) | [`configs/stage2_narrow_path.yaml`](configs/stage2_narrow_path.yaml) |
| Adversarial audit report | [`artifacts/STAGE2_AUDIT_REPORT.md`](artifacts/STAGE2_AUDIT_REPORT.md) |
| User verification checklist | [`artifacts/STAGE2_VERIFICATION.md`](artifacts/STAGE2_VERIFICATION.md) |
| Target inference report | [`artifacts/stage2_target_inference_report.md`](artifacts/stage2_target_inference_report.md) |
| Resolved gate state | [`artifacts/stage2_narrow_path_state.json`](artifacts/stage2_narrow_path_state.json) |
| Label eligibility gate (Stage 3 join target) | `tables/stage2_label_eligibility.parquet` |
| Geometry self-consistency report | [`artifacts/stage2_geometry_selfcheck.md`](artifacts/stage2_geometry_selfcheck.md) |
| Reviewer audit sheet (blank columns) | [`artifacts/stage2_ambiguous_audit.csv`](artifacts/stage2_ambiguous_audit.csv) |
| Per-fold thresholds | [`artifacts/stage2_fold_thresholds.json`](artifacts/stage2_fold_thresholds.json) |
| Output manifest (SHA-256) | [`artifacts/stage2_output_manifest.json`](artifacts/stage2_output_manifest.json) |
| Probability distributions (candidate rows) | `tables/stage2_target_probabilities.parquet` |
| Failed-pass assignments (pass rows) | `tables/stage2_failed_pass_assignments.parquet` |
| Held-out completed-pass validation | `tables/stage2_completed_pass_validation.parquet` |

Stage 2 outputs a **distribution over four candidates** for every failed pass;
`inferred_targetId` is populated only where the provisional auto-accept rule fires.
`toReceiverId` is the **interceptor** and is never a candidate or a target — the
candidate set comes from the passer's own lineup, so this holds structurally and is
asserted in `tests/test_stage2.py` against the real tables.

Current run: 89 failed passes, 60 auto-accepted (67.4%), 27 ambiguous (30.3%), 2
unresolved (2.2%). Strict unresolved clears the 10% kill criterion; the pre-audit
figure (unresolved + ambiguous, 32.6%) does not.

### Narrow path is ACTIVE and binding

No human audit of failed-pass targets exists (`human_audit_status:
deferred_waived`) and pre-audit unresolved breaches the kill line, so Stage 0's
`on_inadequate_reliability` clause is the Stage 2 acceptance mode. It is locked in
[`configs/stage2_narrow_path.yaml`](configs/stage2_narrow_path.yaml) and enforced
by `src/passing_windows/narrow_path.py` — not merely described in a report.

Stage 3+ must join `tables/stage2_label_eligibility.parquet` and take completion
labels through `narrow_path.receiver_completion_training_labels`, which returns
**completed passes with known receivers only** (4530 of 4619). All 89 failed
passes are restricted to touch-level turnover and pass-attempt features. The 60
auto-accepted inferred targets are opt-in via `usable_as_named_failed_target` and
require an exclusion sensitivity check in the same artifact.

Held-out completed-pass top-1 agreement 0.819 and auto-accept precision 0.968 are
**diagnostics of the scoring machinery measured on completed passes**. They are
not failed-pass precision and may not be cited as such; the pipeline scans its own
generated reports for that phrasing and fails on it.

## Stage 2 handover notes

- `candidate_states.parquet` is the materialized decision-state layer: one row per
  (touch, 5 Hz frame, candidate receiver) with the defensive assignment resolved
  from the `matchups` interval feed.
- `tracking_frames.parquet` retains every 25 Hz frame within ±1 s of a pass
  release/end (±2 s around non-complete releases), so ball flight can be read
  without re-streaming the raw feed.
- `receiverLoc`, `distance`, `receiverRegion`, and `receiverId` are null for
  **every** non-complete pass in the raw feed. Use `pass_outcome_class` and
  `failed_pass_inference_eligible`, not `complete.astype(bool)`.

Do not use `data/aggregates/*.csv` as contemporaneous model covariates.

## Stage 3 artifacts

| Artifact | Path |
|---|---|
| Candidate features (aggregate) | `tables/stage3_candidate_features.parquet` |
| Support-rejection table | `tables/stage3_support_rejection.parquet` |
| Per-game caches + `_DONE_STAGE3` | `tables/by_game/{id}/` |
| Candidate report | [`artifacts/stage3_candidate_report.md`](artifacts/stage3_candidate_report.md) |
| Verification checklist | [`artifacts/STAGE3_VERIFICATION.md`](artifacts/STAGE3_VERIFICATION.md) |
| Geometry self-check | [`artifacts/stage3_geometry_selfcheck.md`](artifacts/stage3_geometry_selfcheck.md) |
| Fold flight-time models | [`artifacts/stage3_fold_flight_models.json`](artifacts/stage3_fold_flight_models.json) |
| Output manifest (SHA-256) | [`artifacts/stage3_output_manifest.json`](artifacts/stage3_output_manifest.json) |
| Rebuild commands | [`artifacts/STAGE3_REBUILD.md`](artifacts/STAGE3_REBUILD.md) |
| Stage 4 handoff | [`artifacts/STAGE3_HANDOFF_STAGE4.md`](artifacts/STAGE3_HANDOFF_STAGE4.md) |

Stage 3 writes **four candidate rows per decision frame** under dual sampling:
primary-eligible **5 Hz** frames for window series, plus **forced pass-release**
frames (even off the stride-5 lattice) so Stage 4 can train passability at
release-time state — see [`artifacts/STAGE3_DUAL_SAMPLING.md`](artifacts/STAGE3_DUAL_SAMPLING.md).
Both `direct_*` and `lead_*` geometry families are retained; primary trajectory
choice is `lead_if_velocity_ok_else_direct`. Flight time is leave-one-game-out
(always fit on all train-fold games, even under `--games`). Completion labels
join `tables/stage2_label_eligibility.parquet` via
`receiver_completion_training_labels`. Stage 4 model columns remain NaN
placeholders.

## Stage 4 artifacts

| Artifact | Path |
|---|---|
| Candidate predictions (aggregate) | `tables/stage4_candidate_predictions.parquet` |
| Passability ablations (long) | `tables/stage4_passability_ablations_long.parquet` |
| Per-game caches + `_DONE_STAGE4` | `tables/by_game/{id}/` |
| Model report | [`artifacts/stage4_model_report.md`](artifacts/stage4_model_report.md) |
| Verification checklist | [`artifacts/STAGE4_VERIFICATION.md`](artifacts/STAGE4_VERIFICATION.md) |
| Fold metrics JSON | [`artifacts/stage4_fold_metrics.json`](artifacts/stage4_fold_metrics.json) |
| Output manifest (SHA-256) | [`artifacts/stage4_output_manifest.json`](artifacts/stage4_output_manifest.json) |
| Regression lock | [`artifacts/STAGE4_REGRESSION_LOCK.md`](artifacts/STAGE4_REGRESSION_LOCK.md) |
| Stage 5 handoff | [`artifacts/STAGE4_HANDOFF_STAGE5.md`](artifacts/STAGE4_HANDOFF_STAGE5.md) |

Stage 4 fits nested leave-one-game-out passability (5 ablations), V_catch,
V_keep, Q/NOV, and a descriptive release-time choice model. Sklearn B-spline +
L2 logistic/Ridge substitutes for the freeze GAM (documented in verification).
Counterfactual `Q`/`NOV` are NaN where `rejected_outside_support`.

## Stage 5 artifacts

| Artifact | Path |
|---|---|
| Candidate series (5 Hz, smoothed) | `tables/stage5_candidate_series.parquet` |
| Windows (+ never_open stubs) | `tables/stage5_windows.parquet` |
| Option-set frames | `tables/stage5_option_set_frames.parquet` |
| Threshold selections | `tables/stage5_threshold_selections.parquet` |
| Fold thresholds JSON | [`artifacts/stage5_fold_thresholds.json`](artifacts/stage5_fold_thresholds.json) |
| Verification | [`artifacts/STAGE5_VERIFICATION.md`](artifacts/STAGE5_VERIFICATION.md) |
| Stage 6 handoff | [`artifacts/STAGE5_HANDOFF_STAGE6.md`](artifacts/STAGE5_HANDOFF_STAGE6.md) |
| Builder status / gaps | [`artifacts/STAGE5_BUILDER_STATUS.md`](artifacts/STAGE5_BUILDER_STATUS.md) |
| Output manifest | [`artifacts/stage5_output_manifest.json`](artifacts/stage5_output_manifest.json) |

Stage 5 segments causal, LOGO-thresholded windows from Stage 4 OOS scores.
Do not refit component models. Full 10-game run: `pipelines/05_segment_windows.py`.
