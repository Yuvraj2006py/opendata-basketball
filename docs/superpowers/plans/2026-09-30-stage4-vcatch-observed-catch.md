# Stage 4 V_catch Observed-Catch Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `stage4_sklearn_logo_v2` that trains `V_catch` on observed true-receiver rows (±0.4s around release) while scoring all candidates.

**Architecture:** Add `observed_catch_training_mask` in `catch_value.py`; default `fit_catch_value` to that population; keep Ridge + splines and feature lists; update Stage 4 pipeline metrics/report to evaluate on the observed-catch holdout mask; retain all-candidate fit as ablation column.

**Tech Stack:** Python, pandas, numpy, sklearn (`Ridge` + `SplineTransformer`), pytest, existing Stage 4 LOGO pipeline.

## Global Constraints

- Model version: `stage4_sklearn_logo_v2`
- Half-window: `0.4` seconds at `FRAME_RATE_HZ = 25.0` (10 frames)
- Nested LOGO only; no random frame/touch/chance splits
- Outcomes from shots + free_throws only; features at projected catch only
- Do not change passability, choice, `V_keep`, or Q/NOV formulas
- Spec: `docs/superpowers/specs/2026-09-30-stage4-vcatch-observed-catch-design.md`

---

### Task 1: Observed-catch training mask + fit default

**Files:**
- Modify: `analysis/passing_windows/src/passing_windows/models/catch_value.py`
- Modify: `analysis/passing_windows/src/passing_windows/models/__init__.py` (`MODEL_VERSION`)
- Test: `analysis/passing_windows/tests/test_stage4.py`

**Interfaces:**
- Produces: `observed_catch_training_mask(df, *, half_window_s=0.4, frame_rate_hz=FRAME_RATE_HZ) -> pd.Series`
- Produces: `fit_catch_value(..., training_population: str = "observed_catch")` where `"observed_catch"` | `"all_candidates"`

- [ ] **Step 1: Write failing mask tests**

```python
from passing_windows.models.catch_value import observed_catch_training_mask

def test_observed_catch_mask_release_and_nearby_5hz():
    df = _toy_candidates(n_frames=30)
    # release at f=10 -> frameIdx=50; true receiver candidateId=100
    mask = observed_catch_training_mask(df, half_window_s=0.4)
    assert mask.loc[(df["frameIdx"] == 50) & (df["candidateId"] == 100)].all()
    # ±10 frames at 25Hz: frameIdx 40..60
    near = df[(df["candidateId"] == 100) & (df["frameIdx"] - 50).abs() <= 10]
    assert mask.loc[near.index].all()
    far = df[(df["candidateId"] == 100) & (df["frameIdx"] - 50).abs() > 10]
    assert not mask.loc[far.index].any()
    other = df[df["candidateId"] != 100]
    assert not mask.loc[other.index].any()


def test_observed_catch_mask_excludes_ineligible_release():
    df = _toy_candidates(n_frames=20)
    df.loc[df["frameIdx"] == 50, "completion_label_eligible"] = False
    mask = observed_catch_training_mask(df)
    assert not mask.any()
```

- [ ] **Step 2: Run tests — expect ImportError / AttributeError**

Run: `pytest analysis/passing_windows/tests/test_stage4.py::test_observed_catch_mask_release_and_nearby_5hz analysis/passing_windows/tests/test_stage4.py::test_observed_catch_mask_excludes_ineligible_release -v`

- [ ] **Step 3: Implement mask + wire into `fit_catch_value`**

```python
def observed_catch_training_mask(
    df: pd.DataFrame,
    *,
    half_window_s: float = 0.4,
    frame_rate_hz: float = FRAME_RATE_HZ,
) -> pd.Series:
    out = pd.Series(False, index=df.index)
    eligible = df["completion_label_eligible"].fillna(False).astype(bool)
    true_rec = df["is_true_receiver"].fillna(False).astype(bool)
    release_true = eligible & true_rec
    out |= release_true
    if not release_true.any():
        return out
    half_frames = float(half_window_s) * float(frame_rate_hz)
    anchors = df.loc[release_true, ["gameId", "touchId", "candidateId", "frameIdx"]].rename(
        columns={"frameIdx": "release_frameIdx", "candidateId": "true_candidateId"}
    )
    base = df.reset_index()
    merged = base.merge(anchors, on=["gameId", "touchId"], how="inner")
    is_5hz = merged["is_model_5hz_frame"].fillna(False).astype(bool)
    near = (
        is_5hz
        & (merged["candidateId"] == merged["true_candidateId"])
        & ((merged["frameIdx"] - merged["release_frameIdx"]).abs() <= half_frames)
    )
    out.loc[merged.loc[near, "index"].unique()] = True
    return out
```

In `fit_catch_value`, after selecting `base`, if `training_population == "observed_catch"` apply mask before `prepare_catch_training`. Set `MODEL_VERSION = "stage4_sklearn_logo_v2"`.

- [ ] **Step 4: Update toy catch test to use enough observed rows; run unit tests**

Run: `pytest analysis/passing_windows/tests/test_stage4.py -k "observed_catch or fit_catch" -v`

- [ ] **Step 5: Commit**

```bash
git add analysis/passing_windows/src/passing_windows/models/catch_value.py analysis/passing_windows/src/passing_windows/models/__init__.py analysis/passing_windows/tests/test_stage4.py
git commit -m "feat(stage4): train V_catch on observed-catch window"
```

---

### Task 2: Pipeline metrics on observed-catch holdout + ablation column

**Files:**
- Modify: `analysis/passing_windows/pipelines/04_fit_component_models.py`
- Modify: `analysis/passing_windows/artifacts/STAGE4_VERIFICATION.md` (via renderer)
- Test: `analysis/passing_windows/tests/test_stage4.py`

**Interfaces:**
- Consumes: `observed_catch_training_mask`, `fit_catch_value(..., training_population=...)`
- Produces: fold metrics keys `overall_catch` (observed-catch), `overall_catch_all_candidates`, `per_game_catch_observed`, optional `spearman`; prediction column `V_catch_ablation_all_candidates`

- [ ] **Step 1: Write failing test for fold-metrics keys (unit-level helper or needs_stage4 soft assert)**

```python
def test_fold_metrics_schema_documents_observed_catch_keys():
    # Documents required keys for v2 reports; used after pipeline refresh.
    required = {"rmse", "mae", "r2", "n", "spearman"}
    assert required  # placeholder replaced by real assert once sample dict built in test helpers
```

Prefer a pure unit test that builds a tiny pred+outcome frame and calls a new helper:

```python
# in catch_value.py or metrics reporting helper
def evaluate_catch_on_mask(pred_df, outcome_df, mask) -> dict
```

- [ ] **Step 2: Implement evaluation helper + pipeline wiring**

In each fold: fit primary `observed_catch` and ablation `all_candidates`; write both prediction columns.  
In aggregate metrics: compute `overall_catch` on observed-catch mask of holdout preds; also all-candidate diagnostic; per-game R²/RMSE; Spearman via `pd.Series.corr(method="spearman")`.  
Update `render_model_report` / `render_verification` notes for observed-catch training.

- [ ] **Step 3: Unit tests for evaluate helper + mask metric keys**

Run: `pytest analysis/passing_windows/tests/test_stage4.py -k "observed_catch or catch" -v`

- [ ] **Step 4: Commit**

```bash
git add analysis/passing_windows/pipelines/04_fit_component_models.py analysis/passing_windows/src/passing_windows/models/catch_value.py analysis/passing_windows/tests/test_stage4.py
git commit -m "feat(stage4): report V_catch metrics on observed-catch holdout"
```

---

### Task 3: Refit Stage 4 (if data present) + verification

**Files:**
- Regenerate: `analysis/passing_windows/tables/stage4_*.parquet`, `artifacts/stage4_*`

- [ ] **Step 1: Run Stage 4 pipeline if Stage 3 features exist**

Run from `analysis/passing_windows`:  
`python pipelines/04_fit_component_models.py --force`  
(or project’s documented CLI)

- [ ] **Step 2: Run full Stage 4 test suite**

`pytest analysis/passing_windows/tests/test_stage4.py -v`

- [ ] **Step 3: Record accept/escalate vs design §2 in model report notes; commit artifacts if repo tracks them**

```bash
git add analysis/passing_windows/artifacts/stage4_model_report.md analysis/passing_windows/artifacts/STAGE4_VERIFICATION.md analysis/passing_windows/artifacts/stage4_fold_metrics.json
git commit -m "chore(stage4): refresh v2 V_catch metrics artifacts"
```

If parquet tables are gitignored, leave them local and note paths in the final summary.

---

## Spec coverage checklist

| Spec requirement | Task |
|---|---|
| Observed-catch ±0.4s mask | Task 1 |
| Score all candidates | Task 1 (predict unchanged) |
| Primary metrics on observed-catch | Task 2 |
| All-candidate ablation | Task 2 |
| Version `stage4_sklearn_logo_v2` | Task 1 |
| Tests for mask + metrics keys | Tasks 1–2 |
| Docs / verification note | Task 2–3 |
| Passability/choice/V_keep unchanged | (no tasks) |
