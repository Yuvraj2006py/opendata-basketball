# Stage 5 Passing Windows MVP Implementation Plan

> **For agentic workers:** Inline execution this session. TDD: tests first in `tests/test_stage5.py`.

**Goal:** Segment causal, LOGO-thresholded passing windows from Stage 4 OOS scores without refitting models.

**Architecture:** `src/passing_windows/windows/` provides smooth → thresholds → segment → labels → option_set. `pipelines/05_segment_windows.py` mirrors Stage 4 restartable LOGO per-game flow.

**Tech Stack:** Python, pandas/numpy, pytest, parquet via existing `io_utils`.

## Global Constraints

- Window series: `is_model_5hz_frame` only; persistence ≥ 0.20s wall-clock (≥2 consecutive 5 Hz samples).
- Causal smooth only; open = q≥open_thr ∧ NOV>0 ∧ not rejected; hysteresis close ≤ open.
- Nested LOGO thresholds; no Stage 4 refit; known-receiver for `used`; carry GATE_COLUMNS.
- MVP session: unit tests + 1-game smoke; document remaining audit gaps in `STAGE5_BUILDER_STATUS.md`.

## Tasks

### Task 1: Red tests (S5-B03,B05–B08,B10,B11,B15+)

**Files:** Create `tests/test_stage5.py`

### Task 2: Windows package

**Files:** `src/passing_windows/windows/{__init__,smooth,thresholds,segment,labels,option_set,leakage}.py`

### Task 3: Pipeline + config

**Files:** `pipelines/05_segment_windows.py`, `configs/stage5_windows.yaml`

### Task 4: Artifacts + README + status + commit

Smoke `--games one`, write verification/handoff/manifest stubs, update README Stage 5.
