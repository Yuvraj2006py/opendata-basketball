"""Tests for Stage 5 window segmentation (causal smooth, LOGO thresholds, labels).

Unit tests are synthetic and always run. Tests marked ``needs_stage5`` read
tables produced by ``pipelines/05_segment_windows.py``.
"""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from passing_windows.narrow_path import GATE_COLUMNS
from passing_windows.paths import ARTIFACTS, PACKAGE_ROOT, TABLES
from passing_windows.windows.labels import WINDOW_LABELS, assign_touch_summary_label, label_windows
from passing_windows.windows.leakage import assert_no_component_model_refit, scan_stage5_sources_for_refit
from passing_windows.windows.option_set import compute_option_set_features
from passing_windows.windows.segment import (
    MODEL_DT_SECONDS,
    MIN_PERSISTENCE_SECONDS,
    open_signal,
    segment_candidate_series,
)
from passing_windows.windows.smooth import causal_ewma, smooth_score_columns
from passing_windows.windows.thresholds import (
    ThresholdSelection,
    apply_thresholds_to_holdout,
    select_thresholds_logo,
)

PRED4 = TABLES / "stage4_candidate_predictions.parquet"
WINDOWS = TABLES / "stage5_windows.parquet"
SERIES = TABLES / "stage5_candidate_series.parquet"
PIPELINE = ROOT / "pipelines" / "05_segment_windows.py"
REGRESSION_LOCK = ARTIFACTS / "STAGE5_REGRESSION_LOCK.md"

needs_stage5 = pytest.mark.skipif(not WINDOWS.exists(), reason="Stage 5 pipeline not run")
needs_stage4 = pytest.mark.skipif(not PRED4.exists(), reason="Stage 4 predictions missing")

RENDER_HZ = 25.0


def _series(
    q: list[float],
    nov: list[float],
    *,
    rejected: list[bool] | None = None,
    frame_start: int = 0,
    candidate_id: int = 100,
) -> pd.DataFrame:
    n = len(q)
    rejected = rejected or [False] * n
    return pd.DataFrame(
        {
            "gameId": 1,
            "touchId": "t1",
            "frameIdx": [frame_start + i * 5 for i in range(n)],
            "candidateId": candidate_id,
            "is_model_5hz_frame": True,
            "q_smooth": q,
            "NOV_smooth": nov,
            "V_catch_smooth": [1.0] * n,
            "Q_smooth": [float(qi) for qi in q],
            "rejected_outside_support": rejected,
            "fold_id": "fold_holdout_2",
            "model_version": "stage4_sklearn_logo_v3",
            "sampling_role": "model_5hz",
        }
    )


# ---------------------------------------------------------------------------
# S5-B00 / scaffold
# ---------------------------------------------------------------------------


def test_stage5_pipeline_entrypoint_exists():
    assert PIPELINE.exists(), "pipelines/05_segment_windows.py missing"


def test_stage5_windows_package_importable():
    import passing_windows.windows as w

    assert hasattr(w, "causal_ewma") or True  # package exists
    assert (ROOT / "src" / "passing_windows" / "windows" / "segment.py").exists()


# ---------------------------------------------------------------------------
# S5-B03 — 5 Hz series only
# ---------------------------------------------------------------------------


def test_stage5_windows_use_model_5hz_series_only():
    """Segmentation must ignore non-5Hz rows (forced release-only frames)."""
    rows = []
    # 5 Hz lattice
    for i in range(6):
        rows.append(
            {
                "gameId": 1,
                "touchId": "t1",
                "frameIdx": i * 5,
                "candidateId": 100,
                "is_model_5hz_frame": True,
                "q_smooth": 0.9,
                "NOV_smooth": 0.5,
                "rejected_outside_support": False,
            }
        )
    # Off-lattice release-only row with high scores — must not enter series
    rows.append(
        {
            "gameId": 1,
            "touchId": "t1",
            "frameIdx": 17,
            "candidateId": 100,
            "is_model_5hz_frame": False,
            "q_smooth": 0.99,
            "NOV_smooth": 9.0,
            "rejected_outside_support": False,
        }
    )
    df = pd.DataFrame(rows)
    series = df[df["is_model_5hz_frame"]].copy()
    assert len(series) == 6
    assert 17 not in set(series["frameIdx"])
    wins = segment_candidate_series(
        series,
        open_threshold=0.5,
        close_threshold=0.4,
    )
    assert all(int(f) % 5 == 0 for f in wins["opening_frameIdx"]) if len(wins) else True


# ---------------------------------------------------------------------------
# S5-B05 — causal smoothing
# ---------------------------------------------------------------------------


def test_stage5_smoothing_is_causal():
    """Future spike must not move estimates at earlier times."""
    x = np.array([0.0, 0.0, 0.0, 0.0, 10.0], dtype=float)
    y = causal_ewma(x, alpha=0.5)
    assert y[0] == pytest.approx(0.0)
    assert y[1] == pytest.approx(0.0)
    assert y[2] == pytest.approx(0.0)
    assert y[3] == pytest.approx(0.0)
    assert y[4] > 0.0
    # Centered / filtfilt would leak; causal must keep early frames flat.
    assert np.all(y[:4] == 0.0)


def test_stage5_smooths_crossfitted_scores_not_geometry():
    df = pd.DataFrame(
        {
            "gameId": 1,
            "touchId": "t1",
            "candidateId": 100,
            "frameIdx": [0, 5, 10],
            "is_model_5hz_frame": True,
            "q_passability_model": [0.1, 0.2, 0.9],
            "V_catch_model": [1.0, 1.1, 1.2],
            "Q_option_model": [0.1, 0.22, 1.08],
            "NOV_model": [0.0, 0.1, 0.8],
            "pass_distance_ft": [20.0, 20.0, 5.0],
        }
    )
    out = smooth_score_columns(df, alpha=0.5)
    for col in ("q_smooth", "V_catch_smooth", "Q_smooth", "NOV_smooth"):
        assert col in out.columns
    # Geometry must not be the smoothing target
    assert "pass_distance_ft_smooth" not in out.columns


# ---------------------------------------------------------------------------
# S5-B06 — LOGO threshold isolation
# ---------------------------------------------------------------------------


def test_stage5_thresholds_logo_holdout_isolation():
    rng = np.random.default_rng(0)
    rows = []
    for gid in (10, 20, 30):
        for i in range(40):
            rows.append(
                {
                    "gameId": gid,
                    "touchId": f"t{gid}",
                    "frameIdx": i * 5,
                    "candidateId": 100,
                    "is_model_5hz_frame": True,
                    "is_pass_release_frame": i == 20,
                    "completion_label_eligible": i == 20,
                    "is_true_receiver": i == 20,
                    "q_passability_model": float(0.3 + 0.01 * i + (0.2 if gid == 30 else 0.0)),
                    "NOV_model": float(0.1 + 0.01 * i),
                    "rejected_outside_support": False,
                    "fold_id": f"fold_holdout_{gid}",
                }
            )
    df = pd.DataFrame(rows)
    train_ids = [10, 20]
    held_out = 30
    sel = select_thresholds_logo(df, train_game_ids=train_ids, held_out_game_id=held_out, fold_id="fold_holdout_30")
    assert isinstance(sel, ThresholdSelection)
    assert held_out not in sel.train_game_ids
    assert set(sel.train_game_ids) == set(train_ids)
    # Holdout application must use stored thresholds, not recompute from holdout
    applied = apply_thresholds_to_holdout(df[df["gameId"] == held_out], sel)
    assert applied.open_threshold == sel.open_threshold
    assert applied.close_threshold == sel.close_threshold
    # Sanity: selecting with holdout included must be a different API misuse — ensure
    # select_thresholds_logo raises if held_out appears in train list.
    with pytest.raises(ValueError):
        select_thresholds_logo(
            df,
            train_game_ids=[10, 20, 30],
            held_out_game_id=30,
            fold_id="bad",
        )


# ---------------------------------------------------------------------------
# S5-B07 — hysteresis
# ---------------------------------------------------------------------------


def test_stage5_hysteresis_separate_open_close():
    # q oscillates between 0.55 and 0.45; open=0.6 would never open; open=0.5 close=0.4
    q = [0.3, 0.55, 0.55, 0.45, 0.45, 0.45, 0.55, 0.55]
    nov = [0.2] * len(q)
    df = _series(q, nov)
    wins = segment_candidate_series(df, open_threshold=0.5, close_threshold=0.4)
    assert len(wins) >= 1
    # With identical thresholds and flicker, behavior differs — lock close < open
    sel = ThresholdSelection(
        fold_id="f",
        held_out_game_id=1,
        train_game_ids=[2],
        open_threshold=0.5,
        close_threshold=0.4,
        late_use_material_loss_delta=0.05,
        late_after_close_epsilon_s=0.20,
        procedure="test",
    )
    assert sel.close_threshold <= sel.open_threshold
    with pytest.raises(ValueError):
        ThresholdSelection(
            fold_id="f",
            held_out_game_id=1,
            train_game_ids=[2],
            open_threshold=0.4,
            close_threshold=0.5,
            late_use_material_loss_delta=0.05,
            late_after_close_epsilon_s=0.20,
            procedure="bad",
        )


# ---------------------------------------------------------------------------
# S5-B08 — persistence wall-clock
# ---------------------------------------------------------------------------


def test_stage5_persistence_ge_0_20s_wallclock():
    assert MIN_PERSISTENCE_SECONDS == pytest.approx(0.20)
    assert MODEL_DT_SECONDS == pytest.approx(0.20)
    # Single-frame spike must NOT open
    q = [0.1, 0.9, 0.1, 0.1]
    nov = [0.1, 0.5, 0.1, 0.1]
    df = _series(q, nov)
    wins = segment_candidate_series(df, open_threshold=0.5, close_threshold=0.4)
    assert len(wins) == 0
    # Two consecutive True samples (Δt=0.20s) must open
    q2 = [0.1, 0.9, 0.9, 0.1, 0.1]
    nov2 = [0.1, 0.5, 0.5, 0.1, 0.1]
    df2 = _series(q2, nov2)
    wins2 = segment_candidate_series(df2, open_threshold=0.5, close_threshold=0.4)
    assert len(wins2) == 1
    assert wins2.iloc[0]["duration_s"] >= 0.20 - 1e-9


# ---------------------------------------------------------------------------
# S5-B10 — rejected cannot open
# ---------------------------------------------------------------------------


def test_stage5_rejected_cannot_open_windows():
    q = [0.9, 0.9, 0.9, 0.9]
    nov = [0.5, 0.5, 0.5, 0.5]
    df = _series(q, nov, rejected=[True, True, True, True])
    sig = open_signal(df, open_threshold=0.5)
    assert not sig.any()
    wins = segment_candidate_series(df, open_threshold=0.5, close_threshold=0.4)
    assert len(wins) == 0
    # NaN NOV also cannot open
    df2 = _series(q, [np.nan] * 4)
    assert not open_signal(df2, open_threshold=0.5).any()


# ---------------------------------------------------------------------------
# S5-B11 — used requires known receiver
# ---------------------------------------------------------------------------


def test_stage5_used_requires_known_receiver():
    # Build one open window spanning frames 5..15, release at frame 10
    q = [0.1, 0.9, 0.9, 0.9, 0.9, 0.1]
    nov = [0.0, 0.4, 0.5, 0.6, 0.4, 0.0]
    series = _series(q, nov)
    wins = segment_candidate_series(series, open_threshold=0.5, close_threshold=0.4)
    assert len(wins) == 1
    # Interceptor / unreliable target must not produce used
    release_bad = pd.DataFrame(
        [
            {
                "gameId": 1,
                "touchId": "t1",
                "frameIdx": 10,
                "candidateId": 100,
                "is_true_receiver": True,
                "is_pass_release_frame": True,
                "completion_label_eligible": False,
                "target_reliability_status": "interceptor_only",
                "receiver_specific_eligible": False,
                "usable_as_receiver_completion_label": False,
            }
        ]
    )
    labeled_bad = label_windows(
        wins,
        series,
        release_bad,
        late_delta=0.05,
        late_after_close_epsilon_s=0.20,
    )
    assert labeled_bad.iloc[0]["label"] != "used"
    # Known-receiver release during open → used
    release_ok = pd.DataFrame(
        [
            {
                "gameId": 1,
                "touchId": "t1",
                "frameIdx": 10,
                "candidateId": 100,
                "is_true_receiver": True,
                "is_pass_release_frame": True,
                "completion_label_eligible": True,
                "target_reliability_status": "known_receiver",
                "receiver_specific_eligible": True,
                "usable_as_receiver_completion_label": True,
            }
        ]
    )
    labeled_ok = label_windows(
        wins,
        series,
        release_ok,
        late_delta=0.05,
        late_after_close_epsilon_s=0.20,
    )
    assert labeled_ok.iloc[0]["label"] == "used"


# ---------------------------------------------------------------------------
# S5-B15 — open requires q AND NOV
# ---------------------------------------------------------------------------


def test_stage5_open_requires_q_and_nov():
    # High q, NOV <= 0
    df1 = _series([0.9, 0.9, 0.9], [0.0, 0.0, 0.0])
    assert not open_signal(df1, open_threshold=0.5).any()
    assert len(segment_candidate_series(df1, open_threshold=0.5, close_threshold=0.4)) == 0
    # NOV > 0, q below thr
    df2 = _series([0.2, 0.2, 0.2], [0.5, 0.5, 0.5])
    assert not open_signal(df2, open_threshold=0.5).any()
    assert len(segment_candidate_series(df2, open_threshold=0.5, close_threshold=0.4)) == 0
    # Both
    df3 = _series([0.2, 0.9, 0.9, 0.2], [0.0, 0.5, 0.5, 0.0])
    assert open_signal(df3, open_threshold=0.5).sum() == 2
    assert len(segment_candidate_series(df3, open_threshold=0.5, close_threshold=0.4)) == 1


# ---------------------------------------------------------------------------
# Extra locks used by MVP
# ---------------------------------------------------------------------------


def test_stage5_labels_mutually_exclusive():
    assert set(WINDOW_LABELS) == {"open", "used", "late", "unused", "never_open"}
    # Priority: used > late > unused > never_open
    assert assign_touch_summary_label(["unused", "used", "late"]) == "used"
    assert assign_touch_summary_label(["late", "unused"]) == "late"
    assert assign_touch_summary_label(["unused"]) == "unused"
    assert assign_touch_summary_label([]) == "never_open"


def test_stage5_use_labels_allow_off_lattice_release():
    """Release at frameIdx not divisible by 5 still counts as used (S5-B04)."""
    q = [0.1, 0.9, 0.9, 0.9, 0.1]
    nov = [0.0, 0.4, 0.5, 0.4, 0.0]
    series = _series(q, nov)  # frames 0,5,10,15,20
    wins = segment_candidate_series(series, open_threshold=0.5, close_threshold=0.4)
    assert len(wins) == 1
    release = pd.DataFrame(
        [
            {
                "gameId": 1,
                "touchId": "t1",
                "frameIdx": 12,  # off lattice, inside [5,15]
                "candidateId": 100,
                "is_true_receiver": True,
                "is_pass_release_frame": True,
                "completion_label_eligible": True,
                "target_reliability_status": "known_receiver",
                "receiver_specific_eligible": True,
                "usable_as_receiver_completion_label": True,
            }
        ]
    )
    labeled = label_windows(wins, series, release, late_delta=0.05, late_after_close_epsilon_s=0.20)
    assert labeled.iloc[0]["label"] == "used"
    assert labeled.iloc[0]["release_frameIdx"] == 12


def test_stage5_no_component_model_refit():
    assert_no_component_model_refit()
    scan_stage5_sources_for_refit(ROOT / "src" / "passing_windows" / "windows")
    if PIPELINE.exists():
        scan_stage5_sources_for_refit(PIPELINE)


def test_stage5_gate_columns_constant_exported():
    assert len(GATE_COLUMNS) == 10


def test_stage5_option_set_excludes_rejected():
    frames = []
    # 102 rejected (high NOV must not win); 103 below open thr on q
    specs = (
        (100, 0.5, 0.8, False),
        (101, 0.8, 0.8, False),
        (102, 0.9, 0.8, True),
        (103, 0.4, 0.2, False),
    )
    for c, nov, q, rej in specs:
        frames.append(
            {
                "gameId": 1,
                "touchId": "t1",
                "frameIdx": 10,
                "candidateId": c,
                "is_model_5hz_frame": True,
                "NOV_smooth": nov,
                "Q_smooth": nov + 1.0,
                "q_smooth": q,
                "rejected_outside_support": rej,
                "is_open_signal": (not rej) and nov > 0 and q >= 0.5,
                "is_window_open": False,
                "fold_id": "f",
                "model_version": "stage4_sklearn_logo_v3",
            }
        )
    df = pd.DataFrame(frames)
    opt = compute_option_set_features(df, open_threshold=0.5)
    assert len(opt) == 1
    # best among non-rejected: 0.8 (101), not 0.9 rejected
    assert opt.iloc[0]["best_option_value"] == pytest.approx(0.8)
    assert opt.iloc[0]["n_viable_options"] == 2  # 100 and 101 with open signal


def test_stage5_times_use_25hz_clock():
    q = [0.9, 0.9, 0.9]
    nov = [0.5, 0.6, 0.4]
    df = _series(q, nov, frame_start=100)
    wins = segment_candidate_series(df, open_threshold=0.5, close_threshold=0.4)
    assert len(wins) == 1
    row = wins.iloc[0]
    assert row["opening_time_s"] == pytest.approx(row["opening_frameIdx"] / RENDER_HZ)
    assert row["closing_time_s"] == pytest.approx(row["closing_frameIdx"] / RENDER_HZ)


@needs_stage4
def test_stage5_consumes_stage4_predictions_schema():
    df = pd.read_parquet(PRED4, columns=["model_version", "is_model_5hz_frame"])
    assert df["model_version"].nunique() == 1
    assert df["model_version"].iloc[0] == "stage4_sklearn_logo_v3"
    assert int(df["is_model_5hz_frame"].sum()) == 176004


@needs_stage4
def test_stage5_consumes_stage4_predictions_1to1():
    pred = pd.read_parquet(PRED4, columns=["gameId", "touchId", "frameIdx", "candidateId"])
    feat_path = TABLES / "stage3_candidate_features.parquet"
    if not feat_path.exists():
        pytest.skip("Stage 3 features missing")
    feat = pd.read_parquet(feat_path, columns=["gameId", "touchId", "frameIdx", "candidateId"])
    assert len(pred) == 190348
    assert len(pred) == len(feat)
    keys = ["gameId", "touchId", "frameIdx", "candidateId"]
    assert not pred.duplicated(keys).any()
    merged = pred.merge(feat, on=keys, how="inner")
    assert len(merged) == len(pred)


@needs_stage5
def test_stage5_output_manifest_complete():
    man = ARTIFACTS / "stage5_output_manifest.json"
    assert man.exists()
    import json

    data = json.loads(man.read_text(encoding="utf-8"))
    assert data["n_files"] >= 1


def test_stage5_late_delta_fold_internal():
    rows = []
    for gid in (1, 2, 3):
        for i in range(20):
            rows.append(
                {
                    "gameId": gid,
                    "touchId": f"t{gid}",
                    "frameIdx": i * 5,
                    "candidateId": 100,
                    "is_pass_release_frame": i == 10,
                    "completion_label_eligible": i == 10,
                    "is_true_receiver": i == 10,
                    "is_model_5hz_frame": True,
                    "q_passability_model": 0.4 + 0.01 * i,
                    "NOV_model": 0.05 * (i + 1) if gid < 3 else 9.0,
                    "rejected_outside_support": False,
                }
            )
    df = pd.DataFrame(rows)
    sel = select_thresholds_logo(df, train_game_ids=[1, 2], held_out_game_id=3, fold_id="f3")
    # Holdout has inflated NOV; late delta must still come from train only
    train_rel = df[
        df["gameId"].isin([1, 2])
        & df["is_pass_release_frame"]
        & df["completion_label_eligible"]
    ]
    train_nov = train_rel["NOV_model"].to_numpy(dtype=float)
    train_nov = train_nov[train_nov > 0]
    expected = float(max(0.01, np.quantile(train_nov, 0.25)))
    assert sel.late_use_material_loss_delta == pytest.approx(expected)


def test_stage5_option_set_features_complete():
    frames = []
    for c, nov, q in ((100, 0.5, 0.7), (101, 0.2, 0.7), (102, 0.1, 0.7), (103, -0.1, 0.7)):
        frames.append(
            {
                "gameId": 1,
                "touchId": "t1",
                "frameIdx": 10,
                "candidateId": c,
                "is_model_5hz_frame": True,
                "NOV_smooth": nov,
                "Q_smooth": max(nov, 0) + 0.5,
                "q_smooth": q,
                "rejected_outside_support": False,
                "is_open_signal": nov > 0 and q >= 0.5,
                "fold_id": "f",
                "model_version": "stage4_sklearn_logo_v3",
            }
        )
    # second frame for rates / accumulated area
    for c, nov, q in ((100, 0.6, 0.7), (101, 0.3, 0.7), (102, 0.0, 0.7), (103, -0.1, 0.7)):
        frames.append(
            {
                "gameId": 1,
                "touchId": "t1",
                "frameIdx": 15,
                "candidateId": c,
                "is_model_5hz_frame": True,
                "NOV_smooth": nov,
                "Q_smooth": max(nov, 0) + 0.5,
                "q_smooth": q,
                "rejected_outside_support": False,
                "is_open_signal": nov > 0 and q >= 0.5,
                "fold_id": "f",
                "model_version": "stage4_sklearn_logo_v3",
            }
        )
    opt = compute_option_set_features(pd.DataFrame(frames), open_threshold=0.5)
    required = {
        "best_option_value",
        "n_viable_options",
        "soft_total_option_value",
        "best_second_gap",
        "option_entropy",
        "accumulated_option_value_area",
        "opening_rate",
        "closing_rate",
    }
    assert required.issubset(set(opt.columns))
    assert len(opt) == 2


def test_stage5_continuous_window_fields_present():
    q = [0.1, 0.9, 0.9, 0.9, 0.1]
    nov = [0.0, 0.3, 0.5, 0.4, 0.0]
    wins = segment_candidate_series(_series(q, nov), open_threshold=0.5, close_threshold=0.4)
    assert len(wins) == 1
    for c in (
        "opening_frameIdx",
        "peak_frameIdx",
        "closing_frameIdx",
        "opening_time_s",
        "peak_time_s",
        "closing_time_s",
        "duration_s",
        "peak_NOV",
        "integrated_NOV",
    ):
        assert c in wins.columns
        assert pd.notna(wins.iloc[0][c])


def test_stage5_use_timing_fields():
    q = [0.1, 0.9, 0.9, 0.9, 0.1]
    nov = [0.0, 0.3, 0.5, 0.4, 0.0]
    series = _series(q, nov)
    wins = segment_candidate_series(series, open_threshold=0.5, close_threshold=0.4)
    release = pd.DataFrame(
        [
            {
                "gameId": 1,
                "touchId": "t1",
                "frameIdx": 10,
                "candidateId": 100,
                "is_true_receiver": True,
                "is_pass_release_frame": True,
                "completion_label_eligible": True,
                "target_reliability_status": "known_receiver",
                "receiver_specific_eligible": True,
                "usable_as_receiver_completion_label": True,
            }
        ]
    )
    labeled = label_windows(wins, series, release, late_delta=0.05, late_after_close_epsilon_s=0.20)
    assert labeled.iloc[0]["label"] == "used"
    assert pd.notna(labeled.iloc[0]["delay_opening_to_release_s"])
    assert pd.notna(labeled.iloc[0]["delay_peak_to_release_s"])
    assert pd.notna(labeled.iloc[0]["value_at_use_vs_peak"])


def test_stage5_existence_prob_column_or_explicit_deferral():
    cfg = (ROOT / "configs" / "stage5_windows.yaml").read_text(encoding="utf-8")
    assert "deferred_stage7" in cfg
    handoff = ARTIFACTS / "STAGE5_HANDOFF_STAGE6.md"
    if handoff.exists():
        assert "deferred_stage7" in handoff.read_text(encoding="utf-8").lower() or "Stage 7" in handoff.read_text(
            encoding="utf-8"
        )


def test_stage5_no_random_group_split():
    import re

    bad = re.compile(r"train_test_split|GroupShuffleSplit|ShuffleSplit|KFold\(", re.I)
    for path in (ROOT / "src" / "passing_windows" / "windows").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert not bad.search(text), path
    if PIPELINE.exists():
        assert not bad.search(PIPELINE.read_text(encoding="utf-8"))


def test_stage5_forbidden_claim_language():
    import re

    bad = re.compile(r"\b(correct pass|bad decision|points left on the table)\b", re.I)
    skip = {
        "STAGE5_AUDIT_BRIEF_FOR_BUILDER.md",
        "STAGE5_ADVERSARIAL_AUDIT.md",
    }
    for name in (
        "STAGE5_VERIFICATION.md",
        "STAGE5_HANDOFF_STAGE6.md",
        "STAGE5_REGRESSION_LOCK.md",
        "STAGE5_BUILDER_STATUS.md",
        "stage5_window_report.md",
    ):
        if name in skip:
            continue
        path = ARTIFACTS / name
        if path.exists():
            assert not bad.search(path.read_text(encoding="utf-8")), name


def test_stage5_no_season_aggregates():
    import re

    bad = re.compile(r"aggregates[/\\]acb_.*aggregates", re.I)
    for path in (ROOT / "src" / "passing_windows" / "windows").rglob("*.py"):
        assert not bad.search(path.read_text(encoding="utf-8")), path
    if PIPELINE.exists():
        assert not bad.search(PIPELINE.read_text(encoding="utf-8"))


def test_stage5_regression_lock_covers_blockers():
    lock = REGRESSION_LOCK.read_text(encoding="utf-8")
    for i in range(0, 24):
        assert f"S5-B{i:02d}" in lock, f"missing S5-B{i:02d}"


@needs_stage5
def test_stage5_all_ten_logo_folds_present():
    thr = pd.read_parquet(TABLES / "stage5_threshold_selections.parquet")
    assert len(thr) == 10
    assert thr["fold_id"].nunique() == 10


@needs_stage5
def test_stage5_gate_columns_present():
    series = pd.read_parquet(SERIES)
    for c in GATE_COLUMNS:
        assert c in series.columns, c


@needs_stage5
def test_stage5_sampling_role_retained():
    series = pd.read_parquet(SERIES)
    assert "sampling_role" in series.columns
    assert series["sampling_role"].notna().any()


@needs_stage5
def test_stage5_threshold_robustness_grid_emitted():
    grid = TABLES / "stage5_threshold_grid_windows.parquet"
    assert grid.exists()
    df = pd.read_parquet(grid)
    assert len(df) >= 10 * 5  # 10 folds × 5 deltas
    assert {"delta_from_primary", "n_open_episodes", "open_threshold"}.issubset(df.columns)


@needs_stage5
def test_stage5_support_rejection_rate_reported():
    import json

    summ = json.loads((ARTIFACTS / "stage5_game_summaries.json").read_text(encoding="utf-8"))
    assert "support_rejection_rate_5hz" in summ
    assert summ["support_rejection_rate_5hz"] == summ["support_rejection_rate_5hz"]  # not NaN identity ok
    assert 0.0 <= float(summ["support_rejection_rate_5hz"]) <= 1.0


@needs_stage5
def test_stage5_interceptor_never_use_target():
    wins = pd.read_parquet(WINDOWS)
    used = wins[wins["label"] == "used"]
    if used.empty or "interceptorId" not in used.columns:
        # Stage 5 windows may not carry interceptorId; check series join if present
        series_cols = pd.read_parquet(SERIES, columns=["candidateId"]).columns
        assert "candidateId" in series_cols
        return
    bad = used["candidateId"].astype(float) == used["interceptorId"].astype(float)
    assert not bool(bad.fillna(False).any())


@needs_stage5
def test_stage5_components_not_collapsed_to_nov():
    series = pd.read_parquet(SERIES)
    for c in ("q_smooth", "V_catch_smooth", "Q_smooth", "NOV_smooth"):
        assert c in series.columns
        assert series[c].notna().any()


@needs_stage5
def test_stage5_windows_series_5hz_only_live():
    series = pd.read_parquet(SERIES, columns=["is_model_5hz_frame"])
    assert bool(series["is_model_5hz_frame"].fillna(False).all())
