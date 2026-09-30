#!/usr/bin/env python
"""Stage 5: segment passing windows from Stage 4 cross-fitted scores.

Restartable per held-out game. Thresholds are selected on the other nine freeze
games (even under ``--games``), then applied to the held-out game only.

    python pipelines/05_segment_windows.py
    python pipelines/05_segment_windows.py --force
    python pipelines/05_segment_windows.py --games 114086
    python pipelines/05_segment_windows.py --verify-manifest
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

STAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE_ROOT / "src"))

from passing_windows.ingest import load_freeze  # noqa: E402
from passing_windows.io_utils import (  # noqa: E402
    build_output_manifest,
    verify_output_manifest,
    write_json,
    write_markdown,
    write_table,
)
from passing_windows.narrow_path import GATE_COLUMNS  # noqa: E402
from passing_windows.paths import ARTIFACTS, CONFIGS, PACKAGE_ROOT, TABLES  # noqa: E402
from passing_windows.windows.labels import label_windows, never_open_stubs  # noqa: E402
from passing_windows.windows.leakage import scan_stage5_sources_for_refit  # noqa: E402
from passing_windows.windows.option_set import compute_option_set_features  # noqa: E402
from passing_windows.windows.segment import mark_frame_open_state, segment_candidate_series  # noqa: E402
from passing_windows.windows.smooth import smooth_score_columns  # noqa: E402
from passing_windows.windows.thresholds import (  # noqa: E402
    ThresholdSelection,
    apply_thresholds_to_holdout,
    select_thresholds_logo,
)

STAGE = 5
FREEZE_ID = "passing_windows_stage0_20250924"
EXPECTED_MODEL_VERSION = "stage4_sklearn_logo_v3"
MARKER_NAME = "_DONE_STAGE5"
SERIES_NAME = "stage5_candidate_series.parquet"
WINDOWS_NAME = "stage5_windows.parquet"
OPTION_NAME = "stage5_option_set_frames.parquet"
JOIN_KEYS = ["gameId", "touchId", "frameIdx", "candidateId"]
SMOOTH_ALPHA = 0.5


def _game_dir(game_id: int) -> Path:
    return TABLES / "by_game" / str(game_id)


def load_stage5_config() -> dict[str, Any]:
    path = CONFIGS / "stage5_windows.yaml"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_stage4_predictions(game_ids: list[int] | None = None) -> pd.DataFrame:
    path = TABLES / "stage4_candidate_predictions.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run Stage 4 first")
    df = pd.read_parquet(path)
    if game_ids is not None:
        df = df[df["gameId"].isin(game_ids)].copy()
    return df


def assert_stage4_identity(df: pd.DataFrame) -> None:
    if "model_version" not in df.columns:
        raise ValueError("Stage 4 predictions missing model_version")
    versions = set(df["model_version"].dropna().unique())
    if versions != {EXPECTED_MODEL_VERSION}:
        raise ValueError(f"expected model_version={EXPECTED_MODEL_VERSION}, got {versions}")


def process_fold(
    fold: dict[str, Any],
    all_pred: pd.DataFrame,
    *,
    force: bool = False,
    cfg: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    fold_id = str(fold["fold_id"])
    held_out = int(fold["held_out_gameId"])
    train_ids = [int(g) for g in fold["train_gameIds"]]

    gdir = _game_dir(held_out)
    marker = gdir / MARKER_NAME
    series_path = gdir / SERIES_NAME
    windows_path = gdir / WINDOWS_NAME
    option_path = gdir / OPTION_NAME
    thr_path = gdir / "stage5_fold_thresholds.json"

    if marker.exists() and series_path.exists() and windows_path.exists() and not force:
        series = pd.read_parquet(series_path)
        windows = pd.read_parquet(windows_path)
        option = pd.read_parquet(option_path) if option_path.exists() else pd.DataFrame()
        summary = {
            "gameId": held_out,
            "fold_id": fold_id,
            "cached": True,
            "n_series": int(len(series)),
            "n_windows": int(len(windows)),
        }
        if thr_path.exists():
            summary["thresholds"] = json.loads(thr_path.read_text(encoding="utf-8"))
        return series, windows, option, summary

    t0 = time.time()
    # Thresholds from train games only (full Stage 4 table for those games)
    train_pred = all_pred[all_pred["gameId"].isin(train_ids)]
    sel = select_thresholds_logo(
        train_pred,
        train_game_ids=train_ids,
        held_out_game_id=held_out,
        fold_id=fold_id,
    )

    holdout = all_pred[all_pred["gameId"] == held_out].copy()
    if holdout.empty:
        raise ValueError(f"{fold_id}: no Stage 4 rows for held-out game {held_out}")
    assert_stage4_identity(holdout)
    apply_thresholds_to_holdout(holdout, sel)

    # Release rows (any lattice) for use/late labels
    releases = holdout[
        holdout.get("is_pass_release_frame", pd.Series(False, index=holdout.index)).fillna(False).astype(bool)
    ].copy()

    # Window series: 5 Hz only
    series = holdout[holdout["is_model_5hz_frame"].fillna(False).astype(bool)].copy()
    alpha = float((cfg or {}).get("smoothing", {}).get("alpha", SMOOTH_ALPHA))
    series = smooth_score_columns(series, alpha=alpha)
    series["open_threshold"] = sel.open_threshold
    series["close_threshold"] = sel.close_threshold
    series["late_use_material_loss_delta"] = sel.late_use_material_loss_delta

    windows = segment_candidate_series(
        series,
        open_threshold=sel.open_threshold,
        close_threshold=sel.close_threshold,
    )
    if len(windows):
        windows = label_windows(
            windows,
            series,
            releases,
            late_delta=sel.late_use_material_loss_delta,
            late_after_close_epsilon_s=sel.late_after_close_epsilon_s,
        )
        windows["fold_id"] = fold_id
        windows["model_version"] = EXPECTED_MODEL_VERSION
        windows["late_use_material_loss_delta"] = sel.late_use_material_loss_delta
    stubs = never_open_stubs(series, windows)
    if len(stubs):
        stubs["fold_id"] = fold_id
        stubs["model_version"] = EXPECTED_MODEL_VERSION
        stubs["open_threshold"] = sel.open_threshold
        stubs["close_threshold"] = sel.close_threshold
        stubs["late_use_material_loss_delta"] = sel.late_use_material_loss_delta
        windows = pd.concat([windows, stubs], ignore_index=True) if len(windows) else stubs

    series = mark_frame_open_state(series, windows[windows["label"] != "never_open"] if len(windows) else windows)
    # Carry GATE_COLUMNS
    for col in GATE_COLUMNS:
        if col not in series.columns and col in holdout.columns:
            # merge from holdout on keys
            pass
    missing_gates = [c for c in GATE_COLUMNS if c not in series.columns]
    if missing_gates:
        carry = JOIN_KEYS + [c for c in GATE_COLUMNS if c in holdout.columns]
        series = series.drop(columns=[c for c in GATE_COLUMNS if c in series.columns], errors="ignore")
        series = series.merge(holdout[carry].drop_duplicates(JOIN_KEYS), on=JOIN_KEYS, how="left")

    option = compute_option_set_features(series, open_threshold=sel.open_threshold)

    gdir.mkdir(parents=True, exist_ok=True)
    write_table(series, series_path)
    write_table(windows, windows_path)
    write_table(option, option_path)
    write_json(sel.to_dict(), thr_path)
    marker.write_text(f"stage=5\nhash={EXPECTED_MODEL_VERSION}\nfold={fold_id}\n", encoding="utf-8")

    summary = {
        "gameId": held_out,
        "fold_id": fold_id,
        "cached": False,
        "n_series": int(len(series)),
        "n_windows": int(len(windows)),
        "n_option_frames": int(len(option)),
        "n_rejected_5hz": int(series["rejected_outside_support"].fillna(False).sum())
        if "rejected_outside_support" in series.columns
        else 0,
        "label_counts": windows["label"].value_counts(dropna=False).to_dict() if len(windows) else {},
        "thresholds": sel.to_dict(),
        "elapsed_s": round(time.time() - t0, 3),
    }
    write_json(summary, gdir / "stage5_game_summary.json")
    return series, windows, option, summary


def render_verification(summaries: list[dict[str, Any]], n_series: int, n_windows: int) -> str:
    lines = [
        "# Stage 5 verification",
        "",
        f"**Freeze:** `{FREEZE_ID}`",
        f"**Upstream model:** `{EXPECTED_MODEL_VERSION}`",
        f"**Games processed:** {len(summaries)}",
        f"**Series rows (5 Hz):** {n_series}",
        f"**Window rows (incl. never_open stubs):** {n_windows}",
        "",
        "## Binding checks (MVP)",
        "",
        "- Window series filtered to `is_model_5hz_frame`",
        "- Causal EWMA only (no future frames)",
        "- Open requires q ≥ open_thr AND NOV > 0 AND not rejected",
        "- Persistence ≥ 0.20s (≥2 consecutive 5 Hz samples)",
        "- Hysteresis: close_threshold ≤ open_threshold",
        "- Thresholds selected on train games only (nested LOGO)",
        "- `used` requires known-receiver narrow-path gates",
        "- Existence probability: `deferred_stage7`",
        "",
        "## Per-game summaries",
        "",
        "| gameId | fold_id | n_series | n_windows | open_thr | close_thr |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for s in summaries:
        thr = s.get("thresholds") or {}
        lines.append(
            f"| {s.get('gameId')} | {s.get('fold_id')} | {s.get('n_series', 0)} | "
            f"{s.get('n_windows', 0)} | {thr.get('open_threshold', '')} | {thr.get('close_threshold', '')} |"
        )
    lines.extend(["", "## Status", "", "MVP ship — see `STAGE5_BUILDER_STATUS.md` for remaining audit gaps.", ""])
    return "\n".join(lines)


def render_handoff(n_series: int, n_windows: int) -> str:
    return "\n".join(
        [
            "# Stage 5 → Stage 6 handoff",
            "",
            "**Status:** Stage 5 MVP window objects available.",
            "",
            "## Primary tables Stage 6 must consume",
            "",
            "| Table | Grain | Notes |",
            "|---|---|---|",
            "| `tables/stage5_windows.parquet` | window episode (+ never_open stubs) | labels + continuous fields |",
            "| `tables/stage5_candidate_series.parquet` | 5 Hz candidate-frame | smoothed q/V_catch/Q/NOV |",
            "| `tables/stage5_option_set_frames.parquet` | touch × 5 Hz frame | option-set features |",
            "| `tables/stage5_threshold_selections.parquet` | fold | LOGO open/close/late |",
            "",
            "## Dual sampling reminder",
            "",
            "- Window series: `is_model_5hz_frame` only",
            "- Use/late timing: release frames (forced OK; off-lattice OK)",
            "",
            f"- Series rows this run: **{n_series}**",
            f"- Window rows this run: **{n_windows}**",
            "",
            "## Deferred to Stage 7",
            "",
            "- `window_existence_probability_under_tracking_perturbation` (status=`deferred_stage7`)",
            "- Full geometry Monte Carlo / label instability adjudication",
            "",
            "## Do not",
            "",
            "- Refit Stage 4 component models",
            "- Treat interceptor as intended target",
            "- Use forbidden claim language (correct/bad decision / points left on the table)",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 5: segment passing windows")
    parser.add_argument("--force", action="store_true", help="recompute even if markers exist")
    parser.add_argument("--games", type=int, nargs="*", help="subset of held-out gameIds to process")
    parser.add_argument("--verify-manifest", action="store_true", help="verify stage5_output_manifest.json and exit")
    args = parser.parse_args(argv)

    if args.verify_manifest:
        man = ARTIFACTS / "stage5_output_manifest.json"
        ok, problems = verify_output_manifest(man, root=PACKAGE_ROOT)
        if not ok:
            print("MANIFEST FAIL", problems, file=sys.stderr)
            return 1
        print("MANIFEST OK")
        return 0

    # Guard: no Stage 4 refit in Stage 5 sources
    scan_stage5_sources_for_refit(STAGE_ROOT / "src" / "passing_windows" / "windows")
    scan_stage5_sources_for_refit(Path(__file__))

    freeze = load_freeze()
    folds = freeze["validation"]["outer_folds"]
    cfg = load_stage5_config()

    # Load full Stage 4 table once (needed for train-fold threshold selection)
    all_pred = load_stage4_predictions()
    assert_stage4_identity(all_pred)

    target_folds = folds
    if args.games:
        want = set(int(g) for g in args.games)
        target_folds = [f for f in folds if int(f["held_out_gameId"]) in want]
        if not target_folds:
            print(f"no folds match --games {args.games}", file=sys.stderr)
            return 1

    series_parts: list[pd.DataFrame] = []
    window_parts: list[pd.DataFrame] = []
    option_parts: list[pd.DataFrame] = []
    thr_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for fold in target_folds:
        held = int(fold["held_out_gameId"])
        print(f"[stage5] fold {fold['fold_id']} held_out={held}")
        try:
            series, windows, option, summary = process_fold(
                fold, all_pred, force=args.force, cfg=cfg
            )
            series_parts.append(series)
            window_parts.append(windows)
            if len(option):
                option_parts.append(option)
            if summary.get("thresholds"):
                thr_rows.append(summary["thresholds"])
            summaries.append(summary)
        except Exception as e:  # noqa: BLE001 — per-fold restartable
            import traceback

            traceback.print_exc()
            errors.append({"gameId": held, "error": str(e)})
            print(f"[stage5] ERROR game {held}: {e}", file=sys.stderr)

    if not series_parts:
        print("[stage5] no outputs", file=sys.stderr)
        return 2

    all_series = pd.concat(series_parts, ignore_index=True)
    all_windows = pd.concat(window_parts, ignore_index=True) if window_parts else pd.DataFrame()
    all_option = pd.concat(option_parts, ignore_index=True) if option_parts else pd.DataFrame()
    thr_df = pd.DataFrame(thr_rows)

    # If only a subset of games was run, merge with any existing full tables
    out_series = TABLES / "stage5_candidate_series.parquet"
    out_windows = TABLES / "stage5_windows.parquet"
    out_option = TABLES / "stage5_option_set_frames.parquet"
    out_thr = TABLES / "stage5_threshold_selections.parquet"

    if args.games and out_series.exists() and not args.force:
        prev = pd.read_parquet(out_series)
        prev = prev[~prev["gameId"].isin(all_series["gameId"].unique())]
        all_series = pd.concat([prev, all_series], ignore_index=True)
    if args.games and out_windows.exists() and not args.force:
        prev_w = pd.read_parquet(out_windows)
        prev_w = prev_w[~prev_w["gameId"].isin(all_windows["gameId"].unique())]
        all_windows = pd.concat([prev_w, all_windows], ignore_index=True)
    if args.games and out_option.exists() and len(all_option) and not args.force:
        prev_o = pd.read_parquet(out_option)
        prev_o = prev_o[~prev_o["gameId"].isin(all_option["gameId"].unique())]
        all_option = pd.concat([prev_o, all_option], ignore_index=True)

    write_table(all_series, out_series)
    write_table(all_windows, out_windows)
    write_table(all_option, out_option)
    write_table(thr_df, out_thr)

    fold_thr_json = {
        "procedure": thr_rows[0].get("procedure") if thr_rows else None,
        "folds": thr_rows,
        "robustness_grid_note": "Primary selection per fold; grid deltas in ThresholdSelection.robustness_grid()",
        "existence_prob_status": "deferred_stage7",
    }
    # Attach robustness grids
    for row in fold_thr_json["folds"]:
        sel = ThresholdSelection(
            fold_id=row["fold_id"],
            held_out_game_id=row["held_out_game_id"],
            train_game_ids=row["train_game_ids"],
            open_threshold=row["open_threshold"],
            close_threshold=row["close_threshold"],
            late_use_material_loss_delta=row["late_use_material_loss_delta"],
            late_after_close_epsilon_s=row.get("late_after_close_epsilon_s", 0.20),
            procedure=row.get("procedure", ""),
            diagnostics=row.get("diagnostics") or {},
        )
        row["robustness_grid"] = sel.robustness_grid()
    write_json(fold_thr_json, ARTIFACTS / "stage5_fold_thresholds.json")

    verification = render_verification(summaries, len(all_series), len(all_windows))
    handoff = render_handoff(len(all_series), len(all_windows))
    write_markdown(verification, ARTIFACTS / "STAGE5_VERIFICATION.md")
    write_markdown(handoff, ARTIFACTS / "STAGE5_HANDOFF_STAGE6.md")

    lock_path = ARTIFACTS / "STAGE5_REGRESSION_LOCK.md"
    if not lock_path.exists():
        write_markdown(
            "\n".join(
                [
                    "# Stage 5 regression lock",
                    "",
                    "| Finding ID | Summary | Regression test |",
                    "|---|---|---|",
                    "| S5-B03 | 5 Hz series only | `test_stage5_windows_use_model_5hz_series_only` |",
                    "| S5-B05 | Causal smoothing | `test_stage5_smoothing_is_causal` |",
                    "| S5-B06 | LOGO threshold isolation | `test_stage5_thresholds_logo_holdout_isolation` |",
                    "| S5-B07 | Hysteresis open/close | `test_stage5_hysteresis_separate_open_close` |",
                    "| S5-B08 | Persistence ≥0.20s | `test_stage5_persistence_ge_0_20s_wallclock` |",
                    "| S5-B10 | Rejected cannot open | `test_stage5_rejected_cannot_open_windows` |",
                    "| S5-B11 | Used = known-receiver | `test_stage5_used_requires_known_receiver` |",
                    "| S5-B15 | Open = q ∧ NOV | `test_stage5_open_requires_q_and_nov` |",
                    "",
                ]
            ),
            lock_path,
        )

    manifest_paths = [
        out_series,
        out_windows,
        out_option,
        out_thr,
        ARTIFACTS / "stage5_fold_thresholds.json",
        ARTIFACTS / "STAGE5_VERIFICATION.md",
        ARTIFACTS / "STAGE5_HANDOFF_STAGE6.md",
        ARTIFACTS / "STAGE5_REGRESSION_LOCK.md",
    ]
    for s in summaries:
        gid = int(s["gameId"])
        manifest_paths.append(_game_dir(gid) / SERIES_NAME)
        manifest_paths.append(_game_dir(gid) / WINDOWS_NAME)
        manifest_paths.append(_game_dir(gid) / OPTION_NAME)

    manifest = build_output_manifest(
        manifest_paths,
        root=PACKAGE_ROOT,
        manifest_path=ARTIFACTS / "stage5_output_manifest.json",
    )
    write_json(manifest, ARTIFACTS / "stage5_output_manifest.json")
    write_json({"games": summaries, "errors": errors}, ARTIFACTS / "stage5_game_summaries.json")

    if errors:
        print(f"[stage5] completed with {len(errors)} fold errors", file=sys.stderr)
        return 2
    print(f"[stage5] wrote series={len(all_series)} windows={len(all_windows)} -> {out_windows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
