"""Stage 5 reporting: window summary markdown + threshold robustness grid."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .segment import segment_candidate_series
from .thresholds import ThresholdSelection


def render_window_report(
    windows: pd.DataFrame,
    series: pd.DataFrame,
    summaries: list[dict[str, Any]],
) -> str:
    """Descriptive Stage 5 window report (no causal claim language)."""
    lines = [
        "# Stage 5 window report",
        "",
        "**Upstream:** `stage4_sklearn_logo_v3`",
        "**Mode:** descriptive window objects under nested LOGO thresholds",
        "",
        "## Population",
        "",
        f"- 5 Hz series rows: **{len(series)}**",
        f"- Window rows (incl. never_open stubs): **{len(windows)}**",
        f"- Games: **{windows['gameId'].nunique() if len(windows) else 0}**",
        "",
    ]
    if "rejected_outside_support" in series.columns:
        rej = float(series["rejected_outside_support"].fillna(False).mean())
        lines.extend(
            [
                "## Support rejection (5 Hz series)",
                "",
                f"- Fraction `rejected_outside_support`: **{rej:.4f}**",
                "- Rejected rows cannot open windows (NOV non-viable)",
                "",
            ]
        )
    if len(windows) and "label" in windows.columns:
        counts = windows["label"].value_counts(dropna=False)
        lines.extend(["## Label counts", "", "| Label | N |", "|---|---:|"])
        for lab, n in counts.items():
            lines.append(f"| `{lab}` | {int(n)} |")
        lines.append("")
        openish = windows[windows["label"].isin(["used", "late", "unused", "open"])]
        if len(openish):
            lines.extend(
                [
                    "## Duration / NOV (non–never_open)",
                    "",
                    f"- Median duration_s: **{float(openish['duration_s'].median()):.3f}**",
                    f"- Mean peak_NOV: **{float(openish['peak_NOV'].mean()):.4f}**",
                    f"- Mean integrated_NOV: **{float(openish['integrated_NOV'].mean()):.4f}**",
                    "",
                ]
            )
    lines.extend(
        [
            "## Per-game label mix",
            "",
            "| gameId | used | late | unused | never_open | n_windows |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    if len(windows) and "label" in windows.columns:
        for gid, grp in windows.groupby("gameId", sort=True):
            vc = grp["label"].value_counts()
            lines.append(
                f"| {int(gid)} | {int(vc.get('used', 0))} | {int(vc.get('late', 0))} | "
                f"{int(vc.get('unused', 0))} | {int(vc.get('never_open', 0))} | {len(grp)} |"
            )
    lines.extend(
        [
            "",
            "## Language note",
            "",
            "Windows are **model-available** intervals under fold-calibrated thresholds.",
            "Unused does not mean a mistake; used does not mean optimal.",
            "",
        ]
    )
    return "\n".join(lines)


def emit_threshold_grid_summaries(
    series: pd.DataFrame,
    thr_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    """Re-segment each holdout series under the fold robustness grid (counts only)."""
    rows: list[dict[str, Any]] = []
    if series.empty or not thr_rows:
        return pd.DataFrame(rows)
    for thr in thr_rows:
        sel = ThresholdSelection(
            fold_id=str(thr["fold_id"]),
            held_out_game_id=int(thr["held_out_game_id"]),
            train_game_ids=[int(g) for g in thr["train_game_ids"]],
            open_threshold=float(thr["open_threshold"]),
            close_threshold=float(thr["close_threshold"]),
            late_use_material_loss_delta=float(thr["late_use_material_loss_delta"]),
            late_after_close_epsilon_s=float(thr.get("late_after_close_epsilon_s", 0.20)),
            procedure=str(thr.get("procedure", "")),
            diagnostics=thr.get("diagnostics") or {},
        )
        game = int(sel.held_out_game_id)
        series_g = series[series["gameId"].astype(int) == game]
        if series_g.empty:
            continue
        for gpoint in sel.robustness_grid():
            wins = segment_candidate_series(
                series_g,
                open_threshold=float(gpoint["open_threshold"]),
                close_threshold=float(gpoint["close_threshold"]),
            )
            dur = wins["duration_s"].to_numpy(dtype=float) if len(wins) else np.array([])
            peak = wins["peak_NOV"].to_numpy(dtype=float) if len(wins) and "peak_NOV" in wins.columns else np.array([])
            rows.append(
                {
                    "fold_id": sel.fold_id,
                    "gameId": game,
                    "grid_id": f"delta_{gpoint['delta_from_primary']:+.2f}",
                    "delta_from_primary": float(gpoint["delta_from_primary"]),
                    "open_threshold": float(gpoint["open_threshold"]),
                    "close_threshold": float(gpoint["close_threshold"]),
                    "is_primary": abs(float(gpoint["delta_from_primary"])) < 1e-12,
                    "n_open_episodes": int(len(wins)),
                    "mean_duration_s": float(np.nanmean(dur)) if dur.size else float("nan"),
                    "median_duration_s": float(np.nanmedian(dur)) if dur.size else float("nan"),
                    "mean_peak_NOV": float(np.nanmean(peak)) if peak.size else float("nan"),
                }
            )
    return pd.DataFrame(rows)
