"""Window labels: open / used / late / unused / never_open."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .segment import RENDER_HZ, frame_to_time_s

WINDOW_LABELS = ("open", "used", "late", "unused", "never_open")

KNOWN_RECEIVER = "known_receiver"


def assign_touch_summary_label(episode_labels: list[str]) -> str:
    """Priority: used > late > unused > never_open."""
    s = set(episode_labels)
    if "used" in s:
        return "used"
    if "late" in s:
        return "late"
    if "unused" in s or "open" in s:
        return "unused"
    return "never_open"


def _is_known_receiver_use(row: pd.Series) -> bool:
    if not bool(row.get("is_true_receiver", False)):
        return False
    status = str(row.get("target_reliability_status", "") or "")
    if status != KNOWN_RECEIVER:
        return False
    if "receiver_specific_eligible" in row.index:
        if not bool(row.get("receiver_specific_eligible", False)):
            return False
    if "usable_as_receiver_completion_label" in row.index:
        if not bool(row.get("usable_as_receiver_completion_label", False)):
            return False
    return True


def _release_uses(
    releases: pd.DataFrame,
) -> pd.DataFrame:
    """Filter to known-receiver true-receiver release rows usable for used/late."""
    if releases is None or releases.empty:
        return pd.DataFrame()
    r = releases.copy()
    mask = r.get("is_pass_release_frame", pd.Series(True, index=r.index)).fillna(False).astype(bool)
    r = r[mask]
    if r.empty:
        return r
    ok = r.apply(_is_known_receiver_use, axis=1)
    return r[ok].copy()


def label_windows(
    windows: pd.DataFrame,
    series: pd.DataFrame,
    releases: pd.DataFrame,
    *,
    late_delta: float,
    late_after_close_epsilon_s: float = 0.20,
) -> pd.DataFrame:
    """Assign episode-level labels and continuous use-timing fields.

    Release frames need not lie on the 5 Hz lattice (S5-B04).
    ``used`` requires known-receiver narrow-path gates (S5-B11).
    """
    if windows is None or windows.empty:
        return pd.DataFrame()

    out = windows.copy()
    out["label"] = "unused"
    out["release_frameIdx"] = pd.NA
    out["delay_opening_to_release_s"] = np.nan
    out["delay_peak_to_release_s"] = np.nan
    out["value_at_use_vs_peak"] = np.nan
    out["NOV_at_release"] = np.nan
    out["existence_prob_status"] = "deferred_stage7"
    out["window_existence_probability_under_tracking_perturbation"] = np.nan
    out["censored_at_touch_end"] = False

    uses = _release_uses(releases)

    for i, w in out.iterrows():
        open_t = float(w["opening_time_s"])
        close_t = float(w["closing_time_s"])
        peak_nov = float(w["peak_NOV"]) if np.isfinite(w["peak_NOV"]) else float("nan")
        peak_t = float(w["peak_time_s"])

        cand_uses = uses
        if not uses.empty:
            cand_uses = uses[
                (uses["gameId"] == w["gameId"])
                & (uses["touchId"] == w["touchId"])
                & (uses["candidateId"] == w["candidateId"])
            ]

        label = "unused"
        chosen: dict[str, Any] | None = None

        if not cand_uses.empty:
            for _, rel in cand_uses.iterrows():
                rel_f = int(rel["frameIdx"])
                rel_t = frame_to_time_s(rel_f)
                nov_at = _nov_at_time(series, w, rel_f)

                in_open = open_t <= rel_t <= close_t
                after_close = close_t < rel_t <= close_t + float(late_after_close_epsilon_s)

                material_loss = False
                if in_open and np.isfinite(peak_nov) and np.isfinite(nov_at) and rel_t >= peak_t:
                    material_loss = (peak_nov - nov_at) >= float(late_delta)

                if in_open and not material_loss:
                    label = "used"
                    chosen = {"frameIdx": rel_f, "nov": nov_at, "time_s": rel_t}
                    break
                if (in_open and material_loss) or after_close:
                    label = "late"
                    chosen = {"frameIdx": rel_f, "nov": nov_at, "time_s": rel_t}
                    # Keep scanning in case a true used exists (used wins)
                    continue

        out.at[i, "label"] = label
        if chosen is not None:
            out.at[i, "release_frameIdx"] = int(chosen["frameIdx"])
            out.at[i, "delay_opening_to_release_s"] = float(chosen["time_s"] - open_t)
            out.at[i, "delay_peak_to_release_s"] = float(chosen["time_s"] - peak_t)
            out.at[i, "NOV_at_release"] = chosen["nov"]
            if np.isfinite(peak_nov) and peak_nov != 0 and np.isfinite(chosen["nov"]):
                out.at[i, "value_at_use_vs_peak"] = float(chosen["nov"] / peak_nov)

    return out


def _nov_at_time(series: pd.DataFrame, window_row: pd.Series, frame_idx: int) -> float:
    """Nearest 5 Hz NOV_smooth at or before release frame (causal)."""
    g = series[
        (series["gameId"] == window_row["gameId"])
        & (series["touchId"] == window_row["touchId"])
        & (series["candidateId"] == window_row["candidateId"])
        & (series["frameIdx"] <= frame_idx)
    ]
    if g.empty:
        return float("nan")
    row = g.sort_values("frameIdx").iloc[-1]
    return float(row["NOV_smooth"]) if np.isfinite(row.get("NOV_smooth", np.nan)) else float("nan")


def never_open_stubs(
    series: pd.DataFrame,
    windows: pd.DataFrame,
) -> pd.DataFrame:
    """Emit never_open rows for candidate-touches with no qualifying window."""
    keys = ["gameId", "touchId", "candidateId"]
    if series.empty:
        return pd.DataFrame()
    all_keys = series[keys].drop_duplicates()
    if windows is not None and not windows.empty:
        opened = windows[keys].drop_duplicates()
        merged = all_keys.merge(opened.assign(_opened=True), on=keys, how="left")
        missing = merged[merged["_opened"].isna()][keys]
    else:
        missing = all_keys
    rows = []
    for r in missing.itertuples(index=False):
        sub = series[
            (series["gameId"] == r.gameId)
            & (series["touchId"] == r.touchId)
            & (series["candidateId"] == r.candidateId)
        ]
        meta = {
            "gameId": r.gameId,
            "touchId": r.touchId,
            "candidateId": r.candidateId,
            "label": "never_open",
            "opening_frameIdx": pd.NA,
            "peak_frameIdx": pd.NA,
            "closing_frameIdx": pd.NA,
            "opening_time_s": np.nan,
            "peak_time_s": np.nan,
            "closing_time_s": np.nan,
            "duration_s": np.nan,
            "peak_NOV": np.nan,
            "integrated_NOV": np.nan,
            "censored_at_touch_end": False,
            "existence_prob_status": "deferred_stage7",
            "window_existence_probability_under_tracking_perturbation": np.nan,
            "delay_opening_to_release_s": np.nan,
            "delay_peak_to_release_s": np.nan,
            "value_at_use_vs_peak": np.nan,
            "NOV_at_release": np.nan,
            "release_frameIdx": pd.NA,
        }
        for c in ("fold_id", "model_version", "sampling_role", "open_threshold", "close_threshold"):
            if c in sub.columns:
                meta[c] = sub.iloc[0][c]
            elif windows is not None and not windows.empty and c in windows.columns:
                meta[c] = windows.iloc[0][c]
        rows.append(meta)
    return pd.DataFrame(rows)
