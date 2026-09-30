"""Causal outcome construction for Stage 4 value models.

Primary target: offensive points in the next 3 seconds of the same chance,
reconstructed from ``shots`` + ``free_throws`` only — never ``chances.ptsScored``.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from passing_windows.velocities import FRAME_RATE_HZ

HORIZON_S = 3.0
HORIZON_FRAMES = int(round(HORIZON_S * FRAME_RATE_HZ))  # 75


def shot_points(row: pd.Series) -> float:
    """Points from a shot event if made; 0 otherwise."""
    if not bool(row.get("outcome", False)):
        return 0.0
    return 3.0 if bool(row.get("three", False)) else 2.0


def free_throw_points(row: pd.Series) -> float:
    return 1.0 if bool(row.get("outcome", False)) else 0.0


def build_scoring_events(
    shots: pd.DataFrame,
    free_throws: pd.DataFrame,
) -> pd.DataFrame:
    """Flatten shots + free throws into (gameId, chanceId, frame, points) rows."""
    parts: list[pd.DataFrame] = []
    if shots is not None and not shots.empty:
        s = shots.copy()
        frame = s["endFrame"].where(s["endFrame"].notna(), s.get("startFrame"))
        contested = s["contested"] if "contested" in s.columns else pd.Series(False, index=s.index)
        contest = (
            s["contestLevel"].astype(str).str.lower()
            if "contestLevel" in s.columns
            else pd.Series("", index=s.index)
        )
        contested_bool = pd.to_numeric(contested, errors="coerce").fillna(0).astype(bool)
        openish = contest.isin({"", "none", "light", "open", "nan"}) | (~contested_bool)
        made = s["outcome"].fillna(False).astype(bool)
        three = s["three"].fillna(False).astype(bool) if "three" in s.columns else False
        pts = np.where(made, np.where(three, 3.0, 2.0), 0.0)
        parts.append(
            pd.DataFrame(
                {
                    "gameId": s["gameId"].astype(int),
                    "chanceId": s["chanceId"],
                    "event_frame": pd.to_numeric(frame, errors="coerce"),
                    "points": pts,
                    "event_family": "shot",
                    "is_shot": True,
                    "is_open_or_light": openish.to_numpy(),
                    "in_paint_touch_proxy": (
                        s["createdFromPaint"].fillna(False).astype(bool)
                        if "createdFromPaint" in s.columns
                        else False
                    ),
                }
            )
        )
    if free_throws is not None and not free_throws.empty:
        ft = free_throws.copy()
        frame = ft["frame"] if "frame" in ft.columns else ft.get("startFrame")
        made = ft["outcome"].fillna(False).astype(bool)
        parts.append(
            pd.DataFrame(
                {
                    "gameId": ft["gameId"].astype(int),
                    "chanceId": ft["chanceId"],
                    "event_frame": pd.to_numeric(frame, errors="coerce"),
                    "points": np.where(made, 1.0, 0.0),
                    "event_family": "free_throw",
                    "is_shot": False,
                    "is_open_or_light": False,
                    "in_paint_touch_proxy": False,
                }
            )
        )
    if not parts:
        return pd.DataFrame(
            columns=[
                "gameId",
                "chanceId",
                "event_frame",
                "points",
                "event_family",
                "is_shot",
                "is_open_or_light",
                "in_paint_touch_proxy",
            ]
        )
    out = pd.concat(parts, ignore_index=True)
    out = out.dropna(subset=["event_frame"])
    out["event_frame"] = out["event_frame"].astype(np.int64)
    return out


def _index_events(events: pd.DataFrame) -> dict[tuple, np.ndarray]:
    """Map (gameId, chanceId) -> sorted array [frame, points, is_shot, open]."""
    out: dict[tuple, np.ndarray] = {}
    if events.empty:
        return out
    for key, grp in events.groupby(["gameId", "chanceId"], sort=False):
        g = grp.sort_values("event_frame")
        out[(int(key[0]), key[1])] = np.column_stack(
            [
                g["event_frame"].to_numpy(dtype=np.int64),
                g["points"].to_numpy(dtype=np.float64),
                g["is_shot"].to_numpy(dtype=np.float64),
                g["is_open_or_light"].to_numpy(dtype=np.float64),
            ]
        )
    return out


def points_in_window(
    events_arr: np.ndarray | None,
    start_frame: int,
    *,
    horizon_frames: int = HORIZON_FRAMES,
) -> dict[str, float]:
    """Sum points / flags in [start_frame, start_frame + horizon] inclusive."""
    empty = {
        "points_next_3s": 0.0,
        "any_shot_next_3s": 0.0,
        "open_shot_next_3s": 0.0,
    }
    if events_arr is None or len(events_arr) == 0:
        return empty
    end = start_frame + int(horizon_frames)
    frames = events_arr[:, 0]
    lo = np.searchsorted(frames, start_frame, side="left")
    hi = np.searchsorted(frames, end, side="right")
    if hi <= lo:
        return empty
    sub = events_arr[lo:hi]
    return {
        "points_next_3s": float(sub[:, 1].sum()),
        "any_shot_next_3s": float(sub[:, 2].max()),
        "open_shot_next_3s": float(((sub[:, 2] > 0) & (sub[:, 3] > 0)).any()),
    }


def _window_stats_vectorized(
    events_arr: np.ndarray,
    start_frames: np.ndarray,
    *,
    horizon_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized window sums for many start frames against one chance's events."""
    n = len(start_frames)
    points = np.zeros(n, dtype=np.float64)
    any_shot = np.zeros(n, dtype=np.float64)
    open_shot = np.zeros(n, dtype=np.float64)
    if events_arr is None or len(events_arr) == 0 or n == 0:
        return points, any_shot, open_shot
    frames = events_arr[:, 0]
    ends = start_frames + int(horizon_frames)
    # For each query, sum points with searchsorted bounds.
    lo = np.searchsorted(frames, start_frames, side="left")
    hi = np.searchsorted(frames, ends, side="right")
    # Cumsums for O(1) range sums.
    pts_c = np.concatenate([[0.0], np.cumsum(events_arr[:, 1])])
    shot_c = np.concatenate([[0.0], np.cumsum(events_arr[:, 2])])
    open_c = np.concatenate([[0.0], np.cumsum((events_arr[:, 2] > 0) & (events_arr[:, 3] > 0))])
    points = pts_c[hi] - pts_c[lo]
    any_shot = ((shot_c[hi] - shot_c[lo]) > 0).astype(np.float64)
    open_shot = ((open_c[hi] - open_c[lo]) > 0).astype(np.float64)
    return points, any_shot, open_shot


def attach_next3s_outcomes(
    candidates: pd.DataFrame,
    shots: pd.DataFrame,
    free_throws: pd.DataFrame,
    *,
    frame_col: str = "frameIdx",
    flight_time_col: str = "flight_time_s",
    use_projected_catch: bool = True,
    horizon_frames: int = HORIZON_FRAMES,
    events_index: dict[tuple, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Attach next-3s offensive points at decision frame or projected catch.

    When ``use_projected_catch`` is True (V_catch), the window starts at
    ``frameIdx + round(flight_time_s * 25)``. When False (V_keep), it starts at
    ``frameIdx``.
    """
    if events_index is None:
        events = build_scoring_events(shots, free_throws)
        indexed = _index_events(events)
    else:
        indexed = events_index

    start_frames = candidates[frame_col].to_numpy(dtype=np.int64)
    if use_projected_catch:
        ft = pd.to_numeric(candidates.get(flight_time_col), errors="coerce").fillna(0.0).to_numpy()
        start_frames = start_frames + np.rint(ft * FRAME_RATE_HZ).astype(np.int64)

    games = candidates["gameId"].to_numpy()
    chances = candidates["chanceId"].to_numpy()
    n = len(candidates)
    points = np.zeros(n, dtype=np.float64)
    any_shot = np.zeros(n, dtype=np.float64)
    open_shot = np.zeros(n, dtype=np.float64)

    order = pd.DataFrame({"gameId": games, "chanceId": chances}).groupby(
        ["gameId", "chanceId"], sort=False
    ).indices
    for key, idxs in order.items():
        arr = indexed.get((int(key[0]), key[1]))
        idxs = np.asarray(idxs, dtype=np.int64)
        p, a, o = _window_stats_vectorized(
            arr if arr is not None else np.zeros((0, 4)),
            start_frames[idxs],
            horizon_frames=horizon_frames,
        )
        points[idxs] = p
        any_shot[idxs] = a
        open_shot[idxs] = o

    out = candidates.copy()
    prefix = "catch_" if use_projected_catch else "keep_"
    out[f"{prefix}points_next_3s"] = points
    out[f"{prefix}any_shot_next_3s"] = any_shot
    out[f"{prefix}open_shot_next_3s"] = open_shot
    out[f"{prefix}outcome_start_frame"] = start_frames
    return out


def precompute_outcome_index(
    shots: pd.DataFrame,
    free_throws: pd.DataFrame,
) -> dict[tuple, np.ndarray]:
    return _index_events(build_scoring_events(shots, free_throws))


def assert_no_pts_scored_source(columns: Iterable[str]) -> None:
    lowered = {str(c).lower() for c in columns}
    if "ptsscored" in lowered or "pts_scored" in lowered:
        raise ValueError("chances.ptsScored (or ptsScored) must not be used as an outcome source")
