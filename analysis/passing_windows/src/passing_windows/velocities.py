"""Causal velocity estimation (current + past frames only)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

FRAME_RATE_HZ = 25.0
DT = 1.0 / FRAME_RATE_HZ

# A backward difference over more than one second is an average over a long
# interval, not an instantaneous velocity, so derivatives reset across bigger
# gaps in the retained-frame timeline.
MAX_CAUSAL_FRAME_GAP = 25

# Elite sprint speed is ~32 ft/s; anything above this cap is a tracking or
# coordinate artifact rather than player motion.
MAX_PLAUSIBLE_PLAYER_SPEED_FPS = 40.0


def causal_velocity_from_positions(
    x: np.ndarray,
    y: np.ndarray,
    frame_idx: np.ndarray | None = None,
    segment_id: Sequence | np.ndarray | None = None,
    max_frame_gap: int | float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backward difference velocity: v[t] = (p[t] - p[t-1]) / dt.

    First sample has NaN velocity. Never uses future frames.
    If frame_idx provided, dt scales by frame gaps (handles stride).
    `segment_id` resets the derivative whenever the segment label changes, so
    positions from different coordinate segments are never differenced.
    `max_frame_gap` resets the derivative across gaps wider than the limit.
    Returns vx, vy, speed.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    vx = np.full(n, np.nan)
    vy = np.full(n, np.nan)
    if n < 2:
        return vx, vy, np.full(n, np.nan)

    if frame_idx is None:
        gaps = np.ones(n - 1)
        dt = np.full(n - 1, DT)
    else:
        fi = np.asarray(frame_idx, dtype=float)
        gaps = np.diff(fi)
        dt = gaps / FRAME_RATE_HZ
        dt = np.where(dt > 0, dt, np.nan)

    usable = np.ones(n - 1, dtype=bool)
    if max_frame_gap is not None:
        usable &= gaps <= float(max_frame_gap)
    if segment_id is not None:
        seg = np.asarray(segment_id, dtype=object)
        usable &= seg[1:] == seg[:-1]

    dt = np.where(usable, dt, np.nan)
    vx[1:] = np.diff(x) / dt
    vy[1:] = np.diff(y) / dt
    speed = np.hypot(vx, vy)
    return vx, vy, speed


def flag_implausible_speed(
    speed: np.ndarray,
    max_speed_fps: float = MAX_PLAUSIBLE_PLAYER_SPEED_FPS,
) -> np.ndarray:
    """True where a finite speed exceeds the physical plausibility cap."""
    s = np.asarray(speed, dtype=float)
    return np.isfinite(s) & (s > float(max_speed_fps))


def orientation_segment_ids(orientation: Iterable) -> np.ndarray:
    """Segment index incrementing whenever attacking orientation changes.

    Missing orientation forms its own segment so unknown-direction frames are
    never differenced against known-direction frames.
    """
    keys = [
        "na" if (v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA) else str(bool(v))
        for v in orientation
    ]
    s = pd.Series(keys, dtype="object")
    return (s != s.shift()).cumsum().to_numpy()


def orientation_sign(orientation: Iterable) -> np.ndarray:
    """Multiplier mapping broadcast-frame vectors into the event frame.

    `broadcast_to_event_xy` is the identity when leftHoop is True and the 180°
    rotation (x, y) -> (-x, -y) otherwise. Both are linear, so a velocity
    rotates by the same scalar.
    """
    out = []
    for v in orientation:
        if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA:
            out.append(np.nan)
        else:
            out.append(1.0 if bool(v) else -1.0)
    return np.asarray(out, dtype=float)


def causal_smooth_speed(speed: np.ndarray, window: int = 3) -> np.ndarray:
    """Causal moving average of speed (past + current only)."""
    if window < 1:
        raise ValueError("window must be >= 1")
    s = np.asarray(speed, dtype=float)
    out = np.full_like(s, np.nan)
    for i in range(len(s)):
        lo = max(0, i - window + 1)
        chunk = s[lo : i + 1]
        if np.all(np.isnan(chunk)):
            continue
        out[i] = float(np.nanmean(chunk))
    return out


def assert_no_future_leakage(
    feature_times: np.ndarray,
    prediction_time: float,
) -> None:
    """Fail loudly if any feature time is after the prediction time."""
    feature_times = np.asarray(feature_times, dtype=float)
    if np.any(feature_times > prediction_time):
        bad = feature_times[feature_times > prediction_time]
        raise AssertionError(
            f"Causal leakage: {len(bad)} feature times after prediction_time={prediction_time}"
        )


def velocities_for_player_track(
    df: pd.DataFrame,
    *,
    x_col: str = "x",
    y_col: str = "y",
    orientation_col: str | None = "leftHoop",
    max_frame_gap: int | float | None = MAX_CAUSAL_FRAME_GAP,
    max_speed_fps: float = MAX_PLAUSIBLE_PLAYER_SPEED_FPS,
    smooth_window: int = 3,
) -> pd.DataFrame:
    """Add causal vx/vy/speed columns to a single-player frame-sorted track.

    Derivatives are taken on the continuous broadcast coordinates and only then
    rotated into the event frame. Differencing event-frame coordinates directly
    is wrong: they flip sign whenever the attacking hoop changes, which turns a
    possession change into an apparent teleport.
    """
    out = df.sort_values("frameIdx").reset_index(drop=True)

    if orientation_col is not None and orientation_col in out.columns:
        orientation = out[orientation_col].tolist()
        segments = orientation_segment_ids(orientation)
        sign = orientation_sign(orientation)
    else:
        segments = np.zeros(len(out), dtype=int)
        sign = np.ones(len(out), dtype=float)

    frames = out["frameIdx"].to_numpy()
    vx, vy, speed = causal_velocity_from_positions(
        out[x_col].to_numpy(),
        out[y_col].to_numpy(),
        frames,
        segment_id=segments,
        max_frame_gap=max_frame_gap,
    )
    vx = vx * sign
    vy = vy * sign

    implausible = flag_implausible_speed(speed, max_speed_fps)
    vx = np.where(implausible, np.nan, vx)
    vy = np.where(implausible, np.nan, vy)
    speed = np.where(implausible, np.nan, speed)

    gap = np.full(len(out), np.nan)
    if len(out) > 1:
        gap[1:] = np.diff(frames.astype(float))

    out["vx"] = vx
    out["vy"] = vy
    out["speed_causal"] = speed
    out["speed_smooth"] = causal_smooth_speed(speed, window=smooth_window)
    out["velocity_segment"] = segments
    out["velocity_frame_gap"] = gap
    out["speed_implausible"] = implausible
    out["velocity_ok"] = np.isfinite(speed)
    return out


def add_causal_velocities(
    players: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("gameId", "playerId"),
    x_col: str = "x",
    y_col: str = "y",
    orientation_col: str | None = "leftHoop",
    max_frame_gap: int | float | None = MAX_CAUSAL_FRAME_GAP,
    max_speed_fps: float = MAX_PLAUSIBLE_PLAYER_SPEED_FPS,
    smooth_window: int = 3,
) -> pd.DataFrame:
    """Per-player causal velocities with orientation/gap resets and speed caps."""
    if players.empty:
        return players.copy()
    group_cols = [c for c in group_cols if c in players.columns]
    parts = []
    for _, g in players.groupby(list(group_cols), sort=False):
        parts.append(
            velocities_for_player_track(
                g,
                x_col=x_col,
                y_col=y_col,
                orientation_col=orientation_col,
                max_frame_gap=max_frame_gap,
                max_speed_fps=max_speed_fps,
                smooth_window=smooth_window,
            )
        )
    return pd.concat(parts, ignore_index=True)
