"""Coordinate conventions, attacking-direction normalization, mirroring."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

# Event + tracking convention (SkillCorner): offensive hoop at negative x.
# Analysis normalization: flip so offense always attacks toward +x.
EVENT_ATTACK_SIGN = -1.0  # hoop at x = EVENT_ATTACK_SIGN * |rim|
COURT_HALF_X = 47.0  # approximate half-court length in feet (FIBA ~45.9; buffer OK)
FRONTCOURT_X_MAX = 0.0  # under event convention, frontcourt offense has x <= 0


def distance_xy(a: Iterable[float], b: Iterable[float]) -> float:
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    return float(np.hypot(ax - bx, ay - by))


def mirror_xy(x: float, y: float) -> tuple[float, float]:
    """Left/right mirror when attacking along x: flip y only."""
    return float(x), float(-y)


def broadcast_to_event_xy(x: float, y: float, left_hoop: bool) -> tuple[float, float]:
    """Map broadcast tracking coords into the event attacking-hoop frame.

    Events always place the offensive hoop at negative x. Tracking is stored in
    broadcast orientation. Empirically:
    - leftHoop=True  → already aligned with events
    - leftHoop=False → rotate 180°: (x, y) -> (-x, -y)
    """
    if left_hoop:
        return float(x), float(y)
    return float(-x), float(-y)


def normalize_attack_xy(x: float, y: float, attack_sign: float = EVENT_ATTACK_SIGN) -> tuple[float, float]:
    """Map event-aligned coords so offense attacks +x.

    Input should already be in the event frame (offensive hoop at negative x).
    Normalized: x_norm = attack_sign * x  => near-hoop x=-40 becomes +40.
    """
    return float(attack_sign * x), float(y)


def is_frontcourt_x(x: float, attack_sign: float = EVENT_ATTACK_SIGN, eps: float = 0.5) -> bool:
    """Frontcourt under event convention: on the offensive (negative-x) half."""
    return float(x) * float(attack_sign) >= -eps  # x and attack_sign same sign, or near mid


def verify_possession_hoop_orientation(
    shot_xs: list[float],
    left_hoop: bool | None,
    attack_sign: float = EVENT_ATTACK_SIGN,
) -> dict:
    """Check shot locations cluster on the attacking side (negative x in events).

    Returns diagnostics; does not use leftHoop to flip (events already normalized).
    leftHoop is recorded for audit only.
    """
    xs = np.asarray(shot_xs, dtype=float)
    n = int(xs.size)
    if n == 0:
        return {
            "n_shots": 0,
            "frac_attacking_side": np.nan,
            "median_x": np.nan,
            "orientation_ok": True,  # no evidence against
            "leftHoop": left_hoop,
        }
    # Attacking side under event convention: x * attack_sign > 0 roughly? 
    # attack_sign=-1, attacking side x < 0 => x * attack_sign > 0
    on_side = (xs * attack_sign) > 0
    frac = float(np.mean(on_side))
    median_x = float(np.median(xs))
    return {
        "n_shots": n,
        "frac_attacking_side": frac,
        "median_x": median_x,
        "orientation_ok": frac >= 0.7 or n < 3,
        "leftHoop": left_hoop,
    }


def apply_mirror_columns(df: pd.DataFrame, x_col: str, y_col: str, suffix: str = "_mirrored") -> pd.DataFrame:
    out = df.copy()
    mirrored = [mirror_xy(x, y) for x, y in zip(out[x_col], out[y_col], strict=True)]
    out[f"{x_col}{suffix}"] = [m[0] for m in mirrored]
    out[f"{y_col}{suffix}"] = [m[1] for m in mirrored]
    return out


def symmetry_feature_check(
    feature_raw: float,
    feature_mirrored: float,
    expect_flip: bool,
    tol: float = 1e-9,
) -> bool:
    """Geometry symmetry: signed features flip under mirror; unsigned stay equal."""
    if expect_flip:
        return abs(feature_raw + feature_mirrored) <= tol
    return abs(feature_raw - feature_mirrored) <= tol
