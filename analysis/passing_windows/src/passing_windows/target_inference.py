"""Stage 2: intended-target inference for failed passes, with explicit uncertainty.

The raw feed gives no intended receiver for a failed pass: `receiverId`,
`receiverLoc`, `distance`, and `receiverRegion` are all null, and `toReceiverId`
is the **interceptor**, never the target. Candidates are therefore scored from
release-time state plus the observed ball flight:

1. four teammates at release (the lineup containing the passer, minus the passer);
2. each candidate's release position projected forward along its causal velocity
   for the empirically supported flight time of a pass of that length;
3. the observed ball direction over the clean part of the flight;
4. flight-time compatibility between the ball's travel and each candidate.

Every channel produces a residual, residuals become a likelihood, and the
likelihood is normalized across the four candidates into a probability
distribution. Hard labels are a downstream decision, not the output.

All numeric decision thresholds are selected inside training folds
(leave-one-game-out, frozen in `stage0_freeze.yaml`) and are **provisional**
until Stage 4 supplies audit labels: the freeze fixes the selection procedure
(`failed_pass_auto_accept_geometric_margin: TBD_TRAINING_FOLD`), and Stage 2
substitutes held-out completed passes for the not-yet-existing human audit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .velocities import FRAME_RATE_HZ

FREEZE_ID = "passing_windows_stage0_20250924"

# --- Physical / court constants (engineering limits, not fold-calibrated) ----
MAX_PROJECTION_SPEED_FPS = 24.0  # Stage 1 max eligible candidate speed = 23.73
MAX_PROJECTION_HORIZON_S = 1.5  # momentum assumption breaks down beyond this
COURT_HALF_LENGTH_FT = 47.0
COURT_HALF_WIDTH_FT = 25.0

# --- Ball-flight extraction limits ------------------------------------------
MIN_BALL_FRAMES = 3  # frames with ball xy inside the flight window
MIN_FLIGHT_DISPLACEMENT_FT = 2.0  # below this the direction is noise
MIN_BALL_SPEED_FPS = 5.0  # below this the flight-time channel is unusable
CLEAN_FLIGHT_MAX_TURN_DEG = 45.0  # deflection / bounce ends the clean segment

# --- Likelihood shape and scale floors --------------------------------------
# Residuals of the *true* receiver are sharply peaked with a long tail (a pass
# tipped at release, a receiver who cut after the ball left). A Gaussian fitted
# to the peak then treats the tail as impossible and returns p_top = 1.0 on
# passes it gets wrong, so the likelihood is Student-t.
STUDENT_T_DOF = 3.0
# An intercepted ball stops short, so the intended target is usually *beyond*
# the interception point. Overshoot (the ball had already flown past the
# candidate) is penalised at full width; undershoot at this multiple of it.
FLIGHT_UNDERSHOOT_SCALE = 3.0
MIN_SIGMA_BALL_FT = 0.25
MIN_SIGMA_POS_FT = 0.5
MIN_SIGMA_FLIGHT_S = 0.03
BEHIND_PASSER_LOGLIK_PENALTY = -6.0  # candidate behind the release direction
PROB_FLOOR = 1e-12

# --- Decision-rule grid (values provisional; procedure frozen) --------------
AUTO_ACCEPT_TARGET_PRECISION = 0.95
P_MIN_GRID: tuple[float, ...] = (0.50, 0.60, 0.70, 0.80, 0.90)
PROB_MARGIN_GRID: tuple[float, ...] = (0.10, 0.20, 0.30, 0.40, 0.50)
# The geometric margin starts above zero on purpose. A margin is required by the
# Stage 2 design ("a fixed geometric margin over the runner-up"), so the fold
# chooses how wide it is, not whether it exists; including 0.0 let coverage
# maximization silently delete the constraint.
GEOM_MARGIN_FT_GRID: tuple[float, ...] = (1.0, 2.0, 4.0, 6.0)

FLIGHT_MODE_SYMMETRIC = "symmetric"
FLIGHT_MODE_ONE_SIDED = "one_sided_overshoot"

DECISION_AUTO_ACCEPT = "auto_accept"
DECISION_AMBIGUOUS = "ambiguous"
DECISION_UNRESOLVED = "unresolved"


# ---------------------------------------------------------------------------
# Candidate sets
# ---------------------------------------------------------------------------


def _clean_id_list(values) -> list[int]:
    if values is None:
        return []
    if isinstance(values, float) and np.isnan(values):
        return []
    out: list[int] = []
    for v in values:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        iv = int(v)
        if iv not in out:
            out.append(iv)
    return out


def candidate_set_for_pass(
    off_player_ids,
    def_player_ids,
    passer_id,
) -> tuple[str, list[int]]:
    """Teammates of the passer at release: the lineup holding the passer, minus them.

    Returns ``(side, candidates)`` where side is ``"off"``, ``"def"``, or
    ``"none"``. The defensive branch is not a fallback for missing data: seven
    failed passes in the ten games are thrown by a player credited inside the
    opponent's chance (a deflection or steal), and their teammates are that
    chance's defensive lineup. A candidate set is only usable when it names
    exactly four teammates.

    The interceptor (`toReceiverId`) plays for the other team, so it can never
    enter a candidate set built this way.
    """
    if passer_id is None or (isinstance(passer_id, float) and np.isnan(passer_id)):
        return "none", []
    pid = int(passer_id)
    off = _clean_id_list(off_player_ids)
    if pid in off:
        return "off", [p for p in off if p != pid]
    dfn = _clean_id_list(def_player_ids)
    if pid in dfn:
        return "def", [p for p in dfn if p != pid]
    return "none", []


# ---------------------------------------------------------------------------
# Ball flight
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BallFlight:
    """Release-anchored description of the observed ball path."""

    available: bool
    reason: str
    release_frame: int | None = None
    release_x: float = np.nan
    release_y: float = np.nan
    dir_x: float = np.nan
    dir_y: float = np.nan
    speed_fps: float = np.nan
    clean_end_frame: int | None = None
    clean_displacement_ft: float = np.nan
    max_turn_deg: float = np.nan
    t_clean_s: float = np.nan
    t_obs_s: float = np.nan
    n_ball_frames: int = 0
    n_clean_frames: int = 0
    deflected: bool = False

    @property
    def speed_usable(self) -> bool:
        return bool(self.available and np.isfinite(self.speed_fps) and self.speed_fps >= MIN_BALL_SPEED_FPS)


def _angle_between_deg(ax: float, ay: float, bx: float, by: float) -> float:
    na = float(np.hypot(ax, ay))
    nb = float(np.hypot(bx, by))
    if na <= 0 or nb <= 0:
        return float("nan")
    cos = float(np.clip((ax * bx + ay * by) / (na * nb), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def extract_ball_flight(
    frames: np.ndarray,
    ball_x: np.ndarray,
    ball_y: np.ndarray,
    start_frame: int,
    end_frame: int,
) -> BallFlight:
    """Build the clean-flight segment of a pass from retained tracking frames.

    ``frames``/``ball_x``/``ball_y`` must cover the pass window and be sorted.
    The clean segment ends at the first frame whose bearing from the release
    point turns more than ``CLEAN_FLIGHT_MAX_TURN_DEG`` away from the initial
    bearing, or where the ball starts travelling back toward the release point.
    Everything after that is a deflection or a bounce and carries no
    information about where the pass was aimed.
    """
    frames = np.asarray(frames)
    bx = np.asarray(ball_x, dtype=float)
    by = np.asarray(ball_y, dtype=float)
    keep = np.isfinite(bx) & np.isfinite(by) & (frames >= start_frame) & (frames <= end_frame)
    frames, bx, by = frames[keep], bx[keep], by[keep]
    t_obs = float((int(end_frame) - int(start_frame)) / FRAME_RATE_HZ)

    if frames.size < MIN_BALL_FRAMES:
        return BallFlight(False, "too_few_ball_frames", n_ball_frames=int(frames.size), t_obs_s=t_obs)

    order = np.argsort(frames, kind="mergesort")
    frames, bx, by = frames[order], bx[order], by[order]
    f0, x0, y0 = int(frames[0]), float(bx[0]), float(by[0])

    dx = bx - x0
    dy = by - y0
    disp = np.hypot(dx, dy)
    moved = np.flatnonzero(disp >= MIN_FLIGHT_DISPLACEMENT_FT)
    if moved.size == 0:
        return BallFlight(
            False,
            "ball_did_not_travel",
            release_frame=f0,
            release_x=x0,
            release_y=y0,
            n_ball_frames=int(frames.size),
            t_obs_s=t_obs,
        )

    ref = int(moved[0])
    ref_x, ref_y = float(dx[ref]), float(dy[ref])
    last = ref
    max_turn = 0.0
    deflected = False
    for k in range(ref + 1, frames.size):
        turn = _angle_between_deg(ref_x, ref_y, float(dx[k]), float(dy[k]))
        if not np.isfinite(turn):
            break
        if turn > CLEAN_FLIGHT_MAX_TURN_DEG or disp[k] < disp[last]:
            deflected = True
            break
        max_turn = max(max_turn, turn)
        last = k

    end_x, end_y = float(dx[last]), float(dy[last])
    travel = float(np.hypot(end_x, end_y))
    dt = float((int(frames[last]) - f0) / FRAME_RATE_HZ)
    if travel < MIN_FLIGHT_DISPLACEMENT_FT or dt <= 0:
        return BallFlight(
            False,
            "degenerate_clean_segment",
            release_frame=f0,
            release_x=x0,
            release_y=y0,
            n_ball_frames=int(frames.size),
            t_obs_s=t_obs,
        )

    return BallFlight(
        available=True,
        reason="ok",
        release_frame=f0,
        release_x=x0,
        release_y=y0,
        dir_x=end_x / travel,
        dir_y=end_y / travel,
        speed_fps=travel / dt,
        clean_end_frame=int(frames[last]),
        clean_displacement_ft=travel,
        max_turn_deg=float(max_turn),
        t_clean_s=dt,
        t_obs_s=t_obs,
        n_ball_frames=int(frames.size),
        n_clean_frames=int(last + 1),
        deflected=deflected,
    )


# ---------------------------------------------------------------------------
# Empirical flight-time support
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FlightTimeModel:
    """Robust monotonic flight time as a function of pass distance.

    Binned medians with an isotonic (cumulative-max) pass, linear interpolation
    inside the support, and a clamped linear extrapolation outside it. Fitted on
    completed passes from training games only.
    """

    edges: np.ndarray
    centers: np.ndarray
    times: np.ndarray
    n_per_bin: np.ndarray
    distance_support_max: float
    n_train: int

    def predict(self, distance_ft) -> np.ndarray:
        d = np.asarray(distance_ft, dtype=float)
        if self.centers.size == 0:
            return np.full(d.shape, np.nan)
        if self.centers.size == 1:
            return np.where(np.isfinite(d), float(self.times[0]), np.nan)
        out = np.interp(d, self.centers, self.times)
        # Clamped linear extrapolation beyond the outer bin centers.
        slope_hi = (self.times[-1] - self.times[-2]) / max(self.centers[-1] - self.centers[-2], 1e-6)
        hi = d > self.centers[-1]
        out = np.where(hi, self.times[-1] + slope_hi * (d - self.centers[-1]), out)
        slope_lo = (self.times[1] - self.times[0]) / max(self.centers[1] - self.centers[0], 1e-6)
        lo = d < self.centers[0]
        out = np.where(lo, np.maximum(self.times[0] + slope_lo * (d - self.centers[0]), 0.04), out)
        return np.where(np.isfinite(d), out, np.nan)


def fit_flight_time_model(
    distance_ft: Iterable[float],
    flight_time_s: Iterable[float],
    *,
    bin_width_ft: float = 2.0,
    min_bin_count: int = 20,
    support_quantile: float = 0.99,
) -> FlightTimeModel:
    d = np.asarray(list(distance_ft), dtype=float)
    t = np.asarray(list(flight_time_s), dtype=float)
    ok = np.isfinite(d) & np.isfinite(t) & (d >= 0) & (t > 0)
    d, t = d[ok], t[ok]
    if d.size == 0:
        return FlightTimeModel(np.array([]), np.array([]), np.array([]), np.array([]), np.nan, 0)

    hi = float(np.ceil(np.nanmax(d) / bin_width_ft) * bin_width_ft)
    edges = np.arange(0.0, hi + bin_width_ft, bin_width_ft)
    idx = np.clip(np.digitize(d, edges) - 1, 0, len(edges) - 2)

    centers: list[float] = []
    times: list[float] = []
    counts: list[int] = []
    pending_d: list[float] = []
    pending_t: list[float] = []
    for b in range(len(edges) - 1):
        sel = idx == b
        pending_d.extend(d[sel].tolist())
        pending_t.extend(t[sel].tolist())
        if len(pending_t) >= min_bin_count:
            centers.append(float(np.median(pending_d)))
            times.append(float(np.median(pending_t)))
            counts.append(len(pending_t))
            pending_d, pending_t = [], []
    if pending_t:
        if centers:
            # Fold the tail remainder into the last bin rather than trusting it alone.
            counts[-1] += len(pending_t)
        else:
            centers.append(float(np.median(pending_d)))
            times.append(float(np.median(pending_t)))
            counts.append(len(pending_t))

    centers_arr = np.asarray(centers, dtype=float)
    times_arr = np.maximum.accumulate(np.asarray(times, dtype=float))  # isotonic
    return FlightTimeModel(
        edges=edges,
        centers=centers_arr,
        times=times_arr,
        n_per_bin=np.asarray(counts, dtype=int),
        distance_support_max=float(np.quantile(d, support_quantile)),
        n_train=int(d.size),
    )


# ---------------------------------------------------------------------------
# Candidate projection and per-candidate evidence
# ---------------------------------------------------------------------------


def project_candidate(
    x: float,
    y: float,
    vx: float,
    vy: float,
    t_s: float,
    *,
    velocity_ok: bool,
    max_speed_fps: float = MAX_PROJECTION_SPEED_FPS,
    horizon_s: float = MAX_PROJECTION_HORIZON_S,
) -> tuple[float, float, bool]:
    """Project a receiver forward on causal velocity; returns (x, y, clipped).

    A missing or rejected velocity means *unknown motion*, not zero motion: the
    candidate is held at its release position and the caller records
    ``velocity_ok=False`` so the assumption stays visible downstream.
    """
    if not np.isfinite(x) or not np.isfinite(y):
        return float("nan"), float("nan"), False
    if not velocity_ok or not np.isfinite(vx) or not np.isfinite(vy) or not np.isfinite(t_s):
        return float(x), float(y), False
    t = float(min(max(t_s, 0.0), horizon_s))
    speed = float(np.hypot(vx, vy))
    clipped = False
    if speed > max_speed_fps:
        scale = max_speed_fps / speed
        vx, vy = vx * scale, vy * scale
        clipped = True
    px, py = float(x + vx * t), float(y + vy * t)
    cx = float(np.clip(px, -COURT_HALF_LENGTH_FT, COURT_HALF_LENGTH_FT))
    cy = float(np.clip(py, -COURT_HALF_WIDTH_FT, COURT_HALF_WIDTH_FT))
    if cx != px or cy != py:
        clipped = True
    return cx, cy, clipped


def candidate_evidence(
    flight: BallFlight,
    candidate_state: dict[str, Any],
    flight_time_model: FlightTimeModel,
    *,
    projection_iterations: int = 2,
) -> dict[str, Any]:
    """Geometric residuals for one candidate against one observed ball flight.

    The candidate's motion is projected only from release-time state; the ball
    path is the sole post-release channel, exactly as the Stage 2 plan allows.
    """
    x = float(candidate_state.get("x", np.nan))
    y = float(candidate_state.get("y", np.nan))
    vx = float(candidate_state.get("vx", np.nan))
    vy = float(candidate_state.get("vy", np.nan))
    velocity_ok = bool(candidate_state.get("velocity_ok", False))
    tracked = bool(np.isfinite(x) and np.isfinite(y))

    out: dict[str, Any] = {
        "candidate_tracked": tracked,
        "candidate_velocity_ok": velocity_ok,
        "candidate_x_event": x if tracked else np.nan,
        "candidate_y_event": y if tracked else np.nan,
        "candidate_vx": vx if velocity_ok else np.nan,
        "candidate_vy": vy if velocity_ok else np.nan,
        "release_distance_ft": np.nan,
        "t_pred_flight_s": np.nan,
        "proj_x_event": np.nan,
        "proj_y_event": np.nan,
        "projection_clipped": False,
        "proj_distance_ft": np.nan,
        "angle_deg": np.nan,
        "along_ft": np.nan,
        "lateral_ft": np.nan,
        "behind_release": False,
        "t_ball_to_candidate_s": np.nan,
        "flight_residual_s": np.nan,
        "outside_distance_support": False,
        "bearing_chord_ft": flight.clean_displacement_ft if flight.available else np.nan,
        "angle_channel_ok": False,
        "lateral_channel_ok": False,
        "flight_channel_ok": False,
    }
    if not tracked or not flight.available:
        return out

    rx, ry = flight.release_x, flight.release_y
    d0 = float(np.hypot(x - rx, y - ry))
    out["release_distance_ft"] = d0

    t_hat = float(flight_time_model.predict(d0))
    px, py, clipped = x, y, False
    for _ in range(max(projection_iterations, 1)):
        px, py, clipped = project_candidate(x, y, vx, vy, t_hat, velocity_ok=velocity_ok)
        d_proj = float(np.hypot(px - rx, py - ry))
        t_hat = float(flight_time_model.predict(d_proj))
    d_proj = float(np.hypot(px - rx, py - ry))

    along = float((px - rx) * flight.dir_x + (py - ry) * flight.dir_y)
    lateral = float(abs((px - rx) * flight.dir_y - (py - ry) * flight.dir_x))
    angle = _angle_between_deg(flight.dir_x, flight.dir_y, px - rx, py - ry)

    out.update(
        {
            "t_pred_flight_s": t_hat,
            "proj_x_event": px,
            "proj_y_event": py,
            "projection_clipped": bool(clipped),
            "proj_distance_ft": d_proj,
            "angle_deg": angle,
            "along_ft": along,
            "lateral_ft": lateral,
            "behind_release": bool(along <= 0.0),
            "outside_distance_support": bool(
                np.isfinite(flight_time_model.distance_support_max)
                and d_proj > flight_time_model.distance_support_max
            ),
            "angle_channel_ok": bool(np.isfinite(angle) and d_proj > 0),
            "lateral_channel_ok": bool(np.isfinite(lateral) and d_proj > 0),
        }
    )

    if flight.speed_usable and along > 0:
        t_ball = along / flight.speed_fps
        out["t_ball_to_candidate_s"] = float(t_ball)
        out["flight_residual_s"] = float(flight.t_clean_s - t_ball)
        out["flight_channel_ok"] = True
    return out


BALL_FLIGHT_COLUMNS = (
    "ball_flight_available",
    "ball_flight_reason",
    "ball_release_frame",
    "ball_release_x_event",
    "ball_release_y_event",
    "ball_dir_x",
    "ball_dir_y",
    "ball_speed_fps",
    "ball_clean_end_frame",
    "ball_clean_displacement_ft",
    "ball_max_turn_deg",
    "ball_t_clean_s",
    "ball_t_obs_s",
    "n_ball_frames",
    "n_clean_ball_frames",
    "ball_deflected",
)


def flight_to_columns(flight: BallFlight) -> dict[str, Any]:
    """Flatten a BallFlight for storage in the cached per-game state table."""
    return {
        "ball_flight_available": flight.available,
        "ball_flight_reason": flight.reason,
        "ball_release_frame": flight.release_frame,
        "ball_release_x_event": flight.release_x,
        "ball_release_y_event": flight.release_y,
        "ball_dir_x": flight.dir_x,
        "ball_dir_y": flight.dir_y,
        "ball_speed_fps": flight.speed_fps,
        "ball_clean_end_frame": flight.clean_end_frame,
        "ball_clean_displacement_ft": flight.clean_displacement_ft,
        "ball_max_turn_deg": flight.max_turn_deg,
        "ball_t_clean_s": flight.t_clean_s,
        "ball_t_obs_s": flight.t_obs_s,
        "n_ball_frames": flight.n_ball_frames,
        "n_clean_ball_frames": flight.n_clean_frames,
        "ball_deflected": flight.deflected,
    }


def flight_from_row(row: Any) -> BallFlight:
    """Rebuild a BallFlight from a cached state row (inverse of flight_to_columns)."""

    def _f(name: str) -> float:
        value = row[name] if name in row else np.nan
        return float(value) if value is not None and pd.notna(value) else float("nan")

    available = bool(row["ball_flight_available"]) if pd.notna(row["ball_flight_available"]) else False
    release_frame = row.get("ball_release_frame")
    clean_end = row.get("ball_clean_end_frame")
    return BallFlight(
        available=available,
        reason=str(row.get("ball_flight_reason", "")),
        release_frame=int(release_frame) if pd.notna(release_frame) else None,
        release_x=_f("ball_release_x_event"),
        release_y=_f("ball_release_y_event"),
        dir_x=_f("ball_dir_x"),
        dir_y=_f("ball_dir_y"),
        speed_fps=_f("ball_speed_fps"),
        clean_end_frame=int(clean_end) if pd.notna(clean_end) else None,
        clean_displacement_ft=_f("ball_clean_displacement_ft"),
        max_turn_deg=_f("ball_max_turn_deg"),
        t_clean_s=_f("ball_t_clean_s"),
        t_obs_s=_f("ball_t_obs_s"),
        n_ball_frames=int(row.get("n_ball_frames") or 0),
        n_clean_frames=int(row.get("n_clean_ball_frames") or 0),
        deflected=bool(row.get("ball_deflected", False)),
    )


EVIDENCE_COLUMNS = (
    "candidate_tracked",
    "candidate_velocity_ok",
    "candidate_x_event",
    "candidate_y_event",
    "candidate_vx",
    "candidate_vy",
    "release_distance_ft",
    "t_pred_flight_s",
    "proj_x_event",
    "proj_y_event",
    "projection_clipped",
    "proj_distance_ft",
    "angle_deg",
    "along_ft",
    "lateral_ft",
    "behind_release",
    "t_ball_to_candidate_s",
    "flight_residual_s",
    "outside_distance_support",
    "bearing_chord_ft",
    "angle_channel_ok",
    "lateral_channel_ok",
    "flight_channel_ok",
)


def _project_candidates_vec(
    x: np.ndarray,
    y: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    t_s: np.ndarray,
    velocity_ok: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised twin of :func:`project_candidate` (see that docstring)."""
    use = velocity_ok & np.isfinite(vx) & np.isfinite(vy) & np.isfinite(t_s)
    t = np.clip(np.where(use, t_s, 0.0), 0.0, MAX_PROJECTION_HORIZON_S)
    speed = np.hypot(np.where(use, vx, 0.0), np.where(use, vy, 0.0))
    over = use & (speed > MAX_PROJECTION_SPEED_FPS)
    scale = np.where(over, MAX_PROJECTION_SPEED_FPS / np.where(speed > 0, speed, 1.0), 1.0)
    px = x + np.where(use, vx * scale * t, 0.0)
    py = y + np.where(use, vy * scale * t, 0.0)
    cx = np.where(use, np.clip(px, -COURT_HALF_LENGTH_FT, COURT_HALF_LENGTH_FT), px)
    cy = np.where(use, np.clip(py, -COURT_HALF_WIDTH_FT, COURT_HALF_WIDTH_FT), py)
    clipped = over | (cx != px) | (cy != py)
    tracked = np.isfinite(x) & np.isfinite(y)
    return (
        np.where(tracked, cx, np.nan),
        np.where(tracked, cy, np.nan),
        clipped & tracked,
    )


def evidence_from_states(
    states: pd.DataFrame,
    flight_time_model: FlightTimeModel,
    *,
    projection_iterations: int = 2,
) -> pd.DataFrame:
    """Score cached release states against one fold's flight-time model.

    Projection depends on the flight-time model, and that model is fitted per
    outer fold, so the per-game cache stores raw release states and the
    residuals are computed here — once per fold — rather than being baked in.

    This is the vectorised twin of :func:`candidate_evidence`; the two are held
    to identical output by `test_vectorised_evidence_matches_scalar`.
    """
    if states.empty:
        return states.copy()

    def _col(name: str, default: float = np.nan) -> np.ndarray:
        if name in states.columns:
            return states[name].to_numpy(dtype=float)
        return np.full(len(states), default, dtype=float)

    x = _col("candidate_x_event")
    y = _col("candidate_y_event")
    vx = _col("candidate_vx")
    vy = _col("candidate_vy")
    velocity_ok = (
        states["candidate_velocity_ok"].fillna(False).to_numpy(dtype=bool)
        if "candidate_velocity_ok" in states.columns
        else np.zeros(len(states), dtype=bool)
    )
    available = states["ball_flight_available"].fillna(False).to_numpy(dtype=bool)
    rx = _col("ball_release_x_event")
    ry = _col("ball_release_y_event")
    dir_x = _col("ball_dir_x")
    dir_y = _col("ball_dir_y")
    speed = _col("ball_speed_fps")
    t_clean = _col("ball_t_clean_s")
    chord = _col("ball_clean_displacement_ft")

    tracked = np.isfinite(x) & np.isfinite(y)
    live = tracked & available

    d0 = np.hypot(x - rx, y - ry)
    t_hat = flight_time_model.predict(np.where(live, d0, np.nan))
    px, py, clipped = x.copy(), y.copy(), np.zeros(len(states), dtype=bool)
    for _ in range(max(projection_iterations, 1)):
        px, py, clipped = _project_candidates_vec(x, y, vx, vy, t_hat, velocity_ok)
        t_hat = flight_time_model.predict(np.where(live, np.hypot(px - rx, py - ry), np.nan))
    d_proj = np.hypot(px - rx, py - ry)

    dx, dy = px - rx, py - ry
    along = dx * dir_x + dy * dir_y
    lateral = np.abs(dx * dir_y - dy * dir_x)
    norm = np.hypot(dx, dy)
    cos = np.divide(
        along, np.where(norm > 0, norm, np.nan), out=np.full(len(states), np.nan), where=norm > 0
    )
    angle = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

    speed_usable = available & np.isfinite(speed) & (speed >= MIN_BALL_SPEED_FPS)
    flight_ok = live & speed_usable & (along > 0)
    t_ball = np.where(flight_ok, along / np.where(speed > 0, speed, np.nan), np.nan)
    geom_ok = live & (d_proj > 0)

    evidence = pd.DataFrame(
        {
            "candidate_tracked": tracked,
            "candidate_velocity_ok": velocity_ok,
            "candidate_x_event": np.where(tracked, x, np.nan),
            "candidate_y_event": np.where(tracked, y, np.nan),
            "candidate_vx": np.where(velocity_ok, vx, np.nan),
            "candidate_vy": np.where(velocity_ok, vy, np.nan),
            "release_distance_ft": np.where(live, d0, np.nan),
            "t_pred_flight_s": np.where(live, t_hat, np.nan),
            "proj_x_event": np.where(live, px, np.nan),
            "proj_y_event": np.where(live, py, np.nan),
            "projection_clipped": clipped & live,
            "proj_distance_ft": np.where(live, d_proj, np.nan),
            "angle_deg": np.where(live, angle, np.nan),
            "along_ft": np.where(live, along, np.nan),
            "lateral_ft": np.where(live, lateral, np.nan),
            "behind_release": live & (along <= 0.0),
            "t_ball_to_candidate_s": t_ball,
            "flight_residual_s": np.where(flight_ok, t_clean - t_ball, np.nan),
            "outside_distance_support": live
            & np.isfinite(flight_time_model.distance_support_max)
            & (d_proj > flight_time_model.distance_support_max),
            "bearing_chord_ft": np.where(available, chord, np.nan),
            "angle_channel_ok": geom_ok & np.isfinite(angle),
            "lateral_channel_ok": geom_ok & np.isfinite(lateral),
            "flight_channel_ok": flight_ok,
        },
        index=states.index,
    )
    base = states.drop(columns=[c for c in EVIDENCE_COLUMNS if c in states.columns])
    return pd.concat([base, evidence], axis=1)


# ---------------------------------------------------------------------------
# Likelihood scales and probabilities
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvidenceScales:
    """Residual scales for the true receiver, fitted on training games.

    The lateral tolerance is not a constant. A bearing estimated from a short
    ball chord is imprecise, and that bearing error fans out with distance, so
    the usable width at a candidate is

        sigma_lateral = sqrt(sigma_pos^2 + (sigma_ball / chord * along)^2)

    with ``sigma_ball`` the ball-position noise that sets the bearing error and
    ``sigma_pos`` the receiver position-plus-projection noise. A 3-frame flight
    therefore separates candidates far more weakly than a 10-frame flight
    instead of pretending to the same precision.
    """

    sigma_ball_ft: float
    sigma_pos_ft: float
    flight_offset_s: float
    sigma_flight_s: float
    nu: float = STUDENT_T_DOF
    undershoot_scale: float = FLIGHT_UNDERSHOOT_SCALE
    n_train_passes: int = 0
    source: str = "training_fold_completed_passes"

    def as_dict(self) -> dict[str, Any]:
        return {
            "sigma_ball_ft": self.sigma_ball_ft,
            "sigma_pos_ft": self.sigma_pos_ft,
            "flight_offset_s": self.flight_offset_s,
            "sigma_flight_s": self.sigma_flight_s,
            "nu": self.nu,
            "undershoot_scale": self.undershoot_scale,
            "n_train_passes": self.n_train_passes,
            "source": self.source,
        }

    def lateral_sigma(self, along_ft: np.ndarray, chord_ft: np.ndarray) -> np.ndarray:
        bearing = self.sigma_ball_ft * np.maximum(along_ft, 0.0) / np.maximum(chord_ft, 1.0)
        return np.sqrt(np.square(self.sigma_pos_ft) + np.square(bearing))


def _robust_scale(values: Iterable[float], floor: float, centre: float = 0.0) -> float:
    v = np.asarray(list(values), dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float(floor)
    return float(max(1.4826 * float(np.median(np.abs(v - centre))), floor))


def _median_or(values: Iterable[float], default: float = 0.0) -> float:
    v = np.asarray(list(values), dtype=float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else float(default)


def fit_evidence_scales(true_receiver_rows: pd.DataFrame) -> EvidenceScales:
    """Fit likelihood widths from the residuals of *known* receivers."""
    rows = true_receiver_rows
    if rows.empty:
        return EvidenceScales(MIN_SIGMA_BALL_FT, MIN_SIGMA_POS_FT, 0.0, MIN_SIGMA_FLIGHT_S)

    chord = np.maximum(rows["bearing_chord_ft"].to_numpy(dtype=float), 1.0)
    along = np.maximum(rows["along_ft"].to_numpy(dtype=float), 0.0)
    lateral = rows["lateral_ft"].to_numpy(dtype=float)
    angle_rad = np.radians(rows["angle_deg"].to_numpy(dtype=float))

    # Bearing error expressed as an equivalent ball-position error in feet.
    sigma_ball = _robust_scale(angle_rad * chord, MIN_SIGMA_BALL_FT)
    # Whatever lateral error the bearing term cannot explain is position noise.
    bearing_component = sigma_ball * along / chord
    residual_pos = np.sqrt(np.maximum(np.square(lateral) - np.square(bearing_component), 0.0))
    sigma_pos = _robust_scale(residual_pos, MIN_SIGMA_POS_FT)

    # The clean-flight segment ends at or before the catch, so the flight-time
    # residual of a known receiver has a systematic offset; centre it rather
    # than letting a fixed bias inflate the width.
    flight = rows["flight_residual_s"]
    offset = _median_or(flight, 0.0)
    sigma_flight = _robust_scale(flight, MIN_SIGMA_FLIGHT_S, centre=offset)

    return EvidenceScales(
        sigma_ball_ft=sigma_ball,
        sigma_pos_ft=sigma_pos,
        flight_offset_s=offset,
        sigma_flight_s=sigma_flight,
        n_train_passes=int(len(rows)),
    )


def _student_t_loglik(z: np.ndarray, nu: float) -> np.ndarray:
    return -0.5 * (nu + 1.0) * np.log1p(np.square(z) / nu)


def candidate_log_likelihood(
    evidence: pd.DataFrame,
    scales: EvidenceScales,
    *,
    flight_mode: str = FLIGHT_MODE_ONE_SIDED,
) -> pd.Series:
    """Per-candidate log likelihood; unavailable channels contribute nothing.

    ``flight_mode``:
    - ``symmetric`` — a completed pass reaches its receiver, so the flight-time
      residual is centred and penalised in both directions.
    - ``one_sided_overshoot`` — an intercepted pass stops early, so a candidate
      farther than the ball got is expected and only weakly penalised
      (``undershoot_scale``), while a candidate the ball had already flown past
      is evidence against.
    """
    if flight_mode not in (FLIGHT_MODE_SYMMETRIC, FLIGHT_MODE_ONE_SIDED):
        raise ValueError(f"unknown flight_mode: {flight_mode}")

    ll = np.zeros(len(evidence), dtype=float)

    lateral = evidence["lateral_ft"].to_numpy(dtype=float)
    along = evidence["along_ft"].to_numpy(dtype=float)
    chord = evidence["bearing_chord_ft"].to_numpy(dtype=float)
    ok = evidence["lateral_channel_ok"].to_numpy(dtype=bool) & np.isfinite(lateral)
    sigma_lat = scales.lateral_sigma(np.nan_to_num(along), np.nan_to_num(chord, nan=1.0))
    z_lat = np.nan_to_num(lateral) / np.where(sigma_lat > 0, sigma_lat, 1.0)
    ll += np.where(ok, _student_t_loglik(z_lat, scales.nu), 0.0)

    resid = evidence["flight_residual_s"].to_numpy(dtype=float) - scales.flight_offset_s
    ok = evidence["flight_channel_ok"].to_numpy(dtype=bool) & np.isfinite(resid)
    effective = (
        resid
        if flight_mode == FLIGHT_MODE_SYMMETRIC
        else np.where(resid >= 0, resid, resid / scales.undershoot_scale)
    )
    z_t = np.nan_to_num(effective) / scales.sigma_flight_s
    ll += np.where(ok, _student_t_loglik(z_t, scales.nu), 0.0)

    behind = evidence["behind_release"].to_numpy(dtype=bool)
    ll += np.where(behind, BEHIND_PASSER_LOGLIK_PENALTY, 0.0)

    # A candidate with no tracking has no evidence at all and must not win by
    # default; it keeps the uniform share only if nothing else scored either.
    untracked = ~evidence["candidate_tracked"].to_numpy(dtype=bool)
    ll = np.where(untracked, -np.inf, ll)
    return pd.Series(ll, index=evidence.index, name="log_likelihood")


def normalize_within_pass(
    evidence: pd.DataFrame,
    log_likelihood: pd.Series,
    *,
    pass_key: str = "passId",
) -> pd.Series:
    """Softmax the log likelihood inside each pass; always returns a simplex.

    A pass whose candidates all scored ``-inf`` (nobody tracked) falls back to
    the uniform distribution: no evidence must not become false confidence.
    """
    if evidence.empty:
        return pd.Series(dtype=float, index=evidence.index, name="target_probability")
    codes = pd.factorize(evidence[pass_key], sort=False)[0]
    n_groups = int(codes.max()) + 1
    ll = log_likelihood.to_numpy(dtype=float)
    finite = np.isfinite(ll)

    group_max = np.full(n_groups, -np.inf)
    np.maximum.at(group_max, codes[finite], ll[finite])
    shifted = np.full(len(ll), -np.inf)
    np.subtract(ll, group_max[codes], out=shifted, where=finite)
    weights = np.where(finite, np.exp(shifted), 0.0)

    totals = np.zeros(n_groups)
    np.add.at(totals, codes, weights)
    sizes = np.bincount(codes, minlength=n_groups).astype(float)
    denom = totals[codes]
    probs = np.where(denom > 0, weights / np.where(denom > 0, denom, 1.0), 1.0 / sizes[codes])
    return pd.Series(probs, index=evidence.index, name="target_probability").clip(lower=0.0)


def attach_probabilities(
    evidence: pd.DataFrame,
    scales: EvidenceScales,
    *,
    flight_mode: str = FLIGHT_MODE_ONE_SIDED,
    pass_key: str = "passId",
    prob_col: str = "target_probability",
) -> pd.DataFrame:
    out = evidence.copy()
    ll = candidate_log_likelihood(out, scales, flight_mode=flight_mode)
    out["log_likelihood"] = ll
    out[prob_col] = normalize_within_pass(out, ll, pass_key=pass_key)
    return out


# ---------------------------------------------------------------------------
# Decision rule
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AutoAcceptRule:
    """Provisional auto-accept rule; selected inside a training fold."""

    p_min: float
    prob_margin: float
    geom_margin_ft: float
    target_precision: float = AUTO_ACCEPT_TARGET_PRECISION
    train_precision: float = np.nan
    train_coverage: float = np.nan
    train_n: int = 0
    met_target: bool = False
    provisional: bool = True
    fold_id: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "fold_id": self.fold_id,
            "p_min": self.p_min,
            "prob_margin": self.prob_margin,
            "geom_margin_ft": self.geom_margin_ft,
            "target_precision": self.target_precision,
            "train_precision": self.train_precision,
            "train_coverage": self.train_coverage,
            "train_n": self.train_n,
            "met_target": self.met_target,
            "provisional": self.provisional,
        }


def summarize_pass_level(
    scored: pd.DataFrame,
    *,
    pass_key: str = "passId",
    prob_col: str = "target_probability",
) -> pd.DataFrame:
    """Collapse candidate rows to one row per pass with top-two diagnostics."""
    if scored.empty:
        return pd.DataFrame()
    ordered = scored.sort_values(
        [pass_key, prob_col, "candidateId"], ascending=[True, False, True], kind="mergesort"
    )
    grouped = ordered.groupby(pass_key, sort=False)
    top = grouped.head(1).set_index(pass_key)
    second = grouped.nth(1).set_index(pass_key)

    out = pd.DataFrame(index=top.index)
    out["top_candidateId"] = top["candidateId"]
    out["p_top"] = top[prob_col]
    out["top_angle_deg"] = top["angle_deg"]
    out["top_lateral_ft"] = top["lateral_ft"]
    out["top_flight_residual_s"] = top["flight_residual_s"]
    out["top_proj_x_event"] = top["proj_x_event"]
    out["top_proj_y_event"] = top["proj_y_event"]
    out["top_candidate_velocity_ok"] = top["candidate_velocity_ok"]
    out["top_outside_distance_support"] = top["outside_distance_support"]
    out["second_candidateId"] = second["candidateId"]
    out["p_second"] = second[prob_col]
    out["second_lateral_ft"] = second["lateral_ft"]
    out["second_angle_deg"] = second["angle_deg"]
    out["prob_margin"] = out["p_top"] - out["p_second"]
    out["geom_margin_ft"] = out["second_lateral_ft"] - out["top_lateral_ft"]
    out["n_candidates"] = grouped.size()
    out["n_candidates_tracked"] = grouped["candidate_tracked"].sum().astype(int)

    p = ordered[prob_col].to_numpy(dtype=float)
    terms = np.where(p > 0, p * np.log2(np.clip(p, PROB_FLOOR, 1.0)), 0.0)
    out["entropy_bits"] = -pd.Series(terms, index=ordered[pass_key].to_numpy()).groupby(level=0).sum()
    return out.reset_index()


def apply_auto_accept(summary: pd.DataFrame, rule: AutoAcceptRule) -> pd.Series:
    """Boolean mask of passes meeting the probability and geometric margins."""
    p_top = summary["p_top"].to_numpy(dtype=float)
    prob_margin = summary["prob_margin"].to_numpy(dtype=float)
    geom = summary["geom_margin_ft"].to_numpy(dtype=float)
    ok = (
        np.isfinite(p_top)
        & (p_top >= rule.p_min)
        & np.isfinite(prob_margin)
        & (prob_margin >= rule.prob_margin)
        & np.isfinite(geom)
        & (geom >= rule.geom_margin_ft)
    )
    return pd.Series(ok, index=summary.index, name="auto_accept")


def select_auto_accept_rule(
    train_summary: pd.DataFrame,
    *,
    truth_col: str = "true_receiverId",
    fold_id: str = "",
    target_precision: float = AUTO_ACCEPT_TARGET_PRECISION,
    p_min_grid: Sequence[float] = P_MIN_GRID,
    prob_margin_grid: Sequence[float] = PROB_MARGIN_GRID,
    geom_margin_grid: Sequence[float] = GEOM_MARGIN_FT_GRID,
) -> AutoAcceptRule:
    """Pick the widest-coverage rule reaching the precision target in training.

    Stage 0 freezes the *procedure* (`failed_pass_auto_accept_geometric_margin:
    TBD_TRAINING_FOLD`, "maximize training-fold audit agreement subject to
    unresolved-rate observability"). No human audit exists yet, so held-out
    completed passes — whose receiver is known — stand in for audit labels. The
    resulting numbers are provisional until Stage 4.
    """
    best: AutoAcceptRule | None = None
    if not train_summary.empty and truth_col in train_summary.columns:
        correct = (train_summary["top_candidateId"] == train_summary[truth_col]).to_numpy(dtype=bool)
        n_total = len(train_summary)
        for p_min in p_min_grid:
            for prob_margin in prob_margin_grid:
                for geom in geom_margin_grid:
                    trial = AutoAcceptRule(p_min, prob_margin, geom, target_precision, fold_id=fold_id)
                    mask = apply_auto_accept(train_summary, trial).to_numpy(dtype=bool)
                    n_acc = int(mask.sum())
                    if n_acc == 0:
                        continue
                    precision = float(correct[mask].mean())
                    coverage = n_acc / max(n_total, 1)
                    cand = AutoAcceptRule(
                        p_min=p_min,
                        prob_margin=prob_margin,
                        geom_margin_ft=geom,
                        target_precision=target_precision,
                        train_precision=precision,
                        train_coverage=coverage,
                        train_n=n_total,
                        met_target=precision >= target_precision,
                        fold_id=fold_id,
                    )
                    if best is None or _rule_is_better(cand, best):
                        best = cand
    if best is None:
        # No usable training signal: fall back to the strictest grid point so a
        # missing fold cannot silently produce permissive auto-accepts.
        return AutoAcceptRule(
            p_min=max(p_min_grid),
            prob_margin=max(prob_margin_grid),
            geom_margin_ft=max(geom_margin_grid),
            target_precision=target_precision,
            fold_id=fold_id,
        )
    return best


def _rule_is_better(cand: AutoAcceptRule, best: AutoAcceptRule) -> bool:
    """Prefer meeting precision, then coverage, then precision, then strictness."""
    key_c = (
        cand.met_target,
        round(cand.train_coverage, 6) if cand.met_target else 0.0,
        round(cand.train_precision, 6),
        -cand.p_min,
        -cand.prob_margin,
        -cand.geom_margin_ft,
    )
    key_b = (
        best.met_target,
        round(best.train_coverage, 6) if best.met_target else 0.0,
        round(best.train_precision, 6),
        -best.p_min,
        -best.prob_margin,
        -best.geom_margin_ft,
    )
    return key_c > key_b


def classify_decisions(
    summary: pd.DataFrame,
    rule: AutoAcceptRule,
    *,
    min_tracked_candidates: int = 2,
) -> pd.DataFrame:
    """Label each pass auto_accept / ambiguous / unresolved with a reason.

    ``unresolved`` means the evidence never existed (no candidate set, no ball
    flight, no tracking). ``ambiguous`` means the evidence exists but does not
    separate the candidates well enough to accept a label without a human, which
    is what the audit sheet is for.
    """
    out = summary.copy()
    accept = apply_auto_accept(out, rule).to_numpy(dtype=bool)

    count_col = "n_candidates_declared" if "n_candidates_declared" in out.columns else "n_candidates"
    no_candidates = out[count_col].to_numpy(dtype=float) != 4
    no_flight = ~out["ball_flight_available"].to_numpy(dtype=bool) if "ball_flight_available" in out.columns else np.zeros(len(out), dtype=bool)
    thin_tracking = out["n_candidates_tracked"].to_numpy(dtype=float) < min_tracked_candidates

    decision = np.where(
        no_candidates,
        DECISION_UNRESOLVED,
        np.where(
            no_flight | thin_tracking,
            DECISION_UNRESOLVED,
            np.where(accept, DECISION_AUTO_ACCEPT, DECISION_AMBIGUOUS),
        ),
    )
    reason = np.where(
        no_candidates,
        "candidate_set_not_four",
        np.where(
            no_flight,
            "no_usable_ball_flight",
            np.where(
                thin_tracking,
                "insufficient_candidate_tracking",
                np.where(accept, "margin_satisfied", "margin_not_satisfied"),
            ),
        ),
    )
    out["decision"] = decision
    out["decision_reason"] = reason
    # An unresolved pass has no inferred target at all; do not leak a hard label.
    out["inferred_targetId"] = np.where(
        out["decision"] == DECISION_AUTO_ACCEPT, out["top_candidateId"], pd.NA
    )
    out["auto_accept_rule_p_min"] = rule.p_min
    out["auto_accept_rule_prob_margin"] = rule.prob_margin
    out["auto_accept_rule_geom_margin_ft"] = rule.geom_margin_ft
    out["auto_accept_rule_provisional"] = rule.provisional
    return out


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def completed_pass_agreement(
    summary: pd.DataFrame,
    *,
    truth_col: str = "true_receiverId",
) -> dict[str, Any]:
    """Top-1 accuracy and auto-accept precision against the known receiver."""
    if summary.empty or truth_col not in summary.columns:
        return {"n": 0, "top1_accuracy": np.nan, "auto_accept_precision": np.nan, "auto_accept_rate": np.nan}
    truth = summary[truth_col]
    correct = (summary["top_candidateId"] == truth).to_numpy(dtype=bool)
    accepted = (summary["decision"] == DECISION_AUTO_ACCEPT).to_numpy(dtype=bool)
    return {
        "n": int(len(summary)),
        "top1_accuracy": float(correct.mean()),
        "auto_accept_rate": float(accepted.mean()),
        "auto_accept_precision": float(correct[accepted].mean()) if accepted.any() else np.nan,
        "n_auto_accepted": int(accepted.sum()),
        "ambiguous_rate": float((summary["decision"] == DECISION_AMBIGUOUS).mean()),
        "unresolved_rate": float((summary["decision"] == DECISION_UNRESOLVED).mean()),
        "truth_in_candidate_set_rate": float(summary.get("truth_in_candidate_set", pd.Series(dtype=bool)).mean())
        if "truth_in_candidate_set" in summary.columns
        else np.nan,
    }


CALIBRATION_BINS: tuple[float, ...] = (0.0, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0001)


def probability_calibration(
    summary: pd.DataFrame,
    *,
    truth_col: str = "true_receiverId",
    bins: Sequence[float] = CALIBRATION_BINS,
) -> pd.DataFrame:
    """Is `p_top` a probability? Bin it and compare to held-out hit rate.

    Stage 2 promises a distribution rather than a label, so the distribution has
    to mean something. This is the evidence for that claim, and it only uses
    held-out completed passes, where the receiver is known.
    """
    if summary.empty or truth_col not in summary.columns:
        return pd.DataFrame(columns=["p_top_bin", "n", "mean_p_top", "empirical_accuracy"])
    frame = summary.copy()
    frame["correct"] = frame["top_candidateId"] == frame[truth_col]
    frame["p_top_bin"] = pd.cut(frame["p_top"], list(bins), right=True, include_lowest=True)
    grouped = frame.groupby("p_top_bin", observed=True).agg(
        n=("correct", "size"),
        mean_p_top=("p_top", "mean"),
        empirical_accuracy=("correct", "mean"),
    )
    out = grouped.reset_index()
    out["p_top_bin"] = out["p_top_bin"].astype(str)
    out["calibration_gap"] = out["mean_p_top"] - out["empirical_accuracy"]
    return out


def unresolved_summary(summary: pd.DataFrame, kill_threshold_pct: float = 10.0) -> dict[str, Any]:
    """Unresolved accounting against the frozen 10% kill criterion.

    Two figures are reported because they answer different questions:
    ``strict`` counts passes with no usable evidence, and ``pre_audit`` counts
    everything without an accepted label — the honest number while the audit
    sheet is still blank.
    """
    n = int(len(summary))
    if n == 0:
        return {"n_failed": 0, "kill_threshold_pct": kill_threshold_pct}
    strict = int((summary["decision"] == DECISION_UNRESOLVED).sum())
    ambiguous = int((summary["decision"] == DECISION_AMBIGUOUS).sum())
    accepted = int((summary["decision"] == DECISION_AUTO_ACCEPT).sum())
    strict_pct = 100.0 * strict / n
    pre_audit_pct = 100.0 * (strict + ambiguous) / n
    return {
        "n_failed": n,
        "n_auto_accepted": accepted,
        "n_ambiguous": ambiguous,
        "n_unresolved_strict": strict,
        "auto_accept_pct": 100.0 * accepted / n,
        "ambiguous_pct": 100.0 * ambiguous / n,
        "unresolved_strict_pct": strict_pct,
        "unresolved_pre_audit_pct": pre_audit_pct,
        "kill_threshold_pct": kill_threshold_pct,
        "strict_breaches_kill": strict_pct > kill_threshold_pct,
        "pre_audit_breaches_kill": pre_audit_pct > kill_threshold_pct,
        "max_unresolved_allowed": int(np.floor(kill_threshold_pct / 100.0 * n)),
    }


def exclusion_sensitivity_stub(
    failed_summary: pd.DataFrame,
    completed_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Per-receiver exposure to dropping every inferred failure (Stage 4 stub).

    Compares receiver-level completion denominators under two specifications:
    completed passes only, versus completed passes plus auto-accepted inferred
    failures. Stage 4 reruns receiver-specific calibration under both.
    """
    rows: list[dict[str, Any]] = []
    comp = completed_summary.copy()
    if not comp.empty:
        comp_counts = comp.groupby("true_receiverId").size().rename("n_completed")
    else:
        comp_counts = pd.Series(dtype=int, name="n_completed")

    acc = failed_summary[failed_summary["decision"] == DECISION_AUTO_ACCEPT]
    if not acc.empty:
        fail_counts = acc.groupby("top_candidateId").size().rename("n_inferred_failed")
    else:
        fail_counts = pd.Series(dtype=int, name="n_inferred_failed")

    tab = pd.concat([comp_counts, fail_counts], axis=1).fillna(0)
    tab.index.name = "receiverId"
    tab = tab.astype(int).reset_index()
    tab["n_with_inferred"] = tab["n_completed"] + tab["n_inferred_failed"]
    tab["completion_rate_excluding_failures"] = 1.0
    tab["completion_rate_including_inferred"] = tab["n_completed"] / tab["n_with_inferred"].clip(lower=1)
    tab["share_of_attempts_inferred"] = tab["n_inferred_failed"] / tab["n_with_inferred"].clip(lower=1)
    rows.extend(tab.to_dict("records"))
    return pd.DataFrame(rows)
