"""Fold-safe empirical pass flight-time models for Stage 3.

Flight time is fit as a robust monotonic function of pass distance from
completed passes with known receivers only (narrow-path
``receiver_specific_eligible``). Each outer fold's held-out game is scored
with a model trained on the other nine games — never on itself.

Distance / spatial support for counterfactual rejection uses the training
fold's *attempted* pass distance distribution (completed + failed attempts),
matching the Stage 0 ``pass_distance_support_rejection`` procedure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

import hashlib
import json

from .geometry import euclidean_distance
from .narrow_path import receiver_completion_training_labels
from .target_inference import FlightTimeModel, fit_flight_time_model
from .velocities import FRAME_RATE_HZ

# Re-export for callers that only need Stage 3 flight-time surfaces.
__all__ = [
    "FlightTimeModel",
    "DistanceSupport",
    "FoldFlightBundle",
    "fit_flight_time_model",
    "observed_flight_time_s",
    "fit_fold_flight_bundles",
    "bundle_for_game",
    "predict_flight_time",
    "outside_distance_support",
    "enrich_attempt_distances",
    "flight_bundle_cache_key",
    "assert_bundles_trainable",
]


@dataclass(frozen=True)
class DistanceSupport:
    """Training-fold attempted-pass distance support for rejection."""

    distance_min_ft: float
    distance_max_ft: float
    support_quantile: float
    n_attempts: int

    def contains(self, distance_ft: float) -> bool:
        if not np.isfinite(distance_ft):
            return False
        return bool(self.distance_min_ft <= distance_ft <= self.distance_max_ft)


@dataclass(frozen=True)
class FoldFlightBundle:
    """Leave-one-game-out flight time + distance support for one held-out game."""

    fold_id: str
    held_out_gameId: int
    train_gameIds: tuple[int, ...]
    flight_model: FlightTimeModel
    distance_support: DistanceSupport
    n_train_completed: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "fold_id": self.fold_id,
            "held_out_gameId": self.held_out_gameId,
            "train_gameIds": list(self.train_gameIds),
            "n_train_completed": self.n_train_completed,
            "flight_time_n_train": int(self.flight_model.n_train),
            "flight_time_bins": int(self.flight_model.centers.size),
            "distance_support_max_ft": float(self.distance_support.distance_max_ft),
            "distance_support_min_ft": float(self.distance_support.distance_min_ft),
            "distance_support_n_attempts": int(self.distance_support.n_attempts),
            "distance_support_quantile": float(self.distance_support.support_quantile),
        }


def observed_flight_time_s(
    start_frame: Iterable[float],
    end_frame: Iterable[float],
    *,
    frame_rate_hz: float = FRAME_RATE_HZ,
) -> np.ndarray:
    """Flight duration from release/start frame to catch/end frame."""
    s = np.asarray(list(start_frame), dtype=float)
    e = np.asarray(list(end_frame), dtype=float)
    dt = (e - s) / float(frame_rate_hz)
    return np.where(np.isfinite(dt) & (dt > 0), dt, np.nan)


def fit_distance_support(
    distances_ft: Iterable[float],
    *,
    low_quantile: float = 0.01,
    high_quantile: float = 0.99,
) -> DistanceSupport:
    d = np.asarray(list(distances_ft), dtype=float)
    d = d[np.isfinite(d) & (d >= 0)]
    if d.size == 0:
        return DistanceSupport(np.nan, np.nan, high_quantile, 0)
    return DistanceSupport(
        distance_min_ft=float(np.quantile(d, low_quantile)),
        distance_max_ft=float(np.quantile(d, high_quantile)),
        support_quantile=float(high_quantile),
        n_attempts=int(d.size),
    )


def _completed_training_rows(
    passes: pd.DataFrame,
    eligibility: pd.DataFrame,
) -> pd.DataFrame:
    """Completed known-receiver passes only, via the narrow-path helper."""
    labels = receiver_completion_training_labels(eligibility)
    if labels.empty or passes.empty:
        return pd.DataFrame()
    # Eligibility keys on passId/gameId; passes table uses `id`.
    pass_key = "id" if "id" in passes.columns and "passId" not in passes.columns else "passId"
    p = passes.copy()
    if pass_key == "id":
        p = p.rename(columns={"id": "passId"})
    cols = ["passId", "gameId", "startFrame", "endFrame", "distance"]
    for optional in ("distance_recomputed", "passerId", "receiverId", "pass_outcome_class"):
        if optional in p.columns:
            cols.append(optional)
    merged = labels.merge(p[cols], on=["passId", "gameId"], how="inner", suffixes=("", "_pass"))
    if "distance_recomputed" in merged.columns:
        dist = merged["distance_recomputed"].where(
            merged["distance_recomputed"].notna(), merged["distance"]
        )
    else:
        dist = merged["distance"]
    merged = merged.assign(
        train_distance_ft=dist.astype(float),
        train_flight_time_s=observed_flight_time_s(merged["startFrame"], merged["endFrame"]),
    )
    return merged


def enrich_attempt_distances(
    passes: pd.DataFrame,
    *,
    tracking_players: pd.DataFrame | None = None,
    tracking_frames: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach ``attempt_distance_ft`` for support rejection (completed + failed).

    Prefer feed ``distance_recomputed`` / ``distance``. When those are null
    (typical for failed attempts), recompute from release tracking: passer XY at
    ``startFrame`` to ball XY at ``endFrame`` (walking back while undetected).
    Fallback: passer → interceptor at release when both are tracked.
    """
    out = passes.copy()
    if "id" in out.columns and "passId" not in out.columns:
        out = out.rename(columns={"id": "passId"})

    base = pd.Series(np.nan, index=out.index, dtype=float)
    if "distance_recomputed" in out.columns:
        base = out["distance_recomputed"].astype(float)
    if "distance" in out.columns:
        base = base.where(np.isfinite(base), out["distance"].astype(float))

    need = ~np.isfinite(base.to_numpy(dtype=float))
    if need.any() and tracking_players is not None and not tracking_players.empty:
        players = tracking_players
        p_idx = players.set_index(["frameIdx", "playerId"], drop=False)
        frames = tracking_frames
        f_by_frame = None
        if frames is not None and not frames.empty and "ball_x_event" in frames.columns:
            f_by_frame = frames.drop_duplicates("frameIdx").set_index("frameIdx")

        dists = base.to_numpy(dtype=float, copy=True)
        for i in np.where(need)[0]:
            row = out.iloc[int(i)]
            sf = row.get("startFrame")
            ef = row.get("endFrame", sf)
            passer_id = row.get("passerId")
            if pd.isna(sf) or pd.isna(passer_id):
                continue
            sf_i, passer_i = int(sf), int(passer_id)
            try:
                prow = p_idx.loc[(sf_i, passer_i)]
                if isinstance(prow, pd.DataFrame):
                    prow = prow.iloc[0]
                px = float(prow.get("x_event", np.nan))
                py = float(prow.get("y_event", np.nan))
            except KeyError:
                continue
            if not (np.isfinite(px) and np.isfinite(py)):
                continue

            bx = by = float("nan")
            if f_by_frame is not None and pd.notna(ef):
                for fr in range(int(ef), sf_i - 1, -1):
                    if fr not in f_by_frame.index:
                        continue
                    frow = f_by_frame.loc[fr]
                    if isinstance(frow, pd.DataFrame):
                        frow = frow.iloc[0]
                    if not bool(frow.get("ball_isDetected", True)):
                        continue
                    bx = float(frow.get("ball_x_event", np.nan))
                    by = float(frow.get("ball_y_event", np.nan))
                    if np.isfinite(bx) and np.isfinite(by):
                        break
            if np.isfinite(bx) and np.isfinite(by):
                dists[int(i)] = euclidean_distance(px, py, bx, by)
                continue

            # Fallback: passer → interceptor at release.
            iid = row.get("interceptorId", row.get("toReceiverId"))
            if pd.isna(iid):
                continue
            try:
                irow = p_idx.loc[(sf_i, int(iid))]
                if isinstance(irow, pd.DataFrame):
                    irow = irow.iloc[0]
                ix = float(irow.get("x_event", np.nan))
                iy = float(irow.get("y_event", np.nan))
            except KeyError:
                continue
            if np.isfinite(ix) and np.isfinite(iy):
                dists[int(i)] = euclidean_distance(px, py, ix, iy)
        base = pd.Series(dists, index=out.index)

    out["attempt_distance_ft"] = base.astype(float)
    return out


def fit_fold_flight_bundles(
    passes: pd.DataFrame,
    eligibility: pd.DataFrame,
    outer_folds: Sequence[dict[str, Any]],
    *,
    attempt_distance_col: str = "attempt_distance_ft",
) -> dict[int, FoldFlightBundle]:
    """Fit one flight-time + support bundle per held-out game (no leakage).

    ``passes`` must cover **all** train-fold games (not a ``--games`` subset).
    Empty train completed or empty attempt support raises.
    """
    completed = _completed_training_rows(passes, eligibility)
    attempts = passes.copy()
    if "id" in attempts.columns and "passId" not in attempts.columns:
        attempts = attempts.rename(columns={"id": "passId"})
    if attempt_distance_col not in attempts.columns:
        # Fall back through recomputed → raw distance for older callers.
        attempts = enrich_attempt_distances(attempts)
        attempt_distance_col = "attempt_distance_ft"

    bundles: dict[int, FoldFlightBundle] = {}
    for fold in outer_folds:
        fold_id = str(fold["fold_id"])
        held_out = int(fold["held_out_gameId"])
        train_ids = tuple(int(g) for g in fold["train_gameIds"])
        if held_out in train_ids:
            raise ValueError(f"{fold_id}: held-out game {held_out} appears in train_gameIds")

        train_completed = completed[completed["gameId"].isin(train_ids)]
        if train_completed.empty:
            raise ValueError(
                f"{fold_id}: zero completed training passes for train_gameIds={list(train_ids)}; "
                "load ALL freeze games for flight fitting (do not subset passes with --games)"
            )
        model = fit_flight_time_model(
            train_completed["train_distance_ft"],
            train_completed["train_flight_time_s"],
        )
        train_attempts = attempts[attempts["gameId"].isin(train_ids)]
        support = fit_distance_support(train_attempts[attempt_distance_col].astype(float))
        if support.n_attempts == 0 or not np.isfinite(support.distance_max_ft):
            raise ValueError(
                f"{fold_id}: empty attempted-pass distance support for train_gameIds={list(train_ids)}"
            )
        bundles[held_out] = FoldFlightBundle(
            fold_id=fold_id,
            held_out_gameId=held_out,
            train_gameIds=train_ids,
            flight_model=model,
            distance_support=support,
            n_train_completed=int(len(train_completed)),
        )
    return bundles


def assert_bundles_trainable(bundles: dict[int, FoldFlightBundle]) -> None:
    for gid, b in bundles.items():
        if b.n_train_completed <= 0:
            raise ValueError(f"fold for game {gid}: n_train_completed=0")
        if b.distance_support.n_attempts <= 0:
            raise ValueError(f"fold for game {gid}: distance_support.n_attempts=0")


def flight_bundle_cache_key(bundle: FoldFlightBundle) -> str:
    """Stable short hash of flight model + support used to invalidate per-game caches."""
    payload = {
        "fold_id": bundle.fold_id,
        "held_out": bundle.held_out_gameId,
        "train": list(bundle.train_gameIds),
        "n_train_completed": bundle.n_train_completed,
        "centers": bundle.flight_model.centers.tolist(),
        "times": bundle.flight_model.times.tolist(),
        "n_per_bin": bundle.flight_model.n_per_bin.tolist(),
        "support_min": float(bundle.distance_support.distance_min_ft),
        "support_max": float(bundle.distance_support.distance_max_ft),
        "support_n": int(bundle.distance_support.n_attempts),
        "schema": 3,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:20]


def bundle_for_game(
    bundles: dict[int, FoldFlightBundle],
    game_id: int,
) -> FoldFlightBundle:
    if int(game_id) not in bundles:
        raise KeyError(f"no fold flight bundle for gameId={game_id}")
    return bundles[int(game_id)]


def predict_flight_time(model: FlightTimeModel, distance_ft: float | np.ndarray) -> np.ndarray:
    return model.predict(distance_ft)


def outside_distance_support(
    distance_ft: float | np.ndarray,
    support: DistanceSupport,
) -> np.ndarray:
    d = np.asarray(distance_ft, dtype=float)
    if not np.isfinite(support.distance_max_ft):
        return np.zeros(d.shape, dtype=bool)
    below = np.isfinite(d) & (d < support.distance_min_ft)
    above = np.isfinite(d) & (d > support.distance_max_ft)
    return below | above


def bundles_to_records(bundles: dict[int, FoldFlightBundle]) -> list[dict[str, Any]]:
    return [b.as_dict() for _, b in sorted(bundles.items())]


def bundle_flight_model_payload(model: FlightTimeModel) -> dict[str, Any]:
    """JSON-serializable flight-model summary (not the full interpolator)."""
    return {
        "n_train": int(model.n_train),
        "n_bins": int(model.centers.size),
        "distance_support_max": float(model.distance_support_max),
        "centers": model.centers.tolist(),
        "times": model.times.tolist(),
        "n_per_bin": model.n_per_bin.tolist(),
    }
