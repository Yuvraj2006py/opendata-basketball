"""Normalize event tables with context, derived fields, and integrity flags."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .coords import EVENT_ATTACK_SIGN, normalize_attack_xy, verify_possession_hoop_orientation
from .joins import ballhandler_in_offense


def _list_len(val) -> int:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0
    if isinstance(val, (list, tuple, np.ndarray)):
        return len(val)
    return 0


def normalize_possessions(possessions: pd.DataFrame, shots: pd.DataFrame) -> pd.DataFrame:
    """Attach hoop-orientation audit and attack-direction metadata."""
    df = possessions.copy()
    df["n_off_ids"] = df["offPlayerIds"].map(_list_len)
    df["n_def_ids"] = df["defPlayerIds"].map(_list_len)
    df["substitution_boundary"] = (df["n_off_ids"] > 5) | (df["n_def_ids"] > 5)
    df["frame_order_ok"] = df["endFrame"] >= df["startFrame"]
    df["elapsed_frames"] = df["endFrame"] - df["startFrame"]
    df["elapsed_game_clock"] = df["startGameClock"] - df["endGameClock"]
    df["attack_sign"] = EVENT_ATTACK_SIGN  # events already normalized

    # Hoop orientation check from shot locations in this possession
    shot_xs_by_poss: dict[str, list[float]] = {}
    if len(shots) and "possessionId" in shots.columns:
        for row in shots.itertuples(index=False):
            loc = getattr(row, "location", None)
            pid = getattr(row, "possessionId", None)
            if pid is None or loc is None:
                continue
            if isinstance(loc, (list, tuple, np.ndarray)) and len(loc) >= 1:
                shot_xs_by_poss.setdefault(str(pid), []).append(float(loc[0]))

    audits = []
    for row in df.itertuples(index=False):
        xs = shot_xs_by_poss.get(str(row.id), [])
        audits.append(
            verify_possession_hoop_orientation(
                xs,
                bool(row.leftHoop) if pd.notna(getattr(row, "leftHoop", None)) else None,
            )
        )
    audit_df = pd.DataFrame(audits)
    for col in audit_df.columns:
        df[f"hoop_{col}"] = audit_df[col].values
    return df


def normalize_chances(chances: pd.DataFrame, possessions: pd.DataFrame) -> pd.DataFrame:
    df = chances.copy()
    df["n_off_ids"] = df["offPlayerIds"].map(_list_len)
    df["n_def_ids"] = df["defPlayerIds"].map(_list_len)
    df["lineup_5v5"] = (df["n_off_ids"] == 5) & (df["n_def_ids"] == 5)
    df["frame_order_ok"] = df["endFrame"] >= df["startFrame"]
    df["elapsed_frames"] = df["endFrame"] - df["startFrame"]
    df["elapsed_game_clock"] = df["startGameClock"] - df["endGameClock"]
    df["zero_elapsed_clock"] = df["elapsed_game_clock"] <= 0
    df["has_frontcourt_frame"] = df["frontcourtFrame"].notna()
    # Prefer boolean usable; also derive from qualityIndex
    if "usable" in df.columns:
        df["usable_flag"] = df["usable"].astype(bool)
    else:
        df["usable_flag"] = df["qualityIndex"] >= 3
    df["usable_matches_qi"] = df["usable_flag"] == (df["qualityIndex"] >= 3)

    poss_cols = [
        "id",
        "leftHoop",
        "substitution_boundary",
        "attack_sign",
        "hoop_orientation_ok",
        "hoop_frac_attacking_side",
        "hoop_median_x",
    ]
    poss_cols = [c for c in poss_cols if c in possessions.columns]
    poss_small = possessions[poss_cols].rename(columns={"id": "possessionId"})
    df = df.merge(poss_small, on="possessionId", how="left", suffixes=("", "_poss"))
    return df


def normalize_touches(touches: pd.DataFrame, chances: pd.DataFrame) -> pd.DataFrame:
    df = touches.copy()
    df["frame_order_ok"] = df["endFrame"] >= df["startFrame"]
    df["elapsed_frames"] = df["endFrame"] - df["startFrame"]
    df["duration_s_from_frames"] = df["elapsed_frames"] / 25.0
    df["touchTime_abs_diff"] = (df["touchTime"] - df["duration_s_from_frames"]).abs()
    df["has_ballhandler"] = df["playerId"].notna()

    chance_cols = [
        "id",
        "usable_flag",
        "transition",
        "frontcourtFrame",
        "qualityIndex",
        "lineup_5v5",
        "frame_order_ok",
        "zero_elapsed_clock",
        "substitution_boundary",
        "offPlayerIds",
        "defPlayerIds",
        "offTeamId",
        "defTeamId",
        "homeStartScore",
        "awayStartScore",
        "period",
        "leftHoop",
        "attack_sign",
        "hoop_orientation_ok",
        "startGameClock",
        "endGameClock",
        "startShotClock",
        "endShotClock",
        "has_frontcourt_frame",
    ]
    available = [c for c in chance_cols if c in chances.columns]
    ch = chances[available].copy()
    ch = ch.rename(
        columns={
            "id": "chanceId",
            "frame_order_ok": "chance_frame_order_ok",
            "offTeamId": "chance_offTeamId",
            "defTeamId": "chance_defTeamId",
            "period": "chance_period",
            "startGameClock": "chance_startGameClock",
            "endGameClock": "chance_endGameClock",
        }
    )
    df = df.merge(ch, on="chanceId", how="left", suffixes=("", "_ch"))
    if "has_frontcourt_frame" not in df.columns and "frontcourtFrame" in df.columns:
        df["has_frontcourt_frame"] = df["frontcourtFrame"].notna()

    # The chance lineup must contain the ball-handler, otherwise the four
    # teammate candidates cannot be formed (see joins.ballhandler_in_offense).
    if "offPlayerIds" in df.columns:
        df["ballhandler_in_offense"] = [
            ballhandler_in_offense(off, bh)
            for off, bh in zip(df["offPlayerIds"], df["playerId"], strict=True)
        ]
    else:
        df["ballhandler_in_offense"] = False
    return df


def normalize_passes(passes: pd.DataFrame) -> pd.DataFrame:
    df = passes.copy()
    df["frame_order_ok"] = df["endFrame"] >= df["startFrame"]
    df["elapsed_frames"] = df["endFrame"] - df["startFrame"]
    df["is_self_pass"] = (
        df["passerId"].notna()
        & df["receiverId"].notna()
        & (df["passerId"] == df["receiverId"])
    )
    df["has_interceptor"] = df["toReceiverId"].notna()
    # Frozen rule: toReceiverId is interceptor, never intended target
    df["interceptorId"] = df["toReceiverId"]

    # `complete` is tri-state in the raw feed: True, False, or null. Coercing it
    # with astype(bool) silently merges the nulls into the failed-pass cohort,
    # but the two groups carry different evidence, so keep them separate.
    df["complete_flag"] = df["complete"].map(
        lambda v: pd.NA if v is None or (isinstance(v, float) and np.isnan(v)) else bool(v)
    ).astype("boolean")
    df["pass_outcome_class"] = np.where(
        df["complete_flag"].isna(),
        "unknown_outcome",
        np.where(df["complete_flag"].fillna(False), "complete", "incomplete_turnover"),
    )
    df["has_endFrame"] = df["endFrame"].notna()
    df["has_receiver_evidence"] = df["receiverLoc"].notna() & df["receiverId"].notna()
    # Stage 2 target-inference population: a failed pass with a recorded end
    # frame. `unknown_outcome` passes have no endFrame, receiver, or interceptor.
    df["is_failed_pass"] = (df["pass_outcome_class"] == "incomplete_turnover")
    df["failed_pass_inference_eligible"] = df["is_failed_pass"] & df["has_endFrame"]
    df["intended_target_unknown"] = (~df["complete_flag"].fillna(False)) | df["turnover"].astype(bool)
    # Distance recomputation from locations when present
    dists = []
    for row in df.itertuples(index=False):
        pl, rl = getattr(row, "passerLoc", None), getattr(row, "receiverLoc", None)
        if (
            isinstance(pl, (list, tuple, np.ndarray))
            and isinstance(rl, (list, tuple, np.ndarray))
            and len(pl) >= 2
            and len(rl) >= 2
        ):
            dists.append(float(np.hypot(float(pl[0]) - float(rl[0]), float(pl[1]) - float(rl[1]))))
        else:
            dists.append(np.nan)
    df["distance_recomputed"] = dists
    df["distance_abs_diff"] = (df["distance"] - df["distance_recomputed"]).abs()

    # Normalized attack coords for passer/receiver locs
    px_n, py_n, rx_n, ry_n = [], [], [], []
    for row in df.itertuples(index=False):
        pl, rl = getattr(row, "passerLoc", None), getattr(row, "receiverLoc", None)
        if isinstance(pl, (list, tuple, np.ndarray)) and len(pl) >= 2:
            x, y = normalize_attack_xy(float(pl[0]), float(pl[1]))
            px_n.append(x)
            py_n.append(y)
        else:
            px_n.append(np.nan)
            py_n.append(np.nan)
        if isinstance(rl, (list, tuple, np.ndarray)) and len(rl) >= 2:
            x, y = normalize_attack_xy(float(rl[0]), float(rl[1]))
            rx_n.append(x)
            ry_n.append(y)
        else:
            rx_n.append(np.nan)
            ry_n.append(np.nan)
    df["passer_x_norm"] = px_n
    df["passer_y_norm"] = py_n
    df["receiver_x_norm"] = rx_n
    df["receiver_y_norm"] = ry_n
    return df


def points_from_shots_and_ft(shots: pd.DataFrame, free_throws: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct points from shots + free_throws only (never chances.ptsScored).

    Shot points: made FG => 3 if three else 2; fouled makes still count FG points.
    FT points: outcome True => 1.
    Grain: one row per (gameId, chanceId) with points_recon.
    """
    rows = []
    if len(shots):
        s = shots.copy()
        s["pts"] = np.where(
            s["outcome"].astype(bool),
            np.where(s["three"].astype(bool), 3, 2),
            0,
        )
        g = s.groupby(["gameId", "chanceId"], dropna=False)["pts"].sum().reset_index(name="shot_pts")
        rows.append(g)
    if len(free_throws):
        ft = free_throws.copy()
        ft["pts"] = np.where(ft["outcome"].astype(bool), 1, 0)
        g = ft.groupby(["gameId", "chanceId"], dropna=False)["pts"].sum().reset_index(name="ft_pts")
        rows.append(g)
    if not rows:
        return pd.DataFrame(columns=["gameId", "chanceId", "shot_pts", "ft_pts", "points_recon"])
    out = rows[0]
    for r in rows[1:]:
        out = out.merge(r, on=["gameId", "chanceId"], how="outer")
    if "shot_pts" not in out.columns:
        out["shot_pts"] = 0
    if "ft_pts" not in out.columns:
        out["ft_pts"] = 0
    out["shot_pts"] = out["shot_pts"].fillna(0).astype(int)
    out["ft_pts"] = out["ft_pts"].fillna(0).astype(int)
    out["points_recon"] = out["shot_pts"] + out["ft_pts"]
    return out


def audit_multipass_touches(passes: pd.DataFrame) -> pd.DataFrame:
    """Identify touches with >1 pass; Stage 1 policy = exclude from primary."""
    p = passes[passes["touchId"].notna()].copy()
    counts = p.groupby(["gameId", "touchId"]).size().reset_index(name="n_passes")
    multi = counts[counts["n_passes"] > 1].copy()
    multi["policy"] = "exclude_from_primary"
    return multi
