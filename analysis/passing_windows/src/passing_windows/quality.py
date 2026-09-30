"""Eligibility filters, exclusion cascade, and CONSORT-style sample flow."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .joins import ballhandler_in_offense

# Ordered exclusion reasons for CONSORT (first-match wins)
EXCLUSION_ORDER = [
    "missing_chance",
    "malformed_chance_frame_order",
    "malformed_touch_frame_order",
    "usable_false",
    "transition_true",
    "zero_elapsed_clock",
    "missing_frontcourt_frame",
    "lineup_not_5v5",
    "substitution_boundary",
    "missing_ballhandler",
    "ballhandler_not_in_offense",
    "self_pass_touch",
    "multipass_touch",
    "inbounds_touch",
    "null_touch_pass_only",  # reserved
]


def annotate_pass_touch_flags(
    touches: pd.DataFrame,
    passes: pd.DataFrame,
    multipass: pd.DataFrame,
) -> pd.DataFrame:
    """Attach pass-derived flags onto touches."""
    df = touches.copy()
    p = passes[passes["touchId"].notna()].copy()

    self_touch_ids = set(
        zip(
            p.loc[p.get("is_self_pass", False) == True, "gameId"],  # noqa: E712
            p.loc[p.get("is_self_pass", False) == True, "touchId"],  # noqa: E712
        )
    ) if "is_self_pass" in p.columns else set()
    if "is_self_pass" in p.columns:
        self_touch_ids = set(
            (int(g), t)
            for g, t in p.loc[p["is_self_pass"].astype(bool), ["gameId", "touchId"]].itertuples(
                index=False, name=None
            )
        )
    else:
        self_touch_ids = set()

    inbound_touch_ids = set(
        (int(g), t)
        for g, t in p.loc[p["inbounds"].astype(bool), ["gameId", "touchId"]].itertuples(
            index=False, name=None
        )
    ) if "inbounds" in p.columns else set()

    multi_ids = set(
        (int(g), t)
        for g, t in multipass[["gameId", "touchId"]].itertuples(index=False, name=None)
    ) if len(multipass) else set()

    df["is_self_pass_touch"] = [
        (int(g), t) in self_touch_ids
        for g, t in zip(df["gameId"], df["id"], strict=True)
    ]
    df["is_inbounds_touch"] = [
        (int(g), t) in inbound_touch_ids
        for g, t in zip(df["gameId"], df["id"], strict=True)
    ]
    df["is_multipass_touch"] = [
        (int(g), t) in multi_ids
        for g, t in zip(df["gameId"], df["id"], strict=True)
    ]
    return df


def primary_exclusion_reason(row: pd.Series) -> str | None:
    """Return first failing primary-population reason, or None if eligible at touch level."""
    if pd.isna(row.get("chanceId")) or pd.isna(row.get("usable_flag")):
        return "missing_chance"
    if row.get("chance_frame_order_ok") is False or (
        pd.notna(row.get("chance_frame_order_ok")) and not bool(row.get("chance_frame_order_ok"))
    ):
        return "malformed_chance_frame_order"
    if row.get("frame_order_ok") is False or not bool(row.get("frame_order_ok", True)):
        return "malformed_touch_frame_order"
    if not bool(row.get("usable_flag", False)):
        return "usable_false"
    if bool(row.get("transition", False)):
        return "transition_true"
    if bool(row.get("zero_elapsed_clock", False)):
        return "zero_elapsed_clock"
    if not bool(row.get("has_frontcourt_frame", False)) and pd.isna(row.get("frontcourtFrame")):
        return "missing_frontcourt_frame"
    # has_frontcourt_frame may be derived
    if "frontcourtFrame" in row and pd.isna(row.get("frontcourtFrame")):
        return "missing_frontcourt_frame"
    if row.get("lineup_5v5") is False or (
        pd.notna(row.get("lineup_5v5")) and not bool(row.get("lineup_5v5"))
    ):
        return "lineup_not_5v5"
    if bool(row.get("substitution_boundary", False)):
        return "substitution_boundary"
    if not bool(row.get("has_ballhandler", True)) or pd.isna(row.get("playerId")):
        return "missing_ballhandler"
    if not ballhandler_in_offense(row.get("offPlayerIds"), row.get("playerId")):
        return "ballhandler_not_in_offense"
    if bool(row.get("is_self_pass_touch", False)):
        return "self_pass_touch"
    if bool(row.get("is_multipass_touch", False)):
        return "multipass_touch"
    if bool(row.get("is_inbounds_touch", False)):
        return "inbounds_touch"
    return None


def classify_touches(touches: pd.DataFrame) -> pd.DataFrame:
    df = touches.copy()
    reasons = [primary_exclusion_reason(row) for _, row in df.iterrows()]
    df["exclusion_reason"] = reasons
    df["primary_touch_eligible"] = [r is None for r in reasons]
    return df


def consort_flow(touches: pd.DataFrame) -> pd.DataFrame:
    """CONSORT-style cumulative exclusion counts overall and by game."""
    n_all = len(touches)
    out = []
    rem = n_all
    out.append({"stage": "all_touches", "gameId": "ALL", "n_remaining": rem, "n_excluded_here": 0})
    for reason in EXCLUSION_ORDER:
        n_ex = int((touches["exclusion_reason"] == reason).sum())
        if n_ex == 0 and reason not in {
            "usable_false",
            "transition_true",
            "inbounds_touch",
            "self_pass_touch",
            "multipass_touch",
            "substitution_boundary",
            "malformed_chance_frame_order",
            "zero_elapsed_clock",
            "missing_frontcourt_frame",
            "lineup_not_5v5",
            "missing_ballhandler",
            "ballhandler_not_in_offense",
            "malformed_touch_frame_order",
            "missing_chance",
        }:
            continue
        rem -= n_ex
        out.append(
            {
                "stage": f"after_exclude_{reason}",
                "gameId": "ALL",
                "n_remaining": rem,
                "n_excluded_here": n_ex,
            }
        )
    out.append(
        {
            "stage": "primary_touch_eligible",
            "gameId": "ALL",
            "n_remaining": int(touches["primary_touch_eligible"].sum()),
            "n_excluded_here": 0,
        }
    )

    # Per-game exclusion tallies (gameId as string for parquet homogeneity)
    for gid, gdf in touches.groupby("gameId"):
        gid_s = str(int(gid))
        out.append(
            {
                "stage": "all_touches",
                "gameId": gid_s,
                "n_remaining": len(gdf),
                "n_excluded_here": 0,
            }
        )
        for reason in EXCLUSION_ORDER:
            n_ex = int((gdf["exclusion_reason"] == reason).sum())
            if n_ex:
                out.append(
                    {
                        "stage": f"excluded_{reason}",
                        "gameId": gid_s,
                        "n_remaining": int(gdf["primary_touch_eligible"].sum()),
                        "n_excluded_here": n_ex,
                    }
                )
        out.append(
            {
                "stage": "primary_touch_eligible",
                "gameId": gid_s,
                "n_remaining": int(gdf["primary_touch_eligible"].sum()),
                "n_excluded_here": 0,
            }
        )
    return pd.DataFrame(out)


def exclusion_concentration(
    touches: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Check exclusions not concentrated by period / score / team / shot-clock."""
    df = touches.copy()
    if "homeStartScore" in df.columns and "awayStartScore" in df.columns:
        df["score_diff_bin"] = pd.cut(
            df["homeStartScore"].fillna(0) - df["awayStartScore"].fillna(0),
            bins=[-100, -15, -5, 5, 15, 100],
            labels=["<=-15", "-15:-5", "-5:5", "5:15", ">=15"],
        )
    else:
        df["score_diff_bin"] = pd.NA

    if "shotClock" in df.columns:
        df["shot_clock_bin"] = pd.cut(
            df["shotClock"].astype(float),
            bins=[-0.1, 2, 7, 14, 24.1],
            labels=["0-2", "2-7", "7-14", "14-24"],
        )
    else:
        df["shot_clock_bin"] = pd.NA

    excl = df[~df["primary_touch_eligible"]].copy()

    def _rate_table(col: str) -> pd.DataFrame:
        if col not in df.columns:
            return pd.DataFrame()
        all_c = df.groupby(col, observed=False).size().rename("n_all")
        ex_c = excl.groupby(col, observed=False).size().rename("n_excluded")
        tab = pd.concat([all_c, ex_c], axis=1).fillna(0)
        tab["n_excluded"] = tab["n_excluded"].astype(int)
        tab["excl_rate"] = tab["n_excluded"] / tab["n_all"].clip(lower=1)
        return tab.reset_index()

    return {
        "by_period": _rate_table("period"),
        "by_off_team": _rate_table("offTeamId"),
        "by_score_diff": _rate_table("score_diff_bin"),
        "by_shot_clock": _rate_table("shot_clock_bin"),
        "by_reason": df.groupby("exclusion_reason", dropna=False)
        .size()
        .reset_index(name="n"),
    }


def frame_eligibility_flags(
    frame_idx: int,
    *,
    game_clock_stopped: bool,
    frontcourt_frame: int | float | None,
    ten_players: bool,
    ballhandler_present: bool,
    ballhandler_x: float | None = None,
) -> dict[str, bool]:
    """Frame-level primary filters (applied after touch eligibility)."""
    live_clock = not bool(game_clock_stopped)
    if frontcourt_frame is None or (isinstance(frontcourt_frame, float) and np.isnan(frontcourt_frame)):
        in_frontcourt = False
    else:
        in_frontcourt = int(frame_idx) >= int(frontcourt_frame)
        if ballhandler_x is not None and not (isinstance(ballhandler_x, float) and np.isnan(ballhandler_x)):
            # Event convention: frontcourt offense x <= ~0
            in_frontcourt = in_frontcourt and (float(ballhandler_x) <= 0.5)
    ok = live_clock and in_frontcourt and ten_players and ballhandler_present
    return {
        "live_clock": live_clock,
        "frontcourt": in_frontcourt,
        "ten_players": ten_players,
        "ballhandler_present": ballhandler_present,
        "primary_frame_eligible": ok,
    }


def sample_model_frames(start_frame: int, end_frame: int, stride: int = 5) -> list[int]:
    """5 Hz sampling on 25 Hz timeline: every `stride` frames within touch."""
    if end_frame < start_frame:
        return []
    return list(range(int(start_frame), int(end_frame) + 1, int(stride)))
