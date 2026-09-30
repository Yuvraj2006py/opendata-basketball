"""Required Stage 1 joins and post-catch chain scaffolding."""

from __future__ import annotations

import numpy as np
import pandas as pd


def join_pass_to_touch(passes: pd.DataFrame, touches: pd.DataFrame) -> pd.DataFrame:
    tcols = ["id", "gameId", "startFrame", "endFrame", "playerId", "chanceId", "possessionId"]
    t = touches[tcols].rename(
        columns={
            "id": "touchId",
            "startFrame": "touch_startFrame",
            "endFrame": "touch_endFrame",
            "playerId": "touch_playerId",
            "chanceId": "touch_chanceId",
            "possessionId": "touch_possessionId",
        }
    )
    out = passes.merge(t, on=["gameId", "touchId"], how="left")
    out["pass_within_touch"] = (
        out["touch_startFrame"].notna()
        & (out["startFrame"] >= out["touch_startFrame"])
        & (out["startFrame"] <= out["touch_endFrame"] + 1)  # one-frame tolerance
    )
    # Ordinary complete passes: release near touch end
    out["release_near_touch_end"] = (
        out["touch_endFrame"].notna()
        & ((out["touch_endFrame"] - out["startFrame"]).abs() <= 2)
    )
    out["passer_matches_touch_player"] = (
        out["passerId"].notna()
        & out["touch_playerId"].notna()
        & (out["passerId"] == out["touch_playerId"])
    )
    return out


EXPECTED_CANDIDATES_PER_TOUCH = 4


def ballhandler_in_offense(chance_off_player_ids, ballhandler_id) -> bool:
    """True when the identified ball-handler belongs to the offensive lineup.

    Some touches are credited to a defender inside an offensive chance (a
    deflection or steal), and a few name a player absent from both recorded
    lineups. Either way the chance lineup cannot yield four teammates, so these
    touches must leave the primary cohort rather than silently produce five
    candidates.
    """
    if ballhandler_id is None or (isinstance(ballhandler_id, float) and np.isnan(ballhandler_id)):
        return False
    if chance_off_player_ids is None:
        return False
    if isinstance(chance_off_player_ids, float) and np.isnan(chance_off_player_ids):
        return False
    return int(ballhandler_id) in {int(x) for x in chance_off_player_ids}


def candidate_lineup_from_chance(
    chance_off_player_ids,
    ballhandler_id: int,
) -> list[int]:
    """Four teammate candidates = chance.offPlayerIds excluding ball-handler."""
    if chance_off_player_ids is None:
        raw: list = []
    elif isinstance(chance_off_player_ids, float) and np.isnan(chance_off_player_ids):
        raw = []
    else:
        raw = list(chance_off_player_ids)
    ids = [int(x) for x in raw if x is not None and not (isinstance(x, float) and np.isnan(x))]
    seen: set[int] = set()
    out = []
    for p in ids:
        if p == int(ballhandler_id) or p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def build_touch_candidates(touches: pd.DataFrame, *, strict: bool = True) -> pd.DataFrame:
    """One row per (touch, candidate receiver) from chance lineup.

    With `strict`, every input touch must yield exactly four unique teammates;
    anything else is a lineup/ball-handler mismatch that must be excluded
    upstream instead of reaching Stage 2.
    """
    rows = []
    offenders = []
    for row in touches.itertuples(index=False):
        off_ids = getattr(row, "offPlayerIds", None)
        bh = getattr(row, "playerId", None)
        if bh is None or (isinstance(bh, float) and np.isnan(bh)):
            offenders.append((row.id, "missing_ballhandler", None))
            continue
        if not ballhandler_in_offense(off_ids, bh):
            offenders.append((row.id, "ballhandler_not_in_offense", int(bh)))
            continue
        cands = candidate_lineup_from_chance(off_ids, int(bh))
        if len(cands) != EXPECTED_CANDIDATES_PER_TOUCH:
            offenders.append((row.id, f"n_candidates={len(cands)}", int(bh)))
            continue
        for j, cid in enumerate(cands):
            rows.append(
                {
                    "gameId": int(row.gameId),
                    "touchId": row.id,
                    "chanceId": row.chanceId,
                    "ballhandlerId": int(bh),
                    "candidateId": int(cid),
                    "candidate_rank": j,
                    "n_candidates": len(cands),
                }
            )
    if strict and offenders:
        raise AssertionError(
            f"{len(offenders)} touches cannot form exactly "
            f"{EXPECTED_CANDIDATES_PER_TOUCH} teammate candidates: {offenders[:10]}"
        )
    return pd.DataFrame(rows)


MATCHUP_OUT_COLS = [
    "assigned_defenderId",
    "matchupId",
    "matchup_startFrame",
    "matchup_endFrame",
    "n_matchup_intervals",
    "matchup_matched",
    "matchup_ambiguous",
]


def interval_join_matchups(
    keys: pd.DataFrame,
    matchups: pd.DataFrame,
    *,
    frame_col: str = "frameIdx",
    off_player_col: str = "offPlayerId",
    prefix: str = "",
) -> pd.DataFrame:
    """Attach defensive assignment for each (game, chance, offPlayer, frame) key.

    `keys` must include gameId, chanceId, frame_col, and off_player_col. Row
    count and column order of `keys` are preserved; matchup columns are appended
    (optionally prefixed) so touch/candidate keys survive the join.

    Matchup intervals for one offensive player inside one chance frequently
    overlap (a switch is recorded before the previous assignment ends), so more
    than one interval can cover a frame. Ties are broken deterministically:
    latest `startFrame` first (most recent assignment), then the shortest
    interval, then the smallest `matchupId`. `n_matchup_intervals` records how
    many intervals covered the frame and `matchup_ambiguous` flags the frames
    where the choice was not unique.
    """
    out = keys.copy()
    named = [f"{prefix}{c}" for c in MATCHUP_OUT_COLS]
    if keys.empty or matchups.empty:
        for col, src in zip(named, MATCHUP_OUT_COLS, strict=True):
            if src in ("matchup_matched", "matchup_ambiguous"):
                out[col] = False
            elif src == "n_matchup_intervals":
                out[col] = 0
            else:
                out[col] = pd.NA
        return out

    m = matchups[["gameId", "chanceId", "offPlayerId", "startFrame", "endFrame", "defPlayerId", "id"]].rename(
        columns={
            "startFrame": "matchup_startFrame",
            "endFrame": "matchup_endFrame",
            "defPlayerId": "assigned_defenderId",
            "id": "matchupId",
        }
    )

    left = out[["gameId", "chanceId", off_player_col, frame_col]].copy()
    left["_key_row"] = np.arange(len(left))
    left = left.rename(columns={off_player_col: "offPlayerId"})

    merged = left.merge(m, on=["gameId", "chanceId", "offPlayerId"], how="inner")
    fr = merged[frame_col]
    covers = (fr >= merged["matchup_startFrame"]) & (fr <= merged["matchup_endFrame"])
    merged = merged[covers].copy()

    if merged.empty:
        for col, src in zip(named, MATCHUP_OUT_COLS, strict=True):
            if src in ("matchup_matched", "matchup_ambiguous"):
                out[col] = False
            elif src == "n_matchup_intervals":
                out[col] = 0
            else:
                out[col] = pd.NA
        return out

    merged["n_matchup_intervals"] = merged.groupby("_key_row")["matchupId"].transform("size")
    merged["_len"] = merged["matchup_endFrame"] - merged["matchup_startFrame"]
    merged["_neg_start"] = -merged["matchup_startFrame"]
    merged = merged.sort_values(
        ["_key_row", "_neg_start", "_len", "matchupId"], kind="mergesort"
    )
    chosen = merged.drop_duplicates(subset=["_key_row"], keep="first").set_index("_key_row")

    out[f"{prefix}assigned_defenderId"] = chosen["assigned_defenderId"].reindex(range(len(out))).to_numpy()
    out[f"{prefix}matchupId"] = chosen["matchupId"].reindex(range(len(out))).to_numpy()
    out[f"{prefix}matchup_startFrame"] = chosen["matchup_startFrame"].reindex(range(len(out))).to_numpy()
    out[f"{prefix}matchup_endFrame"] = chosen["matchup_endFrame"].reindex(range(len(out))).to_numpy()
    n_int = chosen["n_matchup_intervals"].reindex(range(len(out))).fillna(0).astype(int).to_numpy()
    out[f"{prefix}n_matchup_intervals"] = n_int
    out[f"{prefix}matchup_matched"] = n_int > 0
    out[f"{prefix}matchup_ambiguous"] = n_int > 1
    return out


def build_candidate_states(
    model_frames: pd.DataFrame,
    candidates: pd.DataFrame,
    matchups: pd.DataFrame,
    players: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per (touch, sampled frame, candidate receiver) with assignments.

    This is the materialized candidate-frame layer the Stage 1 plan requires:
    touch membership x chance lineup x defensive-assignment interval join.
    """
    if model_frames.empty or candidates.empty:
        return pd.DataFrame()

    cand = candidates[
        ["gameId", "touchId", "chanceId", "ballhandlerId", "candidateId", "candidate_rank"]
    ]
    frame_cols = [
        c
        for c in [
            "gameId",
            "touchId",
            "chanceId",
            "possessionId",
            "frameIdx",
            "period",
            "gameClock",
            "shotClock",
            "ballhandlerId",
            "primary_frame_eligible",
        ]
        if c in model_frames.columns
    ]
    states = model_frames[frame_cols].merge(
        cand.drop(columns=["ballhandlerId"]), on=["gameId", "touchId", "chanceId"], how="inner"
    )
    states = states.sort_values(["gameId", "touchId", "frameIdx", "candidate_rank"]).reset_index(
        drop=True
    )

    states = interval_join_matchups(
        states, matchups, frame_col="frameIdx", off_player_col="candidateId"
    )

    if players is not None and len(players):
        pcols = [
            c
            for c in ["gameId", "frameIdx", "playerId", "x_event", "y_event", "vx", "vy", "speed_causal", "velocity_ok", "predError"]
            if c in players.columns
        ]
        subset = [c for c in ["gameId", "frameIdx", "playerId"] if c in pcols]
        if "frameIdx" not in subset or "playerId" not in subset:
            subset = [c for c in ["frameIdx", "playerId"] if c in pcols]
        pl = players[pcols].drop_duplicates(subset=subset)
        rename_map = {
            "playerId": "candidateId",
            "x_event": "candidate_x_event",
            "y_event": "candidate_y_event",
            "vx": "candidate_vx",
            "vy": "candidate_vy",
            "speed_causal": "candidate_speed_causal",
            "velocity_ok": "candidate_velocity_ok",
            "predError": "candidate_predError",
        }
        pl = pl.rename(columns={k: v for k, v in rename_map.items() if k in pl.columns})
        on_cols = [c for c in ["gameId", "frameIdx", "candidateId"] if c in pl.columns and c in states.columns]
        if "frameIdx" not in on_cols or "candidateId" not in on_cols:
            on_cols = [c for c in ["frameIdx", "candidateId"] if c in pl.columns and c in states.columns]
        states = states.merge(pl, on=on_cols, how="left")
        states["candidate_tracked"] = states["candidate_x_event"].notna()
    return states


def build_post_catch_chain(
    passes: pd.DataFrame,
    touches: pd.DataFrame,
    shots: pd.DataFrame,
    free_throws: pd.DataFrame,
) -> pd.DataFrame:
    """Completed pass → receiver next touch → subsequent events in same chance.

    Stores join keys and validation flags for Stage 4; does not fit models.
    """
    completed = passes[
        passes["complete"].astype(bool)
        & passes["receiverId"].notna()
        & passes["touchId"].notna()
        & ~passes.get("is_self_pass", False)
    ].copy()
    if "is_self_pass" in passes.columns:
        completed = passes[
            passes["complete"].astype(bool)
            & passes["receiverId"].notna()
            & passes["touchId"].notna()
            & ~passes["is_self_pass"].astype(bool)
        ].copy()

    t = touches[
        ["id", "gameId", "chanceId", "playerId", "startFrame", "endFrame"]
    ].rename(
        columns={
            "id": "receiver_touchId",
            "playerId": "receiver_touch_playerId",
            "startFrame": "receiver_touch_startFrame",
            "endFrame": "receiver_touch_endFrame",
        }
    )

    rows = []
    touches_by_chance = {
        (int(r.gameId), r.chanceId): touches[
            (touches["gameId"] == r.gameId) & (touches["chanceId"] == r.chanceId)
        ]
        for r in completed[["gameId", "chanceId"]].drop_duplicates().itertuples(index=False)
    }

    for row in completed.itertuples(index=False):
        key = (int(row.gameId), row.chanceId)
        ch_touches = touches_by_chance.get(key)
        next_touch_id = None
        next_start = None
        next_end = None
        if ch_touches is not None and len(ch_touches):
            cand = ch_touches[
                (ch_touches["playerId"] == row.receiverId)
                & (ch_touches["startFrame"] >= row.endFrame - 1)
            ].sort_values("startFrame")
            if len(cand):
                nt = cand.iloc[0]
                next_touch_id = nt["id"]
                next_start = int(nt["startFrame"])
                next_end = int(nt["endFrame"])

        # Subsequent shot / FT in same chance after catch
        next_shot_id = None
        next_ft_id = None
        catch_frame = int(row.endFrame) if pd.notna(row.endFrame) else None
        if catch_frame is not None:
            sh = shots[
                (shots["gameId"] == row.gameId)
                & (shots["chanceId"] == row.chanceId)
                & (shots["startFrame"] >= catch_frame)
            ].sort_values("startFrame")
            if len(sh):
                next_shot_id = sh.iloc[0]["id"]
            ft = free_throws[
                (free_throws["gameId"] == row.gameId)
                & (free_throws["chanceId"] == row.chanceId)
                & (free_throws["frame"] >= catch_frame)
            ].sort_values("frame")
            if len(ft):
                next_ft_id = ft.iloc[0]["id"]

        rows.append(
            {
                "gameId": int(row.gameId),
                "passId": row.id,
                "chanceId": row.chanceId,
                "possessionId": row.possessionId,
                "passer_touchId": row.touchId,
                "pass_startFrame": int(row.startFrame),
                "pass_endFrame": int(row.endFrame) if pd.notna(row.endFrame) else pd.NA,
                "passerId": int(row.passerId) if pd.notna(row.passerId) else pd.NA,
                "receiverId": int(row.receiverId),
                "receiver_next_touchId": next_touch_id,
                "receiver_touch_startFrame": next_start,
                "receiver_touch_endFrame": next_end,
                "next_shotId": next_shot_id,
                "next_freeThrowId": next_ft_id,
                "has_receiver_touch": next_touch_id is not None,
            }
        )
    return pd.DataFrame(rows)


def clock_agreement(
    event_frame: int,
    tracking_frame_idx: int,
    event_game_clock: float | None,
    tracking_game_clock: float | None,
    event_wall_clock: int | None = None,
    tracking_wall_clock: int | None = None,
    frame_tol: int = 1,
    clock_tol: float = 0.05,
) -> dict:
    """Check event vs tracking clocks within one-frame tolerance."""
    frame_ok = abs(int(event_frame) - int(tracking_frame_idx)) <= frame_tol
    clock_ok = True
    if event_game_clock is not None and tracking_game_clock is not None:
        if not (pd.isna(event_game_clock) or pd.isna(tracking_game_clock)):
            clock_ok = abs(float(event_game_clock) - float(tracking_game_clock)) <= clock_tol
    wall_ok = True
    if event_wall_clock is not None and tracking_wall_clock is not None:
        if not (pd.isna(event_wall_clock) or pd.isna(tracking_wall_clock)):
            # wallClock in ms; one frame ~40ms
            wall_ok = abs(int(event_wall_clock) - int(tracking_wall_clock)) <= 50
    return {
        "frame_ok": frame_ok,
        "game_clock_ok": clock_ok,
        "wall_clock_ok": wall_ok,
        "ok": frame_ok and clock_ok and wall_ok,
    }
