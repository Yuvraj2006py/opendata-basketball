#!/usr/bin/env python
"""Stage 1: build canonical event/tracking tables, eligibility, and audits.

Restartable: skips games whose per-game parquet markers exist unless --force.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running without install: add src to path
STAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE_ROOT / "src"))

from passing_windows.aliases import load_alias_map, load_alias_table  # noqa: E402
from passing_windows.audit import (  # noqa: E402
    audit_candidate_sets,
    audit_causal_velocity,
    audit_causal_velocity_boundaries,
    audit_event_tracking_clocks,
    audit_invariants_primary,
    audit_matchup_interval_join,
    audit_mirroring_symmetry,
    audit_output_manifest,
    audit_pass_outcome_classes,
    audit_pass_release_vs_touch_end,
    audit_points_reconstruction,
    audit_receiver_loc_vs_tracking,
    audit_synthetic_lane_geometry,
    spot_check_calculations_one_game,
)
from passing_windows.coords import broadcast_to_event_xy, normalize_attack_xy  # noqa: E402
from passing_windows.ingest import (  # noqa: E402
    attach_player_aliases,
    events_to_tables,
    game_ids_from_freeze,
    iter_tracking_frames,
    load_freeze,
    load_game_data,
    tracking_frame_is_live,
)
from passing_windows.io_utils import (  # noqa: E402
    build_output_manifest,
    refresh_sign_off_hashes,
    verify_output_manifest,
    write_json,
    write_markdown,
    write_table,
)
from passing_windows.joins import (  # noqa: E402
    build_candidate_states,
    build_post_catch_chain,
    build_touch_candidates,
    interval_join_matchups,
    join_pass_to_touch,
)
from passing_windows.normalize import (  # noqa: E402
    audit_multipass_touches,
    normalize_chances,
    normalize_passes,
    normalize_possessions,
    normalize_touches,
    points_from_shots_and_ft,
)
from passing_windows.paths import ARTIFACTS, PACKAGE_ROOT, TABLES  # noqa: E402
from passing_windows.quality import (  # noqa: E402
    annotate_pass_touch_flags,
    classify_touches,
    consort_flow,
    exclusion_concentration,
    frame_eligibility_flags,
    sample_model_frames,
)
from passing_windows.velocities import (  # noqa: E402
    MAX_CAUSAL_FRAME_GAP,
    MAX_PLAUSIBLE_PLAYER_SPEED_FPS,
    add_causal_velocities,
)


def _game_table_dir(game_id: int) -> Path:
    return TABLES / "by_game" / str(game_id)


# Stage 2 reconstructs ball flight for failed passes, so Stage 1 retains the
# full 25 Hz window around every pass rather than only its endpoints.
PASS_WINDOW_FRAMES = 25  # +/- 1.0 s at 25 Hz: covers an ordinary pass flight
# Non-complete passes have no endFrame at all in the raw feed, so their release
# window is widened to cover the flight without an end anchor.
FAILED_PASS_WINDOW_FRAMES = 50  # +/- 2.0 s at 25 Hz


def _needed_frames_for_game(
    touches: pd.DataFrame,
    passes: pd.DataFrame,
    shots: pd.DataFrame,
    stride: int = 5,
) -> tuple[set[int], set[int], set[int]]:
    """Frames needed for joins/audits/model scaffolding.

    Returns (all needed frames, model-sample frames, pass-window frames).
    """
    model_frames: set[int] = set()
    pass_window: set[int] = set()
    for row in touches.itertuples(index=False):
        if not bool(getattr(row, "primary_touch_eligible", False)):
            continue
        model_frames.update(
            sample_model_frames(int(row.startFrame), int(row.endFrame), stride=stride)
        )
        # also keep endpoints
        model_frames.add(int(row.startFrame))
        model_frames.add(int(row.endFrame))
    for row in passes.itertuples(index=False):
        failed = str(getattr(row, "pass_outcome_class", "complete")) != "complete"
        half = FAILED_PASS_WINDOW_FRAMES if failed else PASS_WINDOW_FRAMES
        if pd.notna(row.startFrame):
            s = int(row.startFrame)
            pass_window.update(range(max(s - half, 0), s + half + 1))
        if pd.notna(getattr(row, "endFrame", None)):
            e = int(row.endFrame)
            pass_window.update(range(max(e - PASS_WINDOW_FRAMES, 0), e + PASS_WINDOW_FRAMES + 1))
    other: set[int] = set()
    for row in shots.itertuples(index=False):
        if pd.notna(row.startFrame):
            other.add(int(row.startFrame))
    return model_frames | pass_window | other, model_frames, pass_window


def _left_hoop_lookup(possessions: pd.DataFrame) -> list[tuple[int, int, bool]]:
    """Return sorted (startFrame, endFrame, leftHoop) intervals for a game."""
    rows = []
    for row in possessions.itertuples(index=False):
        if pd.isna(row.startFrame) or pd.isna(row.endFrame):
            continue
        lh = bool(row.leftHoop) if pd.notna(getattr(row, "leftHoop", None)) else True
        rows.append((int(row.startFrame), int(row.endFrame), lh))
    rows.sort(key=lambda r: r[0])
    return rows


def _left_hoop_at(frame_idx: int, intervals: list[tuple[int, int, bool]]) -> bool | None:
    for start, end, lh in intervals:
        if start <= frame_idx <= end:
            return lh
    return None


def _process_tracking(
    game_id: int,
    needed: set[int],
    possessions: pd.DataFrame,
    model_sample_frames: set[int] | None = None,
    pass_window_frames: set[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Stream tracking; keep needed frames fully; summarize quality on all frames."""
    quality = {
        "gameId": game_id,
        "n_frames_total": 0,
        "n_live": 0,
        "n_dead": 0,
        "n_ten_players": 0,
        "n_clock_stopped": 0,
        "n_clock_running": 0,
        "predError_sum": 0.0,
        "predError_n": 0,
        "ball_predError_sum": 0.0,
        "ball_predError_n": 0,
    }
    frame_rows = []
    player_rows = []
    intervals = _left_hoop_lookup(possessions)
    model_sample_frames = model_sample_frames or set()
    pass_window_frames = pass_window_frames or set()

    for frame in iter_tracking_frames(game_id):
        quality["n_frames_total"] += 1
        live = tracking_frame_is_live(frame)
        if live:
            quality["n_live"] += 1
        else:
            quality["n_dead"] += 1
        if frame.get("gameClockStopped"):
            quality["n_clock_stopped"] += 1
        else:
            quality["n_clock_running"] += 1

        home = frame.get("homePlayers") or []
        away = frame.get("awayPlayers") or []
        if len(home) == 5 and len(away) == 5:
            quality["n_ten_players"] += 1
        for p in home + away:
            pe = p.get("predError")
            if pe is not None:
                quality["predError_sum"] += float(pe)
                quality["predError_n"] += 1
        ball = frame.get("ball") or {}
        if ball.get("predError") is not None:
            quality["ball_predError_sum"] += float(ball["predError"])
            quality["ball_predError_n"] += 1

        fi = int(frame["frameIdx"])
        if fi not in needed:
            continue

        left_hoop = _left_hoop_at(fi, intervals)
        n_home, n_away = len(home), len(away)
        ball_xyz = ball.get("xyz")
        ball_x = ball_y = ball_z = None
        ball_x_event = ball_y_event = None
        if ball_xyz:
            ball_x, ball_y, ball_z = ball_xyz[0], ball_xyz[1], ball_xyz[2]
            if left_hoop is not None and ball_x is not None and ball_y is not None:
                ball_x_event, ball_y_event = broadcast_to_event_xy(
                    float(ball_x), float(ball_y), left_hoop
                )
        frame_rows.append(
            {
                "gameId": game_id,
                "frameIdx": fi,
                "wallClock": frame.get("wallClock"),
                "gameClock": frame.get("gameClock"),
                "gameClockStopped": bool(frame.get("gameClockStopped")),
                "period": frame.get("period"),
                "shotClock": frame.get("shotClock"),
                "n_home": n_home,
                "n_away": n_away,
                "ten_players": n_home == 5 and n_away == 5,
                "live": live,
                "leftHoop": left_hoop,
                "is_model_sample_frame": fi in model_sample_frames,
                "is_pass_window_frame": fi in pass_window_frames,
                "ball_x": ball_x,
                "ball_y": ball_y,
                "ball_z": ball_z,
                "ball_x_event": ball_x_event,
                "ball_y_event": ball_y_event,
                "ball_speed": ball.get("speed"),
                "ball_isDetected": ball.get("isDetected"),
                "ball_predError": ball.get("predError"),
            }
        )
        for side, plist in (("home", home), ("away", away)):
            for p in plist:
                xyz = p.get("xyz") or [None, None, None]
                x, y, z = xyz[0], xyz[1], xyz[2]
                x_event = y_event = xn = yn = None
                if x is not None and y is not None and left_hoop is not None:
                    x_event, y_event = broadcast_to_event_xy(float(x), float(y), left_hoop)
                    xn, yn = normalize_attack_xy(x_event, y_event)
                player_rows.append(
                    {
                        "gameId": game_id,
                        "frameIdx": fi,
                        "teamSide": side,
                        "playerId": int(p["playerId"]),
                        "jersey": p.get("jersey"),
                        "x": x,
                        "y": y,
                        "z": z,
                        "leftHoop": left_hoop,
                        "x_event": x_event,
                        "y_event": y_event,
                        "x_norm": xn,
                        "y_norm": yn,
                        "speed": p.get("speed"),
                        "isDetected": p.get("isDetected"),
                        "predError": p.get("predError"),
                    }
                )

    frames = pd.DataFrame(frame_rows)
    players = pd.DataFrame(player_rows)

    if len(players):
        # Differentiate the continuous broadcast coordinates and rotate the
        # result into the event frame. Differencing x_event/y_event directly
        # turns every attacking-hoop change into an apparent teleport.
        players = add_causal_velocities(
            players,
            group_cols=("gameId", "playerId"),
            x_col="x",
            y_col="y",
            orientation_col="leftHoop",
        )
        quality["n_speed_implausible"] = int(players["speed_implausible"].sum())
        quality["max_speed_causal_fps"] = (
            float(players["speed_causal"].max()) if players["speed_causal"].notna().any() else None
        )
        quality["velocity_ok_rate"] = float(players["velocity_ok"].mean())

    quality["predError_mean"] = (
        quality["predError_sum"] / quality["predError_n"] if quality["predError_n"] else None
    )
    quality["ball_predError_mean"] = (
        quality["ball_predError_sum"] / quality["ball_predError_n"]
        if quality["ball_predError_n"]
        else None
    )
    quality["live_rate"] = quality["n_live"] / max(quality["n_frames_total"], 1)
    quality["ten_player_rate_of_live"] = (
        quality["n_ten_players"] / max(quality["n_live"], 1)
    )
    return frames, players, quality


def build_touch_model_frames(
    touches: pd.DataFrame,
    frames: pd.DataFrame,
    players: pd.DataFrame,
    stride: int = 5,
) -> pd.DataFrame:
    """Candidate decision-state scaffolding at 5 Hz for primary-eligible touches."""
    primary = touches[touches["primary_touch_eligible"]].copy()
    if primary.empty or frames.empty:
        return pd.DataFrame()

    fr_idx = frames.set_index("frameIdx", drop=False)
    # player lookup
    pl = players.set_index(["frameIdx", "playerId"]) if len(players) else None

    rows = []
    for trow in primary.itertuples(index=False):
        fids = sample_model_frames(int(trow.startFrame), int(trow.endFrame), stride=stride)
        bh = int(trow.playerId) if pd.notna(trow.playerId) else None
        for fi in fids:
            if fi not in fr_idx.index:
                continue
            f = fr_idx.loc[fi]
            if isinstance(f, pd.DataFrame):
                f = f.iloc[0]
            bh_x = bh_y = None
            bh_present = False
            if pl is not None and bh is not None and (fi, bh) in pl.index:
                pr = pl.loc[(fi, bh)]
                if isinstance(pr, pd.DataFrame):
                    pr = pr.iloc[0]
                xe = pr["x_event"] if "x_event" in pr.index else pd.NA
                ye = pr["y_event"] if "y_event" in pr.index else pd.NA
                if pd.notna(xe) and pd.notna(ye):
                    bh_x, bh_y = float(xe), float(ye)
                else:
                    bh_x, bh_y = float(pr["x"]), float(pr["y"])
                bh_present = True
            flags = frame_eligibility_flags(
                fi,
                game_clock_stopped=bool(f["gameClockStopped"]),
                frontcourt_frame=getattr(trow, "frontcourtFrame", None),
                ten_players=bool(f["ten_players"]),
                ballhandler_present=bool(bh_present),
                ballhandler_x=float(bh_x) if bh_x is not None else None,
            )
            rows.append(
                {
                    "gameId": int(trow.gameId),
                    "touchId": trow.id,
                    "chanceId": trow.chanceId,
                    "possessionId": trow.possessionId,
                    "frameIdx": fi,
                    "period": f.get("period"),
                    "gameClock": f.get("gameClock"),
                    "shotClock": f.get("shotClock"),
                    "ballhandlerId": bh,
                    "ballhandler_x": bh_x,
                    "ballhandler_y": bh_y,
                    **flags,
                }
            )
    return pd.DataFrame(rows)


def process_game(game_id: int, alias_map: dict, force: bool = False) -> dict:
    gdir = _game_table_dir(game_id)
    marker = gdir / "_DONE"
    if marker.exists() and not force:
        print(f"[{game_id}] skip (cached)")
        return {"gameId": game_id, "cached": True}

    print(f"[{game_id}] loading events...")
    tables = events_to_tables(game_id)
    tables = attach_player_aliases(tables, alias_map)
    game_meta = load_game_data(game_id)

    possessions = normalize_possessions(tables["possessions"], tables["shots"])
    chances = normalize_chances(tables["chances"], possessions)
    touches = normalize_touches(tables["touches"], chances)
    passes = normalize_passes(tables["passes"])
    multipass = audit_multipass_touches(passes)
    touches = annotate_pass_touch_flags(touches, passes, multipass)
    touches = classify_touches(touches)

    points = points_from_shots_and_ft(tables["shots"], tables["free_throws"])
    points = points[points["gameId"] == game_id] if len(points) else points

    passes_joined = join_pass_to_touch(passes, touches)
    candidates = build_touch_candidates(
        touches[touches["primary_touch_eligible"]], strict=True
    )
    post_catch = build_post_catch_chain(
        passes, touches, tables["shots"], tables["free_throws"]
    )

    needed, model_sample, pass_window = _needed_frames_for_game(
        touches, passes, tables["shots"], stride=5
    )
    print(f"[{game_id}] streaming tracking ({len(needed)} needed frames)...")
    frames, players, tq = _process_tracking(
        game_id, needed, possessions, model_sample, pass_window
    )

    model_frames = build_touch_model_frames(touches, frames, players, stride=5)
    matchups = tables["matchups"]
    if len(model_frames):
        # Defensive assignment for the ball-handler at each decision state.
        model_frames = interval_join_matchups(
            model_frames,
            matchups,
            frame_col="frameIdx",
            off_player_col="ballhandlerId",
            prefix="ballhandler_",
        )
    candidate_states = build_candidate_states(model_frames, candidates, matchups, players)

    # Write per-game tables
    gdir.mkdir(parents=True, exist_ok=True)
    write_table(possessions, gdir / "possessions.parquet")
    write_table(chances, gdir / "chances.parquet")
    write_table(touches, gdir / "touches.parquet")
    write_table(passes, gdir / "passes.parquet")
    write_table(passes_joined, gdir / "passes_joined.parquet")
    write_table(tables["shots"], gdir / "shots.parquet")
    write_table(tables["free_throws"], gdir / "free_throws.parquet")
    write_table(tables["matchups"], gdir / "matchups.parquet")
    write_table(multipass, gdir / "multipass_touches.parquet")
    write_table(points, gdir / "chance_points_recon.parquet")
    write_table(candidates, gdir / "touch_candidates.parquet")
    write_table(post_catch, gdir / "post_catch_chain.parquet")
    write_table(frames, gdir / "tracking_frames.parquet")
    write_table(players, gdir / "tracking_players.parquet")
    write_table(model_frames, gdir / "touch_model_frames.parquet")
    write_table(candidate_states, gdir / "candidate_states.parquet")
    write_json(
        {
            "gameId": game_id,
            "homeTeam": game_meta.get("homeTeam", {}).get("teamName"),
            "awayTeam": game_meta.get("awayTeam", {}).get("teamName"),
            "tracking_quality": tq,
            "n_primary_touches": int(touches["primary_touch_eligible"].sum()),
            "n_model_frames": len(model_frames),
            "n_model_frames_eligible": int(model_frames["primary_frame_eligible"].sum())
            if len(model_frames)
            else 0,
            "n_candidate_states": len(candidate_states),
            "n_retained_tracking_frames": len(frames),
            "n_pass_window_frames": len(pass_window),
        },
        gdir / "game_summary.json",
    )
    marker.write_text("ok\n", encoding="utf-8")
    print(f"[{game_id}] done primary_touches={int(touches['primary_touch_eligible'].sum())}")
    return {"gameId": game_id, "cached": False, "tracking_quality": tq}


def concat_games(game_ids: list[int], name: str) -> pd.DataFrame:
    parts = []
    for gid in game_ids:
        path = _game_table_dir(gid) / f"{name}.parquet"
        if path.exists():
            parts.append(pd.read_parquet(path))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def render_sample_flow_md(flow: pd.DataFrame, touches: pd.DataFrame, conc: dict) -> str:
    all_flow = flow[flow["gameId"].astype(str) == "ALL"]
    lines = [
        "# Stage 1 sample flow (CONSORT-style)",
        "",
        "**Freeze ID:** `passing_windows_stage0_20250924`",
        "",
        "## Overall cascade (first-match exclusion)",
        "",
        "| Stage | Remaining | Excluded here |",
        "|---|---:|---:|",
    ]
    for row in all_flow.itertuples(index=False):
        lines.append(
            f"| `{row.stage}` | {row.n_remaining} | {row.n_excluded_here} |"
        )
    lines += [
        "",
        f"**Primary-eligible touches:** {int(touches['primary_touch_eligible'].sum())} / {len(touches)}",
        "",
        "## Exclusions by reason",
        "",
        "| Reason | N |",
        "|---|---:|",
    ]
    by_reason = conc["by_reason"].fillna({"exclusion_reason": "primary_eligible"})
    for row in by_reason.itertuples(index=False):
        reason = row.exclusion_reason if pd.notna(row.exclusion_reason) else "(eligible)"
        lines.append(f"| `{reason}` | {row.n} |")

    lines += ["", "## Per-game primary eligible", "", "| gameId | all touches | primary |", "|---|---:|---:|"]
    for gid, gdf in touches.groupby("gameId"):
        lines.append(
            f"| {gid} | {len(gdf)} | {int(gdf['primary_touch_eligible'].sum())} |"
        )

    lines += ["", "## Exclusion concentration checks", ""]
    for key in ("by_period", "by_off_team", "by_score_diff", "by_shot_clock"):
        tab = conc.get(key, pd.DataFrame())
        lines.append(f"### {key}")
        lines.append("")
        if tab is None or tab.empty:
            lines.append("_No data_")
        else:
            lines.append(tab.to_markdown(index=False))
        lines.append("")
    lines.append(
        "**Interpretation note:** large rate swings by bin warrant Stage 2+ sensitivity; "
        "they do not invalidate the freeze but must be reported."
    )
    return "\n".join(lines) + "\n"


def render_join_audit_md(results: dict) -> str:
    lines = [
        "# Stage 1 join audit",
        "",
        "**Freeze ID:** `passing_windows_stage0_20250924`",
        "",
        "| Check | Result | Detail |",
        "|---|---|---|",
    ]
    for name, res in results.items():
        ok = res.get("ok")
        status = "PASS" if ok else "FAIL"
        lines.append(f"| `{name}` | **{status}** | `{res}` |")
    lines.append("")
    return "\n".join(lines) + "\n"


def render_calc_audit_md(calcs: dict) -> str:
    lines = [
        "# Stage 1 calculation audit",
        "",
        "Every derived Stage 1 field is documented with formula, units, spot-check, and pitfalls.",
        "",
        "## Registry",
        "",
        "| Field | Formula | Units | Spot-check | Pitfalls |",
        "|---|---|---|---|---|",
        "| `elapsed_frames` | `endFrame - startFrame` | frames (25 fps) | one-game identity + 10-game totals | Negative => malformed; excluded from primary |",
        "| `duration_s_from_frames` | `elapsed_frames / 25` | seconds | corr vs `touchTime` | Clock stoppage can diverge slightly from wall time |",
        "| `elapsed_game_clock` | `startGameClock - endGameClock` | seconds | zero-elapsed flagged | Game clock pauses; not wall duration |",
        "| `distance_recomputed` | `hypot(passerLoc - receiverLoc)` | feet | max abs diff vs `distance` ~0 | Null locs => NaN |",
        "| `speed_causal` | `(p[t]-p[t-1]) / (Δframe/25)` on **broadcast** coords, then rotated to the event frame | ft/s | future-mutate invariant + real-data boundary audit | First frame NaN; no future frames; **never difference `x_event`** (it flips sign at attacking-hoop changes); NaN across orientation segments, frame gaps > 25, and speeds > 40 ft/s |",
        "| `velocity_segment` | cumulative count of `leftHoop` changes per player | index | boundary audit | Segment change forces a derivative reset |",
        "| `speed_implausible` | `speed_causal > 40` ft/s before nulling | bool | 0 retained | Cap is a physical plausibility screen, not a model choice |",
        "| `assigned_defenderId` | matchup interval covering the frame for that offensive player | player id | interval-containment audit | Intervals overlap; ties broken by latest `startFrame`, then shortest, then smallest `matchupId` |",
        "| `pass_outcome_class` | `complete` / `incomplete_turnover` / `unknown_outcome` from tri-state `complete` | category | class counts vs raw feed | `complete` is nullable; `astype(bool)` silently merges nulls into failures |",
        "| `x_norm,y_norm` | `x_norm = attack_sign * x` (`attack_sign=-1`) | feet | hoop side frac | Events already attack −x |",
        "| `usable_flag` | `qualityIndex >= 3` | bool | equals vendor `usable` | Do not use QI as continuous covariate without care |",
        "| `points_recon` | made FG 2/3 + made FT 1 | points | manual sum == recon sum | **Never** use `chances.ptsScored` for outcomes |",
        "| `interceptorId` | copy of `toReceiverId` | player id | complete passes null | Never intended target |",
        "| `primary_touch_eligible` | first-match exclusion cascade | bool | invariants audit | Order matters for CONSORT |",
        "| `ballhandler_in_offense` | `playerId ∈ chance.offPlayerIds` | bool | exactly-4-candidates audit | Defender-credited touches inside an offensive chance otherwise yield five candidates |",
        "| `primary_frame_eligible` | live ∧ frontcourt ∧ 10 players ∧ BH present | bool | model-frame rates | `predError` caps TBD_TRAINING_FOLD |",
        "",
        "## Spot-check results",
        "",
    ]
    for k, v in calcs.items():
        status = "PASS" if (isinstance(v, dict) and v.get("ok", True)) or v is True else "INFO"
        if isinstance(v, dict) and "ok" in v:
            status = "PASS" if v["ok"] else "FAIL"
        lines.append(f"### `{k}` — {status}")
        lines.append("")
        lines.append("```")
        lines.append(str(v))
        lines.append("```")
        lines.append("")
    lines += [
        "## KNOWN_ISSUES handled in Stage 1",
        "",
        "- Self-passes excluded from primary",
        "- Malformed chance/touch frame order excluded",
        "- Zero-elapsed chances excluded",
        "- `toReceiverId` renamed semantically to interceptor; never used as target",
        "- Multi-pass touches excluded from primary (audit table retained)",
        "- Season aggregates not loaded",
        "- `ptsScored` not used for `points_recon`",
        "- Touches whose ball-handler is absent from `chance.offPlayerIds` excluded",
        "  (`ballhandler_not_in_offense`), so every primary touch forms exactly four",
        "  teammate candidates",
        "- Causal velocities differentiated on continuous broadcast coordinates and",
        "  reset at orientation/gap boundaries, replacing event-coordinate differencing",
        "",
        "## Known raw-data limitations recorded for Stage 2",
        "",
        "- `receiverLoc`, `distance`, `receiverRegion`, and `receiverId` are null for",
        "  **every** non-complete pass in the raw feed. Stage 1 drops nothing; the",
        "  channel does not exist. Stage 2 must infer targets from ball trajectory,",
        "  pass direction, and empirical flight-time support.",
        "- `complete` is tri-state. `unknown_outcome` passes additionally have no",
        "  `endFrame`, no receiver, and no interceptor, so they cannot enter",
        "  failed-pass target inference; use `failed_pass_inference_eligible`.",
        "",
    ]
    return "\n".join(lines) + "\n"


def render_quality_md(
    tq_rows: list[dict],
    model_frames: pd.DataFrame,
    touches: pd.DataFrame,
    players: pd.DataFrame,
    frames: pd.DataFrame,
    candidate_states: pd.DataFrame,
    passes: pd.DataFrame,
    velocity_audit: dict,
    matchup_audit: dict,
    pass_class_audit: dict,
) -> str:
    tq = pd.DataFrame(tq_rows)
    lines = [
        "# Stage 1 tracking quality report",
        "",
        "**Freeze ID:** `passing_windows_stage0_20250924`",
        "",
        "## Per-game tracking summary",
        "",
    ]
    if len(tq):
        show = tq[
            [
                c
                for c in [
                    "gameId",
                    "n_frames_total",
                    "n_live",
                    "n_dead",
                    "live_rate",
                    "n_ten_players",
                    "ten_player_rate_of_live",
                    "predError_mean",
                    "ball_predError_mean",
                    "max_speed_causal_fps",
                    "velocity_ok_rate",
                ]
                if c in tq.columns
            ]
        ]
        lines.append(show.to_markdown(index=False))

    lines += [
        "",
        "## Tracking retention (Stage 2 ball-flight windows)",
        "",
        f"- Retained 25 Hz frames: **{len(frames)}**",
        f"- Pass-window half-width: **±{PASS_WINDOW_FRAMES} frames** (±{PASS_WINDOW_FRAMES / 25:.1f} s) "
        "around every pass release and recorded end frame",
        f"- Non-complete passes have no `endFrame`, so their release window is widened to "
        f"**±{FAILED_PASS_WINDOW_FRAMES} frames** (±{FAILED_PASS_WINDOW_FRAMES / 25:.1f} s)",
    ]
    if "is_pass_window_frame" in frames.columns:
        lines.append(
            f"- Frames retained for pass windows: **{int(frames['is_pass_window_frame'].sum())}**"
        )
    if "is_model_sample_frame" in frames.columns:
        lines.append(
            f"- Frames retained as 5 Hz model samples: **{int(frames['is_model_sample_frame'].sum())}**"
        )
    lines.append(
        "- Stage 2 can therefore read ball trajectory around any pass directly from "
        "`tracking_frames.parquet` without re-streaming the raw feed."
    )

    lines += [
        "",
        "## Causal velocity plausibility",
        "",
        f"- Velocity frame-gap reset: **> {MAX_CAUSAL_FRAME_GAP} frames** "
        f"({MAX_CAUSAL_FRAME_GAP / 25:.1f} s)",
        f"- Plausibility cap: **{MAX_PLAUSIBLE_PLAYER_SPEED_FPS} ft/s** "
        "(elite sprint is ~32 ft/s)",
        f"- Max retained `speed_causal`: **{velocity_audit.get('max_speed_fps')}** ft/s",
        f"- 99.9th percentile `speed_causal`: **{velocity_audit.get('p99_9_speed_fps')}** ft/s",
        f"- Rows above cap after fix: **{velocity_audit.get('n_speed_over_cap')}**",
        f"- Rows with null velocity (segment starts, wide gaps, missing coords): "
        f"**{velocity_audit.get('n_speed_null')}** of {len(players)}",
        "",
        "Velocities are differentiated on continuous broadcast coordinates and then",
        "rotated into the event frame. Differencing `x_event`/`y_event` directly made",
        "every attacking-hoop change look like a teleport (previously up to 1169 ft/s).",
        "",
        "## Matchup interval join coverage",
        "",
        f"- Candidate states: **{len(candidate_states)}**",
        f"- Defensive assignment coverage (all states): "
        f"**{matchup_audit.get('coverage_all_states')}**",
        f"- Coverage among primary-eligible states: "
        f"**{matchup_audit.get('coverage_primary_eligible_states')}**",
        f"- Unmatched states: **{matchup_audit.get('n_unmatched')}** "
        f"({matchup_audit.get('unmatched_rate')})",
        f"- States where overlapping intervals required tie-breaking: "
        f"**{matchup_audit.get('n_ambiguous_overlapping_intervals')}** "
        f"({matchup_audit.get('ambiguous_rate')})",
        f"- Max intervals covering a single frame: "
        f"**{matchup_audit.get('max_intervals_covering_one_frame')}**",
        f"- Deterministic resolution rule: {matchup_audit.get('resolution_rule')}",
        "",
        "## Pass outcome classes and failed-pass evidence",
        "",
        "| Class | N | with `endFrame` | with `receiverId` | with `receiverLoc` | with interceptor |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cls, stats in (pass_class_audit.get("by_class") or {}).items():
        lines.append(
            f"| `{cls}` | {stats['n']} | {stats['n_with_endFrame']} | "
            f"{stats['n_with_receiverId']} | {stats['n_with_receiverLoc']} | "
            f"{stats['n_with_interceptor']} |"
        )
    lines += [
        "",
        "`receiverLoc` / `distance` / `receiverRegion` / `receiverId` are absent from the",
        "raw feed for every non-complete pass, so Stage 2 cannot compare candidates to a",
        "recorded receiver location for the failed-pass population. `unknown_outcome`",
        "passes also lack an `endFrame` entirely and are excluded from failed-pass target",
        "inference via `failed_pass_inference_eligible`.",
    ]
    if "failed_pass_inference_eligible" in passes.columns:
        lines.append("")
        lines.append(
            f"- Failed passes eligible for target inference: "
            f"**{int(passes['failed_pass_inference_eligible'].sum())}**"
        )

    lines += [
        "",
        "## Model-frame eligibility (5 Hz primary touches)",
        "",
    ]
    if len(model_frames):
        n = len(model_frames)
        n_ok = int(model_frames["primary_frame_eligible"].sum())
        lines.append(f"- Model frames built: **{n}**")
        lines.append(f"- Primary-frame eligible: **{n_ok}** ({n_ok / max(n, 1):.1%})")
        for col in ("live_clock", "frontcourt", "ten_players", "ballhandler_present"):
            if col in model_frames.columns:
                lines.append(f"- `{col}` true: {int(model_frames[col].sum())} / {n}")
    else:
        lines.append("_No model frames_")
    lines += [
        "",
        "## TBD_TRAINING_FOLD tracking caps",
        "",
        "Stage 0 freezes **selection procedure** only for `predError` caps.",
        "Stage 1 reports distributions but does **not** apply numeric caps yet.",
        "Caps will be chosen inside each outer-fold training set so tracking-failure",
        "among otherwise-eligible decision states stays under the 20% kill line.",
        "",
        f"Primary-eligible touches: {int(touches['primary_touch_eligible'].sum())}",
        "",
    ]
    return "\n".join(lines) + "\n"


def render_verification_md(
    n_touches: int,
    n_primary: int,
    n_frames: int,
    n_frames_ok: int,
    n_passes: int,
    join_results: dict,
    n_candidate_states: int,
    n_retained_tracking_frames: int,
    passes: pd.DataFrame,
) -> str:
    lines = [
        "# Stage 1 USER VERIFICATION CHECKLIST",
        "",
        "**Freeze ID:** `passing_windows_stage0_20250924`",
        "**Purpose:** Approve the canonical data layer before Stage 2 (failed-pass target inference).",
        "",
        "Mark each item **APPROVE** / **REJECT** / **CHANGE REQUEST**.",
        "",
        "## A. Sample sizes",
        "",
        f"| Metric | Value |",
        f"|---|---:|",
        f"| All touches | {n_touches} |",
        f"| Primary-eligible touches | {n_primary} |",
        f"| Model frames (5 Hz, primary touches) | {n_frames} |",
        f"| Primary-eligible model frames | {n_frames_ok} |",
        f"| Candidate states (frame x candidate) | {n_candidate_states} |",
        f"| Retained 25 Hz tracking frames | {n_retained_tracking_frames} |",
        f"| Passes (all) | {n_passes} |",
        f"| Passes: complete | {int((passes['pass_outcome_class'] == 'complete').sum())} |",
        f"| Passes: incomplete/turnover | {int((passes['pass_outcome_class'] == 'incomplete_turnover').sum())} |",
        f"| Passes: unknown outcome (no endFrame) | {int((passes['pass_outcome_class'] == 'unknown_outcome').sum())} |",
        "",
        "## B. Joins and geometry tests",
        "",
        "| # | Check | Pipeline result | Verify |",
        "|---|---|---|---|",
    ]
    for i, (name, res) in enumerate(join_results.items(), 1):
        status = "PASS" if res.get("ok") else "FAIL"
        lines.append(f"| B{i} | `{name}` | {status} | ☐ |")
    lines += [
        "",
        "## C. Exclusions and frozen rules",
        "",
        "| # | Rule | Verify |",
        "|---|---|---|",
        "| C1 | Self-passes excluded from primary | ☐ |",
        "| C2 | Malformed / zero-elapsed chances excluded | ☐ |",
        "| C3 | Multi-pass touches excluded (audited table retained) | ☐ |",
        "| C4 | Inbounds touches excluded | ☐ |",
        "| C5 | Substitution-boundary possessions excluded | ☐ |",
        "| C6 | `toReceiverId` not used as intended target | ☐ |",
        "| C7 | Points reconstructed from shots+FT only | ☐ |",
        "| C8 | Season aggregates not used | ☐ |",
        "| C9 | Player aliases added; original IDs preserved | ☐ |",
        "| C10 | `predError` caps still TBD_TRAINING_FOLD | ☐ |",
        "| C11 | Ball-handler outside `chance.offPlayerIds` excluded; every primary touch forms exactly 4 candidates | ☐ |",
        "| C12 | Causal velocities differenced on continuous coords; reset at orientation/gap boundaries | ☐ |",
        "| C13 | Matchup interval join materialized with deterministic overlap resolution | ☐ |",
        "| C14 | Failed-pass evidence limits documented (`receiverLoc` absent for all non-complete passes) | ☐ |",
        "",
        "## D. Artifacts present",
        "",
        "| # | Artifact | Verify |",
        "|---|---|---|",
        "| D1 | `artifacts/stage1_sample_flow.md` | ☐ |",
        "| D2 | `artifacts/stage1_join_audit.md` | ☐ |",
        "| D3 | `artifacts/stage1_calculation_audit.md` | ☐ |",
        "| D4 | `artifacts/stage1_quality_report.md` | ☐ |",
        "| D5 | `artifacts/stage1_output_manifest.json` (verify with `--verify-manifest`) | ☐ |",
        "| D6 | `tables/*.parquet` consolidated outputs | ☐ |",
        "| D7 | `tables/candidate_states.parquet` defensive-assignment layer | ☐ |",
        "",
        "## Sign-off",
        "",
        "| Role | Name | Date | Overall |",
        "|---|---|---|---|",
        "| User / analyst | | | APPROVE / REJECT |",
        "",
        "**If APPROVE:** Stage 2 may begin.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 1 canonical data build")
    parser.add_argument("--force", action="store_true", help="Rebuild even if cached")
    parser.add_argument("--games", nargs="*", type=int, default=None, help="Subset of game IDs")
    parser.add_argument(
        "--verify-manifest",
        action="store_true",
        help="Only re-verify artifacts/stage1_output_manifest.json and exit",
    )
    parser.add_argument(
        "--refresh-sign-off",
        action="store_true",
        help="Re-hash STAGE1_VERIFICATION.md after approval is recorded, then exit",
    )
    args = parser.parse_args()

    if args.refresh_sign_off:
        refresh_sign_off_hashes(
            ARTIFACTS / "stage1_output_manifest.json",
            [ARTIFACTS / "STAGE1_VERIFICATION.md"],
            PACKAGE_ROOT,
        )
        print("Sign-off hashes refreshed.")
        args.verify_manifest = True

    if args.verify_manifest:
        result = audit_output_manifest(
            ARTIFACTS / "stage1_output_manifest.json", PACKAGE_ROOT
        )
        print(json.dumps(result, indent=2))
        return 0 if result.get("ok") else 2

    t0 = time.time()
    freeze = load_freeze()
    game_ids = args.games or game_ids_from_freeze(freeze)
    alias_map = load_alias_map()
    TABLES.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    tq_rows = []
    for gid in game_ids:
        try:
            summary = process_game(gid, alias_map, force=args.force)
            if summary.get("tracking_quality"):
                tq_rows.append(summary["tracking_quality"])
            elif (_game_table_dir(gid) / "game_summary.json").exists():
                meta = json.loads((_game_table_dir(gid) / "game_summary.json").read_text(encoding="utf-8"))
                if "tracking_quality" in meta:
                    tq_rows.append(meta["tracking_quality"])
        except Exception:
            traceback.print_exc()
            return 1

    print("Consolidating tables...")
    possessions = concat_games(game_ids, "possessions")
    chances = concat_games(game_ids, "chances")
    touches = concat_games(game_ids, "touches")
    passes = concat_games(game_ids, "passes")
    passes_joined = concat_games(game_ids, "passes_joined")
    shots = concat_games(game_ids, "shots")
    free_throws = concat_games(game_ids, "free_throws")
    matchups = concat_games(game_ids, "matchups")
    multipass = concat_games(game_ids, "multipass_touches")
    points = concat_games(game_ids, "chance_points_recon")
    candidates = concat_games(game_ids, "touch_candidates")
    post_catch = concat_games(game_ids, "post_catch_chain")
    frames = concat_games(game_ids, "tracking_frames")
    players = concat_games(game_ids, "tracking_players")
    model_frames = concat_games(game_ids, "touch_model_frames")
    candidate_states = concat_games(game_ids, "candidate_states")

    for name, df in [
        ("possessions", possessions),
        ("chances", chances),
        ("touches", touches),
        ("passes", passes),
        ("passes_joined", passes_joined),
        ("shots", shots),
        ("free_throws", free_throws),
        ("matchups", matchups),
        ("multipass_touches", multipass),
        ("chance_points_recon", points),
        ("touch_candidates", candidates),
        ("post_catch_chain", post_catch),
        ("tracking_frames", frames),
        ("tracking_players", players),
        ("touch_model_frames", model_frames),
        ("candidate_states", candidate_states),
        ("player_id_aliases", load_alias_table()),
    ]:
        write_table(df, TABLES / f"{name}.parquet")

    flow = consort_flow(touches)
    write_table(flow, TABLES / "consort_flow.parquet")
    conc = exclusion_concentration(touches)

    # Audits
    print("Running audits...")
    join_results = {
        "event_tracking_clocks_passes": audit_event_tracking_clocks(passes, frames, sample_n=300),
        "pass_release_vs_touch_end": audit_pass_release_vs_touch_end(passes_joined),
        "receiver_loc_vs_tracking": audit_receiver_loc_vs_tracking(passes, players),
        "mirroring_symmetry": audit_mirroring_symmetry(),
        "causal_velocity": audit_causal_velocity(),
        "causal_velocity_boundaries": audit_causal_velocity_boundaries(players),
        "synthetic_lane_geometry": audit_synthetic_lane_geometry(),
        "candidate_sets_exactly_four": audit_candidate_sets(candidates, touches),
        "matchup_interval_join": audit_matchup_interval_join(candidate_states, matchups),
        "pass_outcome_classes": audit_pass_outcome_classes(passes),
    }
    try:
        inv = audit_invariants_primary(touches, passes)
        join_results["primary_invariants"] = inv
    except AssertionError as e:
        join_results["primary_invariants"] = {"ok": False, "error": str(e)}

    calc_results = {
        "points_reconstruction": audit_points_reconstruction(shots, free_throws, chances, points),
        "spot_check_game_114243": spot_check_calculations_one_game(114243, touches, passes, chances),
        "totals_10_games": {
            "n_touches": len(touches),
            "n_primary_touches": int(touches["primary_touch_eligible"].sum()),
            "n_passes": len(passes),
            "n_self_passes": int(passes["is_self_pass"].sum()) if "is_self_pass" in passes.columns else None,
            "n_multipass_touches": len(multipass),
            "n_chances": len(chances),
            "n_usable_chances": int(chances["usable_flag"].sum()) if "usable_flag" in chances.columns else None,
            "n_model_frames": len(model_frames),
            "n_model_frames_eligible": int(model_frames["primary_frame_eligible"].sum())
            if len(model_frames)
            else 0,
            "post_catch_with_receiver_touch_rate": float(post_catch["has_receiver_touch"].mean())
            if len(post_catch)
            else None,
            "hoop_orientation_ok_rate": float(possessions["hoop_orientation_ok"].mean())
            if "hoop_orientation_ok" in possessions.columns
            else None,
            "ok": True,
        },
        "mirroring": join_results["mirroring_symmetry"],
        "causal_velocity": join_results["causal_velocity"],
        "causal_velocity_boundaries": join_results["causal_velocity_boundaries"],
        "candidate_sets_exactly_four": join_results["candidate_sets_exactly_four"],
        "matchup_interval_join": join_results["matchup_interval_join"],
        "pass_outcome_classes": join_results["pass_outcome_classes"],
        "lane_geometry": join_results["synthetic_lane_geometry"],
        "tracking_retention": {
            "pass_window_frames_half_width": PASS_WINDOW_FRAMES,
            "failed_pass_window_frames_half_width": FAILED_PASS_WINDOW_FRAMES,
            "n_retained_tracking_frames": len(frames),
            "n_pass_window_frames": int(frames["is_pass_window_frame"].sum())
            if "is_pass_window_frame" in frames.columns
            else None,
            "n_model_sample_frames": int(frames["is_model_sample_frame"].sum())
            if "is_model_sample_frame" in frames.columns
            else None,
            "ok": True,
        },
    }

    write_markdown(render_sample_flow_md(flow, touches, conc), ARTIFACTS / "stage1_sample_flow.md")
    write_markdown(render_join_audit_md(join_results), ARTIFACTS / "stage1_join_audit.md")
    write_markdown(render_calc_audit_md(calc_results), ARTIFACTS / "stage1_calculation_audit.md")
    write_markdown(
        render_quality_md(
            tq_rows,
            model_frames,
            touches,
            players,
            frames,
            candidate_states,
            passes,
            join_results["causal_velocity_boundaries"],
            join_results["matchup_interval_join"],
            join_results["pass_outcome_classes"],
        ),
        ARTIFACTS / "stage1_quality_report.md",
    )
    write_markdown(
        render_verification_md(
            len(touches),
            int(touches["primary_touch_eligible"].sum()),
            len(model_frames),
            int(model_frames["primary_frame_eligible"].sum()) if len(model_frames) else 0,
            len(passes),
            join_results,
            len(candidate_states),
            len(frames),
            passes,
        ),
        ARTIFACTS / "STAGE1_VERIFICATION.md",
    )

    # Environment pin (written before the manifest so it can be hashed)
    env = {
        "python": sys.version,
        "packages": {
            name: md.version(name)
            for name in ("pandas", "numpy", "pyarrow", "pyyaml", "pytest")
            if _has_pkg(name)
        },
        "causal_velocity": {
            "max_frame_gap_frames": MAX_CAUSAL_FRAME_GAP,
            "max_plausible_player_speed_fps": MAX_PLAUSIBLE_PLAYER_SPEED_FPS,
        },
        "tracking_retention": {
            "pass_window_frames_half_width": PASS_WINDOW_FRAMES,
            "failed_pass_window_frames_half_width": FAILED_PASS_WINDOW_FRAMES,
        },
    }
    write_json(env, ARTIFACTS / "stage1_environment.json")

    # Manifest, generated last so every hashed artifact is already final.
    # Named explicitly rather than globbed: a case-insensitive `stage1_*` glob
    # also matches the hand-written STAGE1_* sign-off/audit documents.
    manifest_path = ARTIFACTS / "stage1_output_manifest.json"
    out_paths = sorted(TABLES.glob("*.parquet")) + [
        ARTIFACTS / name
        for name in (
            "stage1_sample_flow.md",
            "stage1_join_audit.md",
            "stage1_calculation_audit.md",
            "stage1_quality_report.md",
            "stage1_environment.json",
        )
    ]
    # include by_game summaries lightly
    for gid in game_ids:
        out_paths.append(_game_table_dir(gid) / "game_summary.json")
    manifest = build_output_manifest(
        out_paths,
        root=PACKAGE_ROOT,
        sign_off_paths=[ARTIFACTS / "STAGE1_VERIFICATION.md"],
        manifest_path=manifest_path,
    )
    manifest["freeze_id"] = freeze.get("freeze_id")
    manifest["stage"] = 1
    manifest["elapsed_seconds"] = round(time.time() - t0, 1)
    write_json(manifest, manifest_path)

    manifest_check = verify_output_manifest(manifest_path, PACKAGE_ROOT)
    print(f"Manifest verification: {manifest_check['ok']} ({manifest_check['n_files']} files)")
    if not manifest_check["ok"]:
        print("MANIFEST MISMATCH:", manifest_check)

    print(f"Stage 1 complete in {time.time() - t0:.1f}s")
    print(f"Primary touches: {int(touches['primary_touch_eligible'].sum())} / {len(touches)}")
    fails = [k for k, v in join_results.items() if not v.get("ok")]
    if fails:
        print("AUDIT FAILURES:", fails)
        return 2
    if not manifest_check["ok"]:
        return 2
    return 0


def _has_pkg(name: str) -> bool:
    try:
        md.version(name)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
