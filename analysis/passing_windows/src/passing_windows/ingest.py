"""Ingest game metadata, dynamic events, and streamed tracking."""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .aliases import add_canonical_column, load_alias_map
from .paths import (
    FREEZE_YAML,
    dynamic_events_path,
    game_data_path,
    tracking_path,
)

EVENT_FAMILIES = [
    "chances",
    "possessions",
    "shots",
    "rebounds",
    "turnovers",
    "free_throws",
    "fouls",
    "timeouts",
    "passes",
    "touches",
    "dribbles",
    "picks",
    "isolations",
    "handoffs",
    "drives",
    "posts",
    "off_ball_screens",
    "closeouts",
    "chance_players",
    "matchups",
]


def load_freeze(path: Path | None = None) -> dict[str, Any]:
    with open(path or FREEZE_YAML, encoding="utf-8") as f:
        return yaml.safe_load(f)


def game_ids_from_freeze(freeze: dict[str, Any] | None = None) -> list[int]:
    freeze = freeze or load_freeze()
    return [int(g) for g in freeze["game_ids"]]


def load_game_data(game_id: int) -> dict[str, Any]:
    with open(game_data_path(game_id), encoding="utf-8") as f:
        return json.load(f)


def load_dynamic_events(game_id: int) -> dict[str, list[dict]]:
    with open(dynamic_events_path(game_id), encoding="utf-8") as f:
        return json.load(f)


def _records_to_frame(records: list[dict], game_id: int) -> pd.DataFrame:
    if not records:
        return pd.DataFrame({"gameId": pd.Series(dtype="int64")})
    df = pd.DataFrame.from_records(records)
    if "gameId" not in df.columns:
        df.insert(0, "gameId", game_id)
    else:
        df["gameId"] = df["gameId"].astype(int)
    return df


def events_to_tables(game_id: int, events: dict[str, list[dict]] | None = None) -> dict[str, pd.DataFrame]:
    events = events or load_dynamic_events(game_id)
    tables: dict[str, pd.DataFrame] = {}
    for family in EVENT_FAMILIES:
        tables[family] = _records_to_frame(events.get(family, []), game_id)
    return tables


def iter_tracking_frames(game_id: int) -> Iterator[dict[str, Any]]:
    """Stream decompressed JSONL tracking frames (do not load all at once)."""
    path = tracking_path(game_id)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def tracking_frame_is_live(frame: dict[str, Any]) -> bool:
    home = frame.get("homePlayers") or []
    away = frame.get("awayPlayers") or []
    ball = frame.get("ball") or {}
    return len(home) > 0 and len(away) > 0 and bool(ball.get("xyz"))


def expand_tracking_frame(frame: dict[str, Any], game_id: int) -> tuple[dict, list[dict]]:
    """Return (frame_row, player_rows) for a single tracking frame."""
    home = frame.get("homePlayers") or []
    away = frame.get("awayPlayers") or []
    ball = frame.get("ball") or {}
    live = tracking_frame_is_live(frame)
    n_home, n_away = len(home), len(away)
    frame_row = {
        "gameId": game_id,
        "frameIdx": int(frame["frameIdx"]),
        "wallClock": frame.get("wallClock"),
        "gameClock": frame.get("gameClock"),
        "gameClockStopped": bool(frame.get("gameClockStopped")),
        "period": frame.get("period"),
        "shotClock": frame.get("shotClock"),
        "n_home": n_home,
        "n_away": n_away,
        "ten_players": n_home == 5 and n_away == 5,
        "live": live,
        "ball_x": ball.get("xyz", [None, None, None])[0] if ball.get("xyz") else None,
        "ball_y": ball.get("xyz", [None, None, None])[1] if ball.get("xyz") else None,
        "ball_z": ball.get("xyz", [None, None, None])[2] if ball.get("xyz") else None,
        "ball_speed": ball.get("speed"),
        "ball_isDetected": ball.get("isDetected"),
        "ball_predError": ball.get("predError"),
    }
    players: list[dict] = []
    for side, plist in (("home", home), ("away", away)):
        for p in plist:
            xyz = p.get("xyz") or [None, None, None]
            players.append(
                {
                    "gameId": game_id,
                    "frameIdx": int(frame["frameIdx"]),
                    "teamSide": side,
                    "playerId": int(p["playerId"]),
                    "jersey": p.get("jersey"),
                    "x": xyz[0],
                    "y": xyz[1],
                    "z": xyz[2],
                    "speed": p.get("speed"),
                    "isDetected": p.get("isDetected"),
                    "predError": p.get("predError"),
                }
            )
    return frame_row, players


def stream_tracking_tables(
    game_id: int,
    *,
    live_only: bool = True,
    frame_filter: set[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stream tracking into frame-level and player-level tables.

    Parameters
    ----------
    live_only:
        Keep only frames with players + ball.
    frame_filter:
        If provided, keep only these frameIdx values (after live filter if live_only).
    """
    frame_rows: list[dict] = []
    player_rows: list[dict] = []
    for frame in iter_tracking_frames(game_id):
        fi = int(frame["frameIdx"])
        if frame_filter is not None and fi not in frame_filter:
            continue
        if live_only and not tracking_frame_is_live(frame):
            if frame_filter is None:
                continue
            # Explicitly requested frames: still emit frame row even if dead
        frow, prows = expand_tracking_frame(frame, game_id)
        if live_only and not frow["live"] and frame_filter is None:
            continue
        frame_rows.append(frow)
        player_rows.extend(prows)
    return pd.DataFrame(frame_rows), pd.DataFrame(player_rows)


def attach_player_aliases(tables: dict[str, pd.DataFrame], alias_map: dict[int, int] | None = None) -> dict[str, pd.DataFrame]:
    """Add *_canonical columns for common player-id fields; preserve originals."""
    alias_map = alias_map or load_alias_map()
    id_cols = {
        "touches": ["playerId", "defenderId"],
        "passes": ["passerId", "receiverId", "toReceiverId"],
        "shots": ["shooterId", "passerId", "blockerId", "closestDefId"],
        "free_throws": ["shooterId"],
        "matchups": ["defPlayerId", "offPlayerId"],
        "turnovers": ["turnedOverId", "stealerId"],
        "chance_players": ["playerId"],
    }
    out: dict[str, pd.DataFrame] = {}
    for name, df in tables.items():
        result = df
        for col in id_cols.get(name, []):
            if col in result.columns:
                result = add_canonical_column(result, col, alias_map)
        out[name] = result
    return out
