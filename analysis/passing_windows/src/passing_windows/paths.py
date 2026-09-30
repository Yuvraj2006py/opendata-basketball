"""Path helpers for the passing_windows analysis project."""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]  # analysis/passing_windows
REPO_ROOT = PACKAGE_ROOT.parents[1]  # opendata-basketball
DATA_ROOT = REPO_ROOT / "data"
CONFIGS = PACKAGE_ROOT / "configs"
ARTIFACTS = PACKAGE_ROOT / "artifacts"
TABLES = PACKAGE_ROOT / "tables"
FREEZE_YAML = CONFIGS / "stage0_freeze.yaml"
ALIASES_CSV = DATA_ROOT / "player_id_aliases.csv"
MATCHES_JSON = DATA_ROOT / "matches.json"


def match_dir(game_id: int | str) -> Path:
    return DATA_ROOT / "matches" / str(game_id)


def game_data_path(game_id: int | str) -> Path:
    gid = str(game_id)
    return match_dir(gid) / f"{gid}_game_data.json"


def dynamic_events_path(game_id: int | str) -> Path:
    gid = str(game_id)
    return match_dir(gid) / f"{gid}_dynamic_events.json"


def tracking_path(game_id: int | str) -> Path:
    gid = str(game_id)
    return match_dir(gid) / f"{gid}_tracking_data.jsonl.gz"
