"""Player ID alias normalization.

Within a game, SkillCorner player IDs are stable. Across games, nine players
appear under two IDs. Use aliases for person-level merges; keep original IDs
for within-game event/tracking joins.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .paths import ALIASES_CSV


def load_alias_map(path: Path | None = None) -> dict[int, int]:
    """Return mapping player_id -> canonical_player_id."""
    csv_path = path or ALIASES_CSV
    df = pd.read_csv(csv_path)
    return {
        int(row.player_id): int(row.canonical_player_id)
        for row in df.itertuples(index=False)
    }


def load_alias_table(path: Path | None = None) -> pd.DataFrame:
    csv_path = path or ALIASES_CSV
    df = pd.read_csv(csv_path)
    for col in ("player_id", "canonical_player_id", "acb_player_id"):
        if col in df.columns:
            df[col] = df[col].astype("Int64")
    return df


def canonicalize_id(player_id: int | None, alias_map: dict[int, int]) -> int | None:
    if player_id is None or (isinstance(player_id, float) and pd.isna(player_id)):
        return None
    pid = int(player_id)
    return alias_map.get(pid, pid)


def add_canonical_column(
    df: pd.DataFrame,
    source_col: str,
    alias_map: dict[int, int],
    out_col: str | None = None,
) -> pd.DataFrame:
    """Add a canonical person-id column; preserve the original source column."""
    out = out_col or f"{source_col}_canonical"
    result = df.copy()
    result[out] = result[source_col].map(
        lambda x: canonicalize_id(None if pd.isna(x) else int(x), alias_map)
        if pd.notna(x)
        else pd.NA
    )
    return result
