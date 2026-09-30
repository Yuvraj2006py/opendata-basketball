"""I/O helpers: parquet writes, hashing, manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def write_table(df: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix == ".csv":
        df.to_csv(path, index=False)
    elif path.suffix in {".feather", ".ft"}:
        df.to_feather(path)
    else:
        raise ValueError(f"Unsupported table suffix: {path.suffix}")
    return path


def read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix in {".feather", ".ft"}:
        return pd.read_feather(path)
    raise ValueError(f"Unsupported table suffix: {path.suffix}")


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _hash_entries(paths: list[Path], root: Path | None) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        rel = str(p.relative_to(root)) if root else str(p)
        # De-duplicate: the same artifact must never be hashed twice, or a single
        # mismatch is reported twice by verification.
        if rel in seen:
            continue
        seen.add(rel)
        entries.append(
            {
                "path": rel,
                "bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            }
        )
    return entries


def build_output_manifest(
    paths: list[Path],
    root: Path | None = None,
    *,
    sign_off_paths: list[Path] | None = None,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Hash generated outputs.

    `manifest_path` is excluded from its own file list: hashing a file that is
    then overwritten by the hash itself can never verify. `sign_off_paths` are
    hashed separately because human sign-off edits them after the build, so they
    are recorded for provenance but not part of strict verification.
    """
    root = Path(root) if root else None
    exclude = {Path(manifest_path).resolve()} if manifest_path else set()
    files = _hash_entries([p for p in paths if Path(p).resolve() not in exclude], root)
    out: dict[str, Any] = {
        "n_files": len(files),
        "files": files,
        "verification": (
            "`files` must hash-verify exactly (see verify_output_manifest). "
            "`sign_off_files` are expected to change when a human records "
            "approval, so they are provenance only."
        ),
    }
    if sign_off_paths:
        out["sign_off_files"] = _hash_entries(sign_off_paths, root)
    return out


def refresh_sign_off_hashes(
    manifest_path: Path,
    sign_off_paths: list[Path],
    root: Path,
) -> dict[str, Any]:
    """Re-hash the sign-off documents in place after approval is recorded.

    Sign-off files are edited by a human after the build, so their build-time
    hash goes stale by design. This records the post-approval hash without
    touching the strictly verified `files` section.
    """
    manifest_path = Path(manifest_path)
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    manifest["sign_off_files"] = _hash_entries(sign_off_paths, Path(root))
    write_json(manifest, manifest_path)
    return manifest


def verify_output_manifest(
    manifest: dict[str, Any] | Path,
    root: Path,
) -> dict[str, Any]:
    """Recompute SHA-256 for every strictly verified manifest entry."""
    if isinstance(manifest, (str, Path)):
        with open(manifest, encoding="utf-8") as f:
            manifest = json.load(f)
    root = Path(root)
    missing, mismatched, duplicates = [], [], []
    seen: set[str] = set()
    for entry in manifest.get("files", []):
        rel = entry["path"]
        if rel in seen:
            duplicates.append(rel)
        seen.add(rel)
        p = root / rel
        if not p.exists():
            missing.append(rel)
            continue
        if sha256_file(p) != entry["sha256"]:
            mismatched.append(rel)
    return {
        "n_files": len(manifest.get("files", [])),
        "n_missing": len(missing),
        "n_mismatched": len(mismatched),
        "n_duplicate_paths": len(duplicates),
        "missing": missing,
        "mismatched": mismatched,
        "duplicate_paths": duplicates,
        "ok": not missing and not mismatched and not duplicates,
    }


def write_json(obj: Any, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
        f.write("\n")
    return path


def write_markdown(text: str, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
