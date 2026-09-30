#!/usr/bin/env python
"""Live Stage 3 adversarial checks (auditor). Safe to re-run after builder lands tables.

Does not implement Stage 3 features. Writes/updates evidence under artifacts/.

    .\\.venv\\Scripts\\python.exe pipelines\\_stage3_adversarial_live_audit.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE_ROOT / "src"))

from passing_windows.ingest import game_ids_from_freeze, load_freeze  # noqa: E402
from passing_windows.io_utils import verify_output_manifest  # noqa: E402
from passing_windows.narrow_path import (  # noqa: E402
    GATE_COLUMNS,
    assert_gate_columns_present,
    load_narrow_path_lock,
    receiver_completion_training_labels,
)
from passing_windows.paths import ARTIFACTS, PACKAGE_ROOT, TABLES  # noqa: E402

OUT_MD = ARTIFACTS / "STAGE3_ADVERSARIAL_AUDIT.md"
OUT_JSON = ARTIFACTS / "STAGE3_AUDIT_CHECKLIST.json"


def _append_live_section(md_path: Path, section: str) -> None:
    text = md_path.read_text(encoding="utf-8") if md_path.exists() else ""
    marker = "## Live verification"
    if marker in text:
        pre = text.split(marker)[0].rstrip()
        text = pre + "\n\n" + section + "\n"
    else:
        text = text.rstrip() + "\n\n" + section + "\n"
    md_path.write_text(text, encoding="utf-8")


def main() -> int:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    features_path = TABLES / "stage3_candidate_features.parquet"
    manifest_path = ARTIFACTS / "stage3_output_manifest.json"
    eligibility_path = TABLES / "stage2_label_eligibility.parquet"

    checks: dict = {"ok": False, "stage3_present": features_path.exists()}
    if not features_path.exists():
        checks["detail"] = "stage3_candidate_features.parquet missing"
        section = (
            "## Live verification\n\n"
            f"**Status:** SKIPPED — `{features_path.as_posix()}` not found.\n"
        )
        _append_live_section(OUT_MD, section)
        print(json.dumps(checks, indent=2))
        return 1

    freeze = load_freeze()
    lock = load_narrow_path_lock()
    features = pd.read_parquet(features_path)
    eligibility = pd.read_parquet(eligibility_path)
    labels = receiver_completion_training_labels(eligibility)
    game_ids = game_ids_from_freeze(freeze)

    # Four candidates
    sizes = features.groupby(["gameId", "touchId", "frameIdx"]).size()
    checks["four_candidates_per_frame"] = {
        "ok": bool((sizes == 4).all()),
        "n_bad": int((sizes != 4).sum()),
        "mode": int(sizes.mode().iloc[0]) if len(sizes) else None,
    }

    # 5 Hz stride among multi-frame touches (release-only frames excluded)
    diffs: list[int] = []
    hz = (
        features[features["is_model_5hz_frame"].astype(bool)]
        if "is_model_5hz_frame" in features.columns
        else features
    )
    for _, g in hz.groupby(["gameId", "touchId"]):
        fr = sorted(g["frameIdx"].unique())
        if len(fr) >= 2:
            diffs.extend(int(d) for d in np.diff(fr))
    checks["model_5hz_stride"] = {
        "ok": (not diffs or all(d % 5 == 0 for d in diffs))
        and ("model_hz" not in features.columns or int(features["model_hz"].iloc[0]) == 5),
        "n_diffs": len(diffs),
    }

    # Null velocity
    missing_vel = ~features["receiver_velocity_ok"].astype(bool)
    checks["null_velocity_is_missing"] = {
        "ok": bool(features.loc[missing_vel, "receiver_vx"].isna().all()) if missing_vel.any() else True
    }

    # Narrow path
    assert_gate_columns_present(eligibility, name="eligibility")
    n_failed_elig = int(
        (
            (eligibility["cohort"] == "failed")
            & eligibility["receiver_specific_eligible"].fillna(False).astype(bool)
        ).sum()
    )
    checks["narrow_path_labels"] = {
        "ok": n_failed_elig == 0 and len(labels) == 4530,
        "n_labels": int(len(labels)),
        "failed_eligible": n_failed_elig,
    }
    present_gates = [c for c in GATE_COLUMNS if c in features.columns]
    checks["stage3_gate_columns"] = {
        "ok": len(present_gates) == len(GATE_COLUMNS),
        "present": present_gates,
        "missing": [c for c in GATE_COLUMNS if c not in features.columns],
    }

    # Completion label yield vs expected release coverage
    n_true = int(features["is_true_receiver"].fillna(False).sum()) if "is_true_receiver" in features.columns else 0
    n_comp_rows = (
        int(features["completion_label_eligible"].fillna(False).sum())
        if "completion_label_eligible" in features.columns
        else 0
    )
    # Count how many labeled passes have a release row in features
    if "passId" in features.columns and "is_pass_release_frame" in features.columns:
        release_pass_ids = set(
            features.loc[features["is_pass_release_frame"].fillna(False), "passId"].dropna().astype(str)
        )
        label_ids = set(labels["passId"].astype(str))
        n_labels_with_release = len(label_ids & release_pass_ids)
    else:
        n_labels_with_release = -1
    checks["release_label_yield"] = {
        "ok": n_labels_with_release >= int(0.9 * len(labels)),
        "n_labels": int(len(labels)),
        "n_labels_with_release_row": n_labels_with_release,
        "n_true_receiver_rows": n_true,
        "n_completion_label_eligible_rows": n_comp_rows,
        "threshold": ">=90% of 4530 labels must have a release feature row",
    }

    # Failed never completion-eligible
    if "cohort" in features.columns and "completion_label_eligible" in features.columns:
        bad = features["completion_label_eligible"].fillna(False) & (features["cohort"] == "failed")
        checks["no_failed_completion_labels"] = {"ok": int(bad.sum()) == 0, "n_bad": int(bad.sum())}
    else:
        checks["no_failed_completion_labels"] = {"ok": False, "detail": "missing columns"}

    # Trajectory rule
    if "trajectory_choice_rule" in features.columns:
        checks["trajectory_rule"] = {
            "ok": bool((features["trajectory_choice_rule"] == "lead_if_velocity_ok_else_direct").all())
        }
    else:
        checks["trajectory_rule"] = {"ok": False}

    # Fold flight JSON
    fold_path = ARTIFACTS / "stage3_fold_flight_models.json"
    if fold_path.exists():
        fold_doc = json.loads(fold_path.read_text(encoding="utf-8"))
        fold_ok = all(
            int(r["held_out_gameId"]) not in set(r["train_gameIds"])
            and int(r.get("n_train_completed", 0)) > 0
            for r in fold_doc.get("folds", [])
        )
        checks["fold_safe_flight_time"] = {
            "ok": fold_ok,
            "n_folds": len(fold_doc.get("folds", [])),
        }
    else:
        checks["fold_safe_flight_time"] = {"ok": False, "detail": "missing stage3_fold_flight_models.json"}

    # Manifest
    if manifest_path.exists():
        checks["manifest"] = verify_output_manifest(manifest_path, root=PACKAGE_ROOT)
    else:
        checks["manifest"] = {"ok": False, "detail": "missing manifest"}

    # Markers
    markers = sum(1 for g in game_ids if (TABLES / "by_game" / str(g) / "_DONE_STAGE3").exists())
    checks["markers"] = {"ok": markers == len(game_ids), "n_done": markers, "n_games": len(game_ids)}

    checks["ok"] = all(
        (v.get("ok") if isinstance(v, dict) else bool(v))
        for k, v in checks.items()
        if k not in ("ok", "stage3_present", "detail")
    )
    checks["narrow_path_lock_id"] = lock.get("lock_id")
    checks["n_candidate_rows"] = int(len(features))

    # Update checklist JSON live_verification block if present
    if OUT_JSON.exists():
        doc = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        doc["live_verification"] = {"ran": True, "checks": checks}
        doc["stage3_artifacts_present"] = True
        if not checks["ok"]:
            doc["verdict"] = "NO-GO"
        OUT_JSON.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")

    lines = [
        "## Live verification",
        "",
        f"**Status:** {'PASS' if checks['ok'] else 'FAIL'}",
        f"**Rows:** {checks['n_candidate_rows']}",
        f"**Narrow-path lock:** `{checks['narrow_path_lock_id']}`",
        "",
        "| Check | Result | Detail |",
        "|---|---|---|",
    ]
    for k, v in checks.items():
        if k in ("ok", "stage3_present", "narrow_path_lock_id", "n_candidate_rows"):
            continue
        ok = v.get("ok") if isinstance(v, dict) else bool(v)
        detail = json.dumps({kk: vv for kk, vv in v.items() if kk != "ok"}) if isinstance(v, dict) else ""
        lines.append(f"| `{k}` | {'PASS' if ok else 'FAIL'} | {detail} |")
    lines.append("")
    _append_live_section(OUT_MD, "\n".join(lines))

    out_live = ARTIFACTS / "STAGE3_LIVE_AUDIT.json"
    out_live.write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(checks, indent=2))
    return 0 if checks["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
