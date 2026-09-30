#!/usr/bin/env python
"""Stage 3: reconstruct candidate pass geometry at every 5 Hz decision state.

Restartable: per-game feature tables are cached under
`tables/by_game/{gameId}/stage3_candidate_features.parquet` behind a
`_DONE_STAGE3` marker. Fold flight-time fitting always reruns (cheap) so
held-out games never leak into their own flight models.

    python pipelines/03_build_candidate_states.py
    python pipelines/03_build_candidate_states.py --force
    python pipelines/03_build_candidate_states.py --games 114086
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

STAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE_ROOT / "src"))

from passing_windows.candidates import (  # noqa: E402
    TRAJECTORY_RULE,
    attach_narrow_path_labels,
    reconstruct_game_candidates,
    summarize_candidates,
)
from passing_windows.flight_time import (  # noqa: E402
    assert_bundles_trainable,
    bundle_flight_model_payload,
    bundles_to_records,
    enrich_attempt_distances,
    fit_fold_flight_bundles,
    flight_bundle_cache_key,
)
from passing_windows.ingest import game_ids_from_freeze, load_freeze  # noqa: E402
from passing_windows.io_utils import (  # noqa: E402
    build_output_manifest,
    verify_output_manifest,
    write_json,
    write_markdown,
    write_table,
)
from passing_windows.narrow_path import (  # noqa: E402
    GATE_COLUMNS,
    NarrowPathViolation,
    assert_gate_columns_present,
    assert_interceptor_never_target,
    assert_no_failed_precision_claim,
    load_narrow_path_lock,
    receiver_completion_training_labels,
)
from passing_windows.paths import ARTIFACTS, PACKAGE_ROOT, TABLES  # noqa: E402

STAGE = 3
FREEZE_ID = "passing_windows_stage0_20250924"
MARKER_NAME = "_DONE_STAGE3"
FEATURES_NAME = "stage3_candidate_features.parquet"
SCHEMA_VERSION = 3


def _game_dir(game_id: int) -> Path:
    return TABLES / "by_game" / str(game_id)


def _load_concat_passes(game_ids: list[int], *, enrich_attempts: bool = False) -> pd.DataFrame:
    parts = []
    for gid in game_ids:
        path = _game_dir(gid) / "passes.parquet"
        if not path.exists():
            continue
        passes = pd.read_parquet(path)
        if enrich_attempts:
            players_path = _game_dir(gid) / "tracking_players.parquet"
            frames_path = _game_dir(gid) / "tracking_frames.parquet"
            players = pd.read_parquet(players_path) if players_path.exists() else None
            frames = pd.read_parquet(frames_path) if frames_path.exists() else None
            passes = enrich_attempt_distances(passes, tracking_players=players, tracking_frames=frames)
        parts.append(passes)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _read_marker_hash(marker: Path) -> str | None:
    if not marker.exists():
        return None
    text = marker.read_text(encoding="utf-8").strip()
    if not text:
        return None
    # New format: "hash=<hex>" or bare hex; legacy "ok" → miss.
    if text.startswith("hash="):
        return text.split("=", 1)[1].strip()
    if text in {"ok", "ok\n"}:
        return None
    return text.split()[0]


def process_game(
    game_id: int,
    flight_bundle,
    eligibility: pd.DataFrame,
    *,
    force: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    gdir = _game_dir(game_id)
    marker = gdir / MARKER_NAME
    out_path = gdir / FEATURES_NAME
    expected_hash = flight_bundle_cache_key(flight_bundle)
    cached_hash = _read_marker_hash(marker)
    if marker.exists() and out_path.exists() and not force and cached_hash == expected_hash:
        df = pd.read_parquet(out_path)
        summary = {"gameId": game_id, "cached": True, **summarize_candidates(df)}
        return df, summary

    t0 = time.time()
    candidate_states = pd.read_parquet(gdir / "candidate_states.parquet")
    touch_model_frames = pd.read_parquet(gdir / "touch_model_frames.parquet")
    tracking_players = pd.read_parquet(gdir / "tracking_players.parquet")
    chances = pd.read_parquet(gdir / "chances.parquet")
    touches = pd.read_parquet(gdir / "touches.parquet")
    passes = pd.read_parquet(gdir / "passes.parquet")
    touch_candidates = pd.read_parquet(gdir / "touch_candidates.parquet")
    matchups = pd.read_parquet(gdir / "matchups.parquet")

    features = reconstruct_game_candidates(
        candidate_states=candidate_states,
        touch_model_frames=touch_model_frames,
        tracking_players=tracking_players,
        chances=chances,
        touches=touches,
        flight_bundle=flight_bundle,
        primary_eligible_only=True,
        passes=passes,
        touch_candidates=touch_candidates,
        matchups=matchups,
        force_pass_releases=True,
    )
    features = attach_narrow_path_labels(features, eligibility, passes)
    write_table(features, out_path)
    marker.write_text(f"hash={expected_hash}\nschema={SCHEMA_VERSION}\n", encoding="utf-8")
    summary = {
        "gameId": game_id,
        "cached": False,
        "elapsed_seconds": round(time.time() - t0, 2),
        "fold_id": flight_bundle.fold_id,
        "n_train_completed_for_flight": flight_bundle.n_train_completed,
        "distance_support_max_ft": float(flight_bundle.distance_support.distance_max_ft),
        "flight_bundle_hash": expected_hash,
        **summarize_candidates(features),
    }
    write_json(summary, gdir / "stage3_game_summary.json")
    return features, summary


def render_candidate_report(
    summary: dict[str, Any],
    fold_records: list[dict[str, Any]],
    game_summaries: list[dict[str, Any]],
    lock: dict[str, Any],
) -> str:
    mix = summary.get("trajectory_mix", {})
    regions = summary.get("court_region_mix", {})
    lines = [
        "# Stage 3 candidate reconstruction report",
        "",
        f"**Freeze ID:** `{FREEZE_ID}`  ",
        f"**Narrow-path lock:** `{lock.get('lock_id')}`  ",
        f"**Trajectory choice rule:** `{TRAJECTORY_RULE}` (explicit; not score-maximizing)",
        "",
        "## Population",
        "",
        f"- Candidate rows (5 Hz primary + forced release × 4 receivers): **{summary.get('n_candidate_rows')}**",
        f"- Games: **{summary.get('n_games')}**",
        f"- Touches: **{summary.get('n_touches')}**",
        f"- Decision frames: **{summary.get('n_frames')}**",
        f"- Model 5 Hz frames: **{summary.get('n_model_5hz_frames')}**",
        f"- Forced release frames: **{summary.get('n_forced_release_frames')}**",
        f"- Candidates per frame (mode): **{summary.get('candidates_per_frame_mode')}**",
        "",
        "## Dual sampling",
        "",
        "- `model_5hz` / `is_model_5hz_frame`: primary-eligible stride-5 window series.",
        "- `pass_release` / `is_forced_release_frame`: every pass `startFrame` with tracking,",
        "  force-inserted even when off the 5 Hz lattice so Stage 4 can train passability",
        "  at release-time state. Overlaps are dual-flagged.",
        "",
        "## Narrow-path label join",
        "",
        f"- Completion-label-eligible candidate rows (release × known receiver pass): "
        f"**{summary.get('n_completion_label_eligible_rows')}**",
        f"- True-receiver rows among those: **{summary.get('n_true_receiver_rows')}**",
        "- Failed passes remain touch-level only; inferred targets are **not** used as",
        "  named receivers for completion calibration.",
        "",
        "## Support rejection and missingness",
        "",
        f"- Rejected outside fold distance support: "
        f"**{summary.get('pct_rejected_outside_support', float('nan')):.2f}%**",
        f"- Missing matchup (`matchup_matched == False`): "
        f"**{summary.get('pct_missing_matchup', float('nan')):.2f}%**",
        f"- Receiver velocity missing/invalid: "
        f"**{summary.get('pct_receiver_velocity_missing', float('nan')):.2f}%**",
        "",
        "## Trajectory mix (primary choice)",
        "",
    ]
    for k, v in sorted(mix.items()):
        lines.append(f"- `{k}`: {v}")
    lines.extend(["", "## Court region mix (chosen trajectory catch)", ""])
    for k, v in sorted(regions.items(), key=lambda kv: -kv[1]):
        lines.append(f"- `{k}`: {v}")
    lines.extend(
        [
            "",
            "## Fold flight-time models (leave-one-game-out)",
            "",
            "| Fold | Held-out | Train completed | Bins | Support max ft |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for rec in fold_records:
        lines.append(
            f"| `{rec['fold_id']}` | {rec['held_out_gameId']} | {rec['n_train_completed']} | "
            f"{rec['flight_time_bins']} | {rec['distance_support_max_ft']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Per-game summaries",
            "",
            "| Game | Rows | % rejected | % missing matchup | Cached | Seconds |",
            "|---:|---:|---:|---:|---|---:|",
        ]
    )
    for g in game_summaries:
        lines.append(
            f"| {g.get('gameId')} | {g.get('n_candidate_rows', 0)} | "
            f"{g.get('pct_rejected_outside_support', float('nan')):.2f} | "
            f"{g.get('pct_missing_matchup', float('nan')):.2f} | "
            f"{g.get('cached')} | {g.get('elapsed_seconds', '—')} |"
        )
    lines.extend(
        [
            "",
            "## Feature families retained",
            "",
            "- Pass geometry: distance, angle, lane clearance, trajectory length, sideline/baseline proximity",
            "- Motion: passer / receiver / assigned-defender velocity components (NaN when missing)",
            "- Timing: flight time, shot clock, touch age, period",
            "- Defensive context: passer pressure, receiver separation, help density, temporal margin, recovery time",
            "- Offensive context: rim distance/angle, paint/corner/wing region, toward-rim speed",
            "- Reliability: lane `predError` mean; detection via tracked flags",
            "- Transparent `geometry_passability_score` plus Stage-4 placeholders (`q_passability_model`, etc.)",
            "",
            "## Constraints obeyed",
            "",
            "1. Joined `tables/stage2_label_eligibility.parquet`.",
            "2. Completion labels via `receiver_completion_training_labels` only.",
            "3. All failed passes touch-level by default (no interceptor-as-target).",
            "4. Missing matchups / null velocities stay missing, not zero.",
            "5. Primary sampling at 5 Hz (stride 5) per Stage 0 freeze.",
            "",
        ]
    )
    return "\n".join(lines)


def render_geometry_selfcheck(checks: dict[str, Any], summary: dict[str, Any]) -> str:
    """Markdown twin of automated geometry / narrow-path self-checks."""
    lines = [
        "# Stage 3 geometry self-check",
        "",
        f"**Freeze ID:** `{FREEZE_ID}`  ",
        f"**Overall:** {'PASS' if checks.get('ok') else 'FAIL'}",
        "",
        "These checks mirror the Stage 2 geometry self-check: machine-verified",
        "invariants of the candidate reconstruction, not Stage 4 model fit quality.",
        "",
        "## Checks",
        "",
        "| Check | Result | Detail |",
        "|---|---|---|",
    ]
    for name, result in checks.items():
        if name == "ok":
            continue
        ok = result.get("ok") if isinstance(result, dict) else bool(result)
        detail = result.get("detail", "") if isinstance(result, dict) else ""
        lines.append(f"| `{name}` | {'PASS' if ok else 'FAIL'} | {detail} |")
    lines.extend(
        [
            "",
            "## Population snapshot",
            "",
            f"- Candidate rows: **{summary.get('n_candidate_rows')}**",
            f"- Decision frames: **{summary.get('n_frames')}**",
            f"- Candidates/frame mode: **{summary.get('candidates_per_frame_mode')}**",
            f"- Rejected outside support: "
            f"**{summary.get('pct_rejected_outside_support', float('nan')):.2f}%**",
            f"- Missing matchup: **{summary.get('pct_missing_matchup', float('nan')):.2f}%**",
            f"- Trajectory mix: `{summary.get('trajectory_mix')}`",
            "",
            "## Semantics locked here",
            "",
            f"- Trajectory rule: `{TRAJECTORY_RULE}` (explicit; never score-maximizing).",
            "- Both `direct_*` and `lead_*` feature families are retained.",
            "- Missing velocities stay NaN (not zero); unmatched matchups leave recovery NaN.",
            "- Fold flight-time models never train on the held-out game.",
            "- Completion labels only via narrow-path `receiver_specific_eligible`.",
            "",
        ]
    )
    return "\n".join(lines)


def render_verification(summary: dict[str, Any], checks: dict[str, Any]) -> str:
    lines = [
        "# Stage 3 verification checklist",
        "",
        f"**Freeze ID:** `{FREEZE_ID}`  ",
        f"**Status:** {'PASS' if checks.get('ok') else 'FAIL'}",
        "",
        "## Counts",
        "",
        f"| Metric | Value |",
        f"|---|---:|",
        f"| Candidate rows | {summary.get('n_candidate_rows')} |",
        f"| Decision frames | {summary.get('n_frames')} |",
        f"| Candidates/frame (mode) | {summary.get('candidates_per_frame_mode')} |",
        f"| % rejected (support) | {summary.get('pct_rejected_outside_support', float('nan')):.2f} |",
        f"| % missing matchup | {summary.get('pct_missing_matchup', float('nan')):.2f} |",
        f"| Completion-label-eligible rows | {summary.get('n_completion_label_eligible_rows')} |",
        "",
        "## Automated checks",
        "",
        "| Check | Result |",
        "|---|---|",
    ]
    for name, result in checks.items():
        if name == "ok":
            continue
        ok = result.get("ok") if isinstance(result, dict) else bool(result)
        detail = result.get("detail", "") if isinstance(result, dict) else ""
        lines.append(f"| `{name}` | {'PASS' if ok else 'FAIL'} {detail} |")
    lines.extend(
        [
            "",
            "## Human sign-off",
            "",
            "| Item | Status |",
            "|---|---|",
            "| Four candidates per primary-eligible frame | ☐ |",
            "| Causal velocity only (no future frames) | ☐ |",
            "| Explicit trajectory choice (not max-score) | ☐ |",
            "| Fold-safe flight time (no held-out leakage) | ☐ |",
            "| Narrow-path eligibility join | ☐ |",
            "| No interceptor-as-target | ☐ |",
            "",
        ]
    )
    return "\n".join(lines)


def run_self_checks(
    features: pd.DataFrame,
    eligibility: pd.DataFrame,
    fold_records: list[dict[str, Any]],
) -> dict[str, Any]:
    checks: dict[str, Any] = {}

    # 4 candidates per frame
    if features.empty:
        checks["four_candidates_per_frame"] = {"ok": False, "detail": "empty"}
    else:
        sizes = features.groupby(["gameId", "touchId", "frameIdx"]).size()
        ok = bool((sizes == 4).all())
        checks["four_candidates_per_frame"] = {
            "ok": ok,
            "detail": f"mode={int(sizes.mode().iloc[0])}; n_bad={int((sizes != 4).sum())}",
        }

    # Explicit trajectory choice present and rule-consistent
    if not features.empty:
        rule_ok = (features["trajectory_choice_rule"] == TRAJECTORY_RULE).all()
        lead_when_vel = features.loc[features["receiver_velocity_ok"].astype(bool), "trajectory_choice"]
        direct_when_no = features.loc[~features["receiver_velocity_ok"].astype(bool), "trajectory_choice"]
        consistent = bool(
            rule_ok
            and (lead_when_vel == "lead").all()
            and (direct_when_no == "direct").all()
        )
        # Both trajectory feature families present
        has_both = "direct_geometry_passability_score" in features.columns and (
            "lead_geometry_passability_score" in features.columns
        )
        checks["explicit_trajectory_choice"] = {
            "ok": consistent and has_both,
            "detail": f"rule_ok={bool(rule_ok)}; both_families={has_both}",
        }

        # Null velocity not coerced to zero
        missing_vel = ~features["receiver_velocity_ok"].astype(bool)
        vx_nan = features.loc[missing_vel, "receiver_vx"].isna().all() if missing_vel.any() else True
        checks["null_velocity_is_missing"] = {
            "ok": bool(vx_nan),
            "detail": "receiver_vx NaN wherever velocity_ok is False",
        }

        missing_mu = ~features["matchup_matched"].astype(bool)
        recovery_nan = (
            features.loc[missing_mu, "target_defender_recovery_s"].isna().all()
            if missing_mu.any()
            else True
        )
        checks["missing_matchup_is_missing"] = {
            "ok": bool(recovery_nan),
            "detail": "target_defender_recovery_s NaN when matchup unmatched",
        }

    # Fold-safe: each game's fold excludes itself from train list
    fold_ok = all(
        int(r["held_out_gameId"]) not in set(r["train_gameIds"]) for r in fold_records
    )
    checks["fold_safe_flight_time"] = {"ok": fold_ok, "detail": f"n_folds={len(fold_records)}"}

    # Narrow-path join
    try:
        assert_gate_columns_present(eligibility, name="eligibility")
        labels = receiver_completion_training_labels(eligibility)
        n_failed_eligible = int(
            (
                (eligibility["cohort"] == "failed")
                & eligibility["receiver_specific_eligible"].fillna(False).astype(bool)
            ).sum()
        )
        checks["narrow_path_label_join"] = {
            "ok": n_failed_eligible == 0 and len(labels) > 0,
            "detail": f"completion_labels={len(labels)}; failed_eligible={n_failed_eligible}",
        }
    except NarrowPathViolation as exc:
        checks["narrow_path_label_join"] = {"ok": False, "detail": str(exc)}

    # No interceptor as candidate / true receiver
    try:
        assert_interceptor_never_target(features, name="features", target_col="true_receiverId")
        clash = 0
        if "interceptorId" in features.columns:
            clash = int(
                (
                    features["candidateId"].notna()
                    & features["interceptorId"].notna()
                    & (features["candidateId"].astype("Int64") == features["interceptorId"].astype("Int64"))
                ).sum()
            )
        checks["no_interceptor_as_target"] = {"ok": clash == 0, "detail": f"clash_rows={clash}"}
    except (NarrowPathViolation, AssertionError) as exc:
        checks["no_interceptor_as_target"] = {"ok": False, "detail": str(exc)}

    # Support rejection column exists and is boolean-like
    if not features.empty:
        col = features["rejected_outside_support"]
        checks["support_rejection_flag"] = {
            "ok": "rejected_outside_support" in features.columns
            and set(col.dropna().unique()).issubset({True, False, 0, 1}),
            "detail": f"pct={100.0 * col.astype(bool).mean():.2f}",
        }

    # 5 Hz sampling — only among model_5hz frames (release frames may break stride)
    if not features.empty and "frameIdx" in features.columns:
        hz = (
            features[features["is_model_5hz_frame"].astype(bool)]
            if "is_model_5hz_frame" in features.columns
            else features
        )
        diffs = []
        for _, g in hz.groupby(["gameId", "touchId"]):
            frames = sorted(g["frameIdx"].unique())
            if len(frames) >= 2:
                diffs.extend(np.diff(frames).tolist())
        ok_stride = all(int(d) % 5 == 0 for d in diffs) if diffs else True
        checks["model_5hz_stride"] = {
            "ok": ok_stride and int(features["model_hz"].iloc[0]) == 5,
            "detail": f"n_diffs_checked={len(diffs)}",
        }

    # Dual sampling + release-label coverage
    if not features.empty:
        has_roles = "is_forced_release_frame" in features.columns and "is_model_5hz_frame" in features.columns
        checks["dual_sampling_flags"] = {
            "ok": has_roles
            and bool(features["is_model_5hz_frame"].astype(bool).any())
            and bool(features["is_forced_release_frame"].astype(bool).any()),
            "detail": (
                f"n_5hz_frames={int(features.loc[features['is_model_5hz_frame'].astype(bool)].groupby(['gameId','touchId','frameIdx']).ngroups)}; "
                f"n_release_frames={int(features.loc[features['is_forced_release_frame'].astype(bool)].groupby(['gameId','touchId','frameIdx']).ngroups)}"
                if has_roles
                else "missing flags"
            ),
        }
        labels = receiver_completion_training_labels(eligibility)
        if "passId" in features.columns and "is_pass_release_frame" in features.columns and not labels.empty:
            # Scope to games actually present in this features table so --games
            # subset runs are not graded against the full-season label set.
            if "gameId" in features.columns and "gameId" in labels.columns:
                feat_games = set(features["gameId"].dropna().astype(int).tolist())
                labels = labels[labels["gameId"].astype(int).isin(feat_games)]
            release_ids = set(
                features.loc[features["is_pass_release_frame"].fillna(False), "passId"]
                .dropna()
                .astype(str)
            )
            label_ids = set(labels["passId"].astype(str))
            n_hit = len(label_ids & release_ids)
            n_lab = len(label_ids)
            checks["release_label_coverage"] = {
                "ok": n_lab == 0 or n_hit >= int(0.9 * n_lab),
                "detail": f"{n_hit}/{n_lab} completion labels (in reconstructed games) have release feature rows",
            }
        else:
            checks["release_label_coverage"] = {"ok": False, "detail": "missing join columns"}

        present_gates = [c for c in GATE_COLUMNS if c in features.columns]
        checks["full_gate_columns"] = {
            "ok": len(present_gates) == len(GATE_COLUMNS),
            "detail": f"present={len(present_gates)}/{len(GATE_COLUMNS)}; missing={[c for c in GATE_COLUMNS if c not in features.columns]}",
        }

    checks["ok"] = all(v.get("ok", False) for k, v in checks.items() if k != "ok")
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 3: build candidate pass geometry")
    parser.add_argument("--force", action="store_true", help="Rebuild per-game caches")
    parser.add_argument("--games", nargs="+", type=int, help="Subset of game IDs")
    parser.add_argument(
        "--verify-manifest",
        action="store_true",
        help="Only verify stage3_output_manifest.json hashes",
    )
    args = parser.parse_args(argv)
    manifest_path = ARTIFACTS / "stage3_output_manifest.json"

    if args.verify_manifest:
        result = verify_output_manifest(manifest_path, root=PACKAGE_ROOT)
        print(json.dumps(result, indent=2))
        return 0 if result.get("ok") else 2

    t0 = time.time()
    freeze = load_freeze()
    lock = load_narrow_path_lock()
    game_ids = args.games or game_ids_from_freeze(freeze)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    eligibility_path = TABLES / "stage2_label_eligibility.parquet"
    if not eligibility_path.exists():
        print("Missing Stage 2 label eligibility table; run Stage 2 first.")
        return 1
    eligibility = pd.read_parquet(eligibility_path)
    try:
        assert_gate_columns_present(eligibility, name="stage2_label_eligibility")
        labels = receiver_completion_training_labels(eligibility)
    except NarrowPathViolation as exc:
        print("NARROW-PATH GATE VIOLATION:", exc)
        return 2
    print(f"Narrow-path completion labels: {len(labels)} (failed eligible: 0 expected)")

    # Flight / support fitting ALWAYS uses every freeze game — never the --games subset.
    all_freeze_ids = game_ids_from_freeze(freeze)
    print(f"Loading passes for flight/support from all {len(all_freeze_ids)} freeze games...")
    passes_for_flight = _load_concat_passes(all_freeze_ids, enrich_attempts=True)
    if passes_for_flight.empty:
        print("No Stage 1 passes found; run Stage 1 first.")
        return 1
    n_attempt_finite = int(passes_for_flight["attempt_distance_ft"].notna().sum()) if "attempt_distance_ft" in passes_for_flight.columns else 0
    print(f"Attempt distances finite: {n_attempt_finite}/{len(passes_for_flight)}")

    try:
        bundles = fit_fold_flight_bundles(
            passes_for_flight, eligibility, freeze["validation"]["outer_folds"]
        )
        assert_bundles_trainable(bundles)
    except Exception:
        traceback.print_exc()
        return 1

    # Reconstruction may be limited by --games; flight models stay fold-complete.
    reconstruct_ids = args.games or all_freeze_ids
    parts: list[pd.DataFrame] = []
    game_summaries: list[dict[str, Any]] = []
    for gid in reconstruct_ids:
        if int(gid) not in bundles:
            print(f"[{gid}] no fold bundle; skip")
            continue
        print(f"[{gid}] reconstructing candidates (fold={bundles[int(gid)].fold_id})...")
        try:
            feats, summary = process_game(
                int(gid), bundles[int(gid)], eligibility, force=args.force
            )
        except Exception:
            traceback.print_exc()
            return 1
        parts.append(feats)
        game_summaries.append(summary)
        print(
            f"[{gid}] rows={summary.get('n_candidate_rows')} "
            f"rejected={summary.get('pct_rejected_outside_support', float('nan')):.1f}% "
            f"cached={summary.get('cached')}"
        )

    if not parts:
        print("No Stage 3 features produced.")
        return 1

    features = pd.concat(parts, ignore_index=True)
    write_table(features, TABLES / "stage3_candidate_features.parquet")

    # Support-rejection flag table (compact)
    reject_cols = [
        c
        for c in [
            "gameId",
            "touchId",
            "frameIdx",
            "candidateId",
            "pass_distance_ft",
            "rejected_outside_support",
            "direct_rejected_outside_support",
            "lead_rejected_outside_support",
            "distance_support_min_ft",
            "distance_support_max_ft",
            "fold_id",
        ]
        if c in features.columns
    ]
    write_table(features[reject_cols], TABLES / "stage3_support_rejection.parquet")

    fold_records = bundles_to_records(bundles)
    for rec in fold_records:
        gid = int(rec["held_out_gameId"])
        rec["flight_model"] = bundle_flight_model_payload(bundles[gid].flight_model)
    write_json(
        {
            "freeze_id": FREEZE_ID,
            "stage": STAGE,
            "trajectory_choice_rule": TRAJECTORY_RULE,
            "folds": fold_records,
        },
        ARTIFACTS / "stage3_fold_flight_models.json",
    )

    summary = summarize_candidates(features)
    summary["elapsed_seconds_total"] = round(time.time() - t0, 1)
    write_json(
        {"overall": summary, "games": game_summaries},
        ARTIFACTS / "stage3_game_summaries.json",
    )

    checks = run_self_checks(features, eligibility, fold_records)
    report_md = render_candidate_report(summary, fold_records, game_summaries, lock)
    verification_md = render_verification(summary, checks)
    selfcheck_md = render_geometry_selfcheck(checks, summary)
    try:
        assert_no_failed_precision_claim(report_md, name="stage3_candidate_report.md")
        assert_no_failed_precision_claim(verification_md, name="STAGE3_VERIFICATION.md")
        assert_no_failed_precision_claim(selfcheck_md, name="stage3_geometry_selfcheck.md")
    except NarrowPathViolation as exc:
        print("NARROW-PATH GATE VIOLATION:", exc)
        return 2

    write_markdown(report_md, ARTIFACTS / "stage3_candidate_report.md")
    write_markdown(verification_md, ARTIFACTS / "STAGE3_VERIFICATION.md")
    write_markdown(selfcheck_md, ARTIFACTS / "stage3_geometry_selfcheck.md")

    outputs = [
        TABLES / "stage3_candidate_features.parquet",
        TABLES / "stage3_support_rejection.parquet",
        ARTIFACTS / "stage3_candidate_report.md",
        ARTIFACTS / "STAGE3_VERIFICATION.md",
        ARTIFACTS / "stage3_geometry_selfcheck.md",
        ARTIFACTS / "stage3_fold_flight_models.json",
        ARTIFACTS / "stage3_game_summaries.json",
        PACKAGE_ROOT / "configs" / "stage2_narrow_path.yaml",
    ]
    # Per-game caches for reconstructed games
    for gid in reconstruct_ids:
        p = _game_dir(gid) / FEATURES_NAME
        if p.exists():
            outputs.append(p)

    manifest = build_output_manifest(
        outputs,
        root=PACKAGE_ROOT,
        manifest_path=manifest_path,
        sign_off_paths=[ARTIFACTS / "STAGE3_VERIFICATION.md"],
    )
    manifest["freeze_id"] = FREEZE_ID
    manifest["stage"] = STAGE
    manifest["narrow_path_lock_id"] = lock.get("lock_id")
    manifest["trajectory_choice_rule"] = TRAJECTORY_RULE
    manifest["n_candidate_rows"] = summary.get("n_candidate_rows")
    manifest["elapsed_seconds"] = summary["elapsed_seconds_total"]
    manifest["self_checks_ok"] = checks.get("ok")
    write_json(manifest, manifest_path)

    verification = verify_output_manifest(manifest_path, root=PACKAGE_ROOT)
    print(f"Stage 3 complete in {time.time() - t0:.1f}s")
    print(
        f"Rows={summary.get('n_candidate_rows')} | "
        f"rejected={summary.get('pct_rejected_outside_support', float('nan')):.2f}% | "
        f"missing_matchup={summary.get('pct_missing_matchup', float('nan')):.2f}% | "
        f"traj={summary.get('trajectory_mix')}"
    )
    if not checks.get("ok"):
        print("SELF-CHECK FAILURES:", {k: v for k, v in checks.items() if k != "ok" and not v.get("ok")})
        return 2
    if not verification.get("ok"):
        print("MANIFEST VERIFICATION FAILED:", verification)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
