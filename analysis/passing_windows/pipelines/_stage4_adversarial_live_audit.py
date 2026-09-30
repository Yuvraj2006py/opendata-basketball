#!/usr/bin/env python
"""Live Stage 4 adversarial checks (auditor). Safe to re-run after builder lands.

Does not implement Stage 4 models. Writes/updates evidence under artifacts/.

Until Stage 4 outputs exist, exits 2 with a clear missing-artifact message.

    .\\.venv\\Scripts\\python.exe pipelines\\_stage4_adversarial_live_audit.py
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

STAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE_ROOT / "src"))

OUT_MD = STAGE_ROOT / "artifacts" / "STAGE4_ADVERSARIAL_AUDIT.md"
OUT_JSON = STAGE_ROOT / "artifacts" / "STAGE4_AUDIT_CHECKLIST.json"
OUT_LIVE = STAGE_ROOT / "artifacts" / "STAGE4_LIVE_AUDIT.json"
REGRESSION_LOCK = STAGE_ROOT / "artifacts" / "STAGE4_REGRESSION_LOCK.md"

PIPELINE = STAGE_ROOT / "pipelines" / "04_fit_component_models.py"
TEST_FILE = STAGE_ROOT / "tests" / "test_stage4.py"
TABLES = STAGE_ROOT / "tables"
ARTIFACTS = STAGE_ROOT / "artifacts"

FORBIDDEN_CLAIM_RE = re.compile(
    r"\b(correct pass|bad decision|points left on the table)\b",
    re.IGNORECASE,
)
FAILED_PRECISION_PROXY_RE = re.compile(
    r"(failed[- ]pass[^\n]{0,80}(precision|agreement)|"
    r"(precision|agreement)[^\n]{0,80}failed[- ]pass)",
    re.IGNORECASE,
)
PTS_SCORED_FEATURE_RE = re.compile(r"ptsScored", re.IGNORECASE)
AGGREGATE_CSV_RE = re.compile(r"aggregates[/\\]acb_.*aggregates", re.IGNORECASE)
RANDOM_SPLIT_RE = re.compile(
    r"train_test_split|GroupShuffleSplit|ShuffleSplit|KFold\(",
    re.IGNORECASE,
)


def _append_live_section(md_path: Path, section: str) -> None:
    text = md_path.read_text(encoding="utf-8") if md_path.exists() else ""
    marker = "## Live verification"
    if marker in text:
        pre = text.split(marker)[0].rstrip()
        # Keep go/no-go language after live section if present
        rest = text.split(marker, 1)[1]
        if "## Go / no-go language" in rest:
            post = "## Go / no-go language" + rest.split("## Go / no-go language", 1)[1]
            text = pre + "\n\n" + section + "\n\n" + post
        else:
            text = pre + "\n\n" + section + "\n"
    else:
        text = text.rstrip() + "\n\n" + section + "\n"
    md_path.write_text(text, encoding="utf-8")


def _update_verdict_header(md_path: Path, verdict: str, note: str) -> None:
    if not md_path.exists():
        return
    text = md_path.read_text(encoding="utf-8")
    text = re.sub(
        r"## Verdict: \*\*[^*]+\*\*[^\n]*",
        f"## Verdict: **{verdict}** ({note})",
        text,
        count=1,
    )
    md_path.write_text(text, encoding="utf-8")


def _missing_artifacts() -> dict[str, list[str]]:
    missing: dict[str, list[str]] = {
        "pipeline": [],
        "tests": [],
        "tables": [],
        "artifacts": [],
    }
    if not PIPELINE.exists():
        missing["pipeline"].append(str(PIPELINE.relative_to(STAGE_ROOT).as_posix()))
    if not TEST_FILE.exists():
        missing["tests"].append(str(TEST_FILE.relative_to(STAGE_ROOT).as_posix()))
    stage4_tables = sorted(TABLES.glob("stage4_*.parquet")) if TABLES.exists() else []
    if not stage4_tables:
        missing["tables"].append("tables/stage4_*.parquet")
    stage4_arts = sorted(ARTIFACTS.glob("stage4_*")) if ARTIFACTS.exists() else []
    # Exclude auditor-owned STAGE4_* uppercase docs
    stage4_arts = [p for p in stage4_arts if p.name.startswith("stage4_")]
    if not stage4_arts:
        missing["artifacts"].append("artifacts/stage4_*")
    return missing


def _flatten_missing(missing: dict[str, list[str]]) -> list[str]:
    out: list[str] = []
    for items in missing.values():
        out.extend(items)
    return out


def _scan_source_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _collect_stage4_sources() -> list[Path]:
    paths: list[Path] = []
    if PIPELINE.exists():
        paths.append(PIPELINE)
    src = STAGE_ROOT / "src" / "passing_windows"
    models = src / "models"
    if models.exists():
        paths.extend(sorted(models.glob("*.py")))
    if src.exists():
        for p in src.rglob("*.py"):
            name = p.name.lower()
            if any(
                k in name
                for k in (
                    "stage4",
                    "passability",
                    "catch_value",
                    "keep",
                    "choice",
                    "component",
                    "nov",
                    "leakage",
                )
            ):
                if p not in paths:
                    paths.append(p)
    return paths


def _check_static_source(checks: dict) -> None:
    sources = _collect_stage4_sources()
    blob = "\n".join(_scan_source_text(p) for p in sources)
    checks["source_scan"] = {
        "ok": True,
        "n_files": len(sources),
        "files": [str(p.relative_to(STAGE_ROOT).as_posix()) for p in sources],
    }
    if not sources:
        checks["source_scan"]["ok"] = False
        checks["source_scan"]["detail"] = "no Stage 4 sources found"
        return

    checks["no_random_group_split"] = {
        "ok": RANDOM_SPLIT_RE.search(blob) is None
        or ("LeaveOneGroupOut" in blob or "logo" in blob.lower() or "held_out" in blob),
        "detail": "flag raw ShuffleSplit/train_test_split if used without LOGO wrapper",
    }
    # Soft: if ShuffleSplit appears AND no logo markers → fail
    if RANDOM_SPLIT_RE.search(blob) and not any(
        k in blob for k in ("held_out_gameId", "train_gameIds", "fold_holdout", "LeaveOneGroupOut")
    ):
        checks["no_random_group_split"]["ok"] = False

    checks["no_season_aggregates"] = {
        "ok": AGGREGATE_CSV_RE.search(blob) is None,
    }
    checks["outcome_not_ptsScored_as_feature"] = {
        "ok": "chances.ptsScored" not in blob
        or "never" in blob.lower()
        or "forbid" in blob.lower(),
        "detail": "ptsScored may appear only as prohibition text",
    }
    if re.search(r"""['\"]ptsScored['\"]""", blob) and "forbid" not in blob.lower():
        # feature-like string use without forbid nearby → soft fail signal
        checks["outcome_not_ptsScored_as_feature"]["ok"] = False

    checks["shotQuality_not_feature"] = {"ok": True}
    if re.search(r"shotQuality", blob):
        # Allowed in forbidden-feature registries and prohibition comments.
        # Fail only if used as an active feature-list entry outside FORBIDDEN_*.
        active = False
        for m in re.finditer(
            r"(PASSABILITY_FEATURES|CATCH_VALUE_FEATURES|KEEP_STATE_FEATURES|"
            r"feature_cols|FEATURE_COLUMNS)\s*[:=\[][^\]]*shotQuality",
            blob,
            re.I | re.S,
        ):
            active = True
        # Also catch X[["shotQuality"]] style
        if re.search(r"""\[\s*['\"]shotQuality['\"]\s*\]|\.get\(\s*['\"]shotQuality['\"]""", blob):
            active = True
        checks["shotQuality_not_feature"] = {
            "ok": not active,
            "detail": "shotQuality may appear only in forbidden lists / comments",
        }

    # Prohibition phrases ("not/never … score-max") are fine.
    # Fail only on affirmative selection of score-max as the trajectory rule.
    affirmative = re.search(
        r"(choose|select|switch(?:es|ed)?|use[sd]?|pick(?:s|ed)?)\s+[^\n]{0,40}score[-_ ]?max",
        blob,
        re.IGNORECASE,
    )
    # Rule must live in Stage 3/4 code or predictions — not only in prose reports.
    code_paths = [
        STAGE_ROOT / "src" / "passing_windows" / "candidates.py",
        STAGE_ROOT / "pipelines" / "04_fit_component_models.py",
    ]
    code_blob = "\n".join(p.read_text(encoding="utf-8") for p in code_paths if p.exists())
    has_rule = "lead_if_velocity_ok_else_direct" in code_blob or "lead_if_velocity_ok_else_direct" in blob
    checks["trajectory_rule_preserved"] = {
        "ok": affirmative is None and has_rule,
        "detail": "require lead_if_velocity_ok_else_direct in code; prohibition phrasing OK",
    }


def _check_tables(checks: dict) -> None:
    import pandas as pd

    from passing_windows.narrow_path import (  # noqa: WPS433
        GATE_COLUMNS,
        load_narrow_path_lock,
        receiver_completion_training_labels,
    )
    from passing_windows.ingest import load_freeze  # noqa: WPS433

    freeze = load_freeze()
    lock = load_narrow_path_lock()
    checks["narrow_path_lock_id"] = lock.get("lock_id")
    checks["freeze_id"] = freeze.get("freeze_id")

    tables = sorted(TABLES.glob("stage4_*.parquet"))
    checks["stage4_tables"] = {
        "ok": len(tables) > 0,
        "files": [p.name for p in tables],
    }
    if not tables:
        return

    # Prefer a scored candidate / predictions table if present
    preferred = None
    for name in (
        "stage4_candidate_scores.parquet",
        "stage4_scored_candidates.parquet",
        "stage4_predictions.parquet",
        "stage4_component_scores.parquet",
    ):
        cand = TABLES / name
        if cand.exists():
            preferred = cand
            break
    if preferred is None:
        preferred = tables[0]

    df = pd.read_parquet(preferred)
    checks["primary_table"] = {
        "ok": True,
        "path": preferred.name,
        "n_rows": int(len(df)),
        "columns": list(df.columns)[:80],
    }

    present_gates = [c for c in GATE_COLUMNS if c in df.columns]
    checks["gate_columns"] = {
        "ok": len(present_gates) == len(GATE_COLUMNS),
        "present": present_gates,
        "missing": [c for c in GATE_COLUMNS if c not in df.columns],
    }

    if "completion_label_eligible" in df.columns:
        if "cohort" in df.columns:
            bad = df["completion_label_eligible"].fillna(False) & (df["cohort"] == "failed")
            checks["no_failed_completion_labels"] = {
                "ok": int(bad.sum()) == 0,
                "n_bad": int(bad.sum()),
            }
        else:
            feat_path = TABLES / "stage3_candidate_features.parquet"
            if feat_path.exists() and {"gameId", "touchId", "frameIdx", "candidateId"}.issubset(
                df.columns
            ):
                feat = pd.read_parquet(feat_path)
                need = ["gameId", "touchId", "frameIdx", "candidateId"]
                if "cohort" not in feat.columns:
                    checks["no_failed_completion_labels"] = {
                        "ok": True,
                        "detail": "completion_label_eligible present; cohort absent upstream",
                    }
                else:
                    labeled = df.loc[
                        df["completion_label_eligible"].fillna(False),
                        need,
                    ]
                    merged = labeled.merge(
                        feat[need + ["cohort"]],
                        on=need,
                        how="left",
                    )
                    n_bad = int((merged["cohort"] == "failed").sum())
                    checks["no_failed_completion_labels"] = {
                        "ok": n_bad == 0,
                        "n_bad": n_bad,
                        "detail": "cohort joined from stage3",
                    }
            else:
                checks["no_failed_completion_labels"] = {
                    "ok": True,
                    "detail": "completion_label_eligible present; cohort not on Stage 4 table",
                }
    elif "receiver_specific_eligible" in df.columns and "cohort" in df.columns:
        bad = df["receiver_specific_eligible"].fillna(False) & (df["cohort"] == "failed")
        checks["no_failed_completion_labels"] = {
            "ok": int(bad.sum()) == 0,
            "n_bad": int(bad.sum()),
        }
    else:
        checks["no_failed_completion_labels"] = {
            "ok": False,
            "detail": "missing completion_label_eligible",
        }

    for col in ("q_passability_model", "q_hat", "q_passability"):
        if col in df.columns:
            n_finite = int(df[col].notna().sum())
            checks["q_oos_predictions"] = {
                "ok": n_finite > 0,
                "column": col,
                "n_finite": n_finite,
            }
            break
    else:
        checks["q_oos_predictions"] = {"ok": False, "detail": "no q column found"}

    for cols in (
        ("V_catch_model", "NOV_model"),
        ("V_catch", "NOV"),
        ("v_catch", "nov"),
    ):
        if all(c in df.columns for c in cols):
            ok = all(int(df[c].notna().sum()) > 0 for c in cols)
            checks["components_not_nov_only"] = {
                "ok": ok,
                "columns": list(cols),
            }
            break
    else:
        checks["components_not_nov_only"] = {
            "ok": False,
            "detail": "V_catch/NOV columns missing",
        }

    if "trajectory_choice_rule" in df.columns:
        checks["trajectory_rule"] = {
            "ok": bool(
                (df["trajectory_choice_rule"] == "lead_if_velocity_ok_else_direct").all()
            )
        }
    else:
        checks["trajectory_rule"] = {"ok": True, "detail": "column absent on this table"}

    if "rejected_outside_support" in df.columns:
        rej = df["rejected_outside_support"].fillna(False).astype(bool)
        n_rej = int(rej.sum())
        if n_rej and "Q_option_model" in df.columns and "NOV_model" in df.columns:
            q_ok = bool(df.loc[rej, "Q_option_model"].isna().all())
            nov_ok = bool(df.loc[rej, "NOV_model"].isna().all())
            checks["support_rejection"] = {
                "ok": q_ok and nov_ok,
                "n_rejected": n_rej,
                "q_nan_rate": float(df.loc[rej, "Q_option_model"].isna().mean()),
                "nov_nan_rate": float(df.loc[rej, "NOV_model"].isna().mean()),
            }
        else:
            checks["support_rejection"] = {
                "ok": True,
                "n_rejected": n_rej,
                "detail": "flag present; Q/NOV mask not checked (columns missing or n_rej=0)",
            }
    else:
        checks["support_rejection"] = {
            "ok": False,
            "detail": "rejected_outside_support missing",
        }

    # Release-label yield vs Stage 2/3 eligibility
    eligibility_path = TABLES / "stage2_label_eligibility.parquet"
    features_path = TABLES / "stage3_candidate_features.parquet"
    if eligibility_path.exists() and features_path.exists():
        eligibility = pd.read_parquet(eligibility_path)
        labels = receiver_completion_training_labels(eligibility)
        features = pd.read_parquet(features_path)
        if "passId" in features.columns and "is_pass_release_frame" in features.columns:
            release_ids = set(
                features.loc[features["is_pass_release_frame"].fillna(False), "passId"]
                .dropna()
                .astype(str)
            )
            label_ids = set(labels["passId"].astype(str))
            n_hit = len(label_ids & release_ids)
            checks["release_label_yield"] = {
                "ok": n_hit >= int(0.9 * len(labels)),
                "n_labels": int(len(labels)),
                "n_hit": n_hit,
            }
        else:
            checks["release_label_yield"] = {
                "ok": False,
                "detail": "stage3 missing release columns",
            }
    else:
        checks["release_label_yield"] = {
            "ok": False,
            "detail": "stage2/stage3 upstream missing",
        }

    # Manifest
    manifest_path = ARTIFACTS / "stage4_output_manifest.json"
    if manifest_path.exists():
        from passing_windows.io_utils import verify_output_manifest  # noqa: WPS433
        from passing_windows.paths import PACKAGE_ROOT  # noqa: WPS433

        checks["manifest"] = verify_output_manifest(manifest_path, root=PACKAGE_ROOT)
    else:
        checks["manifest"] = {"ok": False, "detail": "missing stage4_output_manifest.json"}

    # Report phrasing (builder artifacts only; skip auditor-owned STAGE4_* docs)
    report_hits = []
    skip_names = {
        "STAGE4_ADVERSARIAL_AUDIT.md",
        "STAGE4_AUDIT_CHECKLIST.json",
        "STAGE4_REGRESSION_LOCK.md",
        "STAGE4_LIVE_AUDIT.json",
    }
    for p in ARTIFACTS.glob("stage4_*"):
        if p.name in skip_names or p.name.startswith("STAGE4_ADVERSARIAL") or p.name.startswith("STAGE4_AUDIT") or p.name.startswith("STAGE4_REGRESSION") or p.name.startswith("STAGE4_LIVE"):
            continue
        if p.suffix.lower() not in {".md", ".txt", ".json"}:
            continue
        text = _scan_source_text(p)
        if FORBIDDEN_CLAIM_RE.search(text):
            report_hits.append({"file": p.name, "kind": "forbidden_claim"})
        if FAILED_PRECISION_PROXY_RE.search(text) and "diagnostic" not in text.lower():
            report_hits.append({"file": p.name, "kind": "failed_precision_proxy"})
    for p in ARTIFACTS.glob("STAGE4_*.md"):
        if p.name in skip_names or p.name in {
            "STAGE4_ADVERSARIAL_AUDIT.md",
            "STAGE4_REGRESSION_LOCK.md",
        }:
            continue
        # Builder verification/handoff still scanned
        if p.name in {"STAGE4_VERIFICATION.md", "STAGE4_HANDOFF_STAGE5.md"}:
            text = _scan_source_text(p)
            if FORBIDDEN_CLAIM_RE.search(text):
                report_hits.append({"file": p.name, "kind": "forbidden_claim"})
            if FAILED_PRECISION_PROXY_RE.search(text) and "diagnostic" not in text.lower():
                report_hits.append({"file": p.name, "kind": "failed_precision_proxy"})
    checks["report_phrasing"] = {"ok": len(report_hits) == 0, "hits": report_hits}

    # Full LOGO coverage: all ten freeze games must have predictions + markers
    from passing_windows.ingest import game_ids_from_freeze  # noqa: WPS433

    game_ids = game_ids_from_freeze(freeze)
    pred_games = sorted(int(g) for g in df["gameId"].dropna().unique()) if "gameId" in df.columns else []
    markers = sum(
        1 for g in game_ids if (TABLES / "by_game" / str(g) / "_DONE_STAGE4").exists()
    )
    checks["logo_ten_fold_coverage"] = {
        "ok": set(pred_games) == set(int(g) for g in game_ids) and markers == len(game_ids),
        "pred_games": pred_games,
        "n_markers": markers,
        "n_freeze_games": len(game_ids),
        "missing_games": sorted(set(int(g) for g in game_ids) - set(pred_games)),
    }

    # Handoff must not claim full Stage3 alignment while coverage is partial
    handoff = ARTIFACTS / "STAGE4_HANDOFF_STAGE5.md"
    if handoff.exists():
        ht = _scan_source_text(handoff)
        claims_align = "aligns 1:1" in ht.lower() or "1:1 with" in ht.lower()
        checks["handoff_coverage_honesty"] = {
            "ok": (not claims_align) or checks["logo_ten_fold_coverage"]["ok"],
            "detail": "handoff claims 1:1 Stage3 align but LOGO coverage incomplete"
            if claims_align and not checks["logo_ten_fold_coverage"]["ok"]
            else "ok",
        }
    else:
        checks["handoff_coverage_honesty"] = {"ok": False, "detail": "missing handoff"}



def _write_concrete_blockers(md_path: Path, blockers: list[dict]) -> None:
    if not md_path.exists():
        return
    text = md_path.read_text(encoding="utf-8")
    marker = "## Concrete blockers (live)"
    section_lines = [
        "## Concrete blockers (live)",
        "",
    ]
    if not blockers:
        section_lines.append("_No open live blockers._")
        section_lines.append("")
    else:
        for b in blockers:
            section_lines.extend(
                [
                    f"### {b['id']} — {b.get('title', b['id'])}",
                    "",
                    f"- **Severity:** {b.get('severity', 'BLOCKER')}",
                    f"- **Exact failure:** {b['exact_failure']}",
                    f"- **Evidence:** {b['evidence']}",
                    f"- **Required fix:** {b['required_fix']}",
                    f"- **Regression test:** `{b['regression_test']}`",
                    "",
                ]
            )
    new_section = "\n".join(section_lines)
    if marker in text:
        # Replace until Live verification
        pre = text.split(marker)[0].rstrip()
        rest = text.split(marker, 1)[1]
        if "## Live verification" in rest:
            post = "## Live verification" + rest.split("## Live verification", 1)[1]
            # Strip go/no-go from post if nested; _append_live_section handles later
            text = pre + "\n\n" + new_section + "\n" + post
        else:
            text = pre + "\n\n" + new_section + "\n"
    else:
        text = text.rstrip() + "\n\n" + new_section + "\n"
    md_path.write_text(text, encoding="utf-8")


def _blockers_from_checks(checks: dict, missing: list[str]) -> list[dict]:
    blockers: list[dict] = []
    if missing:
        blockers.append(
            {
                "id": "S4-B24" if any("04_fit" in m or "test_stage4" in m for m in missing) else "S4-B25",
                "title": "Missing Stage 4 artifacts",
                "severity": "BLOCKER",
                "exact_failure": "Required Stage 4 paths absent: " + ", ".join(missing),
                "evidence": "pipelines/_stage4_adversarial_live_audit.py:_missing_artifacts",
                "required_fix": "Land pipeline, tests, tables/stage4_*.parquet, and artifacts/stage4_*",
                "regression_test": "test_stage4_pipeline_entrypoint_exists",
            }
        )
        # Always emit both if both families missing
        if any("stage4_" in m for m in missing):
            blockers.append(
                {
                    "id": "S4-B25",
                    "title": "Stage 4 tables/artifacts missing",
                    "severity": "BLOCKER",
                    "exact_failure": "No stage4 parquet/artifact outputs on disk",
                    "evidence": "glob tables/stage4_*.parquet + artifacts/stage4_*",
                    "required_fix": "Emit scored tables + hashed manifest + verification",
                    "regression_test": "test_stage4_output_manifest_complete",
                }
            )
        return blockers

    mapping = [
        ("no_random_group_split", "S4-B01", "Random/group split without LOGO", "test_stage4_no_random_group_split"),
        ("no_season_aggregates", "S4-B23", "Season aggregate covariate use", "test_stage4_no_season_aggregates"),
        ("shotQuality_not_feature", "S4-B16", "shotQuality used as feature", "test_stage4_shotQuality_not_feature"),
        ("gate_columns", "S4-B08", "GATE_COLUMNS incomplete on Stage 4 table", "test_stage4_gate_columns_present"),
        ("no_failed_completion_labels", "S4-B06", "Failed cohort in completion training", "test_stage4_no_failed_completion_labels"),
        ("q_oos_predictions", "S4-B11", "q OOS predictions missing", "test_stage4_q_oos_predictions_present"),
        ("components_not_nov_only", "S4-B19", "q/V_catch/NOV components missing", "test_stage4_components_not_collapsed_to_nov"),
        ("trajectory_rule", "S4-B21", "Trajectory rule not lead_if_velocity_ok_else_direct", "test_stage4_trajectory_rule_unchanged"),
        ("support_rejection", "S4-B22", "rejected_outside_support not carried/masked", "test_stage4_support_rejection_masked"),
        ("release_label_yield", "S4-B05", "Release-label yield below 90%", "test_stage4_release_label_yield"),
        ("manifest", "S4-B25", "stage4_output_manifest missing or mismatched", "test_stage4_output_manifest_complete"),
        ("report_phrasing", "S4-B20", "Forbidden claim or failed-precision proxy language", "test_stage4_forbidden_claim_language"),
        ("logo_ten_fold_coverage", "S4-B27", "Fewer than 10 LOGO folds / games scored", "test_stage4_all_ten_logo_folds_present"),
        ("handoff_coverage_honesty", "S4-B28", "Handoff claims full Stage3 align without full coverage", "test_stage4_handoff_matches_coverage"),
    ]
    for key, fid, title, test_name in mapping:
        v = checks.get(key)
        if not isinstance(v, dict):
            continue
        if v.get("ok"):
            continue
        blockers.append(
            {
                "id": fid,
                "title": title,
                "severity": "BLOCKER",
                "exact_failure": title + f" (check `{key}` failed)",
                "evidence": json.dumps({kk: vv for kk, vv in v.items() if kk != "ok"})[:500],
                "required_fix": f"Fix Stage 4 so live check `{key}` passes",
                "regression_test": test_name,
            }
        )
    return blockers


def main() -> int:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    missing_map = _missing_artifacts()
    missing = _flatten_missing(missing_map)

    checks: dict = {
        "ok": False,
        "stage4_present": len(missing) == 0,
        "missing": missing_map,
    }

    if missing:
        msg = (
            "STAGE4_MISSING_ARTIFACTS: Stage 4 outputs not found. "
            "Missing: " + ", ".join(missing) + ". "
            "Builder must land pipelines/04_fit_component_models.py, "
            "tests/test_stage4.py, tables/stage4_*.parquet, and artifacts/stage4_* "
            "before live audit can proceed."
        )
        print(msg, file=sys.stderr)
        checks["detail"] = msg
        blockers = _blockers_from_checks(checks, missing)
        section = (
            "## Live verification\n\n"
            "**Status:** FAIL — missing Stage 4 artifacts\n\n"
            f"```\n{msg}\n```\n\n"
            "| Check | Result | Detail |\n"
            "|---|---|---|\n"
            f"| `artifacts_present` | FAIL | {json.dumps(missing_map)} |\n"
        )
        _append_live_section(OUT_MD, section)
        _write_concrete_blockers(OUT_MD, blockers)
        _update_verdict_header(OUT_MD, "NO-GO", "Stage 4 artifacts missing — resume builder")

        if OUT_JSON.exists():
            doc = json.loads(OUT_JSON.read_text(encoding="utf-8"))
            doc["verdict"] = "NO-GO"
            doc["live_audit_ok"] = False
            doc["stage4_artifacts_present"] = False
            doc["stage4_pipeline_present"] = PIPELINE.exists()
            doc["stage4_tests_present"] = TEST_FILE.exists()
            doc["blockers_remaining"] = blockers
            doc["live_verification"] = {"ran": True, "checks": checks}
            doc["stage5_may_begin"] = False
            OUT_JSON.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")

        OUT_LIVE.write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(checks, indent=2))
        return 2

    # Stage 4 present — run live checks
    try:
        _check_static_source(checks)
        _check_tables(checks)
    except Exception as exc:  # noqa: BLE001 — auditor must report, not crash silently
        checks["runtime_error"] = {"ok": False, "error": repr(exc)}
        print(f"STAGE4_LIVE_AUDIT_ERROR: {exc!r}", file=sys.stderr)

    checks["ok"] = all(
        (v.get("ok") if isinstance(v, dict) else bool(v))
        for k, v in checks.items()
        if k
        not in (
            "ok",
            "stage4_present",
            "missing",
            "detail",
            "narrow_path_lock_id",
            "freeze_id",
            "primary_table",
        )
    )

    blockers = _blockers_from_checks(checks, [])
    verdict = "SAFE TO ACCEPT" if checks["ok"] and not blockers else (
        "CONDITIONAL GO" if checks.get("q_oos_predictions", {}).get("ok") and not any(
            b["id"].startswith("S4-B") for b in blockers
        )
        else "NO-GO"
    )
    # Stricter: any blocker → NO-GO
    if blockers:
        verdict = "NO-GO"

    lines = [
        "## Live verification",
        "",
        f"**Status:** {'PASS' if checks['ok'] else 'FAIL'}",
        f"**Verdict:** {verdict}",
        f"**Narrow-path lock:** `{checks.get('narrow_path_lock_id', '')}`",
        "",
        "| Check | Result | Detail |",
        "|---|---|---|",
    ]
    for k, v in checks.items():
        if k in (
            "ok",
            "stage4_present",
            "missing",
            "detail",
            "narrow_path_lock_id",
            "freeze_id",
        ):
            continue
        if not isinstance(v, dict):
            continue
        ok = v.get("ok")
        detail = json.dumps({kk: vv for kk, vv in v.items() if kk != "ok"})
        if len(detail) > 300:
            detail = detail[:297] + "..."
        lines.append(f"| `{k}` | {'PASS' if ok else 'FAIL'} | {detail} |")
    lines.append("")
    _append_live_section(OUT_MD, "\n".join(lines))
    _write_concrete_blockers(OUT_MD, blockers)
    _update_verdict_header(
        OUT_MD,
        verdict,
        "live audit complete" if checks["ok"] else "open blockers — resume builder",
    )

    if OUT_JSON.exists():
        doc = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        doc["verdict"] = verdict.replace(" ", "_")
        doc["live_audit_ok"] = bool(checks["ok"])
        doc["stage4_artifacts_present"] = True
        doc["stage4_pipeline_present"] = True
        doc["stage4_tests_present"] = TEST_FILE.exists()
        doc["blockers_remaining"] = blockers
        doc["live_verification"] = {"ran": True, "checks": checks}
        doc["stage5_may_begin"] = checks["ok"] and not blockers
        # Update checklist statuses for checks we evaluated
        status_map = {
            "S4-B01": "no_random_group_split",
            "S4-B06": "no_failed_completion_labels",
            "S4-B08": "gate_columns",
            "S4-B11": "q_oos_predictions",
            "S4-B16": "shotQuality_not_feature",
            "S4-B19": "components_not_nov_only",
            "S4-B21": "trajectory_rule",
            "S4-B22": "support_rejection",
            "S4-B23": "no_season_aggregates",
            "S4-B05": "release_label_yield",
            "S4-B25": "manifest",
            "S4-B20": "report_phrasing",
            "S4-B27": "logo_ten_fold_coverage",
            "S4-B28": "handoff_coverage_honesty",
            "S4-B24": None,
        }
        for item in doc.get("checklist", []):
            key = status_map.get(item["id"])
            if key is None and item["id"] == "S4-B24":
                item["status"] = "PASS" if PIPELINE.exists() and TEST_FILE.exists() else "FAIL"
                continue
            if key and key in checks:
                item["status"] = "PASS" if checks[key].get("ok") else "FAIL"
        OUT_JSON.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")

    OUT_LIVE.write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(checks, indent=2))
    return 0 if checks["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
