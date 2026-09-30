"""Independent Stage 2 pre-Stage-3 readiness audit.

Recomputes kill rates, lock triggers, gate columns, eligibility labels,
interceptor rules, reviewer-blank rules, simplex checks, and report phrasing
from on-disk tables — does not trust prior checklist marks.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from passing_windows.narrow_path import (  # noqa: E402
    GATE_COLUMNS,
    AUDIT_DEFERRED_WAIVED,
    NarrowPathViolation,
    assert_audit_sheet_on_disk_is_blank,
    assert_interceptor_never_target,
    assert_narrow_path_gates,
    assert_no_failed_precision_claim,
    evaluate_narrow_path,
    load_narrow_path_lock,
    receiver_completion_training_labels,
)
from passing_windows.paths import ARTIFACTS, TABLES  # noqa: E402

REVIEWER_COLUMNS = (
    "reviewer_decision",
    "reviewer_confidence",
    "reviewer_is_intended_target",
    "reviewer_reason",
)


def main() -> int:
    issues: list[str] = []
    notes: list[str] = []
    checks: dict[str, str] = {}

    lock = load_narrow_path_lock()
    elig = pd.read_parquet(TABLES / "stage2_label_eligibility.parquet")
    failed = pd.read_parquet(TABLES / "stage2_failed_pass_assignments.parquet")
    completed = pd.read_parquet(TABLES / "stage2_completed_pass_validation.parquet")
    probs = pd.read_parquet(TABLES / "stage2_target_probabilities.parquet")
    state = json.loads((ARTIFACTS / "stage2_narrow_path_state.json").read_text(encoding="utf-8"))
    audit_path = ARTIFACTS / "stage2_ambiguous_audit.csv"
    audit = pd.read_csv(audit_path)

    report_paths = {
        "inference_report": ARTIFACTS / "stage2_target_inference_report.md",
        "narrow_path_lock": ARTIFACTS / "STAGE2_NARROW_PATH_LOCK.md",
        "verification": ARTIFACTS / "STAGE2_VERIFICATION.md",
        "reaudit": ARTIFACTS / "STAGE2_REAUDIT_REPORT.md",
        "geometry_selfcheck": ARTIFACTS / "stage2_geometry_selfcheck.md",
        "audit_report": ARTIFACTS / "STAGE2_AUDIT_REPORT.md",
    }
    reports = {k: p.read_text(encoding="utf-8") for k, p in report_paths.items() if p.exists()}

    # --- Cohort / kill rates ---
    n_failed = len(failed)
    exp_failed = int(lock["kill_criterion"]["n_failed_passes"])
    n_auto = int((failed["decision"] == "auto_accept").sum())
    n_amb = int((failed["decision"] == "ambiguous").sum())
    n_unr = int((failed["decision"] == "unresolved").sum())
    pre_audit = (n_amb + n_unr) / n_failed * 100
    strict = n_unr / n_failed * 100
    if n_failed != exp_failed:
        issues.append(f"failed count {n_failed} != lock {exp_failed}")
    else:
        notes.append(f"failed count matches lock ({n_failed})")
    if abs(pre_audit - float(lock["kill_criterion"]["measured_unresolved_pre_audit_pct"])) > 0.25:
        issues.append(
            f"pre_audit pct drift live={pre_audit:.3f} lock={lock['kill_criterion']['measured_unresolved_pre_audit_pct']}"
        )
    if abs(strict - float(lock["kill_criterion"]["measured_unresolved_strict_pct"])) > 0.25:
        issues.append(
            f"strict pct drift live={strict:.3f} lock={lock['kill_criterion']['measured_unresolved_strict_pct']}"
        )
    checks["kill_rates"] = "PASS" if not any("pct drift" in i or "failed count" in i for i in issues) else "FAIL"

    # --- Narrow path triggers ---
    np_state = evaluate_narrow_path(
        {"unresolved_pre_audit_pct": pre_audit, "unresolved_strict_pct": strict},
        lock=lock,
        human_audit_status=lock["human_audit"]["status"],
    )
    if not np_state["narrow_path_active"]:
        issues.append("narrow_path_active False from live triggers")
    for trigger in (
        "pre_audit_unresolved_rate_exceeds_frozen_threshold",
        "human_audit_status_is_not_complete",
    ):
        if trigger not in np_state["triggers"]:
            issues.append(f"missing trigger {trigger}")
    if lock["human_audit"]["status"] != AUDIT_DEFERRED_WAIVED:
        issues.append(f"unexpected human_audit.status={lock['human_audit']['status']}")
    if not bool(state.get("narrow_path_active")):
        issues.append("stage2_narrow_path_state.json narrow_path_active is not True")
    checks["narrow_path_triggers"] = (
        "PASS"
        if not any(x in " ".join(issues) for x in ("narrow_path", "missing trigger", "human_audit"))
        else "CHECK"
    )

    # --- Gate columns (lock requires full set on every written_to table) ---
    for name, df in (
        ("eligibility", elig),
        ("failed", failed),
        ("completed", completed),
        ("probabilities", probs),
    ):
        missing = [c for c in GATE_COLUMNS if c not in df.columns]
        if missing:
            issues.append(f"{name} missing gate columns {missing}")
        else:
            notes.append(f"{name}: all {len(GATE_COLUMNS)} gate columns present")
    checks["gate_columns"] = (
        "PASS" if not any("missing gate columns" in i for i in issues) else "FAIL"
    )

    # --- Library fatal suite ---
    try:
        gate_summary = assert_narrow_path_gates(
            failed=failed,
            completed=completed,
            eligibility=elig,
            state=state if "narrow_path_active" in state else np_state,
        )
        notes.append(
            "assert_narrow_path_gates OK "
            f"(eligible={gate_summary.get('n_receiver_specific_eligible')}, "
            f"touch_only={gate_summary.get('n_touch_level_only')})"
        )
        checks["assert_narrow_path_gates"] = "PASS"
    except NarrowPathViolation as exc:
        issues.append(f"assert_narrow_path_gates: {exc}")
        checks["assert_narrow_path_gates"] = "FAIL"
    except Exception as exc:  # noqa: BLE001
        # State from JSON may lack keys expected by assert; merge with np_state.
        merged = {**np_state, **state}
        try:
            gate_summary = assert_narrow_path_gates(
                failed=failed,
                completed=completed,
                eligibility=elig,
                state=merged,
            )
            notes.append(
                "assert_narrow_path_gates OK (merged state) "
                f"(eligible={gate_summary.get('n_receiver_specific_eligible')}, "
                f"touch_only={gate_summary.get('n_touch_level_only')})"
            )
            checks["assert_narrow_path_gates"] = "PASS"
        except Exception as exc2:  # noqa: BLE001
            issues.append(f"assert_narrow_path_gates: {exc} / retry: {exc2}")
            checks["assert_narrow_path_gates"] = "FAIL"

    # --- Interceptor never target ---
    try:
        assert_interceptor_never_target(failed, name="failed")
        assert_interceptor_never_target(probs, name="probabilities", target_col="inferred_targetId")
        if {"candidateId", "interceptorId"}.issubset(probs.columns):
            clash = (
                pd.to_numeric(probs["candidateId"], errors="coerce")
                == pd.to_numeric(probs["interceptorId"], errors="coerce")
            ) & probs["candidateId"].notna() & probs["interceptorId"].notna()
            if int(clash.sum()):
                issues.append(f"probabilities: interceptor==candidateId on {int(clash.sum())} rows")
            else:
                notes.append("probabilities: interceptor never in candidate set")
        checks["interceptor_rule"] = "PASS"
    except NarrowPathViolation as exc:
        issues.append(str(exc))
        checks["interceptor_rule"] = "FAIL"

    # --- Training labels ---
    labels = receiver_completion_training_labels(elig)
    n_labels = len(labels)
    n_completed_known = int(
        ((elig["cohort"] == "completed") & elig["usable_as_receiver_completion_label"].fillna(False)).sum()
    )
    failed_elig = elig[elig["cohort"] == "failed"]
    if (labels["cohort"] == "failed").any():
        issues.append("training labels include failed cohort")
    if failed_elig["usable_as_receiver_completion_label"].fillna(False).any():
        issues.append("failed rows marked usable_as_receiver_completion_label")
    if not failed_elig["usable_for_touch_level_turnover"].fillna(False).all():
        issues.append("failed rows missing usable_for_touch_level_turnover")
    if n_labels != n_completed_known:
        issues.append(f"training labels {n_labels} != completed known {n_completed_known}")
    else:
        notes.append(f"Stage 3 training labels = {n_labels} completed known-receiver passes")
    auto = failed[failed["decision"] == "auto_accept"]
    if len(auto) and not auto["usable_as_named_failed_target"].fillna(False).all():
        issues.append("auto_accept rows missing usable_as_named_failed_target")
    if len(auto) and not auto["named_failed_target_requires_sensitivity_check"].fillna(False).all():
        issues.append("auto_accept rows missing sensitivity requirement flag")
    non_auto_named = failed[
        (failed["decision"] != "auto_accept") & failed["usable_as_named_failed_target"].fillna(False)
    ]
    if len(non_auto_named):
        issues.append(f"non-auto rows usable_as_named_failed_target: {len(non_auto_named)}")
    checks["label_policy"] = "PASS" if not any("label" in i.lower() or "auto_accept" in i or "failed rows" in i for i in issues[-12:]) else "CHECK"

    # --- Reviewer blank ---
    try:
        assert_audit_sheet_on_disk_is_blank(
            audit_path,
            REVIEWER_COLUMNS,
            human_audit_status=lock["human_audit"]["status"],
        )
        notes.append("audit sheet reviewer columns blank on disk")
        checks["reviewer_blank"] = "PASS"
    except NarrowPathViolation as exc:
        issues.append(str(exc))
        checks["reviewer_blank"] = "FAIL"

    # --- Probability simplex ---
    pcol = next(
        (
            c
            for c in ("target_probability", "probability", "p_target", "p", "prob")
            if c in probs.columns
        ),
        None,
    )
    if pcol is None:
        cand = [c for c in probs.columns if "prob" in c.lower()]
        issues.append(f"no probability column found; candidates={cand[:12]}")
        checks["simplex"] = "FAIL"
    else:
        sums = probs.groupby("passId")[pcol].sum()
        bad = int(((sums - 1.0).abs() > 1e-5).sum())
        if bad:
            issues.append(f"simplex violations on {bad} passes (col={pcol})")
            checks["simplex"] = "FAIL"
        else:
            notes.append(f"probabilities form simplex per pass ({pcol})")
            checks["simplex"] = "PASS"

    # --- LOGO folds ---
    mism = 0
    for _, row in failed[["fold_id", "gameId"]].drop_duplicates().iterrows():
        fid = str(row["fold_id"])
        gid = str(int(row["gameId"]))
        if f"holdout_{gid}" not in fid and gid not in fid:
            mism += 1
    if mism:
        issues.append(f"fold_id/gameId mismatches: {mism}")
        checks["logo_folds"] = "FAIL"
    else:
        notes.append("fold_id encodes leave-one-game-out holdout game")
        checks["logo_folds"] = "PASS"

    # --- Eligibility uniqueness ---
    dups = int(elig.duplicated(["passId"]).sum())
    if dups:
        issues.append(f"duplicate passIds in eligibility: {dups}")
        checks["eligibility_unique"] = "FAIL"
    else:
        notes.append("eligibility passId unique")
        checks["eligibility_unique"] = "PASS"

    # --- Report phrasing ---
    phrasing_fail = False
    for name, text in reports.items():
        try:
            assert_no_failed_precision_claim(text, name=name)
        except NarrowPathViolation as exc:
            issues.append(str(exc))
            phrasing_fail = True
    checks["report_phrasing"] = "FAIL" if phrasing_fail else "PASS"
    if not phrasing_fail:
        notes.append(f"no forbidden failed-precision phrasing in {len(reports)} reports")

    # --- Completed-pass diagnostics present and not oversold ---
    diag_cols = [c for c in completed.columns if any(x in c.lower() for x in ("agree", "correct", "accept", "top1"))]
    notes.append(f"completed diagnostic columns present: {diag_cols[:12]}")

    # Reconcile eligibility vs failed/completed counts
    elig_failed = int((elig["cohort"] == "failed").sum())
    elig_completed = int((elig["cohort"] == "completed").sum())
    if elig_failed != n_failed:
        issues.append(f"eligibility failed {elig_failed} != failed table {n_failed}")
    if elig_completed != len(completed):
        issues.append(f"eligibility completed {elig_completed} != completed table {len(completed)}")

    # Soft-check label_policy check key more carefully
    label_issue_markers = (
        "training labels",
        "usable_as_receiver_completion_label",
        "usable_for_touch_level",
        "usable_as_named_failed",
        "sensitivity",
        "non-auto",
    )
    if any(any(m in i for m in label_issue_markers) for i in issues):
        checks["label_policy"] = "FAIL"
    else:
        checks["label_policy"] = "PASS"

    if any("narrow_path" in i or "missing trigger" in i or "human_audit" in i for i in issues):
        checks["narrow_path_triggers"] = "FAIL"
    else:
        checks["narrow_path_triggers"] = "PASS"

    verdict = "PASS" if not issues else "FAIL"
    payload = {
        "verdict": verdict,
        "checks": checks,
        "issues": issues,
        "notes": notes,
        "counts": {
            "failed": n_failed,
            "auto_accept": n_auto,
            "ambiguous": n_amb,
            "unresolved": n_unr,
            "pre_audit_unresolved_pct": round(pre_audit, 4),
            "strict_unresolved_pct": round(strict, 4),
            "completed": len(completed),
            "probability_rows": len(probs),
            "eligibility_rows": len(elig),
            "stage3_training_labels": n_labels,
            "audit_sheet_rows": len(audit),
        },
        "narrow_path_state_live": np_state,
    }
    out_json = ARTIFACTS / "STAGE2_PRE_STAGE3_AUDIT.json"
    out_md = ARTIFACTS / "STAGE2_PRE_STAGE3_AUDIT.md"
    out_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Stage 2 pre–Stage 3 readiness audit",
        "",
        f"**Verdict:** `{verdict}`",
        "",
        "Fresh independent audit of on-disk Stage 2 tables against "
        "`configs/stage2_narrow_path.yaml` and Stage 0 kill/fallback rules.",
        "",
        "## Checks",
        "",
        "| Check | Result |",
        "|---|---|",
    ]
    for k, v in checks.items():
        lines.append(f"| `{k}` | **{v}** |")
    lines += [
        "",
        "## Counts",
        "",
        f"- Failed passes: **{n_failed}** (auto={n_auto}, ambiguous={n_amb}, unresolved={n_unr})",
        f"- Pre-audit unresolved: **{pre_audit:.2f}%** (freeze kill line 10%; breach → narrow path)",
        f"- Strict unresolved: **{strict:.2f}%**",
        f"- Stage 3 default training labels (`receiver_specific_eligible`): **{n_labels}**",
        "",
        "## Issues",
        "",
    ]
    if issues:
        lines.extend(f"- {i}" for i in issues)
    else:
        lines.append("- None.")
    lines += ["", "## Notes", ""]
    lines.extend(f"- {n}" for n in notes)
    lines += [
        "",
        "## Stage 3 go / no-go",
        "",
    ]
    if verdict == "PASS":
        lines += [
            "**CONDITIONAL GO** under `target_inference_mode: narrow_path` only.",
            "",
            "Mandatory constraints unchanged:",
            "1. Join `tables/stage2_label_eligibility.parquet`.",
            "2. Completion labels only via `receiver_completion_training_labels` / `receiver_specific_eligible`.",
            "3. Failed passes = touch-level turnover evidence by default.",
            "4. Auto-accepted failures opt-in only with exclusion sensitivity.",
            "5. Never fill reviewer columns; never cite completed-pass proxy as failed-pass precision.",
            "",
            "Remaining non-blocking residual risk: human failed-pass audit is still "
            "`deferred_waived` — that is intentional under the lock, not a silent miss.",
        ]
    else:
        lines += [
            "**NO-GO** until issues above are resolved.",
        ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"verdict": verdict, "n_issues": len(issues), "checks": checks, "counts": payload["counts"]}, indent=2))
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
