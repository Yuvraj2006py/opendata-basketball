#!/usr/bin/env python
"""Live Stage 5 adversarial checks. Safe to re-run after builder lands outputs.

    python pipelines/_stage5_adversarial_live_audit.py
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

STAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE_ROOT / "src"))

from passing_windows.narrow_path import GATE_COLUMNS  # noqa: E402

ARTIFACTS = STAGE_ROOT / "artifacts"
TABLES = STAGE_ROOT / "tables"
WINDOWS_PKG = STAGE_ROOT / "src" / "passing_windows" / "windows"
PIPELINE = STAGE_ROOT / "pipelines" / "05_segment_windows.py"
TEST_FILE = STAGE_ROOT / "tests" / "test_stage5.py"
OUT_MD = ARTIFACTS / "STAGE5_ADVERSARIAL_AUDIT.md"
OUT_LIVE = ARTIFACTS / "STAGE5_LIVE_AUDIT.json"
REGRESSION_LOCK = ARTIFACTS / "STAGE5_REGRESSION_LOCK.md"

FORBIDDEN_CLAIM_RE = re.compile(
    r"\b(correct pass|bad decision|points left on the table)\b",
    re.IGNORECASE,
)
RANDOM_SPLIT_RE = re.compile(
    r"train_test_split|GroupShuffleSplit|ShuffleSplit|KFold\(",
    re.IGNORECASE,
)
AGGREGATE_RE = re.compile(r"aggregates[/\\]acb_.*aggregates", re.IGNORECASE)
FORBIDDEN_FIT = {
    "fit_passability_ablation",
    "fit_all_passability_ablations",
    "fit_catch_value",
    "fit_keep_state",
    "fit_choice_model",
}


def _check(name: str, ok: bool, detail: str = "") -> dict:
    return {"check": name, "status": "PASS" if ok else "FAIL", "detail": detail}


def main() -> int:
    results: list[dict] = []
    blockers_failed: list[str] = []

    required_tables = [
        TABLES / "stage5_candidate_series.parquet",
        TABLES / "stage5_windows.parquet",
        TABLES / "stage5_option_set_frames.parquet",
        TABLES / "stage5_threshold_selections.parquet",
        TABLES / "stage5_threshold_grid_windows.parquet",
    ]
    missing = [str(p.name) for p in required_tables if not p.exists()]
    results.append(_check("tables_present", not missing, ",".join(missing) or "all present"))
    if missing:
        blockers_failed.append("S5-tables")

    for path in (PIPELINE, WINDOWS_PKG, TEST_FILE, REGRESSION_LOCK):
        ok = path.exists()
        results.append(_check(f"exists:{path.name}", ok))
        if not ok:
            blockers_failed.append(path.name)

    # Source scans
    src_files = list(WINDOWS_PKG.rglob("*.py")) + ([PIPELINE] if PIPELINE.exists() else [])
    random_hits = []
    agg_hits = []
    fit_hits = []
    for f in src_files:
        text = f.read_text(encoding="utf-8")
        if RANDOM_SPLIT_RE.search(text):
            random_hits.append(f.name)
        if AGGREGATE_RE.search(text):
            agg_hits.append(f.name)
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name in FORBIDDEN_FIT:
                        fit_hits.append(f"{f.name}:{alias.name}")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in FORBIDDEN_FIT:
                    fit_hits.append(f"{f.name}:{node.func.id}")
    results.append(_check("no_random_splits", not random_hits, str(random_hits)))
    results.append(_check("no_season_aggregates", not agg_hits, str(agg_hits)))
    results.append(_check("no_component_refit", not fit_hits, str(fit_hits)))
    if random_hits:
        blockers_failed.append("S5-B19")
    if fit_hits:
        blockers_failed.append("S5-B02")

    # Forbidden language in Stage 5 builder/ship markdown (exclude audit briefs quoting rules)
    claim_hits = []
    skip = {"STAGE5_AUDIT_BRIEF_FOR_BUILDER.md", "STAGE5_ADVERSARIAL_AUDIT.md"}
    for md in list(ARTIFACTS.glob("STAGE5*.md")) + list(ARTIFACTS.glob("stage5*.md")):
        if md.name in skip:
            continue
        if FORBIDDEN_CLAIM_RE.search(md.read_text(encoding="utf-8")):
            claim_hits.append(md.name)
    results.append(_check("forbidden_claim_language", not claim_hits, str(claim_hits)))
    if claim_hits:
        blockers_failed.append("S5-B14")

    # Live table checks
    if (TABLES / "stage5_candidate_series.parquet").exists():
        import pandas as pd

        series = pd.read_parquet(TABLES / "stage5_candidate_series.parquet")
        windows = pd.read_parquet(TABLES / "stage5_windows.parquet")
        thr = pd.read_parquet(TABLES / "stage5_threshold_selections.parquet")
        grid = pd.read_parquet(TABLES / "stage5_threshold_grid_windows.parquet")

        results.append(_check("series_5hz_only", bool(series["is_model_5hz_frame"].fillna(False).all())))
        results.append(_check("n_series_176004", len(series) == 176004, str(len(series))))
        results.append(_check("ten_folds", len(thr) == 10 and thr["fold_id"].nunique() == 10, str(len(thr))))
        results.append(_check("grid_rows_ge_50", len(grid) >= 50, str(len(grid))))
        missing_gates = [c for c in GATE_COLUMNS if c not in series.columns]
        results.append(_check("gate_columns", not missing_gates, str(missing_gates)))
        for c in ("q_smooth", "V_catch_smooth", "Q_smooth", "NOV_smooth"):
            results.append(_check(f"col_{c}", c in series.columns and series[c].notna().any()))
        if "rejected_outside_support" in series.columns:
            # No open episode may be built only from rejected — covered by unit; check rate reported
            rej = float(series["rejected_outside_support"].fillna(False).mean())
            results.append(_check("rejection_rate_finite", 0.0 <= rej <= 1.0, f"{rej:.4f}"))
        labels = set(windows["label"].dropna().unique()) if "label" in windows.columns else set()
        results.append(
            _check(
                "labels_subset",
                labels.issubset({"open", "used", "late", "unused", "never_open"}),
                str(sorted(labels)),
            )
        )
        if missing_gates:
            blockers_failed.append("S5-B12")
        if len(thr) != 10:
            blockers_failed.append("S5-B21")
        if not bool(series["is_model_5hz_frame"].fillna(False).all()):
            blockers_failed.append("S5-B03")

    # Regression lock covers B00-B23
    if REGRESSION_LOCK.exists():
        lock = REGRESSION_LOCK.read_text(encoding="utf-8")
        missing_ids = [f"S5-B{i:02d}" for i in range(0, 24) if f"S5-B{i:02d}" not in lock]
        results.append(_check("regression_lock_blockers", not missing_ids, str(missing_ids)))
        if missing_ids:
            blockers_failed.append("S5-B23")

    # Manifest
    man = ARTIFACTS / "stage5_output_manifest.json"
    if man.exists():
        data = json.loads(man.read_text(encoding="utf-8"))
        results.append(_check("manifest_n_files", int(data.get("n_files", 0)) >= 5, str(data.get("n_files"))))
    else:
        results.append(_check("manifest_n_files", False, "missing"))
        blockers_failed.append("S5-B22")

    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    verdict = "SAFE TO ACCEPT" if n_fail == 0 and not blockers_failed else "CONDITIONAL / FIX REQUIRED"
    note = "live audit complete" if n_fail == 0 else f"{n_fail} checks failed; blockers={blockers_failed}"

    md = "\n".join(
        [
            "# Stage 5 adversarial audit (Passing Windows)",
            "",
            f"**Auditor role:** live adversarial checks (`pipelines/_stage5_adversarial_live_audit.py`)",
            f"**Freeze:** `passing_windows_stage0_20250924`",
            f"**Narrow-path lock:** `passing_windows_stage2_narrow_path_20260924`",
            f"**Upstream model:** `stage4_sklearn_logo_v3`",
            f"**Brief:** `artifacts/STAGE5_AUDIT_BRIEF_FOR_BUILDER.md`",
            "",
            f"## Verdict: **{verdict}** ({note})",
            "",
            "### Live checks",
            "",
            "| Check | Status | Detail |",
            "|---|---|---|",
        ]
    )
    for r in results:
        md += f"\n| `{r['check']}` | **{r['status']}** | {r.get('detail', '')} |"
    md += "\n\n### Residual / deferred\n\n"
    md += "- Existence probability under tracking perturbation → Stage 7\n"
    md += "- Label-stability kill (>15% flip) adjudication → Stage 7\n"
    md += "- Full geometry Monte Carlo → Stage 7\n"
    md += "\n### Go / no-go language\n\n"
    if n_fail == 0:
        md += "Stage 5 may proceed to Stage 6 under the narrow-path lock. "
        md += "Do not claim unused windows are mistakes; keep q and V_catch visible.\n"
    else:
        md += "Do not start Stage 6 headline claims until FAIL checks are cleared.\n"

    OUT_MD.write_text(md, encoding="utf-8")
    OUT_LIVE.write_text(
        json.dumps({"verdict": verdict, "note": note, "results": results, "blockers_failed": blockers_failed}, indent=2),
        encoding="utf-8",
    )
    print(verdict, note)
    for r in results:
        if r["status"] == "FAIL":
            print("FAIL", r["check"], r.get("detail"))
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
