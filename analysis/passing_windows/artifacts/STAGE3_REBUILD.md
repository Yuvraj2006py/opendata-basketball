# Stage 0–3 rebuild commands

Exact PowerShell commands to rebuild Stages 1–3 end-to-end from the project
venv. Stage 0 is freeze/config only (no pipeline). Run from
`analysis/passing_windows`.

## Prerequisites

```powershell
cd C:\Users\yuvi2\Downloads\opendata-basketball\analysis\passing_windows
.\.venv\Scripts\python.exe -c "import pandas, pyarrow, yaml, numpy; print('ok')"
```

Upstream raw data must exist at `..\..\data` (matches + tracking + events).
Stage 0 freeze: `configs/stage0_freeze.yaml`.
Narrow-path lock (binding): `configs/stage2_narrow_path.yaml`.

## Full rebuild (force)

```powershell
cd C:\Users\yuvi2\Downloads\opendata-basketball\analysis\passing_windows

.\.venv\Scripts\python.exe pipelines\01_build_canonical_data.py --force
.\.venv\Scripts\python.exe pipelines\01_build_canonical_data.py --verify-manifest

.\.venv\Scripts\python.exe pipelines\02_infer_failed_pass_targets.py --force
.\.venv\Scripts\python.exe pipelines\02_infer_failed_pass_targets.py --verify-manifest

.\.venv\Scripts\python.exe pipelines\03_build_candidate_states.py --force
.\.venv\Scripts\python.exe pipelines\03_build_candidate_states.py --verify-manifest
```

Expected exit codes: **0** for each command. Manifest verify exits **2** on hash mismatch.

## Restartable / cached rebuild

Omit `--force` to reuse per-game markers (`_DONE`, `_DONE_STAGE2`, `_DONE_STAGE3`)
and skip games whose feature tables already exist. Fold flight-time models in
Stage 3 always re-fit (cheap) so held-out games never leak.

```powershell
.\.venv\Scripts\python.exe pipelines\01_build_canonical_data.py
.\.venv\Scripts\python.exe pipelines\02_infer_failed_pass_targets.py
.\.venv\Scripts\python.exe pipelines\03_build_candidate_states.py
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_stage1.py tests\test_stage2.py tests\test_stage3.py -q
```

## Approximate runtime (10 games, force)

| Stage | Notes |
|---|---|
| 1 | Tracking stream + joins (longest I/O) |
| 2 | Target inference over failed + completed passes |
| 3 | ~2–3 min/game candidate geometry; ~20–30 min total |

## Stage 3 outputs to confirm

| Path | Role |
|---|---|
| `tables/by_game/{id}/stage3_candidate_features.parquet` | Per-game candidates |
| `tables/by_game/{id}/_DONE_STAGE3` | Cache marker |
| `tables/stage3_candidate_features.parquet` | Aggregate |
| `tables/stage3_support_rejection.parquet` | Compact reject flags |
| `artifacts/stage3_candidate_report.md` | Population + fold report |
| `artifacts/STAGE3_VERIFICATION.md` | Checklist |
| `artifacts/stage3_geometry_selfcheck.md` | Automated geometry checks |
| `artifacts/stage3_fold_flight_models.json` | LOGO flight models |
| `artifacts/stage3_output_manifest.json` | SHA-256 manifest |
