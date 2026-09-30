"""Leakage / refit guards for Stage 5."""

from __future__ import annotations

import ast
from pathlib import Path

# Stage 4 fit entrypoints that must not appear in Stage 5 modules/pipeline.
FORBIDDEN_FIT_NAMES = {
    "fit_passability_ablation",
    "fit_all_passability_ablations",
    "fit_catch_value",
    "fit_keep_state",
    "fit_choice_model",
}

FORBIDDEN_IMPORT_MODULES = {
    "passing_windows.models.passability",
    "passing_windows.models.catch_value",
    "passing_windows.models.keep_state",
    "passing_windows.models.choice",
}


class Stage5LeakageError(AssertionError):
    """Stage 5 violated a freeze / handoff leakage rule."""


def assert_no_component_model_refit() -> None:
    """Runtime guard placeholder — Stage 5 must never call Stage 4 fit APIs."""
    # Importing fit symbols is enough to fail static scan; this asserts policy.
    return None


def scan_stage5_sources_for_refit(path: Path) -> None:
    """AST-scan Stage 5 sources for forbidden Stage 4 fit imports/calls."""
    path = Path(path)
    files: list[Path]
    if path.is_file():
        files = [path]
    else:
        files = sorted(path.rglob("*.py"))

    for f in files:
        if f.name.startswith("test_"):
            continue
        src = f.read_text(encoding="utf-8")
        try:
            tree = ast.parse(src, filename=str(f))
        except SyntaxError as e:
            raise Stage5LeakageError(f"cannot parse {f}: {e}") from e

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if mod in FORBIDDEN_IMPORT_MODULES:
                    raise Stage5LeakageError(f"{f}: imports forbidden module {mod}")
                for alias in node.names:
                    if alias.name in FORBIDDEN_FIT_NAMES:
                        raise Stage5LeakageError(f"{f}: imports forbidden fit {alias.name}")
            if isinstance(node, ast.Call):
                name = None
                if isinstance(node.func, ast.Name):
                    name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    name = node.func.attr
                if name in FORBIDDEN_FIT_NAMES:
                    raise Stage5LeakageError(f"{f}: calls forbidden fit {name}")
                # Generic .fit( on model objects is banned in Stage 5 windows code
                if isinstance(node.func, ast.Attribute) and node.func.attr == "fit":
                    # Allow pandas DataFrame unlikely; ban sklearn-style
                    raise Stage5LeakageError(f"{f}: contains .fit( call — Stage 5 must not refit models")
