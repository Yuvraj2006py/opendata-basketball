"""Stage 2 narrow-path label gate — machine enforcement of the frozen fallback.

Stage 0 froze a kill criterion (`unresolved failed-pass targets exceed 10% OR
audit reliability is poor`) and a fallback (`on_inadequate_reliability`: retain
failed passes only for touch-level turnover modelling, restrict receiver-specific
completion calibration to outcomes with reliable targets).

Both halves of that criterion fired:

1. pre-audit unresolved is 29/89 = 32.6%, above the frozen 10% line;
2. no human review of failed-pass targets exists, so audit reliability is not
   poor — it is *unmeasured*, which the gate treats identically.

The narrow path is therefore the binding Stage 2 acceptance mode, locked in
`configs/stage2_narrow_path.yaml`. This module turns that lock into columns and
assertions so a downstream stage cannot silently use a provisional label: a
consumer that joins `tables/stage2_label_eligibility.parquet` and filters on
`receiver_specific_eligible` gets completed passes with known receivers, and
nothing else, by default.

The held-out completed-pass numbers (top-1 agreement, auto-accept precision) are
**diagnostics**. A completed pass reaches its receiver and is geometrically
easier than an intercepted one, so they bound failed-pass precision from above
rather than measuring it, and :func:`assert_no_failed_precision_claim` exists so
that misuse fails loudly rather than reading well.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import yaml

from .paths import CONFIGS
from .target_inference import DECISION_AMBIGUOUS, DECISION_AUTO_ACCEPT, DECISION_UNRESOLVED

NARROW_PATH_YAML = CONFIGS / "stage2_narrow_path.yaml"

LOCK_ID = "passing_windows_stage2_narrow_path_20260924"
MODE_NARROW_PATH = "narrow_path"
MODE_FULL = "full_receiver_specific"

# Human audit states. Only `complete` retires the narrow path, and nothing in
# Stage 2 may set it: it requires reviewer columns that a person filled in.
AUDIT_DEFERRED_WAIVED = "deferred_waived"
AUDIT_COMPLETE = "complete"

RELIABILITY_KNOWN_RECEIVER = "known_receiver"
RELIABILITY_INFERRED_PROVISIONAL = "inferred_provisional_unaudited"
RELIABILITY_AMBIGUOUS = "ambiguous_unreviewed"
RELIABILITY_NO_EVIDENCE = "no_evidence"

RELIABILITY_STATUSES = (
    RELIABILITY_KNOWN_RECEIVER,
    RELIABILITY_INFERRED_PROVISIONAL,
    RELIABILITY_AMBIGUOUS,
    RELIABILITY_NO_EVIDENCE,
)

GATE_COLUMNS: tuple[str, ...] = (
    "narrow_path_active",
    "target_inference_mode",
    "target_reliability_status",
    "human_audit_status",
    "usable_as_receiver_completion_label",
    "usable_as_named_failed_target",
    "named_failed_target_requires_sensitivity_check",
    "usable_for_touch_level_turnover",
    "receiver_specific_eligible",
    "label_gate_reason",
)

BOOLEAN_GATE_COLUMNS: tuple[str, ...] = (
    "narrow_path_active",
    "usable_as_receiver_completion_label",
    "usable_as_named_failed_target",
    "named_failed_target_requires_sensitivity_check",
    "usable_for_touch_level_turnover",
    "receiver_specific_eligible",
)

COHORT_FAILED = "failed"
COHORT_COMPLETED = "completed"

ELIGIBILITY_COLUMNS: tuple[str, ...] = (
    "passId",
    "gameId",
    "fold_id",
    "cohort",
    "pass_outcome_class",
    "passerId",
    "interceptorId",
    "true_receiverId",
    "inferred_targetId",
    "decision",
    "decision_reason",
    "p_top",
    "prob_margin",
    "geom_margin_ft",
    *GATE_COLUMNS,
)

REVIEWER_BLANK_SENTINELS = {"", "nan", "none", "<na>"}


class NarrowPathViolation(AssertionError):
    """A Stage 2 output broke a gate the narrow-path lock declares fatal."""


# ---------------------------------------------------------------------------
# Lock
# ---------------------------------------------------------------------------


def load_narrow_path_lock(path: Path | None = None) -> dict[str, Any]:
    with open(path or NARROW_PATH_YAML, encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate_narrow_path(
    unresolved: dict[str, Any],
    *,
    lock: dict[str, Any] | None = None,
    human_audit_status: str | None = None,
) -> dict[str, Any]:
    """Decide whether the narrow path is active for this run, and say why.

    The lock declares `target_inference_mode: narrow_path`, but the triggers are
    recomputed from the live run rather than trusted, so the gate cannot go
    stale if a future re-run resolves more targets or an audit is completed.
    """
    lock = lock or load_narrow_path_lock()
    audit_status = human_audit_status or lock["human_audit"]["status"]
    threshold = float(lock["kill_criterion"]["frozen_threshold_pct"])

    pre_audit_pct = float(unresolved.get("unresolved_pre_audit_pct", np.nan))
    strict_pct = float(unresolved.get("unresolved_strict_pct", np.nan))
    breaches_pre_audit = bool(np.isfinite(pre_audit_pct) and pre_audit_pct > threshold)
    breaches_strict = bool(np.isfinite(strict_pct) and strict_pct > threshold)
    audit_incomplete = audit_status != AUDIT_COMPLETE

    triggers: list[str] = []
    if breaches_pre_audit:
        triggers.append("pre_audit_unresolved_rate_exceeds_frozen_threshold")
    if breaches_strict:
        triggers.append("strict_unresolved_rate_exceeds_frozen_threshold")
    if audit_incomplete:
        triggers.append("human_audit_status_is_not_complete")

    active = bool(triggers) or lock["target_inference_mode"] == MODE_NARROW_PATH
    return {
        "lock_id": lock["lock_id"],
        "narrow_path_active": active,
        "target_inference_mode": MODE_NARROW_PATH if active else MODE_FULL,
        "locked_mode": lock["target_inference_mode"],
        "human_audit_status": audit_status,
        "kill_threshold_pct": threshold,
        "unresolved_strict_pct": strict_pct,
        "unresolved_pre_audit_pct": pre_audit_pct,
        "triggers": triggers,
        "completed_pass_proxy_role": lock["completed_pass_proxy"]["role"],
    }


# ---------------------------------------------------------------------------
# Gate columns
# ---------------------------------------------------------------------------


def _decision(frame: pd.DataFrame) -> np.ndarray:
    if "decision" not in frame.columns:
        return np.full(len(frame), "", dtype=object)
    return frame["decision"].astype(object).to_numpy()


def _known_receiver(frame: pd.DataFrame) -> np.ndarray:
    if "true_receiverId" not in frame.columns:
        return np.zeros(len(frame), dtype=bool)
    known = frame["true_receiverId"].notna().to_numpy()
    if "truth_in_candidate_set" in frame.columns:
        known = known & frame["truth_in_candidate_set"].fillna(False).to_numpy(dtype=bool)
    return known


def _has_label(frame: pd.DataFrame) -> np.ndarray:
    if "inferred_targetId" not in frame.columns:
        return np.zeros(len(frame), dtype=bool)
    return frame["inferred_targetId"].notna().to_numpy()


def apply_label_gates(
    frame: pd.DataFrame,
    *,
    cohort: str,
    state: dict[str, Any],
) -> pd.DataFrame:
    """Attach the narrow-path gate columns to a pass-level Stage 2 table.

    ``cohort`` is ``"failed"`` or ``"completed"``. The completed cohort carries
    its receiver in the feed, so it is the only source of reliable
    receiver-specific completion labels; the failed cohort carries at best an
    unaudited geometric inference.
    """
    if cohort not in (COHORT_FAILED, COHORT_COMPLETED):
        raise ValueError(f"unknown cohort: {cohort}")
    out = frame.copy()
    if out.empty:
        for column in GATE_COLUMNS:
            out[column] = pd.Series(dtype=bool if column in BOOLEAN_GATE_COLUMNS else object)
        out["cohort"] = pd.Series(dtype=object)
        return out

    active = bool(state["narrow_path_active"])
    n = len(out)
    decision = _decision(out)
    known = _known_receiver(out)
    labelled = _has_label(out)

    if cohort == COHORT_COMPLETED:
        reliability = np.where(known, RELIABILITY_KNOWN_RECEIVER, RELIABILITY_NO_EVIDENCE)
        completion_label = known
        named_failed = np.zeros(n, dtype=bool)
        reason = np.where(
            known,
            "completed_pass_known_receiver",
            "completed_pass_receiver_missing_from_candidate_set",
        )
    else:
        reliability = np.select(
            [
                decision == DECISION_AUTO_ACCEPT,
                decision == DECISION_AMBIGUOUS,
                decision == DECISION_UNRESOLVED,
            ],
            [
                RELIABILITY_INFERRED_PROVISIONAL,
                RELIABILITY_AMBIGUOUS,
                RELIABILITY_NO_EVIDENCE,
            ],
            default=RELIABILITY_NO_EVIDENCE,
        )
        # A failed pass never carries a known receiver: the feed nulls it and
        # `toReceiverId` is the interceptor. Under the narrow path no failed
        # pass may become a named-receiver completion label, whatever the
        # geometry said.
        completion_label = np.zeros(n, dtype=bool) if active else labelled
        named_failed = (decision == DECISION_AUTO_ACCEPT) & labelled
        reason = np.select(
            [
                named_failed & active,
                decision == DECISION_AUTO_ACCEPT,
                decision == DECISION_AMBIGUOUS,
            ],
            [
                "narrow_path_auto_accepted_provisional_sensitivity_required",
                "auto_accepted_inferred_target",
                "narrow_path_ambiguous_unreviewed_touch_level_only",
            ],
            default="narrow_path_no_evidence_touch_level_only",
        )

    out["narrow_path_active"] = active
    out["target_inference_mode"] = state["target_inference_mode"]
    out["target_reliability_status"] = reliability
    out["human_audit_status"] = state["human_audit_status"]
    out["usable_as_receiver_completion_label"] = completion_label
    out["usable_as_named_failed_target"] = named_failed
    out["named_failed_target_requires_sensitivity_check"] = named_failed & active
    # Every pass attempt, resolved or not, is evidence at the touch level. That
    # is the whole point of the narrow path: nothing is thrown away, it is only
    # barred from being attributed to a named receiver.
    out["usable_for_touch_level_turnover"] = np.ones(n, dtype=bool)
    # Safe by default. Auto-accepted failures stay out of this flag while the
    # narrow path is active; a downstream stage that wants them must opt in via
    # `usable_as_named_failed_target` and publish the exclusion sensitivity.
    out["receiver_specific_eligible"] = completion_label if active else (completion_label | named_failed)
    out["label_gate_reason"] = reason
    out["cohort"] = cohort
    return out


def build_label_eligibility(
    failed: pd.DataFrame,
    completed: pd.DataFrame,
) -> pd.DataFrame:
    """One gated row per Stage 2 pass — the join target for every later stage."""
    parts = [frame for frame in (failed, completed) if not frame.empty]
    if not parts:
        return pd.DataFrame(columns=list(ELIGIBILITY_COLUMNS))
    stacked = pd.concat(parts, ignore_index=True)
    for column in ELIGIBILITY_COLUMNS:
        if column not in stacked.columns:
            stacked[column] = pd.NA
    return stacked[list(ELIGIBILITY_COLUMNS)].sort_values(["cohort", "gameId", "passId"]).reset_index(drop=True)


def receiver_completion_training_labels(eligibility: pd.DataFrame) -> pd.DataFrame:
    """The only Stage 3+ entry point for named-receiver completion labels.

    Filtering happens here rather than at each call site so the gate cannot be
    forgotten, and the result is re-asserted before it is handed back.
    """
    assert_gate_columns_present(eligibility, name="eligibility")
    if eligibility.empty:
        return eligibility.copy()
    selected = eligibility[eligibility["receiver_specific_eligible"].fillna(False).astype(bool)].copy()
    assert_no_unreliable_completion_labels(selected, name="receiver_completion_training_labels")
    return selected


# ---------------------------------------------------------------------------
# Assertions — fatal by design
# ---------------------------------------------------------------------------


def assert_gate_columns_present(frame: pd.DataFrame, *, name: str) -> None:
    missing = [c for c in GATE_COLUMNS if c not in frame.columns]
    if missing:
        raise NarrowPathViolation(f"{name}: missing narrow-path gate columns {missing}")


def assert_interceptor_never_target(
    frame: pd.DataFrame,
    *,
    name: str,
    target_col: str = "inferred_targetId",
) -> None:
    """The one rule that holds under every path: the interceptor is not the target."""
    if frame.empty or "interceptorId" not in frame.columns:
        return
    interceptor = pd.to_numeric(frame["interceptorId"], errors="coerce")
    for column in (target_col, "candidateId", "top_candidateId"):
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        clash = (values == interceptor) & values.notna() & interceptor.notna()
        if int(clash.sum()):
            raise NarrowPathViolation(
                f"{name}: {int(clash.sum())} row(s) name the interceptor in `{column}`"
            )


def assert_no_unreliable_completion_labels(frame: pd.DataFrame, *, name: str) -> None:
    """`usable_as_receiver_completion_label` implies a receiver the feed gave us."""
    if frame.empty:
        return
    assert_gate_columns_present(frame, name=name)
    flagged = frame["usable_as_receiver_completion_label"].fillna(False).astype(bool)
    if not flagged.any():
        return
    bad_status = flagged & (frame["target_reliability_status"] != RELIABILITY_KNOWN_RECEIVER)
    if int(bad_status.sum()):
        offenders = sorted(set(frame.loc[bad_status, "target_reliability_status"].astype(str)))
        raise NarrowPathViolation(
            f"{name}: {int(bad_status.sum())} completion label(s) with unreliable "
            f"target_reliability_status {offenders}"
        )
    if "true_receiverId" in frame.columns:
        missing = flagged & frame["true_receiverId"].isna()
        if int(missing.sum()):
            raise NarrowPathViolation(
                f"{name}: {int(missing.sum())} completion label(s) have no known receiver"
            )
    if "cohort" in frame.columns:
        from_failed = flagged & (frame["cohort"] == COHORT_FAILED)
        if int(from_failed.sum()):
            raise NarrowPathViolation(
                f"{name}: {int(from_failed.sum())} completion label(s) drawn from the failed cohort"
            )


def assert_named_failed_targets_are_accepted(frame: pd.DataFrame, *, name: str) -> None:
    if frame.empty or "usable_as_named_failed_target" not in frame.columns:
        return
    flagged = frame["usable_as_named_failed_target"].fillna(False).astype(bool)
    if not flagged.any():
        return
    if "decision" in frame.columns:
        not_accepted = flagged & (frame["decision"] != DECISION_AUTO_ACCEPT)
        if int(not_accepted.sum()):
            raise NarrowPathViolation(
                f"{name}: {int(not_accepted.sum())} named failed target(s) are not auto-accepted"
            )
    if "inferred_targetId" in frame.columns:
        unlabelled = flagged & frame["inferred_targetId"].isna()
        if int(unlabelled.sum()):
            raise NarrowPathViolation(
                f"{name}: {int(unlabelled.sum())} named failed target(s) carry no target id"
            )


def assert_reviewer_columns_blank(
    audit_sheet: pd.DataFrame,
    reviewer_columns: Sequence[str],
    *,
    human_audit_status: str,
) -> None:
    """No fabricated reviewer decisions while the audit is deferred/waived.

    A filled column here would be the single most damaging thing Stage 2 could
    produce: an invented human label that later stages treat as ground truth.
    """
    if human_audit_status == AUDIT_COMPLETE or audit_sheet.empty:
        return
    for column in reviewer_columns:
        if column not in audit_sheet.columns:
            continue
        values = audit_sheet[column].astype(str).str.strip().str.lower()
        filled = ~values.isin(REVIEWER_BLANK_SENTINELS)
        if int(filled.sum()):
            raise NarrowPathViolation(
                f"stage2_ambiguous_audit.csv: {int(filled.sum())} value(s) in `{column}` "
                f"while human_audit_status={human_audit_status}; reviewer fields must stay "
                "blank until a real blinded review is recorded"
            )


def assert_audit_sheet_on_disk_is_blank(
    path: Path,
    reviewer_columns: Sequence[str],
    *,
    human_audit_status: str,
) -> None:
    """Check the sheet already on disk before Stage 2 overwrites it.

    Checking only the freshly built sheet would miss the case that matters: a
    file someone edited between runs. Refusing is right in both directions — it
    surfaces a fabricated decision, and it stops a re-run from silently erasing
    a review that really was performed.
    """
    path = Path(path)
    if human_audit_status == AUDIT_COMPLETE or not path.exists():
        return
    assert_reviewer_columns_blank(
        pd.read_csv(path), reviewer_columns, human_audit_status=human_audit_status
    )


def assert_no_failed_precision_claim(text: str, *, name: str) -> None:
    """Guard the wording of generated reports against proxy abuse.

    The completed-pass proxy cannot validate failed-pass labels, so a generated
    artifact must never phrase it as if it could.
    """
    lowered = text.lower()
    forbidden = (
        "failed-pass precision of",
        "failed-pass auto-accept precision is",
        "audited failed-pass precision",
        "validated failed-pass targets",
    )
    hits = [phrase for phrase in forbidden if phrase in lowered]
    if hits:
        raise NarrowPathViolation(f"{name}: proxy-abuse phrasing {hits}")


def assert_narrow_path_gates(
    failed: pd.DataFrame,
    completed: pd.DataFrame,
    eligibility: pd.DataFrame,
    *,
    state: dict[str, Any],
) -> dict[str, Any]:
    """Every fatal gate in one call; raises :class:`NarrowPathViolation` on any breach."""
    for name, frame in (("failed_assignments", failed), ("completed_validation", completed)):
        assert_gate_columns_present(frame, name=name)
        assert_interceptor_never_target(frame, name=name)
        assert_no_unreliable_completion_labels(frame, name=name)
        assert_named_failed_targets_are_accepted(frame, name=name)

    assert_gate_columns_present(eligibility, name="label_eligibility")
    assert_interceptor_never_target(eligibility, name="label_eligibility")
    assert_no_unreliable_completion_labels(eligibility, name="label_eligibility")
    assert_named_failed_targets_are_accepted(eligibility, name="label_eligibility")

    if state["narrow_path_active"] and not failed.empty:
        leaked = failed["usable_as_receiver_completion_label"].fillna(False).astype(bool)
        if int(leaked.sum()):
            raise NarrowPathViolation(
                f"failed_assignments: {int(leaked.sum())} failed pass(es) flagged as "
                "receiver completion labels while the narrow path is active"
            )
    return gate_summary(eligibility, state=state)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def gate_summary(eligibility: pd.DataFrame, *, state: dict[str, Any]) -> dict[str, Any]:
    """Counts behind each gate, for the report and the verification checklist."""
    if eligibility.empty:
        return {**state, "n_passes": 0}

    def _count(mask: Iterable[bool]) -> int:
        return int(pd.Series(list(mask)).fillna(False).astype(bool).sum())

    failed = eligibility["cohort"] == COHORT_FAILED
    return {
        **state,
        "n_passes": int(len(eligibility)),
        "n_failed": int(failed.sum()),
        "n_completed": int((~failed).sum()),
        "n_receiver_completion_labels": _count(eligibility["usable_as_receiver_completion_label"]),
        "n_named_failed_targets_optional": _count(eligibility["usable_as_named_failed_target"]),
        "n_receiver_specific_eligible": _count(eligibility["receiver_specific_eligible"]),
        "n_touch_level_only": int(
            (failed & ~eligibility["receiver_specific_eligible"].fillna(False).astype(bool)).sum()
        ),
        "reliability_mix": {
            str(k): int(v) for k, v in eligibility["target_reliability_status"].value_counts().items()
        },
    }
