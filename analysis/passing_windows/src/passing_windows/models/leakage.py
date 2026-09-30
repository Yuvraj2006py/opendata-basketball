"""Leakage and narrow-path guards for Stage 4 fitting."""

from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd

from passing_windows.narrow_path import (
    NarrowPathViolation,
    assert_interceptor_never_target,
    assert_no_failed_precision_claim,
)

from .features import assert_no_forbidden_features
from .outcomes import assert_no_pts_scored_source

FORBIDDEN_SPLIT_TOKENS = (
    "random_frame",
    "random_touch",
    "random_chance",
    "shuffle_split",
    "train_test_split",
)


class Stage4LeakageError(AssertionError):
    """Causal / label / split leakage detected."""


def assert_logo_only(split_description: str) -> None:
    text = split_description.lower()
    for tok in FORBIDDEN_SPLIT_TOKENS:
        if tok in text:
            raise Stage4LeakageError(
                f"forbidden split {tok!r} in {split_description!r}; nested LOGO only"
            )
    if "leave-one-game" not in text and "logo" not in text and "leave_one_game" not in text:
        raise Stage4LeakageError(
            f"split must be nested leave-one-game-out; got {split_description!r}"
        )


def assert_held_out_absent_from_train(
    train_game_ids: Sequence[int],
    held_out_game_id: int,
) -> None:
    if int(held_out_game_id) in {int(g) for g in train_game_ids}:
        raise Stage4LeakageError(
            f"held-out game {held_out_game_id} appears in train_game_ids={list(train_game_ids)}"
        )


def assert_train_rows_exclude_held_out(train_df: pd.DataFrame, held_out_game_id: int) -> None:
    if train_df.empty:
        return
    if int(held_out_game_id) in set(train_df["gameId"].astype(int).unique()):
        raise Stage4LeakageError(
            f"training frame contains held-out gameId={held_out_game_id}"
        )


def assert_completion_labels_narrow_path(labeled: pd.DataFrame) -> None:
    """Completion training rows must be narrow-path eligible known receivers."""
    if labeled.empty:
        return
    if "completion_label_eligible" in labeled.columns:
        bad = labeled[~labeled["completion_label_eligible"].fillna(False).astype(bool)]
        if len(bad):
            raise NarrowPathViolation(
                f"passability training includes {len(bad)} non-eligible rows"
            )
    if "receiver_specific_eligible" in labeled.columns:
        bad = labeled[~labeled["receiver_specific_eligible"].fillna(False).astype(bool)]
        if len(bad):
            raise NarrowPathViolation(
                f"passability training includes {len(bad)} non-receiver_specific_eligible rows"
            )
    if "target_reliability_status" in labeled.columns:
        bad = labeled[labeled["target_reliability_status"].astype(str) != "known_receiver"]
        if len(bad):
            raise NarrowPathViolation(
                f"completion labels with non-known_receiver status: {len(bad)}"
            )
    assert_interceptor_never_target(
        labeled, name="stage4_completion_labels", target_col="true_receiverId"
    )


def assert_no_season_aggregate_covariates(columns: Iterable[str]) -> None:
    for c in columns:
        cl = str(c).lower()
        if "season_avg" in cl or "season_aggregate" in cl or cl.endswith("_season"):
            raise Stage4LeakageError(f"season aggregate covariate forbidden: {c}")
    assert_no_forbidden_features(list(columns), name="stage4_features")


def assert_causal_feature_frame(df: pd.DataFrame, feature_cols: Sequence[str]) -> None:
    assert_no_season_aggregate_covariates(feature_cols)
    assert_no_pts_scored_source(df.columns)
    for c in feature_cols:
        if str(c).startswith("future_") or str(c).endswith("_lead_future"):
            raise Stage4LeakageError(f"future-looking feature forbidden: {c}")


def assert_trajectory_choice_honored(df: pd.DataFrame) -> None:
    """Primary geometry must match the explicit trajectory_choice, not score-max."""
    if df.empty or "trajectory_choice" not in df.columns:
        return
    if "trajectory_choice_rule" in df.columns:
        bad = df["trajectory_choice_rule"].dropna().astype(str).unique().tolist()
        unexpected = [r for r in bad if r != "lead_if_velocity_ok_else_direct"]
        if unexpected:
            raise Stage4LeakageError(f"unexpected trajectory_choice_rule values: {unexpected}")


def scan_reports_for_forbidden_claims(texts: Sequence[str]) -> None:
    for i, text in enumerate(texts):
        assert_no_failed_precision_claim(text, name=f"stage4_report[{i}]")
