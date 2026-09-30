"""Tests for Stage 1 canonical data layer."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from passing_windows.aliases import canonicalize_id, load_alias_map
from passing_windows.audit import (
    audit_candidate_sets,
    audit_causal_velocity,
    audit_causal_velocity_boundaries,
    audit_matchup_interval_join,
    audit_mirroring_symmetry,
    audit_pass_outcome_classes,
    audit_synthetic_lane_geometry,
)
from passing_windows.coords import broadcast_to_event_xy, mirror_xy, normalize_attack_xy, symmetry_feature_check
from passing_windows.geometry import euclidean_distance, lane_clearance, point_to_segment_distance
from passing_windows.io_utils import build_output_manifest, verify_output_manifest, write_json
from passing_windows.joins import (
    ballhandler_in_offense,
    build_candidate_states,
    build_touch_candidates,
    candidate_lineup_from_chance,
    clock_agreement,
    interval_join_matchups,
    join_pass_to_touch,
)
from passing_windows.normalize import normalize_passes
from passing_windows.quality import EXCLUSION_ORDER, primary_exclusion_reason, sample_model_frames
from passing_windows.velocities import (
    MAX_CAUSAL_FRAME_GAP,
    MAX_PLAUSIBLE_PLAYER_SPEED_FPS,
    add_causal_velocities,
    assert_no_future_leakage,
    causal_velocity_from_positions,
    flag_implausible_speed,
    orientation_segment_ids,
)


def test_alias_canonicalize():
    m = load_alias_map()
    assert canonicalize_id(59237, m) == 59129  # Dedovic -> Djedovic
    assert canonicalize_id(59129, m) == 59129
    assert canonicalize_id(999999, m) == 999999


def test_normalize_attack_puts_hoop_at_positive_x():
    x_norm, y_norm = normalize_attack_xy(-40.0, 5.0, attack_sign=-1.0)
    assert x_norm == pytest.approx(40.0)
    assert y_norm == pytest.approx(5.0)


def test_broadcast_to_event_left_hoop_identity():
    assert broadcast_to_event_xy(10.0, -4.0, left_hoop=True) == (10.0, -4.0)
    assert broadcast_to_event_xy(10.0, -4.0, left_hoop=False) == (-10.0, 4.0)


def test_mirror_flips_y_only():
    assert mirror_xy(10.0, 4.0) == (10.0, -4.0)
    assert symmetry_feature_check(4.0, -4.0, expect_flip=True)
    assert symmetry_feature_check(12.0, 12.0, expect_flip=False)


def test_causal_velocity_no_future():
    x = np.array([0.0, 1.0, 2.0, 3.0, 50.0])
    y = np.zeros(5)
    frames = np.arange(5)
    vx, _, _ = causal_velocity_from_positions(x, y, frames)
    x2 = x.copy()
    x2[-1] = -100.0
    vx2, _, _ = causal_velocity_from_positions(x2, y, frames)
    assert vx[3] == pytest.approx(vx2[3])
    assert_no_future_leakage(np.array([1, 2, 3]), prediction_time=3)
    with pytest.raises(AssertionError):
        assert_no_future_leakage(np.array([1, 2, 3, 4]), prediction_time=3)


def test_audit_helpers_pass():
    assert audit_causal_velocity()["ok"]
    assert audit_mirroring_symmetry()["ok"]
    assert audit_synthetic_lane_geometry()["ok"]


def test_geometry_lane():
    assert point_to_segment_distance(5, 0, 0, 0, 10, 0) == pytest.approx(0.0)
    assert point_to_segment_distance(5, 3, 0, 0, 10, 0) == pytest.approx(3.0)
    assert lane_clearance((0, 0), (10, 0), (5, 0), 1.0) < 0
    assert euclidean_distance(0, 0, 3, 4) == pytest.approx(5.0)


def test_clock_agreement():
    r = clock_agreement(100, 100, 500.0, 500.0, 4000, 4000)
    assert r["ok"]
    r2 = clock_agreement(100, 103, 500.0, 500.0)
    assert not r2["frame_ok"]


def test_candidate_lineup():
    cands = candidate_lineup_from_chance([1, 2, 3, 4, 5], 3)
    assert cands == [1, 2, 4, 5]


def test_sample_model_frames_5hz():
    assert sample_model_frames(0, 20, stride=5) == [0, 5, 10, 15, 20]


def test_join_pass_to_touch():
    touches = pd.DataFrame(
        [
            {
                "id": "t1",
                "gameId": 1,
                "startFrame": 100,
                "endFrame": 120,
                "playerId": 7,
                "chanceId": "c1",
                "possessionId": "p1",
            }
        ]
    )
    passes = pd.DataFrame(
        [
            {
                "id": "pass1",
                "gameId": 1,
                "touchId": "t1",
                "startFrame": 119,
                "endFrame": 125,
                "passerId": 7,
                "receiverId": 8,
                "complete": True,
                "chanceId": "c1",
                "possessionId": "p1",
            }
        ]
    )
    joined = join_pass_to_touch(passes, touches)
    assert bool(joined.iloc[0]["pass_within_touch"])
    assert bool(joined.iloc[0]["release_near_touch_end"])
    assert bool(joined.iloc[0]["passer_matches_touch_player"])


def _eligible_touch_row(**overrides) -> pd.Series:
    row = {
        "chanceId": "c",
        "usable_flag": True,
        "chance_frame_order_ok": True,
        "frame_order_ok": True,
        "transition": False,
        "zero_elapsed_clock": False,
        "frontcourtFrame": 10,
        "has_frontcourt_frame": True,
        "lineup_5v5": True,
        "substitution_boundary": False,
        "has_ballhandler": True,
        "playerId": 1,
        "offPlayerIds": [1, 2, 3, 4, 5],
        "is_self_pass_touch": False,
        "is_multipass_touch": False,
        "is_inbounds_touch": False,
    }
    row.update(overrides)
    return pd.Series(row)


def _touches_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Blocker 1: exactly four teammate candidates per primary touch
# --------------------------------------------------------------------------


def test_ballhandler_in_offense_detects_defender_credited_touch():
    assert ballhandler_in_offense([1, 2, 3, 4, 5], 3)
    # Ball-handler recorded on the defensive side of the chance.
    assert not ballhandler_in_offense([1, 2, 3, 4, 5], 9)
    assert not ballhandler_in_offense(None, 3)
    assert not ballhandler_in_offense([1, 2, 3, 4, 5], None)


def test_eligible_touch_row_is_eligible():
    assert primary_exclusion_reason(_eligible_touch_row()) is None


def test_exclusion_ballhandler_not_in_offense():
    """The real defect: ball-handler outside chance.offPlayerIds must be excluded."""
    row = _eligible_touch_row(playerId=9, offPlayerIds=[1, 2, 3, 4, 5])
    assert primary_exclusion_reason(row) == "ballhandler_not_in_offense"


def test_ballhandler_rule_is_in_consort_order():
    assert "ballhandler_not_in_offense" in EXCLUSION_ORDER
    # Must be evaluated after a ball-handler is known to exist.
    assert EXCLUSION_ORDER.index("missing_ballhandler") < EXCLUSION_ORDER.index(
        "ballhandler_not_in_offense"
    )


def test_build_touch_candidates_exactly_four_per_touch():
    touches = _touches_frame(
        [
            {
                "gameId": 1,
                "id": "t1",
                "chanceId": "c1",
                "playerId": 3,
                "offPlayerIds": [1, 2, 3, 4, 5],
            }
        ]
    )
    cands = build_touch_candidates(touches)
    assert len(cands) == 4
    assert sorted(cands["candidateId"]) == [1, 2, 4, 5]
    assert set(cands["n_candidates"]) == {4}
    assert 3 not in set(cands["candidateId"])


def test_build_touch_candidates_rejects_five_candidate_touch():
    """Regression: a ball-handler outside offPlayerIds produced five candidates."""
    touches = _touches_frame(
        [
            {
                "gameId": 114243,
                "id": "touch-114243-2-194",
                "chanceId": "c1",
                "playerId": 59181,
                "offPlayerIds": [35028, 59220, 59221, 59217, 59301],
            }
        ]
    )
    with pytest.raises(AssertionError, match="exactly"):
        build_touch_candidates(touches)
    # Non-strict mode drops the touch instead of emitting five candidates.
    lenient = build_touch_candidates(touches, strict=False)
    assert lenient.empty


def test_build_touch_candidates_drops_duplicate_lineup_ids():
    touches = _touches_frame(
        [
            {
                "gameId": 1,
                "id": "t1",
                "chanceId": "c1",
                "playerId": 3,
                "offPlayerIds": [1, 2, 3, 4, 5, 5],
            }
        ]
    )
    cands = build_touch_candidates(touches)
    assert len(cands) == 4
    assert cands["candidateId"].nunique() == 4


def test_audit_candidate_sets_detects_five_candidates():
    touches = pd.DataFrame(
        [
            {
                "gameId": 1,
                "id": "t1",
                "chanceId": "c1",
                "playerId": 9,
                "offPlayerIds": [1, 2, 3, 4, 5],
                "primary_touch_eligible": True,
            }
        ]
    )
    bad = pd.DataFrame(
        [
            {
                "gameId": 1,
                "touchId": "t1",
                "chanceId": "c1",
                "ballhandlerId": 9,
                "candidateId": cid,
                "candidate_rank": i,
                "n_candidates": 5,
            }
            for i, cid in enumerate([1, 2, 3, 4, 5])
        ]
    )
    res = audit_candidate_sets(bad, touches)
    assert not res["ok"]
    assert res["max_candidates_per_touch"] == 5
    assert not res["checks"]["all_touches_exactly_4"]
    assert not res["checks"]["no_primary_ballhandler_outside_offense"]


def test_audit_candidate_sets_passes_on_clean_input():
    touches = pd.DataFrame(
        [
            {
                "gameId": 1,
                "id": "t1",
                "chanceId": "c1",
                "playerId": 3,
                "offPlayerIds": [1, 2, 3, 4, 5],
                "primary_touch_eligible": True,
            }
        ]
    )
    res = audit_candidate_sets(build_touch_candidates(touches), touches)
    assert res["ok"], res
    assert res["max_candidates_per_touch"] == 4


# --------------------------------------------------------------------------
# Blocker 3: velocity discontinuities at coordinate boundaries
# --------------------------------------------------------------------------


def _orientation_flip_track() -> pd.DataFrame:
    """Player walking steadily in broadcast space while the hoop side flips."""
    return pd.DataFrame(
        {
            "gameId": [1] * 5,
            "playerId": [7] * 5,
            "frameIdx": [0, 5, 10, 15, 20],
            "x": [40.0, 40.5, 41.0, 41.5, 42.0],
            "y": [0.0, 0.0, 0.0, 0.0, 0.0],
            "leftHoop": [True, True, False, False, False],
        }
    )


def test_orientation_segment_ids_split_on_change():
    seg = orientation_segment_ids([True, True, False, False, None, None, True])
    assert list(seg) == [1, 1, 2, 2, 3, 3, 4]


def test_velocity_resets_at_orientation_boundary():
    """Regression: differencing event coords across a hoop flip gave ~400 ft/s."""
    track = _orientation_flip_track()

    # What the old implementation did: difference event-aligned coordinates.
    x_event = [broadcast_to_event_xy(x, y, lh)[0] for x, y, lh in zip(track["x"], track["y"], track["leftHoop"])]
    _, _, naive_speed = causal_velocity_from_positions(
        np.array(x_event), track["y"].to_numpy(), track["frameIdx"].to_numpy()
    )
    assert naive_speed[2] > MAX_PLAUSIBLE_PLAYER_SPEED_FPS

    out = add_causal_velocities(track)
    # Boundary row carries no velocity at all.
    assert np.isnan(out.loc[2, "speed_causal"])
    assert not bool(out.loc[2, "velocity_ok"])
    # Every other estimate is the true 2.5 ft/s walking speed.
    assert out.loc[1, "speed_causal"] == pytest.approx(2.5)
    assert out.loc[3, "speed_causal"] == pytest.approx(2.5)
    assert out["speed_causal"].max() < MAX_PLAUSIBLE_PLAYER_SPEED_FPS


def test_velocity_rotates_into_event_frame():
    out = add_causal_velocities(_orientation_flip_track())
    # leftHoop True: event frame equals broadcast frame.
    assert out.loc[1, "vx"] == pytest.approx(2.5)
    # leftHoop False: broadcast is rotated 180 degrees, so vx flips sign while
    # the speed magnitude is unchanged.
    assert out.loc[3, "vx"] == pytest.approx(-2.5)
    assert out.loc[3, "speed_causal"] == pytest.approx(2.5)


def test_velocity_resets_across_wide_frame_gap():
    track = pd.DataFrame(
        {
            "gameId": [1, 1, 1],
            "playerId": [7, 7, 7],
            "frameIdx": [0, 5, 5 + MAX_CAUSAL_FRAME_GAP + 1],
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 0.0, 0.0],
            "leftHoop": [True, True, True],
        }
    )
    out = add_causal_velocities(track)
    assert out.loc[1, "speed_causal"] == pytest.approx(5.0)
    assert np.isnan(out.loc[2, "speed_causal"])
    assert out.loc[2, "velocity_frame_gap"] == MAX_CAUSAL_FRAME_GAP + 1


def test_implausible_speed_is_flagged_and_nulled():
    track = pd.DataFrame(
        {
            "gameId": [1, 1],
            "playerId": [7, 7],
            "frameIdx": [0, 1],
            "x": [0.0, 100.0],  # 100 ft in one 25 Hz frame
            "y": [0.0, 0.0],
            "leftHoop": [True, True],
        }
    )
    out = add_causal_velocities(track)
    assert bool(out.loc[1, "speed_implausible"])
    assert np.isnan(out.loc[1, "speed_causal"])
    assert np.isnan(out.loc[1, "vx"])
    assert flag_implausible_speed(np.array([1.0, 1e3, np.nan])).tolist() == [False, True, False]


def test_causal_velocity_segment_argument_blocks_differencing():
    x = np.array([-40.0, -39.0, 39.0, 40.0])
    frames = np.array([0, 1, 2, 3])
    _, _, seg_speed = causal_velocity_from_positions(
        x, np.zeros(4), frames, segment_id=np.array([0, 0, 1, 1])
    )
    assert np.isnan(seg_speed[2])
    _, _, raw_speed = causal_velocity_from_positions(x, np.zeros(4), frames)
    assert raw_speed[2] > MAX_PLAUSIBLE_PLAYER_SPEED_FPS


def test_audit_causal_velocity_boundaries_catches_corrupt_track():
    corrupt = pd.DataFrame(
        {
            "gameId": [1, 1, 1],
            "playerId": [7, 7, 7],
            "frameIdx": [0, 5, 10],
            "leftHoop": [True, True, False],
            "speed_causal": [np.nan, 2.5, 400.0],
            "speed_implausible": [False, False, False],
            "velocity_frame_gap": [np.nan, 5.0, 5.0],
        }
    )
    res = audit_causal_velocity_boundaries(corrupt)
    assert not res["ok"]
    assert res["n_speed_over_cap"] == 1
    assert not res["checks"]["no_velocity_at_orientation_boundary"]

    clean = add_causal_velocities(_orientation_flip_track())
    assert audit_causal_velocity_boundaries(clean)["ok"]


# --------------------------------------------------------------------------
# Blocker 2: matchup interval join
# --------------------------------------------------------------------------


def _matchups_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_matchup_interval_join_boundaries_are_inclusive():
    matchups = _matchups_frame(
        [
            {
                "id": "m1",
                "gameId": 1,
                "chanceId": "c1",
                "offPlayerId": 2,
                "defPlayerId": 20,
                "startFrame": 100,
                "endFrame": 120,
            }
        ]
    )
    keys = pd.DataFrame(
        {
            "gameId": [1, 1, 1, 1],
            "chanceId": ["c1"] * 4,
            "candidateId": [2, 2, 2, 2],
            "frameIdx": [99, 100, 120, 121],
        }
    )
    out = interval_join_matchups(keys, matchups, off_player_col="candidateId")
    assert out["matchup_matched"].tolist() == [False, True, True, False]
    assert out.loc[1, "assigned_defenderId"] == 20
    assert out.loc[2, "assigned_defenderId"] == 20
    assert not out["matchup_ambiguous"].any()


def test_matchup_interval_join_resolves_overlaps_deterministically():
    """Overlapping assignments must resolve to the most recent, shortest interval."""
    matchups = _matchups_frame(
        [
            {"id": "m1", "gameId": 1, "chanceId": "c1", "offPlayerId": 2, "defPlayerId": 20, "startFrame": 100, "endFrame": 140},
            {"id": "m2", "gameId": 1, "chanceId": "c1", "offPlayerId": 2, "defPlayerId": 21, "startFrame": 105, "endFrame": 130},
            {"id": "m3", "gameId": 1, "chanceId": "c1", "offPlayerId": 2, "defPlayerId": 22, "startFrame": 105, "endFrame": 115},
        ]
    )
    keys = pd.DataFrame(
        {"gameId": [1], "chanceId": ["c1"], "candidateId": [2], "frameIdx": [110]}
    )
    out = interval_join_matchups(keys, matchups, off_player_col="candidateId")
    assert out.loc[0, "n_matchup_intervals"] == 3
    assert bool(out.loc[0, "matchup_ambiguous"])
    # latest startFrame (105) then shortest interval (m3) wins.
    assert out.loc[0, "matchupId"] == "m3"
    assert out.loc[0, "assigned_defenderId"] == 22

    # Determinism: row order of the matchup feed must not change the answer.
    shuffled = matchups.iloc[::-1].reset_index(drop=True)
    out2 = interval_join_matchups(keys, shuffled, off_player_col="candidateId")
    assert out2.loc[0, "matchupId"] == "m3"


def test_matchup_interval_join_preserves_keys_and_row_count():
    matchups = _matchups_frame(
        [
            {"id": "m1", "gameId": 1, "chanceId": "c1", "offPlayerId": 2, "defPlayerId": 20, "startFrame": 100, "endFrame": 120},
            {"id": "m2", "gameId": 1, "chanceId": "c1", "offPlayerId": 2, "defPlayerId": 21, "startFrame": 100, "endFrame": 120},
        ]
    )
    keys = pd.DataFrame(
        {
            "gameId": [1, 1],
            "chanceId": ["c1", "c1"],
            "touchId": ["t1", "t1"],
            "candidateId": [2, 3],
            "frameIdx": [110, 110],
            "candidate_rank": [0, 1],
        }
    )
    out = interval_join_matchups(keys, matchups, off_player_col="candidateId")
    assert len(out) == len(keys)
    for col in keys.columns:
        assert out[col].tolist() == keys[col].tolist()
    # Candidate 3 has no matchup interval at all.
    assert out["matchup_matched"].tolist() == [True, False]
    assert out.loc[1, "n_matchup_intervals"] == 0
    # Tie on start and length resolves by smallest matchupId.
    assert out.loc[0, "matchupId"] == "m1"


def test_matchup_join_handles_empty_matchups():
    keys = pd.DataFrame(
        {"gameId": [1], "chanceId": ["c1"], "candidateId": [2], "frameIdx": [110]}
    )
    out = interval_join_matchups(keys, pd.DataFrame(), off_player_col="candidateId")
    assert len(out) == 1
    assert not bool(out.loc[0, "matchup_matched"])


def test_build_candidate_states_materializes_join():
    model_frames = pd.DataFrame(
        {
            "gameId": [1, 1],
            "touchId": ["t1", "t1"],
            "chanceId": ["c1", "c1"],
            "frameIdx": [100, 105],
            "ballhandlerId": [3, 3],
            "primary_frame_eligible": [True, True],
        }
    )
    candidates = build_touch_candidates(
        _touches_frame(
            [
                {
                    "gameId": 1,
                    "id": "t1",
                    "chanceId": "c1",
                    "playerId": 3,
                    "offPlayerIds": [1, 2, 3, 4, 5],
                }
            ]
        )
    )
    matchups = _matchups_frame(
        [
            {"id": f"m{p}", "gameId": 1, "chanceId": "c1", "offPlayerId": p, "defPlayerId": 20 + p, "startFrame": 90, "endFrame": 110}
            for p in (1, 2, 4, 5)
        ]
    )
    states = build_candidate_states(model_frames, candidates, matchups)
    assert len(states) == 8  # 2 frames x 4 candidates
    assert states["matchup_matched"].all()
    assert not states.duplicated(subset=["gameId", "touchId", "frameIdx", "candidateId"]).any()
    res = audit_matchup_interval_join(states, matchups)
    assert res["ok"], res
    assert res["coverage_all_states"] == pytest.approx(1.0)


def test_audit_matchup_join_detects_interval_violation():
    matchups = _matchups_frame(
        [
            {"id": "m1", "gameId": 1, "chanceId": "c1", "offPlayerId": 2, "defPlayerId": 20, "startFrame": 100, "endFrame": 120}
        ]
    )
    states = pd.DataFrame(
        {
            "gameId": [1],
            "touchId": ["t1"],
            "chanceId": ["c1"],
            "frameIdx": [500],
            "candidateId": [2],
            "assigned_defenderId": [20],
            "matchupId": ["m1"],
            "matchup_startFrame": [100],
            "matchup_endFrame": [120],
            "n_matchup_intervals": [1],
            "matchup_matched": [True],
            "matchup_ambiguous": [False],
        }
    )
    res = audit_matchup_interval_join(states, matchups)
    assert not res["ok"]
    assert not res["checks"]["chosen_interval_contains_frame"]


# --------------------------------------------------------------------------
# Major 1: output manifest validity
# --------------------------------------------------------------------------


def test_manifest_excludes_itself_and_dedupes(tmp_path):
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.txt").write_text("beta", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")

    manifest = build_output_manifest(
        [tmp_path / "a.txt", tmp_path / "a.txt", manifest_path, tmp_path / "b.txt"],
        root=tmp_path,
        manifest_path=manifest_path,
    )
    paths = [f["path"] for f in manifest["files"]]
    assert paths == ["a.txt", "b.txt"]
    assert len(paths) == len(set(paths))
    assert "manifest.json" not in paths

    write_json(manifest, manifest_path)
    assert verify_output_manifest(manifest_path, tmp_path)["ok"]


def test_manifest_verification_detects_tampering(tmp_path):
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest = build_output_manifest(
        [tmp_path / "a.txt"], root=tmp_path, manifest_path=manifest_path
    )
    write_json(manifest, manifest_path)
    assert verify_output_manifest(manifest_path, tmp_path)["ok"]

    (tmp_path / "a.txt").write_text("tampered", encoding="utf-8")
    result = verify_output_manifest(manifest_path, tmp_path)
    assert not result["ok"]
    assert result["mismatched"] == ["a.txt"]

    (tmp_path / "a.txt").unlink()
    assert verify_output_manifest(manifest_path, tmp_path)["missing"] == ["a.txt"]


def test_manifest_verification_flags_duplicate_entries(tmp_path):
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    manifest = build_output_manifest([tmp_path / "a.txt"], root=tmp_path)
    manifest["files"] = manifest["files"] * 2
    result = verify_output_manifest(manifest, tmp_path)
    assert not result["ok"]
    assert result["n_duplicate_paths"] == 1


def test_manifest_sign_off_files_are_separate(tmp_path):
    (tmp_path / "gen.md").write_text("generated", encoding="utf-8")
    (tmp_path / "SIGNOFF.md").write_text("pending", encoding="utf-8")
    manifest = build_output_manifest(
        [tmp_path / "gen.md"], root=tmp_path, sign_off_paths=[tmp_path / "SIGNOFF.md"]
    )
    assert [f["path"] for f in manifest["files"]] == ["gen.md"]
    assert [f["path"] for f in manifest["sign_off_files"]] == ["SIGNOFF.md"]
    # Human sign-off edits must not break strict verification.
    (tmp_path / "SIGNOFF.md").write_text("APPROVE", encoding="utf-8")
    assert verify_output_manifest(manifest, tmp_path)["ok"]


# --------------------------------------------------------------------------
# Stage 2 scout findings: failed-pass evidence and outcome classes
# --------------------------------------------------------------------------


def _raw_passes_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": "p_complete",
                "gameId": 1,
                "startFrame": 100,
                "endFrame": 110,
                "passerId": 1,
                "receiverId": 2,
                "toReceiverId": None,
                "complete": True,
                "turnover": False,
                "passerLoc": [0.0, 0.0],
                "receiverLoc": [3.0, 4.0],
                "distance": 5.0,
            },
            {
                "id": "p_turnover",
                "gameId": 1,
                "startFrame": 200,
                "endFrame": 215,
                "passerId": 1,
                "receiverId": None,
                "toReceiverId": 20,
                "complete": False,
                "turnover": True,
                "passerLoc": [0.0, 0.0],
                "receiverLoc": None,
                "distance": None,
            },
            {
                "id": "p_unknown",
                "gameId": 1,
                "startFrame": 300,
                "endFrame": None,
                "passerId": 1,
                "receiverId": None,
                "toReceiverId": None,
                "complete": None,
                "turnover": False,
                "passerLoc": [0.0, 0.0],
                "receiverLoc": None,
                "distance": None,
            },
        ]
    )


def test_pass_outcome_class_separates_tri_state_complete():
    """`complete` is nullable; astype(bool) merged nulls into the failure cohort."""
    out = normalize_passes(_raw_passes_frame())
    assert out["pass_outcome_class"].tolist() == [
        "complete",
        "incomplete_turnover",
        "unknown_outcome",
    ]
    assert out["has_endFrame"].tolist() == [True, True, False]
    assert out["is_failed_pass"].tolist() == [False, True, False]
    # Only the turnover pass can enter Stage 2 target inference.
    assert out["failed_pass_inference_eligible"].tolist() == [False, True, False]


def test_failed_passes_have_no_receiver_evidence():
    out = normalize_passes(_raw_passes_frame())
    failed = out[out["pass_outcome_class"] != "complete"]
    assert failed["receiverLoc"].isna().all()
    assert failed["distance"].isna().all()
    assert failed["receiverId"].isna().all()
    assert out.loc[0, "has_receiver_evidence"]


def test_audit_pass_outcome_classes_reports_evidence_gaps():
    out = normalize_passes(_raw_passes_frame())
    res = audit_pass_outcome_classes(out)
    assert res["ok"], res
    assert res["by_class"]["unknown_outcome"]["n_with_endFrame"] == 0
    assert res["by_class"]["incomplete_turnover"]["n_with_receiverLoc"] == 0
    assert res["by_class"]["complete"]["n_with_receiverLoc"] == 1


def test_exclusion_malformed():
    row = pd.Series(
        {
            "chanceId": "c",
            "usable_flag": True,
            "chance_frame_order_ok": False,
            "frame_order_ok": True,
            "transition": False,
            "zero_elapsed_clock": False,
            "frontcourtFrame": 10,
            "has_frontcourt_frame": True,
            "lineup_5v5": True,
            "substitution_boundary": False,
            "has_ballhandler": True,
            "playerId": 1,
            "is_self_pass_touch": False,
            "is_multipass_touch": False,
            "is_inbounds_touch": False,
        }
    )
    assert primary_exclusion_reason(row) == "malformed_chance_frame_order"
