"""Stage 4 cross-fitted component models (passability, value, choice)."""

from __future__ import annotations

MODEL_VERSION = "stage4_sklearn_logo_v3"
SCHEMA_VERSION = 1

ABLATION_NAMES: tuple[str, ...] = (
    "distance_only",
    "nearest_defender_distance",
    "static_lane_geometry_no_velocity",
    "kinematic_geometry_no_uncertainty",
    "full_model",
)

PRIMARY_ABLATION = "full_model"

__all__ = [
    "MODEL_VERSION",
    "SCHEMA_VERSION",
    "ABLATION_NAMES",
    "PRIMARY_ABLATION",
]
