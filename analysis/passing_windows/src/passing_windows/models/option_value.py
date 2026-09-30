"""Option value Q and net option value NOV."""

from __future__ import annotations

import numpy as np
import pandas as pd

V_FAIL_PRIMARY = 0.0


def compute_option_value(
    q: np.ndarray | pd.Series,
    v_catch: np.ndarray | pd.Series,
    *,
    v_fail: float = V_FAIL_PRIMARY,
) -> np.ndarray:
    """Q(j,t) = q * V_catch + (1 - q) * V_fail."""
    q_arr = np.asarray(q, dtype=float)
    v_arr = np.asarray(v_catch, dtype=float)
    return q_arr * v_arr + (1.0 - q_arr) * float(v_fail)


def compute_nov(
    q_option: np.ndarray | pd.Series,
    v_keep: np.ndarray | pd.Series,
) -> np.ndarray:
    """NOV(j,t) = Q(j,t) - V_keep(t)."""
    return np.asarray(q_option, dtype=float) - np.asarray(v_keep, dtype=float)


def attach_option_columns(
    df: pd.DataFrame,
    *,
    q_col: str = "q_passability_model",
    v_catch_col: str = "V_catch_model",
    v_keep_col: str = "V_keep_model",
    reject_col: str = "rejected_outside_support",
    mask_rejected_for_counterfactual: bool = True,
    v_fail: float = V_FAIL_PRIMARY,
) -> pd.DataFrame:
    """Write Q_option_model, NOV_model, V_fail; mask rejected rows for counterfactuals."""
    out = df.copy()
    out["V_fail"] = float(v_fail)
    q = out[q_col].to_numpy(dtype=float)
    vc = out[v_catch_col].to_numpy(dtype=float)
    vk = out[v_keep_col].to_numpy(dtype=float)
    q_opt = compute_option_value(q, vc, v_fail=v_fail)
    nov = compute_nov(q_opt, vk)
    if mask_rejected_for_counterfactual and reject_col in out.columns:
        rejected = out[reject_col].fillna(False).astype(bool).to_numpy()
        q_opt = q_opt.astype(float)
        nov = nov.astype(float)
        q_opt[rejected] = np.nan
        nov[rejected] = np.nan
    out["Q_option_model"] = q_opt
    out["NOV_model"] = nov
    return out
