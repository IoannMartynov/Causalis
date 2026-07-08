from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from causalis.data_contracts.causal_diagnostic_data import UnconfoundednessDiagnosticData
from causalis.data_contracts.causal_estimate import CausalEstimate
from causalis.dgp.causaldata import CausalData


def make_overlap_data_and_estimate(
    m_hat: np.ndarray,
    d: np.ndarray,
    *,
    normalize_ipw: bool = False,
    overlap_threshold: Optional[float] = None,
    overlap_policy: str = "clip",
    include_d_in_diag: bool = True,
) -> Tuple[CausalData, CausalEstimate]:
    m = np.asarray(m_hat, dtype=float).ravel()
    d_arr = (np.asarray(d, dtype=float).ravel() > 0.5).astype(int)
    if m.size != d_arr.size:
        raise ValueError("m_hat and d must have the same length.")

    n = int(m.size)
    x = np.linspace(-1.0, 1.0, n, dtype=float)
    y = 0.5 * d_arr + 0.2 * x

    df = pd.DataFrame(
        {
            "y": y,
            "d": d_arr,
            "x1": x + 0.1,
            "x2": x * x + 0.3,
        }
    )
    data = CausalData(df=df, treatment="d", outcome="y", confounders=["x1", "x2"])

    diag = UnconfoundednessDiagnosticData(
        m_hat=m,
        d=d_arr,
        overlap_policy=str(overlap_policy),
        overlap_threshold=float(overlap_threshold or 0.0),
        normalize_ipw=bool(normalize_ipw),
    )
    if not include_d_in_diag:
        diag = diag.model_copy(update={"d": None})

    n_treated = int(d_arr.sum())
    n_control = int(n - n_treated)
    y_t = y[d_arr == 1]
    y_c = y[d_arr == 0]

    model_options = {"normalize_ipw": bool(normalize_ipw), "overlap_policy": str(overlap_policy)}
    if overlap_threshold is not None:
        model_options["overlap_threshold"] = float(overlap_threshold)

    estimate = CausalEstimate(
        estimand="ATE",
        model="IRM",
        model_options=model_options,
        value=0.0,
        ci_upper_absolute=0.1,
        ci_lower_absolute=-0.1,
        alpha=0.05,
        p_value=1.0,
        is_significant=False,
        n_treated=n_treated,
        n_control=n_control,
        treatment_mean=float(y_t.mean()) if y_t.size else 0.0,
        control_mean=float(y_c.mean()) if y_c.size else 0.0,
        outcome="y",
        treatment="d",
        confounders=["x1", "x2"],
        diagnostic_data=diag,
    )

    return data, estimate
