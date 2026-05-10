from __future__ import annotations

import pandas as pd

from causalis.data_contracts.iv_causal_data import IVCausalData
from causalis.shared.confounders_balance import _compute_balance_table


def iv_confounders_balance(data: IVCausalData) -> pd.DataFrame:
    """
    Compute confounder balance diagnostics between instrument groups.

    Produces a DataFrame containing expanded confounder columns (after one-hot
    encoding categorical variables if present) with:
      - confounders: name of the confounder
      - mean_z_0: mean value for rows with Z=0
      - mean_z_1: mean value for rows with Z=1
      - abs_diff: abs(mean_z_1 - mean_z_0)
      - smd: standardized mean difference (Cohen's d using pooled std)
      - ks_pvalue: p-value for the KS test (rounded to 5 decimal places, non-scientific)

    Parameters
    ----------
    data : IVCausalData
        The IV causal dataset containing exactly one binary instrument column.

    Returns
    -------
    pd.DataFrame
        Balance table sorted by |smd| (descending).

    Examples
    --------
    >>> from causalis.scenarios.iv.refutation import iv_confounders_balance
    >>> # Assuming 'causal_data' is an IVCausalData object
    >>> balance_df = iv_confounders_balance(causal_data)
    >>> balance_df.head()
    """
    if not isinstance(data, IVCausalData):
        raise TypeError("iv_confounders_balance expects an IVCausalData object.")

    instrument = data.instruments[0]
    z = data.df[instrument].astype(int).to_numpy(copy=False)
    mask_z_0 = z == 0
    mask_z_1 = z == 1

    if not mask_z_0.any() or not mask_z_1.any():
        raise ValueError(
            "The instrument must have at least one row in both Z=0 and Z=1 groups."
        )

    balance = _compute_balance_table(
        df=data.df,
        confounders=data.confounders,
        mask_d_0=mask_z_0,
        mask_d_1=mask_z_1,
    )

    return balance.rename(columns={"mean_d_0": "mean_z_0", "mean_d_1": "mean_z_1"})


__all__ = ["iv_confounders_balance"]
