"""
Causalis dataclass for storing cross-sectional DataFrame and column metadata
for instrumental variables causal inference.
"""

from __future__ import annotations

from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
import pandas.api.types as pdtypes
from pydantic import Field, field_validator

from causalis.data_contracts.causaldata import CausalData


class IVCausalData(CausalData):
    """
    Container for instrumental variables causal inference datasets.

    Extends :class:`CausalData` with exactly one instrument column. The stored
    DataFrame is restricted to outcome, treatment, instrument, confounder, and
    optional user_id columns.

    Attributes
    ----------
    df : pd.DataFrame
        DataFrame restricted to the columns used by the IV analysis.
    treatment_name : str
        Column name representing the endogenous treatment variable.
    outcome_name : str
        Column name representing the outcome variable.
    instruments_names : List[str]
        Name of the instrument column, stored as a single-item list.
    confounders_names : List[str]
        Names of the confounder columns (may be empty).
    user_id_name : str, optional
        Column name representing the unique identifier for each observation/user.
    """

    instruments_names: List[str] = Field(alias="instruments")

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        treatment: str,
        outcome: str,
        instruments: Union[str, List[str]],
        confounders: Optional[Union[str, List[str]]] = None,
        user_id: Optional[str] = None,
        **kwargs: Any,
    ) -> "IVCausalData":
        """
        Friendly constructor for IVCausalData.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the data.
        treatment : str
            Column name representing the endogenous treatment variable.
        outcome : str
            Column name representing the outcome variable.
        instruments : Union[str, List[str]]
            Column name(s) representing the instrumental variable(s).
        confounders : Union[str, List[str]], optional
            Column name(s) representing the observed confounders/covariates.
        user_id : str, optional
            Column name representing the unique identifier for each observation/user.
        **kwargs : Any
            Additional arguments passed to the Pydantic model constructor.

        Returns
        -------
        IVCausalData
            A validated IVCausalData instance.
        """
        return cls(
            df=df,
            treatment=treatment,
            outcome=outcome,
            instruments=instruments,
            confounders=confounders,
            user_id=user_id,
            **kwargs,
        )

    @field_validator("instruments_names", mode="before")
    @classmethod
    def _normalize_instruments(cls, v: Any) -> List[str]:
        """
        Normalize instruments to a single-item list of strings.

        Parameters
        ----------
        v : Any
            The instruments input, which can be a string or a list of strings.

        Returns
        -------
        List[str]
            A single-item list containing the instrument column name.

        Raises
        ------
        TypeError
            If any instrument name is not a string or if the input type is invalid.
        ValueError
            If no instruments are supplied or more than one instrument is supplied.
        """
        if v is None:
            raise TypeError(
                "instruments must be a string or a list of strings (cannot be None)."
            )
        if isinstance(v, str):
            out = [v]
        elif isinstance(v, list):
            for item in v:
                if not isinstance(item, str):
                    raise TypeError(
                        f"All instrument names must be strings. Found {type(item).__name__}: {item}"
                    )
            out = v
        else:
            raise TypeError("instruments must be a string or a list of strings.")

        if not out:
            raise ValueError("instruments cannot be empty.")
        if len(out) != 1:
            raise ValueError("IVCausalData requires exactly one instrument column.")
        return out

    def _get_roles(self) -> dict[str, str]:
        """
        Get the primary roles and their column names, including instruments.

        Returns
        -------
        dict[str, str]
            Mapping of role names to column names.
        """
        roles = super()._get_roles()
        for i, instrument in enumerate(self.instruments_names):
            roles[f"instrument[{i}]"] = instrument
        return roles

    def _get_additional_roles_error_msg(self) -> str:
        """
        Hook for adding instrument roles to inherited error messages.

        Returns
        -------
        str
            Additional role names for error messages.
        """
        return "/instruments"

    def _validate_additional_roles(self, df: pd.DataFrame):
        """
        Validate instrument columns.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to validate.

        Raises
        ------
        ValueError
            If treatment or instrument columns are non-binary, or the instrument
            column is non-numeric/non-boolean or constant.
        """
        if not self._is_binary_series(df[self.treatment_name]):
            raise ValueError(
                f"Column '{self.treatment_name}' specified as treatment must be binary encoded with values 0/1."
            )

        for col in self.instruments_names:
            if not (
                pdtypes.is_numeric_dtype(df[col]) or pdtypes.is_bool_dtype(df[col])
            ):
                raise ValueError(
                    f"Column '{col}' specified as instruments must contain only int, float, or bool values."
                )
            if not self._is_binary_series(df[col]):
                raise ValueError(
                    f"Column '{col}' specified as instrument must be binary encoded with values 0/1."
                )
            if df[col].nunique(dropna=False) <= 1:
                raise ValueError(
                    f"Column '{col}' specified as instrument is constant (has zero variance / single unique value), "
                    f"which is not allowed for instrumental variables causal inference."
                )

    @staticmethod
    def _is_binary_series(series: pd.Series) -> bool:
        """
        Check whether a Series contains only binary 0/1 or boolean values.

        Parameters
        ----------
        series : pd.Series
            Series to validate.

        Returns
        -------
        bool
            True when all non-missing values are binary.
        """
        if pdtypes.is_bool_dtype(series):
            return True
        values = series.dropna().unique()
        return set(values).issubset({0, 1})

    def _check_duplicate_column_values(self, df: pd.DataFrame):
        """
        Check for identical values in different columns, including instruments.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to check.

        Raises
        ------
        ValueError
            If two columns have identical values.
        """
        cols = (
            [self.outcome_name, self.treatment_name]
            + self.instruments_names
            + self.confounders_names
        )
        if self.user_id_name:
            cols.append(self.user_id_name)

        cols = list(dict.fromkeys(cols))

        def _values_equal_ignore_dtype(a: pd.Series, b: pd.Series) -> bool:
            return np.array_equal(
                a.to_numpy(dtype=object, copy=False),
                b.to_numpy(dtype=object, copy=False),
            )

        signatures = self._column_value_signatures(df, cols)

        for candidates in signatures.values():
            if len(candidates) < 2:
                continue

            for i, col1 in enumerate(candidates):
                for col2 in candidates[i + 1 :]:
                    if not _values_equal_ignore_dtype(df[col1], df[col2]):
                        continue
                    col1_role = self._get_column_type(col1)
                    col2_role = self._get_column_type(col2)
                    raise ValueError(
                        f"Columns '{col1}' ({col1_role}) and '{col2}' ({col2_role}) have identical values, "
                        f"which is not allowed for instrumental variables causal inference. Only column names differ."
                    )

    def _get_column_type(self, column_name: str) -> str:
        """
        Determine the type/role of a column.

        Parameters
        ----------
        column_name : str
            The name of the column.

        Returns
        -------
        str
            The role of the column.
        """
        if column_name in self.instruments_names:
            return "instrument"
        return super()._get_column_type(column_name)

    @property
    def instruments(self) -> List[str]:
        """
        List of instrument column names.

        Returns
        -------
        List[str]
            Names of the instrument columns.
        """
        return list(self.instruments_names)

    @property
    def Z(self) -> pd.DataFrame:
        """
        Design matrix of instruments.

        Returns
        -------
        pd.DataFrame
            The DataFrame containing only instrument columns.
        """
        return self.df[self.instruments_names].copy()

    def get_df(
        self,
        columns: Optional[List[str]] = None,
        include_treatment: bool = True,
        include_outcome: bool = True,
        include_confounders: bool = True,
        include_user_id: bool = False,
        include_instruments: bool = True,
    ) -> pd.DataFrame:
        """
        Get a DataFrame with specified columns.

        Parameters
        ----------
        columns : List[str], optional
            Specific column names to include.
        include_treatment : bool, default True
            Whether to include the treatment column.
        include_outcome : bool, default True
            Whether to include the outcome column.
        include_confounders : bool, default True
            Whether to include confounder columns.
        include_user_id : bool, default False
            Whether to include the user_id column.
        include_instruments : bool, default True
            Whether to include instrument columns.

        Returns
        -------
        pd.DataFrame
            A copy of the internal DataFrame with selected columns.

        Raises
        ------
        ValueError
            If any specified columns do not exist.
        """
        cols_to_include = []
        if columns is not None:
            cols_to_include.extend(columns)

        if columns is None and not any(
            [
                include_outcome,
                include_confounders,
                include_treatment,
                include_user_id,
                include_instruments,
            ]
        ):
            return self.df.iloc[:, 0:0].copy()

        if include_outcome:
            cols_to_include.append(self.outcome_name)
        if include_confounders:
            cols_to_include.extend(self.confounders_names)
        if include_treatment:
            cols_to_include.append(self.treatment_name)
        if include_instruments:
            cols_to_include.extend(self.instruments_names)
        if include_user_id and self.user_id_name:
            cols_to_include.append(self.user_id_name)

        seen = set()
        cols_to_include = [x for x in cols_to_include if not (x in seen or seen.add(x))]

        missing = [c for c in cols_to_include if c not in self.df.columns]
        if missing:
            raise ValueError(f"Column(s) {missing} do not exist in the DataFrame.")

        return self.df[cols_to_include].copy()

    def __repr__(self) -> str:
        res = (
            f"{self.__class__.__name__}(df={self.df.shape}, "
            f"treatment='{self.treatment_name}', "
            f"outcome='{self.outcome_name}', "
            f"instruments={self.instruments_names}, "
            f"confounders={self.confounders_names}"
        )
        if self.user_id_name:
            res += f", user_id='{self.user_id_name}'"
        res += ")"
        return res
