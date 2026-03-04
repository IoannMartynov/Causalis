from __future__ import annotations
import pandas as pd
import pandas.api.types as pdtypes
from typing import Union, List, Optional, Any
from pydantic import Field
from .causaldata import CausalData


class CausalDataInstrumental(CausalData):
    """
    Container for causal inference datasets with causaldata_instrumental variables.

    Attributes
    ----------
    instrument_name : str
        Column name representing the causaldata_instrumental variable.
    """
    instrument_name: str = Field(alias="instrument")

    @classmethod
    def from_df(
            cls,
            df: pd.DataFrame,
            treatment: str,
            outcome: str,
            confounders: Optional[Union[str, List[str]]] = None,
            user_id: Optional[str] = None,
            instrument: str = None,
            **kwargs: Any
    ) -> 'CausalDataInstrumental':
        """
        Friendly constructor for CausalDataInstrumental.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the data_contracts.
        treatment : str
            Column name representing the treatment variable.
        outcome : str
            Column name representing the outcome  variable.
        confounders : Union[str, List[str]], optional
            Column name(s) representing the confounders/covariates.
        user_id : str, optional
            Column name representing the unique identifier for each observation/user.
        instrument : str, optional
            Column name representing the causaldata_instrumental variable.
        **kwargs : Any
            Additional arguments passed to the Pydantic model constructor.

        Returns
        -------
        CausalDataInstrumental
            A validated CausalDataInstrumental instance.
        """
        return cls(
            df=df,
            treatment=treatment,
            outcome=outcome,
            confounders=confounders,
            user_id=user_id,
            instrument=instrument,
            **kwargs
        )

    def _get_roles(self) -> dict[str, str]:
        """
        Include instrument in primary roles.

        Returns
        -------
        dict[str, str]
            Mapping of role names to column names including instrument.
        """
        roles = super()._get_roles()
        if self.instrument_name:
            roles["instrument"] = self.instrument_name
        return roles

    def _get_additional_roles_error_msg(self) -> str:
        """
        Add instrument to the role disjoint error message.

        Returns
        -------
        str
            Additional string for the error message.
        """
        return "/instrument"

    def _validate_additional_roles(self, df: pd.DataFrame):
        """
        Validate instrument column type and variance.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to validate.

        Raises
        ------
        ValueError
            If the instrument column is not numeric/bool or is constant.
        """
        instrument = self.instrument_name
        if instrument:
            if not (pdtypes.is_numeric_dtype(df[instrument]) or pdtypes.is_bool_dtype(df[instrument])):
                raise ValueError(f"Column '{instrument}' specified as instrument must contain only int, float, or bool values.")
            if df[instrument].nunique(dropna=False) <= 1:
                raise ValueError(
                    f"Column '{instrument}' specified as instrument is constant (has zero variance / single unique value), "
                    f"which is not allowed for causal inference."
                )

    @property
    def instrument(self) -> pd.Series:
        """
        instrument column as a Series.

        Returns
        -------
        pd.Series
            The instrument column.
        """
        if not self.instrument_name or self.instrument_name not in self.df.columns:
            return pd.Series(dtype=float)
        return self.df[self.instrument_name]

    def get_df(
            self,
            columns: Optional[List[str]] = None,
            include_treatment: bool = True,
            include_outcome: bool = True,
            include_confounders: bool = True,
            include_user_id: bool = False,
            include_instrument: bool = False
    ) -> pd.DataFrame:
        """
        Get a DataFrame with specified columns including instrument.

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
        include_instrument : bool, default False
            Whether to include the instrument column.

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

        if columns is None and not any([include_outcome, include_confounders, include_treatment, include_user_id, include_instrument]):
            return self.df.copy()

        if include_outcome:
            cols_to_include.append(self.outcome_name)
        if include_confounders:
            cols_to_include.extend(self.confounders_names)
        if include_treatment:
            cols_to_include.append(self.treatment_name)
        if include_user_id and self.user_id_name:
            cols_to_include.append(self.user_id_name)
        if include_instrument and self.instrument_name:
            cols_to_include.append(self.instrument_name)

        # Remove duplicates while preserving order
        seen = set()
        cols_to_include = [x for x in cols_to_include if not (x in seen or seen.add(x))]

        # Validate existence
        missing = [c for c in cols_to_include if c not in self.df.columns]
        if missing:
            raise ValueError(f"Column(s) {missing} do not exist in the DataFrame.")

        return self.df[cols_to_include].copy()

    def __repr__(self) -> str:
        res = super().__repr__()
        # remove the closing ')'
        res = res[:-1]
        if self.instrument_name:
            res += f", instrument='{self.instrument_name}'"
        res += ")"
        return res
