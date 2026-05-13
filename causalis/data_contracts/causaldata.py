"""
Causalis Dataclass for storing Cross-sectional DataFrame and column metadata for causal inference.
"""

from __future__ import annotations

import hashlib

import pandas as pd
import numpy as np
import pandas.api.types as pdtypes
from typing import Union, List, Optional, Any, ClassVar
from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator


class CausalData(BaseModel):
    """
    Container for causal inference datasets.

    Wraps a pandas DataFrame and stores the names of treatment, outcome, and optional confounder columns.
    The stored DataFrame is restricted to only those columns.
    Uses Pydantic for validation and as a data_contracts contract.

    Attributes
    ----------
    df : pd.DataFrame
        The DataFrame containing the data_contracts restricted to outcome, treatment, and confounder columns.
        NaN values are not allowed in the used columns.
    treatment_name : str
        Column name representing the treatment variable.
    outcome_name : str
        Column name representing the outcome variable.
    confounders_names : List[str]
        Names of the confounder columns (may be empty).
    user_id_name : str, optional
        Column name representing the unique identifier for each observation/user.
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True,
        extra="forbid",
    )

    df: pd.DataFrame
    treatment_name: str = Field(alias="treatment")
    outcome_name: str = Field(alias="outcome")
    confounders_names: List[str] = Field(alias="confounders", default_factory=list)
    user_id_name: Optional[str] = Field(alias="user_id", default=None)

    @classmethod
    def from_df(
            cls,
            df: pd.DataFrame,
            treatment: str,
            outcome: str,
            confounders: Optional[Union[str, List[str]]] = None,
            user_id: Optional[str] = None,
            **kwargs: Any
    ) -> 'CausalData':
        """
        Friendly constructor for CausalData.

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
        **kwargs : Any
            Additional arguments passed to the Pydantic model constructor.

        Returns
        -------
        CausalData
            A validated CausalData instance.
        """
        return cls(
            df=df,
            treatment=treatment,
            outcome=outcome,
            confounders=confounders,
            user_id=user_id,
            **kwargs
        )

    @field_validator("confounders_names", mode="before")
    @classmethod
    def _normalize_confounders(cls, v: Any) -> List[str]:
        """
        Normalize confounders to a list of unique strings.

        Parameters
        ----------
        v : Any
            The confounders input, which can be None, a string, or a list of strings.

        Returns
        -------
        List[str]
            A list of unique confounder column names.

        Raises
        -------
        TypeError
            If any confounder name is not a string or if the input type is invalid.
        """
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        if isinstance(v, list):
            for item in v:
                if not isinstance(item, str):
                    raise TypeError(f"All confounder names must be strings. Found {type(item).__name__}: {item}")
            seen = set()
            unique_confs = []
            for c in v:
                if c not in seen:
                    unique_confs.append(c)
                    seen.add(c)
            return unique_confs
        raise TypeError("confounders must be None, a string, or a list of strings.")

    @model_validator(mode='after')
    def _validate_and_normalize(self) -> 'CausalData':
        """
        Perform complex validation and normalize the stored DataFrame.

        This validator:
        1. Checks for duplicate column names in the input DataFrame.
        2. Ensures role columns (treatment, outcome, user_id) are disjoint.
        3. Verifies that all specified columns exist in the DataFrame.
        4. Validates types and checks for constant variance in outcome, treatment, and confounders.
        5. Ensures no NaN values are present in used columns.
        6. Subsets the DataFrame to used columns and coerces binary treatment to int8.
        7. Checks for duplicate column values.
        8. Verifies user_id uniqueness.

        Returns
        -------
        CausalData
            The validated and normalized CausalData instance.

        Raises
        ------
        ValueError
            If any validation step fails (e.g., missing columns, NaN values, constant roles).
        """
        df = self.df
        treatment = self.treatment_name
        outcome = self.outcome_name
        confounders = self.confounders_names
        user_id = self.user_id_name

        # 0. Guard against duplicate column names
        if df.columns.has_duplicates:
            dupes = df.columns[df.columns.duplicated()].unique().tolist()
            raise ValueError(f"DataFrame has duplicate column names: {dupes}. This is not supported.")

        # 1. Disjoint role validation
        roles = self._get_roles()

        # Check for overlaps between primary roles
        role_names = list(roles.keys())
        for i, r1 in enumerate(role_names):
            for r2 in role_names[i+1:]:
                if roles[r1] == roles[r2]:
                    raise ValueError(f"Column '{roles[r1]}' cannot be both {r1} and {r2}.")

        overlap = [c for c in confounders if c in set(roles.values())]
        if overlap:
            raise ValueError(
                "confounder columns must be disjoint from treatment/outcome/user_id" + self._get_additional_roles_error_msg() + "; overlapping columns: "
                + ", ".join(overlap)
            )

        # 2. Check if all specified columns exist in df
        all_columns = set(df.columns)
        for role_name, col in roles.items():
            if col not in all_columns:
                raise ValueError(f"Column '{col}' specified as {role_name} does not exist in the DataFrame.")
        
        for col in confounders:
            if col not in all_columns:
                raise ValueError(f"Column '{col}' specified as confounders does not exist in the DataFrame.")

        # 4. Validate types and check for constant variance
        # Outcome
        if not (pdtypes.is_numeric_dtype(df[outcome]) or pdtypes.is_bool_dtype(df[outcome])):
            raise ValueError(f"Column '{outcome}' specified as outcome must contain only int, float, or bool values.")
        if self._is_constant_series(df[outcome]):
            raise ValueError(
                f"Column '{outcome}' specified as outcome is constant (has zero variance / single unique value), "
                f"which is not allowed for causal inference."
            )

        # Treatment
        if not (pdtypes.is_numeric_dtype(df[treatment]) or pdtypes.is_bool_dtype(df[treatment])):
            raise ValueError(f"Column '{treatment}' specified as treatment must contain only int, float, or bool values.")
        if self._is_constant_series(df[treatment]):
            raise ValueError(
                f"Column '{treatment}' specified as treatment is constant (has zero variance / single unique value), "
                f"which is not allowed for causal inference."
            )

        self._validate_additional_roles(df)

        # confounders
        for col in confounders:
            if not (pdtypes.is_numeric_dtype(df[col]) or pdtypes.is_bool_dtype(df[col])):
                raise ValueError(f"Column '{col}' specified as confounders must contain only int, float, or bool values.")
            
            if self._is_constant_series(df[col]):
                raise ValueError(
                    f"Column '{col}' specified as confounder is constant (has zero variance / single unique value), "
                    f"which is not allowed for causal inference."
                )

        # 5. Check for NaN values in used columns
        cols_to_check = list(roles.values()) + confounders
        
        # Unique columns preserving order
        cols_to_check = list(dict.fromkeys(cols_to_check))
        
        if df[cols_to_check].isna().any().any():
            raise ValueError("DataFrame contains NaN values in used columns, which are not allowed.")

        # 5. Store only the relevant columns and canonicalize treatment.
        self.df = df[cols_to_check].copy()
        self.df[treatment] = self._validate_and_cast_binary_treatment(self.df[treatment], treatment)
        for col in self.df.columns:
            if col != user_id and pdtypes.is_bool_dtype(self.df[col]):
                self.df[col] = self.df[col].astype("int8")
            
            # Final safeguard for numeric types (excluding user_id)
            if col != user_id:
                if not pdtypes.is_numeric_dtype(self.df[col]):
                    raise ValueError(
                        f"All columns in stored DataFrame must be numeric; column '{col}' has dtype {self.df[col].dtype}."
                    )

        # 6. Check for duplicate column values on the normalized subset
        self._check_duplicate_column_values(self.df)
        self._check_user_id_uniqueness(self.df)

        return self

    @staticmethod
    def _validate_and_cast_binary_treatment(series: pd.Series, name: str) -> pd.Series:
        """Validate exact {0,1}/boolean treatment encoding and cast to int8."""
        if pdtypes.is_bool_dtype(series):
            return series.astype("int8")

        if not pdtypes.is_numeric_dtype(series):
            raise ValueError(
                f"Column '{name}' specified as treatment must contain only int, float, or bool values."
            )

        values = pd.unique(series)
        if not np.isfinite(np.asarray(values, dtype=float)).all():
            raise ValueError(
                f"Column '{name}' specified as treatment must contain only finite binary values 0/1."
            )

        value_set = set(np.asarray(values, dtype=float).tolist())
        if value_set != {0.0, 1.0}:
            raise ValueError(
                f"Column '{name}' specified as treatment must be binary encoded in {{0,1}}. "
                f"Found values: {sorted(value_set)}"
            )
        return series.astype("int8")

    @staticmethod
    def _is_constant_series(series: pd.Series) -> bool:
        """Return whether all values are equal without building a unique-value set."""
        if len(series) == 0:
            return True

        values = series.to_numpy(copy=False)
        try:
            return bool(np.all(values == values[0]))
        except (TypeError, ValueError):
            return bool(series.nunique(dropna=False) <= 1)

    def _get_roles(self) -> dict[str, str]:
        """
        Get the primary roles and their column names.

        Returns
        -------
        dict[str, str]
            Mapping of role names to column names.
        """
        roles = {}
        if self.user_id_name:
            roles["user_id"] = self.user_id_name
        roles["outcome"] = self.outcome_name
        roles["treatment"] = self.treatment_name
        return roles

    def _get_additional_roles_error_msg(self) -> str:
        """
        Hook for subclasses to add roles to error messages.

        Returns
        -------
        str
            Additional string for the error message.
        """
        return ""

    def _validate_additional_roles(self, df: pd.DataFrame):
        """
        Hook for subclasses to validate additional roles.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to validate.
        """
        pass

    def _check_user_id_uniqueness(self, df: pd.DataFrame):
        """
        Check if user_id column has unique values.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to check.

        Raises
        ------
        ValueError
            If the user_id column contains duplicate values.
        """
        if self.user_id_name and df[self.user_id_name].duplicated().any():
            raise ValueError(f"Column '{self.user_id_name}' specified as user_id contains duplicate values.")

    def _check_duplicate_column_values(self, df: pd.DataFrame):
        """
        Check for identical values in different columns.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to check.

        Raises
        ------
        ValueError
            If two columns have identical values.
        """
        cols = [self.outcome_name, self.treatment_name] + self.confounders_names
        if self.user_id_name:
            cols.append(self.user_id_name)

        # Unique columns preserving order
        cols = list(dict.fromkeys(cols))

        def _values_equal_ignore_dtype(a: pd.Series, b: pd.Series) -> bool:
            # NaNs are forbidden earlier, so array_equal is safe here.
            return np.array_equal(
                a.to_numpy(dtype=object, copy=False),
                b.to_numpy(dtype=object, copy=False),
            )

        signatures: dict[tuple[str, int, str], list[str]] = {}
        for col in cols:
            signature = self._column_value_signature(df[col])
            signatures.setdefault(signature, []).append(col)

        for candidates in signatures.values():
            if len(candidates) < 2:
                continue

            for i, col1 in enumerate(candidates):
                for j in range(i + 1, len(candidates)):
                    col2 = candidates[j]
                    if _values_equal_ignore_dtype(df[col1], df[col2]):
                        col1_role = self._get_column_type(col1)
                        col2_role = self._get_column_type(col2)
                        raise ValueError(
                            f"Columns '{col1}' ({col1_role}) and '{col2}' ({col2_role}) have identical values, "
                            f"which is not allowed for causal inference. Only column names differ."
                        )

    @staticmethod
    def _column_value_signature(series: pd.Series) -> tuple[str, int, str]:
        """Return a dtype-agnostic value fingerprint for duplicate-column screening."""
        hasher = hashlib.blake2b(digest_size=16)

        if pdtypes.is_numeric_dtype(series) or pdtypes.is_bool_dtype(series):
            values = series.to_numpy(dtype=np.float64, copy=True)
            # Keep dtype-agnostic equality consistent for 0.0 and -0.0.
            values[values == 0.0] = 0.0
            hasher.update(np.ascontiguousarray(values).view(np.uint8))
            return ("numeric", len(series), hasher.hexdigest())

        hashed = pd.util.hash_pandas_object(series, index=False, categorize=True).to_numpy(
            dtype=np.uint64,
            copy=False,
        )
        hasher.update(np.ascontiguousarray(hashed).view(np.uint8))
        return ("object", len(series), hasher.hexdigest())


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
            The role of the column ('outcome', 'treatment', 'user_id', 'confounder', or 'unknown').
        """
        if column_name == self.outcome_name:
            return "outcome"
        elif column_name == self.treatment_name:
            return "treatment"
        elif self.user_id_name and column_name == self.user_id_name:
            return "user_id"
        elif column_name in self.confounders_names:
            return "confounder"
        return "unknown"

    @property
    def outcome(self) -> pd.Series:
        """
        Outcome column as a Series.

        Returns
        -------
        pd.Series
            The outcome column.
        """
        if not self.outcome_name or self.outcome_name not in self.df.columns:
            return pd.Series(dtype=float)
        return self.df[self.outcome_name]

    @property
    def treatment(self) -> pd.Series:
        """
        Treatment column as a Series.

        Returns
        -------
        pd.Series
            The treatment column.
        """
        if not self.treatment_name or self.treatment_name not in self.df.columns:
            return pd.Series(dtype=float)
        return self.df[self.treatment_name]

    @property
    def confounders(self) -> List[str]:
        """
        List of confounder column names.

        Returns
        -------
        List[str]
            Names of the confounder columns.
        """
        return list(self.confounders_names)

    @property
    def user_id(self) -> pd.Series:
        """
        user_id column as a Series.

        Returns
        -------
        pd.Series
            The user_id column.
        """
        if not self.user_id_name or self.user_id_name not in self.df.columns:
            return pd.Series(dtype=object)
        return self.df[self.user_id_name]

    @property
    def X(self) -> pd.DataFrame:
        """
        Design matrix of confounders.

        Returns
        -------
        pd.DataFrame
            The DataFrame containing only confounder columns.
        """
        if not self.confounders_names:
            return self.df.iloc[:, 0:0].copy()
        return self.df[self.confounders_names].copy()

    def get_df(
            self,
            columns: Optional[List[str]] = None,
            include_treatment: bool = True,
            include_outcome: bool = True,
            include_confounders: bool = True,
            include_user_id: bool = False,
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

        if columns is None and not any([include_outcome, include_confounders, include_treatment, include_user_id]):
            return self.df.iloc[:, 0:0].copy()  # empty frame with same index

        if include_outcome:
            cols_to_include.append(self.outcome_name)
        if include_confounders:
            cols_to_include.extend(self.confounders_names)
        if include_treatment:
            cols_to_include.append(self.treatment_name)
        if include_user_id and self.user_id_name:
            cols_to_include.append(self.user_id_name)

        # Remove duplicates while preserving order
        seen = set()
        cols_to_include = [x for x in cols_to_include if not (x in seen or seen.add(x))]

        # Validate existence
        missing = [c for c in cols_to_include if c not in self.df.columns]
        if missing:
            raise ValueError(f"Column(s) {missing} do not exist in the DataFrame.")

        return self.df[cols_to_include].copy()

    def __repr__(self) -> str:
        res = (
            f"{self.__class__.__name__}(df={self.df.shape}, "
            f"treatment='{self.treatment_name}', "
            f"outcome='{self.outcome_name}', "
            f"confounders={self.confounders_names}"
        )
        if self.user_id_name:
            res += f", user_id='{self.user_id_name}'"
        res += ")"
        return res
