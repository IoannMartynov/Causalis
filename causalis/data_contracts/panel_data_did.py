from __future__ import annotations

from datetime import date, datetime
from typing import Any, Hashable, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator


TimeLike = Union[str, date, datetime, pd.Timestamp, pd.Period]


class PanelDataDID(BaseModel):
    """Validated long-format panel contract for canonical/simultaneous-adoption DID.

    Required fields
    ---------------
    df : pandas.DataFrame
        Long-format panel data.
    y : str
        Outcome column name in ``df``.
    unit_col : str
        Unit identifier column name in ``df``.
    time_col : str
        Calendar time column name in ``df``.
        Preferred input format is ``pandas.Period`` values with a regular
        frequency (for example monthly ``Period['M']``). Datetime/string values
        are accepted only when a regular frequency can be inferred.
    treated_time : str
        Binary treatment-assignment column in ``df`` (0/1 or False/True).
    covariates : sequence of str, optional
        Optional numeric covariate column names in ``df``. The input alias
        ``covariants`` is accepted for convenience.
    cluster_col : str, optional
        Optional cluster identifier column name in ``df``. The input alias
        ``cluster`` is accepted for convenience. ``unit_col`` and ``time_col``
        are valid cluster columns.

    Notes
    -----
    Extra keyword arguments are rejected.
    The contract derives ``treated_units``, ``control_units``,
    ``treatment_start``, and ``time_freq`` from the input data.
    This contract is for non-staggered difference-in-differences designs:
    every ever-treated unit must be untreated before the common adoption time
    and treated at/after that time; control units must never be treated.
    The model stores a validated internal dataframe snapshot used by all contract
    methods; mutating the public ``df`` attribute after construction does not
    affect validated contract behavior.
    Outcome ``y`` must not contain null/NaN values. Represent missing panel
    periods by omitting unit-time rows, not by keeping rows with ``NaN`` outcome.
    For fiscal quarter/year semantics, pass ``time_col`` explicitly as
    ``pandas.Period`` with the desired fiscal frequency.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        frozen=True,
        populate_by_name=True,
    )

    df: pd.DataFrame = Field(..., description="Long-format panel data.")
    y: str = Field(..., description="Outcome column in df.")
    unit_col: str = Field(..., description="Unit identifier column in df.")
    time_col: str = Field(..., description="Calendar time column in df.")
    treated_time: str = Field(..., description="Binary treatment-assignment column in df.")
    covariates: tuple[str, ...] = Field(
        default_factory=tuple,
        validation_alias=AliasChoices("covariates", "covariants"),
        serialization_alias="covariates",
        description="Optional numeric covariate columns in df.",
    )
    cluster_col: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("cluster_col", "cluster"),
        serialization_alias="cluster_col",
        description="Optional cluster identifier column in df.",
    )

    _treated_units: Optional[tuple[Hashable, ...]] = PrivateAttr(default=None)
    _control_units: Optional[tuple[Hashable, ...]] = PrivateAttr(default=None)
    _treatment_start: Optional[pd.Period] = PrivateAttr(default=None)
    _time_freq: Optional[str] = PrivateAttr(default=None)
    _n_pre_periods: Optional[int] = PrivateAttr(default=None)
    _n_post_periods: Optional[int] = PrivateAttr(default=None)
    _df_validated: Optional[pd.DataFrame] = PrivateAttr(default=None)

    @staticmethod
    def _normalize_column_name(value: Any, field_name: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{field_name} entries must be strings.")
        out = value.strip()
        if not out:
            raise ValueError(f"{field_name} entries must be non-empty strings.")
        return out

    @field_validator("covariates", mode="before")
    @classmethod
    def _normalize_covariates(cls, value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            raw = [value]
        elif isinstance(value, (list, tuple)):
            raw = list(value)
        else:
            raise TypeError("covariates must be None, a string, or a list/tuple of strings.")

        seen = set()
        out = []
        for item in raw:
            covariate = cls._normalize_column_name(item, "covariates")
            if covariate not in seen:
                out.append(covariate)
                seen.add(covariate)
        return tuple(out)

    @field_validator("cluster_col", mode="before")
    @classmethod
    def _normalize_cluster_col(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        return cls._normalize_column_name(value, "cluster_col")

    @staticmethod
    def _canonical_period_freq(freq: str) -> Optional[str]:
        try:
            return pd.period_range(start="2000-01-01", periods=1, freq=freq).freqstr
        except Exception:
            return None

    def _normalize_inferred_freq(self, freq: Optional[str]) -> Optional[str]:
        if freq is None:
            return None

        raw = str(freq).upper()
        candidates = [raw]
        if raw.startswith("MS") or raw.startswith("ME"):
            candidates.append("M")
        elif raw.startswith("QS") or raw.startswith("Q"):
            candidates.append("Q")
        elif raw.startswith("AS") or raw.startswith("YS") or raw.startswith("A") or raw.startswith("Y"):
            candidates.append("Y")

        for candidate in candidates:
            canonical = self._canonical_period_freq(candidate)
            if canonical is not None:
                return canonical
        return None

    def _infer_period_freq_from_datetimes(self, dt: pd.Series) -> str:
        unique_times = pd.DatetimeIndex(sorted(pd.Index(dt.unique()).tolist()))
        if unique_times.empty:
            raise ValueError(f"{self.time_col!r} must contain at least one time value.")
        if len(unique_times) < 3:
            raise ValueError(
                f"Could not infer a regular period frequency from {self.time_col!r} with fewer than 3 unique "
                "datetime values. Provide pandas.Period values in the time column."
            )

        inferred = pd.infer_freq(unique_times)
        normalized = self._normalize_inferred_freq(inferred)
        if normalized is not None:
            return normalized

        raise ValueError(
            f"Could not infer a regular period frequency from {self.time_col!r}. "
            "Use pandas.Period values with an explicit frequency."
        )

    def _reject_numeric_time_series(self, s: pd.Series) -> None:
        if pd.api.types.is_numeric_dtype(s):
            raise ValueError(
                f"{self.time_col!r} must represent explicit calendar time, not numeric values. "
                "Use pandas.Period or datetime-like values."
            )

        numeric_cast = pd.to_numeric(s, errors="coerce")
        if bool(numeric_cast.notna().all()):
            raise ValueError(
                f"{self.time_col!r} must represent explicit calendar time, not numeric-like values. "
                "Use pandas.Period or datetime-like values."
            )

    def _coerce_period_series(self, s: pd.Series) -> tuple[pd.Series, str]:
        if isinstance(s.dtype, pd.PeriodDtype):
            freq = pd.PeriodIndex(s).freqstr
            if freq is None:
                raise ValueError(f"{self.time_col!r} period frequency is missing.")
            return s, freq

        if isinstance(s.dtype, pd.DatetimeTZDtype):
            raise ValueError(
                f"{self.time_col!r} contains timezone-aware datetimes; normalize to naive/UTC first."
            )

        if bool(s.map(lambda v: isinstance(v, pd.Period)).all()):
            period_index = pd.PeriodIndex(s)
            freq = period_index.freqstr
            if freq is None:
                raise ValueError(f"{self.time_col!r} period frequency is missing.")
            return pd.Series(period_index, index=s.index, name=s.name), freq

        self._reject_numeric_time_series(s)
        dt = pd.to_datetime(s, errors="raise")
        if getattr(dt.dt, "tz", None) is not None:
            raise ValueError(
                f"{self.time_col!r} contains timezone-aware datetimes; normalize to naive/UTC first."
            )

        freq = self._infer_period_freq_from_datetimes(dt)
        try:
            return dt.dt.to_period(freq), freq
        except Exception as exc:
            raise ValueError(
                f"Failed to coerce {self.time_col!r} to Period[{freq}]. "
                "Use pandas.Period values with explicit frequency."
            ) from exc

    @model_validator(mode="after")
    def _validate_schema(self) -> "PanelDataDID":
        df = self.df
        if df.empty:
            raise ValueError("df must contain at least one row.")

        covariates = list(self.covariates)
        cluster_col = self.cluster_col
        primary_roles = {self.unit_col, self.time_col, self.y, self.treated_time}

        required = primary_roles | set(covariates)
        if cluster_col is not None:
            required.add(cluster_col)
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        role_columns = [self.unit_col, self.time_col, self.y, self.treated_time]
        if len(set(role_columns)) != len(role_columns):
            raise ValueError("Column role names must be distinct across unit_col, time_col, y, treated_time.")

        covariate_role_overlap = sorted(set(covariates).intersection(primary_roles))
        if covariate_role_overlap:
            raise ValueError(
                "covariates must be distinct from unit_col, time_col, y, and treated_time. "
                f"Overlapping columns: {covariate_role_overlap}"
            )
        if cluster_col is not None:
            if cluster_col in {self.y, self.treated_time}:
                raise ValueError("cluster_col must be distinct from y and treated_time.")
            if cluster_col in set(covariates):
                raise ValueError("cluster_col must be distinct from covariates.")

        if df[self.unit_col].isna().any():
            raise ValueError(f"{self.unit_col!r} contains nulls.")
        if df[self.time_col].isna().any():
            raise ValueError(f"{self.time_col!r} contains nulls.")
        if df[self.y].isna().any():
            raise ValueError(
                f"{self.y!r} contains nulls. Represent missing periods by omitting rows, not NaN outcomes."
            )
        if df[self.treated_time].isna().any():
            raise ValueError(f"{self.treated_time!r} contains nulls.")
        for covariate in covariates:
            if df[covariate].isna().any():
                raise ValueError(f"Covariate {covariate!r} contains nulls.")
        if cluster_col is not None and df[cluster_col].isna().any():
            raise ValueError(f"{cluster_col!r} contains nulls.")

        df = df.copy()

        try:
            coerced_time, time_freq = self._coerce_period_series(df[self.time_col])
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"Failed to normalize {self.time_col!r}.") from exc
        df[self.time_col] = coerced_time

        y_num = pd.to_numeric(df[self.y], errors="coerce")
        created_nan = y_num.isna() & ~df[self.y].isna()
        if bool(created_nan.any()):
            raise ValueError(f"{self.y!r} contains non-numeric values.")
        df[self.y] = y_num

        for covariate in covariates:
            covariate_num = pd.to_numeric(df[covariate], errors="coerce")
            created_covariate_nan = covariate_num.isna() & ~df[covariate].isna()
            if bool(created_covariate_nan.any()):
                raise ValueError(f"Covariate {covariate!r} contains non-numeric values.")
            if not np.isfinite(covariate_num.to_numpy(dtype=float)).all():
                raise ValueError(f"Covariate {covariate!r} must contain only finite numeric values.")
            if covariate_num.nunique(dropna=False) <= 1:
                raise ValueError(f"Covariate {covariate!r} is constant.")
            df[covariate] = covariate_num

        treatment_raw = df[self.treated_time]
        treatment_allowed = {0, 1, True, False}
        if not set(treatment_raw.dropna().unique()).issubset(treatment_allowed):
            raise ValueError(f"{self.treated_time!r} must be boolean or 0/1.")
        treatment = treatment_raw.astype(int)
        df[self.treated_time] = treatment

        treatment_conflicts = (
            df.groupby([self.unit_col, self.time_col], sort=False)[self.treated_time]
            .nunique(dropna=False)
            .gt(1)
        )
        if bool(treatment_conflicts.any()):
            raise ValueError(
                f"{self.treated_time!r} must be consistent within each ({self.unit_col}, {self.time_col}) cell."
            )

        if bool(df.duplicated([self.unit_col, self.time_col]).any()):
            raise ValueError(
                f"Duplicate (unit,time) rows found in [{self.unit_col}, {self.time_col}]. "
                "Aggregate duplicated rows before contract construction."
            )

        treated_rows = df[df[self.treated_time] == 1]
        if treated_rows.empty:
            raise ValueError(f"{self.treated_time!r} must have at least one treated row (value 1).")

        units = pd.Index(df[self.unit_col].unique()).tolist()
        treated_units = tuple(pd.Index(treated_rows[self.unit_col].unique()).tolist())
        control_units = tuple(u for u in units if u not in set(treated_units))
        if not control_units:
            raise ValueError("Need at least one never-treated control unit.")

        starts_by_unit = (
            treated_rows.groupby(self.unit_col, sort=False)[self.time_col]
            .min()
            .to_dict()
        )
        treatment_starts = sorted(set(starts_by_unit.values()))
        if len(treatment_starts) != 1:
            raise ValueError(
                f"{self.treated_time!r} must define simultaneous adoption; "
                f"found treatment starts by unit: {starts_by_unit}"
            )
        treatment_start = treatment_starts[0]

        treated_unit_rows = df[df[self.unit_col].isin(treated_units)]
        if bool(
            (
                treated_unit_rows.loc[treated_unit_rows[self.time_col] < treatment_start, self.treated_time]
                != 0
            ).any()
        ):
            raise ValueError(f"{self.treated_time!r} for treated_units must be 0 before treatment_start.")
        if bool(
            (
                treated_unit_rows.loc[treated_unit_rows[self.time_col] >= treatment_start, self.treated_time]
                != 1
            ).any()
        ):
            raise ValueError(f"{self.treated_time!r} for treated_units must be 1 at/after treatment_start.")

        control_rows = df[df[self.unit_col].isin(control_units)]
        if bool((control_rows[self.treated_time] != 0).any()):
            raise ValueError(f"{self.treated_time!r} must be 0 for all never-treated control units.")

        projected_cols = [self.unit_col, self.time_col, self.treated_time, self.y]
        projected_cols.extend(covariates)
        if cluster_col is not None and cluster_col not in projected_cols:
            projected_cols.append(cluster_col)
        projected = df[projected_cols].copy()

        object.__setattr__(self, "_df_validated", projected.copy(deep=True))
        object.__setattr__(self, "df", projected)
        object.__setattr__(self, "_treated_units", treated_units)
        object.__setattr__(self, "_control_units", control_units)
        object.__setattr__(self, "_treatment_start", treatment_start)
        object.__setattr__(self, "_time_freq", time_freq)

        analysis_times = list(self.analysis_times())
        expected_times = list(
            pd.period_range(start=min(analysis_times), end=max(analysis_times), freq=self.time_freq)
        )
        if analysis_times != expected_times:
            missing_times = sorted(set(expected_times) - set(analysis_times))
            raise ValueError(
                "Analysis time axis has gaps relative to inferred time_freq. "
                f"Missing periods: {missing_times}"
            )

        pre_times = list(self.pre_times())
        post_times = list(self.post_times())
        if not pre_times:
            raise ValueError("No pre-treatment periods available.")
        if not post_times:
            raise ValueError("No post-treatment periods available.")

        analysis_df = self.df_analysis()
        missing_cells: list[tuple[pd.Period, str]] = []
        for time in analysis_times:
            time_rows = analysis_df[analysis_df[self.time_col] == time]
            if not bool(time_rows[self.unit_col].isin(treated_units).any()):
                missing_cells.append((time, "treated"))
            if not bool(time_rows[self.unit_col].isin(control_units).any()):
                missing_cells.append((time, "control"))
        if missing_cells:
            raise ValueError(
                "Each analysis period must contain at least one treated-group row and one control-group row. "
                f"Missing cells: {missing_cells}"
            )

        object.__setattr__(self, "_n_pre_periods", len(pre_times))
        object.__setattr__(self, "_n_post_periods", len(post_times))

        return self

    @property
    def treated_units(self) -> Sequence[Hashable]:
        if self._treated_units is None:
            raise RuntimeError("treated_units metadata is not initialized.")
        return self._treated_units

    @property
    def control_units(self) -> Sequence[Hashable]:
        if self._control_units is None:
            raise RuntimeError("control_units metadata is not initialized.")
        return self._control_units

    @property
    def treatment_start(self) -> pd.Period:
        if self._treatment_start is None:
            raise RuntimeError("treatment_start metadata is not initialized.")
        return self._treatment_start

    @property
    def time_freq(self) -> str:
        if self._time_freq is None:
            raise RuntimeError("time_freq metadata is not initialized.")
        return self._time_freq

    @property
    def n_pre_periods(self) -> int:
        if self._n_pre_periods is None:
            raise RuntimeError("n_pre_periods metadata is not initialized.")
        return self._n_pre_periods

    @property
    def n_post_periods(self) -> int:
        if self._n_post_periods is None:
            raise RuntimeError("n_post_periods metadata is not initialized.")
        return self._n_post_periods

    @property
    def last_post_period(self) -> pd.Period:
        post_times = self.post_times()
        if not post_times:
            raise RuntimeError("last_post_period is not available because there are no post-treatment periods.")
        return post_times[-1]

    @property
    def design_type(self) -> Literal["canonical_2x2", "simultaneous_adoption"]:
        if self.n_pre_periods == 1 and self.n_post_periods == 1:
            return "canonical_2x2"
        return "simultaneous_adoption"

    @property
    def has_covariates(self) -> bool:
        return bool(self.covariates)

    @property
    def has_cluster(self) -> bool:
        return self.cluster_col is not None

    def _validated_df(self) -> pd.DataFrame:
        if self._df_validated is None:
            raise RuntimeError("validated dataframe snapshot is not initialized.")
        return self._df_validated

    def df_analysis(self) -> pd.DataFrame:
        return self._validated_df().copy()

    def covariate_frame(self) -> pd.DataFrame:
        """Return a copy of the validated covariate design columns."""

        return self.df_analysis()[list(self.covariates)].copy()

    def cluster_series(self) -> pd.Series:
        """Return a copy of the validated cluster identifier column."""

        if self.cluster_col is None:
            raise RuntimeError("cluster_col is not set.")
        return self.df_analysis()[self.cluster_col].copy()

    def df_for_did(
        self,
        *,
        treated_group_col: str = "treated_group",
        post_col: str = "post",
    ) -> pd.DataFrame:
        """Return analysis data with derived DID group and post indicators."""

        reserved = set(self.df_analysis().columns)
        extra_cols = {treated_group_col, post_col}
        if len(extra_cols) != 2:
            raise ValueError("treated_group_col and post_col must be distinct.")
        overlap = reserved.intersection(extra_cols)
        if overlap:
            raise ValueError(f"Derived DID column names conflict with contract columns: {sorted(overlap)}")

        df = self.df_analysis()
        df[treated_group_col] = df[self.unit_col].isin(set(self.treated_units)).astype(int)
        df[post_col] = (df[self.time_col] >= self.treatment_start).astype(int)
        return df

    def pre_times(self) -> Sequence[pd.Period]:
        times = pd.Index(self.df_analysis()[self.time_col].unique()).tolist()
        return sorted([t for t in times if t < self.treatment_start])

    def post_times(self) -> Sequence[pd.Period]:
        times = pd.Index(self.df_analysis()[self.time_col].unique()).tolist()
        return sorted([t for t in times if t >= self.treatment_start])

    def analysis_times(self) -> Sequence[pd.Period]:
        times = pd.Index(self.df_analysis()[self.time_col].unique()).tolist()
        return sorted(times)

    def time_to_index(self) -> dict[pd.Period, int]:
        times = self.analysis_times()
        return {t: i for i, t in enumerate(times)}

    def treatment_start_idx(self) -> int:
        mapping = self.time_to_index()
        if self.treatment_start not in mapping:
            raise ValueError("treatment_start is not present in analysis time axis.")
        return mapping[self.treatment_start]

    def cell_counts(self) -> pd.DataFrame:
        """Return treated/control row counts by analysis period and pre/post cell."""

        df = self.df_analysis()
        counts_input = pd.DataFrame(
            {
                "_did_time": df[self.time_col].to_numpy(),
                "_did_post": (df[self.time_col] >= self.treatment_start).astype(int).to_numpy(),
                "_did_treated_group": df[self.unit_col].isin(set(self.treated_units)).astype(int).to_numpy(),
            }
        )
        counts = (
            counts_input.groupby(["_did_time", "_did_post", "_did_treated_group"], observed=True)
            .size()
            .rename("n")
            .reset_index()
        )
        counts = counts.rename(columns={"_did_time": self.time_col, "_did_post": "post"})
        counts["group"] = counts["_did_treated_group"].map({0: "control", 1: "treated"})
        return counts[[self.time_col, "post", "group", "n"]]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(df={self.df.shape}, "
            f"y={self.y!r}, "
            f"unit_col={self.unit_col!r}, "
            f"time_col={self.time_col!r}, "
            f"treated_time={self.treated_time!r}, "
            f"covariates={list(self.covariates)!r}, "
            f"cluster_col={self.cluster_col!r}, "
            f"time_freq={self.time_freq!r}, "
            f"design_type={self.design_type!r}, "
            f"treatment_start={self.treatment_start!r}, "
            f"last_post_period={self.last_post_period!r}, "
            f"n_pre_periods={self.n_pre_periods!r}, "
            f"n_post_periods={self.n_post_periods!r}, "
            f"treated_units={list(self.treated_units)!r}, "
            f"control_units={list(self.control_units)!r})"
        )
