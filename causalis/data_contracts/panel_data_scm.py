from __future__ import annotations

from datetime import date, datetime
from typing import Hashable, Optional, Sequence, Tuple, Union

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator


TimeLike = Union[str, date, datetime, pd.Timestamp, pd.Period]


class PanelDataSCM(BaseModel):
    """Validated long-format panel contract for Synthetic Control estimators.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-format panel data, one row per observed ``(unit, time)`` cell.
    unit_col : str
        Column name that identifies observational units.
    time_col : str
        Column name containing explicit calendar time values.
    time_freq : str, default="M"
        Regular panel frequency alias understood by pandas Period, for example
        ``"D"``, ``"W"``, ``"M"``, ``"Q"``, or ``"Y"``.
    y : str
        Outcome column name.
    treated_unit : Hashable
        Identifier of the treated unit.
    treatment_start : TimeLike
        First treated period (inclusive): pre periods satisfy
        ``t < treatment_start`` and post periods satisfy
        ``t >= treatment_start``.
    donor_units : sequence of Hashable, optional
        Explicit donor pool. If ``None``, all non-treated units are donors.
    time_window : tuple(TimeLike or None, TimeLike or None), optional
        Inclusive analysis window ``(t_min, t_max)`` after time coercion.
    pre_periods : sequence of TimeLike, optional
        Explicit pre-treatment periods. If provided, they override inferred pre periods.
    post_periods : sequence of TimeLike, optional
        Explicit post-treatment periods. If provided, they override inferred post periods.
    covariate_cols : sequence of str, optional
        Additional covariate columns.
    observed_col : str, optional
        Optional boolean/0-1 column indicating whether outcome is observed.
    weights_col : str, optional
        Optional non-negative row weights.
    allow_missing_outcome : bool, default=True
        If ``False``, requires fully observed numeric outcomes.
    allow_duplicate_unit_time : bool, default=False
        If ``False``, requires unique ``(unit_col, time_col)`` after coercion.
    strict_observed_mask : bool, default=True
        If ``True``, requires observed mask to match outcome missingness exactly
        and forbids null values in ``observed_col``.
    allow_gapped_time_axis : bool, default=False
        If ``False``, requires contiguous analysis periods at ``time_freq``.

    Notes
    -----
    All time-like fields are normalized to ``pandas.Period`` at ``time_freq``.
    Timezone-aware datetimes are rejected; normalize to naive or UTC timestamps
    before constructing this contract.

    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        frozen=True,
    )

    # ---------- core data ----------
    df: pd.DataFrame = Field(..., description="Long-format panel data.")
    unit_col: str = Field(..., description="Unit identifier column in df.")
    time_col: str = Field(..., description="Calendar time column in df.")
    time_freq: str = Field(
        default="M",
        description="Regular panel frequency, e.g. 'D', 'W', 'M', 'Q', 'Y'.",
    )
    y: str = Field(..., description="Outcome column in df.")

    # ---------- adoption spec ----------
    treated_unit: Hashable = Field(..., description="ID of the treated unit.")
    treatment_start: TimeLike = Field(
        ...,
        description=(
            "First treated period (inclusive). "
            "Pre: t < treatment_start, Post: t >= treatment_start."
        ),
    )

    # ---------- optional selectors ----------
    donor_units: Optional[Sequence[Hashable]] = Field(
        default=None,
        description="Optional explicit donor pool. If None, all non-treated units are donors.",
    )

    time_window: Optional[Tuple[Optional[TimeLike], Optional[TimeLike]]] = Field(
        default=None,
        description="Inclusive analysis period window: (t_min, t_max). Use None for open ends.",
    )

    pre_periods: Optional[Sequence[TimeLike]] = Field(
        default=None,
        description="Optional explicit pre-treatment periods.",
    )
    post_periods: Optional[Sequence[TimeLike]] = Field(
        default=None,
        description="Optional explicit post-treatment periods.",
    )

    # ---------- optional extras ----------
    covariate_cols: Sequence[str] = Field(
        default_factory=tuple,
        description="Optional covariate columns.",
    )
    observed_col: Optional[str] = Field(
        default=None,
        description="Optional boolean/0-1 column indicating whether outcome is observed.",
    )
    weights_col: Optional[str] = Field(
        default=None,
        description="Optional non-negative row weights.",
    )

    # ---------- behavior flags ----------
    allow_missing_outcome: bool = Field(
        default=True,
        description="If False, requires y to be numeric and fully observed.",
    )
    allow_duplicate_unit_time: bool = Field(
        default=False,
        description="If False, requires uniqueness of (unit_col, time_col).",
    )
    strict_observed_mask: bool = Field(
        default=True,
        description="If True, observed_col must match y missingness exactly.",
    )
    allow_gapped_time_axis: bool = Field(
        default=False,
        description=(
            "If False, requires contiguous analysis periods at time_freq "
            "(no gaps between min and max analysis times)."
        ),
    )

    def _canonical_time_freq(self) -> str:
        """Validate and canonicalize ``time_freq``.

        Returns
        -------
        str
            Canonical pandas period frequency string.

        Raises
        ------
        ValueError
            If ``time_freq`` is empty or not a valid pandas period alias.
        """
        raw = str(self.time_freq).strip()
        if not raw:
            raise ValueError("time_freq must be a non-empty pandas frequency alias.")
        try:
            return pd.period_range(start="2000-01-01", periods=1, freq=raw).freqstr
        except Exception as exc:
            raise ValueError(
                f"Invalid time_freq={self.time_freq!r}. Use pandas aliases like 'D', 'W', 'M', 'Q', or 'Y'."
            ) from exc

    def _coerce_period_series(self, s: pd.Series, freq: str) -> pd.Series:
        """Normalize a time series to ``Period`` at the requested frequency.

        Parameters
        ----------
        s : pandas.Series
            Input time column.
        freq : str
            Canonical period frequency.

        Returns
        -------
        pandas.Series
            Series with period dtype at ``freq``.
        """
        if isinstance(s.dtype, pd.PeriodDtype):
            src_freq = pd.PeriodIndex(s).freqstr
            if src_freq != freq:
                raise ValueError(
                    f"{self.time_col!r} has Period frequency {src_freq!r}, expected {freq!r}."
                )
            return s
        if isinstance(s.dtype, pd.DatetimeTZDtype):
            raise ValueError(
                f"{self.time_col!r} contains timezone-aware datetimes; normalize to naive/UTC before contract construction."
            )
        dt = pd.to_datetime(s, errors="raise")
        if getattr(dt.dt, "tz", None) is not None:
            raise ValueError(
                f"{self.time_col!r} contains timezone-aware datetimes; normalize to naive/UTC before contract construction."
            )
        return dt.dt.to_period(freq)

    def _coerce_period_value(self, t: Optional[TimeLike], freq: str) -> Optional[pd.Period]:
        """Normalize a scalar to ``Period`` at the requested frequency.

        Parameters
        ----------
        t : TimeLike or None
            Time value to normalize.
        freq : str
            Canonical period frequency.

        Returns
        -------
        pandas.Period or None
            Coerced period, or ``None`` for open interval boundaries.
        """
        if t is None:
            return None
        if isinstance(t, pd.Period):
            if t.freqstr != freq:
                raise ValueError(
                    f"Period value {t!r} has frequency {t.freqstr!r}, expected {freq!r}."
                )
            return t
        if isinstance(t, pd.Timestamp) and t.tz is not None:
            raise ValueError(
                "Timezone-aware treatment/time-window/pre/post values are not supported; normalize to naive/UTC first."
            )
        ts = pd.to_datetime(pd.Series([t]), errors="raise").iloc[0]
        if isinstance(ts, pd.Timestamp) and ts.tz is not None:
            raise ValueError(
                "Timezone-aware treatment/time-window/pre/post values are not supported; normalize to naive/UTC first."
            )
        return ts.to_period(freq)

    def _coerce_period_sequence(
        self,
        values: Sequence[TimeLike],
        *,
        name: str,
        freq: str,
    ) -> list[pd.Period]:
        """Coerce and validate explicit period lists.

        Parameters
        ----------
        values : sequence of TimeLike
            User-provided period values.
        name : str
            Field name for validation messages.
        freq : str
            Canonical period frequency.

        Returns
        -------
        list of pandas.Period
            Coerced periods preserving user order.

        Raises
        ------
        ValueError
            If values are invalid, contain ``None``, or include duplicates.
        """
        try:
            coerced_raw = [self._coerce_period_value(v, freq) for v in values]
        except Exception as exc:
            raise ValueError(f"{name} contains invalid calendar periods.") from exc

        if any(v is None for v in coerced_raw):
            raise ValueError(f"{name} must not contain None.")

        coerced = [v for v in coerced_raw if v is not None]
        if len(set(coerced)) != len(coerced):
            raise ValueError(f"{name} must contain unique periods.")
        return coerced

    @model_validator(mode="after")
    def _validate_schema(self) -> "PanelDataSCM":
        """Validate schema and normalize estimator-facing state.

        Returns
        -------
        PanelDataSCM
            Validated instance with normalized period-based time fields.

        Raises
        ------
        ValueError
            If schema, typing, or panel consistency constraints are violated.
        """
        df = self.df
        if df.empty:
            raise ValueError("df must contain at least one row.")

        required = {self.unit_col, self.time_col, self.y}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        if df[self.unit_col].isna().any():
            raise ValueError(f"{self.unit_col!r} contains nulls.")
        if df[self.time_col].isna().any():
            raise ValueError(f"{self.time_col!r} contains nulls.")

        role_columns = [self.unit_col, self.time_col, self.y, *list(self.covariate_cols)]
        if self.observed_col is not None:
            role_columns.append(self.observed_col)
        if self.weights_col is not None:
            role_columns.append(self.weights_col)
        if len(set(role_columns)) != len(role_columns):
            raise ValueError(
                "Column role names must be distinct across unit_col, time_col, y, "
                "covariate_cols, observed_col, and weights_col."
            )

        if len(set(self.covariate_cols)) != len(self.covariate_cols):
            raise ValueError("covariate_cols must contain unique column names.")

        for col in self.covariate_cols:
            if col not in df.columns:
                raise ValueError(f"covariate_cols contains missing column: {col}")
        if self.observed_col is not None and self.observed_col not in df.columns:
            raise ValueError(f"observed_col not found: {self.observed_col}")
        if self.weights_col is not None and self.weights_col not in df.columns:
            raise ValueError(f"weights_col not found: {self.weights_col}")

        keep_columns = [self.unit_col, self.time_col, self.y]
        for col in self.covariate_cols:
            if col not in keep_columns:
                keep_columns.append(col)
        if self.observed_col is not None and self.observed_col not in keep_columns:
            keep_columns.append(self.observed_col)
        if self.weights_col is not None and self.weights_col not in keep_columns:
            keep_columns.append(self.weights_col)
        df = df.loc[:, keep_columns].copy()

        time_freq = self._canonical_time_freq()
        try:
            df[self.time_col] = self._coerce_period_series(df[self.time_col], time_freq)
            treatment_start = self._coerce_period_value(self.treatment_start, time_freq)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(
                f"Failed to coerce {self.time_col!r} / treatment_start to Period[{time_freq}]."
            ) from exc

        if treatment_start is None:
            raise ValueError("treatment_start must be a valid calendar time value.")

        # Duplicate checks happen after coercion because coarse frequencies can
        # collapse multiple timestamps into the same analysis period.
        if not self.allow_duplicate_unit_time:
            has_dup = df.duplicated([self.unit_col, self.time_col]).any()
            if has_dup:
                raise ValueError(
                    f"Duplicate (unit,time) rows found in [{self.unit_col}, {self.time_col}] "
                    f"after coercion to Period[{time_freq}]. Aggregate first or set "
                    "allow_duplicate_unit_time=True."
                )

        time_window = self.time_window
        if time_window is not None:
            try:
                t_min, t_max = time_window
                t_min = self._coerce_period_value(t_min, time_freq)
                t_max = self._coerce_period_value(t_max, time_freq)
            except Exception as exc:
                raise ValueError("time_window contains invalid calendar periods.") from exc
            if t_min is not None and t_max is not None and t_min > t_max:
                raise ValueError("time_window must satisfy t_min <= t_max.")
            time_window = (t_min, t_max)

        pre_periods = self.pre_periods
        if pre_periods is not None:
            pre_periods = self._coerce_period_sequence(pre_periods, name="pre_periods", freq=time_freq)
            if any(t >= treatment_start for t in pre_periods):
                raise ValueError("pre_periods must contain only periods < treatment_start.")

        post_periods = self.post_periods
        if post_periods is not None:
            post_periods = self._coerce_period_sequence(post_periods, name="post_periods", freq=time_freq)
            if any(t < treatment_start for t in post_periods):
                raise ValueError("post_periods must contain only periods >= treatment_start.")

        y_num = pd.to_numeric(df[self.y], errors="coerce")
        if not self.allow_missing_outcome and y_num.isna().any():
            raise ValueError(
                f"{self.y!r} must be numeric and non-missing when allow_missing_outcome=False."
            )
        if self.allow_missing_outcome:
            created_nan = y_num.isna() & ~df[self.y].isna()
            if created_nan.any():
                raise ValueError(f"{self.y!r} contains non-numeric values.")
        df[self.y] = y_num

        if self.observed_col is not None:
            obs = df[self.observed_col]
            allowed = {0, 1, True, False}
            if not set(obs.dropna().unique()).issubset(allowed):
                raise ValueError(f"{self.observed_col!r} must be boolean or 0/1.")
            obs_bool = obs.astype("boolean")
            df[self.observed_col] = obs_bool

            if self.strict_observed_mask:
                if obs_bool.isna().any():
                    raise ValueError(
                        f"{self.observed_col!r} contains nulls; strict_observed_mask=True "
                        "requires explicit True/False for every row."
                    )
                y_is_na = df[self.y].isna()
                mismatch = ((obs_bool == False) & (~y_is_na)) | ((obs_bool == True) & y_is_na)
                if mismatch.any():
                    raise ValueError(
                        "observed_col/outcome mismatch: "
                        "observed=False requires y missing, observed=True requires y present."
                    )

        if self.weights_col is not None:
            weights = pd.to_numeric(df[self.weights_col], errors="coerce")
            if weights.isna().any():
                raise ValueError(f"{self.weights_col!r} contains non-numeric values.")
            if (weights < 0).any():
                raise ValueError(f"{self.weights_col!r} must be non-negative.")
            df[self.weights_col] = weights

        units = pd.Index(df[self.unit_col].unique())
        if self.treated_unit not in set(units):
            raise ValueError(f"treated_unit={self.treated_unit!r} not found in {self.unit_col!r}.")

        donor_units = self.donor_units
        if donor_units is not None:
            donor_list = list(donor_units)
            donor_set = set(donor_list)
            if len(donor_set) != len(donor_list):
                raise ValueError("donor_units must contain unique unit ids.")
            if self.treated_unit in donor_set:
                raise ValueError("donor_units must not include treated_unit.")
            missing_donors = donor_set - set(units)
            if missing_donors:
                raise ValueError(f"donor_units contain unknown unit ids: {sorted(missing_donors)}")
            if len(donor_set) < 2:
                raise ValueError("donor_units must contain at least 2 unique units.")
            donor_units = tuple(donor_list)
        elif len(units) < 3:
            raise ValueError("Need at least 2 donor units.")

        if pre_periods is not None and post_periods is not None:
            pre_set = set(pre_periods)
            post_set = set(post_periods)
            if pre_set & post_set:
                raise ValueError("pre_periods and post_periods must be disjoint.")
            if pre_periods and post_periods and max(pre_periods) >= min(post_periods):
                raise ValueError("Expected all pre_periods < all post_periods.")

        # Persist normalized state before helper-based checks.
        object.__setattr__(self, "df", df)
        object.__setattr__(self, "time_freq", time_freq)
        object.__setattr__(self, "treatment_start", treatment_start)
        object.__setattr__(self, "time_window", time_window)
        object.__setattr__(self, "pre_periods", tuple(pre_periods) if pre_periods is not None else None)
        object.__setattr__(self, "post_periods", tuple(post_periods) if post_periods is not None else None)
        object.__setattr__(self, "donor_units", donor_units)

        analysis_df = self.df_analysis()
        if analysis_df.empty:
            raise ValueError("No rows remain after donor/time filtering.")

        donor_units_in_analysis = sorted(
            set(analysis_df.loc[analysis_df[self.unit_col] != self.treated_unit, self.unit_col].tolist())
        )
        if len(donor_units_in_analysis) < 2:
            raise ValueError(
                "Need at least 2 donors with rows in analysis data after donor/time filtering."
            )

        if self.pre_periods is not None or self.post_periods is not None:
            # Coverage is checked against filtered analysis data to prevent
            # explicit periods from silently pointing outside the analysis window.
            available_times = set(pd.Index(analysis_df[self.time_col].unique()).tolist())

            if self.pre_periods is not None:
                missing_pre = set(self.pre_periods) - available_times
                if missing_pre:
                    raise ValueError(
                        f"pre_periods contain periods not present in analysis data: {sorted(missing_pre)}"
                    )

            if self.post_periods is not None:
                missing_post = set(self.post_periods) - available_times
                if missing_post:
                    raise ValueError(
                        f"post_periods contain periods not present in analysis data: {sorted(missing_post)}"
                    )

        pre_times = list(self.pre_times())
        post_times = list(self.post_times())
        if not pre_times:
            raise ValueError("No pre-treatment periods available after filters.")
        if not post_times:
            raise ValueError("No post-treatment periods available after filters.")

        donor_pre_rows = analysis_df[
            (analysis_df[self.unit_col].isin(donor_units_in_analysis))
            & (analysis_df[self.time_col].isin(pre_times))
        ]
        donor_pre_units = set(donor_pre_rows[self.unit_col].unique().tolist())
        donors_without_pre = sorted(set(donor_units_in_analysis) - donor_pre_units)
        if donors_without_pre:
            raise ValueError(
                "Each donor must have at least one pre-treatment row in analysis data. "
                f"Donors without pre rows: {donors_without_pre}"
            )

        if not self.allow_gapped_time_axis:
            times = list(self.analysis_times())
            expected = list(pd.period_range(start=min(times), end=max(times), freq=self.time_freq))
            if times != expected:
                raise ValueError(
                    "Analysis time axis has gaps relative to time_freq. "
                    "Fill missing periods, restrict time_window, or set allow_gapped_time_axis=True."
                )

        treated_post = analysis_df[
            (analysis_df[self.unit_col] == self.treated_unit)
            & (analysis_df[self.time_col].isin(post_times))
        ]

        observed_post_times = set(pd.Index(treated_post[self.time_col].unique()).tolist())
        missing_post_times = sorted(set(post_times) - observed_post_times)

        unobserved_times = set()

        if self.observed_col is not None and not treated_post.empty:
            obs_mask = treated_post[self.observed_col] == True
            unobserved_times.update(
                pd.Index(treated_post.loc[~obs_mask.fillna(False), self.time_col].unique()).tolist()
            )

        if not treated_post.empty:
            missing_y = treated_post[self.y].isna()
            unobserved_times.update(
                pd.Index(treated_post.loc[missing_y, self.time_col].unique()).tolist()
            )

        if missing_post_times or unobserved_times:
            bad_times = sorted(set(missing_post_times) | set(unobserved_times))
            raise ValueError(
                "treated_unit must have observed y in all post-treatment periods. "
                f"Missing/unobserved treated post periods: {bad_times}"
            )

        return self

    def donor_pool(self) -> Sequence[Hashable]:
        """Return donor units used in analysis.

        Returns
        -------
        sequence of Hashable
            Explicit donor pool if provided, otherwise all non-treated units.
        """
        if self.donor_units is not None:
            return list(self.donor_units)

        units = pd.Index(self.df[self.unit_col].unique())
        return [u for u in units.tolist() if u != self.treated_unit]

    def _apply_time_window(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply inclusive time-window filtering.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe that includes ``time_col``.

        Returns
        -------
        pandas.DataFrame
            Filtered dataframe. If no ``time_window`` is set, input is returned.
        """
        if self.time_window is None:
            return df

        t_min, t_max = self.time_window
        out = df
        if t_min is not None:
            out = out[out[self.time_col] >= t_min]
        if t_max is not None:
            out = out[out[self.time_col] <= t_max]
        return out

    def df_analysis(self) -> pd.DataFrame:
        """Build the estimator-facing analysis dataframe.

        Returns
        -------
        pandas.DataFrame
            Data restricted to treated plus donor units and filtered by time window.
        """
        keep_units = set(self.donor_pool()) | {self.treated_unit}
        out = self.df[self.df[self.unit_col].isin(keep_units)].copy()
        return self._apply_time_window(out)

    def pre_times(self) -> Sequence[pd.Period]:
        """Return pre-treatment periods used by estimators.

        Returns
        -------
        sequence of pandas.Period
            Explicit ``pre_periods`` when provided, else inferred periods with
            ``t < treatment_start``.
        """
        if self.pre_periods is not None:
            return list(self.pre_periods)

        times = pd.Index(self.df_analysis()[self.time_col].unique()).tolist()
        return sorted([t for t in times if t < self.treatment_start])

    def post_times(self) -> Sequence[pd.Period]:
        """Return post-treatment periods used by estimators.

        Returns
        -------
        sequence of pandas.Period
            Explicit ``post_periods`` when provided, else inferred periods with
            ``t >= treatment_start``.
        """
        if self.post_periods is not None:
            return list(self.post_periods)

        times = pd.Index(self.df_analysis()[self.time_col].unique()).tolist()
        return sorted([t for t in times if t >= self.treatment_start])

    def analysis_times(self) -> Sequence[pd.Period]:
        """Return sorted unique time axis of analysis data.

        Returns
        -------
        sequence of pandas.Period
            Sorted analysis periods.
        """
        times = pd.Index(self.df_analysis()[self.time_col].unique()).tolist()
        return sorted(times)

    def time_to_index(self) -> dict[pd.Period, int]:
        """Build dense integer mapping for matrix estimators.

        Returns
        -------
        dict of pandas.Period to int
            Mapping from analysis period to zero-based integer index.
        """
        times = self.analysis_times()
        return {t: i for i, t in enumerate(times)}

    def treatment_start_idx(self) -> int:
        """Return index position of treatment start on analysis time axis.

        Returns
        -------
        int
            Zero-based index corresponding to ``treatment_start``.

        Raises
        ------
        ValueError
            If ``treatment_start`` is outside the analysis time axis.
        """
        mapping = self.time_to_index()
        if self.treatment_start not in mapping:
            raise ValueError("treatment_start is not present in analysis time axis.")
        return mapping[self.treatment_start]

    def __repr__(self) -> str:
        """Return concise debug representation."""
        donor_units = list(self.donor_pool())
        res = (
            f"{self.__class__.__name__}(df={self.df.shape}, "
            f"unit_col={self.unit_col!r}, "
            f"time_col={self.time_col!r}, "
            f"time_freq={self.time_freq!r}, "
            f"y={self.y!r}, "
            f"treated_unit={self.treated_unit!r}, "
            f"treatment_start={self.treatment_start!r}, "
            f"donor_units={donor_units!r}"
        )
        if self.time_window is not None:
            res += f", time_window={self.time_window!r}"
        if self.pre_periods is not None:
            res += f", pre_periods={list(self.pre_periods)!r}"
        if self.post_periods is not None:
            res += f", post_periods={list(self.post_periods)!r}"
        if self.covariate_cols:
            res += f", covariate_cols={list(self.covariate_cols)!r}"
        if self.observed_col is not None:
            res += f", observed_col={self.observed_col!r}"
        if self.weights_col is not None:
            res += f", weights_col={self.weights_col!r}"
        res += ")"
        return res
