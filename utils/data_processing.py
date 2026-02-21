"""
Data processing utilities.
Handles data transformations, feature engineering, and sanitization.

Day 1 additions:
- filter_by_date_range(): restrict dataset to a fixed analysis window (A -> B)
- build_hourly_report_count(): create hourly time series with missing hours filled as 0
"""
import polars as pl
import re
from config import MAX_DURATION_HOURS


def add_time_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add time-based features to the DataFrame.

    Args:
        df: Input DataFrame with started_at and completed_at columns

    Returns:
        DataFrame with additional time features (date, hour, weekday)
    """
    return df.with_columns([
        pl.col("started_at").cast(pl.Datetime, strict=False),
        pl.col("completed_at").cast(pl.Datetime, strict=False),
        pl.col("started_at").dt.date().alias("date"),
        pl.col("started_at").dt.hour().alias("hour"),
        pl.col("started_at").dt.weekday().alias("weekday"),  # 0=Monday ... 6=Sunday
    ])


def quality_sanitize(df: pl.DataFrame) -> pl.DataFrame:
    """
    Sanitize data by filtering out invalid or suspicious records.

    Filters:
    - Non-null duration_seconds
    - Positive duration
    - Duration less than MAX_DURATION_HOURS

    Args:
        df: Input DataFrame

    Returns:
        Sanitized DataFrame
    """
    df2 = df.with_columns([
        pl.col("duration_seconds").cast(pl.Int64, strict=False),
        pl.col("started_at").cast(pl.Datetime, strict=False),
        pl.col("completed_at").cast(pl.Datetime, strict=False),
    ])

    max_seconds = 60 * 60 * MAX_DURATION_HOURS
    df2 = df2.filter(
        pl.col("duration_seconds").is_not_null() &
        (pl.col("duration_seconds") > 0) &
        (pl.col("duration_seconds") < max_seconds) &
        pl.col("started_at").is_not_null()
    )
    return df2


# ----------------------------
# Day 1 helpers
# ----------------------------

def filter_by_date_range(df: pl.DataFrame, start_date: str, end_date: str) -> pl.DataFrame:
    """
    Restrict dataset to a fixed analysis window [start_date, end_date] using Date (not Datetime).
    This avoids timezone / parsing issues.

    Args:
        df: Input DataFrame (must have 'date' or 'started_at')
        start_date: "YYYY-MM-DD"
        end_date: "YYYY-MM-DD"
    """
    # Ensure 'date' exists; if not, create it from started_at
    if "date" not in df.columns:
        df = df.with_columns(
            pl.col("started_at").cast(pl.Datetime, strict=False),
            pl.col("started_at").dt.date().alias("date")
        )

    start_d = pl.lit(start_date).str.strptime(pl.Date, strict=False)
    end_d = pl.lit(end_date).str.strptime(pl.Date, strict=False)

    return df.filter(
        (pl.col("date") >= start_d) &
        (pl.col("date") <= end_d)
    )



def build_hourly_report_count(df: pl.DataFrame) -> pl.DataFrame:
    """
    Build an hourly time series: one row per hour with count of reports.
    Missing hours are filled with 0.

    Output columns:
      - timestamp_hour (Datetime)
      - report_count (Int64)
    """
    df2 = df.with_columns(pl.col("started_at").cast(pl.Datetime, strict=False))

    hourly = (
        df2.with_columns(
            pl.col("started_at").dt.truncate("1h").alias("timestamp_hour")
        )
        .group_by("timestamp_hour")
        .agg(pl.len().alias("report_count"))
        .sort("timestamp_hour")
    )

    if hourly.height == 0:
        # No data in range/filters; return empty with correct schema
        return pl.DataFrame(
            {
                "timestamp_hour": pl.Series([], dtype=pl.Datetime),
                "report_count": pl.Series([], dtype=pl.Int64),
            }
        )

    min_ts = hourly.select(pl.col("timestamp_hour").min()).item()
    max_ts = hourly.select(pl.col("timestamp_hour").max()).item()

    full_hours = pl.datetime_range(
        start=min_ts,
        end=max_ts,
        interval="1h",
        eager=True,
    ).alias("timestamp_hour")

    full_df = pl.DataFrame({"timestamp_hour": full_hours})

    hourly_filled = (
        full_df.join(hourly, on="timestamp_hour", how="left")
        .with_columns(pl.col("report_count").fill_null(0).cast(pl.Int64))
        .sort("timestamp_hour")
    )

    return hourly_filled


def enrich_hourly_ts_with_features(hourly_ts: pl.DataFrame, df_raw: pl.DataFrame) -> pl.DataFrame:
    """
    Enrich hourly time series with operational and seasonal features for modeling.
    Optimized version using pure Polars event-based logic for backlog calculation.
    """
    # Use temporary names for internal logic that matches trainer.py
    df = hourly_ts.rename({'timestamp_hour': 'ds', 'report_count': 'y'})
    
    # 1. Time-based features
    df = df.with_columns([
        pl.col("ds").dt.hour().alias("hour"),
        pl.col("ds").dt.weekday().alias("weekday"),
        pl.col("ds").dt.day().alias("day_of_month"),
        pl.col("ds").dt.month().alias("month"),
        (pl.col("ds").dt.weekday() >= 5).cast(pl.Int64).alias("is_weekend"),
        ((pl.col("ds").dt.hour() >= 8) & (pl.col("ds").dt.hour() <= 18)).cast(pl.Int64).alias("is_peak_hour")
    ])
    
    # 2. Argentina Holidays
    import holidays
    ar_holidays = holidays.CountryHoliday('AR')
    df = df.with_columns(
        pl.col("ds").map_elements(lambda x: 1 if x.date() in ar_holidays else 0, return_dtype=pl.Int64).alias("is_holiday")
    )
    
    # is_month_end and interactions
    df = df.with_columns([
        (pl.col("ds").dt.month() != pl.col("ds").dt.offset_by("1d").dt.month()).cast(pl.Int64).alias("is_month_end"),
        (pl.col("is_weekend") * pl.col("is_peak_hour")).alias("weekend_x_peak"),
        (pl.col("is_holiday") * pl.col("is_peak_hour")).alias("holiday_x_peak"),
        (((pl.col("is_weekend") == 1) | (pl.col("is_holiday") == 1)) & (pl.col("is_peak_hour") == 1)).cast(pl.Int64).alias("weekend_or_holiday_x_peak")
    ])

    # 3. Cyclical encoding
    import numpy as np
    df = df.with_columns([
        (pl.col("hour") * 2 * np.pi / 24).sin().alias("hour_sin"),
        (pl.col("hour") * 2 * np.pi / 24).cos().alias("hour_cos"),
        (pl.col("weekday") * 2 * np.pi / 7).sin().alias("weekday_sin"),
        (pl.col("weekday") * 2 * np.pi / 7).cos().alias("weekday_cos"),
        (pl.col("month") * 2 * np.pi / 12).sin().alias("month_sin"),
        (pl.col("month") * 2 * np.pi / 12).cos().alias("month_cos"),
    ])

    # 4. Fourier Features
    # Create a global index for t
    df = df.with_row_index("t")
    for name, period in {"daily": 24, "weekly": 24 * 7}.items():
        for i in range(1, 4):
            df = df.with_columns([
                (pl.col("t") * 2 * np.pi * i / period).sin().alias(f"fourier_{name}_sin_{i}"),
                (pl.col("t") * 2 * np.pi * i / period).cos().alias(f"fourier_{name}_cos_{i}")
            ])
    df = df.drop("t")

    # 5. Memory Features (Lags & Rolling)
    df = df.with_columns([
        pl.col("y").shift(1).fill_null(0).alias("y_lag_1"),
        pl.col("y").shift(2).fill_null(0).alias("y_lag_2"),
        pl.col("y").shift(3).fill_null(0).alias("y_lag_3"),
        pl.col("y").shift(24).fill_null(0).alias("y_lag_24"),
        pl.col("y").shift(168).fill_null(0).alias("y_lag_168"),
    ])
    
    # Rolling stats (on shifted y to avoid leakage)
    df = df.with_columns([
        pl.col("y_lag_1").rolling_mean(window_size=3).fill_null(0).alias("y_rolling_mean_3"),
        pl.col("y_lag_1").rolling_mean(window_size=24).fill_null(0).alias("y_rolling_mean_24"),
        pl.col("y_lag_1").rolling_mean(window_size=168).fill_null(0).alias("y_rolling_mean_168"),
        pl.col("y_lag_1").rolling_std(window_size=3).fill_null(0).alias("y_rolling_std_3"),
        pl.col("y_lag_1").rolling_std(window_size=24).fill_null(0).alias("y_rolling_std_24"),
    ])

    # Log-scale lags (to match trainer.py's log_ features)
    for col in ["y_lag_1", "y_lag_24", "y_lag_168", "y_rolling_mean_3", "y_rolling_mean_24", "y_rolling_mean_168"]:
        df = df.with_columns(
            pl.col(col).clip(0, None).log1p().alias(f"log_{col}")
        )
    
    # 6. Optimized Backlog Calculation
    # We define an event: +1 for creation, -1 for start processing
    # Note: Using existing names in df_raw if they differ. 
    # trainer.py uses 'CreationDate' and 'StartedProcessingOn'
    # Current utils usage expects 'creation_date' and 'started_at'
    creations_col = "CreationDate" if "CreationDate" in df_raw.columns else "creation_date"
    starts_col = "StartedProcessingOn" if "StartedProcessingOn" in df_raw.columns else "started_at"
    
    creations = df_raw.select(pl.col(creations_col).alias("time"), pl.lit(1).alias("change"))
    starts = df_raw.select(pl.col(starts_col).alias("time"), pl.lit(-1).alias("change"))
    
    events = pl.concat([creations, starts]).sort("time")
    backlog_history = events.with_columns(
        pl.col("change").cum_sum().alias("backlog")
    ).select(["time", "backlog"])
    
    # Map backlog to hourly timestamps using backward asof join
    df = df.join_asof(
        backlog_history,
        left_on="ds",
        right_on="time",
        strategy="backward"
    ).fill_null(0).drop("time")
    
    # 7. Report Type mapping (factorize equivalent)
    # trainer.py factorizes main_report_type. 
    # Here we might need to handle it or leave as 0 if not available
    df = df.with_columns(pl.lit(0).alias("tipo_reporte_id"))
    
    # 8. Parameter date-span feature
    if "avg_param_span_days" not in df.columns:
        param_spans = extract_param_date_span(df_raw)
        if param_spans.height > 0:
            df = df.join_asof(
                param_spans.sort("timestamp_hour"),
                left_on="ds",
                right_on="timestamp_hour",
                strategy="backward"
            ).with_columns(
                pl.col("avg_param_span_days").fill_null(0)
            )
        else:
            df = df.with_columns(pl.lit(0.0).alias("avg_param_span_days"))

    # Return with original column names
    return df.rename({'ds': 'timestamp_hour', 'y': 'report_count'})


def extract_param_date_span(df_raw: pl.DataFrame) -> pl.DataFrame:
    """
    Parse the Parameters column to extract start/end date spans.
    Handles XML, JSON, and query-string formats via regex.

    Returns a DataFrame with columns:
      - timestamp_hour (Datetime): truncated to the hour of started_at
      - avg_param_span_days (Float64): mean date-range span for that hour
    """
    if "Parameters" not in df_raw.columns or "started_at" not in df_raw.columns:
        return pl.DataFrame({
            "timestamp_hour": pl.Series([], dtype=pl.Datetime),
            "avg_param_span_days": pl.Series([], dtype=pl.Float64),
        })

    # Regex that captures date-like values after common key names
    # Handles: XML <StartDate>2024-01-01</StartDate>
    #          JSON "startDate":"2024-01-01"
    #          QS   startDate=2024-01-01
    DATE_RE = re.compile(
        r'(?:start(?:date|Date|_date)|fecha(?:inicio|Inicio))[":\s=>]*'
        r'([\"\']?)'
        r'(\d{4}-\d{2}-\d{2})'
        r'["\'>]?',
        re.IGNORECASE,
    )
    END_RE = re.compile(
        r'(?:end(?:date|Date|_date)|fecha(?:fin|Fin))[":\s=>]*'
        r'(["\']?)'
        r'(\d{4}-\d{2}-\d{2})'
        r'["\'>]?',
        re.IGNORECASE,
    )

    import pandas as pd

    params_pd = df_raw.select([
        pl.col("started_at").cast(pl.Datetime, strict=False),
        pl.col("Parameters").cast(pl.Utf8, strict=False),
    ]).to_pandas()

    spans = []
    for _, row in params_pd.iterrows():
        raw = row["Parameters"]
        if not raw or not isinstance(raw, str):
            continue
        sm = DATE_RE.search(raw)
        em = END_RE.search(raw)
        if sm and em:
            try:
                start_d = pd.to_datetime(sm.group(2))
                end_d   = pd.to_datetime(em.group(2))
                span    = (end_d - start_d).days
                if 0 <= span <= 3650:  # sanity: 0 to 10 years
                    spans.append({"started_at": row["started_at"], "span_days": float(span)})
            except Exception:
                pass

    if not spans:
        return pl.DataFrame({
            "timestamp_hour": pl.Series([], dtype=pl.Datetime),
            "avg_param_span_days": pl.Series([], dtype=pl.Float64),
        })

    spans_df = pl.DataFrame(spans).with_columns(
        pl.col("started_at").cast(pl.Datetime, strict=False).dt.truncate("1h").alias("timestamp_hour")
    )

    return (
        spans_df
        .group_by("timestamp_hour")
        .agg(pl.col("span_days").mean().alias("avg_param_span_days"))
        .sort("timestamp_hour")
    )

