"""
Data processing utilities.
Handles data transformations, feature engineering, and sanitization.

Day 1 additions:
- filter_by_date_range(): restrict dataset to a fixed analysis window (A -> B)
- build_hourly_report_count(): create hourly time series with missing hours filled as 0
"""
import polars as pl
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
