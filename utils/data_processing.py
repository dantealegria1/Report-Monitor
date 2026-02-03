"""
Data processing utilities.
Handles data transformations, feature engineering, and sanitization.
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
        pl.col("started_at").cast(pl.Datetime),
        pl.col("completed_at").cast(pl.Datetime),
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
        pl.col("duration_seconds").cast(pl.Int64, strict=False)
    ])
    
    # Filter impossible/suspicious records
    max_seconds = 60 * 60 * MAX_DURATION_HOURS
    df2 = df2.filter(
        pl.col("duration_seconds").is_not_null() &
        (pl.col("duration_seconds") > 0) &
        (pl.col("duration_seconds") < max_seconds)
    )
    return df2
