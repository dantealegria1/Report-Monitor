"""
Statistical utilities for anomaly detection and drift analysis.
"""
import polars as pl
import pandas as pd


def robust_z_mad(expr_x: pl.Expr, expr_median: pl.Expr, expr_mad: pl.Expr) -> pl.Expr:
    """
    Calculate robust Z-score using Median Absolute Deviation (MAD).
    
    Formula: 0.6745 * (x - median) / MAD
    
    Args:
        expr_x: Expression for the value
        expr_median: Expression for the median
        expr_mad: Expression for the MAD
        
    Returns:
        Polars expression for robust Z-score
    """
    return (pl.lit(0.6745) * (expr_x - expr_median) / (expr_mad + 1e-9))


def ks_distance_approx(
    df: pl.DataFrame,
    col: str,
    group_cols: list[str],
    window_a_days: int = 7,
    window_b_days: int = 30
) -> pl.DataFrame:
    """
    Simplified drift detection using quantile comparison.
    
    Compares quantiles between two time windows:
    - Window A: Last window_a_days days
    - Window B: Previous window_b_days days before window A
    
    Args:
        df: Input DataFrame with a 'date' column
        col: Column name to analyze
        group_cols: Columns to group by
        window_a_days: Size of recent window in days
        window_b_days: Size of comparison window in days
        
    Returns:
        DataFrame with quantile deltas by group
    """
    max_d = df["date"].max()
    if max_d is None:
        return pl.DataFrame()
    
    a_start = max_d - pd.Timedelta(days=window_a_days - 1)
    b_end = a_start - pd.Timedelta(days=1)
    b_start = b_end - pd.Timedelta(days=window_b_days - 1)
    
    df_a = df.filter((pl.col("date") >= a_start) & (pl.col("date") <= max_d))
    df_b = df.filter((pl.col("date") >= b_start) & (pl.col("date") <= b_end))
    
    if df_a.height < 10 or df_b.height < 10:
        return pl.DataFrame()
    
    qa = df_a.group_by(group_cols).agg([
        pl.col(col).quantile(0.50).alias("A_p50"),
        pl.col(col).quantile(0.90).alias("A_p90"),
        pl.len().alias("A_n"),
    ])
    qb = df_b.group_by(group_cols).agg([
        pl.col(col).quantile(0.50).alias("B_p50"),
        pl.col(col).quantile(0.90).alias("B_p90"),
        pl.len().alias("B_n"),
    ])
    
    out = qa.join(qb, on=group_cols, how="inner").with_columns([
        (pl.col("A_p50") - pl.col("B_p50")).alias("delta_p50"),
        (pl.col("A_p90") - pl.col("B_p90")).alias("delta_p90"),
        (pl.col("A_p90") / (pl.col("B_p90") + 1e-9)).alias("ratio_p90"),
    ]).sort("ratio_p90", descending=True)
    
    return out
