"""
Chart components.
Handles all visualization and chart rendering.
"""
import streamlit as st
import polars as pl


def render_execution_trends(df_filtered: pl.DataFrame):
    """
    Render execution trends chart by status.
    
    Args:
        df_filtered: Filtered DataFrame
    """
    st.subheader("Execution Trends (by status)")
    evol = df_filtered.group_by(["date", "execution_status"]).agg(
        pl.len().alias("count")
    ).to_pandas()
    
    if not evol.empty:
        piv = evol.pivot(index="date", columns="execution_status", values="count").fillna(0)
        st.line_chart(piv)


def render_slowest_reports(df_filtered: pl.DataFrame):
    """
    Render top 10 slowest reports chart.
    
    Args:
        df_filtered: Filtered DataFrame
    """
    st.subheader("Top 10 Slowest Reports (success average)")
    df_success = df_filtered.filter(pl.col("execution_status") == "success")
    
    slow = (
        df_success.group_by("ReportName")
        .agg(pl.col("duration_seconds").mean().alias("average"))
        .sort("average", descending=True)
        .head(10)
        .to_pandas()
    )
    
    if not slow.empty:
        st.bar_chart(data=slow, x="ReportName", y="average")
    else:
        st.info("No successful data for the range.")


def render_top_failures(df_filtered: pl.DataFrame):
    """
    Render top 10 reports with most failures.
    
    Args:
        df_filtered: Filtered DataFrame
    """
    st.subheader("Top 10 Reports with Most Failures")
    df_failed = df_filtered.filter(pl.col("execution_status") == "failed")
    
    top_fail = (
        df_failed.group_by("ReportName")
        .agg(pl.len().alias("failures"))
        .sort("failures", descending=True)
        .head(10)
        .to_pandas()
    )
    
    if not top_fail.empty:
        st.bar_chart(data=top_fail, x="ReportName", y="failures")
    else:
        st.info("No failures in the range.")


def render_duration_distribution(df_filtered: pl.DataFrame, key: str = "slider_bin_size"):
    """
    Render duration distribution histogram.
    
    Args:
        df_filtered: Filtered DataFrame
        key: Unique key for the slider widget
    """
    st.subheader("Duration Distribution (success) - approximate bins")
    df_success = df_filtered.filter(pl.col("execution_status") == "success")
    
    if df_success.height > 0:
        bin_size = st.slider("Bin Size (sec)", 1, 120, 10, key=key)
        hist = (
            df_success
            .with_columns(((pl.col("duration_seconds") / bin_size).floor() * bin_size).alias("bin"))
            .group_by("bin")
            .agg(pl.len().alias("count"))
            .sort("bin")
            .to_pandas()
        )
        st.bar_chart(hist, x="bin", y="count")
    else:
        st.info("No successful data for histogram.")


def render_hourly_load(df_filtered: pl.DataFrame):
    """
    Render hourly load saturation chart.
    
    Args:
        df_filtered: Filtered DataFrame
    """
    st.subheader("Hourly Load (count)")
    hourly_load = (
        df_filtered.group_by("hour")
        .agg(pl.len().alias("count"))
        .sort("hour")
        .to_pandas()
    )
    
    if not hourly_load.empty:
        st.area_chart(data=hourly_load, x="hour", y="count")


def render_heatmap(df_filtered: pl.DataFrame):
    """
    Render hour vs weekday heatmap.
    
    Args:
        df_filtered: Filtered DataFrame
    """
    st.subheader("Heatmap Hour vs Weekday (count)")
    heat = (
        df_filtered.group_by(["weekday", "hour"])
        .agg(pl.len().alias("count"))
        .to_pandas()
    )
    
    if not heat.empty:
        heat_piv = heat.pivot(index="weekday", columns="hour", values="count").fillna(0)
        st.dataframe(heat_piv, use_container_width=True)
