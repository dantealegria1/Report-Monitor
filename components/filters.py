"""
Sidebar filters component.
Handles all filtering UI and logic.
"""
import streamlit as st
import polars as pl
from config import TOP_REPORTS_LIMIT


def render_sidebar_filters(df: pl.DataFrame) -> dict:
    """
    Render sidebar filters and return filter selections.
    
    Args:
        df: Input DataFrame to extract filter options from
        
    Returns:
        Dictionary containing filter selections:
        - date_range: tuple of (start_date, end_date)
        - report_types: list of selected report types
        - statuses: list of selected statuses
        - report_names: list of selected report names
    """
    st.sidebar.title("Analysis Filters")
    
    # Date filter
    min_date = df["date"].min()
    max_date = df["date"].max()
    date_selection = st.sidebar.date_input("Date Range", [min_date, max_date])
    
    # Report type filter
    report_types = sorted(df["ReportType"].unique().to_list())
    selected_types = st.sidebar.multiselect(
        "Report Types",
        report_types,
        default=report_types
    )
    
    # Status filter
    status_list = ["success", "failed", "cancelled"]
    selected_status = st.sidebar.multiselect(
        "Status",
        status_list,
        default=status_list
    )
    
    # Report name filter (top reports by volume)
    top_names = (
        df.group_by("ReportName")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .head(TOP_REPORTS_LIMIT)["ReportName"]
        .to_list()
    )
    selected_names = st.sidebar.multiselect(
        f"Reports (Top {TOP_REPORTS_LIMIT} by volume)",
        top_names,
        default=[]
    )
    
    return {
        "date_range": date_selection,
        "report_types": selected_types,
        "statuses": selected_status,
        "report_names": selected_names
    }


def apply_filters(df: pl.DataFrame, filters: dict) -> pl.DataFrame:
    """
    Apply filters to the DataFrame.
    
    Args:
        df: Input DataFrame
        filters: Dictionary of filter selections from render_sidebar_filters
        
    Returns:
        Filtered DataFrame
    """
    mask = pl.lit(True)
    
    # Date range filter
    if len(filters["date_range"]) == 2:
        start_date, end_date = filters["date_range"]
        mask = mask & (pl.col("date") >= start_date) & (pl.col("date") <= end_date)
    
    # Report type filter
    if filters["report_types"]:
        mask = mask & pl.col("ReportType").is_in(filters["report_types"])
    
    # Status filter
    if filters["statuses"]:
        mask = mask & pl.col("execution_status").is_in(filters["statuses"])
    
    # Report name filter
    if filters["report_names"]:
        mask = mask & pl.col("ReportName").is_in(filters["report_names"])
    
    return df.filter(mask)
