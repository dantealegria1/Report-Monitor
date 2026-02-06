"""
Time Series Analysis Page
Displays hourly report count time series for forecasting analysis.
"""
import streamlit as st
import polars as pl

# Configuration and validation
from config import validate_config

# Database operations
from db.database import load_reports_data

# Data processing utilities
from utils.data_processing import (
    add_time_features, quality_sanitize,
    filter_by_date_range, build_hourly_report_count
)

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(page_title="Time Series Analysis", layout="wide", page_icon="📊")

# ----------------------------
# Main Content
# ----------------------------
try:
    # Validate configuration
    validate_config()
    
    # Load and process data
    df_raw = load_reports_data()
    df = add_time_features(df_raw)
    df = quality_sanitize(df)
    
    # Filter to analysis window
    df = filter_by_date_range(df, "2024-01-01", "2026-12-31")
    
    # Create hourly time series
    hourly_ts = build_hourly_report_count(df)
    
    # ----------------------------
    # Display Time Series Data
    # ----------------------------
    st.title("Time Series Analysis")
    st.markdown("""
    This page displays the hourly time series of executed reports. 
    The data is prepared for forecasting analysis and anomaly detection.
    """)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Hours", f"{hourly_ts.height:,}")
    
    with col2:
        if hourly_ts.height > 0:
            avg_count = hourly_ts.select(pl.col("report_count").mean()).item()
            st.metric("Average Reports/hr", f"{avg_count:.2f}")
            
    with col3:
        if hourly_ts.height > 0:
            max_count = hourly_ts.select(pl.col("report_count").max()).item()
            st.metric("Peak Reports/hr", f"{max_count:,}")
            
    with col4:
        if hourly_ts.height > 0:
            max_date = hourly_ts.select(pl.col("timestamp_hour").max()).item()
            st.metric("Latest Data", max_date.strftime("%Y-%m-%d %H:%M"))
    
    st.divider()
    
    # Main Time Series Chart
    st.subheader("Report Count Over Time")
    
    days_to_show = st.slider("Visualization Window (days)", min_value=1, max_value=60, value=14)
    hours_to_show = int(days_to_show * 24)
    
    plot_data = hourly_ts.tail(hours_to_show).with_columns(
        pl.col("timestamp_hour").dt.strftime("%Y-%m-%dT%H:%M:%S").alias("timestamp_str")
    )
    
    st.vega_lite_chart(
        plot_data.to_dicts(),
        {
            "mark": {"type": "area", "color": "#1f77b4", "opacity": 0.6},
            "encoding": {
                "x": {"field": "timestamp_str", "type": "temporal", "title": "Time"},
                "y": {"field": "report_count", "type": "quantitative", "title": "Report Count"},
                "tooltip": [
                    {"field": "timestamp_str", "type": "temporal", "title": "Time"},
                    {"field": "report_count", "type": "quantitative", "title": "Count"}
                ]
            },
            "width": "container",
            "height": 400
        },
        use_container_width=True
    )
    
    st.divider()
    
    # Detailed Data View
    st.subheader("Data Inspector")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        num_rows = st.number_input(
            "Rows to show",
            min_value=10,
            max_value=1000,
            value=48,
            step=10
        )
    
    st.dataframe(
        hourly_ts.head(num_rows),
        use_container_width=True,
        height=300
    )
    
    # Statistics Section
    with st.expander("Descriptive Statistics"):
        if hourly_ts.height > 0:
            stats = hourly_ts.select([
                pl.col("report_count").mean().alias("Mean"),
                pl.col("report_count").median().alias("Median"),
                pl.col("report_count").std().alias("Std Dev"),
                pl.col("report_count").min().alias("Min"),
                pl.col("report_count").max().alias("Max"),
                pl.col("report_count").quantile(0.25).alias("Q1 (25%)"),
                pl.col("report_count").quantile(0.75).alias("Q3 (75%)"),
            ])
            st.dataframe(stats, use_container_width=True)
    
    # Download option
    st.divider()
    csv_data = hourly_ts.write_csv()
    st.download_button(
        label="Download CSV Data",
        data=csv_data,
        file_name="hourly_time_series.csv",
        mime="text/csv"
    )

except Exception as e:
    st.error(f"Error en la aplicación: {e}")
    st.exception(e)
