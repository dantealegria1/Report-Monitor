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

# Inject Custom CSS
try:
    with open("assets/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

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
    
    # Summary Metrics
    st.markdown("### Summary Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Hours", f"{hourly_ts.height}")
    m2.metric("Total Count", f"{hourly_ts['report_count'].sum():,}")
    m3.metric("Average/Hour", f"{hourly_ts['report_count'].mean():.2f}")
    m4.metric("Max/Hour", f"{hourly_ts['report_count'].max()}")
            
    st.divider()
    
    # Main Time Series Chart
    st.subheader("Hourly Execution Trend")
    
    col_ctrl1, col_ctrl2 = st.columns([2, 1])
    with col_ctrl1:
        days_to_show = st.slider("Visualization Window (days)", min_value=1, max_value=60, value=14, key="slider_ts_window")
    with col_ctrl2:
        show_rolling = st.checkbox("Show Moving Average (24h)", value=True)
    
    hours_to_show = int(days_to_show * 24)
    
    # Calculate rolling if requested
    plot_data = hourly_ts.tail(hours_to_show)
    if show_rolling:
        plot_data = plot_data.with_columns(
            pl.col("report_count").rolling_mean(window_size=24).alias("rolling_mean")
        )

    plot_data = plot_data.with_columns(
        pl.col("timestamp_hour").dt.strftime("%Y-%m-%dT%H:%M:%S").alias("timestamp_str")
    )
    
    # Advanced Vega-Lite Chart
    layers = [
        # Base Area
        {
            "mark": {"type": "area", "color": "#1f77b4", "opacity": 0.4},
            "encoding": {
                "x": {"field": "timestamp_str", "type": "temporal", "title": "Time"},
                "y": {"field": "report_count", "type": "quantitative", "title": "Report Count"},
                "tooltip": [
                    {"field": "timestamp_str", "type": "temporal", "title": "Date/Time"},
                    {"field": "report_count", "type": "quantitative", "title": "Count"}
                ]
            }
        }
    ]
    
    if show_rolling:
        layers.append({
            "mark": {"type": "line", "color": "#ff7f0e", "strokeWidth": 2},
            "encoding": {
                "x": {"field": "timestamp_str", "type": "temporal"},
                "y": {"field": "rolling_mean", "type": "quantitative"},
                "tooltip": [
                    {"field": "rolling_mean", "type": "quantitative", "title": "Moving Average (24h)"}
                ]
            }
        })

    st.vega_lite_chart(
        plot_data.to_dicts(),
        {
            "layer": layers,
            "width": "container",
            "height": 450
        },
        use_container_width=True
    )
    
    # Data Inspector & Stats
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Data Inspector")
        st.dataframe(hourly_ts.tail(100).to_pandas(), use_container_width=True)
    with c2:
        st.subheader("Descriptive Statistics")
        st.dataframe(hourly_ts.select("report_count").to_pandas().describe(), use_container_width=True)

    # Download link
    csv = hourly_ts.to_pandas().to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Hourly TS Data as CSV",
        data=csv,
        file_name="hourly_timeseries.csv",
        mime="text/csv",
    )

except Exception as e:
    st.error(f"Error en la aplicación: {e}")
    st.exception(e)
