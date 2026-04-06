"""
Report Monitor
 - Streamlit Application
Main application file orchestrating all components.
"""
import streamlit as st

# Configuration and validation
from config import validate_config

# Database operations
from db.database import load_reports_data

# Data processing utilities
from utils.data_processing import (
    add_time_features, quality_sanitize,
    filter_by_date_range, build_hourly_report_count
)

# UI Components
from components.filters import (
    render_sidebar_filters,
    apply_filters,
    render_presentation_mode_toggle,
    apply_presentation_mode,
)
from components.kpis import render_kpis, render_health_checks
from components.charts import (
    render_execution_trends,
    render_slowest_reports,
    render_top_failures,
    render_duration_distribution,
    render_hourly_load,
    render_heatmap,
    render_double_run_detection
)
from components.anomaly_detection import render_anomaly_detection, render_drift_analysis

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(page_title="Report Monitor", layout="wide")

# Inject Custom CSS
with open("assets/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ----------------------------
# Main Application
# ----------------------------
try:
    # Validate configuration
    validate_config()
    
    # Load and process data
    @st.cache_data(show_spinner="Loading global data...")
    def get_central_data():
        df_raw = load_reports_data()
        df = add_time_features(df_raw)
        df = quality_sanitize(df)
        
        # Enforce global analysis window for 2025
        df = filter_by_date_range(df, "2024-01-01", "2026-03-10")
        
        # Create hourly time series
        hourly_ts = build_hourly_report_count(df)
        return df_raw, df, hourly_ts

    df_raw, df, hourly_ts = get_central_data()
    
    # Store in session state for pages
    st.session_state["df_raw"] = df_raw
    st.session_state["df_all"] = df
    st.session_state["hourly_ts"] = hourly_ts
    
    presentation_mode = render_presentation_mode_toggle()
    df = apply_presentation_mode(df, presentation_mode)


    # Render sidebar filters
    filters = render_sidebar_filters(df)
    df_filtered = apply_filters(df, filters)
    st.session_state["df_filtered"] = df_filtered
    st.session_state["global_filters"] = filters
    
    # ----------------------------
    # Main Content
    # ----------------------------
    st.title("Report Monitor")
    
    # Main dashboard
    st.header("Overview")
    
    # 1. KPIs
    render_kpis(df_filtered)
    
    # 2. Charts
    col1, col2 = st.columns(2)
    with col1:
        render_execution_trends(df_filtered)
    with col2:
        render_slowest_reports(df_filtered)
        
    render_duration_distribution(df_filtered, key="dist_top")
    
    # 3. Health Checks
    render_health_checks(df_raw, df_filtered)
    
    # Top failures
    st.divider()
    render_top_failures(df_filtered)
    
    # Duration distribution
    st.divider()
    render_duration_distribution(df_filtered, key="dist_bottom")
    
    # Hourly load and heatmap
    st.divider()
    render_hourly_load(df_filtered)
    render_heatmap(df_filtered)
    
    # Double-Run Detection
    st.divider()
    render_double_run_detection(df, filters)

    # Anomaly detection & Drift Analysis moved to pages/5_Anomaly_Detection.py
    # st.divider()
    # render_anomaly_detection(df_filtered)
    # st.divider()
    # render_drift_analysis(df_filtered)

except Exception as e:
    st.error(f"Application error: {e}")
