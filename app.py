"""
IaaS Report Monitor - Streamlit Application
Main application file orchestrating all components.
"""
import streamlit as st

# Configuration and validation
from config import validate_config

# Database operations
from db.database import load_reports_data

# Data processing utilities
from utils.data_processing import add_time_features, quality_sanitize

# UI Components
from components.filters import render_sidebar_filters, apply_filters
from components.kpis import render_kpis, render_health_checks
from components.charts import (
    render_execution_trends,
    render_slowest_reports,
    render_top_failures,
    render_duration_distribution,
    render_hourly_load,
    render_heatmap
)
from components.anomaly_detection import render_anomaly_detection, render_drift_analysis

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(page_title="IaaS Report Monitor", layout="wide")

# ----------------------------
# Main Application
# ----------------------------
try:
    # Validate configuration
    validate_config()
    
    # Load and process data
    df_raw = load_reports_data()
    df = add_time_features(df_raw)
    df = quality_sanitize(df)
    
    # Render sidebar filters
    filters = render_sidebar_filters(df)
    df_filtered = apply_filters(df, filters)
    
    # ----------------------------
    # Main Content
    # ----------------------------
    st.title("Panel de Control: Optimizacion de Reportes IaaS")
    
    # Health checks
    render_health_checks(df_raw, df_filtered)
    
    # KPIs
    render_kpis(df_filtered)
    
    # Charts Section
    st.divider()
    c1, c2 = st.columns(2)
    
    with c1:
        render_execution_trends(df_filtered)
    
    with c2:
        render_slowest_reports(df_filtered)
    
    # Top failures
    st.divider()
    render_top_failures(df_filtered)
    
    # Duration distribution
    st.divider()
    render_duration_distribution(df_filtered)
    
    # Hourly load and heatmap
    st.divider()
    render_hourly_load(df_filtered)
    render_heatmap(df_filtered)
    
    # Anomaly detection
    st.divider()
    render_anomaly_detection(df_filtered)
    
    # Drift analysis
    st.divider()
    render_drift_analysis(df_filtered)

except Exception as e:
    st.error(f"Error en la aplicacion: {e}")