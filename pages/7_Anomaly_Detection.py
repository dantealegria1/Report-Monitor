import streamlit as st
import polars as pl
from db.database import load_reports_data
from utils.data_processing import (
    add_time_features, quality_sanitize, filter_by_date_range
)
from components.filters import (
    render_sidebar_filters,
    apply_filters,
    render_presentation_mode_toggle,
    apply_presentation_mode,
)
from components.anomaly_detection import render_anomaly_detection

st.set_page_config(page_title="Anomaly Detection", layout="wide")

# Inject Custom CSS
try:
    with open("assets/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

st.title("Anomaly Detection & Forensics")

# Load Data from session state
df_raw = st.session_state.get("df_raw")
df = st.session_state.get("df_all")

# Fallback if accessed directly
if df_raw is None or df is None:
    with st.spinner("Initializing data..."):
        df_raw = load_reports_data()
        df = add_time_features(df_raw)
        df = quality_sanitize(df)
        df = filter_by_date_range(df, "2025-01-01", "2025-12-31")
        
        st.session_state["df_raw"] = df_raw
        st.session_state["df_all"] = df

presentation_mode = render_presentation_mode_toggle()
df = apply_presentation_mode(df, presentation_mode)

# Filters
# Re-use filters from sidebar to keep consistency
filters = render_sidebar_filters(df)
df_filtered = apply_filters(df, filters)

# Render Main Component (Contains Tabs 1-4)
render_anomaly_detection(df_filtered)
