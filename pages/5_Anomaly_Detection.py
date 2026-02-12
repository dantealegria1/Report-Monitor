import streamlit as st
import polars as pl
from db.database import load_reports_data
from utils.data_processing import (
    add_time_features, quality_sanitize,
    filter_by_date_range, build_hourly_report_count
)
from components.filters import render_sidebar_filters, apply_filters
from components.anomaly_detection import render_anomaly_detection, render_drift_analysis

st.set_page_config(page_title="Anomaly Detection", layout="wide", page_icon="🚨")

# Inject Custom CSS
with open("assets/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("🚨 Anomaly Detection & Forensics")

# Load Data
df_raw = load_reports_data()
df = add_time_features(df_raw)
df = quality_sanitize(df)

# Filters
# Re-use filters from sidebar to keep consistency
filters = render_sidebar_filters(df)
df_filtered = apply_filters(df, filters)

# Tab structure
tab1, tab2, tab3 = st.tabs(["Anomaly Detection", "Drift Analysis", "Report Inspector"])

with tab1:
    render_anomaly_detection(df_filtered)

with tab2:
    render_drift_analysis(df_filtered)

with tab3:
    st.subheader("Report Execution History")
    st.markdown("Inspect the full execution history of a specific report.")
    
    unique_reports = df_filtered["ReportName"].unique().sort()
    
    selected_report = st.selectbox("Select Report to Inspect", unique_reports)
    
    if selected_report:
        df_report = df_filtered.filter(pl.col("ReportName") == selected_report).sort("started_at", descending=True)
        
        # Metrics for this report
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total Executions", df_report.height)
        with c2:
            st.metric("Avg Duration", f"{df_report['duration_seconds'].mean():.2f}s")
        with c3:
            fail_count = df_report.filter(pl.col("execution_status") != "success").height
            st.metric("Failures", fail_count)
            
        st.dataframe(
            df_report.select([
                "started_at", "execution_status", "duration_seconds", 
                "hour", "weekday", "ReportId", "Parameters"
            ]).to_pandas(),
            use_container_width=True
        )

        st.subheader("Performance over Time (Daily Average)")
        
        # Aggregate by day for cleaner visualization
        df_daily = (
            df_report.with_columns(pl.col("started_at").dt.date().alias("date"))
            .group_by("date")
            .agg([
                pl.col("duration_seconds").mean().alias("avg_duration"),
                pl.len().alias("count")
            ])
            .sort("date")
        )

        st.vega_lite_chart(
             df_daily.to_pandas(),
             {
                 "mark": "line",
                 "encoding": {
                     "x": {"field": "date", "type": "temporal", "title": "Date"},
                     "y": {"field": "avg_duration", "type": "quantitative", "title": "Avg Duration (s)"},
                     "tooltip": ["date", "avg_duration", "count"]
                 },
                 "width": "container",
                 "height": 400
             },
             use_container_width=True
        )
