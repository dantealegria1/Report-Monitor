"""
Anomaly detection component.
Handles anomaly detection and drift analysis UI.
"""
import streamlit as st
import polars as pl
from utils.statistics import robust_z_mad, ks_distance_approx


def render_anomaly_detection(df_filtered: pl.DataFrame):
    """
    Render anomaly detection section with robust and contextual methods.
    
    Args:
        df_filtered: Filtered DataFrame
    """
    st.subheader("Anomaly Detection (Robust + Contextual)")
    
    method = st.radio(
        "Method",
        ["Robust by ReportName (MAD)", "Contextual (ReportName + hour)", "Isolation Forest (Multivariate)"],
        horizontal=True
    )
    
    z_threshold = st.slider("Sensitivity Threshold (robust z)", 2.0, 10.0, 3.5)
    
    if df_filtered.height == 0:
        st.info("No data available with current filters.")
        return
    
    if method == "Robust by ReportName (MAD)":
        df_anom = df_filtered.with_columns([
            pl.col("duration_seconds").median().over("ReportName").alias("mediana_rep"),
            (pl.col("duration_seconds") - pl.col("duration_seconds").median().over("ReportName")).abs().median().over("ReportName").alias("mad_rep"),
        ]).with_columns([
            robust_z_mad(
                pl.col("duration_seconds"),
                pl.col("mediana_rep"),
                pl.col("mad_rep")
            ).alias("score")
        ]).filter(
            (pl.col("execution_status") == "success") &
            (pl.col("mad_rep") > 0) &
            (pl.col("score") > z_threshold)
        ).sort("score", descending=True)
    elif method == "Contextual (ReportName + hour)":
        # Contextual: by ReportName + hour
        df_anom = df_filtered.with_columns([
            pl.col("duration_seconds").median().over(["ReportName", "hour"]).alias("mediana_ctx"),
            (pl.col("duration_seconds") - pl.col("duration_seconds").median().over(["ReportName", "hour"])).abs().median().over(["ReportName", "hour"]).alias("mad_ctx"),
        ]).with_columns([
            robust_z_mad(
                pl.col("duration_seconds"),
                pl.col("mediana_ctx"),
                pl.col("mad_ctx")
            ).alias("score")
        ]).filter(
            (pl.col("execution_status") == "success") &
            (pl.col("mad_ctx") > 0) &
            (pl.col("score") > z_threshold)
        ).sort("score", descending=True)
    else: # Isolation Forest
        from sklearn.ensemble import IsolationForest
        import pandas as pd
        
        # Prepare features for Isolation Forest
        df_prep = df_filtered

        # Select features
        # User requested: duration, hour, weekday, ReportId
        features = ["duration_seconds", "hour", "weekday", "ReportId"]
        
        # Ensure ReportId is available and numeric
        if "ReportId" not in df_prep.columns:
             # Fallback if ReportId is missing for some reason (e.g. old cached data)
             features.remove("ReportId")
        
        X = df_prep.select(features).to_pandas().fillna(0)
        
        # Train Isolation Forest
        # contamination='auto' or small value. 
        # We can map z_threshold to contamination roughly: higher threshold = lower contamination
        contamination = 0.01 + (10.0 - z_threshold) / 100.0 # Mapping 2.0->0.09, 10.0->0.01
        
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        preds = iso_forest.fit_predict(X) #-1 for outlier, 1 for inlier
        scores = iso_forest.decision_function(X) # lower is more anomalous
        
        # Add back to dataframe
        df_anom = df_prep.with_columns([
            pl.Series("iso_pred", preds),
            pl.Series("iso_score", scores)
        ]).filter(
            pl.col("iso_pred") == -1
        ).with_columns(
             # Invert score for consistency (higher is more anomalous in our UI)
             (pl.col("iso_score") * -1).alias("score") 
        ).sort("score", descending=True)
    
    if df_anom.height > 0:
        st.warning(f"Detected {df_anom.height} anomalies (score > {z_threshold}).")
        st.dataframe(
            df_anom.select([
                "started_at", "ReportType", "ReportName", "hour",
                "duration_seconds", "execution_status",
                pl.col("score"),
            ]).to_pandas(),
            use_container_width=True
        )
    else:
        st.success("No anomalies detected under this threshold.")


def render_drift_analysis(df_filtered: pl.DataFrame):
    """
    Render drift analysis section.
    
    Args:
        df_filtered: Filtered DataFrame
    """
    st.subheader("Performance Drift (Quantile Proxy)")
    st.caption("Compares duration quantiles: last 7 days vs previous 30-day window (approx).")
    
    drift_level = st.selectbox("Analysis Level", ["By ReportType", "By ReportName (Top 50)"])
    
    df_success = df_filtered.filter(pl.col("execution_status") == "success")
    
    if df_success.height > 0:
        if drift_level == "Por ReportType":
            drift_tbl = ks_distance_approx(df_success, "duration_seconds", ["ReportType"])
        else:
            top50 = (
                df_success.group_by("ReportName")
                .agg(pl.len().alias("n"))
                .sort("n", descending=True)
                .head(50)["ReportName"]
                .to_list()
            )
            drift_tbl = ks_distance_approx(
                df_success.filter(pl.col("ReportName").is_in(top50)),
                "duration_seconds",
                ["ReportName"]
            )
        
        if drift_tbl.height > 0:
            st.dataframe(drift_tbl.to_pandas(), use_container_width=True)
        else:
            st.info("Insufficient data to compare windows (requires at least ~10 executions per window).")
    else:
        st.info("No successful data for drift analysis.")
