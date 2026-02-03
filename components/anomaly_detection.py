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
    st.subheader("Deteccion de Anomalias (Robusto + Contextual)")
    
    method = st.radio(
        "Metodo",
        ["Robusto por ReportName (MAD)", "Contextual (ReportName + hora)"],
        horizontal=True
    )
    
    z_threshold = st.slider("Umbral de Sensibilidad (z robusto)", 2.0, 10.0, 3.5)
    
    if df_filtered.height == 0:
        st.info("No hay datos con los filtros actuales.")
        return
    
    if method == "Robusto por ReportName (MAD)":
        df_anom = df_filtered.with_columns([
            pl.col("duration_seconds").median().over("ReportName").alias("mediana_rep"),
            (pl.col("duration_seconds") - pl.col("duration_seconds").median().over("ReportName")).abs().median().over("ReportName").alias("mad_rep"),
        ]).with_columns([
            robust_z_mad(
                pl.col("duration_seconds"),
                pl.col("mediana_rep"),
                pl.col("mad_rep")
            ).alias("z_robusto")
        ]).filter(
            (pl.col("execution_status") == "success") &
            (pl.col("mad_rep") > 0) &
            (pl.col("z_robusto") > z_threshold)
        ).sort("z_robusto", descending=True)
    else:
        # Contextual: by ReportName + hour
        df_anom = df_filtered.with_columns([
            pl.col("duration_seconds").median().over(["ReportName", "hour"]).alias("mediana_ctx"),
            (pl.col("duration_seconds") - pl.col("duration_seconds").median().over(["ReportName", "hour"])).abs().median().over(["ReportName", "hour"]).alias("mad_ctx"),
        ]).with_columns([
            robust_z_mad(
                pl.col("duration_seconds"),
                pl.col("mediana_ctx"),
                pl.col("mad_ctx")
            ).alias("z_robusto")
        ]).filter(
            (pl.col("execution_status") == "success") &
            (pl.col("mad_ctx") > 0) &
            (pl.col("z_robusto") > z_threshold)
        ).sort("z_robusto", descending=True)
    
    if df_anom.height > 0:
        st.warning(f"Se detectaron {df_anom.height} anomalias (z robusto > {z_threshold}).")
        st.dataframe(
            df_anom.select([
                "started_at", "ReportType", "ReportName", "hour",
                "duration_seconds", "execution_status",
                pl.col("z_robusto"),
            ]).to_pandas(),
            use_container_width=True
        )
    else:
        st.success("No hay anomalias detectadas bajo este umbral.")


def render_drift_analysis(df_filtered: pl.DataFrame):
    """
    Render drift analysis section.
    
    Args:
        df_filtered: Filtered DataFrame
    """
    st.subheader("Drift de rendimiento (proxy por cuantiles)")
    st.caption("Compara cuantiles de duración: últimos 7 días vs ventana previa de 30 días (aprox).")
    
    drift_level = st.selectbox("Nivel de análisis", ["Por ReportType", "Por ReportName (Top 50)"])
    
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
            st.info("No hay suficientes datos para comparar ventanas (se requieren al menos ~10 ejecuciones por ventana).")
    else:
        st.info("Sin datos success para drift.")
