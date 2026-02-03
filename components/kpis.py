"""
KPI metrics component.
Displays key performance indicators.
"""
import streamlit as st
import polars as pl


def render_kpis(df_filtered: pl.DataFrame):
    """
    Render the 4 main KPI metrics.
    
    Args:
        df_filtered: Filtered DataFrame to calculate KPIs from
    """
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    total_execs = df_filtered.height
    
    df_success = df_filtered.filter(pl.col("execution_status") == "success")
    df_failed = df_filtered.filter(pl.col("execution_status") == "failed")
    
    avg_dur = float(df_success["duration_seconds"].mean()) if df_success.height > 0 else 0.0
    fail_rate = (df_failed.height / total_execs * 100) if total_execs > 0 else 0.0
    
    kpi1.metric("Total Ejecuciones", f"{total_execs:,}")
    kpi2.metric("Duracion Promedio (success)", f"{avg_dur:.2f}s")
    kpi3.metric("Tasa de Fallos", f"{fail_rate:.1f}%", delta_color="inverse")
    
    # Variability: median of MADs (robust measure)
    if df_success.height > 0:
        df_mad = df_success.group_by("ReportName").agg([
            pl.col("duration_seconds").median().alias("mediana"),
            (pl.col("duration_seconds") - pl.col("duration_seconds").median()).abs().median().alias("MAD")
        ])
        mad_global = float(df_mad["MAD"].median()) if df_mad.height > 0 else 0.0
        kpi4.metric("Variabilidad (MAD mediana)", f"{mad_global:.2f}s")
    else:
        kpi4.metric("Variabilidad (MAD mediana)", "N/A")


def render_health_checks(df_raw: pl.DataFrame, df_filtered: pl.DataFrame):
    """
    Render data quality and health checks section.
    
    Args:
        df_raw: Raw unfiltered DataFrame
        df_filtered: Filtered DataFrame
    """
    with st.expander("Calidad de datos / Health checks", expanded=False):
        if df_filtered.height == 0:
            st.info("No hay datos con los filtros actuales.")
        else:
            # Null counts per column
            nulls = df_raw.select([
                pl.all().null_count().name.suffix("_nulls")
            ]).to_pandas().T.reset_index()
            nulls.columns = ["columna", "nulos"]
            
            last_ts = df_filtered["started_at"].max()
            first_ts = df_filtered["started_at"].min()
            st.write(f"**Ventana actual:** {first_ts}  →  {last_ts}")
            
            st.dataframe(nulls, use_container_width=True)
