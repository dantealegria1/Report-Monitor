"""
Chart components.
Handles all visualization and chart rendering.
"""
import streamlit as st
import polars as pl


def render_execution_trends(df_filtered: pl.DataFrame):
    """
    Render execution trends chart by status.
    
    Args:
        df_filtered: Filtered DataFrame
    """
    st.subheader("Tendencia de Ejecuciones (por estatus)")
    evol = df_filtered.group_by(["date", "execution_status"]).agg(
        pl.len().alias("cantidad")
    ).to_pandas()
    
    if not evol.empty:
        piv = evol.pivot(index="date", columns="execution_status", values="cantidad").fillna(0)
        st.line_chart(piv)


def render_slowest_reports(df_filtered: pl.DataFrame):
    """
    Render top 10 slowest reports chart.
    
    Args:
        df_filtered: Filtered DataFrame
    """
    st.subheader("Top 10 Reportes mas Lentos (promedio success)")
    df_success = df_filtered.filter(pl.col("execution_status") == "success")
    
    slow = (
        df_success.group_by("ReportName")
        .agg(pl.col("duration_seconds").mean().alias("promedio"))
        .sort("promedio", descending=True)
        .head(10)
        .to_pandas()
    )
    
    if not slow.empty:
        st.bar_chart(data=slow, x="ReportName", y="promedio")
    else:
        st.info("Sin datos success para el rango.")


def render_top_failures(df_filtered: pl.DataFrame):
    """
    Render top 10 reports with most failures.
    
    Args:
        df_filtered: Filtered DataFrame
    """
    st.subheader("Top 10 Reportes con mas Fallos")
    df_failed = df_filtered.filter(pl.col("execution_status") == "failed")
    
    top_fail = (
        df_failed.group_by("ReportName")
        .agg(pl.len().alias("fallos"))
        .sort("fallos", descending=True)
        .head(10)
        .to_pandas()
    )
    
    if not top_fail.empty:
        st.bar_chart(data=top_fail, x="ReportName", y="fallos")
    else:
        st.info("No hay fallos en el rango.")


def render_duration_distribution(df_filtered: pl.DataFrame):
    """
    Render duration distribution histogram.
    
    Args:
        df_filtered: Filtered DataFrame
    """
    st.subheader("Distribucion de Duracion (success) - bins aproximados")
    df_success = df_filtered.filter(pl.col("execution_status") == "success")
    
    if df_success.height > 0:
        bin_size = st.slider("Tamaño de bin (seg)", 1, 120, 10)
        hist = (
            df_success
            .with_columns(((pl.col("duration_seconds") / bin_size).floor() * bin_size).alias("bin"))
            .group_by("bin")
            .agg(pl.len().alias("conteo"))
            .sort("bin")
            .to_pandas()
        )
        st.bar_chart(hist, x="bin", y="conteo")
    else:
        st.info("Sin datos success para histograma.")


def render_hourly_load(df_filtered: pl.DataFrame):
    """
    Render hourly load saturation chart.
    
    Args:
        df_filtered: Filtered DataFrame
    """
    st.subheader("Saturacion por Hora (conteo)")
    hourly_load = (
        df_filtered.group_by("hour")
        .agg(pl.len().alias("conteo"))
        .sort("hour")
        .to_pandas()
    )
    
    if not hourly_load.empty:
        st.area_chart(data=hourly_load, x="hour", y="conteo")


def render_heatmap(df_filtered: pl.DataFrame):
    """
    Render hour vs weekday heatmap.
    
    Args:
        df_filtered: Filtered DataFrame
    """
    st.subheader("Heatmap Hora vs Dia de semana (conteo)")
    heat = (
        df_filtered.group_by(["weekday", "hour"])
        .agg(pl.len().alias("conteo"))
        .to_pandas()
    )
    
    if not heat.empty:
        heat_piv = heat.pivot(index="weekday", columns="hour", values="conteo").fillna(0)
        st.dataframe(heat_piv, use_container_width=True)
