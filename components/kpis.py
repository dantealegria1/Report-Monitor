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
    
    kpi1.metric("Total Executions", f"{total_execs:,}")
    kpi2.metric("Average Duration (success)", f"{avg_dur:.2f}s")
    kpi3.metric("Failure Rate", f"{fail_rate:.1f}%", delta_color="inverse")
    
    # Variability: median of MADs (robust measure)
    if df_success.height > 0:
        df_mad = df_success.group_by("ReportName").agg([
            pl.col("duration_seconds").median().alias("mediana"),
            (pl.col("duration_seconds") - pl.col("duration_seconds").median()).abs().median().alias("MAD")
        ])
        mad_global = float(df_mad["MAD"].median()) if df_mad.height > 0 else 0.0
        kpi4.metric("Variability (median MAD)", f"{mad_global:.2f}s")
    else:
        kpi4.metric("Variability (median MAD)", "N/A")


def render_health_checks(df_raw: pl.DataFrame, df_filtered: pl.DataFrame):
    """
    Render data quality and health checks section.
    
    Args:
        df_raw: Raw unfiltered DataFrame
        df_filtered: Filtered DataFrame
    """
    with st.expander("Data Quality / Health Checks", expanded=False):
        if df_filtered.height == 0:
            st.info("No data available with current filters.")
        else:
            # Null counts per column
            nulls = df_raw.select([
                pl.all().null_count().name.suffix("_nulls")
            ]).to_pandas().T.reset_index()
            nulls.columns = ["column", "nulls"]
            
            last_ts = df_filtered["started_at"].max()
            first_ts = df_filtered["started_at"].min()
            st.write(f"**Current Window:** {first_ts}  to  {last_ts}")
            
            st.dataframe(nulls, use_container_width=True)


def render_mase_kpi(mase: float | None):
    """
    Render MASE (Mean Absolute Scaled Error) KPI with color coding.
    
    Args:
        mase: The calculated MASE value
    """
    if mase is None:
        st.metric("MASE (vs Seasonal Naive)", "N/A", help="Not enough data to calculate MASE")
        return

    # Color coding logic
    # < 1.0 : Green (Better than naive)
    # 1.0 - 1.1 : Yellow (Similar to naive)
    # > 1.1 : Red (Worse than naive)
    
    improvement_pct = (1 - mase) * 100

    if mase < 1.0:
        # Si el MASE baja (ej. 0.57), la mejora es positiva (43%)
        # Usamos inverse para que un delta negativo (reducción de error) sea verde
        st.metric(
            label="MASE",
            value=f"{mase:.2f}",
            delta=f"-{improvement_pct:.1f}% error vs Naive",
            delta_color="inverse", 
            help="If MASE < 1, the model is more accurate than repeating the previous value."
        )
    elif mase <= 1.1:
        st.metric(
            label="MASE",
            value=f"{mase:.2f}",
            delta="Similar to Baseline",
            delta_color="off"
        )
    else:
        # Si el MASE sube (ej. 1.2), el error aumentó un 20%
        # Con inverse, un delta positivo (más error) se pintará de rojo
        st.metric(
            label="MASE",
            value=f"{mase:.2f}",
            delta=f"+{(mase - 1)*100:.1f}% error vs Naive",
            delta_color="inverse"
        )
    

