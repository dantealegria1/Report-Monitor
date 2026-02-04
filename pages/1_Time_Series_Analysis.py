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
    st.title("📊 Análisis de Series de Tiempo")
    st.markdown("""
    Esta página muestra la serie de tiempo horaria del conteo de reportes ejecutados.
    Los datos están preparados para análisis de forecasting y detección de anomalías.
    """)
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total de Horas",
            f"{hourly_ts.height:,}",
            help="Número total de horas en la serie de tiempo"
        )
    
    with col2:
        if hourly_ts.height > 0:
            min_date = hourly_ts.select(pl.col("timestamp_hour").min()).item()
            st.metric(
                "Fecha Inicial",
                min_date.strftime("%Y-%m-%d %H:%M") if min_date else "N/A"
            )
    
    with col3:
        if hourly_ts.height > 0:
            max_date = hourly_ts.select(pl.col("timestamp_hour").max()).item()
            st.metric(
                "Fecha Final",
                max_date.strftime("%Y-%m-%d %H:%M") if max_date else "N/A"
            )
    
    st.divider()
    
    # Display options
    st.subheader("Vista de Datos")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        num_rows = st.number_input(
            "Número de filas a mostrar",
            min_value=10,
            max_value=1000,
            value=48,
            step=10,
            help="Muestra las primeras N filas de la serie de tiempo"
        )
    
    with col2:
        show_stats = st.checkbox("Mostrar estadísticas descriptivas", value=False)
    
    # Display dataframe
    st.dataframe(
        hourly_ts.head(num_rows),
        use_container_width=True,
        height=400
    )
    
    # Show statistics if requested
    if show_stats and hourly_ts.height > 0:
        st.divider()
        st.subheader("Estadísticas Descriptivas")
        
        stats = hourly_ts.select([
            pl.col("report_count").mean().alias("Media"),
            pl.col("report_count").median().alias("Mediana"),
            pl.col("report_count").std().alias("Desviación Estándar"),
            pl.col("report_count").min().alias("Mínimo"),
            pl.col("report_count").max().alias("Máximo"),
            pl.col("report_count").quantile(0.25).alias("Q1 (25%)"),
            pl.col("report_count").quantile(0.75).alias("Q3 (75%)"),
        ])
        
        st.dataframe(stats, use_container_width=True)
    
    # Download option
    st.divider()
    st.subheader("Exportar Datos")
    
    csv_data = hourly_ts.write_csv()
    st.download_button(
        label="📥 Descargar CSV",
        data=csv_data,
        file_name="hourly_time_series.csv",
        mime="text/csv",
        help="Descarga la serie de tiempo completa en formato CSV"
    )

except Exception as e:
    st.error(f"Error en la aplicación: {e}")
    st.exception(e)
