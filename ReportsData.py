import os
import pyodbc
import polars as pl
import numpy as np
from dotenv import load_dotenv

load_dotenv()

def run_etl_process(start_date=None, end_date=None):
    # Configuracion de conexion
    DB_USER = os.getenv("DB_USER", "sa")
    DB_PASS = os.getenv("DB_PASSWORD", "Strong!Passw0rd")
    DB_HOST = os.getenv("DB_HOST", "localhost,1433")
    DB_NAME = os.getenv("DB_NAME", "vt4")
    
    conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={DB_HOST};DATABASE={DB_NAME};UID={DB_USER};PWD={DB_PASS};TrustServerCertificate=yes;"

    # Consulta SQL con parametros opcionales de fecha
    query = """
    SELECT
        ra.StartedProcessingOn AS started_at,
        DATEDIFF(SECOND, ra.StartedProcessingOn, ra.CompletionDate) AS duration_seconds,
        CASE
            WHEN ra.Cancelled = 1 THEN 'cancelled'
            WHEN ra.Failed = 1 OR ra.ErrorMessage IS NOT NULL THEN 'failed'
            ELSE 'success'
        END AS execution_status,
        TRIM(r.ReportName) AS ReportName,
        TRIM(q.ReportType) AS ReportType
    FROM reportsdata ra
    JOIN reports r ON ra.ReportId = r.Id
    JOIN ReportQueues q ON q.Id = r.QueueNumber
    WHERE ra.StartedProcessingOn IS NOT NULL
    """
    
    if start_date:
        query += f" AND ra.StartedProcessingOn >= '{start_date}'"
    if end_date:
        query += f" AND ra.StartedProcessingOn <= '{end_date}'"

    print(f"Iniciando procesamiento: {start_date or 'Inicio'} a {end_date or 'Actualidad'}")

    try:
        with pyodbc.connect(conn_str) as conn:
            df = pl.read_database(query=query, connection=conn)
    except Exception as e:
        print(f"Error de conexion: {e}")
        return

    # Transformacion de datos y Feature Engineering
    df = df.with_columns([
        pl.col("started_at").cast(pl.Datetime),
        pl.col("started_at").dt.truncate("1mo").alias("month_year"),
        pl.col("started_at").dt.hour().alias("hour"),
        pl.col("started_at").dt.weekday().alias("day_of_week")
    ]).filter(pl.col("duration_seconds") >= 0)

    # 1. Metricas de Precision del Modelo (MAE, RMSE)
    # Requerido para evaluar el rendimiento predictivo segun pagina 29 de la tesina
    df_success = df.filter(pl.col("execution_status") == "success")
    
    df_metrics = df_success.with_columns([
        pl.col("duration_seconds").mean().over("ReportName").alias("prediction")
    ]).with_columns([
        (pl.col("duration_seconds") - pl.col("prediction")).alias("error")
    ]).with_columns([
        pl.col("error").abs().alias("AE"),
        (pl.col("error")**2).alias("SE")
    ])

    # Exportacion de metricas de error mensual
    df_metrics.group_by("month_year").agg([
        (pl.col("AE").mean()).alias("MAE"),
        (pl.col("SE").mean().sqrt()).alias("RMSE")
    ]).sort("month_year").write_csv("pbi_metricas_error_mensual.csv")

    # 2. Evolucion Temporal de Tiempos Promedio
    df_success.group_by(["month_year", "ReportName"]).agg(
        pl.col("duration_seconds").mean().alias("avg_dur")
    ).sort(["month_year", "avg_dur"]).write_csv("pbi_evolucion_tiempos.csv")

    # 3. Carga por Dia y Hora (Heatmap de saturacion)
    df.group_by(["day_of_week", "hour"]).agg(
        pl.len().alias("count")
    ).sort(["day_of_week", "hour"]).write_csv("pbi_carga_horaria.csv")

    # 4. Analisis de Reportes Criticos e Impacto
    df_success.group_by("ReportName").agg([
        pl.len().alias("execs"),
        pl.col("duration_seconds").mean().alias("avg_dur"),
        pl.col("duration_seconds").sum().alias("total_impact_s"),
        pl.col("duration_seconds").std().alias("variability_std")
    ]).sort("total_impact_s", descending=True).write_csv("pbi_reportes_criticos.csv")

    # 5. Eficiencia por Tipo de Reporte (Success Rate)
    df.group_by("ReportType").agg([
        pl.len().alias("total"),
        (pl.col("execution_status") == "success").sum().alias("ok")
    ]).with_columns(
        (pl.col("ok") / pl.col("total") * 100).alias("success_rate")
    ).write_csv("pbi_eficiencia_tipo.csv")

    # 6. Deteccion de Anomalias (Z-Score)
    # Implementacion segun avance de solucion en pagina 31
    df.with_columns([
        ((pl.col("duration_seconds") - pl.col("duration_seconds").mean().over("ReportName")) / 
         pl.col("duration_seconds").std().over("ReportName")).alias("z_score")
    ]).filter(pl.col("z_score") > 3).write_csv("pbi_excepciones_criticas.csv")

    print("Proceso completado. Archivos CSV generados para Power BI.")

if __name__ == "__main__":
    # Parametros ejemplo: run_etl_process(start_date="2024-01-01", end_date="2024-12-31")
    run_etl_process()