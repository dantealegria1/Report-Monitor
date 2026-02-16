"""
Database operations module.
Handles database connections and data retrieval.
"""
import streamlit as st
import pyodbc
import polars as pl
from config import DB_USER, DB_PASS, DB_HOST, DB_NAME, CACHE_TTL


def get_connection_string():
    """Build and return the database connection string."""
    return (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        f"SERVER={DB_HOST};DATABASE={DB_NAME};UID={DB_USER};PWD={DB_PASS};"
        "TrustServerCertificate=yes;"
    )


@st.cache_data(ttl=CACHE_TTL)
def load_reports_data():
    """
    Load report execution data from the database.
    
    Returns:
        pl.DataFrame: Polars DataFrame containing report execution data
    """
    query = """
    SELECT 
        ra.CreationDate AS creation_date,
        ra.StartedProcessingOn AS started_at,
        ra.CompletionDate AS completed_at,
        DATEDIFF(SECOND, ra.StartedProcessingOn, ra.CompletionDate) AS duration_seconds,
        CASE 
            WHEN ra.Cancelled = 1 THEN 'cancelled'
            WHEN ra.Failed = 1 OR ra.ErrorMessage IS NOT NULL THEN 'failed'
            ELSE 'success'
        END AS execution_status,
        TRIM(r.ReportName) AS ReportName,
        r.Id AS ReportId,
        ra.Parameters,
        TRIM(q.ReportType) AS ReportType
    FROM reportsdata ra
    JOIN reports r ON ra.ReportId = r.Id
    JOIN ReportQueues q ON q.Id = r.QueueNumber
    WHERE ra.StartedProcessingOn IS NOT NULL
    """
    
    conn_str = get_connection_string()
    with pyodbc.connect(conn_str) as conn:
        return pl.read_database(query=query, connection=conn)
