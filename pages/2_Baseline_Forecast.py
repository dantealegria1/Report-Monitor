import streamlit as st
import polars as pl
import math
from datetime import datetime, timedelta

st.set_page_config(page_title="Baseline Forecast", layout="wide")

# Inject Custom CSS
try:
    with open("assets/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

def _infer_columns(hourly_ts: pl.DataFrame) -> tuple[str, str]:
    """
    Intenta adivinar columnas típicas:
    - timestamp: timestamp_hour / hour / ts / timestamp
    - y: y / count / report_count / reports / n
    """
    cols = set(hourly_ts.columns)

    ts_candidates = ["timestamp_hour", "hour", "ts", "timestamp"]
    y_candidates = ["y", "count", "report_count", "reports", "n"]

    ts_col = next((c for c in ts_candidates if c in cols), None)
    y_col = next((c for c in y_candidates if c in cols), None)

    if not ts_col or not y_col:
        raise ValueError(
            f"No pude inferir columnas. Columnas disponibles: {hourly_ts.columns}\n"
            f"Necesito algo tipo timestamp_hour + report_count (o y/count)."
        )

    return ts_col, y_col


def add_naive_predictions(df: pl.DataFrame, y_col: str) -> pl.DataFrame:
    return df.with_columns(
        [
            pl.col(y_col).shift(1).alias("y_pred_last_hour"),
            pl.col(y_col).shift(168).alias("y_pred_last_week"),  # 24*7
        ]
    )


def _mae_rmse(df: pl.DataFrame, y_col: str, pred_col: str) -> tuple[float | None, float | None, int]:
    d = df.select([pl.col(y_col), pl.col(pred_col)]).drop_nulls()
    n = d.height
    if n == 0:
        return None, None, 0

    mae = (
        d.with_columns((pl.col(y_col) - pl.col(pred_col)).abs().alias("ae"))
        .select(pl.col("ae").mean())
        .item()
    )
    mse = (
        d.with_columns(((pl.col(y_col) - pl.col(pred_col)) ** 2).alias("se"))
        .select(pl.col("se").mean())
        .item()
    )
    rmse = math.sqrt(mse) if mse is not None else None
    return float(mae), float(rmse), n


def render_baseline_page(hourly_ts: pl.DataFrame):
    ts_col, y_col = _infer_columns(hourly_ts)

    df = hourly_ts

    # Si timestamp viene como string, parsea. Si ya es datetime, no lo toques.
    if df.schema[ts_col] == pl.Utf8:
        df = df.with_columns(
            pl.col(ts_col)
            .str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S%.f", strict=False)
            .alias(ts_col)
        )

    df = (
        df.sort(ts_col)
          .with_columns(pl.col(y_col).fill_null(0))
    )

    df = add_naive_predictions(df, y_col=y_col)

    mae1, rmse1, n1 = _mae_rmse(df, y_col, "y_pred_last_hour")
    mae2, rmse2, n2 = _mae_rmse(df, y_col, "y_pred_last_week")

    st.title("Baseline Forecast Comparison")
    st.write("""
        Baseline models serve as a reference point for more complex models like Prophet.
        If a complex model cannot beat these simple baselines, it may be over-configured or 
        the data might have high noise levels.
    """)

    # --- Metrics Section ---
    st.subheader("Performance Metrics (Test Set)")
    
    c_met1, c_met2 = st.columns(2)
    with c_met1:
        st.markdown("**Baseline 1: Last Hour (t-1)**")
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE", "N/A" if mae1 is None else f"{mae1:.4f}")
        m2.metric("RMSE", "N/A" if rmse1 is None else f"{rmse1:.4f}")
        m3.metric("Samples", f"{n1}")

    with c_met2:
        st.markdown("**Baseline 2: Same Hour Last Week (t-168)**")
        m4, m5, m6 = st.columns(3)
        m4.metric("MAE", "N/A" if mae2 is None else f"{mae2:.4f}")
        m5.metric("RMSE", "N/A" if rmse2 is None else f"{rmse2:.4f}")
        m6.metric("Samples", f"{n2}")

    st.divider()

    # --- Visualization Section ---
    st.subheader("Comparison: Actual vs Baseline")
    
    # Calculate min/max dates for picker
    # polars .item() returns native python datetime objects
    min_date = df.select(pl.col(ts_col).min()).item()
    max_date = df.select(pl.col(ts_col).max()).item()
    
    # Default to last 7 days
    default_start = max_date - timedelta(days=7)

    c_date1, c_date2 = st.columns([1, 3])
    with c_date1:
        date_range = st.date_input(
            "Select Date Range",
            value=(default_start, max_date),
            min_value=min_date,
            max_value=max_date,
            key="baseline_date_range"
        )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        # Convert date objects to datetime for comparison
        start_dt = datetime.combine(start_date, datetime.min.time())
        end_dt = datetime.combine(end_date, datetime.max.time())
        
        tail_plot = df.filter(
            (pl.col(ts_col) >= start_dt) & (pl.col(ts_col) <= end_dt)
        )
    else:
        st.warning("Please select a valid date range (Start - End).")
        tail_plot = df.tail(24*7)

    tail_plot = (
        tail_plot.with_columns(
            pl.col(ts_col).dt.strftime("%Y-%m-%dT%H:%M:%S").alias("timestamp_str")
        )
        .rename({y_col: "actual"})
    )

    # Combined Comparison Chart
    st.markdown("**Actual vs Naive Baselines**")
    
    chart_data = tail_plot.select([
        "timestamp_str",
        "actual",
        "y_pred_last_hour",
        "y_pred_last_week"
    ]).to_dicts()

    if chart_data:
        st.vega_lite_chart(
        chart_data,
        {
            "encoding": {"x": {"field": "timestamp_str", "type": "temporal", "title": "Time"}},
            "transform": [
                {"fold": ["actual", "y_pred_last_hour", "y_pred_last_week"], "as": ["Variable", "Value"]}
            ],
            "mark": "line",
            "encoding": {
                "x": {"field": "timestamp_str", "type": "temporal", "title": "Time"},
                "y": {"field": "Value", "type": "quantitative", "title": "Report Count"},
                "color": {
                    "field": "Variable", 
                    "type": "nominal",
                    "scale": {
                        "domain": ["actual", "y_pred_last_hour", "y_pred_last_week"],
                        "range": ["#1f77b4", "#ff7f0e", "#2ca02c"]
                    },
                    "legend": {"title": "Series", "orient": "bottom"}
                },
                "strokeDash": {
                    "field": "Variable",
                    "scale": {
                        "domain": ["actual", "y_pred_last_hour", "y_pred_last_week"],
                        "range": [[1,0], [5,5], [2,2]]
                    }
                },
                "opacity": {
                    "condition": {"test": "datum.Variable === 'actual'", "value": 0.8},
                    "value": 1
                }
            },
            "height": 450,
            "width": "container"
        },
        use_container_width=True
    )

    st.divider()

    # --- Residual Analysis ---
    st.subheader("Residual Analysis (Last Hour)")
    
    tail_res = tail_plot.with_columns(
        (pl.col("actual") - pl.col("y_pred_last_hour")).alias("residual")
    ).drop_nulls()
    
    c_res1, c_res2 = st.columns(2)
    
    with c_res1:
        st.markdown("**Residuals over Time**")
        st.vega_lite_chart(
            tail_res.select(["timestamp_str", "residual"]).to_dicts(),
            {
                "mark": {"type": "bar", "color": "#d62728"},
                "encoding": {
                    "x": {"field": "timestamp_str", "type": "temporal", "title": "Time"},
                    "y": {"field": "residual", "type": "quantitative", "title": "Error"}
                },
                "height": 300
            },
                width='stretch'
            )
        
    with c_res2:
        st.markdown("**Error Distribution**")
        st.vega_lite_chart(
            tail_res.select("residual").to_dicts(),
            {
                "mark": {"type": "bar", "color": "#9467bd"},
                "encoding": {
                    "x": {"bin": True, "field": "residual", "title": "Error Magnitude"},
                    "y": {"aggregate": "count", "title": "Frequency"}
                },
                "width": "container",
                "height": 300
            },
                width='stretch'
            )

    st.divider()
    st.subheader("Sample Data")
    st.dataframe(
        tail_plot.select(["timestamp_str", "actual", "y_pred_last_hour", "y_pred_last_week"])
        .tail(50),
        width='stretch'
    )


# ----------------------------
# Page Entry
# ----------------------------
df_raw = st.session_state.get("df_raw")
hourly_ts = st.session_state.get("hourly_ts")

# Fallback if accessed directly
if df_raw is None or hourly_ts is None:
    from db.database import load_reports_data
    from utils.data_processing import add_time_features, quality_sanitize, filter_by_date_range, build_hourly_report_count
    
    with st.spinner("Initializing data..."):
        df_raw = load_reports_data()
        df = add_time_features(df_raw)
        df = quality_sanitize(df)
        df = filter_by_date_range(df, "2025-01-01", "2025-12-31")
        hourly_ts = build_hourly_report_count(df)
        
        st.session_state["df_raw"] = df_raw
        st.session_state["df_all"] = df
        st.session_state["hourly_ts"] = hourly_ts

if hourly_ts is None:
    st.warning("No data available for display.")
else:
    if not isinstance(hourly_ts, pl.DataFrame):
        st.error(f"hourly_ts is not a Polars DataFrame. Type: {type(hourly_ts)}")
    else:
        render_baseline_page(hourly_ts)
