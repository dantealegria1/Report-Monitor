import streamlit as st
import polars as pl
import math

st.set_page_config(page_title="Baseline Forecast", layout="wide")

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

    st.title("Baseline Forecast (Naive)")
    st.markdown("""
    Naive baselines are simple but powerful benchmarks. 
    They assume the future will look like the recent past.
    """)

    # --- Metrics Section ---
    st.subheader("Performance Metrics (Test Set)")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("**Baseline 1: Last Hour** (y at t-1)")
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE", "N/A" if mae1 is None else f"{mae1:.4f}")
        m2.metric("RMSE", "N/A" if rmse1 is None else f"{rmse1:.4f}")
        m3.metric("Samples", f"{n1}")

    with col2:
        st.info("**Baseline 2: Same Hour Last Week** (y at t-168)")
        m4, m5, m6 = st.columns(3)
        m4.metric("MAE", "N/A" if mae2 is None else f"{mae2:.4f}")
        m5.metric("RMSE", "N/A" if rmse2 is None else f"{rmse2:.4f}")
        m6.metric("Samples", f"{n2}")

    st.divider()

    # --- Visualization Section ---
    st.subheader("Forecasting Visualization")
    days = st.slider("Visualization Window (days)", min_value=1, max_value=30, value=7)
    hours = int(days * 24)

    tail = df.tail(hours)
    tail_plot = (
        tail.with_columns(
            pl.col(ts_col).dt.strftime("%Y-%m-%dT%H:%M:%S").alias("timestamp_str")
        )
        .rename({y_col: "actual"})
    )

    # Combined Comparison Chart
    st.markdown("**Actual vs Naive Baselines**")
    st.vega_lite_chart(
        tail_plot.to_dicts(),
        {
            "mark": {"type": "line", "point": False},
            "encoding": {
                "x": {"field": "timestamp_str", "type": "temporal", "title": "Time"},
                "y": {"field": "value", "type": "quantitative", "title": "Report Count"},
                "color": {
                    "field": "series", 
                    "type": "nominal",
                    "scale": {
                        "domain": ["actual", "y_pred_last_hour", "y_pred_last_week"],
                        "range": ["#1f77b4", "#ff7f0e", "#2ca02c"]
                    }
                }
            },
            "transform": [
                {"fold": ["actual", "y_pred_last_hour", "y_pred_last_week"], "as": ["series", "value"]}
            ],
            "width": "container",
            "height": 400
        },
        use_container_width=True,
    )

    st.divider()

    # --- Residual Analysis ---
    st.subheader("Residual Analysis (Last Hour Baseline)")
    
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
                "width": "container",
                "height": 300
            },
            use_container_width=True
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
            use_container_width=True
        )

    st.divider()
    st.subheader("Sample Data")
    st.dataframe(
        tail_plot.select(["timestamp_str", "actual", "y_pred_last_hour", "y_pred_last_week"])
        .tail(50),
        use_container_width=True
    )


# ----------------------------
# Page Entry
# ----------------------------
hourly_ts = st.session_state.get("hourly_ts")

if hourly_ts is None:
    st.warning("No encontré hourly_ts en session_state. Abre primero la página principal para generarlo.")
else:
    if not isinstance(hourly_ts, pl.DataFrame):
        st.error(f"hourly_ts no es un Polars DataFrame. Tipo: {type(hourly_ts)}")
    else:
        render_baseline_page(hourly_ts)
