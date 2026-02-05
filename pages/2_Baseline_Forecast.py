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

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE (last hour)", "N/A" if mae1 is None else f"{mae1:.4f}")
    c2.metric("RMSE (last hour)", "N/A" if rmse1 is None else f"{rmse1:.4f}")
    c3.metric("Rows used", f"{n1}")

    st.caption("Baseline 2 (same hour last week) requiere por lo menos 168 horas de historia.")

    c4, c5, c6 = st.columns(3)
    c4.metric("MAE (last week)", "N/A" if mae2 is None else f"{mae2:.4f}")
    c5.metric("RMSE (last week)", "N/A" if rmse2 is None else f"{rmse2:.4f}")
    c6.metric("Rows used", f"{n2}")

    st.divider()

    # Ventana para gráficas
    days = st.slider("Ventana (días)", min_value=1, max_value=30, value=5)
    hours = int(days * 24)

    tail = df.tail(hours)

    tail_plot = (
        tail.with_columns(
            pl.col(ts_col).dt.strftime("%Y-%m-%dT%H:%M:%S").alias("timestamp_hour")
        )
        .rename({y_col: "y"})
    )

    st.subheader("Real vs predicción (last hour)")
    chart_data = tail_plot.select(["timestamp_hour", "y", "y_pred_last_hour"]).to_dicts()

    # Debug opcional para confirmar que hay filas
    # st.write("chart rows:", len(chart_data))
    # st.write(chart_data[:3])

    st.vega_lite_chart(
        chart_data,
        {
            "mark": "line",
            "encoding": {
                "x": {"field": "timestamp_hour", "type": "temporal"},
                "y": {"field": "value", "type": "quantitative"},
                "color": {"field": "series", "type": "nominal"},
            },
            "transform": [{"fold": ["y", "y_pred_last_hour"], "as": ["series", "value"]}],
        },
        use_container_width=True,
    )

    st.subheader("Error absoluto por hora (last hour baseline)")
    tail_err = tail_plot.with_columns(
        (pl.col("y") - pl.col("y_pred_last_hour")).abs().alias("abs_error")
    )
    err_data = tail_err.select(["timestamp_hour", "abs_error"]).to_dicts()

    st.vega_lite_chart(
        err_data,
        {
            "mark": "line",
            "encoding": {
                "x": {"field": "timestamp_hour", "type": "temporal"},
                "y": {"field": "abs_error", "type": "quantitative"},
            },
        },
        use_container_width=True,
    )


    # Error absoluto
    st.subheader("Error absoluto por hora (last hour baseline)")
    tail_err = tail_plot.with_columns(
        (pl.col("y") - pl.col("y_pred_last_hour")).abs().alias("abs_error")
    )
    err_data = tail_err.select(["timestamp_hour", "abs_error"]).to_dicts()

    st.vega_lite_chart(
        {"values": err_data},
        {
            "mark": "line",
            "encoding": {
                "x": {"field": "timestamp_hour", "type": "temporal"},
                "y": {"field": "abs_error", "type": "quantitative"},
            },
        },
        use_container_width=True,
    )

    st.divider()
    st.subheader("Muestra de datos")
    st.dataframe(
        tail_plot.select(["timestamp_hour", "y", "y_pred_last_hour", "y_pred_last_week"])
        .tail(50)
        .to_dicts()
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
