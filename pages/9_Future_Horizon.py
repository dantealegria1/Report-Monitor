"""
Future Horizon Page
Uses the trained Prophet + XGBoost hybrid model to forecast
report counts for the next 15 or 30 days.
"""
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from prophet.serialize import model_from_json
import plotly.graph_objects as go
import json
import holidays
from datetime import datetime, timedelta

st.set_page_config(page_title="Future Horizon", layout="wide")

try:
    with open("assets/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

# ─────────────────────────────────────────────
# Load models
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        with open("prophet_model.json", "r") as f:
            prophet = model_from_json(f.read())
        xgb_model = xgb.XGBRegressor()
        xgb_model.load_model("xgboost_model.json")
        with open("label_mapping.json", "r") as f:
            label_map = json.load(f)
        return prophet, xgb_model, label_map
    except Exception as e:
        return None, None, None

prophet_model, xgb_model, label_map = load_models()

st.title("Future Horizon — Report Load Forecast")
st.markdown(
    "Projection of expected report executions using the **Hybrid Prophet + XGBoost** model. "
    "Lags are estimated recursively from the last known actuals."
)

if prophet_model is None:
    st.warning(
        "Trained models not found. Please run `trainer.py` first to generate "
        "`prophet_model.json` and `xgboost_model.json`."
    )
    st.stop()

# ─────────────────────────────────────────────
# Sidebar controls
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("Forecast Settings")
    horizon_days = st.selectbox("Forecast Horizon", [15, 30], index=0, format_func=lambda x: f"{x} days")
    granularity = st.radio("View Granularity", ["Hourly", "Daily"], index=1, horizontal=True)
    backlog_seed = st.number_input(
        "Current Backlog (pending reports)",
        min_value=0, value=10,
        help="Estimated number of reports currently queued. Held constant for the forecast horizon."
    )
    report_type_id = st.selectbox(
        "Load Profile",
        options=list(label_map.keys()),
        format_func=lambda k: label_map[k],
        index=0
    )

# ─────────────────────────────────────────────
# Seed lags from session state (hourly_ts) or defaults
# ─────────────────────────────────────────────
hourly_ts = st.session_state.get("hourly_ts")

if hourly_ts is not None:
    import polars as pl
    ts_pd = hourly_ts.to_pandas()
    if "timestamp_hour" in ts_pd.columns:
        ts_pd = ts_pd.rename(columns={"timestamp_hour": "ds", "report_count": "y"})
    last_actuals = ts_pd["y"].values[-3:]  # last 3 known hourly counts
    seed_lag1 = float(last_actuals[-1]) if len(last_actuals) >= 1 else 10.0
    seed_lag2 = float(last_actuals[-2]) if len(last_actuals) >= 2 else 10.0
    seed_roll3 = float(np.mean(last_actuals[-3:])) if len(last_actuals) >= 3 else 10.0
    last_known_date = ts_pd["ds"].max()
else:
    seed_lag1, seed_lag2, seed_roll3 = 10.0, 10.0, 10.0
    last_known_date = datetime.now() - timedelta(hours=1)

# ─────────────────────────────────────────────
# Generate future timestamps
# ─────────────────────────────────────────────
# Start prediction from the next hour after the dataset ends
start_forecast = pd.Timestamp(last_known_date) + pd.Timedelta(hours=1)
future_hours = pd.date_range(start=start_forecast, periods=horizon_days * 24, freq="h")
ar_holidays = holidays.CountryHoliday("AR")

# ─────────────────────────────────────────────
# Prophet base forecast (vectorised — fast)
# ─────────────────────────────────────────────
future_df = pd.DataFrame({"ds": future_hours})
with st.spinner("Running Prophet forecast…"):
    prophet_forecast = prophet_model.predict(future_df)

prophet_yhat   = prophet_forecast["yhat"].values
prophet_lower  = prophet_forecast["yhat_lower"].values
prophet_upper  = prophet_forecast["yhat_upper"].values
prophet_yhat   = np.clip(prophet_yhat, 0, None)
prophet_lower  = np.clip(prophet_lower, 0, None)
prophet_upper  = np.clip(prophet_upper, 0, None)

# ─────────────────────────────────────────────
# XGBoost recursive forecast
# ─────────────────────────────────────────────
# Feature list must match trainer.py. We'll check what the model expects.
XGB_FEATURES = [
    "backlog", "hour", "weekday", "day_of_month", "month",
    "is_month_end", "is_holiday", "is_weekend", "tipo_reporte_id",
    "y_lag_1", "y_lag_2", "y_rolling_mean_3",
    "avg_param_span_days",
]

# Check expected features from the booster
try:
    booster = xgb_model.get_booster()
    model_features = booster.feature_names
    if model_features:
        # Filter our list to only include features the model actually expects
        # (This handles the case where the model was trained WITHOUT avg_param_span_days)
        XGB_FEATURES = [f for f in XGB_FEATURES if f in model_features]
except Exception:
    pass # Fallback to trying the full list or letting XGBoost complain if mismatch

lag1, lag2, roll3 = seed_lag1, seed_lag2, seed_roll3
xgb_adjustments = []

with st.spinner("Running XGBoost recursive forecast…"):
    for i, ts in enumerate(future_hours):
        # Build feature row
        row = {
            "backlog":          backlog_seed,
            "hour":             ts.hour,
            "weekday":          ts.weekday(),
            "day_of_month":     ts.day,
            "month":            ts.month,
            "is_month_end":     int(ts.is_month_end),
            "is_holiday":       1 if ts in ar_holidays else 0,
            "is_weekend":       1 if ts.weekday() >= 5 else 0,
            "tipo_reporte_id":  int(report_type_id),
            "y_lag_1":          lag1,
            "y_lag_2":          lag2,
            "y_rolling_mean_3": roll3,
            "avg_param_span_days": 0.0, # Default for future assumption
        }
        
        # Select only the features the model expects
        # (If the model doesn't use avg_param_span_days, it won't be in XGB_FEATURES)
        # We need to ensure we pass a DataFrame with the exact columns
        # If XGB_FEATURES has features not in `row` (unlikely given our construction), we'd error.
        # If `row` has features not in XGB_FEATURES, we just drop them here.
        
        # Only pass columns that are in XGB_FEATURES
        row_subset = {k: v for k, v in row.items() if k in XGB_FEATURES}
        
        # Safe dataframe creation
        X = pd.DataFrame([row_subset])[XGB_FEATURES]
        adj = float(xgb_model.predict(X)[0])
        xgb_adjustments.append(adj)

        # Update lags recursively using the hybrid prediction
        hybrid_step = max(0.0, prophet_yhat[i] + adj)
        lag2 = lag1
        lag1 = hybrid_step
        roll3 = (roll3 * 2 + hybrid_step) / 3  # exponential-ish rolling mean

xgb_adj_arr = np.array(xgb_adjustments)
hybrid_yhat  = np.clip(prophet_yhat + xgb_adj_arr, 0, None)
hybrid_lower = np.clip(prophet_lower + xgb_adj_arr, 0, None)
hybrid_upper = np.clip(prophet_upper + xgb_adj_arr, 0, None)

# ─────────────────────────────────────────────
# Build result DataFrame
# ─────────────────────────────────────────────
forecast_df = pd.DataFrame({
    "ds":            future_hours,
    "prophet_yhat":  prophet_yhat,
    "hybrid_yhat":   hybrid_yhat,
    "hybrid_lower":  hybrid_lower,
    "hybrid_upper":  hybrid_upper,
})

# ─────────────────────────────────────────────
# Summary KPIs
# ─────────────────────────────────────────────
total_pred  = int(forecast_df["hybrid_yhat"].sum())
peak_hour   = forecast_df.loc[forecast_df["hybrid_yhat"].idxmax(), "ds"]
daily_agg   = forecast_df.groupby(forecast_df["ds"].dt.date)["hybrid_yhat"].sum()
peak_day    = daily_agg.idxmax()
peak_day_v  = int(daily_agg.max())

k1, k2, k3 = st.columns(3)
k1.metric(f"Total Predicted ({horizon_days}d)", f"{total_pred:,}")
k2.metric("Peak Day", f"{peak_day}  ({peak_day_v:,} reports)")
k3.metric("Peak Hour", peak_hour.strftime("%Y-%m-%d %H:00"))

st.divider()

# ─────────────────────────────────────────────
# Chart
# ─────────────────────────────────────────────
if granularity == "Daily":
    plot_df = forecast_df.copy()
    plot_df["date"] = plot_df["ds"].dt.date
    plot_df = plot_df.groupby("date").agg(
        hybrid_yhat=("hybrid_yhat", "sum"),
        hybrid_lower=("hybrid_lower", "sum"),
        hybrid_upper=("hybrid_upper", "sum"),
        prophet_yhat=("prophet_yhat", "sum"),
    ).reset_index()
    x_col = "date"
else:
    plot_df = forecast_df.copy()
    x_col = "ds"

fig = go.Figure()

# Uncertainty ribbon
fig.add_trace(go.Scatter(
    x=pd.concat([plot_df[x_col], plot_df[x_col][::-1]]),
    y=pd.concat([plot_df["hybrid_upper"], plot_df["hybrid_lower"][::-1]]),
    fill="toself",
    fillcolor="rgba(31,119,180,0.15)",
    line=dict(color="rgba(255,255,255,0)"),
    hoverinfo="skip",
    name="Uncertainty Band",
))

# Prophet base
fig.add_trace(go.Scatter(
    x=plot_df[x_col],
    y=plot_df["prophet_yhat"],
    name="Prophet Base",
    line=dict(color="#aaaaaa", dash="dash", width=1.5),
    hovertemplate="%{x}<br>Prophet: %{y:.1f}<extra></extra>",
))

# Hybrid forecast
fig.add_trace(go.Scatter(
    x=plot_df[x_col],
    y=plot_df["hybrid_yhat"],
    name="Hybrid Forecast",
    line=dict(color="#1f77b4", width=3),
    hovertemplate="%{x}<br>Hybrid: <b>%{y:.1f}</b><extra></extra>",
))

# Last-known actuals overlay (last 30 days)
if hourly_ts is not None:
    if granularity == "Daily":
        actuals_plot = ts_pd.copy()
        actuals_plot["date"] = pd.to_datetime(actuals_plot["ds"]).dt.date
        actuals_plot = actuals_plot.groupby("date")["y"].sum().reset_index()
        actuals_plot = actuals_plot[actuals_plot["date"] >= (pd.Timestamp(last_known_date) - timedelta(days=30)).date()]
        fig.add_trace(go.Scatter(
            x=actuals_plot["date"],
            y=actuals_plot["y"],
            name="Actuals (last 30d)",
            line=dict(color="#2ecc71", width=2),
            hovertemplate="%{x}<br>Actual: <b>%{y}</b><extra></extra>",
        ))
    else:
        actuals_plot = ts_pd[ts_pd["ds"] >= pd.Timestamp(last_known_date) - timedelta(days=7)].copy()
        fig.add_trace(go.Scatter(
            x=actuals_plot["ds"],
            y=actuals_plot["y"],
            name="Actuals (last 7d)",
            line=dict(color="#2ecc71", width=2),
            hovertemplate="%{x}<br>Actual: <b>%{y}</b><extra></extra>",
        ))

# Vertical line at start of forecast
fig.add_vline(
    x=pd.Timestamp(last_known_date).timestamp() * 1000,
    line_dash="dot",
    line_color="rgba(255,255,255,0.4)",
    annotation_text="Last Actuals",
    annotation_position="top right",
)

fig.update_layout(
    title=f"Hybrid Forecast — Next {horizon_days} Days ({granularity})",
    xaxis_title="Date",
    yaxis_title="Report Executions",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=480,
    hovermode="x unified",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#cccccc"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
)

st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# Daily summary table
# ─────────────────────────────────────────────
st.subheader("Daily Forecast Summary")
daily_summary = forecast_df.copy()
daily_summary["date"] = daily_summary["ds"].dt.date
daily_summary = (
    daily_summary.groupby("date")
    .agg(
        predicted_reports=("hybrid_yhat", "sum"),
        lower_bound=("hybrid_lower", "sum"),
        upper_bound=("hybrid_upper", "sum"),
        peak_hour=("hybrid_yhat", lambda s: int(s.idxmax() % 24)),
    )
    .reset_index()
)
daily_summary["predicted_reports"] = daily_summary["predicted_reports"].round(1)
daily_summary["lower_bound"]       = daily_summary["lower_bound"].round(1)
daily_summary["upper_bound"]       = daily_summary["upper_bound"].round(1)
daily_summary.columns = ["Date", "Predicted Reports", "Lower Bound", "Upper Bound", "Peak Hour"]

st.dataframe(daily_summary, use_container_width=True, hide_index=True)

# Download
csv = daily_summary.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download Daily Forecast CSV",
    data=csv,
    file_name=f"forecast_{horizon_days}d.csv",
    mime="text/csv",
)
