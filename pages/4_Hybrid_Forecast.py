import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from prophet.serialize import model_from_json
import json
import plotly.graph_objects as go
from datetime import datetime, timedelta
import holidays
import os

# Page Config
st.set_page_config(page_title="Forecasting Analysis", layout="wide")

# Inject Custom CSS
try:
    with open("assets/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

# --- LOAD MODELS & METRICS ---
@st.cache_resource
def load_all_resources():
    try:
        # Prophet (legacy/comparison)
        with open('prophet_model.json', 'r') as f:
            m = model_from_json(f.read())
        
        # New Quantile Models
        xgb_p50 = xgb.XGBRegressor()
        xgb_p50.load_model('xgboost_model_p50.json')
        
        xgb_p10 = xgb.XGBRegressor()
        xgb_p10.load_model('xgboost_model_p10.json')
        
        xgb_p90 = xgb.XGBRegressor()
        xgb_p90.load_model('xgboost_model_p90.json')
        
        with open('label_mapping.json', 'r') as f:
            label_map = json.load(f)
            
        with open('metrics.json', 'r') as f:
            perf_metrics = json.load(f)

        with open('feature_list.json', 'r') as f:
            feature_list = json.load(f)
            
        return m, (xgb_p10, xgb_p50, xgb_p90), label_map, perf_metrics, feature_list
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None

m, xgb_models, label_map, perf_metrics, feature_list = load_all_resources()

# --- HEADER ---
st.title("Advanced Forecasting: XGBoost + Quantile Bands")
st.markdown("""
    *High-precision forecasting using log-transformed XGBoost with uncertainty intervals (p10, p50, p90).*
""")

if xgb_models is None:
    st.warning("Trained models not detected. Please run the training process (`trainer.py`) to enable this page.")
    st.stop()

xgb_p10, xgb_p50, xgb_p90 = xgb_models

# --- SIDEBAR: INPUTS ---
with st.sidebar:
    st.header("Operational Parameters")
    target_date = st.date_input("Prediction Date", datetime.now() + timedelta(days=1))
    target_hour = st.slider("Hour of Day", 0, 23, 12)
    backlog_input = st.number_input("Current Backlog (Pending Reports)", min_value=0, value=10)

    st.divider()
    st.subheader("Recent Memory (Lags)")
    lag1 = st.number_input("Load at T-1", min_value=0, value=15)
    lag2 = st.number_input("Load at T-2", min_value=0, value=12)
    lag24 = st.number_input("Load at T-24 (Yesterday)", min_value=0, value=20)
    lag168 = st.number_input("Load at T-168 (Last Week)", min_value=0, value=18)

    report_types = list(label_map.values())
    selected_type = st.selectbox("Load Profile", report_types)

# --- INFERENCE ---
dt = datetime.combine(target_date, datetime.min.time()).replace(hour=target_hour)
weekday = dt.weekday()
day_of_month = dt.day
month = dt.month
is_month_end = int(pd.Timestamp(dt).is_month_end)
is_weekend = 1 if weekday >= 5 else 0
ar_holidays = holidays.CountryHoliday('AR')
is_holiday = 1 if dt in ar_holidays else 0
is_peak_hour = 1 if 8 <= target_hour <= 18 else 0
report_type_id = [k for k, v in label_map.items() if v == selected_type][0]

# Build feature vector for XGBoost (matching trainer.py)
row = {
    "backlog": backlog_input,
    "hour": target_hour,
    "weekday": weekday,
    "day_of_month": day_of_month,
    "month": month,
    "is_month_end": is_month_end,
    "is_holiday": is_holiday,
    "is_weekend": is_weekend,
    "is_peak_hour": is_peak_hour,
    "weekend_x_peak": is_weekend * is_peak_hour,
    "holiday_x_peak": is_holiday * is_peak_hour,
    "weekend_or_holiday_x_peak": int((is_weekend or is_holiday) and is_peak_hour),
    "hour_sin": np.sin(2 * np.pi * target_hour / 24),
    "hour_cos": np.cos(2 * np.pi * target_hour / 24),
    "weekday_sin": np.sin(2 * np.pi * weekday / 7),
    "weekday_cos": np.cos(2 * np.pi * weekday / 7),
    "month_sin": np.sin(2 * np.pi * month / 12),
    "month_cos": np.cos(2 * np.pi * month / 12),
    "y_lag_1": float(lag1),
    "y_lag_2": float(lag2),
    "y_lag_3": float(lag2),
    "y_lag_24": float(lag24),
    "y_lag_168": float(lag168),
    "y_rolling_mean_3": (lag1 + lag2 + 10) / 3,
    "y_rolling_mean_24": float(lag24),
    "y_rolling_mean_168": float(lag168),
    "y_rolling_std_3": np.std([lag1, lag2, 10]),
    "y_rolling_std_24": 0.0,
    "log_y_lag_1": np.log1p(lag1),
    "log_y_lag_24": np.log1p(lag24),
    "log_y_lag_168": np.log1p(lag168),
    "log_y_rolling_mean_3": np.log1p((lag1 + lag2 + 10) / 3),
    "log_y_rolling_mean_24": np.log1p(lag24),
    "log_y_rolling_mean_168": np.log1p(lag168),
    "tipo_reporte_id": int(report_type_id),
    "avg_param_span_days": 0.0
}
# Fourier approximations
for name, period in {"daily": 24, "weekly": 24 * 7}.items():
    for k in range(1, 4):
        row[f"fourier_{name}_sin_{k}"] = 0.0
        row[f"fourier_{name}_cos_{k}"] = 0.0

xgb_features = pd.DataFrame([row]).reindex(columns=feature_list, fill_value=0)

# Predictions
xgb_pred = np.expm1(xgb_p50.predict(xgb_features)[0])
xgb_p10_val = np.expm1(xgb_p10.predict(xgb_features)[0])
xgb_p90_val = np.expm1(xgb_p90.predict(xgb_features)[0])

# Enforce logic
xgb_p10_val = min(xgb_p10_val, xgb_pred)
xgb_p90_val = max(xgb_p90_val, xgb_pred)

# --- TABS ---
tab1, tab2, tab3 = st.tabs([
    "Performance Dashboard", 
    "Simulation & Intervals", 
    "Model Insights"
])

with tab1:
    st.subheader("Model Performance")
    hourly_ts = st.session_state.get("hourly_ts")
    df_raw = st.session_state.get("df_raw")

    # Fallback logic
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

    if hourly_ts is not None:
        from utils.data_processing import enrich_hourly_ts_with_features
        import polars as pl
        
        if 'hour_sin' not in hourly_ts.columns:
            with st.spinner("Enriching data with new feature set..."):
                hourly_ts = enrich_hourly_ts_with_features(hourly_ts, df_raw)
                st.session_state["hourly_ts"] = hourly_ts

        df_pd = hourly_ts.to_pandas()
        rename_map = {'timestamp_hour': 'ds', 'report_count': 'y'}
        df_pd = df_pd.rename(columns={k: v for k, v in rename_map.items() if k in df_pd.columns})
            
        split_idx = int(len(df_pd) * 0.8)
        test_df = df_pd.iloc[split_idx:].copy()
        
        test_xgb_features = test_df[feature_list]
        test_df['y_pred_xgb'] = np.expm1(xgb_p50.predict(test_xgb_features)).clip(min=0)

        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{perf_metrics['mae']:.4f}")
        c2.metric("RMSE", f"{perf_metrics['rmse']:.4f}")
        c3.metric("MASE", f"{perf_metrics['mase']:.4f}")
            
        st.divider()
        st.subheader("Latest Predictions")
        st.dataframe(test_df[['ds', 'y', 'y_pred_xgb']].tail(15), use_container_width=True)
    else:
        st.warning("Please load data in the Home page first.")

with tab2:
    st.subheader("Inference Visualization")
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Predicted (p50)", f"{xgb_pred:.2f}")
    mc2.metric("Lower Bound (p10)", f"{xgb_p10_val:.2f}")
    mc3.metric("Upper Bound (p90)", f"{xgb_p90_val:.2f}")

    st.divider()
    fig = go.Figure()
    if hourly_ts is not None:
        context_df = df_pd[df_pd['ds'] >= dt - timedelta(hours=48)].copy()
        fig.add_trace(go.Scatter(x=context_df['ds'], y=context_df['y'], name="Actuals", line=dict(color='gray', width=1)))
        
    # Prediction point
    fig.add_trace(go.Scatter(x=[dt], y=[xgb_pred], mode='markers', marker=dict(color='red', size=12), name="p50"))
    # Ribbon-like bar for interval
    fig.add_trace(go.Scatter(x=[dt, dt], y=[xgb_p10_val, xgb_p90_val], mode='lines', line=dict(color='rgba(255,0,0,0.3)', width=10), name="Uncertainty [p10-p90]"))

    fig.update_layout(height=450, template="plotly_dark", title="Report Load Estimation")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Model Interpretability")
    st.info("The new model implementation uses logs and quantile regression to handle workload spikes.")
    with st.expander("Active Features"):
        st.write(feature_list)
    
    st.latex(r"\log(y+1)_{pred} = \text{XGBoost}(X_{operational})")
    st.latex(r"y_{final} = \exp(\log(y+1)_{pred}) - 1")
