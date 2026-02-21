import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from prophet.serialize import model_from_json
import plotly.graph_objects as go
from datetime import timedelta
import json
import holidays
import os

st.set_page_config(page_title="Future Horizon", layout="wide")

st.title("🔭 Future Horizon: 15-Day Recursive Projection")
st.markdown("""
    This module performs **auto-regressive multi-step forecasting** using the new log-transformed XGBoost models. 
    It predicts one hour at a time and feeds the result back as input for the next hour to capture the recursive operational dynamics.
""")

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    try:
        with open("prophet_model.json", "r") as f:
            prophet = model_from_json(f.read())
            
        xgb_p50 = xgb.XGBRegressor()
        xgb_p50.load_model("xgboost_model_p50.json")

        xgb_p10 = xgb.XGBRegressor()
        xgb_p10.load_model("xgboost_model_p10.json")

        xgb_p90 = xgb.XGBRegressor()
        xgb_p90.load_model("xgboost_model_p90.json")
            
        with open("label_mapping.json", "r") as f:
            label_map = json.load(f)

        with open("feature_list.json", "r") as f:
            feature_list = json.load(f)
            
        return prophet, (xgb_p10, xgb_p50, xgb_p90), label_map, feature_list
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

prophet, xgb_models, label_map, feature_list = load_models()

if prophet is None or xgb_models is None:
    st.warning("Prediction models not detected. Please run training first.")
    st.stop()

xgb_p10, xgb_p50, xgb_p90 = xgb_models

# --- SIDEBAR: OPERATIONAL CONTEXT ---
with st.sidebar:
    st.header("🔮 Horizon Settings")
    forecast_days = st.slider("Forecast Horizon (Days)", 1, 15, 7)
    periods = forecast_days * 24
    
    st.divider()
    st.header("🛠️ Scenario Setup")
    backlog_start = st.number_input("Starting Backlog", min_value=0, value=10)
    
    report_types = list(label_map.values())
    selected_type = st.selectbox("Industry Profile", report_types)
    report_type_id = [k for k, v in label_map.items() if v == selected_type][0]

# --- RECURSIVE FORECASTING ENGINE ---
hourly_ts = st.session_state.get("hourly_ts")
df_raw = st.session_state.get("df_raw")

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

if hourly_ts is not None:
    from utils.data_processing import enrich_hourly_ts_with_features
    # Check for full features
    if 'hour_sin' not in hourly_ts.columns:
        with st.spinner("Preparing historical state..."):
            hourly_ts = enrich_hourly_ts_with_features(hourly_ts, df_raw)
            st.session_state["hourly_ts"] = hourly_ts

    df_pd = hourly_ts.to_pandas()
    rename_map = {'timestamp_hour': 'ds', 'report_count': 'y'}
    df_pd = df_pd.rename(columns={k: v for k, v in rename_map.items() if k in df_pd.columns})
    
    last_dt = df_pd['ds'].iloc[-1]
    last_y = df_pd['y'].iloc[-1]
    
    # 1. Base Prophet Forecast (for seasonal reference)
    future_dates = pd.date_range(start=last_dt + timedelta(hours=1), periods=periods, freq='h')
    future_df = pd.DataFrame({'ds': future_dates})
    p_forecast = prophet.predict(future_df)
    
    # 2. Recursive XGBoost Prediction
    with st.spinner(f"Simulating {periods} hours recursively..."):
        history = df_pd.tail(168).copy() # Use last week for initial lags
        results = []
        ar_holidays = holidays.CountryHoliday('AR')
        
        # Iterative prediction
        for i, current_dt in enumerate(future_dates):
            # Build current feature row
            weekday = current_dt.weekday()
            is_peak = 1 if 8 <= current_dt.hour <= 18 else 0
            is_weekend = 1 if weekday >= 5 else 0
            is_holiday = 1 if current_dt in ar_holidays else 0
            
            # Extract Lags from history
            y_L1 = history['y'].iloc[-1]
            y_L2 = history['y'].iloc[-2] if len(history) > 1 else y_L1
            y_L24 = history['y'].iloc[-24] if len(history) >= 24 else y_L1
            y_L168 = history['y'].iloc[-168] if len(history) >= 168 else y_L1
            
            # Rolling (simplified for recursive)
            y_roll3 = history['y'].tail(3).mean()
            y_roll24 = history['y'].tail(24).mean()
            y_roll168 = history['y'].tail(168).mean()
            
            # Interaction
            row = {
                "backlog": backlog_start, # Simplified: stays constant or moves towards 0? 
                "hour": current_dt.hour,
                "weekday": weekday,
                "day_of_month": current_dt.day,
                "month": current_dt.month,
                "is_month_end": int(pd.Timestamp(current_dt).is_month_end),
                "is_holiday": is_holiday,
                "is_weekend": is_weekend,
                "is_peak_hour": is_peak,
                "weekend_x_peak": is_weekend * is_peak,
                "holiday_x_peak": is_holiday * is_peak,
                "weekend_or_holiday_x_peak": int((is_weekend or is_holiday) and is_peak),
                "hour_sin": np.sin(2 * np.pi * current_dt.hour / 24),
                "hour_cos": np.cos(2 * np.pi * current_dt.hour / 24),
                "weekday_sin": np.sin(2 * np.pi * weekday / 7),
                "weekday_cos": np.cos(2 * np.pi * weekday / 7),
                "month_sin": np.sin(2 * np.pi * current_dt.month / 12),
                "month_cos": np.cos(2 * np.pi * current_dt.month / 12),
                "y_lag_1": float(y_L1),
                "y_lag_2": float(y_L2),
                "y_lag_3": float(y_L2),
                "y_lag_24": float(y_L24),
                "y_lag_168": float(y_L168),
                "y_rolling_mean_3": float(y_roll3),
                "y_rolling_mean_24": float(y_roll24),
                "y_rolling_mean_168": float(y_roll168),
                "y_rolling_std_3": float(history['y'].tail(3).std()),
                "y_rolling_std_24": float(history['y'].tail(24).std()),
                "log_y_lag_1": np.log1p(y_L1),
                "log_y_lag_24": np.log1p(y_L24),
                "log_y_lag_168": np.log1p(y_L168),
                "log_y_rolling_mean_3": np.log1p(y_roll3),
                "log_y_rolling_mean_24": np.log1p(y_roll24),
                "log_y_rolling_mean_168": np.log1p(y_roll168),
                "tipo_reporte_id": int(report_type_id),
                "avg_param_span_days": 0.0
            }
            # Fourier placeholders (recursive t offset would be better, but staying stationary for now)
            for name, p in {"daily": 24, "weekly": 168}.items():
                for k in range(1, 4):
                    row[f"fourier_{name}_sin_{k}"] = 0.0
                    row[f"fourier_{name}_cos_{k}"] = 0.0

            X = pd.DataFrame([row]).reindex(columns=feature_list, fill_value=0)
            
            # Predict
            pred_p50 = np.expm1(xgb_p50.predict(X)[0])
            pred_p10 = np.expm1(xgb_p10.predict(X)[0])
            pred_p90 = np.expm1(xgb_p90.predict(X)[0])
            
            # Ensure non-negative and logical bounds
            pred_p50 = max(0, pred_p50)
            pred_p10 = min(pred_p50, max(0, pred_p10))
            pred_p90 = max(pred_p50, pred_p90)
            
            results.append({
                'ds': current_dt,
                'yhat': pred_p50,
                'p10': pred_p10,
                'p90': pred_p90
            })
            
            # Update history for next iteration
            new_row = pd.DataFrame({'ds': [current_dt], 'y': [pred_p50]})
            history = pd.concat([history, new_row]).tail(168)
        
    res_df = pd.DataFrame(results)

    # --- PLOTTING ---
    fig = go.Figure()
    
    # Historical Context (Last 3 days)
    hist_show = df_pd.tail(72)
    fig.add_trace(go.Scatter(x=hist_show['ds'], y=hist_show['y'], name="Historical Load", line=dict(color='white', width=1.5)))
    
    # Future Forecast (p50)
    fig.add_trace(go.Scatter(x=res_df['ds'], y=res_df['yhat'], name="XGBoost Projection (p50)", line=dict(color='#1f77b4', width=3)))
    
    # Quantile Ribbon
    fig.add_trace(go.Scatter(
        x=pd.concat([res_df['ds'], res_df['ds'][::-1]]),
        y=pd.concat([res_df['p90'], res_df['p10'][::-1]]),
        fill='toself',
        fillcolor='rgba(31,119,180,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name="Uncertainty Interval (p10-p90)"
    ))

    fig.update_layout(
        title=f"Resource Demand Projection: Next {forecast_days} Days",
        template="plotly_dark",
        height=550,
        legend=dict(orientation="h", y=1.1),
        xaxis_title="Time",
        yaxis_title="Estimated Reports/Hour"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Projection Summary")
        st.metric("Peak Expected Load", f"{res_df['yhat'].max():.2f}")
        st.metric("Average Hourly Load", f"{res_df['yhat'].mean():.2f}")
    with c2:
        st.info("""
            **Note on Recursive Forecasting**: 
            Predictions become less certain as the horizon expands (error propagation). 
            The shaded area represents the 80% prediction interval (p10 to p90).
        """)

else:
    st.warning("Please load data in the Home page first.")
