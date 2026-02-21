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

st.title("Future Horizon: Detailed Recursive Projection")
st.markdown("""
    *Auto-regressive multi-step forecasting using log-XGBoost. This model captures recursive operational dynamics by feeding predictions back into the feature set.*
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
    st.header("Horizon Settings")
    forecast_days = st.slider("Forecast Horizon (Days)", 1, 15, 7)
    periods = forecast_days * 24
    
    st.divider()
    st.header("Scenario Setup")
    backlog_start = st.number_input("Starting Backlog", min_value=0, value=10)
    
    report_types = list(label_map.values())
    selected_type = st.selectbox("Industry Profile", report_types)
    report_type_id = [k for k, v in label_map.items() if v == selected_type][0]
    
    st.divider()
    st.header("Operational Rules")
    rtw_multiplier = st.slider("Return-to-Work Impact (+%)", 0, 50, 10, help="Extra load accumulated during weekends/holidays that hits on the first workday.") / 100.0

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
    
    first_ds = df_pd['ds'].min()
    first_ds = df_pd['ds'].min()
    last_dt = df_pd['ds'].iloc[-1]
    
    # 1. Base Prophet Forecast (for seasonal reference/components)
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
            
            # Post-holiday spike logic for simulation
            # Check if yesterday was a holiday or weekend
            yesterday_dt = current_dt - timedelta(days=24) # Simple 24h lookback
            was_off = 1 if (yesterday_dt in ar_holidays or yesterday_dt.weekday() >= 5) else 0
            is_post_holiday = 1 if (weekday < 5 and is_holiday == 0 and was_off == 1) else 0
            
            # Extract Lags from history
            y_L1 = history['y'].iloc[-1]
            y_L24 = history['y'].iloc[-24] if len(history) >= 24 else y_L1
            y_L168 = history['y'].iloc[-168] if len(history) >= 168 else y_L1
            
            # Rolling
            y_roll3 = history['y'].tail(3).mean()
            y_roll24 = history['y'].tail(24).mean()
            y_roll168 = history['y'].tail(168).mean()
            
            row = {
                "backlog": backlog_start, 
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
                "y_lag_24": float(y_L24),
                "y_lag_168": float(y_L168),
                "log_y_lag_1": np.log1p(y_L1),
                "log_y_lag_24": np.log1p(y_L24),
                "log_y_lag_168": np.log1p(y_L168),
                "tipo_reporte_id": int(report_type_id),
                "days_since_start": (current_dt - first_ds).days,
                "is_post_holiday": is_post_holiday,
            }

            X = pd.DataFrame([row]).reindex(columns=feature_list, fill_value=0)
            
            # Predict
            pred_p50 = np.expm1(xgb_p50.predict(X)[0])
            pred_p10 = np.expm1(xgb_p10.predict(X)[0])
            pred_p90 = np.expm1(xgb_p90.predict(X)[0])
            
            # Apply Return-to-Work multiplier if it's a post-holiday spike
            if is_post_holiday == 1:
                pred_p50 *= (1.0 + rtw_multiplier)
                pred_p10 *= (1.0 + rtw_multiplier)
                pred_p90 *= (1.0 + rtw_multiplier)

            pred_p50 = max(0, pred_p50)
            pred_p10 = min(pred_p50, max(0, pred_p10))
            pred_p90 = max(pred_p50, pred_p90)
            
            results.append({
                'ds': current_dt,
                'yhat': pred_p50,
                'p10': pred_p10,
                'p90': pred_p90
            })
            
            # Update history
            new_row = pd.DataFrame({'ds': [current_dt], 'y': [pred_p50]})
            history = pd.concat([history, new_row]).tail(168)
        
    # --- AGGREGATION TO DAILY ---
    res_df = pd.DataFrame(results)
    res_df['date'] = res_df['ds'].dt.date
    res_daily = res_df.groupby('date').agg({
        'yhat': 'sum',
        'p10': 'sum',
        'p90': 'sum'
    }).reset_index()
    res_daily['ds'] = pd.to_datetime(res_daily['date'])

    hist_daily = df_pd.copy()
    hist_daily['date'] = hist_daily['ds'].dt.date
    hist_daily = hist_daily.groupby('date').agg({'y': 'sum'}).reset_index()
    hist_daily['ds'] = pd.to_datetime(hist_daily['date'])

    # --- YEAR-OVER-YEAR COMPARISON DATA ---
    # Find data from same period last year (approx 364 days to align weekdays)
    yoy_dates = res_daily['ds'] - timedelta(days=364)
    yoy_df = hist_daily[hist_daily['ds'].isin(yoy_dates)].copy()
    # Align dates for plotting
    yoy_df['plot_ds'] = yoy_df['ds'] + timedelta(days=364)
    yoy_year = (res_daily['ds'].dt.year.min()) - 1

    # Historical stats for context
    hist_max = hist_daily['y'].max()
    hist_p90 = hist_daily['y'].quantile(0.90)
    # Comparison against last complete year available
    comp_year = 2025 if (df_pd['ds'].dt.year == 2025).any() else 2024
    hist_avg_comp = hist_daily[hist_daily['ds'].dt.year == comp_year]['y'].mean()

    # --- MAIN VISUALIZATION ---
    st.subheader("Daily Impact Projection")
    
    fig = go.Figure()
    
    # Historical Context (Last 14 days)
    hist_show = hist_daily.tail(14)
    fig.add_trace(go.Scatter(
        x=hist_show['ds'], y=hist_show['y'], 
        name="Recent History", 
        line=dict(color='rgba(255,255,255,0.3)', width=1.5, dash='dot'),
        opacity=0.6,
        line_shape='spline'
    ))
    
    # Year-Over-Year Comparison (Context)
    if not yoy_df.empty:
        fig.add_trace(go.Scatter(
            x=yoy_df['plot_ds'], y=yoy_df['y'],
            name=f"Same Period {yoy_year}",
            line=dict(color='#ff00ff', width=2, dash='dash'),
            opacity=0.5,
            line_shape='spline',
            hovertemplate=f"{yoy_year} Equivalent: %{{y:.1f}}<extra></extra>"
        ))

    # Uncertainty Interval
    fig.add_trace(go.Scatter(
        x=pd.concat([res_daily['ds'], res_daily['ds'][::-1]]),
        y=pd.concat([res_daily['p90'], res_daily['p10'][::-1]]),
        fill='toself',
        fillcolor='rgba(0, 242, 254, 0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        name="Uncertainty Range (p10-p90)",
        hoverinfo='skip'
    ))

    # Projection (p50)
    fig.add_trace(go.Scatter(
        x=res_daily['ds'], y=res_daily['yhat'], 
        name="Recursive Projection (p50)", 
        line=dict(color='#00f2fe', width=4),
        line_shape='spline',
        hovertemplate="<b>Date: %{x}</b><br>Projected Total: %{y:.1f}<extra></extra>"
    ))
    
    # Capacity Markers (Horizontal Reference Lines)
    fig.add_hline(y=hist_max, line_dash="dash", line_color="rgba(255,0,0,0.5)", 
                  annotation_text="Historical Max", annotation_position="top left")
    fig.add_hline(y=hist_p90, line_dash="dot", line_color="rgba(255,165,0,0.5)", 
                  annotation_text="90th Percentile", annotation_position="bottom left")

    # Highlight Peaks
    top_days = res_daily.sort_values('yhat', ascending=False).head(3)
    fig.add_trace(go.Scatter(
        x=top_days['ds'], y=top_days['yhat'],
        mode='markers',
        marker=dict(color='#00f2fe', size=12, line=dict(color='white', width=2)),
        name="Projected Peak Values"
    ))

    fig.update_layout(
        title=f"Daily Report Volume Projection: Next {forecast_days} Days",
        template="plotly_dark",
        height=550,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.1, x=0),
        xaxis_title="Calendar",
        yaxis_title="Total Daily Reports",
        margin=dict(r=0, l=0)
    )
    st.plotly_chart(fig, width='stretch')

    # --- METRICS BARS ---
    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Max Daily Load", f"{res_daily['yhat'].max():.1f}", 
                  help="The highest predicted total load for a single day.")
    with c2:
        st.metric("Stress Test Peak", f"{res_daily['p90'].max():.1f}", 
                  help="Conservative upper bound (90th percentile).")
    with c3:
        avg_v = res_daily['yhat'].mean()
        # Compare against baseline year avg
        delta_avg = ((avg_v - hist_avg_comp) / hist_avg_comp * 100) if hist_avg_comp > 0 else 0
        st.metric("Avg Daily Load", f"{avg_v:.1f}", 
                  delta=f"{delta_avg:+.1f}% vs {comp_year} Avg",
                  delta_color="inverse")
    with c4:
        total_v = res_daily['yhat'].sum()
        st.metric("Total Horizon Volume", f"{int(total_v)}", 
                  help="Total cumulative reports expected in the forecast window.")

    # --- SEASONALITY COMPONENTS ---
    st.subheader("Structural Components")
    st.info("Aggregated daily impact of structural cycles.")
    
    # Aggregate prophet forecast to daily average
    p_daily = p_forecast.copy()
    p_daily['date'] = p_daily['ds'].dt.date
    p_daily = p_daily.groupby('date').agg({
        'trend': 'mean',
        'weekly': 'mean',
        'daily': 'mean'
    }).reset_index()
    p_daily['ds'] = pd.to_datetime(p_daily['date'])

    comp_fig = go.Figure()
    # Trend (Line)
    comp_fig.add_trace(go.Scatter(x=p_daily['ds'], y=p_daily['trend'], name="Underlying Trend", line=dict(color='#ff9ff3', width=2), line_shape='spline'))
    # Weekly (Area)
    comp_fig.add_trace(go.Scatter(x=p_daily['ds'], y=p_daily['weekly'], name="Weekly Variance", fill='tozeroy', line=dict(color='#00d2d3', width=1.5), line_shape='spline'))
    
    comp_fig.update_layout(
        height=350,
        template="plotly_dark",
        legend=dict(orientation="h", y=1.2),
        margin=dict(r=0, l=0, b=0),
        yaxis_title="Relative Impact"
    )
    st.plotly_chart(comp_fig, width='stretch')

else:
    st.warning("Please load data in the Home page first.")
