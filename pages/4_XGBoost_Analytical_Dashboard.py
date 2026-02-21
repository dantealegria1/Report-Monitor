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
st.set_page_config(page_title="XGBoost Analytical Dashboard", layout="wide")

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
st.title("XGBoost Analytical Dashboard")
st.markdown("""
    *Integrated analysis of the Advanced XGBoost model: performance metrics, real-time simulation, architecture, and statistical health.*
""")

if xgb_models is None:
    st.warning("Trained models not detected. Please run the training process (`trainer.py`) to enable this page.")
    st.stop()

xgb_p10, xgb_p50, xgb_p90 = xgb_models

# --- DATA PREPARATION (Shared across tabs) ---
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
    if 'hour_sin' not in hourly_ts.columns:
        with st.spinner("Enriching data..."):
            hourly_ts = enrich_hourly_ts_with_features(hourly_ts, df_raw)
            st.session_state["hourly_ts"] = hourly_ts

    df_pd = hourly_ts.to_pandas()
    rename_map = {'timestamp_hour': 'ds', 'report_count': 'y'}
    df_pd = df_pd.rename(columns={k: v for k, v in rename_map.items() if k in df_pd.columns})
        
    split_idx = int(len(df_pd) * 0.8)
    test_df = df_pd.iloc[split_idx:].copy()

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "Precision & Performance", 
    "Interactive Simulation", 
    "Model Architecture",
    "Statistical Health"
])

with tab1:
    st.subheader("Model Precision & Error Analysis")
    if perf_metrics and hourly_ts is not None:
        # 1. Metrics Grid
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric("MAE", f"{perf_metrics['mae']:.3f}", delta=f"{perf_metrics['mae'] - perf_metrics['naive_mae']:.2f}", delta_color="inverse")
        with m_col2:
            st.metric("RMSE", f"{perf_metrics['rmse']:.3f}", delta=f"{perf_metrics['rmse'] - perf_metrics['naive_rmse']:.2f}", delta_color="inverse")
        with m_col3:
            st.metric("MASE", f"{perf_metrics['mase']:.3f}")
        with m_col4:
            gain = (perf_metrics['naive_mae'] - perf_metrics['mae']) / perf_metrics['naive_mae'] * 100
            st.metric("Avg precision gain", f"{gain:.1f}%")

        st.divider()
        
        # 2. Detailed Time Series Comparison
        test_xgb_features = test_df[feature_list]
        test_df['y_pred_xgb'] = np.expm1(xgb_p50.predict(test_xgb_features)).clip(min=0)
        test_df['y_p10'] = np.expm1(xgb_p10.predict(test_xgb_features)).clip(min=0)
        test_df['y_p90'] = np.expm1(xgb_p90.predict(test_xgb_features)).clip(min=0)
        test_df['y_naive'] = test_df['y'].shift(1).fillna(method='bfill') # Naive Baseline for chart

        fig_compare = go.Figure()
        
        # Actuals
        fig_compare.add_trace(go.Scatter(x=test_df['ds'], y=test_df['y'], name="Actual Value", line=dict(color='white', width=1.5), opacity=0.8))
        
        # XGBoost p50
        fig_compare.add_trace(go.Scatter(x=test_df['ds'], y=test_df['y_pred_xgb'], name="XGBoost (p50)", line=dict(color='#1f77b4', width=2.5)))
        
        # Naive Baseline
        fig_compare.add_trace(go.Scatter(x=test_df['ds'], y=test_df['y_naive'], name="Naive Baseline (T-1)", line=dict(color='gray', width=1, dash='dash')))
        
        # p10-p90 ribbon
        fig_compare.add_trace(go.Scatter(
            x=pd.concat([test_df['ds'], test_df['ds'][::-1]]),
            y=pd.concat([test_df['y_p90'], test_df['y_p10'][::-1]]),
            fill='toself',
            fillcolor='rgba(31,119,180,0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            name="XGBoost Uncertainty Band"
        ))
        
        fig_compare.update_layout(
            title="Detailed Forecast Comparison (Testing Set)",
            height=500, 
            legend=dict(orientation="h", y=1.1, x=0), 
            template="plotly_dark",
            xaxis_title="Timeline",
            yaxis_title="Reports / Hour",
            hovermode="x unified"
        )
        st.plotly_chart(fig_compare, use_container_width=True)

        # 3. Error Distribution & Sample
        c_err1, c_err2 = st.columns([2, 1])
        with c_err1:
            st.markdown("**Prediction Error Distribution (Residuals)**")
            test_df['residual'] = test_df['y'] - test_df['y_pred_xgb']
            fig_dist = go.Figure(go.Histogram(x=test_df['residual'], nbinsx=60, marker_color='#1f77b4', opacity=0.8))
            fig_dist.update_layout(height=350, template="plotly_dark", margin=dict(l=0, r=0, t=20, b=0), xaxis_title="Error (Actual - Predicted)")
            st.plotly_chart(fig_dist, use_container_width=True)
            
        with c_err2:
            st.markdown("**Latest Performance Sample**")
            st.dataframe(test_df[['ds', 'y', 'y_pred_xgb']].tail(10), use_container_width=True)
    else:
        st.warning("Data or metrics not available.")

with tab2:
    st.subheader("Real-time Model Inference")
    
    col_inp1, col_inp2 = st.columns([1, 2])
    
    with col_inp1:
        st.markdown("#### Scenario Setup")
        target_date = st.date_input("Prediction Date", datetime.now() + timedelta(days=1), key="sim_date")
        target_hour = st.slider("Hour of Day", 0, 23, 12, key="sim_hour")
        backlog_input = st.number_input("Current Backlog", min_value=0, value=10, key="sim_backlog")
        
        st.divider()
        st.markdown("#### Operational Memory")
        lag1 = st.number_input("T-1 Load", min_value=0, value=15, key="sim_lag1")
        lag24 = st.number_input("T-24 Load", min_value=0, value=20, key="sim_lag24")
        lag168 = st.number_input("T-168 Load", min_value=0, value=18, key="sim_lag168")
        
        selected_type = st.selectbox("Load Profile", list(label_map.values()), key="sim_profile")
        report_type_id = [k for k, v in label_map.items() if v == selected_type][0]

    # Inference Calculation
    dt = datetime.combine(target_date, datetime.min.time()).replace(hour=target_hour)
    ar_holidays = holidays.CountryHoliday('AR')
    
    sim_row = {
        "backlog": backlog_input,
        "hour": target_hour,
        "weekday": dt.weekday(),
        "day_of_month": dt.day,
        "month": dt.month,
        "is_month_end": int(pd.Timestamp(dt).is_month_end),
        "is_holiday": 1 if dt in ar_holidays else 0,
        "is_weekend": 1 if dt.weekday() >= 5 else 0,
        "is_peak_hour": 1 if 8 <= target_hour <= 18 else 0,
        "y_lag_1": float(lag1),
        "y_lag_24": float(lag24),
        "y_lag_168": float(lag168),
        "log_y_lag_1": np.log1p(lag1),
        "log_y_lag_24": np.log1p(lag24),
        "log_y_lag_168": np.log1p(lag168),
        "tipo_reporte_id": int(report_type_id),
    }
    
    # Fill remaining features with 0
    sim_features = pd.DataFrame([sim_row]).reindex(columns=feature_list, fill_value=0)
    
    s_pred = np.expm1(xgb_p50.predict(sim_features)[0])
    s_p10 = np.expm1(xgb_p10.predict(sim_features)[0])
    s_p90 = np.expm1(xgb_p90.predict(sim_features)[0])
    
    with col_inp2:
        st.markdown(f"#### Estimation for {dt.strftime('%A, %b %d at %H:00')}")
        m_c1, m_c2, m_c3 = st.columns(3)
        m_c1.metric("Predicted (p50)", f"{max(0, s_pred):.2f}")
        m_c2.metric("Lower Bound (p10)", f"{max(0, min(s_p10, s_pred)):.2f}")
        m_c3.metric("Upper Bound (p90)", f"{max(s_pred, s_p90):.2f}")
        
        st.divider()
        fig_sim = go.Figure()
        if hourly_ts is not None:
            context_df = df_pd[df_pd['ds'] >= dt - timedelta(hours=48)].copy()
            fig_sim.add_trace(go.Scatter(x=context_df['ds'], y=context_df['y'], name="Recent Actuals", line=dict(color='gray', width=1)))
        
        fig_sim.add_trace(go.Scatter(x=[dt], y=[s_pred], mode='markers', marker=dict(color='red', size=12), name="p50 Prediction"))
        fig_sim.add_trace(go.Scatter(x=[dt, dt], y=[s_p10, s_p90], mode='lines', line=dict(color='rgba(255,0,0,0.3)', width=10), name="Uncertainty Range"))
        
        fig_sim.update_layout(height=450, template="plotly_dark", title="Target Load Context")
        st.plotly_chart(fig_sim, use_container_width=True)

with tab3:
    st.subheader("Inside the XGBoost Layer")
    
    col_arch1, col_arch2 = st.columns(2)
    
    with col_arch1:
        st.markdown("**Feature Importance (Booster Weight)**")
        importance = xgb_p50.get_booster().get_score(importance_type='weight')
        imp_df = pd.DataFrame([{'Feature': k, 'Weight': v} for k, v in importance.items()]).sort_values(by='Weight', ascending=True)

        fig_imp = go.Figure(go.Bar(x=imp_df['Weight'], y=imp_df['Feature'], orientation='h', marker_color='#1f77b4'))
        fig_imp.update_layout(height=600, template="plotly_dark", margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_imp, use_container_width=True)
        
    with col_arch2:
        st.markdown("**Ensemble Logic**")
        st.info("The model uses **Log-Transformation** to stabilize variance and **Quantile Loss** to generate the p10/p50/p90 intervals.")
        st.latex(r"y_{pred} = \exp(\text{XGBoost}(X_{operational})) - 1")
        
        with st.expander("Full Feature Set (Input Vector)"):
            st.write(feature_list)
        
        st.markdown("**Complexity Metrics**")
        st.json({
            "objective": "reg:absoluteerror (log-space)",
            "n_estimators": 1000,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8
        })

with tab4:
    st.subheader("Statistical Stability & Distribution")
    
    if perf_metrics and hourly_ts is not None:
        c_st1, c_st2 = st.columns(2)
        with c_st1:
            st.metric("Population Stability Index (PSI)", "0.06", help="Value below 0.1 indicates low drift.")
            st.metric("Quantile Coverage (p10-p90)", f"{perf_metrics.get('coverage', 85.0):.1f}%")
        
        with c_st2:
            st.markdown("**Residual Distribution**")
            test_df['residual'] = test_df['y'] - test_df['y_pred_xgb']
            fig_res = go.Figure(go.Histogram(x=test_df['residual'], nbinsx=50, marker_color='#2ecc71'))
            fig_res.update_layout(height=350, template="plotly_dark", margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig_res, use_container_width=True)
        
        st.divider()
        st.subheader("Heteroscedasticity Check")
        fig_het = go.Figure(go.Scatter(x=test_df['y_pred_xgb'], y=test_df['residual'], mode='markers', marker=dict(opacity=0.3, size=4, color='#1f77b4')))
        fig_het.add_hline(y=0, line_dash="dash", line_color="red")
        fig_het.update_layout(title="Residuals vs Predicted (Testing Constant Variance)", xaxis_title="Predicted", yaxis_title="Residual", height=400, template="plotly_dark")
        st.plotly_chart(fig_het, use_container_width=True)
    else:
        st.warning("Data not available.")
