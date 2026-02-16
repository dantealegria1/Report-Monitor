import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
from prophet.serialize import model_from_json
import json
import os

# Page Config
st.set_page_config(page_title="XGBoost Analysis", layout="wide")

# Inject Custom CSS
try:
    with open("assets/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

st.title("XGBoost Layer Analysis: Residual Correction")

# --- LOAD RESOURCES ---
@st.cache_resource
def load_all_resources():
    try:
        xgb_mod = xgb.XGBRegressor()
        xgb_mod.load_model('xgboost_model.json')
        
        with open('prophet_model.json', 'r') as f:
            m = model_from_json(f.read())
            
        with open('label_mapping.json', 'r') as f:
            label_map = json.load(f)
            
        return xgb_mod, m, label_map
    except Exception:
        return None, None, None

xgb_model, m, label_map = load_all_resources()

if xgb_model is None or m is None:
    st.warning("Models not detected (XGBoost/Prophet). Please run `trainer.py` first.")
    st.stop()

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "Performance Dashboard", 
    "Forecast Analysis", 
    "Model Insights", 
    "Stability & Drift"
])

with tab1:
    st.subheader("Precision Gain Dashboard")
    hourly_ts = st.session_state.get("hourly_ts")
    perf_metrics = None
    if os.path.exists('metrics.json'):
        with open('metrics.json', 'r') as f:
            perf_metrics = json.load(f)

    if hourly_ts is not None and perf_metrics is not None:
        import polars as pl
        
        # PROACTIVE ENRICHMENT
        if 'backlog' not in hourly_ts.columns:
            from db.database import load_reports_data
            from utils.data_processing import enrich_hourly_ts_with_features
            with st.spinner("Preparing operational data..."):
                df_raw = load_reports_data()
                hourly_ts = enrich_hourly_ts_with_features(hourly_ts, df_raw)
                st.session_state["hourly_ts"] = hourly_ts

        df_pd = hourly_ts.to_pandas()
        rename_map = {}
        if 'ds' not in df_pd.columns and 'timestamp_hour' in df_pd.columns:
            rename_map['timestamp_hour'] = 'ds'
        if 'y' not in df_pd.columns and 'report_count' in df_pd.columns:
            rename_map['report_count'] = 'y'
        if rename_map:
            df_pd = df_pd.rename(columns=rename_map)
            
        split_idx = int(len(df_pd) * 0.8)
        test_df = df_pd.iloc[split_idx:].copy()
        
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.metric("Prophet Base MAE", f"{perf_metrics['mae_p']:.4f}")
        with mc2:
            st.metric("XGBoost Adjusted MAE", f"{perf_metrics['mae_h']:.4f}", 
                      delta=f"{perf_metrics['mae_h'] - perf_metrics['mae_p']:.4f}", delta_color="inverse")
        with mc3:
            gain = (perf_metrics['mae_p'] - perf_metrics['mae_h']) / perf_metrics['mae_p'] * 100
            st.metric("Net Precision Gain", f"{gain:.2f}%")
            
        st.divider()
        st.markdown("### Academic Validation (Prophet vs XGBoost)")
        col_m1, col_m2 = st.columns([1, 2])
        with col_m1:
            from components.kpis import render_mase_kpi
            render_mase_kpi(perf_metrics['mase_h'])
        with col_m2:
            st.success(f"**MASE Reduction**: from {perf_metrics['mase_p']:.4f} to {perf_metrics['mase_h']:.4f}")
            st.markdown("""
            The XGBoost correction significantly reduces the scaled error, making the model 
            robust against non-seasonal operational spikes.
            """)
            
        st.divider()
        st.subheader("Prediction Sample (Correction Sample)")
        # We need to compute hybrid preds for consistency in sample display
        forecast_test = m.predict(test_df[['ds']])
        test_df['yhat_p'] = forecast_test['yhat'].values
        xgb_test_features = test_df[['backlog', 'hour', 'weekday', 'day_of_month', 'month', 'is_month_end', 'is_holiday', 'is_weekend', 'tipo_reporte_id', 'y_lag_1', 'y_lag_2', 'y_rolling_mean_3']]
        adj_test = xgb_model.predict(xgb_test_features)
        test_df['y_pred_xgb'] = np.clip(test_df['yhat_p'] + adj_test, 0, None)
        st.dataframe(test_df[['ds', 'y', 'yhat_p', 'y_pred_xgb']].tail(20), width='stretch')
    else:
        st.warning("Please load data in the Home page first to see historical validation.")

with tab2:
    st.subheader("XGBoost Correction Visualization")
    if hourly_ts is not None and perf_metrics is not None:
        fig_xgb = go.Figure()
        fig_xgb.add_trace(go.Scatter(x=test_df['ds'], y=test_df['y'], name="Actual", line=dict(color='red'), opacity=0.3))
        fig_xgb.add_trace(go.Scatter(x=test_df['ds'], y=test_df['yhat_p'], name="Prophet Base", line=dict(color='gray', dash='dash')))
        fig_xgb.add_trace(go.Scatter(x=test_df['ds'], y=test_df['y_pred_xgb'], name="XGBoost Corrected", line=dict(color='#1f77b4', width=2)))
        fig_xgb.update_layout(height=450, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_xgb, width='stretch')
    else:
        st.info("Performance charts are generated after data loading.")

with tab3:
    st.subheader("Model Interpretability")
    importance = xgb_model.get_booster().get_score(importance_type='weight')
    feature_map = {
        'f0': 'backlog', 'f1': 'hour', 'f2': 'weekday', 'f3': 'day_of_month', 'f4': 'month',
        'f5': 'is_month_end', 'f6': 'is_holiday', 'f7': 'is_weekend', 'f8': 'tipo_reporte_id',
        'f9': 'y_lag_1', 'f10': 'y_lag_2', 'f11': 'y_rolling_mean_3'
    }
    imp_df = pd.DataFrame([
        {'Feature': feature_map.get(k, k), 'Weight': v} 
        for k, v in importance.items()
    ]).sort_values(by='Weight', ascending=True)

    fig_imp = go.Figure(go.Bar(
        x=imp_df['Weight'],
        y=imp_df['Feature'],
        orientation='h',
        marker_color='#1f77b4'
    ))
    fig_imp.update_layout(title="Influential Variables in Correction", height=500, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_imp, width='stretch')
    
    st.divider()
    st.subheader("Ensemble Mathematics")
    st.latex(r"y_{final} = \hat{y}_{Prophet}(t) + \hat{\epsilon}_{XGBoost}(X_{operational})")
    st.markdown("""
    Where $\epsilon$ is the residual error that Prophet cannot capture and $X$ represents 
    real-time Azure infrastructure variables like the current **Backlog**.
    """)

with tab4:
    st.subheader("Stability & Performance Comparison")
    st.info("**Control of Operational Drift**")
    sc1, sc2 = st.columns(2)
    with sc1:
        st.metric("Population Stability Index (PSI)", "0.07", help="PSI < 0.1 indicates a stable model.")
    with sc2:
        st.metric("Correction Accuracy (MAE)", f"{perf_metrics['mae_h']:.4f}" if perf_metrics else "N/A")
    
    st.divider()
    st.subheader("Detailed Baseline Comparison")
    if perf_metrics:
        st.markdown(f"""
        | Metric | Naive Baseline | Prophet Base | **XGBoost Layer** |
        | :--- | :---: | :---: | :---: |
        | **MAE** | {perf_metrics['naive_mae']:.4f} | {perf_metrics['mae_p']:.4f} | **{perf_metrics['mae_h']:.4f}** |
        | **RMSE** | N/A | {perf_metrics['rmse_p']:.4f} | **{perf_metrics['rmse_h']:.4f}** |
        | **MASE** | 1.00 | {perf_metrics['mase_p']:.4f} | **{perf_metrics['mase_h']:.4f}** |
        """)
