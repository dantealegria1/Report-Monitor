import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from prophet.serialize import model_from_json
import json
import plotly.graph_objects as go
from datetime import datetime, timedelta
import holidays

# Page Config
st.set_page_config(page_title="Hybrid Forecast", layout="wide")

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
        with open('prophet_model.json', 'r') as f:
            m = model_from_json(f.read())
        
        xgb_model = xgb.XGBRegressor()
        xgb_model.load_model('xgboost_model.json')
        
        with open('label_mapping.json', 'r') as f:
            label_map = json.load(f)
            
        with open('metrics.json', 'r') as f:
            perf_metrics = json.load(f)
            
        return m, xgb_model, label_map, perf_metrics
    except Exception:
        return None, None, None, None

m, xgb_model, label_map, perf_metrics = load_all_resources()

# --- HEADER ---
st.title("Hybrid Forecasting: IaaS Optimization")
st.markdown("""
    *Integration of Prophet (Seasonality) and XGBoost (Operational Dynamics) for Azure resource sizing.*
""")

if m is None:
    st.warning("Trained models not detected. Please run the training process (`trainer.py`) to enable this page.")
    st.stop()

# --- SIDEBAR: INPUTS ---
with st.sidebar:
    st.header("Operational Parameters")
    target_date = st.date_input("Prediction Date", datetime.now() + timedelta(days=1))
    target_hour = st.slider("Hour of Day", 0, 23, 12)
    backlog_input = st.number_input("Current Backlog (Pending Reports)", min_value=0, value=10)

    st.divider()
    st.subheader("Recent Memory (Lags)")
    lag1 = st.number_input("Load at T-1 (Previous Hour)", min_value=0, value=15)
    lag2 = st.number_input("Load at T-2 (2 Hours Ago)", min_value=0, value=12)
    roll_mean = (lag1 + lag2 + 10) / 3

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
report_type_id = [k for k, v in label_map.items() if v == selected_type][0]

future_df = pd.DataFrame({'ds': [dt]})
forecast = m.predict(future_df)
base_pred = forecast.iloc[0]['yhat']

xgb_features = pd.DataFrame([[
    backlog_input, target_hour, weekday, day_of_month, month,
    is_month_end, is_holiday, is_weekend, int(report_type_id),
    lag1, lag2, roll_mean
]], columns=['backlog', 'hour', 'weekday', 'day_of_month', 'month', 'is_month_end', 'is_holiday', 'is_weekend', 'tipo_reporte_id', 'y_lag_1', 'y_lag_2', 'y_rolling_mean_3'])

adjustment = xgb_model.predict(xgb_features)[0]
hybrid_pred = max(0, base_pred + adjustment)

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "Performance Dashboard", 
    "Forecast Analysis", 
    "Model Insights", 
    "Stability & Drift"
])

with tab1:
    st.subheader("Performance Comparison Dashboard")
    hourly_ts = st.session_state.get("hourly_ts")
    if hourly_ts is not None:
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
        
        # Prediction for Metrics comparison
        forecast_test = m.predict(test_df[['ds']])
        test_df['yhat_p'] = forecast_test['yhat'].values
        xgb_test_features = test_df[['backlog', 'hour', 'weekday', 'day_of_month', 'month', 'is_month_end', 'is_holiday', 'is_weekend', 'tipo_reporte_id', 'y_lag_1', 'y_lag_2', 'y_rolling_mean_3']]
        adj_test = xgb_model.predict(xgb_test_features)
        test_df['y_pred_hybrid'] = np.clip(test_df['yhat_p'] + adj_test, 0, None)

        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.metric("Prophet Base MAE", f"{perf_metrics['mae_p']:.4f}")
        with mc2:
            st.metric("Hybrid MAE", f"{perf_metrics['mae_h']:.4f}", 
                      delta=f"{perf_metrics['mae_h'] - perf_metrics['mae_p']:.4f}", delta_color="inverse")
        with mc3:
            improvement = (perf_metrics['mae_p'] - perf_metrics['mae_h']) / perf_metrics['mae_p'] * 100
            st.metric("Gain vs Prophet", f"{improvement:.2f}%")
            
        st.divider()
        st.markdown("### Academic Validation (MASE)")
        col_m1, col_m2 = st.columns([1, 2])
        with col_m1:
            from components.kpis import render_mase_kpi
            render_mase_kpi(perf_metrics['mase_h'])
        with col_m2:
            st.info(f"**Prophet Base MASE**: {perf_metrics['mase_p']:.4f}  \n"
                    f"**Hybrid Adjusted MASE**: {perf_metrics['mase_h']:.4f}  \n"
                    "A MASE < 1.0 indicates performance superior to the historical naive model.")
            
        st.divider()
        st.subheader("Prediction Sample (Latest Records)")
        st.dataframe(test_df[['ds', 'y', 'yhat_p', 'y_pred_hybrid']].tail(20), width='stretch')
    else:
        st.warning("Please load data in the Home page first to see historical validation.")

with tab2:
    st.subheader("Hybrid Forecast Simulation")
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Prophet Base", f"{base_pred:.2f}")
    with c2: st.metric("XGBoost Adjustment", f"{adjustment:+.2f}")
    with c3: st.metric("Final Hybrid", f"{hybrid_pred:.2f}")

    st.divider()
    sim_hours = pd.date_range(start=dt - timedelta(hours=12), end=dt + timedelta(hours=12), freq='h')
    sim_df = pd.DataFrame({'ds': sim_hours})
    prophet_sim = m.predict(sim_df)['yhat'].values
    sim_features = pd.DataFrame({
        'backlog': [backlog_input] * len(sim_hours),
        'hour': sim_hours.hour,
        'weekday': sim_hours.weekday,
        'day_of_month': sim_hours.day,
        'month': sim_hours.month,
        'is_month_end': [int(pd.Timestamp(t).is_month_end) for t in sim_hours],
        'is_holiday': [1 if t in ar_holidays else 0 for t in sim_hours],
        'is_weekend': [1 if t.weekday() >= 5 else 0 for t in sim_hours],
        'tipo_reporte_id': [int(report_type_id)] * len(sim_hours),
        'y_lag_1': [lag1] * len(sim_hours),
        'y_lag_2': [lag2] * len(sim_hours),
        'y_rolling_mean_3': [roll_mean] * len(sim_hours)
    })
    xgb_sim_adj = xgb_model.predict(sim_features)
    hybrid_sim = np.clip(prophet_sim + xgb_sim_adj, 0, None)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sim_hours, y=prophet_sim, name="Prophet Base", line=dict(color='gray', dash='dash')))
    fig.add_trace(go.Scatter(x=sim_hours, y=hybrid_sim, name="Hybrid Model", fill='tonexty', line=dict(color='#1f77b4', width=3)))
    fig.add_trace(go.Scatter(x=[dt], y=[hybrid_pred], mode='markers', marker=dict(color='red', size=12), name="Current Selection"))
    fig.update_layout(title="Resource Projection (24h Window)", xaxis_title="Time", yaxis_title="Report Counts", legend=dict(orientation="h"), height=450)
    st.plotly_chart(fig, width='stretch')

with tab3:
    st.subheader("Hybrid Model Insights")
    st.markdown("""
    The Hybrid Model combines **long-term seasonality** from Prophet with **short-term operational adjustments** from XGBoost.
    
    ### Net Gain Calculation
    The residual error from Prophet is processed by XGBoost, which interprets operational variables (Backlog, Lags) to provide a more precise final count.
    """)
    
    st.info(f"Current Profit Improvement (RMSE): **{((perf_metrics['rmse_p'] - perf_metrics['rmse_h']) / perf_metrics['rmse_p'] * 100):.2f}%**")
    
    with st.expander("XGBoost Internal Parameters"):
        st.write(xgb_model.get_params())

with tab4:
    st.subheader("Advanced Validation & Diagnostics")
    
    # Activity 1.2: Hybrid Cross-Validation (Backtesting)
    st.markdown("### Activity 1.2: Hybrid Cross-Validation (Rolling Windows)")
    st.markdown("""
    This section implements **rolling window backtesting** to validate the Hybrid model's performance across multiple time periods.
    The process applies XGBoost corrections to Prophet's cross-validation results.
    """)
    
    if st.button("Run Hybrid Backtesting", type="primary"):
        if hourly_ts is not None and m is not None and xgb_model is not None:
            with st.spinner("Running cross-validation with rolling windows..."):
                try:
                    from prophet.diagnostics import cross_validation, performance_metrics
                    
                    # Run Prophet CV
                    df_cv = cross_validation(
                        m, 
                        initial='180 days', 
                        period='30 days', 
                        horizon='15 days',
                        parallel=None
                    )
                    
                    # Apply XGBoost correction to CV results
                    # For each CV prediction, we need to extract operational features
                    df_cv_enriched = df_cv.copy()
                    
                    # Merge with hourly_ts to get operational features
                    df_pd_full = hourly_ts.to_pandas()
                    if 'ds' not in df_pd_full.columns and 'timestamp_hour' in df_pd_full.columns:
                        df_pd_full = df_pd_full.rename(columns={'timestamp_hour': 'ds'})
                    
                    df_cv_enriched = df_cv_enriched.merge(
                        df_pd_full[['ds', 'backlog', 'hour', 'weekday', 'day_of_month', 'month', 
                                    'is_month_end', 'is_holiday', 'is_weekend', 'tipo_reporte_id', 
                                    'y_lag_1', 'y_lag_2', 'y_rolling_mean_3']],
                        on='ds',
                        how='left'
                    )
                    
                    # Fill any missing operational features with defaults
                    df_cv_enriched = df_cv_enriched.fillna(0)
                    
                    # Apply XGBoost correction
                    xgb_cv_features = df_cv_enriched[['backlog', 'hour', 'weekday', 'day_of_month', 'month', 
                                                       'is_month_end', 'is_holiday', 'is_weekend', 'tipo_reporte_id', 
                                                       'y_lag_1', 'y_lag_2', 'y_rolling_mean_3']]
                    xgb_cv_adj = xgb_model.predict(xgb_cv_features)
                    df_cv_enriched['yhat_hybrid'] = np.clip(df_cv_enriched['yhat'] + xgb_cv_adj, 0, None)
                    
                    # Calculate metrics for Prophet and Hybrid
                    df_cv_enriched['ae_prophet'] = np.abs(df_cv_enriched['y'] - df_cv_enriched['yhat'])
                    df_cv_enriched['ae_hybrid'] = np.abs(df_cv_enriched['y'] - df_cv_enriched['yhat_hybrid'])
                    
                    mae_cv_prophet = df_cv_enriched['ae_prophet'].mean()
                    mae_cv_hybrid = df_cv_enriched['ae_hybrid'].mean()
                    
                    rmse_cv_prophet = np.sqrt((df_cv_enriched['ae_prophet'] ** 2).mean())
                    rmse_cv_hybrid = np.sqrt((df_cv_enriched['ae_hybrid'] ** 2).mean())
                    
                    st.success("Cross-validation completed successfully!")
                    
                    # Display results
                    col_cv1, col_cv2, col_cv3 = st.columns(3)
                    with col_cv1:
                        st.metric("Prophet CV MAE", f"{mae_cv_prophet:.4f}")
                    with col_cv2:
                        st.metric("Hybrid CV MAE", f"{mae_cv_hybrid:.4f}",
                                  delta=f"{mae_cv_hybrid - mae_cv_prophet:.4f}", delta_color="inverse")
                    with col_cv3:
                        improvement_cv = (mae_cv_prophet - mae_cv_hybrid) / mae_cv_prophet * 100
                        st.metric("CV Improvement", f"{improvement_cv:.2f}%")
                    
                    # Plot CV results
                    fig_cv = go.Figure()
                    fig_cv.add_trace(go.Scatter(x=df_cv_enriched['ds'], y=df_cv_enriched['y'], 
                                                name="Actual", mode='markers', marker=dict(size=3, color='red', opacity=0.3)))
                    fig_cv.add_trace(go.Scatter(x=df_cv_enriched['ds'], y=df_cv_enriched['yhat'], 
                                                name="Prophet CV", mode='markers', marker=dict(size=3, color='gray')))
                    fig_cv.add_trace(go.Scatter(x=df_cv_enriched['ds'], y=df_cv_enriched['yhat_hybrid'], 
                                                name="Hybrid CV", mode='markers', marker=dict(size=3, color='#1f77b4')))
                    fig_cv.update_layout(title="Cross-Validation Results (Rolling Windows)", 
                                         xaxis_title="Date", yaxis_title="Report Count", 
                                         legend=dict(orientation="h"), height=400)
                    st.plotly_chart(fig_cv, width='stretch')
                    
                except Exception as e:
                    st.error(f"Cross-validation failed: {str(e)}")
                    st.info("This may occur if there's insufficient data for the specified CV parameters.")
        else:
            st.warning("Please ensure data is loaded and models are trained.")
    
    st.divider()
    
    # Activity 1.3: Residuals Analysis
    st.markdown("### Activity 1.3: Residuals Analysis (White Noise Verification)")
    st.markdown("""
    This section analyzes the **residuals** (prediction errors) to verify if they represent **white noise**, 
    which indicates that the model has captured all systematic patterns in the data.
    """)
    
    if hourly_ts is not None and perf_metrics is not None:
        # Calculate residuals
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
        
        # Generate predictions
        forecast_test = m.predict(test_df[['ds']])
        test_df['yhat_p'] = forecast_test['yhat'].values
        xgb_test_features = test_df[['backlog', 'hour', 'weekday', 'day_of_month', 'month', 
                                      'is_month_end', 'is_holiday', 'is_weekend', 'tipo_reporte_id', 
                                      'y_lag_1', 'y_lag_2', 'y_rolling_mean_3']]
        adj_test = xgb_model.predict(xgb_test_features)
        test_df['y_pred_hybrid'] = np.clip(test_df['yhat_p'] + adj_test, 0, None)
        
        # Calculate residuals
        test_df['residual_prophet'] = test_df['y'] - test_df['yhat_p']
        test_df['residual_hybrid'] = test_df['y'] - test_df['y_pred_hybrid']
        
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            st.markdown("#### Residuals Distribution (Histogram)")
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(x=test_df['residual_prophet'], name="Prophet Residuals", 
                                            opacity=0.5, marker_color='gray', nbinsx=30))
            fig_hist.add_trace(go.Histogram(x=test_df['residual_hybrid'], name="Hybrid Residuals", 
                                            opacity=0.7, marker_color='#1f77b4', nbinsx=30))
            fig_hist.update_layout(title="Residuals Distribution", xaxis_title="Residual", 
                                   yaxis_title="Frequency", barmode='overlay', 
                                   legend=dict(orientation="h"), height=350)
            st.plotly_chart(fig_hist, width='stretch')
            
            st.info("""
            **Interpretation**: A bell-shaped (normal) distribution centered at zero indicates good model fit. 
            The Hybrid model should show tighter clustering around zero compared to Prophet.
            """)
        
        with col_res2:
            st.markdown("#### Autocorrelation Function (ACF)")
            from statsmodels.tsa.stattools import acf
            
            # Calculate ACF for both models
            acf_prophet = acf(test_df['residual_prophet'].dropna(), nlags=24, fft=True)
            acf_hybrid = acf(test_df['residual_hybrid'].dropna(), nlags=24, fft=True)
            
            fig_acf = go.Figure()
            fig_acf.add_trace(go.Bar(x=list(range(len(acf_prophet))), y=acf_prophet, 
                                     name="Prophet ACF", marker_color='gray', opacity=0.5))
            fig_acf.add_trace(go.Bar(x=list(range(len(acf_hybrid))), y=acf_hybrid, 
                                     name="Hybrid ACF", marker_color='#1f77b4', opacity=0.7))
            fig_acf.add_hline(y=0, line_dash="dash", line_color="black")
            fig_acf.add_hline(y=1.96/np.sqrt(len(test_df)), line_dash="dot", line_color="red", 
                              annotation_text="95% CI")
            fig_acf.add_hline(y=-1.96/np.sqrt(len(test_df)), line_dash="dot", line_color="red")
            fig_acf.update_layout(title="Autocorrelation of Residuals", xaxis_title="Lag (hours)", 
                                  yaxis_title="ACF", barmode='group', 
                                  legend=dict(orientation="h"), height=350)
            st.plotly_chart(fig_acf, width='stretch')
            
            st.info("""
            **Interpretation**: ACF values within the confidence interval (red dotted lines) indicate **white noise**. 
            The Hybrid model should show lower autocorrelation, confirming it has captured more patterns.
            """)
        
        # Statistical summary
        st.divider()
        st.markdown("#### Residuals Statistical Summary")
        summary_data = {
            "Model": ["Prophet Base", "Hybrid"],
            "Mean Residual": [test_df['residual_prophet'].mean(), test_df['residual_hybrid'].mean()],
            "Std Dev": [test_df['residual_prophet'].std(), test_df['residual_hybrid'].std()],
            "Skewness": [test_df['residual_prophet'].skew(), test_df['residual_hybrid'].skew()],
            "Kurtosis": [test_df['residual_prophet'].kurtosis(), test_df['residual_hybrid'].kurtosis()]
        }
        st.dataframe(pd.DataFrame(summary_data), width='stretch')
        st.caption("""
        **White Noise Criteria**: Mean ≈ 0, Low Std Dev, Skewness ≈ 0, Kurtosis ≈ 3 (normal distribution).
        """)
    else:
        st.warning("Please load data in the Home page first to see residuals analysis.")
