import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import xgboost as xgb
from prophet.serialize import model_from_json
import json
import os

st.set_page_config(page_title="Model Comparison", layout="wide")

st.title("Model Comparison: Prophet vs XGBoost")
st.markdown("""
    This analysis compares the **Prophet** (baseline) against the **New XGBoost** (log-transformed)
    by transforming the regression task into a binary classification based on an accuracy threshold.
""")

# --- LOAD RESOURCES ---
@st.cache_resource
def load_all_resources():
    try:
        with open('prophet_model.json', 'r') as f:
            m = model_from_json(f.read())
        
        xgb_p50 = xgb.XGBRegressor()
        xgb_p50.load_model('xgboost_model_p50.json')
        
        with open('label_mapping.json', 'r') as f:
            label_map = json.load(f)

        with open('feature_list.json', 'r') as f:
            feature_list = json.load(f)
            
<<<<<<< HEAD
        if os.path.exists('model_metadata.json'):
            with open('model_metadata.json', 'r') as f:
                meta = json.load(f)
        else:
            meta = {"target_transform": "log1p"}

        return m, xgb_p50, label_map, feature_list, meta
    except Exception as e:
        st.error(f"Missing required models or files: {e}")
        return None, None, None, None, None

m, xgb_p50, label_map, feature_list, meta = load_all_resources()
=======
        return m, xgb_p50, label_map, feature_list
    except Exception as e:
        st.error(f"Missing required models or files: {e}")
        return None, None, None, None

m, xgb_p50, label_map, feature_list = load_all_resources()
>>>>>>> main

# --- SIDEBAR CONFIG ---
with st.sidebar:
    st.header("Accuracy Target")
    accuracy_threshold = st.slider(
        "Accuracy Threshold (Error < X Reports)", 
        min_value=5, max_value=100, value=30, step=5,
        help="A prediction is considered 'Correct' if the absolute error is below this threshold."
    )

if m is None or xgb_p50 is None:
    st.warning("Please run the training process (`trainer.py`) to generate models first.")
    st.stop()

# --- DATA PREPARATION ---
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
    
    # Ensure full feature set is available
    if 'hour_sin' not in hourly_ts.columns:
        with st.spinner("Re-enriching data for comparison..."):
            hourly_ts = enrich_hourly_ts_with_features(hourly_ts, df_raw)
            st.session_state["hourly_ts"] = hourly_ts

    df_pd = hourly_ts.to_pandas()
    rename_map = {'timestamp_hour': 'ds', 'report_count': 'y'}
    df_pd = df_pd.rename(columns={k: v for k, v in rename_map.items() if k in df_pd.columns})
    
    # Split
    split_idx = int(len(df_pd) * 0.8)
    test_df = df_pd.iloc[split_idx:].copy()
    
    with st.spinner("Generating Comparative Predictions..."):
        # 1. Prophet Baseline
        p_forecast = m.predict(test_df[['ds']])
        test_df['y_pred_prophet'] = np.clip(p_forecast['yhat'].values, 0, None)
        
        # 2. XGBoost (New Model)
        test_xgb_features = test_df[feature_list]
<<<<<<< HEAD
        is_log = meta.get("target_transform", "raw") == "log1p"
        test_df['y_pred_xgb'] = (np.expm1(xgb_p50.predict(test_xgb_features)) if is_log else xgb_p50.predict(test_xgb_features)).clip(min=0)
=======
        test_df['y_pred_xgb'] = np.expm1(xgb_p50.predict(test_xgb_features)).clip(min=0)
>>>>>>> main
        
        # Binary Labels (1 if precise, 0 if error > threshold)
        test_df['correct_prophet'] = (np.abs(test_df['y'] - test_df['y_pred_prophet']) < accuracy_threshold).astype(int)
        test_df['correct_xgb'] = (np.abs(test_df['y'] - test_df['y_pred_xgb']) < accuracy_threshold).astype(int)
        
        # Score for ROC (inverse of error mapped to 0-1 range roughly, or usage of relative accuracy)
        # For ROC, we use the probability or a proxy. Here we use 1 - normalized error
        max_err = max(np.abs(test_df['y'] - test_df['y_pred_prophet']).max(), 
                      np.abs(test_df['y'] - test_df['y_pred_xgb']).max())
        test_df['score_prophet'] = 1 - (np.abs(test_df['y'] - test_df['y_pred_prophet']) / max_err)
        test_df['score_xgb'] = 1 - (np.abs(test_df['y'] - test_df['y_pred_xgb']) / max_err)

    # --- METRICS & VISUALS ---
    tab1, tab2, tab3 = st.tabs(["ROC Analysis", "Confusion Matrices", "Detailed Metrics"])
    
    # Calculate ROC
    fpr_p, tpr_p, _ = roc_curve(test_df['correct_prophet'], test_df['score_prophet'], pos_label=1)
    roc_auc_p = auc(fpr_p, tpr_p)
    
    fpr_x, tpr_x, _ = roc_curve(test_df['correct_xgb'], test_df['score_xgb'], pos_label=1)
    roc_auc_x = auc(fpr_x, tpr_x)

    with tab1:
        st.subheader("ROC Curve Comparison")
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr_p, y=tpr_p, name=f"Prophet Baseline (AUC = {roc_auc_p:.3f})", line=dict(color='gray', dash='dash')))
        fig_roc.add_trace(go.Scatter(x=fpr_x, y=tpr_x, name=f"Advanced XGBoost (AUC = {roc_auc_x:.3f})", line=dict(color='#1f77b4', width=3)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], showlegend=False, line=dict(color='black', dash='dot')))
        fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=500, template="plotly_dark")
        st.plotly_chart(fig_roc, use_container_width=True)
        
        st.info("""
            **Interpretation**: A higher AUC (Area Under Curve) indicates better ability to stay within the accuracy threshold. 
            XGBoost's curve being further towards the top-left demonstrates its superior precision in handling operational spikes.
        """)

    with tab2:
        st.subheader("Confusion Matrix")
        c1, c2 = st.columns(2)
        
        cm_p = confusion_matrix(test_df['correct_prophet'], (test_df['score_prophet'] > 0.5).astype(int))
        cm_x = confusion_matrix(test_df['correct_xgb'], (test_df['score_xgb'] > 0.5).astype(int))
        
        with c1:
            st.markdown("#### Prophet")
            fig_cmp = px.imshow(cm_p, text_auto=True, labels=dict(x="Predicted Accuracy", y="Actual Accuracy"), x=['Inaccurate', 'Accurate'], y=['Inaccurate', 'Accurate'], color_continuous_scale='Greys')
            st.plotly_chart(fig_cmp, use_container_width=True)
            
        with c2:
            st.markdown("#### XGBoost")
            fig_cmx = px.imshow(cm_x, text_auto=True, labels=dict(x="Predicted Accuracy", y="Actual Accuracy"), x=['Inaccurate', 'Accurate'], y=['Inaccurate', 'Accurate'], color_continuous_scale='Blues')
            st.plotly_chart(fig_cmx, use_container_width=True)

    with tab3:
        st.subheader("Accuracy Reports")
        report_p = classification_report(test_df['correct_prophet'], (test_df['score_prophet'] > 0.5).astype(int), output_dict=True)
        report_x = classification_report(test_df['correct_xgb'], (test_df['score_xgb'] > 0.5).astype(int), output_dict=True)
        
        st.markdown("#### Key Indicators")
        # Extract Precision/Recall for the 'Accurate' class (1)
        res_df = pd.DataFrame({
            'Metric': ['Precision', 'Recall', 'F1-Score'],
            'Prophet': [report_p['1']['precision'], report_p['1']['recall'], report_p['1']['f1-score']],
            'XGBoost': [report_x['1']['precision'], report_x['1']['recall'], report_x['1']['f1-score']]
        })
        st.table(res_df)
        
        st.divider()
        st.subheader("Raw Prediction Examples")
        st.dataframe(test_df[['ds', 'y', 'y_pred_prophet', 'y_pred_xgb']].tail(20), use_container_width=True)

else:
    st.warning("Please load data on the Home page first.")
