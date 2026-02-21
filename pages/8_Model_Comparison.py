"""
Model Comparison: Prophet vs Hybrid (Prophet + XGBoost)
Evaluates forecasting models using AUC-ROC by transforming regression into binary classification.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import polars as pl
from datetime import datetime

# Page configuration
st.set_page_config(page_title="Model Comparison", page_icon="⚖️", layout="wide")

st.title("⚖️ Model Comparison: Prophet vs Hybrid")
st.markdown("""
This page evaluates the forecasting accuracy by transforming the regression problem into binary classification.
A prediction is considered **accurate** if the absolute error is ≤ 30 reports.
""")

# Accuracy threshold
ACCURACY_THRESHOLD = 30

@st.cache_data
def load_comparison_data():
    """
    Load actual Prophet and Hybrid model predictions from trained models.
    Uses the test set (last 20% of data) for evaluation.
    """
    try:
        import xgboost as xgb
        from prophet.serialize import model_from_json
        import json
        import os
        
        # Check if models exist
        if not os.path.exists('prophet_model.json') or not os.path.exists('xgboost_model.json'):
            st.warning("⚠️ Models not found. Please run `trainer.py` first to train the models.")
            return None
        
        # Load trained models
        with open('prophet_model.json', 'r') as f:
            prophet_model = model_from_json(f.read())
        
        xgb_model = xgb.XGBRegressor()
        xgb_model.load_model('xgboost_model.json')
        
        # Load hourly time series data from session state
        hourly_ts = st.session_state.get("hourly_ts")
        
        if hourly_ts is None:
            st.warning("⚠️ No data found in session state. Please load data from the Home page first.")
            return None
        
        # Convert to pandas if needed
        if isinstance(hourly_ts, pl.DataFrame):
            df = hourly_ts.to_pandas()
        else:
            df = hourly_ts.copy()
        
        # Ensure proper column names
        rename_map = {}
        if 'ds' not in df.columns and 'timestamp_hour' in df.columns:
            rename_map['timestamp_hour'] = 'ds'
        if 'y' not in df.columns and 'report_count' in df.columns:
            rename_map['report_count'] = 'y'
        if rename_map:
            df = df.rename(columns=rename_map)
        
        # Ensure we have required features for XGBoost
        required_features = ['backlog', 'hour', 'weekday', 'day_of_month', 'month',
                           'is_month_end', 'is_holiday', 'is_weekend', 'tipo_reporte_id',
                           'y_lag_1', 'y_lag_2', 'y_rolling_mean_3']
        
        missing_features = [f for f in required_features if f not in df.columns]
        if missing_features:
            st.warning(f"⚠️ Missing features: {missing_features}. Enriching data...")
            from db.database import load_reports_data
            from utils.data_processing import enrich_hourly_ts_with_features
            df_raw = load_reports_data()
            hourly_ts_enriched = enrich_hourly_ts_with_features(
                pl.from_pandas(df) if not isinstance(hourly_ts, pl.DataFrame) else hourly_ts,
                df_raw
            )
            df = hourly_ts_enriched.to_pandas()
            st.session_state["hourly_ts"] = hourly_ts_enriched
            
            # Enrichment utility returns timestamp_hour/report_count names.
            # Normalize back to Prophet-compatible ds/y names for this page.
            rename_map = {}
            if 'ds' not in df.columns and 'timestamp_hour' in df.columns:
                rename_map['timestamp_hour'] = 'ds'
            if 'y' not in df.columns and 'report_count' in df.columns:
                rename_map['report_count'] = 'y'
            if rename_map:
                df = df.rename(columns=rename_map)
        
        # Final schema guard before model inference
        required_base_cols = ['ds', 'y']
        missing_base_cols = [c for c in required_base_cols if c not in df.columns]
        if missing_base_cols:
            st.error(f"❌ Missing required columns for comparison: {missing_base_cols}")
            st.caption(f"Available columns: {list(df.columns)}")
            return None
        
        # Use test set (last 20% of data)
        split_idx = int(len(df) * 0.8)
        test_df = df.iloc[split_idx:].copy()
        
        # Generate Prophet predictions
        prophet_forecast = prophet_model.predict(test_df[['ds']])
        test_df['y_pred_prophet'] = np.clip(prophet_forecast['yhat'].values, 0, None)
        
        # Generate XGBoost adjustments
        xgb_features = test_df[required_features]
        xgb_adjustments = xgb_model.predict(xgb_features)
        
        # Hybrid predictions = Prophet + XGBoost correction
        test_df['y_pred_hybrid'] = np.clip(test_df['y_pred_prophet'] + xgb_adjustments, 0, None)
        
        # Prepare output DataFrame
        comparison_df = pd.DataFrame({
            'ds': test_df['ds'],
            'y_true': test_df['y'],
            'y_pred_prophet': test_df['y_pred_prophet'],
            'y_pred_hybrid': test_df['y_pred_hybrid']
        })
        
        return comparison_df
    
    except Exception as e:
        st.error(f"❌ Error loading comparison data: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None


def calculate_binary_classification(y_true, y_pred, threshold=ACCURACY_THRESHOLD):
    """
    Transform regression predictions into binary classification.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        threshold: Maximum acceptable absolute error
    
    Returns:
        is_accurate: Binary labels (1 = accurate, 0 = inaccurate)
        probability_scores: Continuous scores for ROC curve
    """
    absolute_error = np.abs(y_true - y_pred)
    
    # Binary classification: 1 if error <= threshold, 0 otherwise
    is_accurate = (absolute_error <= threshold).astype(int)
    
    # Probability score: inverse of normalized absolute error
    # Higher score = better prediction (lower error)
    max_error = np.max(absolute_error)
    if max_error > 0:
        probability_scores = 1 - (absolute_error / max_error)
    else:
        probability_scores = np.ones_like(absolute_error)
    
    return is_accurate, probability_scores


def plot_roc_curves(y_true_prophet, prob_prophet, y_true_hybrid, prob_hybrid):
    """
    Plot ROC curves for both Prophet and Hybrid models.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Prophet ROC curve
    fpr_prophet, tpr_prophet, _ = roc_curve(y_true_prophet, prob_prophet)
    auc_prophet = auc(fpr_prophet, tpr_prophet)
    
    # Hybrid ROC curve
    fpr_hybrid, tpr_hybrid, _ = roc_curve(y_true_hybrid, prob_hybrid)
    auc_hybrid = auc(fpr_hybrid, tpr_hybrid)
    
    # Plot curves
    ax.plot(fpr_prophet, tpr_prophet, color='#FF6B6B', linewidth=2.5, 
            label=f'Prophet (AUC = {auc_prophet:.3f})')
    ax.plot(fpr_hybrid, tpr_hybrid, color='#4ECDC4', linewidth=2.5, 
            label=f'Hybrid (AUC = {auc_hybrid:.3f})')
    
    # Diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Random Classifier')
    
    # Styling
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curve Comparison: Prophet vs Hybrid Model', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    plt.tight_layout()
    
    return fig, auc_prophet, auc_hybrid


def plot_confusion_matrices(y_true_prophet, y_pred_prophet_binary, 
                            y_true_hybrid, y_pred_hybrid_binary):
    """
    Plot confusion matrices for both models side by side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Prophet confusion matrix
    cm_prophet = confusion_matrix(y_true_prophet, y_pred_prophet_binary)
    disp_prophet = ConfusionMatrixDisplay(confusion_matrix=cm_prophet, 
                                          display_labels=['Inaccurate', 'Accurate'])
    disp_prophet.plot(ax=axes[0], cmap='Reds', colorbar=False)
    axes[0].set_title('Prophet Model\nConfusion Matrix', fontsize=13, fontweight='bold')
    axes[0].grid(False)
    
    # Hybrid confusion matrix
    cm_hybrid = confusion_matrix(y_true_hybrid, y_pred_hybrid_binary)
    disp_hybrid = ConfusionMatrixDisplay(confusion_matrix=cm_hybrid, 
                                         display_labels=['Inaccurate', 'Accurate'])
    disp_hybrid.plot(ax=axes[1], cmap='Greens', colorbar=False)
    axes[1].set_title('Hybrid Model\nConfusion Matrix', fontsize=13, fontweight='bold')
    axes[1].grid(False)
    
    plt.tight_layout()
    
    return fig, cm_prophet, cm_hybrid


def calculate_classification_metrics(cm):
    """
    Calculate precision, recall, F1-score from confusion matrix.
    
    Confusion matrix format:
    [[TN, FP],
     [FN, TP]]
    """
    tn, fp, fn, tp = cm.ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    return {
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1_score,
        'Accuracy': accuracy,
        'True Positives': tp,
        'True Negatives': tn,
        'False Positives': fp,
        'False Negatives': fn
    }


# Main execution
st.markdown("---")

# Workflow Information
with st.expander("ℹ️ How This Page Works", expanded=False):
    st.markdown("""
    ### Data Source
    This page uses **actual predictions** from trained models:
    
    1. **Prophet Model**: Loaded from `prophet_model.json` (trained via `trainer.py`)
    2. **XGBoost Model**: Loaded from `xgboost_model.json` (trained via `trainer.py`)
    3. **Test Data**: Last 20% of your hourly time series data
    
    ### Workflow
    1. ✅ Run `trainer.py` to train both models
    2. ✅ Load data in the Home page to populate session state
    3. ✅ Navigate to this page to see the comparison
    
    ### Evaluation Method
    - Predictions are transformed into **binary classification** (accurate vs inaccurate)
    - Threshold: `|error| ≤ 30 reports`
    - ROC curves show the model's ability to distinguish accurate from inaccurate predictions
    """)

st.markdown("---")

# Load data
with st.spinner("Loading actual model predictions from test set..."):
    df = load_comparison_data()

if df is not None:
    st.success(f"✅ Loaded {len(df)} predictions for comparison")
    
    # Display sample data
    with st.expander("📊 View Sample Data"):
        st.dataframe(df.head(20), use_container_width=True)
    
    st.markdown("---")
    
    # Calculate binary classifications
    st.subheader("🎯 Binary Classification Transformation")
    st.markdown(f"""
    **Accuracy Definition:** A prediction is accurate if `|y_true - y_pred| ≤ {ACCURACY_THRESHOLD}` reports.
    
    **Probability Score:** Calculated as `1 - (absolute_error / max_error)` to provide continuous values for ROC analysis.
    """)
    
    # Prophet classification
    is_accurate_prophet, prob_scores_prophet = calculate_binary_classification(
        df['y_true'].values, 
        df['y_pred_prophet'].values
    )
    
    # Hybrid classification
    is_accurate_hybrid, prob_scores_hybrid = calculate_binary_classification(
        df['y_true'].values, 
        df['y_pred_hybrid'].values
    )
    
    # Display accuracy rates
    col1, col2, col3 = st.columns(3)
    
    with col1:
        prophet_accuracy_rate = is_accurate_prophet.mean() * 100
        st.metric("Prophet Accuracy Rate", f"{prophet_accuracy_rate:.2f}%")
    
    with col2:
        hybrid_accuracy_rate = is_accurate_hybrid.mean() * 100
        st.metric("Hybrid Accuracy Rate", f"{hybrid_accuracy_rate:.2f}%")
    
    with col3:
        improvement = hybrid_accuracy_rate - prophet_accuracy_rate
        st.metric("Improvement", f"{improvement:+.2f}%", 
                 delta=f"{improvement:.2f}%")
    
    st.markdown("---")
    
    # ROC Curve Analysis
    st.subheader("📈 ROC Curve Analysis")
    
    fig_roc, auc_prophet, auc_hybrid = plot_roc_curves(
        is_accurate_prophet, prob_scores_prophet,
        is_accurate_hybrid, prob_scores_hybrid
    )
    
    st.pyplot(fig_roc)
    
    # AUC Interpretation
    st.markdown("### 📊 AUC-ROC Interpretation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Prophet AUC-ROC", f"{auc_prophet:.4f}")
        if auc_prophet >= 0.9:
            st.success("Excellent performance")
        elif auc_prophet >= 0.8:
            st.info("Good performance")
        elif auc_prophet >= 0.7:
            st.warning("Fair performance")
        else:
            st.error("Poor performance")
    
    with col2:
        st.metric("Hybrid AUC-ROC", f"{auc_hybrid:.4f}", 
                 delta=f"{auc_hybrid - auc_prophet:+.4f}")
        if auc_hybrid >= 0.9:
            st.success("Excellent performance")
        elif auc_hybrid >= 0.8:
            st.info("Good performance")
        elif auc_hybrid >= 0.7:
            st.warning("Fair performance")
        else:
            st.error("Poor performance")
    
    st.markdown("---")
    
    # Confusion Matrices
    st.subheader("🔍 Confusion Matrix Analysis")
    st.markdown(f"Based on threshold: **|error| ≤ {ACCURACY_THRESHOLD}**")
    
    # Use threshold of 0.5 on probability scores for binary classification
    y_pred_prophet_binary = (prob_scores_prophet >= 0.5).astype(int)
    y_pred_hybrid_binary = (prob_scores_hybrid >= 0.5).astype(int)
    
    fig_cm, cm_prophet, cm_hybrid = plot_confusion_matrices(
        is_accurate_prophet, y_pred_prophet_binary,
        is_accurate_hybrid, y_pred_hybrid_binary
    )
    
    st.pyplot(fig_cm)
    
    st.markdown("---")
    
    # Detailed Metrics
    st.subheader("📋 Detailed Classification Metrics")
    
    metrics_prophet = calculate_classification_metrics(cm_prophet)
    metrics_hybrid = calculate_classification_metrics(cm_hybrid)
    
    # Create comparison table
    metrics_df = pd.DataFrame({
        'Metric': ['Precision', 'Recall', 'F1-Score', 'Accuracy'],
        'Prophet': [
            f"{metrics_prophet['Precision']:.4f}",
            f"{metrics_prophet['Recall']:.4f}",
            f"{metrics_prophet['F1-Score']:.4f}",
            f"{metrics_prophet['Accuracy']:.4f}"
        ],
        'Hybrid': [
            f"{metrics_hybrid['Precision']:.4f}",
            f"{metrics_hybrid['Recall']:.4f}",
            f"{metrics_hybrid['F1-Score']:.4f}",
            f"{metrics_hybrid['Accuracy']:.4f}"
        ],
        'Improvement': [
            f"{metrics_hybrid['Precision'] - metrics_prophet['Precision']:+.4f}",
            f"{metrics_hybrid['Recall'] - metrics_prophet['Recall']:+.4f}",
            f"{metrics_hybrid['F1-Score'] - metrics_prophet['F1-Score']:+.4f}",
            f"{metrics_hybrid['Accuracy'] - metrics_prophet['Accuracy']:+.4f}"
        ]
    })
    
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    # Detailed counts
    st.markdown("### Prediction Counts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Prophet Model:**")
        st.write(f"- True Positives: {metrics_prophet['True Positives']}")
        st.write(f"- True Negatives: {metrics_prophet['True Negatives']}")
        st.write(f"- False Positives: {metrics_prophet['False Positives']}")
        st.write(f"- False Negatives: {metrics_prophet['False Negatives']}")
    
    with col2:
        st.markdown("**Hybrid Model:**")
        st.write(f"- True Positives: {metrics_hybrid['True Positives']}")
        st.write(f"- True Negatives: {metrics_hybrid['True Negatives']}")
        st.write(f"- False Positives: {metrics_hybrid['False Positives']}")
        st.write(f"- False Negatives: {metrics_hybrid['False Negatives']}")
    
    st.markdown("---")
    
    # Key Insights
    st.subheader("💡 Key Insights")
    
    st.markdown(f"""
    1. **AUC-ROC Improvement:** The Hybrid model shows a **{(auc_hybrid - auc_prophet):.4f}** improvement in AUC-ROC score.
    
    2. **Accuracy Threshold:** Predictions within ±{ACCURACY_THRESHOLD} reports are considered accurate for operational planning.
    
    3. **Model Performance:** The Hybrid model (Prophet + XGBoost) achieves **{hybrid_accuracy_rate:.2f}%** accuracy 
       compared to **{prophet_accuracy_rate:.2f}%** for Prophet alone.
    
    4. **Practical Impact:** The improved accuracy translates to better resource allocation and reduced operational risks in IaaS environments.
    """)
    
else:
    st.error("❌ Failed to load comparison data. Please check data availability.")

st.markdown("---")
st.caption("Model Comparison Dashboard | Thesis Project - Report Monitoring System")
