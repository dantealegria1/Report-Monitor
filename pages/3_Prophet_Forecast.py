"""
Prophet Forecast Page (Day 3)
Trains Prophet model and compares performance against naive baseline.
"""
import streamlit as st
import polars as pl
import pandas as pd
import numpy as np
import math
import plotly.graph_objects as go

# Prophet utilities
from utils.prophet_forecast import (
    train_test_split_temporal,
    train_prophet_model,
    generate_prophet_predictions,
    compute_metrics,
    get_forecast_components,
    compute_mase,
    run_prophet_backtesting
)
from components.kpis import render_mase_kpi

# Baseline utilities for comparison
from utils.baseline import add_naive_predictions

st.set_page_config(page_title="Prophet Forecast", layout="wide")

# Inject Custom CSS
try:
    with open("assets/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

def _infer_columns(hourly_ts: pl.DataFrame) -> tuple[str, str]:
    """
    Infer timestamp and target columns from DataFrame.
    """
    cols = set(hourly_ts.columns)
    
    ts_candidates = ["timestamp_hour", "hour", "ts", "timestamp"]
    y_candidates = ["y", "count", "report_count", "reports", "n"]
    
    ts_col = next((c for c in ts_candidates if c in cols), None)
    y_col = next((c for c in y_candidates if c in cols), None)
    
    if not ts_col or not y_col:
        raise ValueError(
            f"Could not infer columns. Available: {hourly_ts.columns}\n"
            f"Need timestamp_hour + report_count (or y/count)."
        )
    
    return ts_col, y_col


def _baseline_metrics(df: pl.DataFrame, y_col: str, pred_col: str) -> tuple[float | None, float | None, int]:
    """Calculate MAE and RMSE for baseline predictions."""
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


def render_prophet_page(hourly_ts: pl.DataFrame):
    ts_col, y_col = _infer_columns(hourly_ts)
    
    # Parse timestamp if needed
    df = hourly_ts
    if df.schema[ts_col] == pl.Utf8:
        df = df.with_columns(
            pl.col(ts_col)
            .str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S%.f", strict=False)
            .alias(ts_col)
        )
    
    df = df.sort(ts_col).with_columns(pl.col(y_col).fill_null(0))
    
    st.title("Prophet Forecast")
    # ----------------------------
    # Model Training & Prediction
    # ----------------------------
    ts_col = "timestamp_hour"
    y_col = "report_count"
    
    # Train/test split configuration
    st.sidebar.header("Model Configuration")
    
    # Simple Mode selection
    config_mode = st.sidebar.radio(
        "Configuration Mode",
        ["Expert Preset (Recommended)", "Custom (Manual)"],
        index=0,
        help="Expert Preset uses optimized parameters to beat the baseline."
    )
    
    if config_mode == "Expert Preset (Recommended)":
        # Expert Preset settings
        changepoint_prior_scale = 0.15
        seasonality_prior_scale = 10.0
        seasonality_mode = 'multiplicative'
        changepoint_range = 0.9
        daily_seasonality = True
        weekly_seasonality = True
        yearly_seasonality = True
        log_transform = True
        growth = 'linear'
        country_holidays = 'US'
        add_lag1 = True
        train_ratio = 0.8
        
        st.sidebar.info("Expert Settings Applied:\n"
                        "- Seasonality: All Enabled\n"
                        "- Holidays: US | Log Transform: On\n"
                        "- Hybrid Mode: On | Ratio: 80%")
    else:
        # Custom manual controls
        yearly_seasonality = st.sidebar.checkbox("Yearly Seasonality", value=True)
        weekly_seasonality = st.sidebar.checkbox("Weekly Seasonality", value=True)
        daily_seasonality = st.sidebar.checkbox("Daily Seasonality", value=True)
        
        log_transform = st.sidebar.checkbox("Log Transform", value=True)
        add_lag1 = st.sidebar.checkbox("Hybrid Mode", value=True)
        
        country_holidays = st.sidebar.selectbox(
            "Country Holidays",
            [None, 'MX', 'US'],
            index=2,
            help="Add holiday effects for a specific country."
        )
        
        train_ratio = st.sidebar.slider(
            "Training Data Ratio",
            key="slider_train_ratio",
            min_value=0.5,
            max_value=0.95,
            value=0.8,
            step=0.05,
            help="Proportion of data used for training"
        )
        
        # Hidden or defaults for others in manual mode for simplicity
        changepoint_prior_scale = 0.05
        seasonality_prior_scale = 10.0
        seasonality_mode = 'additive'
        changepoint_range = 0.8
        growth = 'linear'

    # Split data
    train_df, test_df = train_test_split_temporal(df, train_ratio=train_ratio, ts_col=ts_col, y_col=y_col)
    # Caching the model training to avoid re-running on every interaction
    @st.cache_resource(show_spinner="Training Prophet model...", ttl=3600)
    def train_prophet_model_cached(train_df, **kwargs):
        return train_prophet_model(train_df, **kwargs)

    # Pre-trained Model Support
    import os
    from prophet.serialize import model_from_json
    
    use_pretrained = False
    if os.path.exists('prophet_model.json'):
        use_pretrained = st.sidebar.checkbox("Load Pre-trained Model (prophet_model.json)", value=True)

    # Load performance metrics if they exist for consistency
    perf_metrics = None
    import os, json
    if os.path.exists('metrics.json'):
        try:
            with open('metrics.json', 'r') as f:
                perf_metrics = json.load(f)
        except:
            pass

    # Train or Load Prophet model
    try:
        if use_pretrained:
            with open('prophet_model.json', 'r') as f:
                model = model_from_json(f.read())
            
            # trainer.py currently DOES NOT use log_transform or internal lag1 in Prophet
            log_used = False 
            add_lag1 = False 
            
            st.success("Pre-trained Prophet model loaded successfully!")
        else:
            model, log_used = train_prophet_model_cached(
                train_df,
                ts_col=ts_col,
                y_col=y_col,
                daily_seasonality=daily_seasonality,
                weekly_seasonality=weekly_seasonality,
                yearly_seasonality=yearly_seasonality,
                changepoint_prior_scale=changepoint_prior_scale,
                seasonality_prior_scale=seasonality_prior_scale,
                seasonality_mode=seasonality_mode,
                changepoint_range=changepoint_range,
                log_transform=log_transform,
                growth=growth,
                country_holidays=country_holidays,
                add_lag1=add_lag1
            )
            st.success("Prophet model trained successfully (Cached)!")
    except Exception as e:
        st.error(f"Error loading/training Prophet model: {e}")
        st.stop()
    
    # Get last training value for lag1 continuity
    last_y_train = train_df.select(y_col).tail(1).item()
    if log_used:
        last_y_train = np.log1p(last_y_train)

    # Generate predictions
    with st.spinner("Generating predictions..."):
        test_predictions = generate_prophet_predictions(
            model, test_df, ts_col=ts_col, y_col=y_col, 
            include_intervals=True, inverse_log=log_used,
            add_lag1=add_lag1, train_df_last_y=last_y_train
        )
        # Also generate train predictions for diagnostic purposes
        # Note: for simplicity, train_predictions uses lag1 if add_lag1 is True, 
        # but internal lag1 is handled within model.predict on training data differently.
        # We pass None for train_df_last_y since it will shift within its own data.
        train_predictions = generate_prophet_predictions(
            model, train_df, ts_col=ts_col, y_col=y_col,
            include_intervals=False, inverse_log=log_used,
            add_lag1=add_lag1
        )
    
    # Compute Prophet metrics
    prophet_test_metrics = compute_metrics(test_df, test_predictions, y_col=y_col, pred_col="y_pred_prophet", ts_col=ts_col)
    prophet_train_metrics = compute_metrics(train_df, train_predictions, y_col=y_col, pred_col="y_pred_prophet", ts_col=ts_col)
    
    # Compute baseline metrics on test set (RESTORED MISSING BLOCK)
    test_with_baseline = add_naive_predictions(test_df, y_col=y_col)
    baseline_mae_lh, baseline_rmse_lh, n_lh = _baseline_metrics(test_with_baseline, y_col, "y_pred_last_hour")
    baseline_mae_lw, baseline_rmse_lw, n_lw = _baseline_metrics(test_with_baseline, y_col, "y_pred_last_week")
    
    # --- Prepare Visualization Data ---
    # Merge test predictions with baseline
    # Merge test predictions with actuals and baseline
    # Merge test predictions with actuals and baseline
    plot_data = (
        test_predictions
        .join(test_df.select([ts_col, y_col]), on=ts_col)
        .rename({y_col: "actual"})
        .join(test_with_baseline.select([ts_col, "y_pred_last_hour", "y_pred_last_week"]), on=ts_col)
    )
    plot_data = plot_data.with_columns(
        pl.col(ts_col).dt.strftime("%Y-%m-%dT%H:%M:%S").alias("timestamp_str")
    )
    
    # Prepare chart layers
    layers = [
        # Confidence Interval
        {
            "mark": {"type": "area", "color": "#1f77b4", "opacity": 0.2},
            "encoding": {
                "x": {"field": "timestamp_str", "type": "temporal"},
                "y": {"field": "yhat_lower", "type": "quantitative"},
                "y2": {"field": "yhat_upper"}
            }
        },
        # Actual values
        {
            "mark": {"type": "line", "color": "#333", "strokeWidth": 1, "opacity": 0.5},
            "encoding": {
                "x": {"field": "timestamp_str", "type": "temporal"},
                "y": {"field": "actual", "type": "quantitative"},
                "tooltip": [{"field": "timestamp_str", "type": "temporal"}, {"field": "actual"}]
            }
        },
        # Prophet Prediction
        {
            "mark": {"type": "line", "color": "#1f77b4", "strokeWidth": 2},
            "encoding": {
                "x": {"field": "timestamp_str", "type": "temporal"},
                "y": {"field": "y_pred_prophet", "type": "quantitative"},
                "tooltip": [{"field": "y_pred_prophet", "title": "Prophet Pred"}]
            }
        }
    ]
    
    # Prepare Residuals data
    # Prepare Residuals data
    # residuals_df is derived from plot_data (which now has 'actual' and 'y_pred_prophet')
    residuals_df = plot_data.with_columns(
        (pl.col("actual") - pl.col("y_pred_prophet")).alias("residual")
    )
    
    # ACF Calculation
    import scipy.stats as stats
    y_values = residuals_df["residual"].to_numpy()
    y_var = np.var(y_values)
    acf_values = []
    if y_var > 0:
        for lag in range(1, 25):
            corr = np.corrcoef(y_values[lag:], y_values[:-lag])[0, 1]
            acf_values.append({"lag": lag, "correlation": corr})

    # --- Metrics & Results Tabs ---
    # --- Metrics & Results Tabs ---
    st.divider()
    
    # --- Global Controls ---
    st.markdown("### Visualization Controls")
    
    # Date Range Picker
    min_date = plot_data.select(pl.col(ts_col).min()).item()
    max_date = plot_data.select(pl.col(ts_col).max()).item()
    default_start = max_date - pd.Timedelta(days=7)
    
    c_global_date1, c_global_date2 = st.columns([1, 3])
    with c_global_date1:
        date_range = st.date_input(
            "Analysis Period",
            value=(default_start, max_date),
            min_value=min_date,
            max_value=max_date,
            key="prophet_date_range_global"
        )
    
    if len(date_range) == 2:
        start_dt, end_dt = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    else:
        start_dt, end_dt = default_start, max_date

    filtered_plot_data = plot_data.filter((pl.col(ts_col) >= start_dt) & (pl.col(ts_col) <= end_dt))
    filtered_test_df = test_df.filter((pl.col(ts_col) >= start_dt) & (pl.col(ts_col) <= end_dt))

    tab1, tab2, tab3, tab4 = st.tabs([
        "Performance Dashboard", 
        "Forecast Analysis", 
        "Model Insights",
        "Historical Validation"
    ])

    with tab1:
        st.subheader("Performance Comparison Dashboard")
        mc1, mc2, mc3 = st.columns(3)
        
        display_mae = prophet_test_metrics['MAE']
        display_rmse = prophet_test_metrics['RMSE']
        if use_pretrained and perf_metrics:
            display_mae = perf_metrics.get('mae_p', display_mae)
            display_rmse = perf_metrics.get('rmse_p', display_rmse)

        with mc1:
            st.metric("Prophet MAE", f"{display_mae:.4f}" if display_mae is not None else "N/A")
        with mc2:
            st.metric("Prophet RMSE", f"{display_rmse:.4f}" if display_rmse is not None else "N/A")
        with mc3:
            if use_pretrained and perf_metrics:
                st.metric("Prophet MASE", f"{perf_metrics.get('mase_p', 0):.4f}")
            
        st.divider()
        st.markdown("### Academic Validation (MASE)")
        col_m1, col_m2 = st.columns([1, 2])
        with col_m1:
            mase_1 = compute_mase(
                training_df=train_df, test_df=test_df, predictions=test_predictions,
                y_col=y_col, pred_col="y_pred_prophet", ts_col=ts_col, m=1
            )
            render_mase_kpi(mase_1)
        with col_m2:
            st.info("MASE < 1.0 indicates the model outperforms the seasonal naive baseline.")

        st.divider()
        st.subheader("Prediction vs Actual (Test Set)")
        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(
            x=filtered_plot_data[ts_col].to_pandas(), y=filtered_plot_data["yhat_upper"].to_numpy(),
            fill=None, mode='lines', line_color='rgba(31, 119, 180, 0)', showlegend=False
        ))
        fig_p.add_trace(go.Scatter(
            x=filtered_plot_data[ts_col].to_pandas(), y=filtered_plot_data["yhat_lower"].to_numpy(),
            fill='tonexty', mode='lines', line_color='rgba(31, 119, 180, 0)',
            fillcolor='rgba(31, 119, 180, 0.2)', name="Confidence Interval"
        ))
        fig_p.add_trace(go.Scatter(x=filtered_plot_data[ts_col].to_pandas(), y=filtered_plot_data["actual"].to_numpy(), name="Actual", line=dict(color='red'), opacity=0.6))
        fig_p.add_trace(go.Scatter(x=filtered_plot_data[ts_col].to_pandas(), y=filtered_plot_data["y_pred_prophet"].to_numpy(), name="Prophet", line=dict(color='#1f77b4', width=3)))
        fig_p.update_layout(height=400, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_p, width='stretch')

        st.divider()
        st.subheader("Prediction Sample (Latest Records)")
        st.dataframe(filtered_plot_data.select([ts_col, "actual", "y_pred_prophet"]).tail(20).to_pandas(), width='stretch')

    with tab2:
        st.subheader("Detailed Time Series Analysis")
        show_ci = st.checkbox("Show Confidence Intervals", value=True, key="prophet_show_ci")
        fig_detail = go.Figure()
        if show_ci:
            fig_detail.add_trace(go.Scatter(x=filtered_plot_data[ts_col].to_pandas(), y=filtered_plot_data["yhat_upper"].to_numpy(), fill=None, mode='lines', line_color='rgba(31, 119, 180, 0)', showlegend=False))
            fig_detail.add_trace(go.Scatter(x=filtered_plot_data[ts_col].to_pandas(), y=filtered_plot_data["yhat_lower"].to_numpy(), fill='tonexty', mode='lines', line_color='rgba(31, 119, 180, 0)', fillcolor='rgba(31, 119, 180, 0.2)', name="Confidence Interval"))
        
        fig_detail.add_trace(go.Scatter(x=filtered_plot_data[ts_col].to_pandas(), y=filtered_plot_data["y_pred_last_hour"].to_numpy(), name="Naive (Last Hour)", line=dict(color='#ff7f0e', dash='dash')))
        fig_detail.add_trace(go.Scatter(x=filtered_plot_data[ts_col].to_pandas(), y=filtered_plot_data["y_pred_last_week"].to_numpy(), name="Naive (Last Week)", line=dict(color='#2ca02c', dash='dot')))
        fig_detail.add_trace(go.Scatter(x=filtered_plot_data[ts_col].to_pandas(), y=filtered_plot_data["actual"].to_numpy(), name="Actual", line=dict(color='red'), opacity=0.6))
        fig_detail.add_trace(go.Scatter(x=filtered_plot_data[ts_col].to_pandas(), y=filtered_plot_data["y_pred_prophet"].to_numpy(), name="Prophet", line=dict(color='#1f77b4', width=3)))
        fig_detail.update_layout(height=450, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_detail, width='stretch')

    with tab3:
        st.subheader("Model Insights")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Seasonality & Trends**")
            forecast_components = get_forecast_components(model, filtered_test_df, ts_col=ts_col, y_col=y_col, inverse_log=log_used, add_lag1=add_lag1, train_df_last_y=last_y_train)
            st.line_chart(pd.DataFrame({'trend': forecast_components['trend']}).set_index(forecast_components['ds']))
        with c2:
            st.markdown("**Error Distribution**")
            fig_res2 = go.Figure(go.Histogram(x=residuals_df["residual"].to_numpy(), nbinsx=30, marker_color="#9467bd"))
            fig_res2.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig_res2, width='stretch')
            
        st.divider()
        st.subheader("Autocorrelation (ACF)")
        st.markdown("High correlation at Lag 1 explains why the baseline is challenging to outperform.")
        if y_var > 0:
            acf_df = pd.DataFrame(acf_values)
            fig_acf = go.Figure(go.Bar(x=acf_df["lag"], y=acf_df["correlation"], marker_color="#1f77b4"))
            fig_acf.update_layout(height=250, margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig_acf, width='stretch')
        
        with st.expander("Technical Model Details"):
            st.write(f"Changepoints detected: {len(model.changepoints)}")
            st.write(f"Seasonality Mode: {seasonality_mode}")

    with tab4:
        st.subheader("Historical Validation (Backtesting)")
        st.markdown(" operação real simulation using Rolling Windows for operational stability analysis.")
        with st.expander("Backtesting Configuration", expanded=True):
            bc1, bc2, bc3 = st.columns(3)
            with bc1: cv_initial = st.text_input("Initial Window", value="21 days")
            with bc2: cv_period = st.text_input("Period", value="3 days")
            with bc3: cv_horizon = st.text_input("Horizon", value="2 days")

        if st.button("Run Cross-Validation"):
            with st.spinner("Running Backtesting..."):
                try:
                    df_cv, df_p = run_prophet_backtesting(model, initial=cv_initial, period=cv_period, horizon=cv_horizon)
                    st.success("Backtesting completed!")
                    st.markdown("### Performance Metrics by Horizon")
                    st.dataframe(df_p, width='stretch')
                    st.line_chart(df_p.set_index('horizon')['mae'])
                except Exception as e:
                    st.error(f"Error: {e}")


# ----------------------------
# Page Entry
# ----------------------------
hourly_ts = st.session_state.get("hourly_ts")

if hourly_ts is None:
    st.warning("No hourly_ts found in session_state. Please open the main page first to generate it.")
else:
    if not isinstance(hourly_ts, pl.DataFrame):
        st.error(f"hourly_ts is not a Polars DataFrame. Type: {type(hourly_ts)}")
    else:
        render_prophet_page(hourly_ts)
