"""
Prophet Forecast Page (Day 3)
Trains Prophet model and compares performance against naive baseline.
"""
import streamlit as st
import polars as pl
import pandas as pd
import numpy as np
import math

# Prophet utilities
from utils.prophet_forecast import (
    train_test_split_temporal,
    train_prophet_model,
    generate_prophet_predictions,
    compute_metrics,
    get_forecast_components
)

# Baseline utilities for comparison
from utils.baseline import add_naive_predictions

st.set_page_config(page_title="Prophet Forecast", layout="wide")


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
    # Forecasting Wiki
    # ----------------------------
    with st.expander("Forecasting Wiki & Guide - How to beat the baseline"):
        st.markdown("""
        ### Understanding the Settings
        *   **Changepoint Prior Scale**: Controls the 'flexibility' of the trend. 
            *   *Increase* if the trend changes rapidly or has sudden shifts.
            *   *Decrease* if the model is following noise too closely (overfitting).
        *   **Seasonality Prior Scale**: Controls how much the model follows periodic patterns (daily/weekly).
            *   *Increase* if you clearly see repeating shapes every day/week.
        *   **Log Transform (log1p)**: It calculates error in percentages rather than absolute units, which handles spikes much better.
        *   **Growth Mode**: 
            *   *Linear*: Best for data that is generally increasing or decreasing.
            *   *Flat*: Best for data that stays around a constant average but has spikes.

        ### Why is the Naive Baseline winning?
        The **Baseline (last hour)** simply says "Next hour will be the same as this hour". 
        *   If your data has high **Autocorrelation** (value at $t \approx t-1$), the baseline is very efficient.
        *   Prophet tries to find the *logical signal*. If the data is very noisy, the signal is harder to find than just copying the previous value.
        
        **Strategy to win:**
        1. Enable **Log Transform**.
        2. Set **Changepoint Prior Scale** higher (0.1 to 0.3) so Prophet can 'react' faster.
        3. Use **Multiplicative** seasonality if the spikes get bigger as the total count increases.
        """)
    
    st.markdown("""
    This page trains a Prophet forecasting model and compares its performance 
    against the naive baseline from Day 2. Use the sidebar to tune settings.
    """)

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
        # User requested settings for Expert Preset
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
                        "- Daily: On | Weekly: On | Yearly: On\n"
                        "- US Holidays | Log Transform: On\n"
                        "- Hybrid Mode: On | Ratio: 80%")
    else:
        # Minimalist manual controls consolidated into one section
        yearly_seasonality = st.sidebar.checkbox("Yearly Seasonality", value=True)
        weekly_seasonality = st.sidebar.checkbox("Weekly Seasonality", value=True)
        daily_seasonality = st.sidebar.checkbox("Daily Seasonality", value=True)
        
        log_transform = st.sidebar.checkbox("Log Transform", value=True)
        add_lag1 = st.sidebar.checkbox("Hybrid Mode", value=True)
        
        country_holidays = st.sidebar.selectbox(
            "Country Holidays",
            [None, 'MX', 'US'],
            index=2, # Default to US as requested
            help="Add holiday effects for a specific country."
        )
        
        train_ratio = st.sidebar.slider(
            "Training Data Ratio",
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
    
    # Display split info
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Hours", f"{df.height:,}")
    col2.metric("Training Hours", f"{train_df.height:,}")
    col3.metric("Test Hours", f"{test_df.height:,}")
    
    st.divider()
    
    # Help message for low performance
    if st.session_state.get("last_improvement", 0) < 0:
        st.info("Tip for Improvement: To beat a strong baseline, try: \n"
                "1. Enabling Log Transform (highly effective for count data)\n"
                "2. Increasing Changepoint Prior Scale to 0.1 or 0.2\n"
                "3. Setting Seasonality Mode to Multiplicative if variance grows with the mean.")
    
    # Train Prophet model
    with st.spinner("Training Prophet model..."):
        try:
            model, log_used = train_prophet_model(
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
            st.success("Prophet model trained successfully!")
        except Exception as e:
            st.error(f"Error training Prophet model: {e}")
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
    
    # Compute baseline metrics on test set
    test_with_baseline = add_naive_predictions(test_df, y_col=y_col)
    baseline_mae_lh, baseline_rmse_lh, n_lh = _baseline_metrics(test_with_baseline, y_col, "y_pred_last_hour")
    baseline_mae_lw, baseline_rmse_lw, n_lw = _baseline_metrics(test_with_baseline, y_col, "y_pred_last_week")
    
    # Display metrics comparison
    st.subheader("Model Performance Comparison")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Prophet MAE (Test)",
            f"{prophet_test_metrics['MAE']:.4f}" if prophet_test_metrics['MAE'] is not None else "N/A",
            help="Mean Absolute Error on test set"
        )
    
    with col2:
        st.metric(
            "Prophet MAE (Train)",
            f"{prophet_train_metrics['MAE']:.4f}" if prophet_train_metrics['MAE'] is not None else "N/A",
            help="Mean Absolute Error on training set (diagnostics)"
        )
    
    with col3:
        if baseline_mae_lh is not None and prophet_test_metrics['MAE'] is not None:
            improvement = ((baseline_mae_lh - prophet_test_metrics['MAE']) / baseline_mae_lh) * 100
            st.session_state["last_improvement"] = improvement
            st.metric(
                "Improvement over Baseline",
                f"{improvement:.2f}%",
                delta=f"{improvement:.2f}%",
                help="Percentage improvement in MAE compared to last hour baseline on Test set"
            )
        else:
            st.metric("Improvement", "N/A")
    
    with col4:
        if prophet_train_metrics['MAE'] and prophet_test_metrics['MAE']:
            ratio = prophet_test_metrics['MAE'] / prophet_train_metrics['MAE']
            st.metric(
                "Overfitting Ratio",
                f"{ratio:.2f}x",
                help="Test MAE / Train MAE. Values > 1.5x suggest overfitting."
            )
    
    st.divider()
    
    # Baseline comparison table
    st.subheader("Baseline Comparison (Test Set)")
    comparison_data = {
        "Model": ["Prophet", "Baseline (last hour)", "Baseline (last week)"],
        "MAE": [
            f"{prophet_test_metrics['MAE']:.4f}" if prophet_test_metrics['MAE'] is not None else "N/A",
            f"{baseline_mae_lh:.4f}" if baseline_mae_lh is not None else "N/A",
            f"{baseline_mae_lw:.4f}" if baseline_mae_lw is not None else "N/A"
        ],
        "RMSE": [
            f"{prophet_test_metrics['RMSE']:.4f}" if prophet_test_metrics['RMSE'] is not None else "N/A",
            f"{baseline_rmse_lh:.4f}" if baseline_rmse_lh is not None else "N/A",
            f"{baseline_rmse_lw:.4f}" if baseline_rmse_lw is not None else "N/A"
        ],
        "Samples": [
            prophet_test_metrics['n_samples'],
            n_lh,
            n_lw
        ]
    }
    st.dataframe(comparison_data, use_container_width=True)
    
    st.divider()
    
    # Visualization controls
    st.subheader("📈 Forecast Visualization")
    days_to_show = st.slider("Visualization Window (days)", min_value=1, max_value=30, value=7)
    hours_to_show = int(days_to_show * 24)
    
    # Prepare data for visualization
    test_with_pred = test_df.join(test_predictions, on=ts_col, how="inner")
    test_with_pred = test_with_pred.tail(hours_to_show)
    
    # Add baseline predictions for comparison
    test_with_pred = add_naive_predictions(test_with_pred, y_col=y_col)
    
    # Convert to format for plotting
    plot_data = (
        test_with_pred.with_columns(
            pl.col(ts_col).dt.strftime("%Y-%m-%dT%H:%M:%S").alias("timestamp_str")
        )
        .rename({y_col: "actual"})
        .select(["timestamp_str", "actual", "y_pred_prophet", "y_pred_last_hour", "yhat_lower", "yhat_upper"])
    )
    
    chart_data = plot_data.to_dicts()
    
    show_intervals = st.checkbox("Show Prediction Intervals (Uncertainty)", value=True)
    
    # Actual vs Predictions chart with intervals
    layers = [
        # Main lines
        {
            "mark": {"type": "line", "point": False},
            "encoding": {
                "x": {"field": "timestamp_str", "type": "temporal", "title": "Time"},
                "y": {"field": "value", "type": "quantitative", "title": "Report Count"},
                "color": {
                    "field": "series",
                    "type": "nominal",
                    "title": "Series",
                    "scale": {
                        "domain": ["actual", "y_pred_prophet", "y_pred_last_hour"],
                        "range": ["#1f77b4", "#ff7f0e", "#2ca02c"]
                    }
                }
            },
            "transform": [
                {
                    "fold": ["actual", "y_pred_prophet", "y_pred_last_hour"],
                    "as": ["series", "value"]
                }
            ]
        }
    ]
    
    if show_intervals:
        layers.insert(0, {
            "mark": {"type": "area", "opacity": 0.3, "color": "#ff7f0e"},
            "encoding": {
                "x": {"field": "timestamp_str", "type": "temporal"},
                "y": {"field": "yhat_lower", "type": "quantitative"},
                "y2": {"field": "yhat_upper", "type": "quantitative"},
                "tooltip": None
            }
        })

    st.vega_lite_chart(
        chart_data,
        {
            "layer": layers,
            "width": "container",
            "height": 450
        },
        use_container_width=True
    )
    
    st.divider()
    
    # Forecast components
    st.subheader("Forecast Components")
    
    with st.expander("View Prophet Forecast Components"):
        forecast_components = get_forecast_components(
            model, test_df.tail(hours_to_show), 
            ts_col=ts_col, y_col=y_col,
            inverse_log=log_used, add_lag1=add_lag1, 
            train_df_last_y=last_y_train
        )
        
        # Trend component
        st.markdown("**Trend Component**")
        trend_data = pd.DataFrame({
            'ds': forecast_components['ds'],
            'trend': forecast_components['trend']
        })
        st.line_chart(trend_data.set_index('ds'))
        
        # Weekly seasonality (if available)
        if 'weekly' in forecast_components.columns:
            st.markdown("**Weekly Seasonality**")
            weekly_data = pd.DataFrame({
                'ds': forecast_components['ds'],
                'weekly': forecast_components['weekly']
            })
            st.line_chart(weekly_data.set_index('ds'))
        
        # Daily seasonality (if available)
        if 'daily' in forecast_components.columns:
            st.markdown("**Daily Seasonality**")
            daily_data = pd.DataFrame({
                'ds': forecast_components['ds'],
                'daily': forecast_components['daily']
            })
            st.line_chart(daily_data.set_index('ds'))
    
    st.divider()
    
    # Residual analysis
    st.subheader("Residual Analysis (Test Set)")
    
    residuals_df = plot_data.with_columns(
        (pl.col("actual") - pl.col("y_pred_prophet")).alias("residual")
    )
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("**Residuals over Time**")
        st.vega_lite_chart(
            residuals_df.select(["timestamp_str", "residual"]).to_dicts(),
            {
                "mark": {"type": "bar", "color": "#d62728"},
                "encoding": {
                    "x": {"field": "timestamp_str", "type": "temporal", "title": "Time"},
                    "y": {"field": "residual", "type": "quantitative", "title": "Residual (Actual - Predicted)"}
                },
                "width": "container",
                "height": 300
            },
            use_container_width=True
        )
        
    with c2:
        st.markdown("**Error Distribution (Histogram)**")
        st.vega_lite_chart(
            residuals_df.select("residual").to_dicts(),
            {
                "mark": {"type": "bar", "color": "#9467bd"},
                "encoding": {
                    "x": {"bin": True, "field": "residual", "title": "Error Magnitude"},
                    "y": {"aggregate": "count", "title": "Frequency"}
                },
                "width": "container",
                "height": 300
            },
            use_container_width=True
        )
    
    st.divider()
    
    # Model Diagnostics
    st.subheader("Model Diagnostics")
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("**Changepoints Detected**")
        # Extract changepoints from model
        cp_dates = model.changepoints
        st.write(f"Number of potential changepoints: {len(cp_dates)}")
        st.write(f"Changepoint Prior Scale: {changepoint_prior_scale:.4f}")
        
    with c2:
        st.markdown("**Seasonality Strength**")
        st.write(f"Seasonality Prior Scale: {seasonality_prior_scale:.2f}")
        st.write(f"Seasonality Mode: {seasonality_mode}")
    
    # Advanced Diagnostics
    with st.expander("Advanced Diagnostics (Autocorrelation)"):
        st.markdown("""
        **Why is the baseline so strong?**
        If the data has high **Autocorrelation at Lag 1**, it means the value at hour *t* is very similar 
        to the value at hour *t-1*. This explains why the 'Last Hour' baseline is difficult to beat.
        """)
        
        # Calculate ACF manually for lags 1-24
        lags = list(range(1, 25))
        acf_values = []
        
        y_values = plot_data.select("actual").to_series()
        y_mean = y_values.mean()
        y_var = y_values.var()
        
        if y_var > 0:
            for lag in lags:
                # Eccnomic definition of ACF: cov(y_t, y_{t-lag}) / var(y)
                # We use Polars shifts to align and calculate correlation
                acf = plot_data.select([
                    pl.corr(pl.col("actual"), pl.col("actual").shift(lag)).alias("acf")
                ]).item()
                acf_values.append({"lag": lag, "correlation": acf if acf is not None else 0.0})
        
            st.vega_lite_chart(
                acf_values,
                {
                    "mark": "bar",
                    "encoding": {
                        "x": {"field": "lag", "type": "quantitative", "title": "Lag (Hours)"},
                        "y": {"field": "correlation", "type": "quantitative", "title": "Autocorrelation"},
                        "color": {
                            "condition": {"test": "datum.correlation > 0.5", "value": "#1f77b4"},
                            "value": "#aec7e8"
                        }
                    },
                    "width": "container",
                    "height": 250
                },
                use_container_width=True
            )
        else:
            st.warning("Variance is zero; cannot calculate autocorrelation.")

    st.divider()
    
    # Data sample
    st.subheader("Sample Predictions")
    sample_data = (
        plot_data.select(["timestamp_str", "actual", "y_pred_prophet", "y_pred_last_hour"])
        .tail(50)
        .to_dicts()
    )
    st.dataframe(sample_data, use_container_width=True)


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
