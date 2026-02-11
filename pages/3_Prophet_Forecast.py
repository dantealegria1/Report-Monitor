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
    
    # Information Alert
    st.info("Advanced forecast based on Meta Prophet. For more details on interpreting these charts, consult the Wiki page.")

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
    
    # Help message for low performance (moved before training/metrics for better visibility)
    if st.session_state.get("last_improvement", 0) < 0:
        st.info("Tip for Improvement: To beat a strong baseline, try: \n"
                "1. Enabling Log Transform (highly effective for count data)\n"
                "2. Increasing Changepoint Prior Scale to 0.1 or 0.2\n"
                "3. Setting Seasonality Mode to Multiplicative if variance grows with the mean.")
    
    
    # Caching the model training to avoid re-running on every interaction
    @st.cache_resource(show_spinner="Training Prophet model...", ttl=3600)
    def train_prophet_model_cached(train_df, **kwargs):
        return train_prophet_model(train_df, **kwargs)

    # Train Prophet model
    try:
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
    
    # ----------------------------
    # Global Visualization Controls
    # ----------------------------
    st.markdown("### Forecast Visualization & Analysis")
    
    # Date Range Picker
    min_date = plot_data.select(pl.col(ts_col).min()).item()
    max_date = plot_data.select(pl.col(ts_col).max()).item()
    
    # Default to last 7 days
    default_start = max_date - pd.Timedelta(days=7)
    
    c_global_date1, c_global_date2 = st.columns([1, 3])
    with c_global_date1:
        date_range = st.date_input(
            "Select Date Range (applies to all charts)",
            value=(default_start, max_date),
            min_value=min_date,
            max_value=max_date,
            key="prophet_date_range_global"
        )
    
    # Determine filter range
    if len(date_range) == 2:
        start_date, end_date = date_range
        # Ensure start_date/end_date are datetime for comparison if they are date objects
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1) # End of day
    else:
        # Fallback if range is invalid
        start_dt = default_start
        end_dt = max_date

    # Filter data for all tabs
    filtered_plot_data = plot_data.filter(
        (pl.col(ts_col) >= start_dt) & (pl.col(ts_col) <= end_dt)
    )
    
    filtered_test_df = test_df.filter(
        (pl.col(ts_col) >= start_dt) & (pl.col(ts_col) <= end_dt)
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "Performance Summary", 
        "Forecast Visualization", 
        "Model Components", 
        "Diagnostics & Residuals"
    ])

    with tab1:
        st.subheader("Model Comparison")
        
        # Key metrics in columns
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.metric(
                "Prophet MAE (Test)",
                f"{prophet_test_metrics['MAE']:.4f}" if prophet_test_metrics['MAE'] is not None else "N/A",
                help="Mean Absolute Error on test set"
            )
        with mc2:
            st.metric(
                "Prophet MAE (Train)",
                f"{prophet_train_metrics['MAE']:.4f}" if prophet_train_metrics['MAE'] is not None else "N/A",
                help="Mean Absolute Error on training set (diagnostics)"
            )
        with mc3:
            if baseline_mae_lh is not None and prophet_test_metrics['MAE'] is not None:
                improvement = ((baseline_mae_lh - prophet_test_metrics['MAE']) / baseline_mae_lh) * 100
                st.session_state["last_improvement"] = improvement
                st.metric(
                    "Improvement vs Baseline",
                    f"{improvement:.2f}%",
                    delta=f"{improvement:.2f}%",
                    help="Percentage improvement in MAE compared to last hour baseline on Test set"
                )
        with mc4:
            if prophet_train_metrics['MAE'] and prophet_test_metrics['MAE']:
                ratio = prophet_test_metrics['MAE'] / prophet_train_metrics['MAE']
                st.metric(
                    "Ratio Overfitting",
                    f"{ratio:.2f}x",
                    help="Test MAE / Train MAE. Values > 1.5x suggest overfitting."
                )

        st.divider()
        st.markdown("**Comparison Table (Test Set)**")
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
        st.markdown("**Prediction Sample**")
        sample_data = (
            filtered_plot_data.select(["timestamp_str", "actual", "y_pred_prophet", "y_pred_last_hour"])
            .tail(20)
            .to_dicts()
        )
        st.dataframe(sample_data, use_container_width=True)

    with tab2:
        st.subheader("Prediction vs Actual")
        
        chart_data = filtered_plot_data.to_dicts()

        # Toggle for Confidence Intervals
        show_ci = st.checkbox("Show Confidence Intervals (yhat_lower/upper)", value=True, key="prophet_show_ci")

        if chart_data:
            # Base layers
            layers = []
            
            # Confidence Interval (Area) - Conditionally added
            if show_ci:
                layers.append({
                    "mark": {"type": "area", "color": "#1f77b4", "opacity": 0.2},
                    "encoding": {
                        "y": {"field": "yhat_lower", "type": "quantitative"},
                        "y2": {"field": "yhat_upper"}
                    }
                })
            
            # Lines with Legend (Folded)
            layers.append({
                "transform": [
                    {"fold": ["actual", "y_pred_prophet", "y_pred_last_hour", "y_pred_last_week"], "as": ["Variable", "Value"]}
                ],
                "mark": "line",
                "encoding": {
                    "y": {"field": "Value", "type": "quantitative", "title": "Report Count"},
                    "color": {
                        "field": "Variable", 
                        "type": "nominal",
                        "scale": {
                            "domain": ["actual", "y_pred_prophet", "y_pred_last_hour", "y_pred_last_week"],
                            "range": ["#ff0303", "#1f77b4", "#ff7f0e", "#2ca02c"]
                        },
                        "legend": {"title": "Model / Series", "orient": "bottom"}
                    },
                    "strokeDash": {
                        "field": "Variable",
                        "scale": {
                            "domain": ["actual", "y_pred_prophet", "y_pred_last_hour", "y_pred_last_week"],
                            "range": [[1,0], [1,0], [5,5], [2,2]]
                        }
                    },
                    "strokeWidth": {
                        "condition": [
                            {"test": "datum.Variable === 'y_pred_prophet'", "value": 3},
                            {"test": "datum.Variable === 'actual'", "value": 2}
                        ],
                        "value": 1.5
                    },
                    "opacity": {
                        "condition": {"test": "datum.Variable === 'actual'", "value": 0.6},
                        "value": 1
                    }
                }
            })

            # Vega-Lite Spec with Fold for Legend
            st.vega_lite_chart(
                chart_data,
                {
                    "encoding": {"x": {"field": "timestamp_str", "type": "temporal", "title": "Time"}},
                    "layer": layers,
                    "width": "container",
                    "height": 500
                },
                use_container_width=True
            )

    with tab3:
        st.subheader("Forecast Components")
        st.markdown("These components show the trends and periodic patterns detected by the model.")
        
        # Use filtered data based on date range instead of potentially crashing hour count
        forecast_components = get_forecast_components(
            model, filtered_test_df, 
            ts_col=ts_col, y_col=y_col,
            inverse_log=log_used, add_lag1=add_lag1, 
            train_df_last_y=last_y_train
        )
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Trend**")
            st.line_chart(pd.DataFrame({'trend': forecast_components['trend']}).set_index(forecast_components['ds']))
        
        with c2:
            if 'daily' in forecast_components.columns:
                st.markdown("**Daily Seasonality**")
                st.line_chart(pd.DataFrame({'daily': forecast_components['daily']}).set_index(forecast_components['ds']))
        
        if 'weekly' in forecast_components.columns:
            st.divider()
            st.markdown("**Weekly Seasonality**")
            st.line_chart(pd.DataFrame({'weekly': forecast_components['weekly']}).set_index(forecast_components['ds']))

    with tab4:
        st.subheader("Residual Analysis & Diagnostics")
        
        rc1, rc2 = st.columns(2)
        with rc1:
            st.markdown("**Residuals over time**")
            st.vega_lite_chart(
                residuals_df.select(["timestamp_str", "residual"]).to_dicts(),
                {
                    "mark": {"type": "bar", "color": "#d62728"},
                    "encoding": {
                        "x": {"field": "timestamp_str", "type": "temporal", "title": "Time"},
                        "y": {"field": "residual", "type": "quantitative", "title": "Error (Actual - predicted)"}
                    },
                    "width": "container",
                    "height": 300
                },
                use_container_width=True
            )
        with rc2:
            st.markdown("**Error Distribution**")
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
        st.subheader("Autocorrelation (ACF)")
        st.markdown("Indicates how much the current value depends on past values. High correlation at Lag 1 explains why the baseline is hard to beat.")
        
        if y_var > 0:
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
        
        st.divider()
        with st.expander("Model Details"):
            st.write(f"Changepoints detected: {len(model.changepoints)}")
            st.write(f"Changepoint Prior Scale: {changepoint_prior_scale:.4f}")
            st.write(f"Seasonality Mode: {seasonality_mode}")


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
