"""
Prophet forecasting utilities for Day 3.
Handles train/test split, model training, predictions, and evaluation metrics.
"""
import polars as pl
import pandas as pd
import numpy as np
import math
from prophet import Prophet
from typing import Tuple, Dict


def prepare_prophet_dataframe(hourly_ts: pl.DataFrame, ts_col: str = "timestamp_hour", y_col: str = "report_count") -> pd.DataFrame:
    """
    Convert Polars DataFrame to Prophet's required format (ds, y columns).
    
    Args:
        hourly_ts: Polars DataFrame with timestamp and count columns
        ts_col: Name of timestamp column
        y_col: Name of target variable column
    
    Returns:
        Pandas DataFrame with 'ds' (datetime) and 'y' (numeric) columns
    """
    # Convert to pandas and rename columns for Prophet
    df_prophet = hourly_ts.select([ts_col, y_col]).to_pandas()
    df_prophet.columns = ['ds', 'y']
    
    # Ensure ds is datetime
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
    
    return df_prophet


def train_test_split_temporal(
    hourly_ts: pl.DataFrame, 
    train_ratio: float = 0.8,
    ts_col: str = "timestamp_hour",
    y_col: str = "report_count"
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Split hourly time series into train/test sets maintaining chronological order.
    
    Args:
        hourly_ts: Polars DataFrame with hourly time series
        train_ratio: Proportion of data for training (default 0.8)
        ts_col: Name of timestamp column
        y_col: Name of target variable column
    
    Returns:
        Tuple of (train_df, test_df) as Polars DataFrames
    """
    # Sort by timestamp to ensure chronological order
    df_sorted = hourly_ts.sort(ts_col)
    
    # Calculate split point
    n_total = df_sorted.height
    n_train = int(n_total * train_ratio)
    
    # Split without shuffling
    train_df = df_sorted.head(n_train)
    test_df = df_sorted.tail(n_total - n_train)
    
    return train_df, test_df


def train_prophet_model(
    train_df: pl.DataFrame,
    ts_col: str = "timestamp_hour",
    y_col: str = "report_count",
    daily_seasonality: bool = True,
    weekly_seasonality: bool = True,
    yearly_seasonality: bool = False,
    changepoint_prior_scale: float = 0.05,
    seasonality_prior_scale: float = 10.0,
    seasonality_mode: str = 'additive',
    changepoint_range: float = 0.8,
    log_transform: bool = False,
    growth: str = 'linear',
    country_holidays: str = None,
    add_lag1: bool = False
) -> Tuple[Prophet, bool]:
    """
    Train Prophet model with optional log transform and lag-1 regressor.
    """
    # Convert to Prophet format
    df_prophet = prepare_prophet_dataframe(train_df, ts_col, y_col)
    
    if log_transform:
        df_prophet['y'] = np.log1p(df_prophet['y'])
    
    # Initialize Prophet
    model = Prophet(
        growth=growth,
        daily_seasonality=daily_seasonality,
        weekly_seasonality=weekly_seasonality,
        yearly_seasonality=yearly_seasonality,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        seasonality_mode=seasonality_mode,
        changepoint_range=changepoint_range,
        interval_width=0.95
    )
    
    if country_holidays:
        model.add_country_holidays(country_name=country_holidays)
        
    if add_lag1:
        # Create lag-1 column (previous hour)
        df_prophet['lag1'] = df_prophet['y'].shift(1).fillna(0)
        model.add_regressor('lag1')
    
    # Train the model
    model.fit(df_prophet)
    
    return model, log_transform


def generate_prophet_predictions(
    model: Prophet,
    test_df: pl.DataFrame,
    ts_col: str = "timestamp_hour",
    y_col: str = "report_count",
    include_intervals: bool = True,
    inverse_log: bool = False,
    add_lag1: bool = False,
    train_df_last_y: float = None
) -> pl.DataFrame:
    """
    Generate predictions for the test period.
    """
    # Create future dataframe for test period
    future = pd.DataFrame({'ds': test_df.select(ts_col).to_pandas()[ts_col]})
    future['ds'] = pd.to_datetime(future['ds'])
    
    if add_lag1:
        # For the test set, 'lag1' needs to be the actual previous value
        # We need the last value of the training set to start the first test lag
        actuals = test_df.select(y_col).to_pandas()[y_col]
        if inverse_log:
            actuals = np.log1p(actuals)
            
        lags = actuals.shift(1).fillna(train_df_last_y if train_df_last_y is not None else 0)
        future['lag1'] = lags
    
    # Generate predictions
    forecast = model.predict(future)
    
    if inverse_log:
        # Inverse of log1p is expm1
        forecast['yhat'] = np.expm1(forecast['yhat'])
        forecast['yhat_lower'] = np.expm1(forecast['yhat_lower'])
        forecast['yhat_upper'] = np.expm1(forecast['yhat_upper'])
    
    # Extract predictions and convert back to Polars
    if include_intervals:
        predictions = pl.DataFrame({
            ts_col: forecast['ds'].values,
            'y_pred_prophet': forecast['yhat'].values.clip(min=0),  # Counts can't be negative
            'yhat_lower': forecast['yhat_lower'].values.clip(min=0),
            'yhat_upper': forecast['yhat_upper'].values.clip(min=0)
        })
    else:
        predictions = pl.DataFrame({
            ts_col: forecast['ds'].values,
            'y_pred_prophet': forecast['yhat'].values.clip(min=0)
        })
    
    return predictions


def compute_metrics(
    test_df: pl.DataFrame,
    predictions: pl.DataFrame,
    y_col: str = "report_count",
    pred_col: str = "y_pred_prophet",
    ts_col: str = "timestamp_hour"
) -> Dict[str, float]:
    """
    Calculate MAE and RMSE for model evaluation.
    
    Args:
        test_df: Test data with actual values
        predictions: Predictions DataFrame
        y_col: Name of actual values column
        pred_col: Name of predictions column
        ts_col: Name of timestamp column for joining
    
    Returns:
        Dictionary with MAE, RMSE, and number of samples
    """
    # Join actual and predicted values
    df_eval = test_df.join(predictions, on=ts_col, how="inner")
    
    # Drop any nulls
    df_eval = df_eval.select([y_col, pred_col]).drop_nulls()
    
    n = df_eval.height
    
    if n == 0:
        return {"MAE": None, "RMSE": None, "n_samples": 0}
    
    # Calculate MAE
    mae = (
        df_eval.with_columns((pl.col(y_col) - pl.col(pred_col)).abs().alias("ae"))
        .select(pl.col("ae").mean())
        .item()
    )
    
    # Calculate RMSE
    mse = (
        df_eval.with_columns(((pl.col(y_col) - pl.col(pred_col)) ** 2).alias("se"))
        .select(pl.col("se").mean())
        .item()
    )
    rmse = math.sqrt(mse) if mse is not None else None
    
    return {
        "MAE": float(mae) if mae is not None else None,
        "RMSE": float(rmse) if rmse is not None else None,
        "n_samples": n
    }


def get_forecast_components(
    model: Prophet, 
    test_df: pl.DataFrame, 
    ts_col: str = "timestamp_hour",
    y_col: str = "report_count",
    inverse_log: bool = False,
    add_lag1: bool = False,
    train_df_last_y: float = None
) -> pd.DataFrame:
    """
    Get forecast components (trend, seasonality) for visualization.
    """
    # Create future dataframe
    future = pd.DataFrame({'ds': test_df.select(ts_col).to_pandas()[ts_col]})
    future['ds'] = pd.to_datetime(future['ds'])
    
    if add_lag1:
        # We need to provide the same regressor values as during prediction
        actuals = test_df.select(y_col).to_pandas()[y_col]
        if inverse_log:
            actuals = np.log1p(actuals)
            
        lags = actuals.shift(1).fillna(train_df_last_y if train_df_last_y is not None else 0)
        future['lag1'] = lags
    
    # Generate full forecast with components
    forecast = model.predict(future)
    return forecast


def compute_mase(
    training_df: pl.DataFrame,
    test_df: pl.DataFrame,
    predictions: pl.DataFrame,
    y_col: str = "report_count",
    pred_col: str = "y_pred_prophet",
    ts_col: str = "timestamp_hour",
    m: int = 1
) -> float | None:
    """
    Calculate Mean Absolute Scaled Error (MASE).
    
    MASE = MAE_model / MAE_naive
    
    where MAE_naive is calculated on the *training* set using one-step naive:
    MAE_naive = mean(|Y_t - Y_{t-m}|) for t=m+1 to T.
    
    Args:
        training_df: Training data (used for denominator)
        test_df: Test data (used for numerator)
        predictions: Model predictions for test data
        y_col: Name of target column
        pred_col: Name of prediction column
        ts_col: Name of timestamp column
        m: Periodicity (default 1 for non-seasonal Naive as per thesis requirement)
        
    Returns:
        MASE value (float). MASE < 1 indicates model is better than Naive.
    """
    # 1. Calculate MAE of the model on Test set
    model_mae_dict = compute_metrics(test_df, predictions, y_col, pred_col, ts_col)
    mae_model = model_mae_dict["MAE"]
    
    if mae_model is None:
        return None

    # 2. Calculate Mean Absolute Error of Naive on Training set
    train_sorted = training_df.sort(ts_col)
    y_train = train_sorted.select(pl.col(y_col)).to_series()
    
    if len(y_train) <= m:
        return None
        
    # Naive errors: |Y_t - Y_{t-m}|
    naive_errors = (y_train - y_train.shift(m)).abs().drop_nulls()
    mae_naive = naive_errors.mean()
    
    if mae_naive == 0 or mae_naive is None:
        return None
        
    # 3. Calculate MASE
    mase = mae_model / mae_naive
    
    return float(mase)


from prophet.diagnostics import cross_validation, performance_metrics

def run_prophet_backtesting(
    model: Prophet,
    initial: str = '730 days',
    period: str = '30 days',
    horizon: str = '365 days',
    parallel: str = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run rolling window cross-validation for Prophet model.
    
    Args:
        model: Trained Prophet model
        initial: Size of the initial training period (e.g., '180 days')
        period: Spacing between cutoff dates (e.g., '30 days')
        horizon: Forecast horizon for each window (e.g., '15 days')
        
    Returns:
        Tuple of (cv_results_df, cv_metrics_df) as Pandas DataFrames
    """
    df_cv = cross_validation(
        model, 
        initial=initial, 
        period=period, 
        horizon=horizon, 
        parallel=parallel
    )
    df_p = performance_metrics(df_cv)
    return df_cv, df_p
