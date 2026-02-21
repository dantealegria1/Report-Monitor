"""
Forecasting Model v4 — XGBoost + avg_param_span_days + Quantile Bands
=======================================================================
What changed vs v3:
  1. avg_param_span_days extracted from Parameters column (JSON/text parsing)
     — reports with wider date spans correlate with higher processing load
  2. Quantile uncertainty bands (p10 / p50 / p90) via three XGBoost models
     — gives a capacity-planning upper bound alongside the point forecast
  3. USE_LGBM early-stopping callback fixed
"""

import re
import json
import warnings
import pyodbc
import numpy as np
import pandas as pd
import polars as pl
import holidays
import xgboost as xgb
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# HELPER — extract date-range span from Parameters column
# ─────────────────────────────────────────────────────────────
_DATE_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2})?|\d{2}/\d{2}/\d{4})"
)

def _parse_span_days(param_str) -> float:
    """
    Extract all dates from a Parameters string (JSON or plain text),
    return the span in days between the earliest and latest date found.
    Returns 0.0 if fewer than 2 dates are found.
    """
    if not isinstance(param_str, str) or not param_str.strip():
        return 0.0
    matches = _DATE_RE.findall(param_str)
    parsed = []
    for m in matches:
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%d/%m/%Y"):
            try:
                parsed.append(datetime.strptime(m, fmt))
                break
            except ValueError:
                continue
    if len(parsed) < 2:
        return 0.0
    return (max(parsed) - min(parsed)).days

# ─────────────────────────────────────────────────────────────
# 0. CONFIG
# ─────────────────────────────────────────────────────────────
COUNTRY     = "AR"
DATA_PATH   = "DatosQuery.csv"
TRAIN_RATIO = 0.80
N_CV_FOLDS  = 5
USE_LGBM    = False   # set True to swap XGBoost for LightGBM


# ─────────────────────────────────────────────────────────────
# 1. DATABASE LOADER
# ─────────────────────────────────────────────────────────────
def load_latest_data_from_db(output_csv: str = DATA_PATH) -> bool:
    print(f"Connecting to database to refresh {output_csv}...")
    try:
        from db.database import get_connection_string
        with open("Query_ReportsData.sql", "r") as f:
            query = f.read()
        query = query.replace("TOP (10)", "").replace("TOP 10", "")
        conn_str = get_connection_string()
        with pyodbc.connect(conn_str) as conn:
            df_sql = pd.read_sql(query, conn)
        df_sql.to_csv(output_csv, index=False)
        print(f"Records saved to {output_csv}")
        return True
    except Exception as e:
        print(f"DB refresh failed: {e}\n   Falling back to existing CSV...")
        return False


# ─────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
def add_fourier_features(
    df: pd.DataFrame,
    t: np.ndarray,
    periods: dict,
    k: int = 3,
) -> pd.DataFrame:
    """Fourier sin/cos features. t is a global time index (not reset per fold)."""
    for name, period in periods.items():
        for i in range(1, k + 1):
            df[f"fourier_{name}_sin_{i}"] = np.sin(2 * np.pi * i * t / period)
            df[f"fourier_{name}_cos_{i}"] = np.cos(2 * np.pi * i * t / period)
    return df


def engineer_features(pdf: pd.DataFrame, raw_df_pl: pl.DataFrame) -> pd.DataFrame:
    """Build all features on top of an hourly-aggregated pandas DataFrame."""
    ch = holidays.CountryHoliday(COUNTRY)

    # ── Calendar ─────────────────────────────────────────────
    pdf["hour"]         = pdf["ds"].dt.hour
    pdf["weekday"]      = pdf["ds"].dt.weekday
    pdf["day_of_month"] = pdf["ds"].dt.day
    pdf["month"]        = pdf["ds"].dt.month
    pdf["is_month_end"] = pdf["ds"].dt.is_month_end.astype(int)
    pdf["is_weekend"]   = (pdf["weekday"] >= 5).astype(int)
    pdf["is_holiday"]   = pdf["ds"].apply(lambda x: int(x in ch))
    pdf["is_peak_hour"] = ((pdf["hour"] >= 8) & (pdf["hour"] <= 18)).astype(int)

    # Interaction features — key for your irregular spike pattern
    pdf["weekend_x_peak"]              = pdf["is_weekend"] * pdf["is_peak_hour"]
    pdf["holiday_x_peak"]              = pdf["is_holiday"] * pdf["is_peak_hour"]
    pdf["weekend_or_holiday_x_peak"]   = (
        ((pdf["is_weekend"] == 1) | (pdf["is_holiday"] == 1)) & (pdf["is_peak_hour"] == 1)
    ).astype(int)

    # ── Cyclical encoding ─────────────────────────────────────
    pdf["hour_sin"]    = np.sin(2 * np.pi * pdf["hour"]    / 24)
    pdf["hour_cos"]    = np.cos(2 * np.pi * pdf["hour"]    / 24)
    pdf["weekday_sin"] = np.sin(2 * np.pi * pdf["weekday"] / 7)
    pdf["weekday_cos"] = np.cos(2 * np.pi * pdf["weekday"] / 7)
    pdf["month_sin"]   = np.sin(2 * np.pi * pdf["month"]   / 12)
    pdf["month_cos"]   = np.cos(2 * np.pi * pdf["month"]   / 12)

    # ── Fourier (global index so folds stay consistent) ───────
    t = np.arange(len(pdf))
    pdf = add_fourier_features(
        pdf, t, periods={"daily": 24, "weekly": 24 * 7}, k=3
    )

    # ── Lag features ─────────────────────────────────────────
    pdf["y_lag_1"]   = pdf["y"].shift(1)
    pdf["y_lag_2"]   = pdf["y"].shift(2)
    pdf["y_lag_3"]   = pdf["y"].shift(3)
    pdf["y_lag_24"]  = pdf["y"].shift(24)    # same hour yesterday
    pdf["y_lag_168"] = pdf["y"].shift(168)   # same hour last week

    shifted = pdf["y"].shift(1)
    pdf["y_rolling_mean_3"]   = shifted.rolling(3).mean()
    pdf["y_rolling_mean_24"]  = shifted.rolling(24).mean()
    pdf["y_rolling_mean_168"] = shifted.rolling(168).mean()
    pdf["y_rolling_std_3"]    = shifted.rolling(3).std()
    pdf["y_rolling_std_24"]   = shifted.rolling(24).std()

    # Log-scale lags — help model learn spike magnitude
    for col in ["y_lag_1", "y_lag_24", "y_lag_168",
                "y_rolling_mean_3", "y_rolling_mean_24", "y_rolling_mean_168"]:
        pdf[f"log_{col}"] = np.log1p(pdf[col].clip(lower=0))

    # ── Backlog ───────────────────────────────────────────────
    print("   Calculating Backlog...")

    def backlog_at(target_ds):
        return raw_df_pl.filter(
            (pl.col("CreationDate") <= target_ds)
            & (pl.col("StartedProcessingOn") > target_ds)
        ).height

    pdf["backlog"] = [backlog_at(ts) for ts in pdf["ds"]]

    # ── Report type label ─────────────────────────────────────
    pdf["tipo_reporte_id"] = pd.factorize(pdf["main_report_type"])[0]

    return pdf


def prepare_data(
    csv_path: str,
    start_date: str = None,
    end_date: str = None,
) -> pd.DataFrame:
    print("Loading and filtering data...")
    df = pl.read_csv(csv_path, ignore_errors=True)

    df = df.with_columns([
        pl.col("CreationDate").str.strptime(
            pl.Datetime, format="%Y-%m-%d %H:%M:%S%.f", strict=False
        ),
        pl.col("StartedProcessingOn").str.strptime(
            pl.Datetime, format="%Y-%m-%d %H:%M:%S%.f", strict=False
        ),
    ]).filter(pl.col("StartedProcessingOn").is_not_null())

    if start_date:
        df = df.filter(
            pl.col("StartedProcessingOn") >= datetime.strptime(start_date, "%Y-%m-%d")
        )
    if end_date:
        df = df.filter(
            pl.col("StartedProcessingOn") <= datetime.strptime(end_date, "%Y-%m-%d")
        )

    hourly_ts = (
        df.with_columns(
            pl.col("StartedProcessingOn").dt.truncate("1h").alias("ds")
        )
        .group_by("ds")
        .agg([
            pl.len().alias("y"),
            pl.col("ReportType").mode().first().alias("main_report_type"),
        ])
        .sort("ds")
    )

    pdf = hourly_ts.to_pandas()

    # ── avg_param_span_days — extract from Parameters column if present ──
    if "Parameters" in df.columns:
        print("   Extracting avg_param_span_days from Parameters column...")
        raw_pd = df.select(["StartedProcessingOn", "Parameters"]).to_pandas()
        raw_pd["ds"] = raw_pd["StartedProcessingOn"].dt.floor("h")
        raw_pd["span_days"] = raw_pd["Parameters"].apply(_parse_span_days)
        span_agg = raw_pd.groupby("ds")["span_days"].mean().reset_index()
        span_agg.columns = ["ds", "avg_param_span_days"]
        pdf = pdf.merge(span_agg, on="ds", how="left")
        pdf["avg_param_span_days"] = pdf["avg_param_span_days"].fillna(0.0)
    else:
        pdf["avg_param_span_days"] = 0.0

    pdf = engineer_features(pdf, df)
    pdf = pdf.dropna().reset_index(drop=True)
    print(f"   Final dataset: {len(pdf):,} hourly observations")
    return pdf


# ─────────────────────────────────────────────────────────────
# 3. FEATURE LIST
# ─────────────────────────────────────────────────────────────
FEATURES = [
    # Operational
    "backlog", "tipo_reporte_id",
    # Calendar
    "hour", "weekday", "day_of_month", "month",
    "is_month_end", "is_weekend", "is_holiday", "is_peak_hour",
    # Interaction — most important for irregular spikes
    "weekend_x_peak", "holiday_x_peak", "weekend_or_holiday_x_peak",
    # Cyclical
    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "month_sin", "month_cos",
    # Fourier
    "fourier_daily_sin_1",  "fourier_daily_cos_1",
    "fourier_daily_sin_2",  "fourier_daily_cos_2",
    "fourier_daily_sin_3",  "fourier_daily_cos_3",
    "fourier_weekly_sin_1", "fourier_weekly_cos_1",
    "fourier_weekly_sin_2", "fourier_weekly_cos_2",
    "fourier_weekly_sin_3", "fourier_weekly_cos_3",
    # Lags
    "y_lag_1", "y_lag_2", "y_lag_3", "y_lag_24", "y_lag_168",
    # Rolling stats
    "y_rolling_mean_3",  "y_rolling_mean_24",  "y_rolling_mean_168",
    "y_rolling_std_3",   "y_rolling_std_24",
    # Log-scale lags
    "log_y_lag_1", "log_y_lag_24", "log_y_lag_168",
    "log_y_rolling_mean_3", "log_y_rolling_mean_24", "log_y_rolling_mean_168",
    # Optional (included if present)
    "avg_param_span_days",
]


# ─────────────────────────────────────────────────────────────
# 4. MODEL
# ─────────────────────────────────────────────────────────────
def build_model(early_stopping: bool = True):
    if USE_LGBM:
        try:
            import lightgbm as lgb
            return lgb.LGBMRegressor(
                n_estimators=1000,
                learning_rate=0.02,
                num_leaves=63,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="mae",
                n_jobs=-1,
            )
        except ImportError:
            print("⚠️  LightGBM not installed — using XGBoost")

    params = dict(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        gamma=1,
        objective="reg:absoluteerror",
        eval_metric="mae",
        tree_method="hist",
    )
    if early_stopping:
        params["early_stopping_rounds"] = 50
    return xgb.XGBRegressor(**params)


def train_model(df: pd.DataFrame, split_idx: int):
    """
    Trains three XGBoost models:
      - p50 (median / point forecast) on log1p(y)  → objective reg:absoluteerror
      - p10 (lower bound)                           → objective reg:quantileerror alpha=0.10
      - p90 (upper bound / capacity planning)       → objective reg:quantileerror alpha=0.90
    Test rows never touch training (leak-free).
    """
    features = [f for f in FEATURES if f in df.columns]
    train_df = df.iloc[:split_idx].copy()
    test_df  = df.iloc[split_idx:].copy()

    y_train_log = np.log1p(train_df["y"].clip(lower=0))
    y_test_log  = np.log1p(test_df["y"].clip(lower=0))

    print(f"\nTraining XGBoost on {len(train_df):,} rows "
          f"(log1p target, {len(features)} features)...")

    def _fit(extra_params: dict, label: str):
        base = dict(
            n_estimators=1000, learning_rate=0.02, max_depth=7,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
            gamma=1, eval_metric="mae", tree_method="hist",
            early_stopping_rounds=50,
        )
        base.update(extra_params)
        if USE_LGBM:
            try:
                import lightgbm as lgb
                lgb_params = dict(
                    n_estimators=1000, learning_rate=0.02, num_leaves=63,
                    min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                    objective="quantile" if "quantile_alpha" in extra_params else "mae",
                    alpha=extra_params.get("quantile_alpha", 0.5),
                    n_jobs=-1,
                )
                m = lgb.LGBMRegressor(**lgb_params)
                m.fit(
                    train_df[features], y_train_log,
                    eval_set=[(test_df[features], y_test_log)],
                    callbacks=[lgb.early_stopping(50, verbose=False),
                               lgb.log_evaluation(period=-1)],
                )
                print(f"   {label} done (LightGBM)")
                return m
            except ImportError:
                print("LightGBM not installed — using XGBoost")
        m = xgb.XGBRegressor(**base)
        m.fit(
            train_df[features], y_train_log,
            eval_set=[(test_df[features], y_test_log)],
            verbose=False,
        )
        print(f"   {label} done")
        return m

    model_p50 = _fit({"objective": "reg:absoluteerror"}, "p50 (point forecast)")
    model_p10 = _fit({"objective": "reg:quantileerror", "quantile_alpha": 0.10}, "p10 (lower bound)")
    model_p90 = _fit({"objective": "reg:quantileerror", "quantile_alpha": 0.90}, "p90 (upper bound)")

    # Predict on full df
    df = df.copy()
    df["y_pred"]    = np.expm1(model_p50.predict(df[features])).clip(min=0)
    df["y_pred_p10"] = np.expm1(model_p10.predict(df[features])).clip(min=0)
    df["y_pred_p90"] = np.expm1(model_p90.predict(df[features])).clip(min=0)

    # Save
    model_p50.save_model("xgboost_model_p50.json")
    model_p10.save_model("xgboost_model_p10.json")
    model_p90.save_model("xgboost_model_p90.json")
    # Keep legacy name pointing at p50 for backward compat
    model_p50.save_model("xgboost_model.json")

    with open("feature_list.json", "w") as f:
        json.dump(features, f)
    label_map = dict(enumerate(pd.factorize(df["main_report_type"])[1]))
    with open("label_mapping.json", "w") as f:
        json.dump(label_map, f)

    print("Models saved (p10 / p50 / p90).")
    return (model_p50, model_p10, model_p90), df, features


# ─────────────────────────────────────────────────────────────
# 5. EVALUATION
# ─────────────────────────────────────────────────────────────
def evaluate_and_report(df: pd.DataFrame, split_idx: int) -> dict:
    train_df = df.iloc[:split_idx]
    test_df  = df.iloc[split_idx:]

    y_train = train_df["y"].values.astype(np.float64)
    y_test  = test_df["y"].values.astype(np.float64)
    y_pred  = test_df["y_pred"].values.astype(np.float64)

    # MASE denominator
    naive_mae_train = np.mean(np.abs(np.diff(y_train)))
    if naive_mae_train < 1e-8:
        naive_mae_train = np.mean(np.abs(y_train - y_train.mean())) + 1e-8

    # Naive baseline: y_t = y_{t-1}
    naive_pred = test_df["y"].shift(1).fillna(train_df["y"].iloc[-1]).values
    naive_mae  = mean_absolute_error(y_test, naive_pred)
    naive_rmse = np.sqrt(mean_squared_error(y_test, naive_pred))

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mase = mae / naive_mae_train

    metrics = {
        "naive_mae": naive_mae, "naive_rmse": naive_rmse,
        "mae": mae, "rmse": rmse, "mase": mase,
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 60)
    print(f"{'Metric':<12} | {'Naive (Baseline)':<18} | {'XGBoost'}")
    print("-" * 60)
    print(f"{'MAE':<12} | {naive_mae:<18.4f} | {mae:.4f}")
    print(f"{'RMSE':<12} | {naive_rmse:<18.4f} | {rmse:.4f}")
    print(f"{'MASE':<12} | {1.0000:<18.4f} | {mase:.4f}")
    print("=" * 60)
    print(f"  MAE  improvement vs Naive: {(naive_mae  - mae)  / naive_mae  * 100:.2f}%")
    print(f"  RMSE improvement vs Naive: {(naive_rmse - rmse) / naive_rmse * 100:.2f}%")
    print("=" * 60)

    # Spike breakdown
    spike_mask = y_test > np.percentile(y_test, 75)
    if spike_mask.sum() > 0:
        spike_mae   = mean_absolute_error(y_test[spike_mask], y_pred[spike_mask])
        spike_naive = mean_absolute_error(y_test[spike_mask], naive_pred[spike_mask])
        print(f"\n  Spike hours (top 25%) — XGBoost MAE : {spike_mae:.4f}")
        print(f"  Spike hours (top 25%) — Naive MAE   : {spike_naive:.4f}")
        print(f"  Spike improvement                   : {(spike_naive - spike_mae) / spike_naive * 100:.2f}%")

    # Quantile coverage (if quantile models were trained)
    if "y_pred_p10" in test_df.columns and "y_pred_p90" in test_df.columns:
        p10 = test_df["y_pred_p10"].values
        p90 = test_df["y_pred_p90"].values
        coverage = np.mean((y_test >= p10) & (y_test <= p90)) * 100
        avg_width = np.mean(p90 - p10)
        print(f"\n  Quantile interval [p10, p90]:")
        print(f"    Coverage (target ≈ 80%) : {coverage:.1f}%")
        print(f"    Avg interval width      : {avg_width:.2f} reports/hr")

    return metrics


# ─────────────────────────────────────────────────────────────
# 6. WALK-FORWARD CV
# ─────────────────────────────────────────────────────────────
def walk_forward_cv(df: pd.DataFrame, n_splits: int = N_CV_FOLDS) -> None:
    print(f"\nWalk-Forward CV ({n_splits} folds)...")
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=0)
    features = [f for f in FEATURES if f in df.columns]

    # Global naive MAE as MASE denominator fallback (avoids MASE=0 on tiny folds)
    global_naive_mae = np.mean(np.abs(np.diff(df["y"].values.astype(np.float64))))
    if global_naive_mae < 1e-8:
        global_naive_mae = df["y"].mean() + 1e-8

    mae_scores, mase_scores = [], []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(df), 1):
        train_fold = df.iloc[train_idx]
        test_fold  = df.iloc[test_idx]

        y_train_log = np.log1p(train_fold["y"].clip(lower=0))

        m = build_model(early_stopping=False)
        m.set_params(n_estimators=300)  # fixed iters in CV (no eval set)
        m.fit(train_fold[features], y_train_log, verbose=False)

        preds  = np.expm1(m.predict(test_fold[features])).clip(min=0)
        y_test = test_fold["y"].values.astype(np.float64)

        fold_naive = np.mean(np.abs(np.diff(train_fold["y"].values.astype(np.float64))))
        denom = fold_naive if fold_naive > 1e-8 else global_naive_mae

        mae_fold  = mean_absolute_error(y_test, preds)
        mase_fold = mae_fold / denom

        mae_scores.append(mae_fold)
        mase_scores.append(mase_fold)
        print(f"   Fold {fold} ({len(train_fold):>5} train / {len(test_fold):>4} test): "
              f"MAE={mae_fold:.4f}  MASE={mase_fold:.4f}")

    print(f"\n   CV Summary — MAE : {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
    print(f"   CV Summary — MASE: {np.mean(mase_scores):.4f} ± {np.std(mase_scores):.4f}")


# ─────────────────────────────────────────────────────────────
# 7. FUTURE INFERENCE (autoregressive)
# ─────────────────────────────────────────────────────────────
def predict_future(df_history: pd.DataFrame, periods: int = 48) -> pd.DataFrame:
    """
    Forecast `periods` hours ahead (autoregressive).
    Returns point forecast (p50) + uncertainty bands (p10, p90).
    Predictions are guaranteed ≥ 0.
    """
    print(f"\nForecasting {periods} periods ahead (autoregressive)...")

    def _load_xgb(path):
        m = xgb.XGBRegressor()
        m.load_model(path)
        return m

    model_p50 = _load_xgb("xgboost_model_p50.json")
    model_p10 = _load_xgb("xgboost_model_p10.json")
    model_p90 = _load_xgb("xgboost_model_p90.json")

    with open("feature_list.json") as f:
        features = json.load(f)

    ch = holidays.CountryHoliday(COUNTRY)
    future_ds = pd.date_range(
        start=df_history["ds"].max(), periods=periods + 1, freq="h"
    )[1:]

    y_buffer = list(df_history["y"].values[-200:])
    results  = []

    for i, ds in enumerate(future_ds):
        row = {}

        # Calendar
        row["hour"]         = ds.hour
        row["weekday"]      = ds.weekday()
        row["day_of_month"] = ds.day
        row["month"]        = ds.month
        row["is_month_end"] = int(ds.is_month_end)
        row["is_weekend"]   = int(ds.weekday() >= 5)
        row["is_holiday"]   = int(ds in ch)
        row["is_peak_hour"] = int(8 <= ds.hour <= 18)
        row["weekend_x_peak"]            = row["is_weekend"] * row["is_peak_hour"]
        row["holiday_x_peak"]            = row["is_holiday"] * row["is_peak_hour"]
        row["weekend_or_holiday_x_peak"] = int(
            (row["is_weekend"] or row["is_holiday"]) and bool(row["is_peak_hour"])
        )

        # Cyclical
        row["hour_sin"]    = np.sin(2 * np.pi * ds.hour      / 24)
        row["hour_cos"]    = np.cos(2 * np.pi * ds.hour      / 24)
        row["weekday_sin"] = np.sin(2 * np.pi * ds.weekday() / 7)
        row["weekday_cos"] = np.cos(2 * np.pi * ds.weekday() / 7)
        row["month_sin"]   = np.sin(2 * np.pi * ds.month     / 12)
        row["month_cos"]   = np.cos(2 * np.pi * ds.month     / 12)

        # Fourier
        t = len(df_history) + i
        for name, period in {"daily": 24, "weekly": 24 * 7}.items():
            for k in range(1, 4):
                row[f"fourier_{name}_sin_{k}"] = np.sin(2 * np.pi * k * t / period)
                row[f"fourier_{name}_cos_{k}"] = np.cos(2 * np.pi * k * t / period)

        # Lags from buffer
        def lag(n):
            return y_buffer[-n] if len(y_buffer) >= n else 0

        row["y_lag_1"]   = lag(1)
        row["y_lag_2"]   = lag(2)
        row["y_lag_3"]   = lag(3)
        row["y_lag_24"]  = lag(24)
        row["y_lag_168"] = lag(168)

        buf = np.array(y_buffer, dtype=np.float64)
        row["y_rolling_mean_3"]   = buf[-3:].mean()   if len(buf) >= 3   else buf.mean()
        row["y_rolling_mean_24"]  = buf[-24:].mean()  if len(buf) >= 24  else buf.mean()
        row["y_rolling_mean_168"] = buf[-168:].mean() if len(buf) >= 168 else buf.mean()
        row["y_rolling_std_3"]    = buf[-3:].std()    if len(buf) >= 3   else 0.0
        row["y_rolling_std_24"]   = buf[-24:].std()   if len(buf) >= 24  else 0.0

        for col in ["y_lag_1", "y_lag_24", "y_lag_168",
                    "y_rolling_mean_3", "y_rolling_mean_24", "y_rolling_mean_168"]:
            row[f"log_{col}"] = np.log1p(max(row[col], 0))

        # Operational (carry forward last known)
        row["backlog"]            = df_history["backlog"].iloc[-1]
        row["tipo_reporte_id"]    = df_history["tipo_reporte_id"].iloc[-1]
        row["avg_param_span_days"] = df_history["avg_param_span_days"].iloc[-1] \
                                     if "avg_param_span_days" in df_history.columns else 0.0

        X = pd.DataFrame([row]).reindex(columns=features, fill_value=0)

        y_hat     = max(float(np.expm1(model_p50.predict(X)[0])), 0)
        y_hat_p10 = max(float(np.expm1(model_p10.predict(X)[0])), 0)
        y_hat_p90 = max(float(np.expm1(model_p90.predict(X)[0])), 0)

        # Enforce logical ordering in case quantile crossing occurs
        y_hat_p10 = min(y_hat_p10, y_hat)
        y_hat_p90 = max(y_hat_p90, y_hat)

        results.append({
            "ds":      ds,
            "y_p10":   round(y_hat_p10, 2),
            "y_pred":  round(y_hat,     2),
            "y_p90":   round(y_hat_p90, 2),
        })
        y_buffer.append(y_hat)  # feed p50 back into lag buffer

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Refresh data
    load_latest_data_from_db(DATA_PATH)

    # 2. Feature engineering
    df_final = prepare_data(
        DATA_PATH, start_date="2025-01-01", end_date="2026-02-10"
    )

    # 3. Split
    split_idx = int(len(df_final) * TRAIN_RATIO)
    print(f"\nTrain: {split_idx:,} rows | Test: {len(df_final) - split_idx:,} rows")

    # 4. Train (returns tuple of 3 models: p50, p10, p90)
    models, df_trained, features_used = train_model(df_final, split_idx)

    # 5. Evaluate
    evaluate_and_report(df_trained, split_idx)

    # 6. Walk-forward CV
    walk_forward_cv(df_final, n_splits=N_CV_FOLDS)

    # 7. Forecast next 48 hours with uncertainty bands
    future = predict_future(df_trained, periods=48)
    future.to_csv("forecast_48h.csv", index=False)
    print("\n48-hour forecast saved to forecast_48h.csv")
    print(f"\n{'Datetime':<22} {'p10':>8} {'p50 (pred)':>12} {'p90':>8}")
    print("-" * 55)
    for _, r in future.iterrows():
        print(f"{str(r['ds']):<22} {r['y_p10']:>8.1f} {r['y_pred']:>12.1f} {r['y_p90']:>8.1f}")