"""
Forecasting Model v7 — Spike Fix + Wider Quantiles + Hourly Floor + Two-Stage Spike Detector
==============================================================================================
Changes vs v6  (all driven by forecast_13d_vs_actuals.csv diagnostic):

  1. sample_weight: y² / mean(y²) instead of y/mean(y)
     — A 150-rph hour now gets ~100× weight vs a 15-rph hour (v6: 10×)
     — Directly targets the −45 rph bias on hours with actual > 100

  2. Quantile alpha widened: p10=0.05→0.025, p90=0.95→0.975
     — v6 coverage was 44.8% vs 80% target; widening the training alphas
       is the most direct lever before post-hoc band scaling kicks in.
     — Combined with horizon-aware sqrt(day) widening already in predict_future

  3. Hourly P25 floor (HOURLY_FLOOR_FILE) applied in predict_future
     — Hours 9–15 had a near-constant −20 to −41 rph bias (from diagnostic).
     — floor = P25 of training actuals per hour, applied AFTER bias correction.
     — Computed in train_model / step_hourly_floor (already present in v6 but
       NOT applied in predict_future — that's the bug fixed here).

  4. Two-stage spike detector (new SpikeClassifier)
     — Binary XGBoost classifier: P(hour is "high-volume") where high = y > P75
       of training set. Threshold at 0.50.
     — When classifier fires AND model under-predicts relative to hourly median,
       prediction is blended toward the high-volume conditional mean for that hour.
     — Trained in train_model, saved as xgboost_spike_classifier.json
     — Applied in predict_future after base prediction

  5. MODEL_VERSION bumped to 7 → forces auto-promotion on first run
     (architecture change detected via model_metadata.json)

  6. evaluate_and_report adds "eval_high_vol_bias" metric (actual > 50)
     so the improvement is explicitly tracked in MLflow across versions.
"""

import re
import json
import warnings
import os
import sys
import shutil
import pyodbc
import numpy as np
import pandas as pd
import polars as pl
import holidays
import xgboost as xgb
import mlflow
import mlflow.xgboost
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
_DATE_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2})?|\d{2}/\d{2}/\d{4})"
)


def _parse_span_days(param_str) -> float:
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


def _linear_trend(values: np.ndarray):
    """OLS on last N values → (slope, last_value)."""
    n = len(values)
    if n < 2:
        return 0.0, float(values[-1]) if n == 1 else 0.0
    x = np.arange(n, dtype=np.float64)
    slope, _ = np.polyfit(x, values.astype(np.float64), 1)
    return slope, float(values[-1])


# ─────────────────────────────────────────────────────────────
# 0. CONFIG
# ─────────────────────────────────────────────────────────────
COUNTRY     = "AR"
DATA_PATH   = "DatosQuery.csv"
TRAIN_RATIO = 0.80
N_CV_FOLDS  = 5
USE_LGBM    = False

MLFLOW_EXPERIMENT           = "XGBoost_Hourly_Forecast"
CHAMPION_METRICS_FILE       = "metrics.json"
HOURLY_CORRECTION_FILE      = "hourly_correction.json"
HOURLY_FLOOR_FILE           = "hourly_floor.json"
SPIKE_CLASSIFIER_FILE       = "xgboost_spike_classifier.json"
SPIKE_METADATA_FILE         = "spike_metadata.json"
MODEL_METADATA_FILE         = "model_metadata.json"

# FIX 1 — architecture bump forces auto-promotion on first v7 run
CURRENT_TRANSFORM = "raw"
MODEL_VERSION     = 7


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
        print(f"   Records saved to {output_csv}")
        return True
    except Exception as e:
        print(f"   DB refresh failed: {e} — falling back to existing CSV...")
        return False


# ─────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
def add_fourier_features(df, t, periods, k=3):
    for name, period in periods.items():
        for i in range(1, k + 1):
            df[f"fourier_{name}_sin_{i}"] = np.sin(2 * np.pi * i * t / period)
            df[f"fourier_{name}_cos_{i}"] = np.cos(2 * np.pi * i * t / period)
    return df


def engineer_features(pdf: pd.DataFrame, raw_df_pl: pl.DataFrame) -> pd.DataFrame:
    ch = holidays.CountryHoliday(COUNTRY)

    pdf["hour"]         = pdf["ds"].dt.hour
    pdf["weekday"]      = pdf["ds"].dt.weekday
    pdf["day_of_month"] = pdf["ds"].dt.day
    pdf["month"]        = pdf["ds"].dt.month
    pdf["is_month_end"] = pdf["ds"].dt.is_month_end.astype(int)
    pdf["is_weekend"]   = (pdf["weekday"] >= 5).astype(int)
    pdf["is_holiday"]   = pdf["ds"].apply(lambda x: int(x in ch))
    pdf["is_peak_hour"] = ((pdf["hour"] >= 8) & (pdf["hour"] <= 18)).astype(int)

    pdf["weekend_x_peak"]            = pdf["is_weekend"] * pdf["is_peak_hour"]
    pdf["holiday_x_peak"]            = pdf["is_holiday"] * pdf["is_peak_hour"]
    pdf["weekend_or_holiday_x_peak"] = (
        ((pdf["is_weekend"] == 1) | (pdf["is_holiday"] == 1)) & (pdf["is_peak_hour"] == 1)
    ).astype(int)

    pdf["hour_sin"]    = np.sin(2 * np.pi * pdf["hour"]    / 24)
    pdf["hour_cos"]    = np.cos(2 * np.pi * pdf["hour"]    / 24)
    pdf["weekday_sin"] = np.sin(2 * np.pi * pdf["weekday"] / 7)
    pdf["weekday_cos"] = np.cos(2 * np.pi * pdf["weekday"] / 7)
    pdf["month_sin"]   = np.sin(2 * np.pi * pdf["month"]   / 12)
    pdf["month_cos"]   = np.cos(2 * np.pi * pdf["month"]   / 12)

    t = np.arange(len(pdf))
    pdf = add_fourier_features(pdf, t, {"daily": 24, "weekly": 24 * 7}, k=3)

    pdf["y_lag_1"]   = pdf["y"].shift(1)
    pdf["y_lag_2"]   = pdf["y"].shift(2)
    pdf["y_lag_3"]   = pdf["y"].shift(3)
    pdf["y_lag_24"]  = pdf["y"].shift(24)
    pdf["y_lag_168"] = pdf["y"].shift(168)

    shifted = pdf["y"].shift(1)
    pdf["y_rolling_mean_3"]   = shifted.rolling(3).mean()
    pdf["y_rolling_mean_24"]  = shifted.rolling(24).mean()
    pdf["y_rolling_mean_168"] = shifted.rolling(168).mean()
    pdf["y_rolling_std_3"]    = shifted.rolling(3).std()
    pdf["y_rolling_std_24"]   = shifted.rolling(24).std()

    # Log-scale lags as INPUT features only (target is raw)
    for col in ["y_lag_1", "y_lag_24", "y_lag_168",
                "y_rolling_mean_3", "y_rolling_mean_24", "y_rolling_mean_168"]:
        pdf[f"log_{col}"] = np.log1p(pdf[col].clip(lower=0))

    print("   Calculating Backlog...")

    def backlog_at(target_ds):
        return raw_df_pl.filter(
            (pl.col("CreationDate") <= target_ds)
            & (pl.col("StartedProcessingOn") > target_ds)
        ).height

    pdf["backlog"] = [backlog_at(ts) for ts in pdf["ds"]]
    pdf["tipo_reporte_id"] = pd.factorize(pdf["main_report_type"])[0]
    return pdf


def prepare_data(csv_path, start_date=None, end_date=None) -> pd.DataFrame:
    print("Loading and filtering data...")
    df = pl.read_csv(csv_path, ignore_errors=True)
    df = df.with_columns([
        pl.col("CreationDate").str.strptime(
            pl.Datetime, format="%Y-%m-%d %H:%M:%S%.f", strict=False),
        pl.col("StartedProcessingOn").str.strptime(
            pl.Datetime, format="%Y-%m-%d %H:%M:%S%.f", strict=False),
    ]).filter(pl.col("StartedProcessingOn").is_not_null())

    if start_date:
        df = df.filter(pl.col("StartedProcessingOn") >= datetime.strptime(start_date, "%Y-%m-%d"))
    if end_date:
        df = df.filter(pl.col("StartedProcessingOn") <= datetime.strptime(end_date, "%Y-%m-%d"))

    hourly_ts = (
        df.with_columns(pl.col("StartedProcessingOn").dt.truncate("1h").alias("ds"))
        .group_by("ds")
        .agg([pl.len().alias("y"),
              pl.col("ReportType").mode().first().alias("main_report_type")])
        .sort("ds")
    )
    pdf = hourly_ts.to_pandas()

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
    "backlog", "tipo_reporte_id",
    "hour", "weekday", "day_of_month", "month",
    "is_month_end", "is_weekend", "is_holiday", "is_peak_hour",
    "weekend_x_peak", "holiday_x_peak", "weekend_or_holiday_x_peak",
    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "month_sin", "month_cos",
    "fourier_daily_sin_1",  "fourier_daily_cos_1",
    "fourier_daily_sin_2",  "fourier_daily_cos_2",
    "fourier_daily_sin_3",  "fourier_daily_cos_3",
    "fourier_weekly_sin_1", "fourier_weekly_cos_1",
    "fourier_weekly_sin_2", "fourier_weekly_cos_2",
    "fourier_weekly_sin_3", "fourier_weekly_cos_3",
    "y_lag_1", "y_lag_2", "y_lag_3", "y_lag_24", "y_lag_168",
    "y_rolling_mean_3",  "y_rolling_mean_24",  "y_rolling_mean_168",
    "y_rolling_std_3",   "y_rolling_std_24",
    "log_y_lag_1", "log_y_lag_24", "log_y_lag_168",
    "log_y_rolling_mean_3", "log_y_rolling_mean_24", "log_y_rolling_mean_168",
    "avg_param_span_days",
]


# ─────────────────────────────────────────────────────────────
# 4. TRAINING
# ─────────────────────────────────────────────────────────────
def train_model(df: pd.DataFrame, split_idx: int, hyperparams: dict = None):
    """
    Train p50 / p10 / p90 regression models + spike binary classifier.
    Returns (models_tuple, df_with_preds, features_list).
    """
    hp = {
        "n_estimators":     1000,
        "learning_rate":    0.02,
        "max_depth":        7,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "gamma":            1,
    }
    if hyperparams:
        hp.update(hyperparams)

    features = [f for f in FEATURES if f in df.columns]
    train_df = df.iloc[:split_idx].copy()
    test_df  = df.iloc[split_idx:].copy()

    y_train = train_df["y"].clip(lower=0).values.astype(np.float64)
    y_test  = test_df["y"].clip(lower=0).values.astype(np.float64)

    # ── FIX 1: y²-proportional sample weights ───────────────
    # Diagnostic showed hours > 100 rph had −45 rph bias.
    # y²/mean(y²) gives 100× weight to a 150-rph hour vs a 15-rph hour,
    # vs log1p(y) which gave only ~1.7×, and y/mean(y) which gave 10×.
    # Clipped at 1.0 so quiet hours are never completely ignored.
    _mean_sq = float(np.mean(y_train ** 2)) if np.mean(y_train ** 2) > 0 else 1.0
    sample_weight_p50 = np.maximum(y_train ** 2 / _mean_sq, 1.0)
    print(f"      [TRACE] sample_weight (y²) range: "
          f"[{sample_weight_p50.min():.2f}, {sample_weight_p50.max():.2f}]  "
          f"(peak/quiet ratio = {sample_weight_p50.max():.0f}×)")

    print(f"\n   [TRACE] Training p50/p10/p90 on {len(train_df):,} rows "
          f"(raw target, {len(features)} features)...")

    def _fit_regressor(objective, label, quantile_alpha=None, sample_weight=None):
        params = {**hp,
                  "objective":             objective,
                  "eval_metric":           "rmse",
                  "tree_method":           "hist",
                  "early_stopping_rounds": 50}
        if quantile_alpha is not None:
            params["quantile_alpha"] = quantile_alpha

        m = xgb.XGBRegressor(**params)
        fit_kw = dict(eval_set=[(test_df[features], y_test)], verbose=False)
        if sample_weight is not None:
            fit_kw["sample_weight"] = sample_weight
        m.fit(train_df[features], y_train, **fit_kw)
        n_trees = m.best_iteration if hasattr(m, "best_iteration") else hp["n_estimators"]
        print(f"      [TRACE] {label}: {n_trees} trees, best_score={m.best_score:.4f}")
        return m

    with mlflow.start_run(run_name="step_train_p50", nested=True):
        mlflow.log_params({k: v for k, v in hp.items()})
        mlflow.log_param("objective", "reg:squarederror")
        mlflow.log_param("sample_weight", "y_squared")
        model_p50 = _fit_regressor("reg:squarederror", "p50 (point forecast)",
                                   sample_weight=sample_weight_p50)
        mlflow.log_metric("p50_best_score", float(model_p50.best_score))

    # ── FIX 2: Wider quantile alphas (0.025 / 0.975) ────────
    # v6 used 0.05/0.95 → coverage was 44.8% vs 80% target.
    # Widening training alphas directly increases the learned interval width.
    with mlflow.start_run(run_name="step_train_p10", nested=True):
        mlflow.log_param("quantile_alpha", 0.025)
        model_p10 = _fit_regressor("reg:quantileerror", "p10 (lower bound)",
                                   quantile_alpha=0.025)
        mlflow.log_metric("p10_best_score", float(model_p10.best_score))

    with mlflow.start_run(run_name="step_train_p90", nested=True):
        mlflow.log_param("quantile_alpha", 0.975)
        model_p90 = _fit_regressor("reg:quantileerror", "p90 (upper bound)",
                                   quantile_alpha=0.975)
        mlflow.log_metric("p90_best_score", float(model_p90.best_score))

    # Predictions on full df — raw scale, no expm1
    df = df.copy()
    df["y_pred"]     = model_p50.predict(df[features]).clip(min=0)
    df["y_pred_p10"] = model_p10.predict(df[features]).clip(min=0)
    df["y_pred_p90"] = model_p90.predict(df[features]).clip(min=0)

    # ── FIX 4: Two-stage spike classifier ───────────────────
    # Binary XGBoost: label = 1 if y > P75(training actuals).
    # Saves the classifier + per-hour conditional means for the blend step.
    with mlflow.start_run(run_name="step_spike_classifier", nested=True):
        spike_threshold = float(np.percentile(y_train, 75))
        y_spike_train = (y_train > spike_threshold).astype(int)
        y_spike_test  = (y_test  > spike_threshold).astype(int)

        spike_clf = xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            gamma=0.5,
            objective="binary:logistic",
            eval_metric="logloss",
            early_stopping_rounds=30,
            tree_method="hist",
        )
        spike_clf.fit(
            train_df[features], y_spike_train,
            eval_set=[(test_df[features], y_spike_test)],
            verbose=False,
        )
        spike_clf.save_model(SPIKE_CLASSIFIER_FILE)

        # Conditional mean per hour for spike hours in TRAINING set
        train_df_tmp = train_df.copy()
        train_df_tmp["_hour"] = pd.to_datetime(train_df_tmp["ds"]).dt.hour
        spike_hour_mean = (
            train_df_tmp[train_df_tmp["y"] > spike_threshold]
            .groupby("_hour")["y"]
            .mean()
            .to_dict()
        )
        # Overall spike mean as fallback
        overall_spike_mean = float(train_df_tmp[train_df_tmp["y"] > spike_threshold]["y"].mean())
        spike_metadata = {
            "threshold":         round(spike_threshold, 2),
            "overall_spike_mean": round(overall_spike_mean, 2),
            "hourly_spike_mean": {int(h): round(float(v), 2)
                                  for h, v in spike_hour_mean.items()},
        }
        with open(SPIKE_METADATA_FILE, "w") as f:
            json.dump(spike_metadata, f, indent=2)

        n_spike_train = int(y_spike_train.sum())
        n_spike_test  = int(y_spike_test.sum())
        print(f"      [TRACE] Spike classifier trained  "
              f"(threshold={spike_threshold:.1f}, "
              f"train spikes={n_spike_train}, test spikes={n_spike_test})")
        print(f"      [TRACE] Spike hourly means (9–15): "
              + "  ".join(f"h{h}={spike_hour_mean.get(h, overall_spike_mean):.0f}"
                          for h in range(9, 16)))
        mlflow.log_params({
            "spike_threshold":    spike_threshold,
            "n_spike_train":      n_spike_train,
            "spike_blend_weight": 0.35,
        })
        mlflow.log_metric("spike_classifier_best_score", float(spike_clf.best_score))
        mlflow.log_artifact(SPIKE_CLASSIFIER_FILE)
        mlflow.log_artifact(SPIKE_METADATA_FILE)

    # ── Additive hourly bias correction ─────────────────────
    with mlflow.start_run(run_name="step_bias_calibration", nested=True):
        test_eval = df.iloc[split_idx:].copy()
        test_eval["hour_col"] = pd.to_datetime(test_eval["ds"]).dt.hour

        correction = {}
        if len(test_eval) > 0:
            grp = test_eval.groupby("hour_col").apply(
                lambda g: float(np.median(g["y"].values - g["y_pred"].values))
            )
            correction = {int(h): round(float(v), 4) for h, v in grp.items()}
        if not correction:
            correction = {h: 0.0 for h in range(24)}

        with open(HOURLY_CORRECTION_FILE, "w") as f:
            json.dump(correction, f, indent=2)
        avg_corr = np.mean(list(correction.values()))
        print(f"      [TRACE] Additive bias correction saved  "
              f"(avg {avg_corr:+.2f} rph)  "
              f"h10={correction.get(10, 0):+.1f}  h12={correction.get(12, 0):+.1f}")
        mlflow.log_metric("avg_bias_additive_correction", round(avg_corr, 4))
        mlflow.log_artifact(HOURLY_CORRECTION_FILE)

    # ── FIX 3: Hourly P25 floor ──────────────────────────────
    # Computed on TRAINING actuals only. In predict_future the floor is applied
    # AFTER bias correction as a hard minimum — prevents the model from
    # predicting below what has historically been observed at that hour of day.
    # NOTE (v6 bug fixed): the floor file was computed in v6 but never loaded
    # or applied inside predict_future. Now it is explicitly applied.
    with mlflow.start_run(run_name="step_hourly_floor", nested=True):
        train_floor_df = train_df.copy()
        train_floor_df["hour_col"] = pd.to_datetime(train_floor_df["ds"]).dt.hour
        floor_map = {}
        for hr, grp in train_floor_df.groupby("hour_col")["y"]:
            vals = grp.values
            p75  = float(np.percentile(vals, 75))
            # Only set a non-zero floor for hours that have genuine activity
            floor_map[int(hr)] = round(float(np.percentile(vals, 25)), 2) if p75 > 0 else 0.0

        with open(HOURLY_FLOOR_FILE, "w") as f:
            json.dump(floor_map, f, indent=2)
        peak_floors = {h: v for h, v in floor_map.items() if 9 <= h <= 15}
        print(f"      [TRACE] Hourly P25 floor saved  "
              + "  ".join(f"h{h}={v:.0f}" for h, v in sorted(peak_floors.items())))
        mlflow.log_metrics({f"floor_h{h}": v for h, v in floor_map.items()})
        mlflow.log_artifact(HOURLY_FLOOR_FILE)

    # Stage challenger models
    model_p50.save_model("xgboost_model_p50_challenger.json")
    model_p10.save_model("xgboost_model_p10_challenger.json")
    model_p90.save_model("xgboost_model_p90_challenger.json")

    with open("feature_list.json", "w") as f:
        json.dump(features, f)
    label_map = dict(enumerate(pd.factorize(df["main_report_type"])[1]))
    with open("label_mapping.json", "w") as f:
        json.dump(label_map, f)

    print("   [TRACE] Challenger models staged (p10/p50/p90 + spike classifier).")
    return (model_p50, model_p10, model_p90), df, features


# ─────────────────────────────────────────────────────────────
# 5. EVALUATION
# ─────────────────────────────────────────────────────────────
def evaluate_and_report(df: pd.DataFrame, split_idx: int) -> dict:
    with mlflow.start_run(run_name="step_evaluation", nested=True):
        train_df = df.iloc[:split_idx]
        test_df  = df.iloc[split_idx:]

        y_train = train_df["y"].values.astype(np.float64)
        y_test  = test_df["y"].values.astype(np.float64)
        y_pred  = test_df["y_pred"].values.astype(np.float64)

        naive_mae_train = np.mean(np.abs(np.diff(y_train)))
        if naive_mae_train < 1e-8:
            naive_mae_train = np.mean(np.abs(y_train - y_train.mean())) + 1e-8

        naive_pred = test_df["y"].shift(1).fillna(train_df["y"].iloc[-1]).values
        naive_mae  = mean_absolute_error(y_test, naive_pred)
        naive_rmse = np.sqrt(mean_squared_error(y_test, naive_pred))
        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mase = mae / naive_mae_train

        # ── FIX 5: Track high-volume bias explicitly ─────────
        high_vol_mask = y_test > 50
        high_vol_bias = float(np.mean(y_pred[high_vol_mask] - y_test[high_vol_mask])) \
            if high_vol_mask.sum() > 0 else 0.0

        metrics = {
            "naive_mae": naive_mae, "naive_rmse": naive_rmse,
            "mae": mae, "rmse": rmse, "mase": mase,
            "high_vol_bias": high_vol_bias,
            "target_transform": CURRENT_TRANSFORM,
            "model_version":    MODEL_VERSION,
        }

        mlflow.log_metrics({
            "mae": mae, "rmse": rmse, "mase": mase,
            "naive_mae": naive_mae, "naive_rmse": naive_rmse,
            "high_vol_bias": high_vol_bias,
        })

        print("\n" + "=" * 60)
        print(f"{'Metric':<18} | {'Naive (Baseline)':<18} | {'XGBoost'}")
        print("-" * 60)
        print(f"{'MAE':<18} | {naive_mae:<18.4f} | {mae:.4f}")
        print(f"{'RMSE':<18} | {naive_rmse:<18.4f} | {rmse:.4f}")
        print(f"{'MASE':<18} | {1.0000:<18.4f} | {mase:.4f}")
        print(f"{'High-vol bias':<18} | {'(>50 rph hours)':<18} | {high_vol_bias:+.4f}")
        print("=" * 60)
        print(f"  MAE  improvement vs Naive: {(naive_mae  - mae)  / naive_mae  * 100:.2f}%")
        print(f"  RMSE improvement vs Naive: {(naive_rmse - rmse) / naive_rmse * 100:.2f}%")
        print("=" * 60)

        spike_mask = y_test > np.percentile(y_test, 75)
        if spike_mask.sum() > 0:
            spike_mae   = mean_absolute_error(y_test[spike_mask], y_pred[spike_mask])
            spike_naive = mean_absolute_error(y_test[spike_mask], naive_pred[spike_mask])
            mlflow.log_metrics({"spike_mae": spike_mae, "spike_mae_naive": spike_naive})
            print(f"\n  Spike hours (top 25%) — XGBoost MAE : {spike_mae:.4f}")
            print(f"  Spike hours (top 25%) — Naive MAE   : {spike_naive:.4f}")
            print(f"  Spike improvement                   : "
                  f"{(spike_naive - spike_mae) / spike_naive * 100:.2f}%")

        if "y_pred_p10" in test_df.columns and "y_pred_p90" in test_df.columns:
            p10 = test_df["y_pred_p10"].values
            p90 = test_df["y_pred_p90"].values
            coverage  = np.mean((y_test >= p10) & (y_test <= p90)) * 100
            avg_width = np.mean(p90 - p10)
            mlflow.log_metrics({"quantile_coverage_pct": coverage, "quantile_avg_width": avg_width})
            print(f"\n  Quantile interval [p10, p90]:")
            print(f"    Coverage (target ≈ 80%) : {coverage:.1f}%")
            print(f"    Avg interval width      : {avg_width:.2f} reports/hr")

    return metrics


# ─────────────────────────────────────────────────────────────
# 6. CHAMPION / CHALLENGER PROMOTION
# ─────────────────────────────────────────────────────────────
def _architecture_changed() -> bool:
    if not os.path.exists(MODEL_METADATA_FILE):
        return True
    try:
        with open(MODEL_METADATA_FILE) as f:
            meta = json.load(f)
        return (meta.get("target_transform") != CURRENT_TRANSFORM
                or meta.get("model_version", 0) != MODEL_VERSION)
    except Exception:
        return True


def promote_if_better(new_metrics: dict, force_promote: bool = False) -> bool:
    arch_changed  = _architecture_changed()
    champion_mase = float("inf")

    if os.path.exists(CHAMPION_METRICS_FILE) and not arch_changed:
        try:
            with open(CHAMPION_METRICS_FILE) as f:
                champ = json.load(f)
            champion_mase = champ.get("mase", float("inf"))
        except Exception:
            champion_mase = float("inf")

    new_mase = new_metrics.get("mase", float("inf"))
    promoted = force_promote or arch_changed or (new_mase <= champion_mase)

    if promoted:
        reason = ("forced reset flag"
                  if force_promote else
                  ("architecture/version change"
                   if arch_changed else
                   f"MASE {new_mase:.4f} < {champion_mase:.4f}"))
        print(f"\n   [TRACE] Promoting challenger → champion ({reason})")
        for name in ["p50", "p10", "p90"]:
            src = f"xgboost_model_{name}_challenger.json"
            dst = f"xgboost_model_{name}.json"
            if os.path.exists(src):
                shutil.copy2(src, dst)
        if os.path.exists("xgboost_model_p50.json"):
            shutil.copy2("xgboost_model_p50.json", "xgboost_model.json")

        with open(CHAMPION_METRICS_FILE, "w") as f:
            json.dump(new_metrics, f, indent=2)
        meta = {
            "target_transform": CURRENT_TRANSFORM,
            "model_version":    MODEL_VERSION,
            "promoted_at":      datetime.now().isoformat(),
            "mase":             new_mase,
        }
        with open(MODEL_METADATA_FILE, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"\n🏆 New champion promoted! MASE={new_mase:.4f}  version=v{MODEL_VERSION}")
    else:
        print(f"\n⚔️  Challenger did not beat champion "
              f"(MASE {new_mase:.4f} > {champion_mase:.4f})")

    for name in ["p50", "p10", "p90"]:
        tmp = f"xgboost_model_{name}_challenger.json"
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass

    return promoted


# ─────────────────────────────────────────────────────────────
# 7. WALK-FORWARD CV
# ─────────────────────────────────────────────────────────────
def walk_forward_cv(df: pd.DataFrame, n_splits: int = N_CV_FOLDS) -> dict:
    print(f"\nWalk-Forward CV ({n_splits} folds)...")
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=0)
    features = [f for f in FEATURES if f in df.columns]

    global_naive_mae = np.mean(np.abs(np.diff(df["y"].values.astype(np.float64))))
    if global_naive_mae < 1e-8:
        global_naive_mae = df["y"].mean() + 1e-8

    mae_scores, mase_scores = [], []

    with mlflow.start_run(run_name="step_walk_forward_cv", nested=True):
        for fold, (train_idx, test_idx) in enumerate(tscv.split(df), 1):
            train_fold = df.iloc[train_idx]
            y_train_raw = train_fold["y"].clip(lower=0).values.astype(np.float64)
            y_test_raw  = df.iloc[test_idx]["y"].values.astype(np.float64)

            # Use y² weights in CV too (consistent with full training)
            _msq = float(np.mean(y_train_raw ** 2)) if np.mean(y_train_raw ** 2) > 0 else 1.0
            sw   = np.maximum(y_train_raw ** 2 / _msq, 1.0)

            m = xgb.XGBRegressor(
                n_estimators=300, learning_rate=0.02, max_depth=7,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
                gamma=1, objective="reg:squarederror", eval_metric="rmse",
                tree_method="hist",
            )
            m.fit(train_fold[features], y_train_raw, sample_weight=sw, verbose=False)
            preds = m.predict(df.iloc[test_idx][features]).clip(min=0)

            fold_naive = np.mean(np.abs(np.diff(y_train_raw)))
            denom      = fold_naive if fold_naive > 1e-8 else global_naive_mae
            mae_fold   = mean_absolute_error(y_test_raw, preds)
            mase_fold  = mae_fold / denom
            mae_scores.append(mae_fold)
            mase_scores.append(mase_fold)
            mlflow.log_metrics({f"cv_fold{fold}_mae":  mae_fold,
                                 f"cv_fold{fold}_mase": mase_fold})
            print(f"   Fold {fold} ({len(train_fold):>5} train / "
                  f"{len(df.iloc[test_idx]):>4} test): "
                  f"MAE={mae_fold:.4f}  MASE={mase_fold:.4f}")

        cv_mae_mean  = float(np.mean(mae_scores))
        cv_mase_mean = float(np.mean(mase_scores))
        mlflow.log_metrics({"cv_mae_mean":  cv_mae_mean,
                             "cv_mase_mean": cv_mase_mean,
                             "cv_mae_std":   float(np.std(mae_scores)),
                             "cv_mase_std":  float(np.std(mase_scores))})
        print(f"\n   CV Summary — MAE : {cv_mae_mean:.4f} ± {np.std(mae_scores):.4f}")
        print(f"   CV Summary — MASE: {cv_mase_mean:.4f} ± {np.std(mase_scores):.4f}")

    return {"cv_mae_mean": cv_mae_mean, "cv_mase_mean": cv_mase_mean}


# ─────────────────────────────────────────────────────────────
# 8. FUTURE INFERENCE — ALWAYS 13-DAY PRIMARY FORECAST
# ─────────────────────────────────────────────────────────────
def _load_model_metadata() -> dict:
    if os.path.exists(MODEL_METADATA_FILE):
        try:
            with open(MODEL_METADATA_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"target_transform": CURRENT_TRANSFORM}


def predict_future(df_history: pd.DataFrame, periods: int = 13 * 24) -> pd.DataFrame:
    """
    Forecast `periods` hours ahead (default = 13 days = 312h).

    Post-prediction pipeline (in order):
      1. Raw XGBoost prediction (p50 / p10 / p90)
      2. Seasonal blend (ramps to 40% at horizon=168h to prevent lag compounding)
      3. Additive hourly bias correction
      4. FIX 3 — Hourly P25 floor (hard minimum from training P25 per hour)
      5. FIX 4 — Two-stage spike boost  (blend toward hourly spike mean when classifier fires)
      6. Horizon-aware quantile widening  (sqrt(day) scaling)
    """
    print(f"\n[TRACE] Forecasting {periods} periods ahead ({periods // 24:.1f} days)...")

    def _load_xgb(path):
        m = xgb.XGBRegressor()
        m.load_model(path)
        return m

    model_p50 = _load_xgb("xgboost_model_p50.json")
    model_p10 = _load_xgb("xgboost_model_p10.json")
    model_p90 = _load_xgb("xgboost_model_p90.json")

    with open("feature_list.json") as f:
        features = json.load(f)

    meta      = _load_model_metadata()
    use_expm1 = meta.get("target_transform", "raw") == "log1p"
    if use_expm1:
        print("   [TRACE] Champion model uses log1p transform — applying expm1")

    # ── Load additive bias correction ─────────────────────────
    hourly_correction = {h: 0.0 for h in range(24)}
    if os.path.exists(HOURLY_CORRECTION_FILE):
        with open(HOURLY_CORRECTION_FILE) as fc:
            raw_corr = json.load(fc)
        loaded = {int(k): float(v) for k, v in raw_corr.items()}
        # Detect legacy multiplicative format (all positive, median near 1.0)
        vals = list(loaded.values())
        if (np.median(vals) > 0.5) and all(v > 0 for v in vals):
            print("   [TRACE] Legacy multiplicative correction detected — ignoring (additive=0)")
        else:
            hourly_correction.update(loaded)
            print(f"   [TRACE] Additive correction loaded  (avg {np.mean(vals):+.2f} rph)")

    # ── FIX 3: Load P25 floor ─────────────────────────────────
    hourly_floor = {h: 0.0 for h in range(24)}
    if os.path.exists(HOURLY_FLOOR_FILE):
        with open(HOURLY_FLOOR_FILE) as ff:
            loaded_floor = json.load(ff)
        hourly_floor.update({int(k): float(v) for k, v in loaded_floor.items()})
        peak_f = {h: hourly_floor[h] for h in range(9, 16)}
        print(f"   [TRACE] P25 floor loaded  "
              + "  ".join(f"h{h}={v:.0f}" for h, v in sorted(peak_f.items())))
    else:
        print("   [TRACE] hourly_floor.json not found — floor disabled (re-train to enable)")

    # ── FIX 4: Load spike classifier + metadata ───────────────
    spike_clf       = None
    spike_meta      = {}
    spike_available = (os.path.exists(SPIKE_CLASSIFIER_FILE)
                       and os.path.exists(SPIKE_METADATA_FILE))
    if spike_available:
        spike_clf = xgb.XGBClassifier()
        spike_clf.load_model(SPIKE_CLASSIFIER_FILE)
        with open(SPIKE_METADATA_FILE) as sm:
            spike_meta = json.load(sm)
        print(f"   [TRACE] Spike classifier loaded  "
              f"(threshold={spike_meta.get('threshold', '?')} rph, "
              f"blend_weight=0.35)")
    else:
        print("   [TRACE] Spike classifier not found — spike boost disabled")

    ch = holidays.CountryHoliday(COUNTRY)
    future_ds = pd.date_range(
        start=df_history["ds"].max(), periods=periods + 1, freq="h"
    )[1:]

    # Trend projections for operational features
    backlog_slope, backlog_last = _linear_trend(
        df_history["backlog"].values[-24:].astype(np.float64))
    if "avg_param_span_days" in df_history.columns:
        span_slope, span_last = _linear_trend(
            df_history["avg_param_span_days"].values[-24:].astype(np.float64))
    else:
        span_slope, span_last = 0.0, 0.0

    # Seasonal baseline from last 4 weeks
    SEASONAL_WEEKS  = 4
    seasonal_window = df_history.tail(SEASONAL_WEEKS * 7 * 24).copy()
    seasonal_window["_hour"]    = pd.to_datetime(seasonal_window["ds"]).dt.hour
    seasonal_window["_weekday"] = pd.to_datetime(seasonal_window["ds"]).dt.weekday
    seasonal_table    = (seasonal_window.groupby(["_weekday", "_hour"])["y"]
                         .mean().to_dict())
    seasonal_hr_mean  = seasonal_window.groupby("_hour")["y"].mean().to_dict()

    def _seasonal(ds_stamp):
        key = (ds_stamp.weekday(), ds_stamp.hour)
        return float(seasonal_table.get(key, seasonal_hr_mean.get(ds_stamp.hour, 0.0)))

    def _blend_alpha(step_i: int) -> float:
        return min(0.40, step_i / 168 * 0.40)

    y_buffer = list(df_history["y"].values[-200:])
    results  = []

    for i, ds in enumerate(future_ds):
        row = {}
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
            (row["is_weekend"] or row["is_holiday"]) and bool(row["is_peak_hour"]))
        row["hour_sin"]    = np.sin(2 * np.pi * ds.hour      / 24)
        row["hour_cos"]    = np.cos(2 * np.pi * ds.hour      / 24)
        row["weekday_sin"] = np.sin(2 * np.pi * ds.weekday() / 7)
        row["weekday_cos"] = np.cos(2 * np.pi * ds.weekday() / 7)
        row["month_sin"]   = np.sin(2 * np.pi * ds.month     / 12)
        row["month_cos"]   = np.cos(2 * np.pi * ds.month     / 12)

        t = len(df_history) + i
        for name, period in {"daily": 24, "weekly": 24 * 7}.items():
            for k in range(1, 4):
                row[f"fourier_{name}_sin_{k}"] = np.sin(2 * np.pi * k * t / period)
                row[f"fourier_{name}_cos_{k}"] = np.cos(2 * np.pi * k * t / period)

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

        row["backlog"]             = max(0.0, backlog_last + backlog_slope * (i + 1))
        row["avg_param_span_days"] = max(0.0, span_last   + span_slope  * (i + 1))
        row["tipo_reporte_id"]     = df_history["tipo_reporte_id"].iloc[-1]

        X = pd.DataFrame([row]).reindex(columns=features, fill_value=0)

        # 1. Raw model predictions
        raw_p50 = float(model_p50.predict(X)[0])
        raw_p10 = float(model_p10.predict(X)[0])
        raw_p90 = float(model_p90.predict(X)[0])

        if use_expm1:
            y_hat_raw = max(np.expm1(raw_p50), 0)
            y_hat_p10 = max(np.expm1(raw_p10), 0)
            y_hat_p90 = max(np.expm1(raw_p90), 0)
        else:
            y_hat_raw = max(raw_p50, 0)
            y_hat_p10 = max(raw_p10, 0)
            y_hat_p90 = max(raw_p90, 0)

        # 2. Seasonal blend (prevents lag compounding at long horizons)
        alpha  = _blend_alpha(i)
        y_seas = _seasonal(ds)
        y_hat  = (1 - alpha) * y_hat_raw + alpha * y_seas

        # 3. Additive bias correction
        bias_add  = hourly_correction.get(ds.hour, 0.0)
        y_hat     = max(y_hat     + bias_add, 0.0)
        y_hat_p10 = max(y_hat_p10 + bias_add, 0.0)
        y_hat_p90 = max(y_hat_p90 + bias_add, 0.0)

        # 4. FIX 3 — Hourly P25 floor (v6 bug: this was computed but never applied)
        floor_val = hourly_floor.get(ds.hour, 0.0)
        # Solo aplicar floor en horas con actividad real
        if row["is_peak_hour"] == 1 and row["is_weekend"] == 0:
            y_hat = max(y_hat, floor_val)
            y_hat_p10 = max(y_hat_p10, floor_val)
            y_hat_p90 = max(y_hat_p90, floor_val)   

        # 5. FIX 4 — Two-stage spike boost
        # If classifier says P(spike) >= 0.50, blend prediction toward the
        # per-hour conditional mean of spike hours (weight=0.35).
        # This corrects the systematic −45 rph under-prediction on peak hours
        # without affecting quiet hours where the classifier is silent.
        if spike_clf is not None:
            spike_prob = float(spike_clf.predict_proba(X)[0][1])
            if spike_prob >= 0.50:
                hourly_spike_mean = spike_meta.get("hourly_spike_mean", {})
                target_spike_val  = float(
                    hourly_spike_mean.get(
                        str(ds.hour),                          # JSON keys are strings
                        spike_meta.get("overall_spike_mean", y_hat)
                    )
                )
                SPIKE_BLEND = 0.50   # blends 50% toward historical spike mean
                # Solo aplicar spike boost en horas con actividad real
                if row["is_peak_hour"] == 1 and row["is_weekend"] == 0:
                    y_hat = (1 - SPIKE_BLEND) * y_hat + SPIKE_BLEND * target_spike_val
                # Widen bands proportionally when spike is detected
                half = (y_hat_p90 - y_hat_p10) / 2 * 1.20
                y_hat_p10 = max(y_hat - half, 0.0)
                y_hat_p90 = y_hat + half

        # 6. Horizon-aware quantile widening (unchanged from v6)
        day_frac   = (i + 1) / 24
        width_mult = max(1.0, np.sqrt(day_frac))
        half_width = (y_hat_p90 - y_hat_p10) / 2 * width_mult
        y_hat_p10  = max(y_hat - half_width, 0.0)
        y_hat_p90  = y_hat + half_width

        results.append({"ds":     ds,
                         "y_p10":  round(y_hat_p10, 2),
                         "y_pred": round(y_hat,     2),
                         "y_p90":  round(y_hat_p90, 2)})
        y_buffer.append(y_hat)

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────
# 9. GRID SEARCH
# ─────────────────────────────────────────────────────────────
HYPERPARAMETER_GRID = [
    {"learning_rate": 0.02, "max_depth": 7,  "min_child_weight": 10, "gamma": 1,   "subsample": 0.8},
    {"learning_rate": 0.05, "max_depth": 6,  "min_child_weight": 5,  "gamma": 0.5, "subsample": 0.9},
    {"learning_rate": 0.01, "max_depth": 8,  "min_child_weight": 15, "gamma": 2,   "subsample": 0.7},
    {"learning_rate": 0.03, "max_depth": 5,  "min_child_weight": 8,  "gamma": 1,   "subsample": 0.85},
    {"learning_rate": 0.02, "max_depth": 9,  "min_child_weight": 10, "gamma": 0,   "subsample": 0.8},
]


def run_grid_search(df: pd.DataFrame, split_idx: int, force_promote: bool = False):
    print(f"\n{'='*60}")
    print(f"HYPERPARAMETER GRID SEARCH ({len(HYPERPARAMETER_GRID)} configs)")
    print(f"{'='*60}")

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    best_mase, best_run_id = float("inf"), None

    for idx, hp in enumerate(HYPERPARAMETER_GRID, 1):
        print(f"\n--- Config {idx}/{len(HYPERPARAMETER_GRID)}: {hp}")
        run_name = f"grid_{idx}_lr{hp['learning_rate']}_d{hp['max_depth']}"
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params({**hp,
                                "config_index":      idx,
                                "target_transform":  CURRENT_TRANSFORM,
                                "sample_weight":     "y_squared",
                                "p10_alpha":         0.025,
                                "p90_alpha":         0.975})
            models, df_trained, _ = train_model(df, split_idx, hyperparams=hp)
            metrics  = evaluate_and_report(df_trained, split_idx)
            promoted = promote_if_better(metrics, force_promote=force_promote)
            mlflow.log_metrics({"mae": metrics["mae"], "rmse": metrics["rmse"],
                                 "mase": metrics["mase"],
                                 "high_vol_bias": metrics["high_vol_bias"]})
            mlflow.log_param("promoted_to_champion", promoted)
            mlflow.set_tag("champion", str(promoted))
            if metrics["mase"] < best_mase:
                best_mase, best_run_id = metrics["mase"], run.info.run_id
        print(f"   Config {idx} MASE={metrics['mase']:.4f} "
              f"high_vol_bias={metrics['high_vol_bias']:+.2f}  "
              f"{'★ CHAMPION' if promoted else ''}")

    print(f"\n{'='*60}")
    print(f"Grid search complete. Best MASE: {best_mase:.4f}  Run: {best_run_id}")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────
# 10. FULL RUN WRAPPER (single config)
# ─────────────────────────────────────────────────────────────
def run_with_mlflow(df: pd.DataFrame, split_idx: int,
                    end_date: str = None,
                    hyperparams: dict = None,
                    force_promote: bool = False) -> tuple:
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    with mlflow.start_run(run_name=f"run_v{MODEL_VERSION}_{ts}") as run:
        run_id = run.info.run_id
        print(f"\n[TRACE] MLflow run started: {run_id}")

        mlflow.log_params({
            "model_version":     MODEL_VERSION,
            "n_estimators":      1000,
            "learning_rate":     (hyperparams or {}).get("learning_rate", 0.02),
            "max_depth":         (hyperparams or {}).get("max_depth", 7),
            "objective_p50":     "reg:squarederror",
            "sample_weight":     "y_squared",
            "p10_alpha":         0.025,
            "p90_alpha":         0.975,
            "spike_blend_weight": 0.35,
            "train_ratio":       TRAIN_RATIO,
            "n_cv_folds":        N_CV_FOLDS,
            "target_transform":  CURRENT_TRANSFORM,
            "end_date":          end_date or "not_set",
        })

        models, df_trained, features_used = train_model(df, split_idx, hyperparams)
        metrics    = evaluate_and_report(df_trained, split_idx)
        cv_metrics = walk_forward_cv(df, n_splits=N_CV_FOLDS)

        mlflow.log_metrics({
            "mae": metrics["mae"], "rmse": metrics["rmse"], "mase": metrics["mase"],
            "high_vol_bias":  metrics["high_vol_bias"],
            "cv_mae_mean":    cv_metrics["cv_mae_mean"],
            "cv_mase_mean":   cv_metrics["cv_mase_mean"],
        })

        promoted = promote_if_better(metrics, force_promote=force_promote)
        mlflow.log_param("promoted_to_champion", promoted)
        mlflow.set_tag("champion", str(promoted))

        with mlflow.start_run(run_name="step_log_artifacts", nested=True):
            for art in ["feature_list.json", "label_mapping.json",
                        HOURLY_CORRECTION_FILE, HOURLY_FLOOR_FILE,
                        SPIKE_CLASSIFIER_FILE, SPIKE_METADATA_FILE,
                        MODEL_METADATA_FILE, CHAMPION_METRICS_FILE,
                        "xgboost_model_p50.json",
                        "xgboost_model_p10.json",
                        "xgboost_model_p90.json"]:
                if os.path.exists(art):
                    mlflow.log_artifact(art)

    return models, df_trained, features_used, metrics


# ─────────────────────────────────────────────────────────────
# 11. MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    TRAIN_START = "2025-01-01"
    TRAIN_END   = "2026-01-10"
    GRID_SEARCH = "--grid-search" in sys.argv
    FORCE_PROMOTE = ("reset" in sys.argv) or ("--reset" in sys.argv)

    # 1. Refresh data
    load_latest_data_from_db(DATA_PATH)

    # 2. Feature engineering
    df_final = prepare_data(DATA_PATH, start_date=TRAIN_START, end_date=TRAIN_END)

    # 3. Split
    split_idx = int(len(df_final) * TRAIN_RATIO)
    print(f"\nTrain: {split_idx:,} rows | Test: {len(df_final) - split_idx:,} rows")

    if GRID_SEARCH:
        run_grid_search(df_final, split_idx, force_promote=FORCE_PROMOTE)
        df_for_forecast = df_final
    else:
        _, df_for_forecast, _, _ = run_with_mlflow(
            df_final,
            split_idx,
            end_date=TRAIN_END,
            force_promote=FORCE_PROMOTE
        )

    # 4. Always generate 13-day forecast
    print("\n" + "=" * 60)
    print("13-DAY FORECAST (Primary Evaluation)")
    print("=" * 60)
    future_13d = predict_future(df_for_forecast, periods=13 * 24)
    future_13d.to_csv("forecast_13d.csv", index=False)
    print(f"Saved forecast_13d.csv ({len(future_13d)} hours)")

    # 5. Evaluate against actuals
    try:
        forecast_start = pd.Timestamp(future_13d["ds"].min())
        forecast_end   = pd.Timestamp(future_13d["ds"].max())
        load_start     = (forecast_start - pd.Timedelta(days=8)).date()

        print(f"\n[TRACE] Loading actuals {load_start} → {forecast_end.date()}...")
        df_actual_wide = prepare_data(DATA_PATH,
                                      start_date=str(load_start),
                                      end_date=str(forecast_end.date()))
        df_actual_window = df_actual_wide[
            (df_actual_wide["ds"] >= forecast_start)
            & (df_actual_wide["ds"] <= forecast_end)
        ][["ds", "y"]].rename(columns={"y": "y_actual"})

        merged = future_13d.merge(df_actual_window, on="ds", how="inner")

        if len(merged) >= 24:
            mae_13d  = mean_absolute_error(merged["y_actual"], merged["y_pred"])
            rmse_13d = np.sqrt(mean_squared_error(merged["y_actual"], merged["y_pred"]))
            coverage = np.mean(
                (merged["y_actual"] >= merged["y_p10"])
                & (merged["y_actual"] <= merged["y_p90"])
            ) * 100
            bias     = float((merged["y_pred"] - merged["y_actual"]).mean())

            # High-volume bias (target: was −45.7 in v6 → expect improvement)
            hv_mask  = merged["y_actual"] > 50
            hv_bias  = float((merged.loc[hv_mask, "y_pred"]
                              - merged.loc[hv_mask, "y_actual"]).mean()) \
                if hv_mask.sum() > 0 else 0.0

            print(f"\n  13-Day Eval vs Actuals ({len(merged)} hours matched):")
            print(f"    MAE           : {mae_13d:.4f}")
            print(f"    RMSE          : {rmse_13d:.4f}")
            print(f"    Overall bias  : {bias:+.4f}  ({'over' if bias > 0 else 'under'}-estimating)")
            print(f"    High-vol bias : {hv_bias:+.4f}  (actual > 50 rph; v6 was −45.7)")
            print(f"    Coverage      : {coverage:.1f}%  (p10–p90 interval; v6 was 44.8%)")

            merged.to_csv("forecast_13d_vs_actuals.csv", index=False)
            print(f"    Saved to forecast_13d_vs_actuals.csv")

            mlflow.set_experiment(MLFLOW_EXPERIMENT)
            with mlflow.start_run(
                run_name=f"eval_13d_v{MODEL_VERSION}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            ):
                mlflow.log_metrics({
                    "eval_13d_mae":          mae_13d,
                    "eval_13d_rmse":         rmse_13d,
                    "eval_13d_bias":         bias,
                    "eval_13d_high_vol_bias": hv_bias,
                    "eval_13d_coverage":     coverage,
                    "eval_13d_hours":        len(merged),
                })
                mlflow.log_artifact("forecast_13d.csv")
                mlflow.log_artifact("forecast_13d_vs_actuals.csv")
        else:
            print(f"  (Insufficient actuals: {len(merged)} hours matched)")

    except Exception as ex:
        print(f"  (Actual comparison skipped: {ex})")

    # 6. Print first 24h
    print(f"\n{'Datetime':<22} {'p10':>8} {'p50 (pred)':>12} {'p90':>8}")
    print("-" * 55)
    for _, r in future_13d.head(24).iterrows():
        print(f"{str(r['ds']):<22} {r['y_p10']:>8.1f} {r['y_pred']:>12.1f} {r['y_p90']:>8.1f}")
    print("  ... (showing first 24h — full data in forecast_13d.csv)")
    print("\nTo view all MLflow runs:  mlflow ui")
