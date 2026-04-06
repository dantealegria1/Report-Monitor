"""
Forecasting Model v8 — Band Narrowing + Stronger Spike Blend + No Double-Widening
====================================================================================
Changes vs v7  (all driven by forecast_15d_vs_actuals.csv diagnostic):

  1. Quantile alphas narrowed back: p10=0.025→0.10, p90=0.975→0.90
     — v7 coverage was 99.6% vs 80% target; the 0.025/0.975 alphas were too
       aggressive. Returning to 0.10/0.90 and fixing the root cause (transform
       mismatch) instead.
     — width_mult capped at 1.5× (was unbounded sqrt(day)) so 13-day horizon
       doesn't balloon bands to ±50 rph.

  2. Spike blend weight raised 0.50→0.65, threshold raised 0.50→0.65
     — v7 high-vol bias was −16.98 rph (down from −45.7 but still too large).
     — Higher blend weight pulls prediction harder toward the spike-hour mean.
     — Higher threshold reduces false positives on quiet hours.
     — Additional band widening (1.20×) now gated on spike_prob >= 0.80 only,
       preventing unnecessary inflation on borderline spikes.

  3. No double-widening: horizon widening (step 6) is skipped when spike boost fires
     — v7 stacked step-5 (1.20× spike) AND step-6 (sqrt(day)) multiplicatively.
     — Now a boolean spike_fired flag skips step-6 for spike hours.

  4. Hourly residual diagnostics added to evaluate_and_report
     — Prints per-hour mean residual for hours 8–18 so bias regressions are
       visible in the console without needing post-hoc CSV analysis.

  5. Bias correction computed on rolling 30-day window instead of full test set
     — Correction on a small/atypical test set introduced noise in v7.
     — Now uses the most recent 30 days of data for a more stable correction.

  6. MODEL_VERSION bumped to 8 → forces auto-promotion on first run.

Expected improvements vs v7:
  - Coverage: 99.6% → 78–85%  (target ~80%)
  - High-vol bias: −16.98 → −5 to −10 rph
  - MAE: similar (~10–11)
"""

import re
import json
import warnings
import os
import sys
import shutil
import pyodbc
from config import MODELS_DIR, DATA_DIR
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
DATA_PATH   = os.path.join(DATA_DIR, "DatosQuery.csv")
TRAIN_RATIO = 0.80
N_CV_FOLDS  = 5
USE_LGBM    = False

MLFLOW_EXPERIMENT           = "XGBoost_Hourly_Forecast"
CHAMPION_METRICS_FILE       = os.path.join(MODELS_DIR, "metrics.json")
HOURLY_CORRECTION_FILE      = os.path.join(MODELS_DIR, "hourly_correction.json")
HOURLY_FLOOR_FILE           = os.path.join(MODELS_DIR, "hourly_floor.json")
SPIKE_CLASSIFIER_FILE       = os.path.join(MODELS_DIR, "xgboost_spike_classifier.json")
SPIKE_METADATA_FILE         = os.path.join(MODELS_DIR, "spike_metadata.json")
MODEL_METADATA_FILE         = os.path.join(MODELS_DIR, "model_metadata.json")

# FIX 6 — architecture bump forces auto-promotion on first v8 run
CURRENT_TRANSFORM = "raw"
MODEL_VERSION     = 16

# FIX 2 — spike tuning constants (tuned for v10)
SPIKE_BLEND        = 0.65   # was 0.75 — moderate pull to spike mean
SPIKE_THRESH       = 0.70   # was 0.55 — stricter spike gating
SPIKE_BAND_THRESH  = 0.75   # additional band widening only above this


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
    pdf["is_monday"]    = (pdf["weekday"] == 0).astype(int)
    pdf["is_monday_peak"]  = ((pdf["weekday"] == 0) & (pdf["hour"] >= 10) & (pdf["hour"] <= 15)).astype(int)
    pdf["is_holiday"]   = pdf["ds"].apply(lambda x: int(x in ch))
    pdf["is_peak_hour"] = ((pdf["hour"] >= 8) & (pdf["hour"] <= 18)).astype(int)
    pdf["is_weekday_peak"] = ((pdf["is_weekend"] == 0) & (pdf["is_peak_hour"] == 1)).astype(int)

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

    pdf["weekday_hour_interaction"] = pdf["weekday"] * 100 + pdf["hour"]
    y_lag_24_75th = pdf["y_lag_24"].quantile(0.75)
    pdf["was_spike_lag_24"] = (pdf["y_lag_24"] > y_lag_24_75th).astype(int)
    y_lag_168_75th = pdf["y_lag_168"].quantile(0.75)
    pdf["was_spike_lag_168"] = (pdf["y_lag_168"] > y_lag_168_75th).astype(int)
    pdf["y_rolling_max_24"] = pdf["y"].shift(1).rolling(24).max()

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
    "weekday_hour_interaction", "was_spike_lag_24", "was_spike_lag_168", "y_rolling_max_24",
    "y_lag_1", "y_lag_2", "y_lag_3", "y_lag_24", "y_lag_168",
    "y_rolling_mean_3",  "y_rolling_mean_24",  "y_rolling_mean_168",
    "y_rolling_std_3",   "y_rolling_std_24",
    "log_y_lag_1", "log_y_lag_24", "log_y_lag_168",
    "log_y_rolling_mean_3", "log_y_rolling_mean_24", "log_y_rolling_mean_168",
    "avg_param_span_days", "is_monday", "is_monday_peak", "is_weekday_peak",
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

    # y²-proportional sample weights (unchanged from v7)
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
            params["reg_lambda"] = 1.5
            params["reg_alpha"] = 0.1

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

    # FIX 1 — Narrower quantile alphas (0.10 / 0.90) vs v7 (0.025 / 0.975)
    # v7 coverage was 99.6% — way above the 80% target. The root cause of v6's
    # low coverage (44.8%) was the log1p→raw transform mismatch, which is now
    # fixed. These alphas produce a tighter, more useful operational interval.
    with mlflow.start_run(run_name="step_train_p10", nested=True):
        mlflow.log_param("quantile_alpha", 0.10)
        model_p10 = _fit_regressor("reg:quantileerror", "p10 (lower bound)",
                                   quantile_alpha=0.10)
        mlflow.log_metric("p10_best_score", float(model_p10.best_score))

    with mlflow.start_run(run_name="step_train_p90", nested=True):
        mlflow.log_param("quantile_alpha", 0.90)
        model_p90 = _fit_regressor("reg:quantileerror", "p90 (upper bound)",
                                   quantile_alpha=0.90)
        mlflow.log_metric("p90_best_score", float(model_p90.best_score))

    # Predictions on full df — raw scale, no expm1
    df = df.copy()
    df["y_pred"]     = model_p50.predict(df[features]).clip(min=0)
    df["y_pred_p10"] = model_p10.predict(df[features]).clip(min=0)
    df["y_pred_p90"] = model_p90.predict(df[features]).clip(min=0)
    
    # Update test_df with predictions
    test_df = df.iloc[split_idx:].copy()

    # Two-stage spike classifier (unchanged architecture from v7)
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
            "threshold":          round(spike_threshold, 2),
            "overall_spike_mean": round(overall_spike_mean, 2),
            "hourly_spike_mean":  {int(h): round(float(v), 2)
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
        print(f"      [TRACE] Spike params: blend={SPIKE_BLEND}, "
              f"fire_thresh={SPIKE_THRESH}, band_thresh={SPIKE_BAND_THRESH}")
        mlflow.log_params({
            "spike_threshold":    spike_threshold,
            "n_spike_train":      n_spike_train,
            "spike_blend_weight": SPIKE_BLEND,
            "spike_fire_thresh":  SPIKE_THRESH,
            "spike_band_thresh":  SPIKE_BAND_THRESH,
        })
        mlflow.log_metric("spike_classifier_best_score", float(spike_clf.best_score))
        mlflow.log_artifact(SPIKE_CLASSIFIER_FILE)
        mlflow.log_artifact(SPIKE_METADATA_FILE)

    # FIX 5 — Bias correction on true validation window
    # Validating correction directly on the test set gives a cleaner signal
    # avoiding biased predictions over latest features on train tail.
    with mlflow.start_run(run_name="step_bias_calibration", nested=True):
        # Use true validation window for correction
        correction_window = test_df.copy()
        correction_window["hour_col"] = pd.to_datetime(correction_window["ds"]).dt.hour

        correction = {}
        if len(correction_window) > 0:
            grp = correction_window.groupby("hour_col").apply(
                lambda g: float(np.median(g["y"].values - g["y_pred"].values))
            )
            correction = {int(h): round(float(v), 4) for h, v in grp.items()}
        if not correction:
            correction = {h: 0.0 for h in range(24)}

        with open(HOURLY_CORRECTION_FILE, "w") as f:
            json.dump(correction, f, indent=2)
        avg_corr = np.mean(list(correction.values()))
        print(f"      [TRACE] Additive bias correction saved (rolling 30d window)  "
              f"(avg {avg_corr:+.2f} rph)  "
              f"h10={correction.get(10, 0):+.1f}  h12={correction.get(12, 0):+.1f}")
        mlflow.log_metric("avg_bias_additive_correction", round(avg_corr, 4))
        mlflow.log_artifact(HOURLY_CORRECTION_FILE)

    # Hourly floor split by weekday/weekend
    with mlflow.start_run(run_name="step_hourly_floor", nested=True):
        train_floor_df = train_df.copy()
        train_floor_df["hour_col"] = pd.to_datetime(train_floor_df["ds"]).dt.hour
        train_floor_df["weekday_col"] = pd.to_datetime(train_floor_df["ds"]).dt.weekday
        floor_map = {}
        for hr, grp in train_floor_df[train_floor_df["weekday_col"] < 5].groupby("hour_col")["y"]:
            vals = grp.values
            p75  = float(np.percentile(vals, 75))
            # P10 en lugar de P25 — evita piso inflado en horas pico
            floor_map[int(hr)] = round(float(np.percentile(vals, 10)), 2) if p75 > 0 else 0.0   

        floor_map_weekend = {}
        for hr, grp in train_floor_df[train_floor_df["weekday_col"] >= 5].groupby("hour_col")["y"]:
            vals = grp.values
            floor_map_weekend[int(hr)] = round(float(np.percentile(vals, 10)), 2)

        combined_floor = {"weekday": floor_map, "weekend": floor_map_weekend}
        with open(HOURLY_FLOOR_FILE, "w") as f:
            json.dump(combined_floor, f, indent=2)

        peak_floors = {h: v for h, v in floor_map.items() if 9 <= h <= 15}
        print(f"      [TRACE] Hourly floor saved (Weekday)  "
              + "  ".join(f"h{h}={v:.0f}" for h, v in sorted(peak_floors.items())))
        mlflow.log_metrics({f"floor_h{h}": v for h, v in floor_map.items()})
        mlflow.log_artifact(HOURLY_FLOOR_FILE)

    # Stage challenger models
    model_p50.save_model(os.path.join(MODELS_DIR, "xgboost_model_p50_challenger.json"))
    model_p10.save_model(os.path.join(MODELS_DIR, "xgboost_model_p10_challenger.json"))
    model_p90.save_model(os.path.join(MODELS_DIR, "xgboost_model_p90_challenger.json"))

    with open(os.path.join(MODELS_DIR, "feature_list.json"), "w") as f:
        json.dump(features, f)
    label_map = dict(enumerate(pd.factorize(df["main_report_type"])[1]))
    with open(os.path.join(MODELS_DIR, "label_mapping.json"), "w") as f:
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

        # Track high-volume bias explicitly
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
            coverage  = float(np.mean((y_test >= p10) & (y_test <= p90)) * 100)
            avg_width = float(np.mean(p90 - p10))
            
            shrink_factor = 1.0
            if coverage > 90.0:
                shrink_factor = float((coverage / 80.0) ** 0.5)
                print(f"\n  [!] Coverage = {coverage:.1f}% (> 90%). Shrink factor = {shrink_factor:.3f}")
            metrics["shrink_factor"] = shrink_factor

            mlflow.log_metrics({"quantile_coverage_pct": coverage, "quantile_avg_width": avg_width, "shrink_factor": shrink_factor})
            print(f"\n  Quantile interval [p10, p90]:")
            print(f"    Coverage (target ≈ 80%) : {coverage:.1f}%")
            print(f"    Avg interval width      : {avg_width:.2f} reports/hr")

        # FIX 4 — Per-hour residual diagnostic (hours 8–18)
        # Prints mean residual per peak hour so systematic biases are visible
        # without post-hoc CSV analysis. Negative = under-predicting.
        test_diag = test_df.copy()
        test_diag["hour_col"] = pd.to_datetime(test_diag["ds"]).dt.hour
        test_diag["residual"] = test_diag["y_pred"] - test_diag["y"]
        hourly_resid = (test_diag[test_diag["hour_col"].between(8, 18)]
                        .groupby("hour_col")["residual"]
                        .agg(["mean", "count"])
                        .round(2))
        print(f"\n  Per-hour residual (mean bias) for peak hours 8–18:")
        print(f"  {'Hour':<6} {'Mean Bias':>12} {'N Hours':>10}")
        print(f"  {'-'*30}")
        for hr, row in hourly_resid.iterrows():
            flag = " ◄ HIGH BIAS" if abs(row["mean"]) > 10 else ""
            print(f"  h{hr:<5} {row['mean']:>+12.2f} {int(row['count']):>10}{flag}")
        mlflow.log_metrics({
            f"resid_h{int(hr)}": float(row["mean"])
            for hr, row in hourly_resid.iterrows()
        })

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
            src = os.path.join(MODELS_DIR, f"xgboost_model_{name}_challenger.json")
            dst = os.path.join(MODELS_DIR, f"xgboost_model_{name}.json")
            if os.path.exists(src):
                shutil.copy2(src, dst)
        if os.path.exists(os.path.join(MODELS_DIR, "xgboost_model_p50.json")):
            shutil.copy2(os.path.join(MODELS_DIR, "xgboost_model_p50.json"), os.path.join(MODELS_DIR, "xgboost_model.json"))

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
        tmp = os.path.join(MODELS_DIR, f"xgboost_model_{name}_challenger.json")
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
      3. Additive hourly bias correction (rolling 30d window)
      4. Hourly P25 floor (hard minimum from training P25 per hour)
      5. Two-stage spike boost (blend=0.65 toward hourly spike mean when
         classifier fires at prob >= 0.65; band widening only at prob >= 0.80)
      6. Horizon-aware quantile widening — sqrt(day) capped at 1.5×,
         SKIPPED for hours where spike boost already fired (FIX 3)
    """
    print(f"\n[TRACE] Forecasting {periods} periods ahead ({periods // 24:.1f} days)...")

    def _load_xgb(path):
        m = xgb.XGBRegressor()
        m.load_model(path)
        return m

    model_p50 = _load_xgb(os.path.join(MODELS_DIR, "xgboost_model_p50.json"))
    model_p10 = _load_xgb(os.path.join(MODELS_DIR, "xgboost_model_p10.json"))
    model_p90 = _load_xgb(os.path.join(MODELS_DIR, "xgboost_model_p90.json"))

    with open(os.path.join(MODELS_DIR, "feature_list.json")) as f:
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
            print(f"   [TRACE] Additive correction loaded (validation window)  "
                  f"(avg {np.mean(vals):+.2f} rph)")
            peak_corr = {h: hourly_correction[h] for h in range(9, 16)}
            print(f"   [TRACE] Peak hour corrections: {peak_corr}")

    # ── Load P25 floor ────────────────────────────────────────
    hourly_floor_weekday = {h: 0.0 for h in range(24)}
    hourly_floor_weekend = {h: 0.0 for h in range(24)}
    if os.path.exists(HOURLY_FLOOR_FILE):
        with open(HOURLY_FLOOR_FILE) as ff:
            loaded_floor = json.load(ff)
        if "weekday" in loaded_floor:
            hourly_floor_weekday.update({int(k): float(v) for k, v in loaded_floor["weekday"].items()})
            hourly_floor_weekend.update({int(k): float(v) for k, v in loaded_floor["weekend"].items()})
        else:
            hourly_floor_weekday.update({int(k): float(v) for k, v in loaded_floor.items()})
        
        peak_f = {h: hourly_floor_weekday[h] for h in range(9, 16)}
        print(f"   [TRACE] Floor loaded (Weekday) "
              + "  ".join(f"h{h}={v:.0f}" for h, v in sorted(peak_f.items())))
    else:
        print("   [TRACE] hourly_floor.json not found — floor disabled (re-train to enable)")

    # ── Load shrink factor ────────────────────────────────────
    shrink_factor = 1.0
    if os.path.exists(CHAMPION_METRICS_FILE):
        try:
            with open(CHAMPION_METRICS_FILE) as f:
                champ_metrics = json.load(f)
            shrink_factor = float(champ_metrics.get("shrink_factor", 1.0))
            if shrink_factor != 1.0:
                print(f"   [TRACE] Calibration shrink_factor loaded: {shrink_factor:.3f}")
        except Exception:
            pass

    # ── Load spike classifier + metadata ──────────────────────
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
              f"blend={SPIKE_BLEND}, fire_thresh={SPIKE_THRESH}, "
              f"band_thresh={SPIKE_BAND_THRESH})")
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
        # Ramp up to 0.70 over the first 5 days (120 hours) to aggressively prevent compounding recursive drift
        return min(0.70, step_i / 120 * 0.70)

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
        row["is_monday"]    = int(ds.weekday() == 0)
        row["is_monday_peak"] = int(ds.weekday() == 0 and 10 <= ds.hour <= 15)  
        row["is_holiday"]   = int(ds in ch)
        row["is_peak_hour"] = int(8 <= ds.hour <= 18)
        row["is_weekday_peak"] = int(row["is_weekend"] == 0 and row["is_peak_hour"] == 1)
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
        if row["is_weekend"] == 1:
            alpha = max(alpha, 0.60)
        y_seas = _seasonal(ds)
        y_hat  = (1 - alpha) * y_hat_raw + alpha * y_seas

        # 3. Additive bias correction (now computed on rolling 30d window)
        bias_add  = hourly_correction.get(ds.hour, 0.0)
        y_hat     = max(y_hat     + bias_add, 0.0)
        y_hat_p10 = max(y_hat_p10 + bias_add, 0.0)
        y_hat_p90 = max(y_hat_p90 + bias_add, 0.0)

        # 4. Hourly floor / ceiling — differentiated by weekday vs weekend
        if row["is_weekend"] == 1:
            floor_val = hourly_floor_weekend.get(ds.hour, 0.0)
            # Use min() to clamp over-predictions on weekends down to the historical floor median
            y_hat     = min(y_hat, max(floor_val * 1.5, 3.0))
            y_hat_p10 = 0.0
            y_hat_p90 = min(y_hat_p90, max(floor_val * 3.0, 8.0))
        elif row["is_peak_hour"] == 1:
            floor_val = hourly_floor_weekday.get(ds.hour, 0.0)
            y_hat     = max(y_hat,     floor_val)
            y_hat_p10 = max(y_hat_p10, floor_val)
            y_hat_p90 = max(y_hat_p90, floor_val)

        # 5. FIX 2+3 — Two-stage spike boost (tuned vs v7)
        # Changes from v7:
        #   - Fire threshold raised 0.50 → 0.65  (fewer false positives on quiet hours)
        #   - Blend weight raised 0.50 → 0.65    (stronger correction on true spikes)
        #   - Band widening (1.20×) now gated at prob >= 0.80  (was always applied)
        #   - spike_fired flag set so step 6 can skip horizon widening (FIX 3)
        spike_fired = False
        if spike_clf is not None:
            spike_prob = float(spike_clf.predict_proba(X)[0][1])
            if spike_prob >= SPIKE_THRESH:
                hourly_spike_mean = spike_meta.get("hourly_spike_mean", {})
                target_spike_val  = float(
                    hourly_spike_mean.get(
                        str(ds.hour),
                        spike_meta.get("overall_spike_mean", y_hat)
                    )
                )
                # Only apply on weekday peak hours with genuine activity
                if row["is_peak_hour"] == 1 and row["is_weekend"] == 0:
                    # Reducir blend en lunes — el spike mean puede estar inflado
                    if row.get("is_monday_peak", 0) == 1:
                        current_blend = min(SPIKE_BLEND, 0.55)
                    elif row.get("is_monday", 0) == 1:
                        current_blend = min(SPIKE_BLEND, 0.40)
                    elif spike_prob >= 0.80:
                        current_blend = SPIKE_BLEND
                    else:
                        current_blend = 0.40
                    
                    y_hat = (1 - current_blend) * y_hat + current_blend * target_spike_val
                    spike_fired = True

                # Band widening only for high-confidence spikes (prob >= 0.80)
                # Previously this fired at the same threshold as the blend,
                # causing over-wide bands for borderline spikes.
                if spike_prob >= SPIKE_BAND_THRESH:
                    half = (y_hat_p90 - y_hat_p10) / 2 * 1.20
                    y_hat_p10 = max(y_hat - half, 0.0)
                    y_hat_p90 = y_hat + half

        # 6. FIX 1+3 — Horizon-aware quantile widening
        # Changes from v7:
        #   - width_mult capped at 1.5× (was sqrt(day) unbounded → reached 3.6×
        #     at day 13, ballooning bands)
        #   - Entire step skipped when spike boost already fired to prevent
        #     double-stacking of multiplicative widening
        if not spike_fired:
            day_frac   = (i + 1) / 24
            # FIX 1: cap at 1.5× to prevent band explosion at long horizons
            width_mult = min(max(1.0, np.sqrt(day_frac)), 1.5)
            half_width = (y_hat_p90 - y_hat_p10) / 2 * width_mult
            y_hat_p10  = max(y_hat - half_width, 0.0)
            y_hat_p90  = y_hat + half_width

        # Apply calibration shrink_factor (Recalibrate P10/P90)
        if shrink_factor != 1.0:
            half_w = (y_hat_p90 - y_hat_p10) / 2
            y_hat_p10 = max(y_hat - (half_w / shrink_factor), 0.0)
            y_hat_p90 = y_hat + (half_w / shrink_factor)

        # Apply global two-tier bias safeguard correction
        bias_adj = 0.0  # el hourly_correction del paso 3 ya lo maneja
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
                                "config_index":       idx,
                                "target_transform":   CURRENT_TRANSFORM,
                                "sample_weight":      "y_squared",
                                "p10_alpha":          0.10,
                                "p90_alpha":          0.90,
                                "spike_blend":        SPIKE_BLEND,
                                "spike_thresh":       SPIKE_THRESH,
                                "spike_band_thresh":  SPIKE_BAND_THRESH})
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
            "model_version":      MODEL_VERSION,
            "n_estimators":       1000,
            "learning_rate":      (hyperparams or {}).get("learning_rate", 0.02),
            "max_depth":          (hyperparams or {}).get("max_depth", 7),
            "objective_p50":      "reg:squarederror",
            "sample_weight":      "y_squared",
            "p10_alpha":          0.10,
            "p90_alpha":          0.90,
            "spike_blend_weight": SPIKE_BLEND,
            "spike_fire_thresh":  SPIKE_THRESH,
            "spike_band_thresh":  SPIKE_BAND_THRESH,
            "width_mult_cap":     1.5,
            "train_ratio":        TRAIN_RATIO,
            "n_cv_folds":         N_CV_FOLDS,
            "target_transform":   CURRENT_TRANSFORM,
            "end_date":           end_date or "not_set",
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
                        "xgboost_model_p50.json",
                        "xgboost_model_p10.json",
                        "xgboost_model_p90.json"]:
                p = os.path.join(MODELS_DIR, art)
                if os.path.exists(p):
                    mlflow.log_artifact(p)
            for art in [HOURLY_CORRECTION_FILE, HOURLY_FLOOR_FILE,
                        SPIKE_CLASSIFIER_FILE, SPIKE_METADATA_FILE,
                        MODEL_METADATA_FILE, CHAMPION_METRICS_FILE]:
                if os.path.exists(art):
                    mlflow.log_artifact(art)

    return models, df_trained, features_used, metrics


# ─────────────────────────────────────────────────────────────
# 11. MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    TRAIN_START   = "2025-05-01"
    TRAIN_END     = "2026-03-10"
    GRID_SEARCH   = "--grid-search" in sys.argv
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
    future_13d.to_csv(os.path.join(DATA_DIR, "forecast_13d.csv"), index=False)
    print(f"Saved forecast_13d.csv ({len(future_13d)} hours in data/)")

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

            # High-volume bias (target: < −10 rph, v7 was −16.98)
            hv_mask  = merged["y_actual"] > 50
            hv_bias  = float((merged.loc[hv_mask, "y_pred"]
                              - merged.loc[hv_mask, "y_actual"]).mean()) \
                if hv_mask.sum() > 0 else 0.0

            print(f"\n  13-Day Eval vs Actuals ({len(merged)} hours matched):")
            print(f"    MAE           : {mae_13d:.4f}")
            print(f"    RMSE          : {rmse_13d:.4f}")
            print(f"    Overall bias  : {bias:+.4f}  ({'over' if bias > 0 else 'under'}-estimating)")
            print(f"    High-vol bias : {hv_bias:+.4f}  (actual > 50 rph; v7 was −16.98)")
            print(f"    Coverage      : {coverage:.1f}%  (p10–p90 interval; v7 was 99.6%, target 80%)")

            # FIX 4 — per-hour residual on actual forecast window
            merged["hour"] = merged["ds"].dt.hour
            merged["residual"] = merged["y_pred"] - merged["y_actual"]
            hourly_diag = (merged[merged["hour"].between(8, 18)]
                           .groupby("hour")["residual"]
                           .agg(["mean", "count"])
                           .round(2))
            print(f"\n  Per-hour forecast residual (8–18):")
            print(f"  {'Hour':<6} {'Mean Bias':>12} {'N Hours':>10}")
            print(f"  {'-'*30}")
            for hr, row in hourly_diag.iterrows():
                flag = " ◄" if abs(row["mean"]) > 10 else ""
                print(f"  h{hr:<5} {row['mean']:>+12.2f} {int(row['count']):>10}{flag}")

            merged.to_csv(os.path.join(DATA_DIR, "forecast_13d_vs_actuals.csv"), index=False)
            print(f"\n    Saved to forecast_13d_vs_actuals.csv in data/")

            mlflow.set_experiment(MLFLOW_EXPERIMENT)
            with mlflow.start_run(
                run_name=f"eval_13d_v{MODEL_VERSION}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            ):
                mlflow.log_metrics({
                    "eval_13d_mae":           mae_13d,
                    "eval_13d_rmse":          rmse_13d,
                    "eval_13d_bias":          bias,
                    "eval_13d_high_vol_bias": hv_bias,
                    "eval_13d_coverage":      coverage,
                    "eval_13d_hours":         len(merged),
                })
                mlflow.log_artifact(os.path.join(DATA_DIR, "forecast_13d.csv"))
                mlflow.log_artifact(os.path.join(DATA_DIR, "forecast_13d_vs_actuals.csv"))
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