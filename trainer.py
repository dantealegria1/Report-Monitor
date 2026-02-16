import os
import pandas as pd
import numpy as np
import polars as pl
from prophet import Prophet
from prophet.serialize import model_to_json
import xgboost as xgb
import json
import holidays
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_mase_custom(y_train, y_test, y_pred):
    y_train = np.array(y_train, dtype=np.float64)
    y_test = np.array(y_test, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    
    naive_mae = np.mean(np.abs(np.diff(y_train)))
    if naive_mae < 1e-8: return 0.0
    
    model_mae = mean_absolute_error(y_test, y_pred)
    return model_mae / naive_mae

def prepare_hybrid_data(csv_path, start_date=None, end_date=None):
    print(f"🚀 Cargando y filtrando datos...")
    df = pl.read_csv(csv_path, ignore_errors=True)
    
    df = df.with_columns([
        pl.col("CreationDate").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S%.f", strict=False),
        pl.col("StartedProcessingOn").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S%.f", strict=False)
    ]).filter(pl.col("StartedProcessingOn").is_not_null())

    if start_date:
        df = df.filter(pl.col("StartedProcessingOn") >= datetime.strptime(start_date, "%Y-%m-%d"))
    if end_date:
        df = df.filter(pl.col("StartedProcessingOn") <= datetime.strptime(end_date, "%Y-%m-%d"))

    # 1. Agregación horaria
    hourly_ts = df.with_columns(
        pl.col("StartedProcessingOn").dt.truncate("1h").alias("ds")
    ).group_by("ds").agg([
        pl.len().alias("y"),
        pl.col("ReportType").mode().first().alias("main_report_type")
    ]).sort("ds")

    # 2. Ingeniería de Características (Pandas para Ventanas Móviles y Lags)
    pdf = hourly_ts.to_pandas()
    
    # --- MEMORIA (Lags) ---
    pdf['y_lag_1'] = pdf['y'].shift(1)
    pdf['y_lag_2'] = pdf['y'].shift(2)
    pdf['y_rolling_mean_3'] = pdf['y'].shift(1).rolling(window=3).mean()
    
    # --- FESTIVIDADES ---
    # Cambia 'AR' por el código de tu país (ej: 'MX', 'ES', 'US')
    country_holidays = holidays.CountryHoliday('AR') 
    pdf['is_holiday'] = pdf['ds'].apply(lambda x: 1 if x in country_holidays else 0)

    # --- PATRONES TEMPORALES ---
    pdf['hour'] = pdf['ds'].dt.hour
    pdf['weekday'] = pdf['ds'].dt.weekday
    pdf['day_of_month'] = pdf['ds'].dt.day
    pdf['month'] = pdf['ds'].dt.month
    pdf['is_month_end'] = pdf['ds'].dt.is_month_end.astype(int)
    pdf['is_weekend'] = pdf['weekday'].apply(lambda x: 1 if x >= 5 else 0)

    # 3. Backlog (Lógica original)
    print("⏳ Calculando Backlog (Variable de estado)...")
    def calculate_backlog_at(target_ds, reports_df):
        return reports_df.filter(
            (pl.col("CreationDate") <= target_ds) & 
            (pl.col("StartedProcessingOn") > target_ds)
        ).height
    
    pdf['backlog'] = [calculate_backlog_at(ts, df) for ts in pdf['ds']]
    
    pdf['tipo_reporte_id'] = pd.factorize(pdf['main_report_type'])[0]
    
    # Limpiar nulos creados por los Lags
    return pdf.dropna().reset_index(drop=True)

def train_hybrid_model(df):
    # STAGE 1: PROPHET (Estacionalidad pura + Feriados)
    print("🤖 Entrenando Stage 1: Prophet...")
    m = Prophet(
        daily_seasonality=True, 
        weekly_seasonality=True, 
        yearly_seasonality=True
    )
    m.add_country_holidays(country_name='AR') # Agrega feriados internos a Prophet
    m.fit(df[['ds', 'y']])
    
    forecast = m.predict(df[['ds']])
    df['yhat_prophet'] = forecast['yhat'].values
    df['residual'] = df['y'] - df['yhat_prophet']

    # STAGE 2: XGBOOST (Ajuste fino con variables operativas)
    print("🌳 Entrenando Stage 2: XGBoost...")
    features = [
        'backlog', 'hour', 'weekday', 'day_of_month', 'month',
        'is_month_end', 'is_holiday', 'is_weekend', 'tipo_reporte_id',
        'y_lag_1', 'y_lag_2', 'y_rolling_mean_3'
    ]
    
    X = df[features]
    y_res = df['residual']

    # Hiperparámetros optimizados para reducir MASE
    xgb_model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:absoluteerror' # Optimiza directamente el MAE
    )
    xgb_model.fit(X, y_res)
    
    df['xgb_adj'] = xgb_model.predict(X)
    df['y_pred_hybrid'] = df['yhat_prophet'] + df['xgb_adj']

    # Guardar
    with open('prophet_model.json', 'w') as fout:
        fout.write(model_to_json(m))
    xgb_model.save_model('xgboost_model.json')
    
    label_map = dict(enumerate(pd.factorize(df['main_report_type'])[1]))
    with open('label_mapping.json', 'w') as f:
        json.dump(label_map, f)

    return m, xgb_model, df

def evaluate_and_report(df):
    # Separación cronológica 80/20
    split_idx = int(len(df) * 0.8)
    train_set = df.iloc[:split_idx].copy()
    test_set = df.iloc[split_idx:].copy()
    
    # --- CÁLCULO DEL BASELINE (NAIVE) ---
    # Para el MASE (denominador): se usa el MAE Naive del set de ENTRENAMIENTO
    y_train_vals = train_set['y'].values.astype(np.float64)
    naive_mae_train = np.mean(np.abs(np.diff(y_train_vals)))
    if naive_mae_train < 1e-8: naive_mae_train = 1.0 # Avoid division by zero
    
    # Para la TABLA COMPARATIVA: usamos el error Naive en el set de TEST (y_t = y_{t-1})
    # Esto es para que sea comparable con el MAE de Prophet e Híbrido en el mismo set.
    y_test_naive_pred = test_set['y'].shift(1).fillna(train_set['y'].iloc[-1]).values
    naive_mae_test = mean_absolute_error(test_set['y'], y_test_naive_pred)
    
    # --- MÉTRICAS PROPHET ---
    mae_p = mean_absolute_error(test_set['y'], test_set['yhat_prophet'])
    rmse_p = np.sqrt(mean_squared_error(test_set['y'], test_set['yhat_prophet']))
    mase_p = mae_p / naive_mae_train # Definición estándar de MASE
    
    # --- MÉTRICAS HÍBRIDO ---
    mae_h = mean_absolute_error(test_set['y'], test_set['y_pred_hybrid'])
    rmse_h = np.sqrt(mean_squared_error(test_set['y'], test_set['y_pred_hybrid']))
    mase_h = mae_h / naive_mae_train

    # Guardar métricas para el Dashboard
    metrics = {
        "mae_p": mae_p, "rmse_p": rmse_p, "mase_p": mase_p,
        "mae_h": mae_h, "rmse_h": rmse_h, "mase_h": mase_h,
        "naive_mae": naive_mae_test
    }
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f)

    print("\n" + "="*65)
    print(f"{'Métrica':<15} | {'Naive (Baseline)':<18} | {'Prophet Base':<15} | {'Híbrido'}")
    print("-"*65)
    # El Baseline por definición tiene MASE = 1
    print(f"{'MAE':<15} | {naive_mae_test:<18.4f} | {mae_p:<15.4f} | {mae_h:.4f}")
    print(f"{'RMSE':<15} | {'N/A':<18} | {rmse_p:<15.4f} | {rmse_h:.4f}")
    print(f"{'MASE':<15} | {1.0000:<18.4f} | {mase_p:<15.4f} | {mase_h:.4f}")
    print("="*65)
    
    improvement_mae = (naive_mae_test - mae_h) / naive_mae_test * 100
    print(f"Mejora del Híbrido vs Baseline Naive: {improvement_mae:.2f}%")
    print(f"Reducción de Error (RMSE) Prophet vs Híbrido: {((rmse_p-rmse_h)/rmse_p)*100:.2f}%")
    print("="*65)

if __name__ == "__main__":
    DATA_PATH = "DatosQuery.csv"
    # Ajusta tu rango aquí
    df_final = prepare_hybrid_data(DATA_PATH, start_date="2025-01-01", end_date="2026-01-19")
    _, _, df_trained = train_hybrid_model(df_final)
    evaluate_and_report(df_trained)