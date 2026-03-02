# Report Monitor System Implementation (v7)

## Overview
The Report Monitor is a Streamlit-based operations intelligence and monitoring application. It performs real-time diagnostics on query execution volumes, failure rates, and statistical drift, while forecasting future network load and resource requirements via machine learning.

## Modeling Pipeline ([trainer.py](file:///c:/Tesina_Avances/trainer.py))
The system employs an end-to-end ML training orchestrator powered by **Polars** (high-speed processing), **XGBoost** (gradient boosting), and **MLflow** (experiment tracking + model governance). It operates on a Champion/Challenger promotion framework.

### 1. Feature Engineering
- **Target Scale:** The base execution counts (`y`: reports per hour) are explicitly treated in their original distribution during `XGBoost` training, avoiding `log1p` target transformations in favor of weighting the loss function.
- **Calendar & Time:** Computes cyclic hour, weekday, and month transformations (`sin`/`cos`). Adds binary features for weekends, holidays, peak hours (`08:00 - 18:00`), and month-ends.
- **Fourier terms:** Captures $24$-hour and $168$-hour (weekly) continuous operational seasonality.
- **Autoregressive Lags:** Exploits `t-1`, `t-2`, `t-3`, `t-24`, and `t-168` lags alongside rolling standard deviations. The lags themselves are log-scaled (`np.log1p`) to compress extreme right-tail deviations before feeding into the tree splits.
- **Operational Triggers:** Calculates a real-time [backlog](file:///c:/Tesina_Avances/trainer.py#194-199) logic via as-of-joins (reports queued but not started), and an `avg_param_span_days` detailing how many days of data a query spans.

### 2. Multi-Model Quantile Regressors (v7 Architecture)
To adequately capture asymmetric demand (burst loads) and uncertainty, the system simultaneously trains three quantile regressors per lifecycle:
- **p50 (Median):** Trains on a standard `reg:squarederror` objective. To combat structural underestimation on high-volume periods ($-45$ rph bias on 100+ rph hours), a dynamic sample weight $\frac{y^2}{mean(y^2)}$ heavily penalizes under-predictions on peak hours.
- **p10 & p90 (Bounds):** Trained on `reg:quantileerror` explicitly capturing the $0.025^{th}$ and $0.975^{th}$ probability mass (a 95% interval).

### 3. Two-Stage Spike Classifier
To prevent structural underestimation of sudden burst loads:
- A secondary XGBoost binary classifier ([xgboost_spike_classifier.json](file:///c:/Tesina_Avances/xgboost_spike_classifier.json)) is concurrently trained to predict whether an upcoming hour will exceed the $75^{th}$ percentile of the training volume.
- During autoregressive inference, if the classifier fires ($P(spike) > 0.5$) AND the base `p50` prediction is lower than the historical seasonal mean for that specific hour of the day during spike events, the prediction is blended upwards towards that known high-volume conditional mean.

### 4. Calibration and Inference Engine Constraints
The [predict_future](file:///c:/Tesina_Avances/trainer.py#704-934) function predicts load 13 days out (312 hours) dynamically applying constraints:
1. **Seasonal Trend Blending:** For horizons far in the future ($>24h$), autoregressive error rapidly compounds. The predictions are gradually blended (increasing seamlessly to 40% weighting) with a strict 4-week historical seasonal average.
2. **Additive Bias Correction:** Evaluates the holdout set median residuals per hour ($0-23$), applying a fixed additive offset to the final output.
3. **Hourly P25 Floor:** A hard algorithmic floor derived from the 25th percentile of authentic historical activity per operational hour ($9-15$). This explicitly guarantees the model will not output unrealistic negative or essentially zero capacities during known busy shifts.
4. **Horizon Uncertainty Widening:** Unseen variance is intrinsically magnified further out. The $p10$ and $p90$ interval bands are proportionally adjusted by $\sqrt{horizon\_day}$ increasing operational safety margins.

### 5. Evaluation & Champion/Challenger Promotion Logic
- **Cross validation:** Conducts a 5-fold Walk-Forward split.
- evaluates model precision versus a *Naive Baseline* (value at `t-1`) tracking MASE (Mean Absolute Scaled Error), RMSE, MAE, and evaluating quantile coverage (expected $\approx 80\%$). 
- Explicitly tracks custom `high_vol_bias` metrics (bias strictly on hours $>50$ rph).
- **Promotion Rule:** The newly trained models are staged as "challengers". If their *Validation MASE* outperforms the *Champion MASE* (stored in [metrics.json](file:///c:/Tesina_Avances/metrics.json)), the models overwrite the main JSON assets predicting forward. Alternatively, a version bump (e.g. into `MODEL_VERSION = 7`) forces an automatic promotion via [model_metadata.json](file:///c:/Tesina_Avances/model_metadata.json) invalidation.

## Anomaly Detection & UI Overview (`pages/`)

### A. Anomaly Detection ([pages/7_Anomaly_Detection.py](file:///c:/Tesina_Avances/pages/7_Anomaly_Detection.py))
Provides multiple interchangeable paradigms to detect uncharacteristic code/query behaviors:
- **Detection Methodologies:** Robust Z-Score via Median Absolute Deviations (MAD), Contextual limits (bound individually per report, per hour of day), and Multivariable Isolation Forests.
- **Forensic Git Diagnostics:** When an anomaly fires, the UI isolates the affected execution via the SQL repository mapping and provides the user the Git History (Commit hashes, authors, diff links) surrounding the very hour the report began deviating.
- **Feature Drift (KS Test):** Employs Continuous Distribution Kolmogorov-Smirnov statistics applied upon rolling time windows (e.g. Previous 30 Days vs Last 7 Days) for measuring degradation in fundamental reporting capacity.

### B. Analytical Dashboards & Alternate Engines
- **Prophet Forecast Base ([pages/3_Prophet_Forecast.py](file:///c:/Tesina_Avances/pages/3_Prophet_Forecast.py)):** Operates as a mathematically rigid alternate additive model integrating explicit `log1p` transforms. It optionally augments standard Prophet behavior with a `Lag-1 Regressor` blending AutoRegressive behavior with deterministic seasonality patterns.
- **XGBoost Dashboard ([pages/4_XGBoost_Analytical_Dashboard.py](file:///c:/Tesina_Avances/pages/4_XGBoost_Analytical_Dashboard.py)):** Interactive UI serving real-time predictions directly from our quantile models ($p10 / p50 / p90$), permitting manual "what-if" input simulations (increasing backlog, altering the lag configurations) displaying immediate live shifts in projected load capacity.
- **Model Comparison ([pages/8_Model_Comparison.py](file:///c:/Tesina_Avances/pages/8_Model_Comparison.py)):** Standardizes Prophet vs XGBoost validation via classification mechanisms — mapping absolute errors under a permissible configured threshold into binary 'precision' translating the regressors into operational AUC / ROC classification curves tracking expected success rate.
