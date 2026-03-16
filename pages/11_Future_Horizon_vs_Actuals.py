import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

st.set_page_config(page_title="Future Horizon vs Actuals", layout="wide")

st.title("Future Horizon vs Actual Results")
st.markdown("""
    *Comparison of the 13-day recursive projection against actual reported values. This allows evaluating the ongoing accuracy of the forecasting engine.*
""")

DATA_FILE = "forecast_15d_vs_actuals.csv"

if not os.path.exists(DATA_FILE):
    st.warning(f"⚠️ Data file `{DATA_FILE}` not found. Please ensure that the model has been trained and actuals are available in the database to generate this comparison.")
    st.stop()

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_FILE)
        df['ds'] = pd.to_datetime(df['ds'])
        return df.sort_values('ds')
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data()

if df is None or df.empty:
    st.warning("No data available for comparison.")
    st.stop()

# Get the first date (start of horizon)
horizon_start = df['ds'].min()
horizon_end = horizon_start + pd.Timedelta(days=15)
actual_end = df['ds'].max()

if actual_end < horizon_end:
    st.warning(f"⚠️ El dataset llega hasta {actual_end.strftime('%Y-%m-%d %H:%M')}, lo cual es menor a los 15 días esperados desde {horizon_start.strftime('%Y-%m-%d %H:%M')}. Mostrando la información disponible.")

df = df[(df['ds'] >= horizon_start) & (df['ds'] <= horizon_end)]

if df.empty:
    st.warning("No data available in the required 15-day range.")
    st.stop()

# --- METRICS ---
st.subheader("Performance Metrics")

y_true = df["y_actual"].values
y_pred = df["y_pred"].values

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
bias = np.mean(y_pred - y_true)

# Pearson correlation safely
if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
    corr, _ = pearsonr(y_true, y_pred)
else:
    corr = np.nan

# High-volume bias (Actual > 50)
hv_mask = y_true > 50
hv_bias = np.mean(y_pred[hv_mask] - y_true[hv_mask]) if np.sum(hv_mask) > 0 else 0.0

# Quantile Coverage
coverage = np.mean((y_true >= df["y_p10"]) & (y_true <= df["y_p90"])) * 100

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("MAE", f"{mae:.2f}", help="Mean Absolute Error (lower is better)")
with c2:
    st.metric("RMSE", f"{rmse:.2f}", help="Root Mean Squared Error (lower is better)")
with c3:
    st.metric("Overall Bias", f"{bias:+.2f}", help="Average error (positive means over-prediction)")
with c4:
    st.metric("Correlation", f"{corr:.2f}" if not np.isnan(corr) else "N/A", help="Pearson Correlation Coefficient (closer to 1.0 is better)")

c5, c6, c7, c8 = st.columns(4)
with c5:
    st.metric("High-Volume Bias", f"{hv_bias:+.2f}", help="Bias for hours where actual volume > 50")
with c6:
    st.metric("Quantile Coverage", f"{coverage:.1f}%", help="Percentage of actuals falling within the P10-P90 interval (target ~80%)")
with c7:
    st.metric("Total Hours Evaluated", f"{len(df)}")
with c8:
    pass

st.divider()

# --- VISUALIZATION ---
st.subheader("Forecast vs Actuals Trend")

fig = go.Figure()

# Actuals
fig.add_trace(go.Scatter(
    x=df['ds'], y=df['y_actual'],
    name="Actual Volume",
    mode="lines+markers",
    line=dict(color='#ff00ff', width=2),
    marker=dict(size=4),
    hovertemplate="<b>Date: %{x}</b><br>Actual: %{y:.1f}<extra></extra>"
))

# Uncertainty Interval (P10 - P90)
fig.add_trace(go.Scatter(
    x=pd.concat([df['ds'], df['ds'][::-1]]),
    y=pd.concat([df['y_p90'], df['y_p10'][::-1]]),
    fill='toself',
    fillcolor='rgba(0, 242, 254, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    name="Uncertainty Range (p10-p90)",
    hoverinfo='skip'
))

# Predictions (P50)
fig.add_trace(go.Scatter(
    x=df['ds'], y=df['y_pred'],
    name="Predicted Volume (p50)",
    mode="lines",
    line=dict(color='#00f2fe', width=3, dash='dash'),
    hovertemplate="<b>Date: %{x}</b><br>Predicted: %{y:.1f}<extra></extra>"
))

fig.update_layout(
    title=f"Prediction vs Actuals ({df['ds'].dt.date.min()} to {df['ds'].dt.date.max()})",
    template="plotly_dark",
    height=550,
    hovermode="x unified",
    legend=dict(orientation="h", y=1.1, x=0),
    xaxis_title="Time",
    yaxis_title="Hourly Reports",
    margin=dict(r=0, l=0)
)
st.plotly_chart(fig, use_container_width=True)

# --- DATA TABLE ---
st.subheader("Raw Data")
with st.expander("View Data Table"):
    st.dataframe(
        df.sort_values('ds', ascending=False).style.format({
            'y_p10': '{:.2f}',
            'y_pred': '{:.2f}',
            'y_p90': '{:.2f}',
            'y_actual': '{:.1f}'
        }),
        use_container_width=True
    )
