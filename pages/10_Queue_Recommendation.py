"""
Queue Recommendation (S/M/L/XL) using clustering + KNN assignment.
"""
import numpy as np
import pandas as pd
import polars as pl
import streamlit as st
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from db.database import load_reports_data
from utils.data_processing import add_time_features, quality_sanitize, filter_by_date_range
from components.filters import (
    render_sidebar_filters,
    apply_filters,
    render_presentation_mode_toggle,
    apply_presentation_mode,
)


st.set_page_config(page_title="Queue Recommendation", layout="wide")
st.title("Queue Recommendation (Quantile Bands + KNN)")
st.caption("Recommend operational queue size labels: S, M, L, XL.")


def build_report_features(df: pl.DataFrame) -> pl.DataFrame:
    keys = ["ReportId", "ReportName", "ReportType", "ReportQueueId"]

    runs = (
        df.group_by(keys)
        .agg([
            pl.len().alias("total_runs"),
            (pl.col("execution_status") == "success").sum().alias("success_runs"),
            (pl.col("execution_status") == "failed").sum().alias("failed_runs"),
            (pl.col("execution_status") == "cancelled").sum().alias("cancelled_runs"),
        ])
    )

    durations = (
        df.filter(pl.col("execution_status") == "success")
        .group_by(keys)
        .agg([
            pl.col("duration_seconds").mean().alias("avg_seconds"),
            pl.col("duration_seconds").median().alias("median_seconds"),
            pl.col("duration_seconds").quantile(0.95).alias("p95_seconds"),
        ])
    )

    out = (
        runs.join(durations, on=keys, how="left")
        .with_columns([
            (pl.col("success_runs") / pl.col("total_runs")).fill_null(0.0).alias("success_rate"),
            (pl.col("failed_runs") / pl.col("total_runs")).fill_null(0.0).alias("failure_rate"),
            pl.col("avg_seconds").fill_null(0.0),
            pl.col("median_seconds").fill_null(0.0),
            pl.col("p95_seconds").fill_null(0.0),
        ])
    )
    return out


# Load and process data from session state
df_raw = st.session_state.get("df_raw")
df = st.session_state.get("df_all")

# Fallback if accessed directly
if df_raw is None or df is None:
    with st.spinner("Initializing data..."):
        df_raw = load_reports_data()
        df = add_time_features(df_raw)
        df = quality_sanitize(df)
        df = filter_by_date_range(df, "2025-01-01", "2025-12-31")
        
        st.session_state["df_raw"] = df_raw
        st.session_state["df_all"] = df

presentation_mode = render_presentation_mode_toggle()
df = apply_presentation_mode(df, presentation_mode)

filters = render_sidebar_filters(df)
df_filtered = apply_filters(df, filters)

if df_filtered.height == 0:
    st.warning("No data available with current filters.")
    st.stop()

report_df = build_report_features(df_filtered)

st.subheader("Model Settings")
c1, c2, c3 = st.columns(3)
with c1:
    min_runs = st.slider("Minimum executions per report", 5, 500, 20, 5)
with c2:
    n_neighbors = st.slider("KNN neighbors", 3, 25, 7, 2)
with c3:
    random_state = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)

report_df = report_df.filter(pl.col("total_runs") >= min_runs)
if report_df.height < 8:
    st.warning("Not enough report-level data after filters. Lower minimum executions or broaden filters.")
    st.stop()

features_pd = report_df.select([
    "ReportId",
    "ReportName",
    "ReportType",
    "ReportQueueId",
    "total_runs",
    "avg_seconds",
    "median_seconds",
    "p95_seconds",
    "failure_rate",
    "success_rate",
]).to_pandas()

X = features_pd[["avg_seconds", "median_seconds", "p95_seconds", "total_runs", "failure_rate"]].copy()
X["total_runs"] = np.log1p(X["total_runs"].astype(float))

# 1) Define queue labels from p95 duration quantile bands
q25, q50, q75 = features_pd["p95_seconds"].quantile([0.25, 0.5, 0.75]).tolist()

def queue_from_p95(value: float) -> str:
    if value <= q25:
        return "S"
    if value <= q50:
        return "M"
    if value <= q75:
        return "L"
    return "XL"

features_pd["recommended_queue"] = features_pd["p95_seconds"].apply(queue_from_p95)

# 2) Train KNN assigner based on quantile-derived queue labels
y = features_pd["recommended_queue"].astype(str)
test_size = 0.25
min_class_count = y.value_counts().min()
can_stratify = (y.nunique() > 1) and (min_class_count >= 2)

if can_stratify:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=int(random_state),
        stratify=y,
    )
else:
    st.warning(
        "Some queue classes have fewer than 2 samples; using non-stratified split for KNN training."
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=int(random_state),
        stratify=None,
    )

knn_k = min(int(n_neighbors), len(X_train))
knn_model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=knn_k)),
])
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)
knn_accuracy = accuracy_score(y_test, y_pred)

features_pd["knn_assigned_queue"] = knn_model.predict(X)

st.subheader("Results")
m1, m2, m3 = st.columns(3)
m1.metric("Reports evaluated", f"{len(features_pd)}")
m2.metric("KNN holdout accuracy", f"{knn_accuracy:.2%}")
m3.metric("Queue bands", "4 (S/M/L/XL)")

st.caption("`recommended_queue` is based on p95 quantile bands; `knn_assigned_queue` is the operational assigner.")
st.caption(f"Band thresholds (p95 seconds): S <= {q25:.1f}, M <= {q50:.1f}, L <= {q75:.1f}, XL > {q75:.1f}")

display_cols = [
    "ReportId",
    "ReportName",
    "current_queue",
    "total_runs",
    "avg_seconds",
    "median_seconds",
    "p95_seconds",
    "failure_rate",
    "recommended_queue",
    "knn_assigned_queue",
]

table_df = (
    features_pd.assign(current_queue=features_pd["ReportQueueId"])[display_cols]
    .sort_values(["knn_assigned_queue", "p95_seconds"], ascending=[True, False])
    .reset_index(drop=True)
)

st.dataframe(table_df, use_container_width=True)

st.subheader("Queue Profile")
queue_profile = (
    table_df.groupby("recommended_queue", as_index=False)[["avg_seconds", "median_seconds", "p95_seconds", "failure_rate"]]
    .mean()
    .sort_values("recommended_queue")
)
st.dataframe(queue_profile, use_container_width=True)

csv_bytes = table_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download queue recommendations (CSV)",
    data=csv_bytes,
    file_name="queue_recommendations.csv",
    mime="text/csv",
)
