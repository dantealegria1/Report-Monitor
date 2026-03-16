"""
Queue Recommendation — NR Focus
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
import streamlit as st

from db.database import load_reports_data
from utils.data_processing import add_time_features, quality_sanitize, filter_by_date_range
from components.filters import (
    render_sidebar_filters, apply_filters,
    render_presentation_mode_toggle, apply_presentation_mode,
)
from queue_clustering import (
    run_clustering, simulate_daily_throughput, simulate_backlog, NR_QUEUE_CONFIG,
)

st.set_page_config(page_title="Queue Recommendation · NR", layout="wide")
st.title("Queue Recommendation — NR Reports")
st.caption(
    "Clusters NR reports into S/M/L/XL using KMeans on log-duration. "
    "Compares current queue assignment vs KNN recommendation and simulates daily throughput."
)


def fmt_s(s) -> str:
    try:
        s = float(s)
    except Exception:
        return "—"
    if np.isnan(s) or s < 0:
        return "—"
    s = int(s)
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {(s % 3600) // 60:02d}m"


# ── Data ───────────────────────────────────────────────────────────────────────
df_raw = st.session_state.get("df_raw")
df     = st.session_state.get("df_all")
if df_raw is None or df is None:
    with st.spinner("Loading data…"):
        df_raw = load_reports_data()
        df = add_time_features(df_raw)
        df = quality_sanitize(df)
        df = filter_by_date_range(df, "2025-01-01", "2030-12-31")
        st.session_state["df_raw"] = df_raw
        st.session_state["df_all"] = df

presentation_mode = render_presentation_mode_toggle()
df = apply_presentation_mode(df, presentation_mode)
filters = render_sidebar_filters(df)
df_filtered = apply_filters(df, filters)

if df_filtered.height == 0:
    st.warning("No data available.")
    st.stop()

df_nr = df_filtered.filter(pl.col("ReportType") == "NR")
if df_nr.height == 0:
    st.warning("No NR reports found.")
    st.stop()

st.info(f"**{df_nr.height:,}** NR execution rows loaded.")

# ── Queue Config ───────────────────────────────────────────────────────────────
with st.expander("📋 Allowed NR Queue Configuration", expanded=False):
    st.caption("The `Id` column is what the DB stores in `r.QueueNumber` (the FK).")
    st.dataframe(NR_QUEUE_CONFIG, use_container_width=True, hide_index=True)

# ── Settings ───────────────────────────────────────────────────────────────────
st.subheader("⚙️ Model Settings")
c1, c2, c3 = st.columns(3)
with c1: min_runs     = st.slider("Min runs per report", 5, 500, 20, 5)
with c2: n_neighbors  = st.slider("KNN neighbors", 3, 25, 7, 2)
with c3: random_state = st.number_input("Random seed", 0, 9999, 42, step=1)

# ── Clustering ─────────────────────────────────────────────────────────────────
with st.spinner("Running KMeans + KNN…"):
    res = run_clustering(df_nr, min_runs=min_runs,
                         n_neighbors=n_neighbors, random_state=int(random_state))

if res["error"]:
    st.warning(res["error"])
    st.stop()

fp: pd.DataFrame = res["features_pd"]
knn_acc: float   = res["knn_accuracy"]
thresholds: dict = res["band_thresholds"]

# ── KPIs ────────────────────────────────────────────────────────────────────────
st.subheader("📊 Clustering Results")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Reports evaluated", len(fp))
m2.metric("KNN accuracy", f"{knn_acc:.2%}")
m3.metric("Size bands", "4")
n_mismatch = int((~fp["size_match"]).sum()) if "size_match" in fp.columns else 0
m4.metric("Size mismatches (current ≠ KNN)", n_mismatch)

thresh_txt = "  ·  ".join(
    f"**{sz}** median p95={fmt_s(thresholds.get(sz, 0))}" for sz in ["S", "M", "L", "XL"]
)
st.caption(f"Cluster thresholds: {thresh_txt}")

# ── p95 distribution ────────────────────────────────────────────────────────────
colors = {"S": "#4caf50", "M": "#2196f3", "L": "#ff9800", "XL": "#f44336"}
fig_dist = go.Figure()
for sz in ["S", "M", "L", "XL"]:
    sub = fp[fp["knn_size"] == sz]["p95_seconds"]
    if len(sub):
        fig_dist.add_trace(go.Box(y=sub, name=sz, marker_color=colors.get(sz, "#888"),
                                   boxpoints="outliers"))
fig_dist.update_layout(yaxis_title="p95 seconds", showlegend=False,
                        height=240, margin=dict(l=0, r=0, t=10, b=0), template="plotly_white")
st.plotly_chart(fig_dist, use_container_width=True)

# ── Comparison table ───────────────────────────────────────────────────────────
st.subheader("🔀 Current Assignment vs. KNN Recommendation")
st.caption(
    "`current_size` = size from DB config Id lookup · "
    "`knn_size` = KMeans + KNN recommendation · "
    "`current_instances` = concurrent slots in current config"
)

tbl = fp.copy()
for raw, pretty in [("avg_seconds","Avg"), ("median_seconds","Median"), ("p95_seconds","p95")]:
    if raw in tbl.columns:
        tbl[pretty] = tbl[raw].apply(fmt_s)

tbl["Size match?"] = tbl["size_match"].map({True: "✅", False: "⚠️"}) if "size_match" in tbl.columns else "—"

disp = [c for c in [
    "ReportId", "ReportName",
    "current_size", "current_queue_number", "current_instances",
    "knn_size", "knn_queue_number", "knn_instances",
    "Size match?",
    "total_runs", "Avg", "Median", "p95", "failure_rate",
] if c in tbl.columns]

st.dataframe(
    tbl.sort_values("p95_seconds" if "p95_seconds" in tbl.columns else "knn_size")
       .reset_index(drop=True)[disp],
    use_container_width=True, height=420,
)

# ── Daily Throughput Simulation ────────────────────────────────────────────────
st.subheader("⏱️ Daily Throughput — Current vs. Recommended Arrangement")
st.caption(
    "**Wall-clock hours per day** = sum of all reports' `duration_seconds / instances` for that day.  \n"
    "With more concurrent instances, reports overlap → less wall-clock time to drain the same workload.  \n"
    "Lower is better."
)

with st.spinner("Simulating daily throughput…"):
    daily = simulate_daily_throughput(df_nr, fp)

if daily.empty:
    st.warning("No data for daily throughput simulation.")
else:
    fig_tp = go.Figure()
    fig_tp.add_trace(go.Scatter(
        x=daily["date"], y=daily["current_wall_hours"],
        name="Current arrangement", mode="lines",
        line=dict(color="#ef6c00", width=2),
        fill="tozeroy", fillcolor="rgba(239,108,0,0.10)",
    ))
    fig_tp.add_trace(go.Scatter(
        x=daily["date"], y=daily["rec_wall_hours"],
        name="Recommended (KNN)", mode="lines",
        line=dict(color="#1565c0", width=2, dash="dash"),
        fill="tozeroy", fillcolor="rgba(21,101,192,0.07)",
    ))
    fig_tp.add_trace(go.Bar(
        x=daily["date"], y=daily["total_work_hours"],
        name="Total work (raw hours)", marker_color="rgba(150,150,150,0.25)",
        yaxis="y2",
    ))
    fig_tp.update_layout(
        xaxis_title="Date",
        yaxis=dict(title="Wall-clock hours to drain"),
        yaxis2=dict(title="Total work-hours", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", y=1.05, x=0),
        height=380, hovermode="x unified", template="plotly_white",
        margin=dict(l=0, r=60, t=40, b=0),
    )
    st.plotly_chart(fig_tp, use_container_width=True)

    # Summary
    tot_curr = daily["current_wall_hours"].sum()
    tot_rec  = daily["rec_wall_hours"].sum()
    tot_work = daily["total_work_hours"].sum()
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Total work (all reports)", f"{tot_work:,.0f} h")
    sc2.metric("Wall-clock current",  f"{tot_curr:,.0f} h")
    sc3.metric("Wall-clock recommended", f"{tot_rec:,.0f} h",
               delta=f"{tot_rec - tot_curr:+,.0f} h", delta_color="inverse")
    sc4.metric("Time saved (estimated)", f"{max(0, tot_curr - tot_rec):,.1f} h")

# ── Hourly Backlog ─────────────────────────────────────────────────────────────
st.subheader("📈 Hourly Backlog (reports created but not yet started)")
st.caption(
    "Arrivals = `creation_date` events · Departures = `started_at` events.  \n"
    "The recommended curve scales departures by `avg(knn_instances) / avg(current_instances)`."
)

with st.spinner("Simulating hourly backlog…"):
    sim_curr = simulate_backlog(df_nr, fp, scenario="current")
    sim_rec  = simulate_backlog(df_nr, fp, scenario="recommended")

if not sim_curr.empty:
    fig_bl = go.Figure()
    fig_bl.add_trace(go.Scatter(
        x=sim_curr["hour"], y=sim_curr["backlog"],
        name="Current", mode="lines",
        line=dict(color="#ef6c00", width=1.5),
        fill="tozeroy", fillcolor="rgba(239,108,0,0.10)",
    ))
    fig_bl.add_trace(go.Scatter(
        x=sim_rec["hour"], y=sim_rec["backlog"],
        name="Recommended (KNN)", mode="lines",
        line=dict(color="#1565c0", width=1.5, dash="dash"),
    ))
    fig_bl.update_layout(
        xaxis_title="Hour", yaxis_title="Reports in queue (waiting to start)",
        legend=dict(orientation="h", y=1.04, x=0),
        height=320, hovermode="x unified",
        template="plotly_white", margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig_bl, use_container_width=True)

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Peak backlog (current)",     int(sim_curr["backlog"].max()))
    p2.metric("Peak backlog (recommended)", int(sim_rec["backlog"].max()),
              delta=f"{int(sim_rec['backlog'].max()) - int(sim_curr['backlog'].max()):+d}",
              delta_color="inverse")
    p3.metric("Avg backlog (current)",      f"{sim_curr['backlog'].mean():.1f}")
    p4.metric("Avg backlog (recommended)",  f"{sim_rec['backlog'].mean():.1f}",
              delta=f"{sim_rec['backlog'].mean() - sim_curr['backlog'].mean():+.1f}",
              delta_color="inverse")

# ── Profile ────────────────────────────────────────────────────────────────────
st.subheader("📋 Profile by KNN Size Band")
pcols = [c for c in ["knn_size", "avg_seconds", "median_seconds", "p95_seconds",
                      "knn_instances", "failure_rate", "total_runs"] if c in fp.columns]
if pcols:
    prof = (fp[pcols].groupby("knn_size")
            .agg({c: "mean" for c in pcols if c != "knn_size" and c != "total_runs"}
                 | ({"total_runs": "sum"} if "total_runs" in pcols else {}))
            .reset_index().rename(columns={"knn_size": "Size"})
            .sort_values("Size"))
    for c in ["avg_seconds", "median_seconds", "p95_seconds"]:
        if c in prof.columns:
            prof[c] = prof[c].apply(fmt_s)
    if "knn_instances" in prof.columns:
        prof["knn_instances"] = prof["knn_instances"].round(1)
    st.dataframe(prof, use_container_width=True, hide_index=True)

# ── CSV ────────────────────────────────────────────────────────────────────────
dl = [c for c in ["ReportId","ReportName","current_size","current_queue_number",
                   "current_instances","knn_size","knn_queue_number","knn_instances",
                   "size_match","total_runs","avg_seconds","p95_seconds","failure_rate"]
      if c in fp.columns]
st.download_button("⬇️ Download CSV", fp[dl].to_csv(index=False).encode(),
                   "nr_queue_recommendations.csv", "text/csv")
