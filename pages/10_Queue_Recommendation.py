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
    run_clustering, simulate_daily_throughput, simulate_backlog,
    simulate_contention, NR_QUEUE_CONFIG,
)

st.set_page_config(page_title="Queue Recommendation · NR", layout="wide")
st.title("Queue Recommendation — NR Reports")
st.caption(
    "Clusters NR reports into size bands using KMeans (optimal k via silhouette) "
    "on enriched features. Compares current queue assignment vs KNN recommendation, "
    "simulates throughput, contention, and cost."
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

fp: pd.DataFrame     = res["features_pd"]
knn_acc: float       = res["knn_accuracy"]
thresholds: dict     = res["band_thresholds"]
optimal_k: int       = res["optimal_k"]
sil_scores: dict     = res["silhouette_scores"]
cost_curr: float     = res["cost_current"]
cost_rec: float      = res["cost_recommended"]

# ── KPIs ────────────────────────────────────────────────────────────────────────
st.subheader("📊 Clustering Results")
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Reports evaluated", len(fp))
m2.metric("KNN accuracy", f"{knn_acc:.2%}")
m3.metric("Optimal k", optimal_k)
n_mismatch = int((~fp["size_match"]).sum()) if "size_match" in fp.columns else 0
m4.metric("Size mismatches", n_mismatch)
avg_dist = fp["mismatch_distance"].mean() if "mismatch_distance" in fp.columns else 0
m5.metric("Avg mismatch dist", f"{avg_dist:.2f}")
if cost_curr and cost_curr > 0:
    pct = (cost_curr - cost_rec) / cost_curr * 100
    m6.metric("Cost savings", f"{pct:+.1f}%")
else:
    m6.metric("Cost savings", "—")

# Cluster thresholds
active_sizes = sorted(fp["knn_size"].unique(), key=lambda s: {"XS":0,"S":1,"M":2,"L":3,"XL":4,"XXL":5}.get(s, 9))
thresh_txt = "  ·  ".join(
    f"**{sz}** median p95={fmt_s(thresholds.get(sz, 0))}" for sz in active_sizes
)
st.caption(f"Cluster thresholds: {thresh_txt}")

# ── Silhouette chart ───────────────────────────────────────────────────────────
if sil_scores:
    with st.expander("🔍 Silhouette Score per k", expanded=False):
        ks = sorted(sil_scores.keys())
        scores = [sil_scores[k] for k in ks]
        colors_sil = ["#1565c0" if k == optimal_k else "#90caf9" for k in ks]
        fig_sil = go.Figure(go.Bar(
            x=[str(k) for k in ks], y=scores,
            marker_color=colors_sil,
            text=[f"{s:.3f}" for s in scores], textposition="outside",
        ))
        fig_sil.update_layout(
            xaxis_title="Number of clusters (k)",
            yaxis_title="Silhouette score",
            height=260, margin=dict(l=0, r=0, t=10, b=0),
            template="plotly_white",
        )
        st.plotly_chart(fig_sil, use_container_width=True)
        st.caption(f"Best k = **{optimal_k}** (silhouette = {sil_scores.get(optimal_k, 0):.4f})")

# ── p95 distribution ────────────────────────────────────────────────────────────
colors = {"XS": "#9c27b0", "S": "#4caf50", "M": "#2196f3", "L": "#ff9800", "XL": "#f44336", "XXL": "#795548"}
fig_dist = go.Figure()
for sz in active_sizes:
    sub = fp[fp["knn_size"] == sz]["p95_seconds"]
    if len(sub):
        fig_dist.add_trace(go.Box(y=sub, name=sz, marker_color=colors.get(sz, "#888"),
                                   boxpoints="outliers"))
fig_dist.update_layout(yaxis_title="p95 seconds", showlegend=False,
                        height=240, margin=dict(l=0, r=0, t=10, b=0), template="plotly_white")
st.plotly_chart(fig_dist, use_container_width=True)

# ── Mismatch distance distribution ────────────────────────────────────────────
if "mismatch_distance" in fp.columns:
    with st.expander("📏 Mismatch Distance Distribution", expanded=False):
        dist_counts = fp["mismatch_distance"].value_counts().sort_index()
        dist_colors = {0: "#4caf50", 1: "#ffeb3b", 2: "#ff9800", 3: "#f44336"}
        fig_md = go.Figure(go.Bar(
            x=[str(int(d)) for d in dist_counts.index],
            y=dist_counts.values,
            marker_color=[dist_colors.get(int(d), "#888") for d in dist_counts.index],
            text=dist_counts.values, textposition="outside",
        ))
        fig_md.update_layout(
            xaxis_title="Mismatch distance (0 = exact match, 3 = worst)",
            yaxis_title="Number of reports",
            height=240, margin=dict(l=0, r=0, t=10, b=0), template="plotly_white",
        )
        st.plotly_chart(fig_md, use_container_width=True)

# ── Peak-Hour Analysis ────────────────────────────────────────────────────────
if "peak_hour_ratio" in fp.columns:
    with st.expander("🕐 Peak-Hour Analysis (09:00–17:00)", expanded=False):
        fig_pk = go.Figure()
        for sz in active_sizes:
            sub = fp[fp["knn_size"] == sz]["peak_hour_ratio"]
            if len(sub):
                fig_pk.add_trace(go.Box(y=sub, name=sz, marker_color=colors.get(sz, "#888"),
                                         boxpoints="outliers"))
        fig_pk.update_layout(
            yaxis_title="% of runs during peak hours",
            showlegend=False, height=240,
            margin=dict(l=0, r=0, t=10, b=0), template="plotly_white",
        )
        st.plotly_chart(fig_pk, use_container_width=True)
        st.caption(
            "Reports with a high peak-hour ratio compete more for queue slots during busy hours."
        )

# ── Comparison table ───────────────────────────────────────────────────────────
st.subheader("🔀 Current Assignment vs. KNN Recommendation")
st.caption(
    "`current_size` = size from DB config Id lookup · "
    "`knn_size` = KMeans + KNN recommendation · "
    "`mismatch_distance` = ordinal distance (0=match, 3=worst)"
)

tbl = fp.copy()
for raw, pretty in [("avg_seconds","Avg"), ("median_seconds","Median"), ("p95_seconds","p95")]:
    if raw in tbl.columns:
        tbl[pretty] = tbl[raw].apply(fmt_s)

# Colored severity  (improvement #7)
severity_map = {0: "✅", 1: "🟡", 2: "🟠", 3: "🔴"}
if "mismatch_distance" in tbl.columns:
    tbl["Severity"] = tbl["mismatch_distance"].map(
        lambda d: severity_map.get(int(d), "❓") if pd.notna(d) else "—"
    )
else:
    tbl["Severity"] = "—"

disp = [c for c in [
    "ReportId", "ReportName",
    "current_size", "current_queue_number", "current_instances",
    "knn_size", "knn_queue_number", "knn_instances",
    "Severity", "mismatch_distance",
    "total_runs", "Avg", "Median", "p95", "failure_rate",
    "peak_hour_ratio",
] if c in tbl.columns]

st.dataframe(
    tbl.sort_values("p95_seconds" if "p95_seconds" in tbl.columns else "knn_size")
       .reset_index(drop=True)[disp],
    use_container_width=True, height=420,
)

# ── Efficiency Comparison ─────────────────────────────────────────────────────
st.subheader("💰 Efficiency Comparison — Current vs. Recommended")
st.caption(
    "Compares resource efficiency (p95 / instances) and assignment quality. "
    "All values are percentages — higher improvement % is better."
)
if cost_curr and cost_curr > 0 and cost_rec is not None:
    efficiency_pct = (1 - cost_rec / cost_curr) * 100  # % improvement
    mismatch_pct   = fp["mismatch_distance"].gt(0).mean() * 100 if "mismatch_distance" in fp.columns else 0
    perfect_pct    = fp["size_match"].mean() * 100 if "size_match" in fp.columns else 0
    avg_dist       = fp["mismatch_distance"].mean() if "mismatch_distance" in fp.columns else 0

    cc1, cc2, cc3, cc4 = st.columns(4)
    cc1.metric("Efficiency improvement", f"{efficiency_pct:+.1f}%",
               help="How much more efficient the recommended arrangement is vs. current")
    cc2.metric("Correct assignments", f"{perfect_pct:.0f}%",
               help="% of reports where current queue matches the recommendation")
    cc3.metric("Mismatched reports", f"{mismatch_pct:.0f}%",
               delta=f"avg distance {avg_dist:.1f}", delta_color="inverse",
               help="% of reports whose current size ≠ recommended size")
    cc4.metric("Resource utilisation gain", 
               f"{(cost_curr - cost_rec) / cost_curr * 100:+.1f}%",
               help="% reduction in total p95/instances cost")
else:
    st.info("Not enough data to compute efficiency comparison.")

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

# ── Queue Contention ──────────────────────────────────────────────────────────
st.subheader("🔧 Queue Contention — Concurrent Slot Utilisation")
st.caption(
    "Hourly count of concurrent instance slots in use.  \n"
    "Higher recommended values mean extra parallelism; the system can drain work faster."
)

with st.spinner("Simulating queue contention…"):
    contention = simulate_contention(df_nr, fp)

if not contention.empty:
    fig_ct = go.Figure()
    fig_ct.add_trace(go.Scatter(
        x=contention["hour"], y=contention["concurrent_current"],
        name="Current slots in use", mode="lines",
        line=dict(color="#ef6c00", width=1.5),
        fill="tozeroy", fillcolor="rgba(239,108,0,0.08)",
    ))
    fig_ct.add_trace(go.Scatter(
        x=contention["hour"], y=contention["concurrent_rec"],
        name="Recommended slots in use", mode="lines",
        line=dict(color="#1565c0", width=1.5, dash="dash"),
    ))
    fig_ct.update_layout(
        xaxis_title="Hour", yaxis_title="Concurrent slots",
        legend=dict(orientation="h", y=1.04, x=0),
        height=320, hovermode="x unified",
        template="plotly_white", margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig_ct, use_container_width=True)

    ct1, ct2, ct3 = st.columns(3)
    ct1.metric("Peak slots (current)", int(contention["concurrent_current"].max()))
    ct2.metric("Peak slots (recommended)", int(contention["concurrent_rec"].max()),
               delta=f"{int(contention['concurrent_rec'].max()) - int(contention['concurrent_current'].max()):+d}")
    ct3.metric("Avg active reports/hour", f"{contention['active_reports'].mean():.1f}")
else:
    st.info("No contention data available (need `started_at` and `finished_at` columns).")

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
                      "knn_instances", "failure_rate", "total_runs",
                      "peak_hour_ratio", "cv"] if c in fp.columns]
if pcols:
    agg_dict = {c: "mean" for c in pcols if c not in ("knn_size", "total_runs")}
    if "total_runs" in pcols:
        agg_dict["total_runs"] = "sum"
    prof = (fp[pcols].groupby("knn_size")
            .agg(agg_dict)
            .reset_index().rename(columns={"knn_size": "Size"})
            .sort_values("Size"))
    for c in ["avg_seconds", "median_seconds", "p95_seconds"]:
        if c in prof.columns:
            prof[c] = prof[c].apply(fmt_s)
    if "knn_instances" in prof.columns:
        prof["knn_instances"] = prof["knn_instances"].round(1)
    if "peak_hour_ratio" in prof.columns:
        prof["peak_hour_ratio"] = prof["peak_hour_ratio"].map(lambda v: f"{v:.1%}")
    if "cv" in prof.columns:
        prof["cv"] = prof["cv"].round(2)
    st.dataframe(prof, use_container_width=True, hide_index=True)

# ── CSV ────────────────────────────────────────────────────────────────────────
dl = [c for c in ["ReportId","ReportName","current_size","current_queue_number",
                   "current_instances","knn_size","knn_queue_number","knn_instances",
                   "size_match","mismatch_distance","total_runs","avg_seconds","p95_seconds",
                   "failure_rate","peak_hour_ratio","cv","cost_current","cost_recommended"]
      if c in fp.columns]
st.download_button("⬇️ Download CSV", fp[dl].to_csv(index=False).encode(),
                   "nr_queue_recommendations.csv", "text/csv")
