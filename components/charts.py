"""
Chart components.
Handles all visualization and chart rendering.
"""
import streamlit as st
import polars as pl


def render_execution_trends(df_filtered: pl.DataFrame):
    """
    Render execution trends chart by status with per-report hover breakdown.

    Args:
        df_filtered: Filtered DataFrame
    """
    import plotly.graph_objects as go
    import pandas as pd

    st.subheader("Execution Trends (by status)")

    # Optional: filter by specific reports
    all_reports = sorted(df_filtered["ReportName"].unique().to_list())
    selected_reports = st.multiselect(
        "Filter by Report (leave empty to show all)",
        options=all_reports,
        default=[],
        key="exec_trends_report_filter",
        placeholder="All reports",
    )

    df_trend = df_filtered
    if selected_reports:
        df_trend = df_filtered.filter(pl.col("ReportName").is_in(selected_reports))

    if df_trend.height == 0:
        st.info("No data for the selected reports.")
        return

    # Aggregate total per (date, status) for the bar heights
    evol = (
        df_trend
        .group_by(["date", "execution_status"])
        .agg(pl.len().alias("count"))
        .sort("date")
        .to_pandas()
    )

    # Aggregate per (date, status, ReportName) for the tooltip breakdown
    detail = (
        df_trend
        .group_by(["date", "execution_status", "ReportName"])
        .agg(pl.len().alias("n"))
        .sort(["date", "execution_status", "n"], descending=[False, False, True])
        .to_pandas()
    )

    # Build a lookup: (date, status) -> formatted report breakdown string
    def build_breakdown(grp: pd.DataFrame, top_n: int = 10) -> str:
        rows = grp.head(top_n)
        lines = [f"  {r['ReportName']}: {int(r['n'])}" for _, r in rows.iterrows()]
        if len(grp) > top_n:
            lines.append(f"  … and {len(grp) - top_n} more")
        return "<br>".join(lines)

    breakdown_map: dict = {}
    for (date_val, status), grp in detail.groupby(["date", "execution_status"]):
        breakdown_map[(date_val, status)] = build_breakdown(grp)

    # Color map for statuses
    status_colors = {
        "success": "#2ecc71",
        "failed":  "#e74c3c",
        "error":   "#e67e22",
        "running": "#3498db",
    }

    all_dates = sorted(evol["date"].unique())
    statuses = sorted(evol["execution_status"].unique())

    fig = go.Figure()

    for status in statuses:
        sub = evol[evol["execution_status"] == status].set_index("date")
        counts = [int(sub.loc[d, "count"]) if d in sub.index else 0 for d in all_dates]
        breakdowns = [breakdown_map.get((d, status), "—") for d in all_dates]

        fig.add_trace(go.Bar(
            name=status,
            x=all_dates,
            y=counts,
            marker_color=status_colors.get(status, "#95a5a6"),
            customdata=breakdowns,
            hovertemplate=(
                "<b>%{x|%Y-%m-%d}</b><br>"
                f"Status: <b>{status}</b><br>"
                "Total: <b>%{y}</b><br>"
                "<br><b>Reports:</b><br>%{customdata}"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        barmode="stack",
        xaxis_title="Date",
        yaxis_title="Executions",
        legend_title="Status",
        height=420,
        hovermode="x unified" if len(statuses) == 1 else "closest",
        margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cccccc"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
    )

    st.plotly_chart(fig, use_container_width=True)



def render_slowest_reports(df_filtered: pl.DataFrame):
    """
    Render top 10 slowest reports chart.
    
    Args:
        df_filtered: Filtered DataFrame
    """
    st.subheader("Top 10 Slowest Reports (success average)")
    df_success = df_filtered.filter(pl.col("execution_status") == "success")
    
    slow = (
        df_success.group_by("ReportName")
        .agg(pl.col("duration_seconds").mean().alias("average"))
        .sort("average", descending=True)
        .head(10)
        .to_pandas()
    )
    
    if not slow.empty:
        st.bar_chart(data=slow, x="ReportName", y="average")
    else:
        st.info("No successful data for the range.")


def render_top_failures(df_filtered: pl.DataFrame):
    """
    Render top 10 reports with most failures.
    
    Args:
        df_filtered: Filtered DataFrame
    """
    st.subheader("Top 10 Reports with Most Failures")
    df_failed = df_filtered.filter(pl.col("execution_status") == "failed")
    
    top_fail = (
        df_failed.group_by("ReportName")
        .agg(pl.len().alias("failures"))
        .sort("failures", descending=True)
        .head(10)
        .to_pandas()
    )
    
    if not top_fail.empty:
        st.bar_chart(data=top_fail, x="ReportName", y="failures")
    else:
        st.info("No failures in the range.")


def render_duration_distribution(df_filtered: pl.DataFrame, key: str = "slider_bin_size"):
    """
    Render duration distribution histogram.
    
    Args:
        df_filtered: Filtered DataFrame
        key: Unique key for the slider widget
    """
    st.subheader("Duration Distribution (success) - approximate bins")
    df_success = df_filtered.filter(pl.col("execution_status") == "success")
    
    if df_success.height > 0:
        bin_size = st.slider("Bin Size (sec)", 1, 120, 10, key=key)
        hist = (
            df_success
            .with_columns(((pl.col("duration_seconds") / bin_size).floor() * bin_size).alias("bin"))
            .group_by("bin")
            .agg(pl.len().alias("count"))
            .sort("bin")
            .to_pandas()
        )
        st.bar_chart(hist, x="bin", y="count")
    else:
        st.info("No successful data for histogram.")


def render_hourly_load(df_filtered: pl.DataFrame):
    """
    Render hourly load saturation chart.
    
    Args:
        df_filtered: Filtered DataFrame
    """
    st.subheader("Hourly Load (count)")
    hourly_load = (
        df_filtered.group_by("hour")
        .agg(pl.len().alias("count"))
        .sort("hour")
        .to_pandas()
    )
    
    if not hourly_load.empty:
        st.area_chart(data=hourly_load, x="hour", y="count")


def render_heatmap(df_filtered: pl.DataFrame):
    """
    Render hour vs weekday heatmap with interactive Plotly visualization.
    
    Args:
        df_filtered: Filtered DataFrame
    """
    import plotly.graph_objects as go
    
    st.subheader("Execution by Day and Hour")
    heat = (
        df_filtered.group_by(["weekday", "hour"])
        .agg(pl.len().alias("count"))
        .to_pandas()
    )
    
    if not heat.empty:
        # Pivot data for heatmap
        heat_piv = heat.pivot(index="weekday", columns="hour", values="count").fillna(0)
        
        # Map weekday numbers to day names
        day_labels = {
            0: 'Monday',
            1: 'Tuesday', 
            2: 'Wednesday',
            3: 'Thursday',
            4: 'Friday',
            5: 'Saturday',
            6: 'Sunday'
        }
        
        # Ensure all weekdays are present (0-6)
        for day in range(7):
            if day not in heat_piv.index:
                heat_piv.loc[day] = 0
        
        # Sort by weekday and apply labels
        heat_piv = heat_piv.sort_index()
        heat_piv.index = [day_labels.get(idx, f'Day {idx}') for idx in heat_piv.index]
        
        # Create Plotly heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heat_piv.values,
            x=[f'{int(h):02d}:00' for h in heat_piv.columns],
            y=heat_piv.index,
            colorscale='Viridis',
            hoverongaps=False,
            hovertemplate='<b>%{y}</b><br>Hour: %{x}<br>Executions: %{z}<extra></extra>',
            colorbar=dict(title="Count")
        ))
        
        fig.update_layout(
            title="Report Execution Heatmap",
            xaxis_title="Hour of Day",
            yaxis_title="Day of Week",
            height=400,
            xaxis=dict(side='bottom'),
            yaxis=dict(autorange='reversed')  # Monday at top
        )
        
        st.plotly_chart(fig, width='stretch')
        
        # Optional: Show data table in expander
        with st.expander("View Raw Data"):
            st.dataframe(heat_piv, width='stretch')
    else:
        st.info("No data available for heatmap.")


from components.filters import apply_filters

def render_double_run_detection(df: pl.DataFrame, filters: dict):
    """
    Detect 'double-run' events: same report triggered multiple times within
    a short time window, suggesting users are re-running the same report.

    Args:
        df: Base DataFrame (unfiltered by status, potentially)
        filters: Current sidebar filters
    """
    import plotly.graph_objects as go
    import copy

    st.subheader("Double-Run Detection")
    st.caption(
        "Identifies cases where the **same report** was triggered more than once "
        "within a short time window — a sign of impatient re-runs."
    )

    col_cfg1, col_cfg2, col_cfg3 = st.columns([2, 1, 1])
    with col_cfg1:
        threshold_sec = st.slider(
            "Time window (seconds)",
            min_value=10, max_value=300, value=60, step=10,
            key="double_run_threshold",
            help="Two executions of the same report within this window count as a double-run."
        )
    with col_cfg2:
        match_params = st.checkbox(
            "Also match Parameters",
            value=False,
            key="double_run_match_params",
            help="When checked, only flag pairs that share the same Parameters value."
        )
    with col_cfg3:
        include_cancelled = st.checkbox(
            "Include 'Cancelled'",
            value=False,
            key="double_run_include_cancelled",
            help="If checked, includes cancelled reports even if filtered out globally."
        )

    # Prepare local filters
    local_filters = copy.deepcopy(filters)
    
    if include_cancelled:
        # Ensure 'cancelled' execution_status is included
        current_statuses = local_filters.get("statuses", [])
        if "cancelled" not in current_statuses:
            current_statuses.append("cancelled")
        local_filters["statuses"] = current_statuses

    # Apply filters specifically for this component
    df_filtered = apply_filters(df, local_filters)

    if df_filtered.height == 0:
        st.info("No data available with current filters.")
        return

    # Group key: ReportName (+ Parameters if requested)
    group_cols = ["ReportName", "Parameters"] if match_params and "Parameters" in df_filtered.columns else ["ReportName"]

    # Sort and compute time diff to previous execution within the same group
    df_sorted = df_filtered.sort(group_cols + ["started_at"])

    df_with_prev = df_sorted.with_columns([
        pl.col("started_at").shift(1).over(group_cols).alias("prev_started_at"),
    ]).with_columns([
        (pl.col("started_at") - pl.col("prev_started_at"))
        .dt.total_seconds()
        .alias("gap_seconds")
    ])

    # Flag double-runs: same group, gap within threshold
    df_doubles = df_with_prev.filter(
        (pl.col("gap_seconds") >= 0) & (pl.col("gap_seconds") <= threshold_sec)
    )

    total_pairs = df_doubles.height

    if total_pairs == 0:
        st.success(f"No double-runs detected within a {threshold_sec}s window.")
        return

    st.warning(f"**{total_pairs}** double-run event(s) detected within a **{threshold_sec}s** window.")

    # Summary metrics
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Double-Run Events", total_pairs)
    with m2:
        unique_reports = df_doubles["ReportName"].n_unique()
        st.metric("Affected Reports", unique_reports)
    with m3:
        avg_gap = df_doubles["gap_seconds"].mean()
        st.metric("Avg Gap", f"{avg_gap:.1f}s")

    # Bar chart: which reports have the most double-runs
    top_offenders = (
        df_doubles
        .group_by("ReportName")
        .agg(pl.len().alias("double_runs"))
        .sort("double_runs", descending=True)
        .head(15)
        .to_pandas()
    )

    fig = go.Figure(go.Bar(
        x=top_offenders["double_runs"],
        y=top_offenders["ReportName"],
        orientation="h",
        marker_color="#e67e22",
        hovertemplate="<b>%{y}</b><br>Double-runs: %{x}<extra></extra>",
    ))
    fig.update_layout(
        title="Top Reports by Double-Run Count",
        xaxis_title="Double-Run Events",
        yaxis=dict(autorange="reversed"),
        height=max(250, len(top_offenders) * 28),
        margin=dict(l=0, r=0, t=40, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cccccc"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Drilldown table
    with st.expander("View Double-Run Event Details"):
        display_cols = ["ReportName", "started_at", "prev_started_at", "gap_seconds", "execution_status"]
        if match_params and "Parameters" in df_doubles.columns:
            display_cols.insert(1, "Parameters")
        available = [c for c in display_cols if c in df_doubles.columns]
        st.dataframe(
            df_doubles.select(available).sort("gap_seconds").to_pandas(),
            use_container_width=True
        )

