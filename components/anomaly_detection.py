"""
Anomaly detection component.
Handles anomaly detection, drift analysis, and model validation (confusion matrix).
"""
import streamlit as st
import polars as pl
from utils.statistics import robust_z_mad, calculate_ks_drift
from utils.data_processing import enrich_hourly_ts_with_features

def get_enriched_backlog(df_raw):
    """Ensure hourly_ts has backlog feature, enriching if necessary."""
    hourly_ts = st.session_state.get("hourly_ts")
    if hourly_ts is None:
        return None
        
    if "backlog" not in hourly_ts.columns:
        with st.spinner("Calculating operational backlog for anomaly detection..."):
            hourly_ts = enrich_hourly_ts_with_features(hourly_ts, df_raw)
            st.session_state["hourly_ts"] = hourly_ts
            
    return hourly_ts

def render_anomaly_detection(df_filtered: pl.DataFrame):
    """
    Render anomaly detection section with robust, contextual, and multivariate methods.
    Added: Isolation Forest (multivariate), Drfit Analysis (KS Test), Confusion Matrix (labeling).
    
    Args:
        df_filtered: Filtered DataFrame
    """
    st.subheader("Anomaly Detection System")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Detection Methods", 
        "Feature Drift (KS Test)", 
        "Report Inspector",
        "Confusion Matrix (Labeling)"
    ])
    
    # ----------------------------
    # TAB 1: Detection Methods
    # ----------------------------
    with tab1:
        st.markdown("### 2.1 Multivariate Anomaly Detection")
        
        c1, c2 = st.columns([2, 1])
        with c1:
            method = st.radio(
                "Selected Method",
                ["Robust Z-Score (Univary)", "Contextual Z-Score (Multivary Proxy)", "Isolation Forest (Multivariate AI)"],
                horizontal=True
            )
        with c2:
            use_adaptive = st.checkbox("Adaptive Baseline", value=False, help="Use rolling window (last 50 runs) instead of global stats. Useful for handling permanent performance shifts like migrations.")

        min_score = st.slider("Minimum Anomaly Score (Filter)", 3.0, 10.0, 5.0, 0.1, help="Show only anomalies with a score higher than this value.")
        
        # Prepare Data
        df_anom = df_filtered.sort(["ReportName", "started_at"])
        
        if method == "Robust Z-Score (Univary)":
            # MAD method
            # Use min_score directly as threshold
            
            if use_adaptive:
                # Rolling window Median & MAD approximation via IQR
                # Window = 50 runs
                df_anom = df_anom.with_columns([
                    pl.col("duration_seconds").rolling_median(window_size=50, min_periods=5).over("ReportName").alias("mediana_rep"),
                    pl.col("duration_seconds").rolling_quantile(0.75, window_size=50, min_periods=5).over("ReportName").alias("p75"),
                    pl.col("duration_seconds").rolling_quantile(0.25, window_size=50, min_periods=5).over("ReportName").alias("p25"),
                ]).with_columns([
                    ((pl.col("p75") - pl.col("p25")) / 1.349).alias("mad_rep") # IQR to MAD aprox
                ]).with_columns([
                    robust_z_mad(
                        pl.col("duration_seconds"),
                        pl.col("mediana_rep"),
                        pl.col("mad_rep")
                    ).alias("score")
                ])
            else:
                # Global
                df_anom = df_anom.with_columns([
                    pl.col("duration_seconds").median().over("ReportName").alias("mediana_rep"),
                    (pl.col("duration_seconds") - pl.col("duration_seconds").median().over("ReportName")).abs().median().over("ReportName").alias("mad_rep"),
                ]).with_columns([
                    robust_z_mad(
                        pl.col("duration_seconds"),
                        pl.col("mediana_rep"),
                        pl.col("mad_rep")
                    ).alias("score")
                ])
                
            df_anom = df_anom.filter(
                (pl.col("execution_status") == "success") &
                (pl.col("mad_rep") > 0) &
                (pl.col("score") > min_score)
            ).sort("score", descending=True)
            
        elif method == "Contextual Z-Score (Multivary Proxy)":
            # Contextual: ReportName + Hour
            # Use min_score directly
            
            if use_adaptive:
                 st.info("Adaptive Baseline uses (ReportName) rolling stats but checks against specific hour context.")
                 # Hybrid: Rolling baseline per report, but maybe check hour deviation?
                 # Actually standard rolling median is best proxy for "recent normal"
                 # Let's use similar logic to Univary but maybe window=30
                 df_anom = df_anom.with_columns([
                    pl.col("duration_seconds").rolling_median(window_size=30, min_periods=5).over("ReportName").alias("mediana_ctx"),
                    pl.col("duration_seconds").rolling_quantile(0.75, window_size=30, min_periods=5).over("ReportName").alias("p75"),
                    pl.col("duration_seconds").rolling_quantile(0.25, window_size=30, min_periods=5).over("ReportName").alias("p25"),
                ]).with_columns([
                    ((pl.col("p75") - pl.col("p25")) / 1.349).alias("mad_ctx") 
                ])
            else:
                 df_anom = df_anom.with_columns([
                    pl.col("duration_seconds").median().over(["ReportName", "hour"]).alias("mediana_ctx"),
                    (pl.col("duration_seconds") - pl.col("duration_seconds").median().over(["ReportName", "hour"])).abs().median().over(["ReportName", "hour"]).alias("mad_ctx"),
                ])

            df_anom = df_anom.with_columns([
                robust_z_mad(
                    pl.col("duration_seconds"),
                    pl.col("mediana_ctx"),
                    pl.col("mad_ctx")
                ).alias("score")
            ]).filter(
                (pl.col("execution_status") == "success") &
                (pl.col("mad_ctx") > 0) &
                (pl.col("score") > min_score)
            ).sort("score", descending=True)
            
        else: # Isolation Forest
            from sklearn.ensemble import IsolationForest
            
            # Enrich with Backlog if possible
            # Need raw data to compute backlog if not cached
            # For simplicity, we assume loaded data is available in a global or cached way, 
            # or strictly rely on session state if available
            hourly_ts = st.session_state.get("hourly_ts")
            
            # Prepare features
            features = ["duration_seconds", "hour", "weekday"]
            
            # Try to add 'backlog' by joining on hour
            if hourly_ts is not None:
                # Ensure backlog is calculated
                if "backlog" not in hourly_ts.columns:
                     # We can't easily calculate it here without df_raw logic which is expensive
                     # Just proceed without backlog or warn
                     st.warning("Backlog feature not available (visit Time Series page to generate). Using base features.")
                else:
                    # Join backlog
                    # df_filtered has 'started_at', hourly_ts has 'timestamp_hour'
                    df_prep = df_filtered.with_columns(
                        pl.col("started_at").dt.truncate("1h").alias("timestamp_hour")
                    ).join(
                        hourly_ts.select(["timestamp_hour", "backlog"]), 
                        on="timestamp_hour", 
                        how="left"
                    ).with_columns(pl.col("backlog").fill_null(0))
                    
                    df_anom = df_prep
                    features.append("backlog")
            
            # Prepare X
            X = df_anom.select(features).to_pandas().fillna(0)
            
            # Use 'auto' contamination to standardise scores
            iso = IsolationForest(contamination='auto', random_state=42)
            preds = iso.fit_predict(X)
            scores = iso.decision_function(X) 
            
            df_anom = df_anom.with_columns([
                pl.Series("iso_pred", preds),
                pl.Series("iso_score", scores)
            ]).with_columns(
                # Scaling: Map inverted score (0.0 to ~0.5) to Z-score like range (3.0 to ~13.0)
                # Formula: 3.0 + (inverted_score * 20.0)
                # 0.0 -> 3.0 (Boundary)
                # 0.1 -> 5.0 (Moderate)
                # 0.2 -> 7.0 (High)
                ((pl.col("iso_score") * -1 * 20.0) + 3.0).alias("score")
            ).filter(
                # Check prediction AND score threshold
                # If iso_pred != -1, it's not anomaly (so usually score ~< 3)
                # But we just use score > min_score which covers it
                pl.col("score") > min_score
            ).sort("score", descending=True)

        if df_anom.height > 0:
            st.error(f"{df_anom.height} Anomalies Detected")
            st.dataframe(
                df_anom.select([
                    "started_at", "ReportName", 
                    "duration_seconds", "score", "execution_status"
                ] + (["backlog"] if "backlog" in df_anom.columns else []) + 
                (["mediana_rep"] if "mediana_rep" in df_anom.columns else [])).to_pandas(),
                use_container_width=True
            )
            
            st.caption("ℹ️ **Score Legend**: 3.0-5.0 (Mild), 5.0-7.0 (Moderate), 7.0+ (Severe). Higher is more anomalous.")
            
            # Save for Histogram access in Tab 4
            st.session_state["latest_anomalies"] = df_anom

            # ---------------------------------------------------------
            # Git Integration: View History for Selected Anomaly
            # ---------------------------------------------------------
            st.divider()
            st.subheader("Root Cause Analysis (Git History)")
            has_report_file_path = "ControllerActionOrSP" in df_anom.columns
            
            # Create a selection list of anomalies
            # Format: "ReportName (Date | Score)"
            anom_options = df_anom.select([
                pl.format(
                    "{} ({} | Score: {})",
                    pl.col("ReportName"),
                    pl.col("started_at").dt.date(),
                    pl.col("score").round(1),
                ).alias("label"),
                pl.col("ReportName"),
                pl.col("started_at"),
                (
                    pl.col("ControllerActionOrSP")
                    if has_report_file_path
                    else pl.lit(None)
                ).alias("ControllerActionOrSP"),
            ]).to_pandas()

            if not has_report_file_path:
                st.info("`ControllerActionOrSP` is not available in this dataset. Git history lookup is disabled.")
            
            selected_label = st.selectbox(
                "Select Anomaly to Inspect Code Changes:",
                anom_options["label"].tolist(),
                key="git_history_select"
            )
            
            if selected_label:
                # Get the corresponding file path
                selected_row = anom_options[anom_options["label"] == selected_label].iloc[0]
                report_name = selected_row["ReportName"]
                anomaly_started_at = selected_row["started_at"]
                file_path_rel = selected_row["ControllerActionOrSP"] + ".sql"
                
                if file_path_rel:
                    from config import REPORTS_REPO_PATH
                    from utils.git_utils import get_git_history, GitError
                    
                    st.markdown(f"**Inspecting File**: `{file_path_rel}` for report *{report_name}*")
                    st.caption(f"Showing commits up to anomaly time: `{anomaly_started_at}`")
                    
                    try:
                        commits = get_git_history(
                            REPORTS_REPO_PATH,
                            file_path_rel,
                            limit=5,
                            before_datetime=anomaly_started_at,
                        )
                        
                        if commits:
                            st.caption(f"Last 5 commits for `{file_path_rel}`:")
                            for c in commits:
                                st.markdown(
                                    f"**{c['date']}** - `{c['hash']}` - **{c['author']}**<br>{c['message']}",
                                    unsafe_allow_html=True
                                )
                                st.divider()
                        else:
                            st.info(f"No git history found for `{file_path_rel}`. File might be renamed, moved or not tracked.")
                            
                    except GitError as e:
                        st.error(f"Git Error: {str(e)}")
                    except Exception as e:
                         st.error(f"Error retrieving history: {e}")
                else:
                    st.warning("No file path associated with this report (ControllerActionOrSP is empty).")

        else:
            st.success("Normal Operation: No anomalies detected.")

    # ----------------------------
    # TAB 2: Feature Drift (KS Test)
    # ----------------------------
    with tab2:
        render_drift_analysis(df_filtered)

    # ----------------------------
    # TAB 3: Report Inspector (Keep existing logic)
    # ----------------------------
    with tab3:
        render_report_inspector(df_filtered)
        
    # ----------------------------
    # TAB 4: Confusion Matrix
    # ----------------------------
    with tab4:
        render_manual_labeling()


def render_drift_analysis(df_filtered: pl.DataFrame):
    """Render drift analysis section using Kolmogorov-Smirnov Test."""
    st.markdown("### 2.2 Drift Analysis (KS Test)")
    st.caption("Statistical validation of distribution shift using **Kolmogorov-Smirnov (KS) Test**. Comparisons of p-value < 0.05 indicate significant drift.")
    
    drift_dim = st.selectbox("Drift Dimension", ["ReportName (Top 50)", "ReportType"])
    
    df_success = df_filtered.filter(pl.col("execution_status") == "success")
    
    if df_success.height < 50:
        st.warning("Insufficient data for robust drift analysis.")
        return

    run_drift = st.button("Run Drift Analysis")
    
    if run_drift:
        with st.spinner("Calculating KS Statistics..."):
            if drift_dim == "ReportType":
                drift_tbl = calculate_ks_drift(df_success, "duration_seconds", ["ReportType"])
            else:
                top50 = (
                    df_success.group_by("ReportName")
                    .agg(pl.len().alias("n"))
                    .sort("n", descending=True)
                    .head(50)["ReportName"] 
                )
                drift_tbl = calculate_ks_drift(
                    df_success.filter(pl.col("ReportName").is_in(top50)),
                    "duration_seconds",
                    ["ReportName"]
                )
            
            if drift_tbl.height > 0:
                # Add status icon
                drift_viz = drift_tbl.with_columns(
                    pl.when(pl.col("is_drift"))
                    .then(pl.lit("Drift"))
                    .otherwise(pl.lit("Stable"))
                    .alias("Status")
                )
                
                st.dataframe(
                    drift_viz.select(["Status", drift_dim.split()[0], "ks_stat", "p_value", "sample_a", "sample_b"]).to_pandas(),
                    use_container_width=True
                )
                
                n_drift = drift_viz.filter(pl.col("is_drift")).height
                if n_drift > 0:
                    st.warning(f"Detected **{n_drift}** entities with significant model drift.")
            else:
                st.info("No sufficient data in windows to compare.")

def render_report_inspector(df_filtered: pl.DataFrame):
    """
    Existing report inspector logic moved here.
    """
    st.subheader("Report Execution History")
    unique_reports = df_filtered["ReportName"].unique().sort()
    selected_report = st.selectbox("Select Report to Inspect", unique_reports)
    
    if selected_report:
        df_report = df_filtered.filter(pl.col("ReportName") == selected_report).sort("started_at", descending=True)
        
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Total Executions", df_report.height)
        with c2: st.metric("Avg Duration", f"{df_report['duration_seconds'].mean():.2f}s")
        with c3:
            fail_count = df_report.filter(pl.col("execution_status") != "success").height
            st.metric("Failures", fail_count)
            
        st.dataframe(
            df_report.select([
                "started_at", "execution_status", "duration_seconds", 
                "hour", "weekday", "ReportId", "Parameters"
            ]).to_pandas(),
            use_container_width=True
        )

        st.subheader("Performance over Time")
        df_daily = (
            df_report.with_columns(pl.col("started_at").dt.date().alias("date"))
            .group_by("date")
            .agg([
                pl.col("duration_seconds").mean().alias("avg_duration"),
                pl.len().alias("count")
            ])
            .sort("date")
        )

        st.vega_lite_chart(
             df_daily.to_pandas(),
             {
                 "mark": "line",
                 "encoding": {
                     "x": {"field": "date", "type": "temporal", "title": "Date"},
                     "y": {"field": "avg_duration", "type": "quantitative", "title": "Avg Duration (s)"},
                     "tooltip": ["date", "avg_duration", "count"]
                 },
                 "width": "container",
                 "height": 400
             },
             width='stretch'
        )

        st.subheader("Daily Executions")
        st.vega_lite_chart(
            df_daily.to_pandas(),
            {
                "mark": {"type": "bar", "color": "#4c78a8", "cornerRadius": 3},
                "encoding": {
                    "x": {"field": "date", "type": "temporal"},
                    "y": {"field": "count", "type": "quantitative"},
                    "tooltip": ["date", "count"]
                },
                "height": 300
            },
            width='stretch'
        )

def render_manual_labeling():
    """
    Render Activity 2.3: Confusion Matrix (Manual Labeling).
    """
    st.markdown("### 2.3 Confusion Matrix Validation")
    st.caption("Label a sample of detected anomalies to estimate model precision.")
    
    # Initialize session state for labels
    if "labels" not in st.session_state:
        st.session_state["labels"] = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
        
    cols = st.columns(4)
    cols[0].metric("True Positives (TP)", st.session_state["labels"]["TP"])
    cols[1].metric("False Positives (FP)", st.session_state["labels"]["FP"])
    cols[2].metric("True Negatives (TN)", st.session_state["labels"]["TN"])
    cols[3].metric("False Negatives (FN)", st.session_state["labels"]["FN"])
    
    tp = st.session_state["labels"]["TP"]
    fp = st.session_state["labels"]["FP"]
    
    if tp + fp > 0:
        precision = tp / (tp + fp)
        recall = tp / (tp + st.session_state["labels"]["FN"] + 0.0001) # Approx if FN is unknown usually
        f1 = 2 * (precision * recall) / (precision + recall + 0.00001)
        st.info(f"**Precision**: {precision:.2f} | **F1-Score**: {f1:.2f}")
    
    st.divider()
    
    # Get anomalies from session (populated in Tab 1)
    df_anom = st.session_state.get("latest_anomalies")
    
    if df_anom is None or df_anom.height == 0:
        st.warning("No anomalies detected in Tab 1. Run detection first.")
        return
        
    st.markdown("#### Validate this Anomaly:")
    
    # Show one random anomaly to label
    if "sample_idx" not in st.session_state:
        st.session_state["sample_idx"] = 0
        
    # Circular buffer
    idx = st.session_state["sample_idx"] % df_anom.height
    row = df_anom.row(idx, named=True)
    
    c1, c2 = st.columns([3, 1])
    with c1:
        st.json({
            "Report": row["ReportName"],
            "Duration": f"{row['duration_seconds']}s",
            "Score": f"{row.get('score', 0):.2f}",
            "Hour": row["hour"],
            "Backlog": row.get("backlog", "N/A")
        })
    with c2:
        if st.button("True Anomaly", type="primary", use_container_width=True):
            st.session_state["labels"]["TP"] += 1
            st.session_state["sample_idx"] += 1
            st.rerun()
            
        if st.button("False Alarm", use_container_width=True):
            st.session_state["labels"]["FP"] += 1
            st.session_state["sample_idx"] += 1
            st.rerun()

    if st.button("Reset Counters"):
        st.session_state["labels"] = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
        st.rerun()
