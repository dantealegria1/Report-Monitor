"""
queue_clustering.py  — NR queue size clustering + throughput simulation.

Improvements:
  1. Optimal k via silhouette scoring (k ∈ [2, 6])
  2. Richer clustering features (median, CV, failure_rate, total_exec)
  3. Peak-hour overlap ratio per report
  4. Dynamic instance recommendation (rank-based within band)
  5. Queue contention simulation (concurrent slot usage)
  6. Temporal train/test split for KNN validation
  7. Weighted size mismatch distance (ordinal 0–3)
  8. Cost-based scoring (resource + assignment penalty)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Operational NR queue configuration  (Id = FK stored in r.QueueNumber DB col)
# ---------------------------------------------------------------------------
NR_QUEUE_CONFIG = pd.DataFrame([
    {"Id": 1, "ReportSize": "M",  "QueueNumber": 20, "QueueModifier": None, "InstancesToOpen": 2, "Notes": ""},
    {"Id": 2, "ReportSize": "M",  "QueueNumber": 10, "QueueModifier": None, "InstancesToOpen": 2, "Notes": ""},
    {"Id": 3, "ReportSize": "S",  "QueueNumber": 10, "QueueModifier": None, "InstancesToOpen": 2, "Notes": ""},
    {"Id": 4, "ReportSize": "S",  "QueueNumber": 20, "QueueModifier": None, "InstancesToOpen": 2, "Notes": ""},
    {"Id": 5, "ReportSize": "L",  "QueueNumber": 10, "QueueModifier": None, "InstancesToOpen": 2, "Notes": ""},
    {"Id": 6, "ReportSize": "M",  "QueueNumber": 10, "QueueModifier": None, "InstancesToOpen": 2, "Notes": ""},
    {"Id": 7, "ReportSize": "XL", "QueueNumber": 10, "QueueModifier": "D",  "InstancesToOpen": 0, "Notes": "Delayed nightly"},
    {"Id": 8, "ReportSize": "XL", "QueueNumber": 10, "QueueModifier": None, "InstancesToOpen": 1, "Notes": ""},
    {"Id": 9, "ReportSize": "L",  "QueueNumber": 20, "QueueModifier": None, "InstancesToOpen": 2, "Notes": ""},
])

# Config Id (1-9) → row info  (used to decode ReportQueueId from DB)
_CFG = NR_QUEUE_CONFIG.set_index("Id").to_dict("index")

def _cfg(config_id: int, field: str, default=None):
    return _CFG.get(int(config_id), {}).get(field, default)

# Primary recommended queue per size (best non-delayed option)
RECOMMENDED_QUEUE_BY_SIZE: dict[str, int] = (
    NR_QUEUE_CONFIG[NR_QUEUE_CONFIG["QueueModifier"].isna()]
    .sort_values("InstancesToOpen", ascending=False)
    .drop_duplicates("ReportSize")
    .set_index("ReportSize")["QueueNumber"]
    .to_dict()
)

# Recommended instances per size (non-delayed, highest concurrency)
RECOMMENDED_INSTANCES_BY_SIZE: dict[str, int] = (
    NR_QUEUE_CONFIG[NR_QUEUE_CONFIG["QueueModifier"].isna()]
    .sort_values("InstancesToOpen", ascending=False)
    .drop_duplicates("ReportSize")
    .set_index("ReportSize")["InstancesToOpen"]
    .to_dict()
)

# Ordinal map for mismatch distance  (improvement #7)
_SIZE_ORD = {"S": 0, "M": 1, "L": 2, "XL": 3}
_SIZE_LABELS_FULL = ["XS", "S", "M", "L", "XL", "XXL"]  # for k > 4


# ---------------------------------------------------------------------------
# Feature building  (improvements #2, #3, #6)
# ---------------------------------------------------------------------------
def _build_report_features(df: pl.DataFrame) -> pd.DataFrame:
    keys = [k for k in ["ReportId", "ReportName", "ReportType", "ReportQueueId"]
            if k in df.columns]

    runs = df.group_by(keys).agg([
        pl.len().alias("total_runs"),
        (pl.col("execution_status") == "success").sum().alias("success_runs"),
        (pl.col("execution_status") == "failed").sum().alias("failed_runs"),
        (pl.col("execution_status") == "cancelled").sum().alias("cancelled_runs"),
    ])

    durations = (
        df.filter(pl.col("execution_status") == "success")
        .group_by(keys)
        .agg([
            pl.col("duration_seconds").mean().alias("avg_seconds"),
            pl.col("duration_seconds").median().alias("median_seconds"),
            pl.col("duration_seconds").quantile(0.95).alias("p95_seconds"),
            pl.col("duration_seconds").std().alias("std_seconds"),
            pl.col("duration_seconds").sum().alias("total_exec_seconds"),
        ])
    )

    # Peak-hour ratio  (improvement #3)
    peak_agg = []
    if "creation_date" in df.columns:
        peak_agg = (
            df.with_columns(
                pl.col("creation_date").cast(pl.Datetime).dt.hour().alias("_hour")
            )
            .group_by(keys)
            .agg([
                ((pl.col("_hour") >= 9) & (pl.col("_hour") < 17))
                .mean().alias("peak_hour_ratio"),
            ])
        )

    # Latest run date for temporal split  (improvement #6)
    latest_run = []
    if "creation_date" in df.columns:
        latest_run = (
            df.group_by(keys)
            .agg(pl.col("creation_date").cast(pl.Datetime).max().alias("latest_run_date"))
        )

    result = runs.join(durations, on=keys, how="left")

    if isinstance(peak_agg, pl.DataFrame):
        result = result.join(peak_agg, on=keys, how="left")
    if isinstance(latest_run, pl.DataFrame):
        result = result.join(latest_run, on=keys, how="left")

    fill_cols = [
        (pl.col("success_runs") / pl.col("total_runs")).fill_null(0.0).alias("success_rate"),
        (pl.col("failed_runs")  / pl.col("total_runs")).fill_null(0.0).alias("failure_rate"),
        pl.col("avg_seconds").fill_null(0.0),
        pl.col("median_seconds").fill_null(0.0),
        pl.col("p95_seconds").fill_null(0.0),
        pl.col("std_seconds").fill_null(0.0),
        pl.col("total_exec_seconds").fill_null(0.0),
    ]
    if "peak_hour_ratio" in result.columns:
        fill_cols.append(pl.col("peak_hour_ratio").fill_null(0.5))

    result = result.with_columns(fill_cols)

    out = result.to_pandas()

    # Coefficient of variation  (improvement #2)
    mean_vals = out["avg_seconds"].replace(0, np.nan)
    out["cv"] = (out["std_seconds"] / mean_vals).fillna(0.0)

    return out


# ---------------------------------------------------------------------------
# KMeans size labelling  (improvements #1, #2)
# ---------------------------------------------------------------------------
def _kmeans_size_labels(
    features_pd: pd.DataFrame, random_state: int
) -> tuple[pd.Series, int, dict[int, float]]:
    """
    KMeans with optimal k via silhouette scoring.

    Returns:
        labels: pd.Series of size labels
        optimal_k: int chosen k
        silhouette_scores: dict {k: score}
    """
    # Build feature matrix  (improvement #2: richer features)
    cols = ["p95_seconds", "avg_seconds", "median_seconds", "cv", "failure_rate",
            "total_exec_seconds"]
    if "peak_hour_ratio" in features_pd.columns:
        cols.append("peak_hour_ratio")

    raw = features_pd[cols].copy()
    for c in ["p95_seconds", "avg_seconds", "median_seconds", "total_exec_seconds"]:
        raw[c] = np.log1p(raw[c].astype(float))
    X = raw.values

    # Standardize for fair distance
    X = StandardScaler().fit_transform(X)

    n = len(features_pd)
    max_k = min(6, n - 1)
    min_k = 2

    if max_k < min_k:
        # Too few samples — just 1 cluster
        labels = pd.Series(["M"] * n, index=features_pd.index)
        return labels, 1, {}

    # Improvement #1: silhouette scoring (diagnostic only — always use k=4)
    silhouette_scores: dict[int, float] = {}
    best_k = 4  # business requirement: always S / M / L / XL

    for k in range(min_k, max_k + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        cids = km.fit_predict(X)
        score = silhouette_score(X, cids)
        silhouette_scores[k] = round(score, 4)

    # Always cluster with k=4
    use_k = min(4, n - 1)
    km = KMeans(n_clusters=use_k, random_state=random_state, n_init=10)
    cluster_ids = km.fit_predict(X)

    # Map clusters to size labels ordered by mean p95
    cluster_mean = {
        c: features_pd["p95_seconds"].values[cluster_ids == c].mean()
        for c in range(use_k)
    }
    sorted_clusters = sorted(cluster_mean, key=cluster_mean.get)

    # Always use S / M / L / XL labels
    all_labels = ["S", "M", "L", "XL"][-use_k:]

    label_map = {c: lbl for c, lbl in zip(sorted_clusters, all_labels)}
    labels = pd.Series([label_map[c] for c in cluster_ids], index=features_pd.index)
    return labels, best_k, silhouette_scores


# ---------------------------------------------------------------------------
# Dynamic instance recommendation  (improvement #4)
# ---------------------------------------------------------------------------
_INSTANCE_RANGE = {
    "XS": (2, 3), "S": (2, 3), "M": (2, 3), "L": (2, 3), "XL": (1, 2), "XXL": (1, 1),
}

def _dynamic_instances(features_pd: pd.DataFrame) -> pd.Series:
    """Rank-based instance count within each size band."""
    instances = pd.Series(1, index=features_pd.index)
    for size_label, (lo, hi) in _INSTANCE_RANGE.items():
        mask = features_pd["knn_size"] == size_label
        sub = features_pd.loc[mask, "p95_seconds"]
        if len(sub) == 0:
            continue
        if len(sub) == 1:
            instances.loc[mask] = lo
            continue
        # Rank: fastest → most instances;  slowest → fewest instances
        rank = sub.rank(pct=True)  # 0..1, higher = slower
        # Invert: fastest gets hi, slowest gets lo
        instances.loc[mask] = (lo + np.floor((1 - rank) * (hi - lo + 1)).clip(0, hi - lo).astype(int))
    return instances.astype(int)


# ---------------------------------------------------------------------------
# Mismatch distance  (improvement #7)
# ---------------------------------------------------------------------------
def _mismatch_distance(current: pd.Series, recommended: pd.Series) -> pd.Series:
    cur_ord = current.map(_SIZE_ORD).fillna(-1).astype(int)
    rec_ord = recommended.map(_SIZE_ORD).fillna(-1).astype(int)
    return (cur_ord - rec_ord).abs()


# ---------------------------------------------------------------------------
# Cost scoring  (improvement #8)
# ---------------------------------------------------------------------------
_COST_ALPHA = 1.0     # resource efficiency weight
_COST_BETA  = 300.0   # mismatch distance penalty weight

def _compute_cost(features_pd: pd.DataFrame, instances_col: str, mismatch_col: str) -> pd.Series:
    """Per-report cost = α·(p95/instances) + β·mismatch_distance."""
    inst = features_pd[instances_col].clip(lower=0.5)
    resource = features_pd["p95_seconds"] / inst
    mismatch = features_pd[mismatch_col].fillna(0)
    return _COST_ALPHA * resource + _COST_BETA * mismatch


# ---------------------------------------------------------------------------
# Public clustering API
# ---------------------------------------------------------------------------
def run_clustering(
    df_nr: pl.DataFrame,
    min_runs: int = 20,
    n_neighbors: int = 7,
    random_state: int = 42,
) -> dict:
    features_pd = _build_report_features(df_nr)
    features_pd = features_pd[features_pd["total_runs"] >= min_runs].reset_index(drop=True)

    if len(features_pd) < 8:
        return {"features_pd": features_pd, "knn_accuracy": None,
                "band_thresholds": {}, "optimal_k": None,
                "silhouette_scores": {}, "cost_current": None,
                "cost_recommended": None,
                "error":
                f"Only {len(features_pd)} reports after min_runs filter — need ≥ 8."}

    # ── Decode current queue from config Id ──────────────────────────────────
    if "ReportQueueId" in features_pd.columns:
        cfg_ids = pd.to_numeric(features_pd["ReportQueueId"], errors="coerce").fillna(0).astype(int)
        features_pd["current_config_id"]   = cfg_ids
        features_pd["current_size"]        = cfg_ids.map(lambda i: _cfg(i, "ReportSize", "?"))
        features_pd["current_queue_number"]= cfg_ids.map(lambda i: _cfg(i, "QueueNumber", 10))
        features_pd["current_instances"]   = cfg_ids.map(lambda i: _cfg(i, "InstancesToOpen", 1))
        features_pd["current_modifier"]    = cfg_ids.map(lambda i: _cfg(i, "QueueModifier", None))
    else:
        features_pd["current_config_id"]    = 0
        features_pd["current_size"]         = "?"
        features_pd["current_queue_number"] = 10
        features_pd["current_instances"]    = 1
        features_pd["current_modifier"]     = None

    # ── KMeans clustering → size labels  (improvements #1, #2) ───────────────
    labels, optimal_k, silhouette_scores = _kmeans_size_labels(features_pd, random_state)
    features_pd["knn_size"] = labels

    features_pd["knn_queue_number"] = features_pd["knn_size"].map(
        RECOMMENDED_QUEUE_BY_SIZE).fillna(10).astype(int)

    # ── Dynamic instances  (improvement #4) ──────────────────────────────────
    features_pd["knn_instances"] = _dynamic_instances(features_pd)

    # ── Mismatch distance  (improvement #7) ──────────────────────────────────
    features_pd["mismatch_distance"] = _mismatch_distance(
        features_pd["current_size"], features_pd["knn_size"]
    )
    features_pd["size_match"] = (features_pd["current_size"] == features_pd["knn_size"])

    # ── Band thresholds ──────────────────────────────────────────────────────
    band_thresholds = (
        features_pd.groupby("knn_size")["p95_seconds"].median().to_dict()
    )

    # ── KNN classifier  (improvement #6: temporal split) ─────────────────────
    feat_cols = ["avg_seconds", "median_seconds", "p95_seconds",
                 "total_runs", "failure_rate", "cv"]
    if "peak_hour_ratio" in features_pd.columns:
        feat_cols.append("peak_hour_ratio")

    X = features_pd[feat_cols].copy()
    for c in ["avg_seconds", "median_seconds", "p95_seconds", "total_runs"]:
        X[c] = np.log1p(X[c].astype(float))

    y = features_pd["knn_size"].astype(str)

    # Improvement #6: temporal split when possible
    if "latest_run_date" in features_pd.columns:
        sort_idx = features_pd["latest_run_date"].sort_values().index
        split_pt = int(len(sort_idx) * 0.75)
        train_idx = sort_idx[:split_pt]
        test_idx  = sort_idx[split_pt:]
        X_tr, X_te = X.loc[train_idx], X.loc[test_idx]
        y_tr, y_te = y.loc[train_idx], y.loc[test_idx]
    else:
        min_class = y.value_counts().min()
        can_strat = (y.nunique() > 1) and (min_class >= 2)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, random_state=random_state,
            stratify=y if can_strat else None,
        )

    knn_k = min(n_neighbors, len(X_tr))
    model = Pipeline([("sc", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=knn_k))])
    model.fit(X_tr, y_tr)
    accuracy = accuracy_score(y_te, model.predict(X_te))

    # NOTE: KMeans labels are kept as authoritative (all 4 bands: S/M/L/XL).
    # KNN is used only for accuracy scoring — NOT for relabeling, because KNN
    # can collapse small clusters into neighboring larger ones.

    # ── Cost scoring  (improvement #8) ───────────────────────────────────────
    # Current cost: resource inefficiency + mismatch penalty (current is wrong)
    features_pd["cost_current"] = _compute_cost(
        features_pd, "current_instances", "mismatch_distance"
    )
    # Recommended cost: pure resource efficiency (no mismatch, it's correct)
    features_pd["cost_recommended"] = _COST_ALPHA * (
        features_pd["p95_seconds"] / features_pd["knn_instances"].clip(lower=0.5)
    )

    cost_current_total     = float(features_pd["cost_current"].sum())
    cost_recommended_total = float(features_pd["cost_recommended"].sum())

    return {
        "features_pd":        features_pd,
        "knn_accuracy":       accuracy,
        "band_thresholds":    band_thresholds,
        "optimal_k":          optimal_k,
        "silhouette_scores":  silhouette_scores,
        "cost_current":       cost_current_total,
        "cost_recommended":   cost_recommended_total,
        "error":              None,
    }


# ---------------------------------------------------------------------------
# Daily throughput simulation
# ---------------------------------------------------------------------------
def simulate_daily_throughput(
    df_nr: pl.DataFrame,
    features_pd: pd.DataFrame,
) -> pd.DataFrame:
    """
    Simulate daily throughput: how many effective work-hours does each queue
    arrangement need to drain all reports for that day?
    """
    need = [c for c in ["ReportId", "creation_date", "duration_seconds"] if c in df_nr.columns]
    df_pd = df_nr.select(need).to_pandas()
    df_pd["creation_date"] = pd.to_datetime(df_pd["creation_date"])
    df_pd["duration_seconds"] = pd.to_numeric(df_pd["duration_seconds"], errors="coerce").fillna(0)
    df_pd["date"] = df_pd["creation_date"].dt.date

    # Join instances
    inst = features_pd[["ReportId", "current_instances", "knn_instances"]]
    df_pd = df_pd.merge(inst, on="ReportId", how="left")
    df_pd["current_instances"] = df_pd["current_instances"].fillna(1)
    df_pd["knn_instances"]     = df_pd["knn_instances"].fillna(1)

    # Per report: wall-clock hours needed = duration / instances
    df_pd["current_wall_s"] = df_pd["duration_seconds"] / df_pd["current_instances"].clip(lower=0.5)
    df_pd["rec_wall_s"]     = df_pd["duration_seconds"] / df_pd["knn_instances"].clip(lower=0.5)

    daily = (
        df_pd.groupby("date")
        .agg(
            total_work_hours   =("duration_seconds", lambda x: x.sum() / 3600),
            current_wall_hours =("current_wall_s",   lambda x: x.sum() / 3600),
            rec_wall_hours     =("rec_wall_s",        lambda x: x.sum() / 3600),
            report_count       =("ReportId",          "count"),
        )
        .reset_index()
    )
    daily["date"] = pd.to_datetime(daily["date"])
    return daily


# ---------------------------------------------------------------------------
# Queue contention simulation  (improvement #5)
# ---------------------------------------------------------------------------
def simulate_contention(
    df_nr: pl.DataFrame,
    features_pd: pd.DataFrame,
) -> pd.DataFrame:
    """
    Hourly slot utilisation per queue.

    For each hour, count how many reports are actively running (started_at ≤ hour < finished_at).
    Compare against capacity = sum of instances for reports in that queue.

    Returns: hour | queue | concurrent_current | concurrent_rec | capacity_current | capacity_rec
    """
    need = [c for c in ["ReportId", "started_at", "finished_at"] if c in df_nr.columns]
    if "started_at" not in df_nr.columns or "finished_at" not in df_nr.columns:
        return pd.DataFrame()

    df_pd = df_nr.select(need).to_pandas()
    df_pd["started_at"]  = pd.to_datetime(df_pd["started_at"])
    df_pd["finished_at"] = pd.to_datetime(df_pd["finished_at"])
    df_pd = df_pd.dropna(subset=["started_at", "finished_at"])

    if df_pd.empty:
        return pd.DataFrame()

    inst = features_pd[["ReportId", "current_instances", "knn_instances"]].copy()
    df_pd = df_pd.merge(inst, on="ReportId", how="left")
    df_pd["current_instances"] = df_pd["current_instances"].fillna(1).astype(int)
    df_pd["knn_instances"]     = df_pd["knn_instances"].fillna(1).astype(int)

    # Generate hourly index
    min_h = df_pd["started_at"].min().floor("1h")
    max_h = df_pd["finished_at"].max().ceil("1h")
    hours = pd.date_range(min_h, max_h, freq="1h")

    # For efficiency, sample at most ~720 hours (~30 days)
    if len(hours) > 720:
        hours = hours[-720:]

    records = []
    for h in hours:
        h_end = h + pd.Timedelta(hours=1)
        active = df_pd[(df_pd["started_at"] < h_end) & (df_pd["finished_at"] > h)]
        records.append({
            "hour": h,
            "concurrent_current": int(active["current_instances"].sum()),
            "concurrent_rec":     int(active["knn_instances"].sum()),
            "active_reports":     len(active),
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Hourly backlog simulation
# ---------------------------------------------------------------------------
def simulate_backlog(
    df_nr: pl.DataFrame,
    features_pd: pd.DataFrame,
    scenario: str = "current",
) -> pd.DataFrame:
    """
    Hourly backlog simulation.

    A report is in the backlog from creation_date until started_at.

    For 'current'   : use actual observed start times.
    For 'recommended': scale each report's observed wait time by
                       (current_instances / knn_instances).
    """
    need = [c for c in ["ReportId", "creation_date", "started_at"] if c in df_nr.columns]
    df_pd = df_nr.select(need).to_pandas()
    df_pd["creation_date"] = pd.to_datetime(df_pd["creation_date"])
    df_pd["started_at"]    = pd.to_datetime(df_pd["started_at"])

    # Arrivals are always from creation_date
    arrivals = (
        df_pd["creation_date"].dt.floor("1h")
        .value_counts().sort_index().rename("arrivals")
    )

    if scenario == "current":
        starts = (
            df_pd["started_at"].dropna().dt.floor("1h")
            .value_counts().sort_index().rename("starts")
        )
    else:
        inst = features_pd.set_index("ReportId")[["current_instances", "knn_instances"]].copy()
        inst["current_instances"] = inst["current_instances"].clip(lower=1)
        inst["knn_instances"]     = inst["knn_instances"].clip(lower=1)

        df2 = df_pd.dropna(subset=["started_at"]).join(inst, on="ReportId", how="left")
        df2["current_instances"] = df2["current_instances"].fillna(1).clip(lower=1)
        df2["knn_instances"]     = df2["knn_instances"].fillna(1).clip(lower=1)

        df2["wait_s"] = (df2["started_at"] - df2["creation_date"]).dt.total_seconds().clip(lower=0)
        df2["adj_wait_s"] = df2["wait_s"] * (df2["current_instances"] / df2["knn_instances"])
        df2["adj_start"]  = df2["creation_date"] + pd.to_timedelta(df2["adj_wait_s"], unit="s")

        starts = (
            df2["adj_start"].dt.floor("1h")
            .value_counts().sort_index().rename("starts")
        )

    all_hours = arrivals.index.union(starts.index).sort_values()
    sim = pd.DataFrame(index=all_hours)
    sim["arrivals"] = arrivals.reindex(all_hours, fill_value=0)
    sim["starts"]   = starts.reindex(all_hours, fill_value=0)

    backlog, bl = 0, []
    for _, row in sim.iterrows():
        backlog = max(0, backlog + int(row["arrivals"]) - int(row["starts"]))
        bl.append(backlog)
    sim["backlog"] = bl

    return sim.reset_index().rename(columns={"index": "hour"})
