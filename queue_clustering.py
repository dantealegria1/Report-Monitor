"""
queue_clustering.py  — NR queue size clustering + throughput simulation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
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
    {"Id": 7, "ReportSize": "XL", "QueueNumber": 10, "QueueModifier": "D",  "InstancesToOpen": 0, "Notes": "Delayed nightly"},
    {"Id": 8, "ReportSize": "XL", "QueueNumber": 10, "QueueModifier": None, "InstancesToOpen": 1, "Notes": ""},
])

# Config Id (1-8) → row info  (used to decode ReportQueueId from DB)
_CFG = NR_QUEUE_CONFIG.set_index("Id").to_dict("index")
# e.g. _CFG[7] = {"ReportSize":"XL","QueueNumber":10,"QueueModifier":"D","InstancesToOpen":0,...}

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


# ---------------------------------------------------------------------------
# Feature building
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
            pl.col("duration_seconds").sum().alias("total_exec_seconds"),
        ])
    )

    return (
        runs.join(durations, on=keys, how="left")
        .with_columns([
            (pl.col("success_runs") / pl.col("total_runs")).fill_null(0.0).alias("success_rate"),
            (pl.col("failed_runs")  / pl.col("total_runs")).fill_null(0.0).alias("failure_rate"),
            pl.col("avg_seconds").fill_null(0.0),
            pl.col("median_seconds").fill_null(0.0),
            pl.col("p95_seconds").fill_null(0.0),
            pl.col("total_exec_seconds").fill_null(0.0),
        ])
        .to_pandas()
    )


# ---------------------------------------------------------------------------
# KMeans size labelling
# ---------------------------------------------------------------------------
def _kmeans_size_labels(features_pd: pd.DataFrame, random_state: int) -> pd.Series:
    """KMeans(k=4) on log1p features → labels ranked S < M < L < XL."""
    X = np.column_stack([
        np.log1p(features_pd["p95_seconds"].values),
        np.log1p(features_pd["avg_seconds"].values),
    ])
    n_k = min(4, len(features_pd))
    km = KMeans(n_clusters=n_k, random_state=random_state, n_init=10)
    cluster_ids = km.fit_predict(X)

    cluster_mean = {
        c: features_pd["p95_seconds"].values[cluster_ids == c].mean()
        for c in range(n_k)
    }
    sorted_clusters = sorted(cluster_mean, key=cluster_mean.get)
    while len(sorted_clusters) < 4:
        sorted_clusters.append(sorted_clusters[-1])
    label_map = {c: lbl for c, lbl in zip(sorted_clusters, ["S", "M", "L", "XL"])}
    return pd.Series([label_map[c] for c in cluster_ids], index=features_pd.index)


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
                "band_thresholds": {}, "error":
                f"Only {len(features_pd)} reports after min_runs filter — need ≥ 8."}

    # ── Decode current queue from config Id ──────────────────────────────────
    # ReportQueueId from DB = r.QueueNumber = FK to ReportQueues.Id (1,2,3,4,5,7,8)
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

    # ── KMeans clustering → size labels ──────────────────────────────────────
    features_pd["knn_size"] = _kmeans_size_labels(features_pd, random_state)
    features_pd["knn_queue_number"] = features_pd["knn_size"].map(
        RECOMMENDED_QUEUE_BY_SIZE).fillna(10).astype(int)
    features_pd["knn_instances"] = features_pd["knn_size"].map(
        RECOMMENDED_INSTANCES_BY_SIZE).fillna(1).astype(int)

    # ── Match: compare SIZE labels (both are S/M/L/XL) ───────────────────────
    features_pd["size_match"] = (
        features_pd["current_size"] == features_pd["knn_size"]
    )

    # ── Band thresholds (median p95 per knn cluster, for display) ────────────
    band_thresholds = (
        features_pd.groupby("knn_size")["p95_seconds"].median().to_dict()
    )

    # ── KNN classifier on top of KMeans labels ───────────────────────────────
    X = features_pd[["avg_seconds", "median_seconds", "p95_seconds",
                      "total_runs", "failure_rate"]].copy()
    for c in ["avg_seconds", "median_seconds", "p95_seconds", "total_runs"]:
        X[c] = np.log1p(X[c].astype(float))

    y = features_pd["knn_size"].astype(str)
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

    features_pd["knn_size"] = model.predict(X)   # refine with actual KNN
    # re-derive after KNN refinement
    features_pd["knn_queue_number"] = features_pd["knn_size"].map(
        RECOMMENDED_QUEUE_BY_SIZE).fillna(10).astype(int)
    features_pd["knn_instances"] = features_pd["knn_size"].map(
        RECOMMENDED_INSTANCES_BY_SIZE).fillna(1).astype(int)
    features_pd["size_match"] = (features_pd["current_size"] == features_pd["knn_size"])

    return {
        "features_pd":     features_pd,
        "knn_accuracy":    accuracy,
        "band_thresholds": band_thresholds,
        "error":           None,
    }


# ---------------------------------------------------------------------------
# Daily throughput simulation
# ---------------------------------------------------------------------------
#
# A report is "in the backlog" from creation_date until started_at.
#
# Throughput model per queue:
#   An instance can handle one report at a time.
#   Daily capacity (seconds of work) = instances × 86 400
#   If daily work > capacity → overflow backlog carried to next day.
#
# We compare:
#   Current : each report uses its current_instances
#   Recommended : each report uses its knn_instances

def simulate_daily_throughput(
    df_nr: pl.DataFrame,
    features_pd: pd.DataFrame,
) -> pd.DataFrame:
    """
    Simulate daily throughput: how many effective work-hours does each queue
    arrangement need to drain all reports for that day?

    Returns pd.DataFrame: date | total_work_hours | current_wall_hours | rec_wall_hours
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


def simulate_backlog(
    df_nr: pl.DataFrame,
    features_pd: pd.DataFrame,
    scenario: str = "current",   # "current" | "recommended"
) -> pd.DataFrame:
    """
    Hourly backlog simulation.

    A report is in the backlog from creation_date until started_at.

    For 'current'   : use actual observed start times.
    For 'recommended': scale each report's observed wait time by
                       (current_instances / knn_instances).
        - More recommended instances  → shorter wait → earlier start → less backlog
        - Fewer recommended instances → longer wait  → later start  → more backlog
        knn_instances is clipped to ≥ 1 (delayed/0-instance queues treated as 1).
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
        # Ground truth: use actual starts
        starts = (
            df_pd["started_at"].dropna().dt.floor("1h")
            .value_counts().sort_index().rename("starts")
        )
    else:
        # Per-report: scale wait time by instances ratio
        inst = features_pd.set_index("ReportId")[["current_instances", "knn_instances"]].copy()
        inst["current_instances"] = inst["current_instances"].clip(lower=1)
        inst["knn_instances"]     = inst["knn_instances"].clip(lower=1)   # treat delayed as 1

        df2 = df_pd.dropna(subset=["started_at"]).join(inst, on="ReportId", how="left")
        df2["current_instances"] = df2["current_instances"].fillna(1).clip(lower=1)
        df2["knn_instances"]     = df2["knn_instances"].fillna(1).clip(lower=1)

        # observed wait * (current / recommended) = adjusted wait
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
