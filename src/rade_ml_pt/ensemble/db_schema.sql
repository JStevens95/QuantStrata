-- Ensemble Analytics DB-ready schema.
-- Compatible with both SQLite and Postgres.

CREATE TABLE IF NOT EXISTS ensemble_versions (
    version         TEXT PRIMARY KEY,
    n_clusters      INTEGER NOT NULL,
    n_trades        INTEGER NOT NULL,
    aggregation     TEXT,
    strategy        TEXT,
    evaluated_at    TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS ensemble_metrics (
    version         TEXT NOT NULL REFERENCES ensemble_versions(version),
    split           TEXT NOT NULL,
    metric_name     TEXT NOT NULL,
    metric_value    REAL,
    PRIMARY KEY (version, split, metric_name)
);

CREATE TABLE IF NOT EXISTS cluster_metrics (
    version         TEXT NOT NULL REFERENCES ensemble_versions(version),
    cluster_id      TEXT NOT NULL,
    split           TEXT NOT NULL,
    metric_name     TEXT NOT NULL,
    metric_value    REAL,
    PRIMARY KEY (version, cluster_id, split, metric_name)
);

CREATE TABLE IF NOT EXISTS cluster_attributes (
    version         TEXT NOT NULL REFERENCES ensemble_versions(version),
    cluster_id      TEXT NOT NULL,
    attribute_name  TEXT NOT NULL,
    attribute_value TEXT,
    PRIMARY KEY (version, cluster_id, attribute_name)
);

CREATE TABLE IF NOT EXISTS cluster_predictions (
    version         TEXT NOT NULL REFERENCES ensemble_versions(version),
    cluster_id      TEXT NOT NULL,
    split           TEXT NOT NULL,
    scenario_idx    INTEGER NOT NULL,
    prediction      REAL NOT NULL,
    target          REAL NOT NULL,
    PRIMARY KEY (version, cluster_id, split, scenario_idx)
);

CREATE INDEX IF NOT EXISTS idx_cp_split
    ON cluster_predictions(version, split);

CREATE TABLE IF NOT EXISTS portfolio_summary (
    version         TEXT NOT NULL REFERENCES ensemble_versions(version),
    split           TEXT NOT NULL,
    scenario_idx    INTEGER NOT NULL,
    prediction      REAL NOT NULL,
    target          REAL NOT NULL,
    PRIMARY KEY (version, split, scenario_idx)
);

CREATE TABLE IF NOT EXISTS portfolio_percentiles (
    version         TEXT NOT NULL REFERENCES ensemble_versions(version),
    split           TEXT NOT NULL,
    percentile      TEXT NOT NULL,
    pred_value      REAL,
    target_value    REAL,
    abs_error       REAL,
    PRIMARY KEY (version, split, percentile)
);

CREATE TABLE IF NOT EXISTS worst_scenarios (
    version         TEXT NOT NULL REFERENCES ensemble_versions(version),
    split           TEXT NOT NULL,
    rank            INTEGER NOT NULL,
    scenario_idx    INTEGER NOT NULL,
    prediction      REAL,
    target          REAL,
    abs_error       REAL,
    PRIMARY KEY (version, split, rank)
);

CREATE TABLE IF NOT EXISTS trade_metrics (
    version         TEXT NOT NULL REFERENCES ensemble_versions(version),
    cluster_id      TEXT NOT NULL,
    split           TEXT NOT NULL,
    trade_id        TEXT NOT NULL,
    mae             REAL,
    rmse            REAL,
    max_ae          REAL,
    p95_ae          REAL,
    mean_residual   REAL,
    std_residual    REAL,
    PRIMARY KEY (version, cluster_id, split, trade_id)
);

CREATE TABLE IF NOT EXISTS trade_cluster_map (
    version         TEXT NOT NULL REFERENCES ensemble_versions(version),
    trade_id        TEXT NOT NULL,
    cluster_id      TEXT NOT NULL,
    PRIMARY KEY (version, trade_id)
);

CREATE TABLE IF NOT EXISTS graph_stats (
    version         TEXT NOT NULL REFERENCES ensemble_versions(version),
    cluster_id      TEXT NOT NULL,
    n_nodes         INTEGER,
    n_edges         INTEGER,
    density         REAL,
    mean_weight     REAL,
    PRIMARY KEY (version, cluster_id)
);

-- JSON-blob tables for group summaries (simple storage).
CREATE TABLE IF NOT EXISTS group_summaries (
    version         TEXT NOT NULL REFERENCES ensemble_versions(version),
    split           TEXT NOT NULL,
    data            TEXT NOT NULL,
    PRIMARY KEY (version, split)
);

CREATE TABLE IF NOT EXISTS group_correlations (
    version         TEXT NOT NULL REFERENCES ensemble_versions(version),
    split           TEXT NOT NULL,
    data            TEXT NOT NULL,
    PRIMARY KEY (version, split)
);
