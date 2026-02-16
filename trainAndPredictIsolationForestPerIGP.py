#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import joblib

DMM_FILE = Path("synthetic_output/dmm_synthetic_flat.csv")
LSP_FILE = Path("synthetic_output/lsp_from_dmm.csv")
EVENTS_FILE = Path("synthetic_output/network_events.csv")

MODEL_FILE = Path("synthetic_output/isolation_forest_dmm.pkl")
SCALER_FILE = Path("synthetic_output/scaler_dmm.pkl")
OUT_FILE = Path("synthetic_output/failure_root_causes_with_igp_models.csv")

IGP_MODEL_DIR = Path("synthetic_output/igp_models")

EVENT_WINDOW = "1h"

# IGP model tuning
MIN_IGP_SAMPLES = 50
IGP_TREES = 75
IGP_CONTAMINATION = 0.05

# Loaders
def load_dmm():
    df = pd.read_csv(DMM_FILE)
    df["delay"] = pd.to_numeric(df["delay"], errors="coerce").fillna(0.0)
    df["resultclassification"] = df["resultclassification"].str.upper().str.strip()
    df["timeissued"] = pd.to_datetime(df["timeissued"], unit="ms", errors="coerce")
    return df


def load_igp():
    df = pd.read_csv(LSP_FILE)

    df["timeissued"] = pd.to_datetime(df["timeissued"], unit="ms", errors="coerce")
    df["igp_ids"] = df["lsp_links"].str.split(";")
    df["igp_delays"] = df["lsp_delays"].str.split(";")

    df = df.explode(["igp_ids", "igp_delays"], ignore_index=True)

    df["igp_id"] = df["igp_ids"].astype(str)
    df["igp_delay_ms"] = pd.to_numeric(df["igp_delays"], errors="coerce").fillna(0.0)
    df["total_delay_ms"] = pd.to_numeric(df["total_delay_ms"], errors="coerce").fillna(0.0)
    df["resultclassification"] = df["resultclassification"].str.upper().str.strip()

    return df[[
        "timeissued",
        "igp_id",
        "igp_delay_ms",
        "total_delay_ms",
        "resultclassification"
    ]]


def load_events():
    if not EVENTS_FILE.exists():
        return pd.DataFrame()

    df = pd.read_csv(EVENTS_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
    return df[["timestamp", "event_type", "severity", "description"]]

# DMM Isolation Forest (global)
def train_dmm_model(df_dmm):
    X = df_dmm[["delay"]].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    if MODEL_FILE.exists() and SCALER_FILE.exists():
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
    else:
        model = IsolationForest(contamination=0.1, random_state=42)
        model.fit(Xs)
        MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_FILE)
        joblib.dump(scaler, SCALER_FILE)

    return model, scaler


def predict_failed_dmm(df_dmm, model, scaler):
    df_fail = df_dmm[df_dmm["resultclassification"] == "FAIL"].copy()
    X = scaler.transform(df_fail[["delay"]].values)

    df_fail["anomaly_score"] = -model.decision_function(X)
    pred = model.predict(X)
    df_fail["is_anomaly"] = np.where(pred == -1, "ANOMALY", "NORMAL")

    return df_fail

# Per-IGP Isolation Forests
def train_per_igp_models(df_igp):
    IGP_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    igp_models = {}

    for igp_id, grp in df_igp.groupby("igp_id"):
        if len(grp) < MIN_IGP_SAMPLES:
            continue

        X = grp[["igp_delay_ms"]].values
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        model = IsolationForest(
            n_estimators=IGP_TREES,
            contamination=IGP_CONTAMINATION,
            random_state=42
        )
        model.fit(Xs)

        igp_models[igp_id] = (model, scaler)

        joblib.dump(
            (model, scaler),
            IGP_MODEL_DIR / f"igp_{igp_id}.pkl"
        )

    print(f"Trained {len(igp_models)} per-IGP models")
    return igp_models


def score_igp_models(df_igp, igp_models):
    df = df_igp.copy()
    df["igp_anomaly_score"] = 0.0
    df["igp_is_anomaly"] = "UNKNOWN"

    for igp_id, (model, scaler) in igp_models.items():
        mask = df["igp_id"] == igp_id
        if not mask.any():
            continue

        X = scaler.transform(df.loc[mask, ["igp_delay_ms"]].values)
        score = -model.decision_function(X)
        pred = model.predict(X)

        df.loc[mask, "igp_anomaly_score"] = score
        df.loc[mask, "igp_is_anomaly"] = np.where(pred == -1, "ANOMALY", "NORMAL")

    return df

# Baseline (PASS-only)
def compute_igp_baseline(df_igp):
    df_pass = df_igp[df_igp["resultclassification"] == "PASS"]

    baseline = (
        df_pass
        .groupby("igp_id")["igp_delay_ms"]
        .agg(igp_mean_ms="mean", igp_std_ms="std")
        .reset_index()
    )

    baseline["igp_std_ms"] = baseline["igp_std_ms"].replace(0, 1.0)
    return baseline

def correlate_root_cause(df_fail, df_igp, baseline):
    # Merge IGP failures with DMM failures using time tolerance
    df = pd.merge_asof(
        df_igp.sort_values("timeissued"),
        df_fail.sort_values("timeissued")[["timeissued", "delay", "anomaly_score", "is_anomaly"]],
        on="timeissued",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=5)   # <-- KEY FIX
    )

    # Keep only rows that matched a failure
    df = df[
        (df["resultclassification"] == "FAIL") &
        df["anomaly_score"].notna()
    ]

    if df.empty:
        print(" No IGP rows matched DMM failures after time alignment")
        return df

    df = df.merge(baseline, on="igp_id", how="left")

    df["z_score"] = (
        (df["igp_delay_ms"] - df["igp_mean_ms"]) /
        df["igp_std_ms"]
    ).fillna(0.0)

    df["combined_score"] = df["z_score"] * (1 + df["igp_anomaly_score"])

    idx = df.groupby("timeissued")["combined_score"].idxmax()

    root = df.loc[idx, [
        "timeissued",
        "igp_id",
        "igp_delay_ms",
        "z_score",
        "igp_anomaly_score",
        "delay",
        "anomaly_score",
        "is_anomaly"
    ]].rename(columns={
        "igp_id": "predicted_root_igp",
        "delay": "dmm_delay_ms"
    })

    return root.sort_values("anomaly_score", ascending=False)


def correlate_events_by_igp(df_root, df_events):
    if df_root.empty or df_events.empty:
        return df_root

    rows = []
    for _, r in df_root.iterrows():
        t = r["timeissued"]

        candidates = (
            df_events[df_events["timestamp"] <= t]
            .sort_values("timestamp")
            .tail(5)
        )

        if not candidates.empty:
            evt = candidates.iloc[-1]
            r["likely_event_type"] = evt["event_type"]
            r["likely_event_severity"] = evt["severity"]
            r["likely_event_description"] = evt["description"]
        else:
            r["likely_event_type"] = "NO_MATCH"
            r["likely_event_severity"] = "NO_MATCH"
            r["likely_event_description"] = "NO_MATCH"

        rows.append(r)
    return pd.DataFrame(rows)

    from sklearn.linear_model import LinearRegression

def compute_igp_trend(df_igp):
    trends = []

    df = df_igp[
        (df_igp["resultclassification"] == "PASS") &
        df_igp["timeissued"].notna()
    ].copy()

    if df.empty:
        return pd.DataFrame(
            columns=["igp_id", "trend_slope_ms_per_hour", "trend_label"]
        )

    # Convert time to numeric seconds
    df["time_num"] = df["timeissued"].astype("int64") / 1e9

    for igp_id, grp in df.groupby("igp_id"):
        if len(grp) < 5:
            continue

        grp = grp.sort_values("time_num")
        X = grp["time_num"].values.reshape(-1, 1)
        y = grp["igp_delay_ms"].values.astype(float)

        model = LinearRegression()
        model.fit(X, y)

        slope = model.coef_[0] * 3600.0  # ms/hour

        if slope > 1.0:
            label = "INCREASING"
        elif slope < -1.0:
            label = "DECREASING"
        else:
            label = "STABLE"

        trends.append({
            "igp_id": igp_id,
            "trend_slope_ms_per_hour": slope,
            "trend_label": label
        })

    return pd.DataFrame(trends)

# Main
def main():
    print("Running DMM + per-IGP Isolation Forest RCA pipeline")

    df_dmm = load_dmm()
    df_igp = load_igp()
    df_events = load_events()

    dmm_model, dmm_scaler = train_dmm_model(df_dmm)
    df_fail = predict_failed_dmm(df_dmm, dmm_model, dmm_scaler)

    igp_models = train_per_igp_models(df_igp)
    df_igp = score_igp_models(df_igp, igp_models)

    baseline = compute_igp_baseline(df_igp)
    trend_df = compute_igp_trend(df_igp)
    df_root = correlate_root_cause(df_fail, df_igp, baseline)

    #For deciding the trend. 
    if not trend_df.empty and not df_root.empty:
        df_root = df_root.merge(
            trend_df,
            left_on="predicted_root_igp",
            right_on="igp_id",
            how="left"
        ).drop(columns=["igp_id"], errors="ignore")

    df_root = correlate_events_by_igp(df_root, df_events)

    # Convert timeissued to epoch milliseconds for output
    df_root = df_root.copy()
    df_root["timeissued"] = (
        pd.to_datetime(df_root["timeissued"], errors="coerce").astype("int64") // 10**6
    )

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_root.to_csv(OUT_FILE, index=False)

    print(f"\nSaved results → {OUT_FILE}")
    print("\nTop 5 root causes:")
    print(df_root.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
