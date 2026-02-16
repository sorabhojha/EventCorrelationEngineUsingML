#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from tqdm import tqdm

OUT_DIR = Path("synthetic_output")
DMM_FILE = OUT_DIR / "dmm_synthetic_flat.csv"
OUT_FILE = OUT_DIR / "network_events.csv"

N_EVENTS = 2000
EVENT_TYPES = [
    "LINK_FLAP", "ROUTE_CHANGE", "INTERFACE_ERROR", "MAINTENANCE",
    "NODE_REBOOT", "CONFIG_UPDATE", "POWER_GLITCH"
]
SEVERITIES = ["INFO", "MINOR", "MAJOR", "CRITICAL"]
LINKS = ["R1-R2", "R2-R3", "R3-R4", "R4-R5", "R5-R6", "R6-R7", "R7-R8"]

random.seed(42)
np.random.seed(42)

def generate_events_relative_to_dmm():
    if not DMM_FILE.exists():
        raise FileNotFoundError(f"{DMM_FILE} not found. Please run the DMM generator first.")

    df_dmm = pd.read_csv(DMM_FILE)
    t_min = int(df_dmm["timeissued"].min())
    t_max = int(df_dmm["timeissued"].max())

    # Define event window: 1 hour before DMM start → right up to DMM start
    start_ms = t_min - (60 * 60 * 1000)   # 1 hour before
    end_ms = t_min - (10 * 1000)          # up to 10 seconds before first DMM record

    timestamps = np.linspace(start_ms, end_ms, N_EVENTS).astype(int)

    rows = []
    for ts in tqdm(timestamps, desc="Generating Pre-DMM Events"):
        event_type = random.choice(EVENT_TYPES)
        severity = random.choices(SEVERITIES, weights=[0.4, 0.3, 0.2, 0.1])[0]
        link = random.choice(LINKS)

        if event_type == "LINK_FLAP":
            desc = f"Link {link} flapped"
        elif event_type == "ROUTE_CHANGE":
            desc = f"IGP route change observed on {link}"
        elif event_type == "INTERFACE_ERROR":
            desc = f"High error rate detected on {link}"
        elif event_type == "MAINTENANCE":
            desc = f"Scheduled maintenance window active on {link}"
        elif event_type == "NODE_REBOOT":
            desc = f"Node connected via {link} rebooted unexpectedly"
        elif event_type == "CONFIG_UPDATE":
            desc = f"Configuration updated for interface on {link}"
        elif event_type == "POWER_GLITCH":
            desc = f"Power glitch detected affecting {link}"
        else:
            desc = f"Generic network event on {link}"

        rows.append({
            "timestamp": ts,
            "event_type": event_type,
            "severity": severity,
            "description": desc
        })

    df_events = pd.DataFrame(rows)
    return df_events


def main():
    OUT_DIR.mkdir(exist_ok=True)
    df_events = generate_events_relative_to_dmm()
    df_events.to_csv(OUT_FILE, index=False)

    print("\n Generated pre-DMM network events aligned by timestamp (ms).")
    print(f"   Output: {OUT_FILE}")
    print(f"   Rows  : {len(df_events)}")
    print(f"   Range : {df_events['timestamp'].min()} → {df_events['timestamp'].max()}")
    print("\nSample events:")
    print(df_events.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
