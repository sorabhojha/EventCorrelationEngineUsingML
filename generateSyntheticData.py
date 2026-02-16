#!/usr/bin/env python3
import random
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm

# Config
OUT_DIR = Path("synthetic_output")
OUT_FILE = OUT_DIR / "dmm_synthetic_flat.csv"
N_SAMPLES = 1_000_000
FAIL_RATIO = 0.10

random.seed(42)
np.random.seed(42)

from datetime import datetime, timedelta

def generate_lsp_dmm_data(n_samples: int, failure_ratio: float) -> pd.DataFrame:
    """
    Generate DMM-level data for a single LSP (LSP-1).
    Timestamps start from 12:00 AM of tomorrow (UTC) and increase monotonically.
    """
    n_fail = int(n_samples * failure_ratio)
    n_pass = n_samples - n_fail

    labels = np.array([1] * n_fail + [0] * n_pass)
    np.random.shuffle(labels)

    rows = []

    tomorrow_midnight = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    timestamp = int(tomorrow_midnight.timestamp() * 1000)

    for label in tqdm(labels, desc="Generating LSP DMM data"):
        # Increment timestamp by 1–5 seconds per row
        timestamp += random.randint(1000, 5000)

        if label == 0:
            delay = np.random.normal(loc=120, scale=10)
        else:
            delay = np.random.normal(loc=180, scale=25)

        delay = max(0, delay)
        minimum = max(0, delay - random.uniform(2, 6))
        maximum = delay + random.uniform(5, 10)
        samplecount = random.randint(10, 25) if label == 0 else random.randint(2, 6)
        suspect = bool(label == 1)
        result = "FAIL" if label == 1 else "PASS"

        rows.append({
            "lsp_id": "LSP-1",
            "timeissued": timestamp,
            "delay": round(delay, 2),
            "minimum": round(minimum, 2),
            "maximum": round(maximum, 2),
            "samplecount": samplecount,
            "suspect": suspect,
            "resultclassification": result
        })

    return pd.DataFrame(rows)

def main():
    OUT_DIR.mkdir(exist_ok=True)
    df = generate_lsp_dmm_data(N_SAMPLES, FAIL_RATIO)
    df.to_csv(OUT_FILE, index=False)

    print("\n  Synthetic DMM data generated for LSP-1")
    print(f"   Output : {OUT_FILE}")
    print(f"   Rows   : {len(df)} (FAIL ≈ {FAIL_RATIO*100:.0f}%)")
    print("\n  Sample rows:")
    print(df.head().to_string(index=False))

if __name__ == "__main__":
    main()
