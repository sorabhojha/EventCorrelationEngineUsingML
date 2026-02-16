#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

DMM_FILE = Path("synthetic_output/dmm_synthetic_flat.csv")
OUT_FILE = Path("synthetic_output/lsp_from_dmm.csv")
np.random.seed(42)

IGP_HOPS = 4               # fixed number of hops
FAILURE_VARIANCE = 0.25    # more variation for failures
SUCCESS_VARIANCE = 0.05    # less variation for success

def generate_lsp_data(df_dmm: pd.DataFrame) -> pd.DataFrame:
    records = []

    for _, row in tqdm(df_dmm.iterrows(), total=len(df_dmm), desc="Generating LSP data"):
        test_time = int(row["timeissued"])
        total_delay = float(row["delay"])
        result = row["resultclassification"]

        igp_links = [f"IGP{i+1}" for i in range(IGP_HOPS)]
        mean_delay = total_delay / IGP_HOPS
        variance = FAILURE_VARIANCE if result == "FAIL" else SUCCESS_VARIANCE

        # Sorabh :: Create LSP Details
        igp_delays = np.random.normal(loc=mean_delay, scale=mean_delay * variance, size=IGP_HOPS)

        # Replace negatives, normalize to total_delay
        igp_delays = np.maximum(igp_delays, 1)
        igp_delays *= total_delay / igp_delays.sum()

        # Convert to int
        igp_delays = np.round(igp_delays).astype(int)

        diff = int(round(total_delay)) - igp_delays.sum()
        if diff != 0:
            igp_delays[0] += diff  # adjust first hop slightly

        records.append({
            "timeissued": test_time,
            "lsp_name": "LSP-1",
            "lsp_links": ";".join(igp_links),
            "lsp_delays": ";".join(map(str, igp_delays)),
            "total_delay_ms": int(round(total_delay)),
            "resultclassification": result
        })

    return pd.DataFrame(records)

def main():
    if not DMM_FILE.is_file(): 
        raise FileNotFoundError(f"DMM file not found: {DMM_FILE}")

    print("\nLoading DMM data …")
    df_dmm = pd.read_csv(DMM_FILE)

    print(f"Generating LSP dataset for {len(df_dmm)} DMM tests …")
    df_lsp = generate_lsp_data(df_dmm)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_lsp.to_csv(OUT_FILE, index=False)

    print(f"\n Generated {len(df_lsp)} LSP rows")
    print(f" Output saved → {OUT_FILE}")
    print("\nSample data:")
    print(df_lsp.head().to_string(index=False))


if __name__ == "__main__":
    main()
