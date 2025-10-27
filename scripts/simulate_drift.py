import pandas as pd
import numpy as np
import os

INPUT_FILE = "data/processed/interactions.csv"
DRIFT_FILE = "data/processed/interactions_drifted.csv"

def simulate_drift():
    df = pd.read_csv(INPUT_FILE)

    # Simulate drift by:
    # - Reducing activity for popular items
    # - Dropping highly active users
    df = df[df["item_idx"] < df["item_idx"].quantile(0.75)]
    df = df[df["user_idx"] < df["user_idx"].quantile(0.75)]

    # Slight noise on ratings
    df["rating"] = df["rating"] + np.random.normal(0, 0.2, size=len(df))
    df["rating"] = df["rating"].clip(1, 5)

    df.to_csv(DRIFT_FILE, index=False)
    print(f"Simulated drifted dataset saved to {DRIFT_FILE}")

if __name__ == "__main__":
    simulate_drift()
