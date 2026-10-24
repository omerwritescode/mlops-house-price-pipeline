"""Replay the incoming batches through the live API so Grafana shows real operational metrics.

    python scripts/replay_traffic.py --api http://localhost:8000 --rps 20
Sends every listing of batches 1..3 to POST /predict (one request per listing, paced at --rps),
plus a small share of deliberately invalid requests so the 4xx path is visible too.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["Area_sqft", "Bedrooms", "Bathrooms", "Age_Years", "Distance_to_Center_km", "Neighborhood_Score",
            "Has_Parking", "Property_Condition", "City_Zone"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--file", default=str(ROOT / "data/raw/capstone_incoming_batches.csv"))
    ap.add_argument("--rps", type=float, default=20.0)
    ap.add_argument("--invalid-share", type=float, default=0.02)
    args = ap.parse_args()

    df = pd.read_csv(args.file)
    s = requests.Session()
    ok = bad = 0
    for batch, part in df.groupby("Batch"):
        t0 = time.time()
        for _, row in part.iterrows():
            rec = {k: (None if pd.isna(row[k]) else row[k].item() if hasattr(row[k], "item") else row[k])
                   for k in FEATURES}
            if random.random() < args.invalid_share:
                rec["Area_sqft"] = -5  # rejected by the schema -> 422
            r = s.post(f"{args.api}/predict", json=rec, timeout=10)
            ok += r.status_code == 200
            bad += r.status_code != 200
            time.sleep(max(0.0, 1.0 / args.rps))
        print(f"batch {batch}: {len(part)} requests in {time.time() - t0:.1f}s")
    print(f"done: {ok} ok, {bad} non-200. Open Grafana at http://localhost:3000")


if __name__ == "__main__":
    main()
