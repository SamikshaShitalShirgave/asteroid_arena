"""Download numbered asteroids from the JPL Small-Body Database Query API.

Usage:
    python -m datasets.fetch_sbdb              # full catalog (takes a few minutes)
    python -m datasets.fetch_sbdb --max 50000  # quick test download
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import requests

API = "https://ssd-api.jpl.nasa.gov/sbdb_query.api"
FIELDS = ["spkid", "pdes", "name", "H", "diameter", "albedo", "neo", "pha", "first_obs"]
OUT = Path("data/raw/sbdb_numbered.csv")


def fetch_page(offset, limit):
    params = {
        "fields": ",".join(FIELDS),
        "sb-kind": "a",       # asteroids only
        "sb-ns": "n",         # numbered objects only
        "sort": "spkid",      # stable order so pages don't overlap
        "limit": limit,
        "limit-from": offset,
    }
    for attempt in range(5):
        r = requests.get(API, params=params, timeout=120)
        if r.status_code == 503:            # API overloaded: wait and retry
            time.sleep(10 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("API kept returning 503; try again later")


def parse(payload):
    """The API returns {"fields": [...], "data": [[...], ...]}."""
    return pd.DataFrame(payload.get("data", []), columns=payload["fields"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", type=int, default=100_000)
    ap.add_argument("--max", type=int, default=None, help="stop after this many rows")
    args = ap.parse_args()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    frames, offset = [], 0
    while True:
        limit = args.page if args.max is None else min(args.page, args.max - offset)
        if limit <= 0:
            break
        df = parse(fetch_page(offset, limit))
        if df.empty:
            break
        frames.append(df)
        offset += len(df)
        print(f"downloaded {offset:,} rows")
        if len(df) < limit:
            break
        time.sleep(1)                        # be polite to a free public service

    data = pd.concat(frames, ignore_index=True)
    data.to_csv(OUT, index=False)
    print(f"saved {len(data):,} asteroids to {OUT}")


if __name__ == "__main__":
    main()