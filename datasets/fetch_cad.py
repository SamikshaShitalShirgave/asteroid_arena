import datetime as dt
import json
from pathlib import Path

import pandas as pd
import requests

API = "https://ssd-api.jpl.nasa.gov/cad.api"
CACHE = Path("data/raw/cad_snapshot.json")   # offline fallback for the demo


def upcoming_approaches(days=30, dist_max="0.05", use_cache_on_error=True):
    today = dt.date.today()
    params = {
        "date-min": today.isoformat(),
        "date-max": (today + dt.timedelta(days=days)).isoformat(),
        "dist-max": dist_max,   # astronomical units
        "sort": "date",
    }
    try:
        r = requests.get(API, params=params, timeout=30)
        r.raise_for_status()
        payload = r.json()
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(payload))
    except (requests.RequestException, ValueError):
        if not (use_cache_on_error and CACHE.exists()):
            raise
        payload = json.loads(CACHE.read_text())
    df = pd.DataFrame(payload.get("data", []), columns=payload["fields"])
    if not df.empty:
        df["dist"] = df["dist"].astype(float)
        df["jd"] = df["jd"].astype(float)   # Julian date = numeric time key
    return df


if __name__ == "__main__":
    print(upcoming_approaches().head(20).to_string())