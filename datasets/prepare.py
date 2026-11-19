"""Usage:  python -m datasets.prepare"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw/sbdb_numbered.csv")
OUT = Path("data/processed")


def load_raw():
    df = pd.read_csv(RAW, dtype={"pdes": str})
    # For numbered asteroids the primary designation IS the number ("433" = Eros).
    df["number"] = pd.to_numeric(df["pdes"], errors="coerce")
    df["H"] = pd.to_numeric(df["H"], errors="coerce")
    df["diameter"] = pd.to_numeric(df["diameter"], errors="coerce")
    return df.dropna(subset=["number"]).astype({"number": "int64"})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_raw()

    # Catalog order (by number) = the natural sorted arrival order.
    number = df.sort_values("number")["number"].to_numpy()
    H = df["H"].dropna().to_numpy()
    diameter = df["diameter"].dropna().to_numpy()

    np.save(OUT / "keys_number.npy", number)
    np.save(OUT / "keys_H.npy", H)
    np.save(OUT / "keys_diameter.npy", diameter)

    lines = [
        f"rows downloaded:          {len(df):,}",
        f"number keys:              {len(number):,} (min {number.min()}, max {number.max()})",
        f"H keys:                   {len(H):,} ({len(np.unique(H)):,} distinct values)",
        f"diameter keys:            {len(diameter):,} "
        f"({len(diameter) / len(df):.1%} of asteroids have a measured diameter)",
        f"diameter median / max km: {np.median(diameter):.2f} / {diameter.max():.1f}",
    ]
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()