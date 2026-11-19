import argparse
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark.runner import STRUCTURES, benchmark_search, benchmark_mixed
from datasets.synthetic import keys as synth_keys, insertion_order, holdout, queries

PROCESSED = Path("data/processed")
RESULTS = Path("results")
BST_SORTED_MAX_N = 20_000          # sorted-input BST build is Theta(n^2)
LINEAR_QUERY_BUDGET = 20_000_000   # n * queries for linear search


def load_keys(name):
    f = PROCESSED / f"keys_{name}.npy"
    if f.exists():
        return np.sort(np.load(f)), "asteroids"
    synthetic = {"number": "sequential", "H": "normal", "diameter": "powerlaw"}[name]
    return synth_keys(synthetic, 1_000_000), f"synthetic-{synthetic}"


def sizes(quick, max_n):
    grid = [100, 300, 1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000]
    if quick:
        grid = [100, 1_000, 10_000, 100_000]
    return [n for n in grid if n <= max_n]


def scaling_experiment(quick):
    rows = []
    m = 500 if quick else 2_000
    repeats = 7 if quick else 30
    for key_name in ["number", "H", "diameter"]:
        all_keys, source = load_keys(key_name)
        for n in sizes(quick, len(all_keys)):
            idx = np.linspace(0, len(all_keys) - 1, n).astype(int)
            data, pool = holdout(all_keys[idx], 0.1, seed=n)
            qs = queries(data, m, "uniform", 0.9, pool, seed=n)
            for order in ["sorted", "random"]:
                ins = insertion_order(data, order, seed=n).tolist()
                for s in STRUCTURES:
                    if s == "bst" and order == "sorted" and n > BST_SORTED_MAX_N:
                        continue
                    if s in ("linear", "binary") and order == "random":
                        continue  # arrays: insertion order doesn't change the search
                    q = qs
                    if s == "linear":
                        q = qs[: max(20, min(m, LINEAR_QUERY_BUDGET // n))]
                    row = benchmark_search(s, ins, q, repeats)
                    row.update(key=key_name, source=source, order=order)
                    rows.append(row)
                    print(f"{key_name:8} n={n:>9,} {order:6} {s:7} "
                          f"{row['median_ns']:>12,.0f} ns  probes={row['mean_probes']:.1f}")
    return pd.DataFrame(rows)


def mixed_experiment(quick):
    rows = []
    n = 20_000 if quick else 200_000
    rng = np.random.default_rng(1)
    k = rng.permutation(synth_keys("uniform", 2 * n, seed=1))
    initial, new = k[:n].tolist(), k[n:].tolist()
    qs = queries(np.sort(initial), 20_000, "uniform", 1.0, seed=2)
    for ratio in [0.0, 0.05, 0.5, 0.95]:
        for s in ["binary", "bst", "rbtree"]:
            row = benchmark_mixed(s, initial, new[:20_000], qs, ratio)
            rows.append(row)
            print(f"mixed n={n:,} writes={ratio:.0%} {s:7} {row['us_per_op']:.2f} us/op")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    env = f"Python {sys.version.split()[0]} | {platform.platform()} | {platform.processor()}"
    (RESULTS / "environment.txt").write_text(env + "\n")
    print(env)
    scaling_experiment(args.quick).to_csv(RESULTS / "scaling.csv", index=False)
    mixed_experiment(args.quick).to_csv(RESULTS / "mixed.csv", index=False)
    print("saved results/scaling.csv and results/mixed.csv")


if __name__ == "__main__":
    main()