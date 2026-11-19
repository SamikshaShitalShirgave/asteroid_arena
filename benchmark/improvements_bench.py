import statistics
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

from algorithms.binary import binary_search
from algorithms.improvements import (
    bisect_search, interpolation_search, hybrid_search,
    LinearLearnedIndex, build_eytzinger, eytzinger_search,
)
from benchmark.experiments import load_keys
from benchmark.runner import time_queries, mean_probes
from datasets.synthetic import holdout, queries

RESULTS = Path("results")


def improvement_experiment(n=1_000_000, m=5_000):
    rows = []
    for key_name in ["number", "H", "diameter"]:
        all_keys, source = load_keys(key_name)
        idx = np.linspace(0, len(all_keys) - 1, min(n, len(all_keys))).astype(int)
        data, pool = holdout(all_keys[idx], 0.1, seed=1)
        arr = data.tolist()
        qs = queries(data, m, "uniform", 0.9, pool, seed=1)
        learned = LinearLearnedIndex(arr)
        eyt = build_eytzinger(sorted(set(arr)))
        variants = {
            "binary (hand-written)": lambda k: binary_search(arr, k),
            "bisect (C)": lambda k: bisect_search(arr, k),
            "interpolation": lambda k: interpolation_search(arr, k),
            "hybrid": lambda k: hybrid_search(arr, k),
            "learned index": learned.search,
            "eytzinger": lambda k: eytzinger_search(eyt, k),
        }
        for name, fn in variants.items():
            samples = time_queries(fn, qs, repeats=15)
            row = {"key": key_name, "source": source, "n": len(arr), "variant": name,
                   "median_ns": statistics.median(samples),
                   "mean_probes": mean_probes(fn, qs)}
            if name == "learned index":
                row["max_error"] = learned.err
            rows.append(row)
            print(f"{key_name:8} {name:22} {row['median_ns']:>8,.0f} ns  probes={row['mean_probes']:.1f}")
    return pd.DataFrame(rows)


class NodeNoSlots:  # same fields as bst.Node, but without __slots__
    def __init__(self, key, value):
        self.key, self.values, self.left, self.right = key, [value], None, None


def memory_experiment(n=100_000):
    from algorithms.bst import Node
    from algorithms.rbtree import RBTree
    rows = []
    for label, make in [
        ("python list (sorted array)", lambda: list(range(n))),
        ("BST nodes with __slots__", lambda: [Node(i, None) for i in range(n)]),
        ("BST nodes without __slots__", lambda: [NodeNoSlots(i, None) for i in range(n)]),
    ]:
        tracemalloc.start()
        obj = make()
        cur, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rows.append({"structure": label, "n": n, "bytes_per_key": cur / n})
        del obj
    tracemalloc.start()
    t = RBTree()
    for i in range(n):
        t.insert(i)
    cur, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rows.append({"structure": "red-black tree", "n": n, "bytes_per_key": cur / n})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


def main():
    RESULTS.mkdir(exist_ok=True)
    improvement_experiment().to_csv(RESULTS / "improvements.csv", index=False)
    memory_experiment().to_csv(RESULTS / "memory.csv", index=False)


if __name__ == "__main__":
    main()