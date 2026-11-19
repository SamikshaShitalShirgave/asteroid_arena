import gc
import random
import statistics
import time

from algorithms.linear import linear_search
from algorithms.binary import binary_search, sorted_insert
from algorithms.bst import BST
from algorithms.rbtree import RBTree

STRUCTURES = ["linear", "binary", "bst", "rbtree"]


def build(name, keys):
    """keys = list in INSERTION order. Returns (structure, build_seconds)."""
    t0 = time.perf_counter()
    if name == "linear":
        s = list(keys)
    elif name == "binary":
        s = sorted(keys)                     # the cost binary search pays up front
    elif name in ("bst", "rbtree"):
        s = BST() if name == "bst" else RBTree()
        for k in keys:
            s.insert(k)
    else:
        raise ValueError(name)
    return s, time.perf_counter() - t0


def search_fn(name, s):
    if name == "linear":
        return lambda k: linear_search(s, k)
    if name == "binary":
        return lambda k: binary_search(s, k)
    return s.search


def bootstrap_ci(samples, reps=2000, seed=0):
    """95% bootstrap confidence interval of the median."""
    rng = random.Random(seed)
    n = len(samples)
    meds = sorted(statistics.median(rng.choices(samples, k=n)) for _ in range(reps))
    return meds[int(0.025 * reps)], meds[int(0.975 * reps)]


def time_queries(fn, queries, repeats=30):
    """Time the whole query batch `repeats` times; return per-query times (ns)."""
    for q in queries[: min(len(queries), 100)]:   # warm-up
        fn(q)
    per_query = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            t0 = time.perf_counter_ns()
            for q in queries:
                fn(q)
            per_query.append((time.perf_counter_ns() - t0) / len(queries))
    finally:
        if gc_was_enabled:
            gc.enable()
    return per_query


def mean_probes(fn, queries):
    total = 0
    for q in queries:
        _, p = fn(q)
        total += p or 0
    return total / len(queries)


def benchmark_search(name, keys_in_order, queries, repeats=30):
    s, build_s = build(name, keys_in_order)
    fn = search_fn(name, s)
    samples = time_queries(fn, queries, repeats)
    lo, hi = bootstrap_ci(samples)
    row = {
        "structure": name, "n": len(keys_in_order), "queries": len(queries),
        "build_s": build_s, "median_ns": statistics.median(samples),
        "ci_low_ns": lo, "ci_high_ns": hi, "mean_probes": mean_probes(fn, queries),
    }
    if name in ("bst", "rbtree"):
        row["height"] = s.height()
    return row


def benchmark_mixed(name, initial_keys, new_keys, queries, write_ratio, seed=0):
    """Dynamic workload: interleave inserts (fraction = write_ratio) and searches."""
    s, _ = build(name, initial_keys)
    if name == "binary":
        insert = lambda k: sorted_insert(s, k)
    elif name == "linear":
        insert = s.append
    else:
        insert = s.insert
    search = search_fn(name, s)
    rng = random.Random(seed)
    ops, ni, qi = [], iter(new_keys), iter(queries)
    for _ in range(len(new_keys) + len(queries)):
        if rng.random() < write_ratio:
            k = next(ni, None)
            if k is not None:
                ops.append((insert, k))
                continue
        q = next(qi, None)
        if q is not None:
            ops.append((search, q))
    gc.disable()
    t0 = time.perf_counter()
    for f, k in ops:
        f(k)
    elapsed = time.perf_counter() - t0
    gc.enable()
    return {"structure": name, "n": len(initial_keys), "write_ratio": write_ratio,
            "ops": len(ops), "us_per_op": elapsed / len(ops) * 1e6}