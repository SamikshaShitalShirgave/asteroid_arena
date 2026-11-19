import numpy as np


def keys(kind, n, seed=0):
    rng = np.random.default_rng(seed)
    if kind == "sequential":       # like asteroid numbers
        return np.arange(1, n + 1)
    if kind == "uniform":
        return np.sort(rng.integers(0, 10 * n, n))
    if kind == "normal":
        return np.sort(rng.normal(0, 1, n))
    if kind == "lognormal":
        return np.sort(rng.lognormal(0, 1.5, n))
    if kind == "powerlaw":         # like asteroid diameters
        return np.sort(rng.pareto(1.5, n) + 1)
    raise ValueError(kind)


def insertion_order(sorted_keys, order, seed=0):
    """'sorted', 'random', or 'nearly' (sorted with 5% random swaps)."""
    rng = np.random.default_rng(seed)
    k = np.array(sorted_keys)
    if order == "sorted":
        return k
    if order == "random":
        return rng.permutation(k)
    if order == "nearly":
        k = k.copy()
        m = max(1, len(k) // 20)
        i, j = rng.integers(0, len(k), m), rng.integers(0, len(k), m)
        k[i], k[j] = k[j], k[i]
        return k
    raise ValueError(order)


def holdout(sorted_keys, frac=0.1, seed=0):
    """Returns (data_keys, miss_pool): held-out keys become realistic misses."""
    rng = np.random.default_rng(seed)
    k = np.asarray(sorted_keys)
    mask = rng.random(len(k)) < frac
    return k[~mask], k[mask]


def queries(sorted_keys, m, pattern="uniform", hit_rate=0.9, miss_pool=None, seed=0):
    """m query keys. pattern: 'uniform' or 'zipf' (a few keys asked very often)."""
    rng = np.random.default_rng(seed)
    k = np.asarray(sorted_keys)
    n = len(k)
    if pattern == "uniform":
        idx = rng.integers(0, n, m)
    elif pattern == "zipf":
        idx = np.minimum(rng.zipf(1.3, m) - 1, n - 1)
        idx = rng.permutation(n)[idx]           # popular keys spread over the range
    else:
        raise ValueError(pattern)
    q = k[idx].astype(float if k.dtype.kind == "f" else np.int64)
    miss = rng.random(m) > hit_rate
    pool = np.setdiff1d(np.asarray(miss_pool), k) if miss_pool is not None else np.array([])
    if len(pool):
        q[miss] = rng.choice(pool, miss.sum())
    else:
        q[miss] = k.max() + 1 + rng.integers(0, n, miss.sum())
    return q.tolist()