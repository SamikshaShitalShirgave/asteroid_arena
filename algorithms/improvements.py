import bisect


def bisect_search(arr, key):
    i = bisect.bisect_left(arr, key)
    return (i if i < len(arr) and arr[i] == key else -1), None  # probes not observable


def interpolation_search(arr, key):
    """O(log log n) on evenly spread keys, O(n) worst case on skewed keys."""
    lo, hi, probes = 0, len(arr) - 1, 0
    while lo <= hi and arr[lo] <= key <= arr[hi]:
        if arr[hi] == arr[lo]:
            probes += 1
            return (lo if arr[lo] == key else -1), probes
        pos = lo + int((key - arr[lo]) * (hi - lo) / (arr[hi] - arr[lo]))
        probes += 1
        v = arr[pos]
        if v == key:
            while pos > 0 and arr[pos - 1] == key:   # first copy of duplicates
                pos -= 1
            return pos, probes
        if v < key:
            lo = pos + 1
        else:
            hi = pos - 1
    return -1, probes


def hybrid_search(arr, key, max_interp_steps=4):
    """Interpolation steps first, then plain binary search."""
    lo, hi, probes, steps = 0, len(arr) - 1, 0, 0
    while lo <= hi and arr[lo] <= key <= arr[hi]:
        if steps < max_interp_steps and arr[hi] != arr[lo]:
            pos = lo + int((key - arr[lo]) * (hi - lo) / (arr[hi] - arr[lo]))
            steps += 1
        else:
            pos = (lo + hi) // 2
        probes += 1
        v = arr[pos]
        if v == key:
            return pos, probes
        if v < key:
            lo = pos + 1
        else:
            hi = pos - 1
    return -1, probes


class LinearLearnedIndex:
    """Fit position ~ a*key + b, record the worst prediction error,
    then search only inside [prediction - err, prediction + err]."""

    def __init__(self, arr):
        self.arr = arr
        n = len(arr)
        if n == 0:
            self.a = self.b = 0.0
            self.err = 0
            return
        mean_k = sum(arr) / n
        mean_i = (n - 1) / 2
        cov = var = 0.0
        for i, k in enumerate(arr):
            dk = k - mean_k
            cov += dk * (i - mean_i)
            var += dk * dk
        self.a = cov / var if var else 0.0
        self.b = mean_i - self.a * mean_k
        err, prev = 0, object()
        for i, k in enumerate(arr):
            if k != prev:                    # with duplicates, target the first copy
                err = max(err, abs(i - self._predict(k)))
                prev = k
        self.err = err + 1

    def _predict(self, key):
        return int(self.a * key + self.b)

    def search(self, key):
        n = len(self.arr)
        p = self._predict(key)
        lo = max(0, p - self.err)
        hi = min(n, p + self.err + 1)
        i = bisect.bisect_left(self.arr, key, lo, hi)
        return (i if i < n and self.arr[i] == key else -1), None


def build_eytzinger(sorted_arr):
    """Breadth-first layout, 1-indexed: children of position k are 2k and 2k+1."""
    n = len(sorted_arr)
    out = [None] * (n + 1)
    it = iter(sorted_arr)
    stack, k = [], 1
    while stack or k <= n:                   # iterative in-order fill
        while k <= n:
            stack.append(k)
            k = 2 * k
        k = stack.pop()
        out[k] = next(it)
        k = 2 * k + 1
    return out


def eytzinger_search(eyt, key):
    n = len(eyt) - 1
    k, probes = 1, 0
    while k <= n:
        probes += 1
        v = eyt[k]
        if v == key:
            return k, probes
        k = 2 * k + (v < key)
    return -1, probes


class CachedSearch:
    """Wraps any search(key) -> (result, probes) with a small FIFO cache."""

    def __init__(self, search_fn, max_size=1024):
        self.search_fn = search_fn
        self.cache = {}
        self.max_size = max_size
        self.hits = 0

    def search(self, key):
        if key in self.cache:
            self.hits += 1
            return self.cache[key], 0
        result, probes = self.search_fn(key)
        if len(self.cache) >= self.max_size:
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = result
        return result, probes