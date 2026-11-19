def linear_search(arr, key):
    """Scan left to right. O(n) worst case, works on unsorted data."""
    probes = 0
    for i, v in enumerate(arr):
        probes += 1
        if v == key:
            return i, probes
    return -1, probes


def linear_search_sorted(arr, key):
    """Early exit on SORTED data: stop once v >= key.
    Misses then cost about n/2 probes on average instead of n."""
    probes = 0
    for i, v in enumerate(arr):
        probes += 1
        if v >= key:
            return (i if v == key else -1), probes
    return -1, probes


def linear_range(arr, lo, hi):
    """All values with lo <= v <= hi. Must look at every element: Theta(n)."""
    return [v for v in arr if lo <= v <= hi], len(arr)