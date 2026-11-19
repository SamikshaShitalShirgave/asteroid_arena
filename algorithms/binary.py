def binary_search(arr, key):
    """Classic iterative binary search. Returns (index or -1, probes)."""
    lo, hi = 0, len(arr) - 1
    probes = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        v = arr[mid]
        probes += 1
        if v == key:
            return mid, probes
        if v < key:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1, probes


def lower_bound(arr, key):
    """First index i with arr[i] >= key (same as bisect.bisect_left)."""
    lo, hi = 0, len(arr)
    probes = 0
    while lo < hi:
        mid = (lo + hi) // 2
        probes += 1
        if arr[mid] < key:
            lo = mid + 1
        else:
            hi = mid
    return lo, probes


def upper_bound(arr, key):
    """First index i with arr[i] > key (same as bisect.bisect_right)."""
    lo, hi = 0, len(arr)
    probes = 0
    while lo < hi:
        mid = (lo + hi) // 2
        probes += 1
        if arr[mid] <= key:
            lo = mid + 1
        else:
            hi = mid
    return lo, probes


def binary_range(arr, lo_key, hi_key):
    """All values in [lo_key, hi_key]: two searches + a slice. O(log n + k)."""
    i, p1 = lower_bound(arr, lo_key)
    j, p2 = upper_bound(arr, hi_key)
    return arr[i:j], p1 + p2


def sorted_insert(arr, key):
    """Keep the list sorted: O(log n) to find the place, O(n) to shift. Returns probes."""
    i, probes = upper_bound(arr, key)
    arr.insert(i, key)
    return probes