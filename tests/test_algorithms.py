import bisect
import math
import random

import pytest
from hypothesis import given, settings, strategies as st

from algorithms.linear import linear_search, linear_search_sorted, linear_range
from algorithms.binary import binary_search, lower_bound, binary_range, sorted_insert
from algorithms.bst import BST
from algorithms.rbtree import RBTree

keys_strategy = st.lists(st.integers(min_value=-1000, max_value=1000), max_size=300)


@given(keys_strategy, st.integers(min_value=-1100, max_value=1100))
def test_array_searches_agree(keys, q):
    arr = sorted(keys)
    present = q in arr
    first = bisect.bisect_left(arr, q)
    i, _ = linear_search(arr, q)
    assert (i != -1) == present and (not present or arr[i] == q)
    i, _ = linear_search_sorted(arr, q)
    assert (i != -1) == present and (not present or i == first)
    i, _ = binary_search(arr, q)
    assert (i != -1) == present and (not present or arr[i] == q)
    assert lower_bound(arr, q)[0] == first


@given(keys_strategy, st.integers(-1100, 1100), st.integers(0, 300))
def test_range_queries(keys, lo, width):
    hi = lo + width
    arr = sorted(keys)
    expected = [k for k in arr if lo <= k <= hi]
    assert binary_range(arr, lo, hi)[0] == expected
    assert sorted(linear_range(keys, lo, hi)[0]) == expected
    distinct = sorted(set(expected))
    for tree in (BST(), RBTree()):
        for k in keys:
            tree.insert(k)
        assert tree.range(lo, hi)[0] == distinct


def test_sorted_insert():
    arr = []
    for k in random.Random(1).sample(range(1000), 200):
        sorted_insert(arr, k)
    assert arr == sorted(arr)


ops_strategy = st.lists(
    st.tuples(st.sampled_from(["insert", "delete", "search"]), st.integers(0, 60)),
    max_size=400,
)


@settings(max_examples=300)
@given(ops_strategy)
def test_trees_match_dict_model(ops):
    model = {}
    bst, rb = BST(), RBTree()
    for op, k in ops:
        if op == "insert":
            model.setdefault(k, []).append(k)
            bst.insert(k, k)
            rb.insert(k, k)
        elif op == "delete":
            expected = model.pop(k, None) is not None
            assert bst.delete(k) == expected
            assert rb.delete(k) == expected
        else:
            assert bst.search(k)[0] == model.get(k)
            assert rb.search(k)[0] == model.get(k)
        rb.check_invariants()  # every red-black rule holds after EVERY operation
    assert list(bst.inorder()) == sorted(model)
    assert list(rb.inorder()) == sorted(model)
    assert len(bst) == len(rb) == len(model)


@pytest.mark.parametrize("n", [100, 1000, 10000])
def test_sorted_input_heights(n):
    bst, rb = BST(), RBTree()
    for k in range(n):          # asteroid numbers arrive in order
        bst.insert(k)
        rb.insert(k)
    assert bst.height() == n                          # degenerate: a linked list
    assert rb.height() <= 2 * math.log2(n + 1)        # red-black height bound
    assert bst.search(n - 1)[1] == n
    assert rb.search(n - 1)[1] <= 2 * math.log2(n + 1)

from algorithms.improvements import (
    bisect_search, interpolation_search, hybrid_search,
    LinearLearnedIndex, build_eytzinger, eytzinger_search, CachedSearch,
)


@given(keys_strategy, st.integers(-1100, 1100))
def test_improved_searches_agree(keys, q):
    arr = sorted(keys)
    present = q in arr
    for fn in (bisect_search, interpolation_search, hybrid_search):
        i, _ = fn(arr, q)
        assert (i != -1) == present and (not present or arr[i] == q), fn.__name__
    i, _ = LinearLearnedIndex(arr).search(q)
    assert (i != -1) == present and (not present or arr[i] == q)


@given(st.lists(st.integers(-1000, 1000), unique=True, max_size=300), st.integers(-1100, 1100))
def test_eytzinger(keys, q):
    arr = sorted(keys)
    eyt = build_eytzinger(arr)
    assert sorted(eyt[1:]) == arr
    k, _ = eytzinger_search(eyt, q)
    assert (k != -1) == (q in arr) and (k == -1 or eyt[k] == q)


def test_cache():
    arr = list(range(100))
    c = CachedSearch(lambda k: binary_search(arr, k))
    assert c.search(5)[0] == 5 and c.search(5) == (5, 0) and c.hits == 1

