"""Asteroid Arena.  Run from the project root:  streamlit run app/arena.py"""
import random
import sys
import time
from pathlib import Path
import base64

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from algorithms.binary import binary_search
from algorithms.bst import BST
from algorithms.improvements import (
    bisect_search, interpolation_search, hybrid_search, LinearLearnedIndex, CachedSearch,
)
from algorithms.rbtree import RBTree, RED
from benchmark.runner import build, search_fn
from datasets.synthetic import keys as synth_keys, insertion_order, queries

st.set_page_config(page_title="Asteroid Arena", page_icon="☄️", layout="wide")
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
COLORS = {"linear": "#d62728", "binary": "#1f77b4", "bst": "#ff7f0e", "rbtree": "#9467bd"}

def set_background(image_path, darken=0.55):
    """Full-page background image with a dark overlay so text stays readable."""
    path = Path(image_path)
    if not path.exists():
        return
    b64 = base64.b64encode(path.read_bytes()).decode()
    mime = "png" if path.suffix.lower() == ".png" else "jpeg"
    st.markdown(f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image:
            linear-gradient(rgba(0,0,0,{darken}), rgba(0,0,0,{darken})),
            url("data:image/{mime};base64,{b64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    [data-testid="stHeader"] {{ background: rgba(0,0,0,0); }}
    [data-testid="stSidebar"] > div:first-child {{ background: rgba(14,17,23,0.85); }}
    </style>
    """, unsafe_allow_html=True)


set_background(ROOT / "app" / "assets" / "space.jpg")

@st.cache_data
def load_keys(name):
    f = PROCESSED / f"keys_{name}.npy"
    if f.exists():
        return np.sort(np.load(f)), "NASA/JPL asteroid catalog"
    synthetic = {"number": "sequential", "H": "normal", "diameter": "powerlaw"}[name]
    return synth_keys(synthetic, 1_000_000), f"synthetic stand-in ({synthetic})"


def sample(keys, n):
    return keys[np.linspace(0, len(keys) - 1, n).astype(int)]


def to_dot(tree, highlight_path=(), max_nodes=63):
    """Graphviz DOT text for the top of a BST or RBTree (rendered in the browser)."""
    nil = getattr(tree, "NIL", None)
    is_rb = nil is not None
    lines = ["digraph{node[shape=circle,style=filled,fontcolor=white,fontsize=10,width=0.4]",
             "edge[arrowsize=0.5]"]
    count, queue = 0, [tree.root] if tree.root not in (None, nil) else []
    while queue and count < max_nodes:
        node = queue.pop(0)
        count += 1
        color = ("#c0392b" if node.color == RED else "#222222") if is_rb else "#e67e22"
        pen = ',penwidth=4,color="#f1c40f"' if node.key in highlight_path else ""
        label = f"{node.key:g}" if isinstance(node.key, float) else str(node.key)
        lines.append(f'"{id(node)}"[label="{label}",fillcolor="{color}"{pen}]')
        for child in (node.left, node.right):
            if child not in (None, nil):
                lines.append(f'"{id(node)}"->"{id(child)}"')
                queue.append(child)
    lines.append("}")
    return "\n".join(lines)


def search_path(tree, key):
    nil = getattr(tree, "NIL", None)
    path, node = [], tree.root
    while node not in (None, nil):
        path.append(node.key)
        if key == node.key:
            break
        node = node.left if key < node.key else node.right
    return path


st.title("☄️ Asteroid Arena")
st.caption("Linear search vs binary search vs binary search tree vs red-black tree, on real asteroid data")
key_name = st.sidebar.selectbox("Key to search by", ["number", "H", "diameter"],
                                format_func={"number": "Asteroid number", "H": "Absolute magnitude H",
                                             "diameter": "Diameter (km)"}.get)
all_keys, source = load_keys(key_name)
st.sidebar.write(f"**Data:** {source}  \n**Keys available:** {len(all_keys):,}")

tab_live, tab_theater, tab_race, tab_scaling, tab_upgrade = st.tabs(
    ["🌍 Incoming", "🌳 Tree Theater", "🏁 The Race", "📈 Scaling Lab", "⚡ Upgrade Lab + Verdict"])

# ---- Tab 1: live close approaches
with tab_live:
    st.subheader("Asteroids passing Earth in the next 30 days")
    if st.button("Fetch latest close approaches"):
        from datasets.fetch_cad import upcoming_approaches
        try:
            st.session_state.cad = upcoming_approaches(days=30)
        except Exception as e:  # no network and no snapshot yet
            st.error(f"Could not fetch data and no offline snapshot exists: {e}")
    cad = st.session_state.get("cad")
    if cad is not None and not cad.empty:
        st.dataframe(cad[["des", "cd", "dist", "v_rel", "h"]], hide_index=True)
        bst, rb = BST(), RBTree()
        cost_bst = cost_rb = 0
        for jd in cad["jd"]:                       # date order = sorted keys
            cost_bst += bst.insert(jd)
            cost_rb += rb.insert(jd)
        st.write(f"Inserted {len(cad)} approaches, keyed by time, in arrival order.")
        st.table(pd.DataFrame({
            "structure": ["BST", "Red-black tree"],
            "probes for all inserts": [cost_bst, cost_rb],
            "height": [bst.height(), rb.height()],
        }))
        st.info("Time-ordered arrival is sorted input: the BST's height equals the number of "
                "approaches, while the red-black tree stays balanced.")

# ---- Tab 2: Tree Theater
with tab_theater:
    st.subheader("Insert keys one by one and watch the shape")
    c1, c2, c3 = st.columns(3)
    order = c1.radio("Insertion order", ["sorted (catalog order)", "random"], horizontal=True)
    n_show = c2.slider("Keys to insert", 1, 63, 15)
    find = c3.number_input("Highlight search for key", value=float(n_show))
    base = sorted(set(int(k) if key_name == "number" else round(float(k), 2)
                      for k in sample(all_keys, 200)))[:63]
    ins = base if order.startswith("sorted") else random.Random(7).sample(base, len(base))
    bst, rb = BST(), RBTree()
    for k in ins[:n_show]:
        bst.insert(k)
        rb.insert(k)
    left, right = st.columns(2)
    with left:
        st.markdown(f"**BST** · height {bst.height()} · search probes {bst.search(find)[1]}")
        st.graphviz_chart(to_dot(bst, search_path(bst, find)))
    with right:
        bound = 2 * np.log2(n_show + 1)
        st.markdown(f"**Red-black tree** · height {rb.height()} (bound {bound:.1f}) · "
                    f"search probes {rb.search(find)[1]} · rotations {rb.rotations}")
        st.graphviz_chart(to_dot(rb, search_path(rb, find)))

# ---- Tab 3: The Race
with tab_race:
    st.subheader("Same queries, four algorithms")
    c1, c2, c3 = st.columns(3)
    n = c1.select_slider("Data size n", [100, 1_000, 10_000, 100_000, 1_000_000], value=10_000)
    m = c2.select_slider("Number of queries", [10, 100, 1_000], value=100)
    order = c3.radio("Tree insertion order", ["random", "sorted"], horizontal=True)
    if st.button("🏁 Start the race"):
        data = sample(all_keys, n)
        qs = queries(data, m, "uniform", 0.9, seed=1)
        ins = insertion_order(data, order, seed=1).tolist()
        rows, bars = [], st.empty()
        for name in ["linear", "binary", "bst", "rbtree"]:
            if name == "bst" and order == "sorted" and n > 20_000:
                rows.append({"algorithm": name, "ms per query": None, "avg probes": None,
                             "note": "skipped: building would take hours (Θ(n²))"})
                continue
            with st.spinner(f"{name} is running..."):
                s, build_s = build(name, ins)
                fn = search_fn(name, s)
                t0 = time.perf_counter()
                probes = sum(fn(q)[1] for q in qs)
                ms = (time.perf_counter() - t0) / m * 1e3
            rows.append({"algorithm": name, "ms per query": ms, "avg probes": probes / m,
                         "build s": build_s, "note": ""})
            df = pd.DataFrame([r for r in rows if r["ms per query"] is not None])
            bars.plotly_chart(px.bar(df, x="algorithm", y="ms per query", color="algorithm",
                                     color_discrete_map=COLORS, log_y=True, text_auto=".4f"),
                              key=f"race-{name}")
        st.dataframe(pd.DataFrame(rows), hide_index=True)

# ---- Tab 4: Scaling Lab
with tab_scaling:
    st.subheader("How running time grows with n")
    f = RESULTS / "scaling.csv"
    if not f.exists():
        st.warning("Run `python -m benchmark.experiments` first to create results/scaling.csv.")
    else:
        df = pd.read_csv(f)
        df = df[df["key"] == key_name].copy()
        df["series"] = df["structure"] + " (" + df["order"] + ")"
        fig = px.line(df.sort_values("n"), x="n", y="median_ns", color="series", markers=True,
                      log_x=True, log_y=True, error_y=df["ci_high_ns"] - df["median_ns"],
                      labels={"median_ns": "ns per search (median)", "n": "n (keys)"})
        st.plotly_chart(fig)
        fits = RESULTS / "fits.csv"
        if fits.exists():
            st.dataframe(pd.read_csv(fits).query("key == @key_name"), hide_index=True)

# ---- Tab 5: Upgrade Lab + Verdict
with tab_upgrade:
    st.subheader("Can we beat binary search?")
    n = st.select_slider("n", [10_000, 100_000, 1_000_000], value=100_000, key="up_n")
    if st.button("Run improvements"):
        arr = sample(all_keys, n).tolist()
        qs = queries(np.array(arr), 300, "uniform", 0.9, seed=3)
        learned = LinearLearnedIndex(arr)
        variants = {
            "binary (Python)": lambda k: binary_search(arr, k),
            "bisect (C)": lambda k: bisect_search(arr, k),
            "interpolation": lambda k: interpolation_search(arr, k),
            "hybrid": lambda k: hybrid_search(arr, k),
            "learned index": learned.search,
        }
        rows = []
        for name, fn in variants.items():
            t0 = time.perf_counter()
            for q in qs:
                fn(q)
            rows.append({"technique": name, "µs per query": (time.perf_counter() - t0) / len(qs) * 1e6})
        st.plotly_chart(px.bar(pd.DataFrame(rows), x="technique", y="µs per query", log_y=True))
        st.caption(f"Learned index error bound: {learned.err:,} positions "
                   f"({'useful' if learned.err < len(arr) / 100 else 'too large: the line does not fit this distribution'})")

    st.subheader("Cache for repeated (Zipf) queries")
    if st.button("Compare uniform vs Zipf queries with a cache"):
        arr = sample(all_keys, 100_000).tolist()
        out = []
        for pattern in ["uniform", "zipf"]:
            qs = queries(np.array(arr), 5_000, pattern, 1.0, seed=4)
            c = CachedSearch(lambda k: binary_search(arr, k))
            for q in qs:
                c.search(q)
            out.append({"query pattern": pattern, "cache hit rate": f"{c.hits / len(qs):.0%}"})
        st.table(pd.DataFrame(out))

    st.subheader("Verdict: which structure for which situation?")
    st.markdown("""
| Situation | Best choice | Why |
|---|---|---|
| Tiny data (about 100 keys or fewer) | Linear or any | Setup costs more than the search saves |
| Static data, many lookups | Sorted array + `bisect` | O(log n) in C, least memory |
| Frequent inserts and deletes | Red-black tree | O(log n) updates; arrays shift O(n) elements |
| Keys arrive sorted (IDs, timestamps) | Red-black tree, never a plain BST | Plain BST degrades to O(n) |
| Evenly spread numeric keys | Interpolation or learned index | Position can be predicted from the key |
| Few repeated hot queries | Add a cache in front | Most lookups never reach the structure |
""")
    st.caption("Replace this table with the conclusions from YOUR measurements before the demo.")