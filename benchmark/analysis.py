from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path("results")
MODELS = {
    "O(1)": lambda n: np.ones_like(n),
    "O(log n)": np.log2,
    "O(n)": lambda n: n,
    "O(n log n)": lambda n: n * np.log2(n),
}


def fit_models(n, t):
    n, t = np.asarray(n, float), np.asarray(t, float)
    out = {}
    for name, f in MODELS.items():
        X = np.column_stack([f(n), np.ones_like(n)])
        coef, *_ = np.linalg.lstsq(X, t, rcond=None)
        rss = float(np.sum((X @ coef - t) ** 2))
        out[name] = len(n) * np.log(rss / len(n) + 1e-12) + 4   # AIC, 2 parameters
    best = min(out, key=out.get)
    slope = np.polyfit(np.log(n), np.log(t), 1)[0]
    return best, slope


def main():
    df = pd.read_csv(RESULTS / "scaling.csv")
    fits = []
    for key, g in df.groupby("key"):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for (s, order), h in g.groupby(["structure", "order"]):
            h = h.sort_values("n")
            label = s if s in ("linear", "binary") else f"{s} ({order} insert)"
            ax.errorbar(h["n"], h["median_ns"],
                        yerr=[h["median_ns"] - h["ci_low_ns"], h["ci_high_ns"] - h["median_ns"]],
                        marker="o", capsize=3, label=label)
            if len(h) >= 3:
                best, slope = fit_models(h["n"], h["median_ns"])
                fits.append({"key": key, "structure": s, "order": order,
                             "best_model": best, "loglog_slope": round(slope, 3)})
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("n (number of keys)")
        ax.set_ylabel("time per search (ns, median, 95% CI)")
        ax.set_title(f"Search time vs data size: {key} keys")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(RESULTS / f"scaling_{key}.png", dpi=150)
        plt.close(fig)
    pd.DataFrame(fits).to_csv(RESULTS / "fits.csv", index=False)
    print(pd.DataFrame(fits).to_string(index=False))


if __name__ == "__main__":
    main()