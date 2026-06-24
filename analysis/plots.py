"""Generate the three figures that tell the project's story.

  fig1_ofi_scatter.png   contemporaneous OFI vs. mid-price change (the signal)
  fig2_r2_decay.png      predictive R^2 across horizons, in- vs out-of-sample
  fig3_latency_hist.png  engine per-update latency distribution (log-x)

Usage:
  python analysis/plots.py <features.csv> [latency.csv] [out_dir]
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ols_r2(x, y):
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return beta, (1 - ss_res / ss_tot if ss_tot > 0 else np.nan)


def bucket(df, dt):
    g = (df["time"] // dt).astype(int)
    out = pd.DataFrame({
        "ofi": df.groupby(g)["ofi"].sum(),
        "mid_first": df.groupby(g)["mid"].first(),
        "mid_last": df.groupby(g)["mid"].last(),
    })
    out["dmid"] = out["mid_last"] - out["mid_first"]
    return out.dropna()


def fig_ofi_scatter(df, path):
    b = bucket(df, 1.0)
    # de-noise the picture: bin OFI and show mean response (binscatter)
    x, y = b["ofi"].values, b["dmid"].values
    beta, r2 = ols_r2(x, y)
    q = pd.qcut(x, 30, duplicates="drop")
    binned = pd.DataFrame({"x": x, "y": y}).groupby(q, observed=True).mean()

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.scatter(x, y, s=4, alpha=0.15, color="#9abedb", label="1s buckets")
    ax.scatter(binned["x"], binned["y"], s=28, color="#c0392b",
               zorder=3, label="binned mean")
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, beta[0] + beta[1] * xs, color="#2c3e50", lw=1.8,
            label=f"OLS fit  (R²={r2:.2f})")
    ax.set_xlabel("Order Flow Imbalance (1s bucket)")
    ax.set_ylabel("Δ mid-price over bucket (ticks)")
    ax.set_title("Contemporaneous OFI explains mid-price moves")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def fig_r2_decay(df, path):
    horizons = [0.5, 1, 2, 5, 10, 30, 60]
    in_s, out_s = [], []
    for h in horizons:
        b = bucket(df, h)
        x = b["ofi"].values
        # contemporaneous in-sample
        _, r2_in = ols_r2(x, b["dmid"].values)
        # predictive t+1 out-of-sample (fit first half, score second)
        n = len(b); split = n // 2
        yx = b["dmid"].shift(-1)
        tr_x, tr_y = x[:split][:-1], yx.values[:split][:-1]
        te_x, te_y = x[split:][:-1], yx.values[split:][:-1]
        m = ~np.isnan(tr_y); beta, _ = ols_r2(tr_x[m], tr_y[m])
        pred = beta[0] + beta[1] * te_x
        mm = ~np.isnan(te_y)
        ss_res = np.sum((te_y[mm] - pred[mm]) ** 2)
        ss_tot = np.sum((te_y[mm] - te_y[mm].mean()) ** 2)
        in_s.append(r2_in); out_s.append(1 - ss_res / ss_tot)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(horizons, in_s, "o-", color="#2c3e50", label="contemporaneous (in-sample)")
    ax.plot(horizons, out_s, "s-", color="#c0392b", label="predictive t+1 (out-of-sample)")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("bucket horizon (seconds, log scale)")
    ax.set_ylabel("R²")
    ax.set_title("Signal is contemporaneous, not predictive")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def fig_latency(lat_path, path):
    # Tail-latency curve: the canonical HFT latency plot. The x-axis stretches
    # the tail so p99 / p99.9 / p99.99 are readable. Per-update work sits at the
    # steady_clock resolution floor (~40 ns on this box), so latencies are
    # quantized -- the curve makes that floor and the tail explicit.
    lat = np.sort(pd.read_csv(lat_path)["ns"].values)
    lat = lat[lat > 0]
    pct = np.array([50, 75, 90, 99, 99.9, 99.99, 99.999])
    vals = np.percentile(lat, pct)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    xpos = -np.log10(1 - pct / 100)          # stretch the tail
    ax.plot(xpos, vals, "o-", color="#2c3e50")
    for xp, v, p in zip(xpos, vals, pct):
        ax.annotate(f"{v:.0f} ns", (xp, v), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=7, color="#c0392b")
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"p{p:g}" for p in pct], fontsize=8)
    ax.set_ylabel("per-update latency (ns)")
    ax.set_title("Order book update tail latency (400k AAPL events)")
    ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    feats = sys.argv[1]
    lat = sys.argv[2] if len(sys.argv) > 2 else None
    out_dir = sys.argv[3] if len(sys.argv) > 3 else "analysis/figures"
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(feats)
    fig_ofi_scatter(df, os.path.join(out_dir, "fig1_ofi_scatter.png"))
    fig_r2_decay(df, os.path.join(out_dir, "fig2_r2_decay.png"))
    print("wrote fig1, fig2")
    if lat and os.path.exists(lat):
        fig_latency(lat, os.path.join(out_dir, "fig3_latency_hist.png"))
        print("wrote fig3")
    else:
        print("no latency file given -> run: lob_replay <msg> --bench build/lat.csv")


if __name__ == "__main__":
    main()
