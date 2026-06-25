"""Build the OFI research feature file from LOBSTER ground-truth data.

We intentionally take the book from the exchange-published *orderbook* file
(exact L1..LM state) and timestamps from the aligned *message* file, rather than
from message-replay. Pure message-replay cannot be bit-exact for a sample that
begins mid-session: the stream references orders (by id) created in the opening
auction that are outside the visible book -- see the engine's --validate mode,
which quantifies that drift. Microstructure studies use the snapshot file for
exactly this reason.

OFI definitions
---------------
Level-m order flow imbalance (Cont, Kukanov & Stoikov 2014; generalised to M
levels by Cont, Cucuringu & Zhang 2023, *Cross-Impact of Order Flow Imbalance
in Equity Markets*, Quantitative Finance 23(10)):

    bid flow   a^b_m = q^b_m              if P^b_m >  P^b_m(prev)
                       q^b_m - q^b_m(prev) if P^b_m == P^b_m(prev)
                       -q^b_m(prev)        if P^b_m <  P^b_m(prev)
    ask flow   a^a_m = -q^a_m             if P^a_m <  P^a_m(prev)
                       q^a_m - q^a_m(prev) if P^a_m == P^a_m(prev)
                       q^a_m(prev)         if P^a_m >  P^a_m(prev)
    OFI_m      = a^b_m - a^a_m

Following CCZ, each level OFI is normalised by the sample-average per-level depth
Q = mean over (t, m) of (q^b_m + q^a_m) / 2 so the levels are comparable and the
units are dimensionless. The *integrated* OFI is the projection of the M
normalised level OFIs onto their first principal component (CCZ show PC1 carries
the great majority of the variance with near-uniform, same-sign weights).

Output columns
--------------
    time,bid,bid_sz,ask,ask_sz,mid,ofi,        <- L1, back-compat with ofi_study
    ofi_1..ofi_M,       ofi_int                <- scalar-Q normalised + PC1
    ofi_pl_1..ofi_pl_M, ofi_int_pl             <- per-level-Q normalised + PC1

Usage: python analysis/build_features.py <message.csv> <orderbook.csv> <out.csv>
                                         [--levels M]
"""
import sys
import numpy as np
import pandas as pd


def level_ofi(pb, qb, pa, qa):
    """Per-level OFI time series from level price/size arrays (length n)."""
    n = len(pb)
    e = np.zeros(n)
    # bid side
    e[1:] += (pb[1:] > pb[:-1]) * qb[1:]
    e[1:] += (pb[1:] == pb[:-1]) * (qb[1:] - qb[:-1])
    e[1:] += (pb[1:] < pb[:-1]) * (-qb[:-1])
    # ask side (subtracted)
    e[1:] -= (pa[1:] < pa[:-1]) * (-qa[1:])
    e[1:] -= (pa[1:] == pa[:-1]) * (qa[1:] - qa[:-1])
    e[1:] -= (pa[1:] > pa[:-1]) * (qa[:-1])
    return e


def integrate(ofi_levels):
    """First-principal-component projection of the (n, M) normalised OFI matrix.

    Returns (integrated_series, weights). Weights are sign-fixed so that the
    component points in the direction of net buy pressure (positive loadings)."""
    X = ofi_levels - ofi_levels.mean(axis=0, keepdims=True)
    # PC1 via SVD of the centred matrix.
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    w = vt[0]
    if w.sum() < 0:          # orient toward buy pressure
        w = -w
    return ofi_levels @ w, w


def main(msg_path: str, ob_path: str, out_path: str, levels: int):
    # message file: time, type, id, size, price, direction
    times = pd.read_csv(msg_path, header=None, usecols=[0], names=["time"])["time"].values

    # orderbook file columns repeat: ask1,asz1,bid1,bsz1, ask2,asz2,bid2,bsz2, ...
    raw = pd.read_csv(ob_path, header=None)
    avail = raw.shape[1] // 4
    M = min(levels, avail)
    if M < levels:
        print(f"warning: file has {avail} levels, using {M}")

    n = min(len(times), len(raw))
    raw = raw.iloc[:n]

    out = pd.DataFrame({"time": times[:n]})
    ofi_levels = np.zeros((n, M))
    for m in range(M):
        pa = raw.iloc[:, 4 * m + 0].values.astype(np.float64)
        qa = raw.iloc[:, 4 * m + 1].values.astype(np.float64)
        pb = raw.iloc[:, 4 * m + 2].values.astype(np.float64)
        qb = raw.iloc[:, 4 * m + 3].values.astype(np.float64)
        ofi_levels[:, m] = level_ofi(pb, qb, pa, qa)
        if m == 0:
            out["bid"], out["bid_sz"] = pb, qb
            out["ask"], out["ask_sz"] = pa, qa
            out["mid"] = 0.5 * (pb + pa)
            out["ofi"] = ofi_levels[:, 0]   # raw L1 OFI, back-compat

    qb_all = raw.iloc[:, [4 * m + 3 for m in range(M)]].values.astype(np.float64)
    qa_all = raw.iloc[:, [4 * m + 1 for m in range(M)]].values.astype(np.float64)
    depth = 0.5 * (qb_all + qa_all)            # (n, M) per-level depth

    # (i) CCZ scalar normalisation: one average depth across all levels.
    Q = depth.mean()
    ofi_norm = ofi_levels / Q
    # (ii) Per-level normalisation: each level by its own average depth Q_m.
    #      Removes the mechanical advantage deep levels get from larger sizes,
    #      so the PC1 weight profile can be compared against (i) for confounds.
    Qm = depth.mean(axis=0)                     # (M,)
    ofi_norm_pl = ofi_levels / Qm

    for m in range(M):
        out[f"ofi_{m + 1}"] = ofi_norm[:, m]
        out[f"ofi_pl_{m + 1}"] = ofi_norm_pl[:, m]

    out["ofi_int"], w = integrate(ofi_norm)
    out["ofi_int_pl"], w_pl = integrate(ofi_norm_pl)

    out.to_csv(out_path, index=False, float_format="%.9g")
    spread = (out.ask - out.bid).mean()
    print(f"wrote {len(out)} rows, {M} levels, to {out_path}")
    print(f"mean spread: {spread:.0f} ticks (${spread/10000:.4f}); "
          f"avg per-level depth Q = {Q:.0f}")
    fmt = lambda a: np.array2string(a, precision=3, suppress_small=True)
    print(f"scalar-Q  PC1 weights: {fmt(w)}  (var share {pc1_share(ofi_norm):.1%})")
    print(f"per-lvl   PC1 weights: {fmt(w_pl)}  (var share {pc1_share(ofi_norm_pl):.1%})")


def pc1_share(X):
    Xc = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    return (s[0] ** 2) / (s ** 2).sum()


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    levels = 10
    if "--levels" in sys.argv:
        levels = int(sys.argv[sys.argv.index("--levels") + 1])
    if len(args) < 3:
        print(__doc__)
        sys.exit(1)
    main(args[0], args[1], args[2], levels)
