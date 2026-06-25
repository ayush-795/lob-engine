"""Order Flow Imbalance predictability study.

Reads the feature CSV produced by `analysis/build_features.py` and tests the
central hypothesis from Cont, Kukanov & Stoikov (2014): contemporaneous order
flow imbalance (OFI) explains short-horizon mid-price changes -- then extends it
to the multi-level / integrated OFI of Cont, Cucuringu & Zhang (2023) and a
Kolm-Turiel-Westray-style horizon sweep.

Predictor families compared head to head:
    L1 OFI (baseline)
    integrated PC1, scalar-Q normalised      (CCZ as usually stated)
    integrated PC1, per-level-Q normalised   (confound control)
    full L1..LM vector OLS                    (supervised upper bound)

We deliberately report this *honestly*: in-sample contemporaneous R^2,
out-of-sample predictive R^2 across horizons (fit 1st half, score 2nd half),
and what survives a conservative half-spread transaction cost.

Usage:  python analysis/ofi_study.py features.csv
"""

import sys
import numpy as np
import pandas as pd


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["spread"] = df["ask"] - df["bid"]
    return df


def level_cols(df, prefix):
    cols = [c for c in df.columns
            if c.startswith(prefix) and c[len(prefix):].isdigit()]
    return sorted(cols, key=lambda c: int(c[len(prefix):]))


def bucket(df: pd.DataFrame, dt: float = 1.0) -> pd.DataFrame:
    """Aggregate event-time rows into fixed clock-time buckets of `dt` seconds:
    summed OFI (every ofi* column) vs. mid-price change over the bucket."""
    g = (df["time"] // dt).astype(int)
    ofi_cols = [c for c in df.columns if c.startswith("ofi")]
    agg = {c: (c, "sum") for c in ofi_cols}
    agg.update(mid_first=("mid", "first"), mid_last=("mid", "last"),
               spread=("spread", "mean"))
    out = df.groupby(g).agg(**agg)
    out["dmid"] = out["mid_last"] - out["mid_first"]
    return out.dropna()


def ols(X, y):
    A = np.column_stack([np.ones(len(X)), np.atleast_2d(X.T).T if X.ndim == 1 else X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return beta


def r2(y, pred):
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def insample_r2(b, feats):
    X = b[feats].values
    beta = ols(X, b.dmid.values)
    A = np.column_stack([np.ones(len(X)), np.atleast_2d(X.T).T if X.ndim == 1 else X])
    return r2(b.dmid.values, A @ beta)


def oos_predictive(b, feats, h):
    """OOS R^2 predicting the cumulative mid change over [t+1, t+h] from `feats`,
    fitting on the first half and scoring on the second. Multivariate-safe."""
    target = b["dmid"].rolling(h).sum().shift(-h)
    X = b[feats].values
    y = target.values
    ok = ~np.isnan(y)
    X, y = X[ok], y[ok]
    split = len(y) // 2
    beta = ols(X[:split], y[:split])
    Xte = np.atleast_2d(X[split:].T).T if X.ndim == 1 else X[split:]
    A = np.column_stack([np.ones(len(Xte)), Xte])
    return r2(y[split:], A @ beta)


def main():
    if len(sys.argv) < 2:
        print("usage: python ofi_study.py features.csv")
        sys.exit(1)

    df = load(sys.argv[1])
    b = bucket(df, dt=1.0)
    n = len(b)
    lvls = level_cols(b, "ofi_")
    lvls_pl = level_cols(b, "ofi_pl_")
    print(f"buckets: {n}, levels: {len(lvls)}, mean spread: {b.spread.mean():.0f} ticks")

    # ---- predictor families: (label, feature columns) ----
    families = [("L1 OFI", ["ofi"])]
    if "ofi_int" in b:
        families.append(("integ PC1 (scalar-Q)", ["ofi_int"]))
    if "ofi_int_pl" in b:
        families.append(("integ PC1 (per-lvl)", ["ofi_int_pl"]))
    if lvls:
        families.append((f"full L1..L{len(lvls)} OLS", lvls))

    # ---- contemporaneous in-sample R^2 ----
    print("\ncontemporaneous in-sample R^2:")
    for label, feats in families:
        print(f"  {label:<22s} {insample_r2(b, feats):.3f}")

    # ---- out-of-sample predictive horizon sweep ----
    horizons = [h for h in (1, 2, 5, 10, 30) if h < n // 4]
    labels = [f for f, _ in families]
    print("\nout-of-sample predictive R^2 (fit 1st half, score 2nd half):")
    print("  horizon | " + " | ".join(f"{l:>20s}" for l in labels))
    print("  " + "-" * (10 + len(labels) * 23))
    for h in horizons:
        cells = " | ".join(f"{oos_predictive(b, fe, h):20.4f}" for _, fe in families)
        print(f"  t+{h:<5d} | {cells}")

    # ---- honest cost check on the best single index ----
    best = "ofi_int_pl" if "ofi_int_pl" in b else "ofi"
    target = b["dmid"].shift(-1)
    X, y = b[best].values, target.values
    ok = ~np.isnan(y)
    X, y = X[ok], y[ok]
    split = len(y) // 2
    beta = ols(X[:split], y[:split])
    pred = beta[0] + beta[1] * X[split:]
    gross, cost = np.abs(pred).mean(), 0.5 * b.spread.mean()
    print(f"\ncost check on '{best}' (t+1): mean |predicted move| {gross:.1f} ticks "
          f"vs half-spread {cost:.1f} ticks")
    print("verdict:", "edge may survive costs" if gross > cost
          else "predictability is real but eaten by transaction costs")


if __name__ == "__main__":
    main()
