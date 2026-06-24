"""Order Flow Imbalance predictability study.

Reads the feature CSV produced by `lob_replay --dump` and tests the central
hypothesis from Cont, Kukanov & Stoikov (2014): contemporaneous order flow
imbalance (OFI) explains short-horizon mid-price changes.

We deliberately report this *honestly*:
  * in-sample R^2 of the contemporaneous regression,
  * out-of-sample predictive R^2 at several horizons,
  * what survives a conservative half-spread transaction cost.

Usage:  python analysis/ofi_study.py features.csv
"""

import sys
import numpy as np
import pandas as pd


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["spread"] = df["ask"] - df["bid"]
    return df


def bucket(df: pd.DataFrame, dt: float = 1.0) -> pd.DataFrame:
    """Aggregate event-time rows into fixed clock-time buckets of `dt` seconds:
    summed OFI vs. mid-price change over the bucket."""
    g = (df["time"] // dt).astype(int)
    out = pd.DataFrame({
        "ofi": df.groupby(g)["ofi"].sum(),
        "mid_first": df.groupby(g)["mid"].first(),
        "mid_last": df.groupby(g)["mid"].last(),
        "spread": df.groupby(g)["spread"].mean(),
    })
    out["dmid"] = out["mid_last"] - out["mid_first"]
    return out.dropna()


def ols(x: np.ndarray, y: np.ndarray):
    """Simple OLS y = a + b x. Returns (a, b, r2)."""
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return beta[0], beta[1], r2


def main():
    if len(sys.argv) < 2:
        print("usage: python ofi_study.py features.csv")
        sys.exit(1)

    df = load(sys.argv[1])
    b = bucket(df, dt=1.0)
    n = len(b)
    print(f"buckets: {n}, mean spread: {b.spread.mean():.0f} ticks")

    # Contemporaneous fit (the CKS regression).
    a, slope, r2 = ols(b.ofi.values, b.dmid.values)
    print(f"\ncontemporaneous:  dmid = {a:.2f} + {slope:.4e} * OFI   R^2 = {r2:.3f}")

    # Out-of-sample predictive power: fit on first half, score next-bucket
    # prediction on second half. This is where most naive 'signals' die.
    split = n // 2
    tr, te = b.iloc[:split], b.iloc[split:]
    a, slope, _ = ols(tr.ofi.values[:-1], tr.dmid.shift(-1).dropna().values)
    pred = a + slope * te.ofi.values[:-1]
    actual = te.dmid.shift(-1).dropna().values
    ss_res = np.sum((actual - pred) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    oos_r2 = 1 - ss_res / ss_tot
    print(f"predictive (t+1): out-of-sample R^2 = {oos_r2:.3f}")

    # Honest cost check: a strategy that trades on predicted direction must pay
    # ~half the spread to cross. Net edge per trade in ticks:
    gross = np.abs(pred).mean()
    cost = 0.5 * b.spread.mean()
    print(f"\nmean |predicted move|: {gross:.1f} ticks vs half-spread cost: {cost:.1f} ticks")
    print("verdict:", "edge may survive costs" if gross > cost
          else "predictability is real but eaten by transaction costs")


if __name__ == "__main__":
    main()
