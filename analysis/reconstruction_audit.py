"""Reconstruction-noise audit for OFI effect sizes (Gap 2).

Every published OFI / deep-LOB study (CCZ 2023, KTW 2023, Briola 2024) runs on a
*reconstructed* book and none reports how the headline result moves under
reconstruction error. The engine's `--validate` mode already measures that error
against LOBSTER's official snapshots; this script asks the next question:

    if the reconstructed depth is wrong by a measurable amount, how much does the
    OFI -> mid-change relationship (slope and R^2) actually shift?

We perturb the ground-truth book with a controlled noise model and recompute the
contemporaneous OFI regression at each noise level. The noise magnitude `sigma`
should ultimately be *calibrated* to the per-level size error that `--validate`
reports for a given reconstruction; here we sweep it so the sensitivity (the
"reconstruction-error budget") is explicit.

Noise model (per level, per snapshot, i.i.d.):
    q_hat = q * exp(sigma * N(0,1))          multiplicative log-normal on sizes
This preserves positivity and non-dimensionality, and mimics depth that drifts
proportionally to its size -- the dominant failure mode of mid-session replay.

Usage: python analysis/reconstruction_audit.py <message.csv> <orderbook.csv>
                                               [--levels M] [--reps R]
"""
import sys
import numpy as np
import pandas as pd

from build_features import level_ofi, integrate


def load_book(msg_path, ob_path, M):
    times = pd.read_csv(msg_path, header=None, usecols=[0], names=["time"])["time"].values
    raw = pd.read_csv(ob_path, header=None)
    avail = raw.shape[1] // 4
    M = min(M, avail)
    n = min(len(times), len(raw))
    raw = raw.iloc[:n].to_numpy(dtype=np.float64)
    times = times[:n]
    pa = raw[:, [4 * m + 0 for m in range(M)]]
    qa = raw[:, [4 * m + 1 for m in range(M)]]
    pb = raw[:, [4 * m + 2 for m in range(M)]]
    qb = raw[:, [4 * m + 3 for m in range(M)]]
    return times, pa, qa, pb, qb, M


def ofi_features(times, pa, qa, pb, qb, M):
    """Return per-bucket (ofi_L1, ofi_int, dmid, spread) at 1s buckets."""
    n = len(times)
    ofi_levels = np.zeros((n, M))
    for m in range(M):
        ofi_levels[:, m] = level_ofi(pb[:, m], qb[:, m], pa[:, m], qa[:, m])
    Q = 0.5 * (qb + qa).mean()
    ofi_int, _ = integrate(ofi_levels / Q)

    mid = 0.5 * (pb[:, 0] + pa[:, 0])
    spread = pa[:, 0] - pb[:, 0]
    g = (times // 1.0).astype(int)
    df = pd.DataFrame({"g": g, "ofi": ofi_levels[:, 0], "ofi_int": ofi_int,
                       "mid": mid, "spread": spread})
    b = df.groupby("g").agg(ofi=("ofi", "sum"), ofi_int=("ofi_int", "sum"),
                            mid_f=("mid", "first"), mid_l=("mid", "last"),
                            spread=("spread", "mean"))
    b["dmid"] = b.mid_l - b.mid_f
    return b.dropna()


def contemp(x, y):
    A = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ beta
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return beta[1], (1 - ss_res / ss_tot if ss_tot > 0 else float("nan"))


def main(msg_path, ob_path, M, reps):
    times, pa, qa, pb, qb, M = load_book(msg_path, ob_path, M)

    # Baseline (clean book).
    b0 = ofi_features(times, pa, qa, pb, qb, M)
    s0, r0 = contemp(b0.ofi.values, b0.dmid.values)
    si0, ri0 = contemp(b0.ofi_int.values, b0.dmid.values)
    print(f"levels: {M}, buckets: {len(b0)}")
    print(f"baseline   L1: slope {s0:.4e}  R^2 {r0:.4f}   "
          f"| integ: slope {si0:.4e}  R^2 {ri0:.4f}\n")

    print("reconstruction-error budget (mean +/- sd over "
          f"{reps} reps):")
    print("  sigma |     L1 R^2      | L1 slope drift |    integ R^2")
    print("  " + "-" * 56)
    rng = np.random.default_rng(0)
    for sigma in (0.0, 0.05, 0.10, 0.20, 0.40):
        r1s, sl1s, ris = [], [], []
        for _ in range(reps if sigma > 0 else 1):
            qbp = qb * np.exp(sigma * rng.standard_normal(qb.shape))
            qap = qa * np.exp(sigma * rng.standard_normal(qa.shape))
            b = ofi_features(times, pa, qap, pb, qbp, M)
            s, r = contemp(b.ofi.values, b.dmid.values)
            _, ri = contemp(b.ofi_int.values, b.dmid.values)
            r1s.append(r); sl1s.append(s); ris.append(ri)
        r1, r1sd = np.mean(r1s), np.std(r1s)
        drift = (np.mean(sl1s) - s0) / s0 * 100
        ri, risd = np.mean(ris), np.std(ris)
        print(f"  {sigma:4.2f}  | {r1:.4f} +/- {r1sd:.4f} | {drift:+12.1f}% | "
              f"{ri:.4f} +/- {risd:.4f}")

    print("\nNote: calibrate sigma to the per-level size error reported by "
          "`lob_replay --validate` for an honest, data-driven error budget.")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    M = int(sys.argv[sys.argv.index("--levels") + 1]) if "--levels" in sys.argv else 10
    reps = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 5
    if len(args) < 2:
        print(__doc__)
        sys.exit(1)
    main(args[0], args[1], M, reps)
