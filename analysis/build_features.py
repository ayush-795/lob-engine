"""Build the OFI research feature file from LOBSTER ground-truth data.

We intentionally take top-of-book from the exchange-published *orderbook* file
(exact L1 state) and timestamps from the aligned *message* file, rather than
from message-replay. Pure message-replay cannot be bit-exact for a sample that
begins mid-session: the stream references orders (by id) created in the opening
auction that are outside the visible book -- see the engine's --validate mode,
which quantifies that drift. Microstructure studies use the snapshot file for
exactly this reason.

Output columns match the C++ --dump tool: time,bid,bid_sz,ask,ask_sz,mid,ofi

Usage: python analysis/build_features.py <message.csv> <orderbook.csv> <out.csv>
"""
import sys
import numpy as np
import pandas as pd


def main(msg_path: str, ob_path: str, out_path: str):
    # message file: time, type, id, size, price, direction
    times = pd.read_csv(msg_path, header=None, usecols=[0], names=["time"])["time"].values

    # orderbook file: ask1,asz1,bid1,bsz1,ask2,...  (we only need level 1)
    ob = pd.read_csv(ob_path, header=None, usecols=[0, 1, 2, 3],
                     names=["ask", "ask_sz", "bid", "bid_sz"])
    n = min(len(times), len(ob))
    ob = ob.iloc[:n].copy()
    ob["time"] = times[:n]
    ob["mid"] = 0.5 * (ob["bid"] + ob["ask"])

    # Order Flow Imbalance (Cont, Kukanov & Stoikov 2014), computed from
    # successive level-1 prices/sizes.
    pb, qb = ob["bid"].values, ob["bid_sz"].values
    pa, qa = ob["ask"].values, ob["ask_sz"].values
    e = np.zeros(n)
    e[1:] = (
        (pb[1:] >= pb[:-1]) * qb[1:] - (pb[1:] <= pb[:-1]) * qb[:-1]
        - (pa[1:] <= pa[:-1]) * qa[1:] + (pa[1:] >= pa[:-1]) * qa[:-1]
    )
    ob["ofi"] = e

    ob = ob[["time", "bid", "bid_sz", "ask", "ask_sz", "mid", "ofi"]]
    ob.to_csv(out_path, index=False, float_format="%.9g")
    print(f"wrote {len(ob)} rows to {out_path}")
    print(f"mean spread: {(ob.ask - ob.bid).mean():.0f} ticks "
          f"(${(ob.ask - ob.bid).mean()/10000:.4f})")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
