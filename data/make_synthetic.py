"""Generate a synthetic LOBSTER-format message file so the pipeline runs
end-to-end without needing licensed data. Replace with real LOBSTER sample
data (https://lobsterdata.com/info/DataSamples.php) for the actual study --
the synthetic stream is only a smoke test of the plumbing.

Output columns: time, type, order_id, size, price, direction
"""
import sys
import numpy as np

def main(path: str, n: int = 200_000, seed: int = 7):
    rng = np.random.default_rng(seed)
    rows = []
    oid = 1
    mid = 1_000_000          # price in 1/10000 dollars (= $100.00)
    live = {}                # id -> (price, size, dir)
    t = 34200.0              # 9:30am in seconds-after-midnight
    for _ in range(n):
        t += rng.exponential(0.001)
        r = rng.random()
        # bias the mid with a slow random walk to create some signal
        mid += int(rng.normal(0, 2))
        if r < 0.55 or not live:                       # add limit order
            direction = 1 if rng.random() < 0.5 else -1
            offset = (1 + rng.integers(0, 5)) * 100
            price = mid - offset if direction == 1 else mid + offset
            size = int(rng.integers(1, 50))
            rows.append((t, 1, oid, size, price, direction))
            live[oid] = (price, size, direction)
            oid += 1
        else:                                          # cancel/execute existing
            vid = rng.choice(list(live.keys()))
            price, size, direction = live[vid]
            etype = 3 if rng.random() < 0.5 else 4
            rows.append((t, etype, vid, size, price, direction))
            del live[vid]

    with open(path, "w") as f:
        for t, ty, i, s, p, d in rows:
            f.write(f"{t:.9f},{ty},{i},{s},{p},{d}\n")
    print(f"wrote {len(rows)} messages to {path}")

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "data/synthetic_messages.csv"
    main(out)
