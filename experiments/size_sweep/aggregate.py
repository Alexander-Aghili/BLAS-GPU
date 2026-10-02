#!/usr/bin/env python3
"""Aggregate raw.csv passes: median across passes of each pass's median, global min."""
import csv
import statistics
import sys
from collections import defaultdict

S = 4  # bytes per FP32 element

def useful(routine, n):
    """(FLOPs, useful bytes) per the article's Table 1 with beta = 0."""
    if routine == "dot":
        return 2 * n, S * 2 * n
    if routine == "gemv":  # s[mn + n + m], m = n
        return 2 * n * n, S * (n * n + 2 * n)
    return 2 * n ** 3, S * 3 * n * n  # gemm: s[mk + kn + mn]

src, dst = sys.argv[1], sys.argv[2]
groups = defaultdict(list)
order = []
for r in csv.DictReader(open(src)):
    key = (r["routine"], r["impl"], r["rev"], int(r["size"]))
    if key not in groups:
        order.append(key)
    groups[key].append(r)

with open(dst, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["routine", "impl", "rev", "size", "time_us_median", "time_us_min", "reps", "gflops",
                "gbps_useful", "rel_err"])
    for key in order:
        rows = groups[key]
        med = statistics.median(float(r["time_us_median"]) for r in rows)
        mn = min(float(r["time_us_min"]) for r in rows)
        reps = sum(int(r["reps"]) for r in rows)
        err = max(float(r["rel_err"]) for r in rows)
        flops, byts = useful(key[0], key[3])
        w.writerow([*key, f"{med:.3f}", f"{mn:.3f}", reps, f"{flops / med / 1e3:.2f}",
                    f"{byts / med / 1e3:.2f}", f"{err:.3e}"])
print(f"wrote {dst}: {len(order)} rows from {sum(len(v) for v in groups.values())} pass rows")
