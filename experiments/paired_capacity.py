#!/usr/bin/env python3
"""Paired capacity comparison from an existing table8_capacity.json.

All configurations were fitted on the same splits, so the difference between
two of them is paired and its standard error is far smaller than the
across-split standard deviation of either. Run this rather than re-running the
sweep:

    python paired_capacity.py results/table8_capacity.json
"""
import json, sys
import numpy as np

d = json.load(open(sys.argv[1]))["results"]
by = {}
for key, per_ds in d.items():
    L, dd = key.replace("L", "").split("_d")
    by[(int(L), int(dd))] = per_ds

for ds in ("cgd", "crc"):
    print(f"\n=== {ds} ===")
    have = {k: np.asarray(v[ds].get("cindex_per_split", []), float)
            for k, v in by.items()}
    have = {k: v for k, v in have.items() if v.size}
    if not have:
        print("  per-split values absent; this JSON predates the change")
        continue
    best = max(have, key=lambda k: have[k].mean())
    ref = (2, 64)
    print(f"  best L={best[0]}, d={best[1]}  C={have[best].mean():.4f}")
    print(f"  default L=2, d=64          C={have[ref].mean():.4f}")
    diff = have[best] - have[ref]
    se = diff.std(ddof=1) / np.sqrt(diff.size)
    lo, hi = diff.mean() - 1.96 * se, diff.mean() + 1.96 * se
    print(f"  paired difference {diff.mean():+.4f} (95% CI {lo:+.4f}, {hi:+.4f}), "
          f"best ahead on {100*(diff>0).mean():.0f}% of splits")
    print("  ->", "the smaller model is reliably better" if lo > 0
          else ("the default is reliably better" if hi < 0
                else "these data do not separate the two"))
    for cfg in sorted(have, key=lambda k: by[k][ds]["params"]):
        dd2 = have[best] - have[cfg]
        se2 = dd2.std(ddof=1) / np.sqrt(dd2.size) or 1e-12
        if dd2.mean() - 1.96 * se2 <= 0:
            print(f"  cheapest not reliably worse than best: "
                  f"L={cfg[0]}, d={cfg[1]} ({by[cfg][ds]['params']:,} params)")
            break
