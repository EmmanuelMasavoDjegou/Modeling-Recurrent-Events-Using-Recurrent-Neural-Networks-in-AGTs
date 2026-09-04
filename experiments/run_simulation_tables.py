#!/usr/bin/env python3
"""
Tables 1-3 -- the main simulation grid.

Regenerates the three simulation tables of the manuscript: one per mean
function (linear, interaction, GAM), each crossing three error distributions,
two dependence mechanisms, three censoring levels and two training sizes.

WHY THIS EXISTS
---------------
The original notebooks produced these tables under three defects: an inverted
Gehan hinge sign, a latent-gap leak that trained the model against uncensored
outcomes, and a missing WRS normalization. The first two partially cancelled,
so the numbers looked plausible while being wrong in a way that grew with the
censoring fraction. The 65% columns are the most affected. Run
`2.Simulation/RNN-AGT_v1.ipynb` first to measure the effect at your scale
before deciding how much of the manuscript's simulation section to revise.

A second, smaller change: `tau` is now solved for to hit each target censoring
fraction rather than fixed at 3000, so the 25/50/65% columns mean what they say.

COST
----
The full grid is 3 mean functions x 3 errors x 2 dependence x 3 censoring x 2
sizes = 108 cells, times `--replicates`. At 500 replicates this is a cluster
job, not a laptop one. Use `--replicates 20` for a first look and
`--mean-funcs interaction` to regenerate one table at a time.

Usage
-----
    python experiments/run_simulation_tables.py --replicates 500
    python experiments/run_simulation_tables.py --replicates 20 --quick
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rnn_agt import data as D
from rnn_agt import latex
from rnn_agt.seeds import make_seeds
from rnn_agt.train import TrainConfig, train_model

MEAN_FUNCS = ("linear", "interaction", "gam")
ERRORS = ("normal", "gumbel", "logistic")
DEPENDENCE = ("frailty", "ar1")
CENSORING = (0.25, 0.50, 0.65)

TABLE_OF = {"interaction": 1, "gam": 2, "linear": 3}
MEAN_LABEL = {
    "linear": "linear mean function",
    "interaction": "interaction mean function",
    "gam": "GAM-type nonlinear mean function",
}


def run_cell(args, mean_func, error, dependence, censoring, n_train, seed):
    """One grid cell: generate, fit, evaluate."""
    seeds = make_seeds(seed)
    rng = seeds.data()
    tau = D.calibrate_tau(
        n_train, D.MEAN_FUNCTIONS[mean_func], error, rng, censoring,
        D.DEPENDENCE_SPECS[dependence],
    )
    train_s = D.make_dataset(
        n_train, mean_func, error, rng, dependence=dependence, tau=tau
    )
    test_s = D.make_dataset(
        args.n_test, mean_func, error, rng, dependence=dependence, tau=tau
    )
    cfg = TrainConfig(
        model="rnn_agt", epochs=args.epochs, pair_sample_s=args.pair_s,
        pair_batch_b=args.batch, hidden_dim=args.hidden,
        gru_layers=args.layers, lr=args.lr, device=args.device,
    )
    res = train_model(train_s, test_s, 3, cfg, seeds)
    achieved = 1.0 - sum(int(s["delta"].sum()) for s in train_s) / max(
        sum(len(s["delta"]) for s in train_s), 1
    )
    return res.metrics, achieved


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", type=int, default=500)
    ap.add_argument("--mean-funcs", nargs="+", default=list(MEAN_FUNCS))
    ap.add_argument("--n-trains", type=int, nargs="+", default=[1000, 5000])
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--pair-s", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/tables123")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.replicates = 2
        args.n_trains = [200]
        args.n_test = 200
        args.epochs = 2

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    grid = list(itertools.product(
        args.mean_funcs, ERRORS, DEPENDENCE, CENSORING, args.n_trains
    ))
    print(f"{len(grid)} cells x {args.replicates} replicates "
          f"= {len(grid) * args.replicates} fits\n")

    acc = defaultdict(list)
    achieved_cens = defaultdict(list)
    t0 = time.time()

    for k, (mf, err, dep, cens, n_tr) in enumerate(grid):
        for rep in range(args.replicates):
            metrics, achieved = run_cell(
                args, mf, err, dep, cens, n_tr, args.seed + 1000 * rep
            )
            acc[(mf, err, dep, cens, n_tr)].append(
                (metrics["test_cindex"], metrics["test_amse"])
            )
            achieved_cens[(mf, err, dep, cens, n_tr)].append(achieved)

        vals = np.array(acc[(mf, err, dep, cens, n_tr)], dtype=float)
        elapsed = time.time() - t0
        rate = elapsed / (k + 1)
        print(
            f"[{k+1:3d}/{len(grid)}] {mf:12s} {err:9s} {dep:8s} "
            f"cens={cens:.2f} n={n_tr:5d}  "
            f"C={np.nanmean(vals[:, 0]):.3f}  AMSE={np.nanmean(vals[:, 1]):6.2f}  "
            f"(achieved cens {np.mean(achieved_cens[(mf, err, dep, cens, n_tr)]):.2f}, "
            f"eta {rate * (len(grid) - k - 1) / 60:.0f} min)",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Emit one LaTeX fragment per mean function
    # ------------------------------------------------------------------
    summary = {}
    for key, vals in acc.items():
        arr = np.array(vals, dtype=float)
        summary["|".join(map(str, key))] = {
            "cindex": float(np.nanmean(arr[:, 0])),
            "amse": float(np.nanmean(arr[:, 1])),
            "cindex_se": float(np.nanstd(arr[:, 0], ddof=1) / np.sqrt(len(arr)))
            if len(arr) > 1 else 0.0,
            "achieved_censoring": float(np.mean(achieved_cens[key])),
            "n_replicates": len(arr),
        }

    with open(f"{args.out}.json", "w") as fh:
        json.dump({"summary": summary, "args": vars(args)}, fh, indent=2)

    for mf in args.mean_funcs:
        rows = []
        for err in ERRORS:
            for dep in DEPENDENCE:
                for cens in CENSORING:
                    cells = [
                        err if (dep == DEPENDENCE[0] and cens == CENSORING[0]) else "",
                        D.DEPENDENCE_SPECS[dep].label if cens == CENSORING[0] else "",
                        f"{int(cens * 100)}",
                    ]
                    for n_tr in args.n_trains:
                        key = (mf, err, dep, cens, n_tr)
                        if key in acc:
                            arr = np.array(acc[key], dtype=float)
                            cells.append(latex.fmt_c_amse(
                                float(np.nanmean(arr[:, 0])),
                                float(np.nanmean(arr[:, 1])),
                            ))
                        else:
                            cells.append(r"\PH{n/a}")
                    rows.append(cells)

        body = latex.table_rows(rows, row_colors=("rowA", "rowB"))
        path = f"{args.out}_table{TABLE_OF[mf]}_{mf}.tex"
        latex.write_fragment(
            path, f"Table {TABLE_OF[mf]}: {MEAN_LABEL[mf]}", body
        )
        print(f"\nwrote {path}")

    max_se = max(v["cindex_se"] for v in summary.values())
    print(f"\nmax Monte Carlo SE on the C-index: {max_se:.4f}")
    print(f"total runtime: {(time.time() - t0) / 60:.1f} min")

    # Report any cell where the achieved censoring missed its target, since a
    # silently-missed target would make a column mean something other than its
    # heading says.
    off = [
        (k, v["achieved_censoring"])
        for k, v in summary.items()
        if abs(v["achieved_censoring"] - float(k.split("|")[3])) > 0.05
    ]
    if off:
        print("\nCells where achieved censoring missed the target by >5 points:")
        for k, a in off:
            print(f"  {k}: achieved {a:.2f}")
        print("Raise calibrate_tau's n_pilot or max_iter for these settings.")
    else:
        print("All cells hit their target censoring fraction within 5 points.")


if __name__ == "__main__":
    main()
