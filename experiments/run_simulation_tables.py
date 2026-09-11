#!/usr/bin/env python3
"""
Tables 1-3 -- the main simulation grid.

Regenerates the three simulation tables of the manuscript: one per mean
function (linear, interaction, GAM), each crossing three error distributions,
two dependence mechanisms, three censoring levels and two training sizes.

`tau` is solved for to hit each target censoring fraction rather than fixed, so
the 25/50/65% columns mean what they say. `simulation/loss_sensitivity.ipynb`
shows how sensitive these numbers are to the loss orientation and the outcome
scale, which is worth reading before interpreting the heavier-censoring columns.

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

#: Training duration is a column in Tables 1-3, and the epoch grid differs by
#: sample size: a larger training set needs fewer passes to reach the same
#: effective number of gradient steps, so reporting both at the same epoch
#: counts would not be a like-for-like comparison.
EPOCH_GRID = {1000: (5, 10, 15), 5000: (2, 3, 5)}
DEFAULT_EPOCHS = (5, 10, 15)


def epochs_for(n_train: int) -> tuple:
    """Epoch checkpoints for a given training size."""
    return EPOCH_GRID.get(n_train, DEFAULT_EPOCHS)

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
    checkpoints = epochs_for(n_train)
    cfg = TrainConfig(
        model="rnn_agt", epochs=max(checkpoints), pair_sample_s=args.pair_s,
        pair_batch_b=args.batch, hidden_dim=args.hidden,
        gru_layers=args.layers, lr=args.lr, device=args.device,
        eval_at_epochs=checkpoints,
    )
    res = train_model(train_s, test_s, 3, cfg, seeds)
    achieved = 1.0 - sum(int(s["delta"].sum()) for s in train_s) / max(
        sum(len(s["delta"]) for s in train_s), 1
    )
    return res.checkpoints, achieved


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", type=int, default=500)
    ap.add_argument("--mean-funcs", nargs="+", default=list(MEAN_FUNCS))
    ap.add_argument("--n-trains", type=int, nargs="+", default=[1000, 5000])
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--epochs", type=int, nargs="+", default=None,
                    help="Override the epoch checkpoints for every sample "
                         "size, e.g. --epochs 5 10 15. By default the grid "
                         "follows EPOCH_GRID.")
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
        args.epochs = [1, 2]

    if args.epochs:
        EPOCH_GRID.clear()
        globals()["DEFAULT_EPOCHS"] = tuple(args.epochs)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    grid = list(itertools.product(
        args.mean_funcs, ERRORS, DEPENDENCE, CENSORING, args.n_trains
    ))
    print(f"{len(grid)} cells x {args.replicates} replicates "
          f"= {len(grid) * args.replicates} fits")
    print("progress format: E<epoch>:<C-index>/<AMSE>; 'shift' is the fitted "
          "intercept\napplied before AMSE (the Gehan objective does not "
          "identify it).\n")

    acc = defaultdict(list)
    achieved_cens = defaultdict(list)
    per_n: dict = {}
    t0 = time.time()

    for k, (mf, err, dep, cens, n_tr) in enumerate(grid):
        cell_start = time.time()
        for rep in range(args.replicates):
            checkpoints, achieved = run_cell(
                args, mf, err, dep, cens, n_tr, args.seed + 1000 * rep
            )
            for ep, m in checkpoints.items():
                acc[(mf, err, dep, cens, n_tr, ep)].append(
                    (m["test_cindex"], m["test_amse"], m.get("location_shift", 0.0))
                )
            achieved_cens[(mf, err, dep, cens, n_tr)].append(achieved)

        eps = epochs_for(n_tr)
        # Report C-index and AMSE together: they answer different questions
        # and can move in opposite directions, so showing only one hides half
        # of what a cell is telling you.
        summary_bits = []
        for ep in eps:
            vals = np.array(acc[(mf, err, dep, cens, n_tr, ep)], dtype=float)
            summary_bits.append(
                f"E{ep}:{np.nanmean(vals[:, 0]):.3f}/{np.nanmean(vals[:, 1]):.2f}"
            )
        # Cost scales with the training size, and the grid alternates between
        # them, so a single running average swings wildly from cell to cell and
        # projects the wrong figure onto everything remaining. Time each
        # training size separately and project each by its own rate.
        cell_time = time.time() - cell_start
        per_n.setdefault(n_tr, []).append(cell_time)
        remaining = 0.0
        for future_n in (g[4] for g in grid[k + 1:]):
            samples = per_n.get(future_n)
            if samples:
                remaining += float(np.mean(samples))
            else:
                # not yet timed at this size: scale from a known one
                known = next(iter(per_n.items()))
                remaining += float(np.mean(known[1])) * (future_n / known[0])
        shift_bits = np.nanmean(
            [np.array(acc[(mf, err, dep, cens, n_tr, ep)], dtype=float)[:, 2].mean()
             for ep in eps]
        )
        print(
            f"[{k+1:3d}/{len(grid)}] {mf:12s} {err:9s} {dep:8s} "
            f"cens={cens:.2f} n={n_tr:5d}  " + "  ".join(summary_bits) +
            f"  (achieved cens "
            f"{np.mean(achieved_cens[(mf, err, dep, cens, n_tr)]):.2f}, "
            f"shift {shift_bits:+.2f}, "
            f"eta {remaining / 60:.0f} min)",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Emit one LaTeX fragment per mean function
    # ------------------------------------------------------------------
    summary = {}
    for key, vals in acc.items():
        arr = np.array(vals, dtype=float)
        cens_key = key[:5]
        summary["|".join(map(str, key))] = {
            "cindex": float(np.nanmean(arr[:, 0])),
            "amse": float(np.nanmean(arr[:, 1])),
            "cindex_se": float(np.nanstd(arr[:, 0], ddof=1) / np.sqrt(len(arr)))
            if len(arr) > 1 else 0.0,
            "location_shift": float(np.nanmean(arr[:, 2])) if arr.shape[1] > 2 else 0.0,
            "achieved_censoring": float(np.mean(achieved_cens[cens_key])),
            "n_replicates": len(arr),
        }

    with open(f"{args.out}.json", "w") as fh:
        json.dump({"summary": summary, "args": vars(args)}, fh, indent=2)

    for mf in args.mean_funcs:
        body = latex.simulation_table(
            acc, mf, ERRORS, DEPENDENCE, CENSORING, args.n_trains, epochs_for
        )
        header = "  %% column order: " + ", ".join(
            f"n={n}/Epoch {ep}" for n in args.n_trains for ep in epochs_for(n)
        )
        path = f"{args.out}_table{TABLE_OF[mf]}_{mf}.tex"
        latex.write_fragment(
            path, f"Table {TABLE_OF[mf]}: {MEAN_LABEL[mf]}", header + "\n" + body
        )
        print(f"\nwrote {path}")

    max_se = max(v["cindex_se"] for v in summary.values())
    print(f"\nmax Monte Carlo SE on the C-index: {max_se:.4f}")

    shifts = [abs(v["location_shift"]) for v in summary.values()]
    if shifts:
        print(f"location shift applied before AMSE: median "
              f"{np.median(shifts):.2f}, max {max(shifts):.2f}")
        print("The Gehan objective does not identify the intercept, so this is "
              "fitted\non the training residuals and applied to both partitions. "
              "AMSE without it\nmeasures level drift rather than prediction error.")
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
