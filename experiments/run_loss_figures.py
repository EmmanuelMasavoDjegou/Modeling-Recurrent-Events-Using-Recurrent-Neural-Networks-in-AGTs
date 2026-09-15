#!/usr/bin/env python3
"""
Figure 2 -- mini-batch training loss across all settings.

Produces the nine panels of `fig:loss_all_layout`: training loss across epochs
for every combination of mean function and error distribution, at
n_train = 1,000 with 25% incomplete follow-up.

    row a-c   interaction   loss_{normal,gumbel,log}_nonlinear.png
    row d-f   GAM-type      loss_{normal,gumbel,log}_n1000_gam.png
    row g-i   linear        loss_{normal,gumbel,log}_n1000_linear.png

`log` abbreviates the *logistic* error distribution, not a log transform. The
filenames are fixed by the \\includegraphics calls in the manuscript and must
not be changed.

Only training loss is plotted, so no test set is generated and no held-out
evaluation is run. Note that the loss is on the Gehan-WRS scale, whose level is
not comparable across error families: their variances differ by more than
threefold, so only the shape of each trajectory is comparable.

Usage
-----
    python experiments/run_loss_figures.py
    python experiments/run_loss_figures.py --epochs 5 --n-train 300   # quick check
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")          # headless: safe under nohup
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rnn_agt import data as D
from rnn_agt.seeds import make_seeds
from rnn_agt.train import TrainConfig, train_model

#: mean function -> filename fragment used by the manuscript
MEAN_FUNCS = {"interaction": "nonlinear",
              "gam": "n1000_gam",
              "linear": "n1000_linear"}
#: error distribution -> filename fragment
ERRORS = {"normal": "normal", "gumbel": "gumbel", "logistic": "log"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--censoring", type=float, default=0.25)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--dependence", default="ar1")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--pair-s", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--outdir", default=None,
                    help="Defaults to manuscript/images if it exists, else "
                         "results/.")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--quick", action="store_true",
                    help="Tiny run to verify the pipeline and the filenames.")
    args = ap.parse_args()

    if args.quick:
        args.n_train, args.epochs, args.pair_s = 250, 3, 10
        args.hidden, args.layers = 16, 1

    outdir = args.outdir or (
        "manuscript/images" if os.path.isdir("manuscript/images") else "results"
    )
    os.makedirs(outdir, exist_ok=True)
    print(f"{len(MEAN_FUNCS) * len(ERRORS)} panels -> {outdir}\n")

    losses, t0 = {}, time.time()
    for mf in MEAN_FUNCS:
        for err in ERRORS:
            seeds = make_seeds(args.seed)
            rng = seeds.data()
            tau = D.calibrate_tau(
                args.n_train, D.MEAN_FUNCTIONS[mf], err, rng,
                args.censoring, D.DEPENDENCE_SPECS[args.dependence],
            )
            tr = D.make_dataset(args.n_train, mf, err, rng,
                                dependence=args.dependence, tau=tau)

            cfg = TrainConfig(
                model="rnn_agt", epochs=args.epochs, pair_sample_s=args.pair_s,
                pair_batch_b=args.batch, hidden_dim=args.hidden,
                gru_layers=args.layers, lr=args.lr, device=args.device,
            )
            res = train_model(tr, None, 3, cfg, seeds)
            losses[(mf, err)] = res.epoch_losses
            print(f"  {mf:12s} {err:9s} "
                  f"loss {res.epoch_losses[0]:.3f} -> {res.epoch_losses[-1]:.3f}",
                  flush=True)

    for (mf, err), curve in losses.items():
        fname = f"loss_{ERRORS[err]}_{MEAN_FUNCS[mf]}.png"
        fig, ax = plt.subplots(figsize=(4.2, 3.0))
        ax.plot(range(1, len(curve) + 1), curve, lw=1.4, color="tab:blue")
        ax.set_xlabel("epoch")
        ax.set_ylabel("mini-batch Gehan-WRS loss")
        # No in-plot title: the manuscript figure already labels every panel,
        # with a row heading per mean function and "(a) Normal errors" beneath
        # each panel, so a title would duplicate it.
        ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, fname), dpi=args.dpi)
        plt.close(fig)
        print("  wrote", fname)

    with open(os.path.join(outdir, "loss_curves.json"), "w") as fh:
        json.dump({f"{mf}|{err}": c for (mf, err), c in losses.items()},
                  fh, indent=2)

    print(f"\n9 panels written in {(time.time() - t0) / 60:.1f} min.")
    print("The raw curves are saved alongside them as loss_curves.json, so the")
    print("figure can be restyled without refitting.")


if __name__ == "__main__":
    main()
