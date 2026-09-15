#!/usr/bin/env python3
"""
Figures 3 and 4 -- high-dimensional covariates.

Sweeps the covariate dimension p with the sample size held fixed, under both a
linear and a nonlinear signal, and writes the eight panels used by
`fig:highdim_linear` and `fig:highdim_nonlinear`.

    Figure 3   linear     amse_c-index_plots.png,  train_test_cindex.png,
                          train_test_amse.png,     loss_trajectorie.png
    Figure 4   nonlinear  the same four names with a `1` suffix

The signals are carried by the first three covariates only; the remaining p-3
are noise:

    linear              z1 - 6 z2 + 4 z3
    highdim_nonlinear   3 z1 - 6 z2 + 4 z3 + 2 z1 z2 - 3 z2^2 + sin(z3)

The quantity of interest is the train-test gap, which widens with p because the
network has more noise dimensions to fit; both curves are plotted rather than
test alone. AMSE is reported with the location shift applied, so it measures
prediction error rather than the drift of an unidentified level.

The filenames are fixed by the \\includegraphics calls in the manuscript. The
unsuffixed names are easy to collide with; nothing else in the repository
writes them.

Usage
-----
    python experiments/run_highdim_figures.py
    python experiments/run_highdim_figures.py --p-values 3 10 50 --epochs 5
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rnn_agt import data as D
from rnn_agt.seeds import make_seeds
from rnn_agt.train import TrainConfig, train_model

#: signal -> filename suffix. Figure 3 is unsuffixed, Figure 4 takes "1".
DGPS = {"linear": "", "highdim_nonlinear": "1"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p-values", type=int, nargs="+",
                    default=[3, 10, 30, 50, 100, 300, 500, 1000])
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--censoring", type=float, default=0.25)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--dependence", default="ar1")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--pair-s", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--quick", action="store_true",
                    help="Tiny run to verify the pipeline and the filenames.")
    ap.add_argument("--resume", action="store_true",
                    help="Skip fits already recorded in the checkpoint. The "
                         "large-p fits are slow, so an interruption is worth "
                         "protecting against.")
    args = ap.parse_args()

    if args.quick:
        args.p_values = [3, 10, 50]
        args.n_train = args.n_test = 250
        args.epochs, args.pair_s = 3, 10
        args.hidden, args.layers = 16, 1

    outdir = args.outdir or (
        "manuscript/images" if os.path.isdir("manuscript/images") else "results"
    )
    os.makedirs(outdir, exist_ok=True)
    n_fits = len(DGPS) * len(args.p_values)
    print(f"{n_fits} fits -> 8 panels in {outdir}\n")

    ckpt = os.path.join(outdir, "highdim.ckpt.pkl")
    done = {}
    if args.resume and os.path.exists(ckpt):
        with open(ckpt, "rb") as fh:
            done = pickle.load(fh)
        print(f"resuming: {len(done)} fits already recorded\n")

    def save_ckpt():
        tmp = ckpt + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump(done, fh)
        os.replace(tmp, ckpt)

    results, t0 = {}, time.time()
    for mf, suffix in DGPS.items():
        rows, curves = [], {}
        for p_dim in args.p_values:
            if (mf, p_dim) in done:
                m, c = done[(mf, p_dim)]
                rows.append(m); curves[p_dim] = c
                print(f"  {mf:18s} p={p_dim:5d}  (from checkpoint)", flush=True)
                continue
            seeds = make_seeds(args.seed + p_dim)
            rng = seeds.data()
            tau = D.calibrate_tau(
                args.n_train, D.MEAN_FUNCTIONS[mf], "normal", rng,
                args.censoring, D.DEPENDENCE_SPECS[args.dependence],
            )
            tr = D.make_dataset(args.n_train, mf, "normal", rng,
                                dependence=args.dependence, tau=tau, p=p_dim)
            te = D.make_dataset(args.n_test, mf, "normal", rng,
                                dependence=args.dependence, tau=tau, p=p_dim)

            cfg = TrainConfig(
                model="rnn_agt", epochs=args.epochs, pair_sample_s=args.pair_s,
                pair_batch_b=args.batch, hidden_dim=args.hidden,
                gru_layers=args.layers, lr=args.lr, device=args.device,
            )
            res = train_model(tr, te, p_dim, cfg, seeds)
            m = {"p": p_dim, **res.metrics, "params": res.n_params}
            rows.append(m)
            curves[p_dim] = res.epoch_losses
            done[(mf, p_dim)] = (m, res.epoch_losses)
            save_ckpt()
            print(f"  {mf:18s} p={p_dim:5d}  "
                  f"train C={res.metrics['train_cindex']:.3f}  "
                  f"test C={res.metrics['test_cindex']:.3f}  "
                  f"test AMSE={res.metrics['test_amse']:.2f}", flush=True)
        results[mf] = (pd.DataFrame(rows), curves)

    for mf, suffix in DGPS.items():
        df, curves = results[mf]

        fig, ax = plt.subplots(figsize=(4.6, 3.4))
        ax.plot(df.p, df.train_cindex, marker="o", label="train")
        ax.plot(df.p, df.test_cindex, marker="s", label="test")
        ax.axhline(0.5, ls="--", c="grey", lw=1)
        ax.set_xscale("log"); ax.set_xlabel("covariate dimension $p$")
        ax.set_ylabel("IPCW C-index"); ax.legend(); ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"train_test_cindex{suffix}.png"),
                    dpi=args.dpi)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(4.6, 3.4))
        ax.plot(df.p, df.train_amse, marker="o", label="train")
        ax.plot(df.p, df.test_amse, marker="s", label="test")
        ax.set_xscale("log"); ax.set_xlabel("covariate dimension $p$")
        ax.set_ylabel("AMSE"); ax.legend(); ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"train_test_amse{suffix}.png"),
                    dpi=args.dpi)
        plt.close(fig)

        # Panel (a): the two metrics against p, side by side. Test only --
        # panels (b) and (c) carry the train curves, so including them here
        # would duplicate those panels rather than add anything.
        # C-index left, AMSE right, matching the subcaption in the manuscript.
        fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2))
        axes[0].plot(df.p, df.test_cindex, marker="s", color="tab:orange")
        axes[0].axhline(0.5, ls="--", c="grey", lw=1)
        axes[0].set_xscale("log")
        axes[0].set_xlabel(r"covariate dimension $p$")
        axes[0].set_ylabel("test IPCW C-index"); axes[0].grid(alpha=.3)
        axes[1].plot(df.p, df.test_amse, marker="s", color="tab:orange")
        axes[1].set_xscale("log")
        axes[1].set_xlabel(r"covariate dimension $p$")
        axes[1].set_ylabel("test AMSE"); axes[1].grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"amse_c-index_plots{suffix}.png"),
                    dpi=args.dpi)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(4.6, 3.4))
        for p_dim in args.p_values:
            ax.plot(range(1, len(curves[p_dim]) + 1), curves[p_dim],
                    lw=1.2, label=f"p={p_dim}")
        ax.set_xlabel("epoch"); ax.set_ylabel("mini-batch Gehan-WRS loss")
        ax.legend(fontsize=6, ncol=2); ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"loss_trajectorie{suffix}.png"),
                    dpi=args.dpi)
        plt.close(fig)

        print(f"  {mf}: 4 panels written with suffix {suffix!r}")

    payload = {mf: {"metrics": df.to_dict(orient="records"),
                    "loss_curves": {str(k): v for k, v in curves.items()}}
               for mf, (df, curves) in results.items()}
    with open(os.path.join(outdir, "highdim_results.json"), "w") as fh:
        json.dump(payload, fh, indent=2)

    print(f"\n8 panels written in {(time.time() - t0) / 60:.1f} min.")
    for mf, (df, _) in results.items():
        gap = df.train_cindex - df.test_cindex
        j = int(np.argmax(gap.values))
        print(f"  {mf}: widest train-test C-index gap {gap.max():.3f} "
              f"at p={int(df.p.values[j])}")


if __name__ == "__main__":
    main()
