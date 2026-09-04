#!/usr/bin/env python3
"""
Table 5 -- robustness to the within-subject dependence mechanism (Reviewer 2, #3).

Runs the interaction mean function at n_train = 1,000 with 50% incomplete
follow-up under five dependence mechanisms, fitting all three rungs of the
ablation ladder under each.

The AR(2) row carries the most diagnostic weight.  NN-AFT conditions on
baseline covariates only and so *cannot* represent lag-two dependence at all,
whereas RNN-AGT can.  A gap that widens at AR(2) relative to AR(1) is direct
evidence that the recurrent state is doing the work claimed for it.  If instead
the RNN-AGT advantage narrows once the dependence stops being linear and
first-order, that is evidence the GRU was exploiting the AR(1) structure
specifically, and the manuscript commits to reporting it either way.

Usage
-----
    python experiments/run_dependence_robustness.py --replicates 500
    python experiments/run_dependence_robustness.py --replicates 5 --quick
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rnn_agt import data as D
from rnn_agt import latex
from rnn_agt.seeds import make_seeds
from rnn_agt.train import TrainConfig, train_model

MECHANISMS = ("frailty", "ar1", "nar1", "ar2", "event_dependent")
MODELS = ("aft_wrs", "nn_aft", "rnn_agt")


def build_configs(args) -> dict:
    common = dict(
        epochs=args.epochs,
        pair_sample_s=args.pair_s,
        pair_batch_b=args.batch,
        device=args.device,
    )
    return {
        "aft_wrs": TrainConfig(model="aft_wrs", lr=args.lr_linear, **common),
        "nn_aft": TrainConfig(
            model="nn_aft", lr=args.lr, hidden_dim=args.hidden,
            gru_layers=args.layers, **common
        ),
        "rnn_agt": TrainConfig(
            model="rnn_agt", lr=args.lr, hidden_dim=args.hidden,
            gru_layers=args.layers, **common
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", type=int, default=500)
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--censoring", type=float, default=0.50)
    ap.add_argument("--mean-func", default="interaction")
    ap.add_argument("--error", default="normal")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--pair-s", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-linear", type=float, default=1e-2,
                    help="AFT-WRS needs a larger step; see README.")
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/table5_dependence")
    ap.add_argument("--quick", action="store_true",
                    help="tiny run to verify the pipeline")
    args = ap.parse_args()

    if args.quick:
        args.replicates, args.n_train, args.n_test, args.epochs = 2, 150, 150, 2

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    configs = build_configs(args)

    acc = defaultdict(lambda: defaultdict(list))
    for rep in range(args.replicates):
        seeds = make_seeds(args.seed + rep)
        rng = seeds.data()
        for mech in MECHANISMS:
            tau = D.calibrate_tau(
                args.n_train, D.MEAN_FUNCTIONS[args.mean_func], args.error,
                rng, args.censoring, D.DEPENDENCE_SPECS[mech],
            )
            train_s = D.make_dataset(
                args.n_train, args.mean_func, args.error, rng,
                dependence=mech, tau=tau,
            )
            test_s = D.make_dataset(
                args.n_test, args.mean_func, args.error, rng,
                dependence=mech, tau=tau,
            )
            for name, cfg in configs.items():
                res = train_model(train_s, test_s, 3, cfg, seeds)
                acc[mech][name].append(
                    (res.metrics["test_cindex"], res.metrics["test_amse"])
                )
        print(f"replicate {rep + 1}/{args.replicates} done", flush=True)

    results, raw = {}, {}
    for mech in MECHANISMS:
        results[mech] = {}
        raw[mech] = {}
        for name in MODELS:
            arr = np.array(acc[mech][name], dtype=float)
            results[mech][name] = {
                "test_cindex": float(np.nanmean(arr[:, 0])),
                "test_amse": float(np.nanmean(arr[:, 1])),
                "cindex_se": float(np.nanstd(arr[:, 0], ddof=1) / np.sqrt(len(arr)))
                if len(arr) > 1 else 0.0,
            }
            raw[mech][name] = arr.tolist()

    with open(f"{args.out}.json", "w") as fh:
        json.dump({"summary": results, "raw": raw, "args": vars(args)}, fh, indent=2)

    body = latex.dependence_table(
        results, MECHANISMS,
        {m: D.DEPENDENCE_SPECS[m].label for m in MECHANISMS},
    )
    latex.write_fragment(
        f"{args.out}.tex", "Table 5: dependence-mechanism robustness", body
    )

    max_se = max(
        results[m][n]["cindex_se"] for m in MECHANISMS for n in MODELS
    )
    print("\n" + body)
    print(f"\nmax Monte Carlo SE on the C-index: {max_se:.4f}")
    print(f"wrote {args.out}.json and {args.out}.tex")


if __name__ == "__main__":
    main()
