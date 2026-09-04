#!/usr/bin/env python3
"""
Table 10 -- model capacity relative to sample size (Reviewer 1, #1).

Sweeps L in {1,2} and d in {8,16,32,64} on both clinical datasets under the
same repeated splits used for Tables 8 and 9, reporting the trainable
parameter count alongside the number of training subjects.  That ratio is what
the reviewer's concern is actually about: the L=2, d=64 configuration carries
several thousand parameters and was inherited from simulations at
n_train in {1000, 5000}, never re-tuned for cohorts of 128 and 403.

If the smaller configurations match or beat L=2, d=64, that is a substantive
finding rather than an inconvenience.  It would mean these cohorts cannot
identify the additional capacity, and the practical recommendation should
become to scale d to cohort size rather than to adopt the simulation-tuned
default.  The script prints the best configuration per dataset so the
comparison is unavoidable.

Usage
-----
    python experiments/run_capacity_sweep.py \
        --cgd data/cgd.csv --crc data/crc.csv --splits 200
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rnn_agt import latex
from rnn_agt.evaluation import run_repeated_splits, summarise
from rnn_agt.models import gru_parameter_count
from rnn_agt.train import TrainConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_ablation_and_splits import prepare_real_data  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cgd", required=True)
    ap.add_argument("--crc", required=True)
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--test-frac", type=float, default=0.30)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--layers", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--dims", type=int, nargs="+", default=[8, 16, 32, 64])
    ap.add_argument("--pair-s", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/table10_capacity")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    datasets = {}
    for name, path in (("cgd", args.cgd), ("crc", args.crc)):
        subs, p = prepare_real_data(path)
        n_train = int(round((1 - args.test_frac) * len(subs)))
        datasets[name] = (subs, p, n_train)
        print(f"{name}: {len(subs)} subjects, ~{n_train} in training, p={p}")

    results, param_counts = {}, {}

    for L in args.layers:
        for d in args.dims:
            cfg = TrainConfig(
                model="rnn_agt", hidden_dim=d, gru_layers=L,
                epochs=args.epochs, pair_sample_s=args.pair_s,
                pair_batch_b=args.batch, lr=args.lr, device=args.device,
            )
            results[(L, d)] = {}
            for ds, (subs, p, n_train) in datasets.items():
                outcomes = run_repeated_splits(
                    subs, p, {"rnn_agt": cfg}, args.seed,
                    n_splits=args.splits, test_frac=args.test_frac,
                )
                c = summarise(outcomes, "test_cindex")["rnn_agt"]
                a = summarise(outcomes, "test_amse")["rnn_agt"]
                results[(L, d)][ds] = {
                    "cindex_mean": c["mean"], "cindex_sd": c["sd"],
                    "amse_mean": a["mean"], "amse_sd": a["sd"],
                    "params": gru_parameter_count(p, d, L),
                    "n_train": n_train,
                    "params_per_subject": gru_parameter_count(p, d, L) / max(n_train, 1),
                }
            param_counts[(L, d)] = gru_parameter_count(
                list(datasets.values())[0][1], d, L
            )
            summary = "  ".join(
                f"{ds}: {results[(L, d)][ds]['cindex_mean']:.3f}"
                for ds in datasets
            )
            print(f"L={L} d={d:3d} params={param_counts[(L, d)]:6,}  {summary}",
                  flush=True)

    body = latex.capacity_table(results, args.layers, args.dims, param_counts)
    latex.write_fragment(f"{args.out}.tex", "Table 10: capacity sweep", body)

    with open(f"{args.out}.json", "w") as fh:
        json.dump(
            {
                "results": {f"L{L}_d{d}": v for (L, d), v in results.items()},
                "args": vars(args),
            },
            fh, indent=2, default=float,
        )

    print("\n" + body)
    print("\n----- Best configuration per dataset -----")
    for ds in datasets:
        best = max(results, key=lambda k: results[k][ds]["cindex_mean"])
        default = (2, 64)
        b, dft = results[best][ds], results[default][ds]
        gap = b["cindex_mean"] - dft["cindex_mean"]
        print(
            f"  {ds}: best L={best[0]}, d={best[1]} "
            f"(C={b['cindex_mean']:.3f}, {b['params']:,} params, "
            f"{b['params_per_subject']:.1f} per training subject)"
        )
        print(
            f"        default L=2, d=64 gives C={dft['cindex_mean']:.3f}; "
            f"gap {gap:+.3f} against an across-split sd of {dft['cindex_sd']:.3f}"
        )
        if gap > 0 and gap < dft["cindex_sd"]:
            print(
                "        The gap is inside one standard deviation, so the "
                "smaller model is not demonstrably better; report it as a wash."
            )
        elif gap > dft["cindex_sd"]:
            print(
                "        The smaller model wins by more than one sd. Update the "
                "configuration reported in Section 5.2 accordingly."
            )


if __name__ == "__main__":
    main()
