#!/usr/bin/env python3
"""
Table 7 -- the two rows added to the existing benchmark table (AE #1).

Fits AFT-WRS and NN-AFT on the same *single fixed split* used in the original
submission, so the new rows sit on the same footing as the PWP-GT, PWP-TT, WLW
and RNN-AGT numbers already in Table 7.

These numbers do not support any comparative claim, and the manuscript's
revised caption says so.  They exist for continuity with the original
submission; the comparisons rest on Table 9.  If you find yourself quoting a
number from here in the discussion, quote Table 9 instead.

Usage
-----
    python experiments/run_benchmark_rows.py \
        --cgd data/cgd.csv --crc data/crc.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rnn_agt import latex  # noqa: E402
from rnn_agt.evaluation import stratified_split  # noqa: E402
from rnn_agt.seeds import make_seeds  # noqa: E402
from rnn_agt.train import TrainConfig, train_model  # noqa: E402
from run_ablation_and_splits import prepare_real_data  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cgd", required=True)
    ap.add_argument("--crc", required=True)
    ap.add_argument("--test-frac", type=float, default=0.30)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--pair-s", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-linear", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=42,
                    help="Fixed seed matching the original submission's split.")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/table7_rows")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    configs = {
        "aft_wrs": TrainConfig(
            model="aft_wrs", lr=args.lr_linear, epochs=args.epochs,
            pair_sample_s=args.pair_s, device=args.device,
        ),
        "nn_aft": TrainConfig(
            model="nn_aft", lr=args.lr, epochs=args.epochs,
            hidden_dim=args.hidden, gru_layers=args.layers,
            pair_sample_s=args.pair_s, device=args.device,
        ),
    }

    results = {name: {} for name in configs}
    for ds, path in (("cgd", args.cgd), ("crc", args.crc)):
        subs, p = prepare_real_data(path)
        seeds = make_seeds(args.seed)
        tr_idx, te_idx = stratified_split(subs, args.test_frac, seeds.split())
        train_s = [subs[i] for i in tr_idx]
        test_s = [subs[i] for i in te_idx]
        print(f"{ds}: {len(train_s)} train / {len(test_s)} test subjects")

        for name, cfg in configs.items():
            res = train_model(train_s, test_s, p, cfg, make_seeds(args.seed))
            results[name][ds] = res.metrics["test_cindex"]
            print(f"  {name:8s} test C-index {res.metrics['test_cindex']:.3f}")

    body = latex.benchmark_rows(results)
    latex.write_fragment(f"{args.out}.tex", "Table 7: two added rows", body)
    with open(f"{args.out}.json", "w") as fh:
        json.dump({"results": results, "args": vars(args)}, fh, indent=2)

    print("\n" + body)


if __name__ == "__main__":
    main()
