#!/usr/bin/env python3
"""
Table 4 -- sub-sampling configuration sensitivity.

Sweeps the pairs-per-anchor `s` and the pair mini-batch size `b`, reporting
test-set C-index and AMSE at two training sizes and their respective epoch
checkpoints, together with runtime.

WHAT THE TABLE IS FOR
---------------------
Theorem A.2 makes the sub-sampled objective conditionally unbiased for the full
loss at every `s`, so `s` controls the *variance* of the gradient estimate, not
its bias. The question the table answers is therefore practical rather than
inferential: how small can `s` be before the extra gradient noise costs
accuracy, given that per-epoch cost scales as O(n' s)?

The driver prints the smallest `s` whose C-index is within one Monte Carlo
standard error of the best, which is the defensible basis for choosing the
default rather than picking the largest value that fits the compute budget.

Usage
-----
    python experiments/run_subsampling_sensitivity.py --replicates 50
    python experiments/run_subsampling_sensitivity.py --quick
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rnn_agt import data as D
from rnn_agt import latex
from rnn_agt.seeds import make_seeds
from rnn_agt.train import TrainConfig, train_model

#: Rows of Table 4.
S_GRID = (5, 10, 15)
B_GRID = (32, 64, 128)
#: Columns: epoch checkpoints per training size, as in Tables 1-3.
N_EPOCHS = {1000: (5, 10), 5000: (2, 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", type=int, default=50)
    ap.add_argument("--s-grid", type=int, nargs="+", default=list(S_GRID))
    ap.add_argument("--b-grid", type=int, nargs="+", default=list(B_GRID))
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--mean-func", default="interaction")
    ap.add_argument("--error", default="normal")
    ap.add_argument("--dependence", default="ar1")
    ap.add_argument("--censoring", type=float, default=0.50)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/table4_subsampling")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.replicates = 2
        args.s_grid, args.b_grid = [5, 10], [32, 64]
        args.n_test = 200

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    n_trains = sorted(N_EPOCHS)
    grid = [(s, b, n) for s in args.s_grid for b in args.b_grid for n in n_trains]

    acc = defaultdict(list)
    runtime = defaultdict(list)
    done: set = set()
    ckpt = f"{args.out}.ckpt.pkl"
    if args.resume and os.path.exists(ckpt):
        with open(ckpt, "rb") as fh:
            st = pickle.load(fh)
        acc = defaultdict(list, st["acc"])
        runtime = defaultdict(list, st["runtime"])
        done = st["done"]
        print(f"resuming: {len(done)}/{len(grid)} cells complete\n")

    def save():
        tmp = ckpt + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump({"acc": dict(acc), "runtime": dict(runtime),
                         "done": done, "args": vars(args)}, fh)
        os.replace(tmp, ckpt)

    print(f"{len(grid)} cells x {args.replicates} replicates "
          f"= {len(grid) * args.replicates} fits")
    print("progress format: E<epoch>:<C-index>/<AMSE>\n")

    t0 = time.time()
    for k, (s_val, b_val, n_tr) in enumerate(grid):
        if (s_val, b_val, n_tr) in done:
            continue
        checkpoints = N_EPOCHS[n_tr]
        cell_start = time.time()

        for rep in range(args.replicates):
            seeds = make_seeds(args.seed + 1000 * rep)
            rng = seeds.data()
            tau = D.calibrate_tau(
                n_tr, D.MEAN_FUNCTIONS[args.mean_func], args.error, rng,
                args.censoring, D.DEPENDENCE_SPECS[args.dependence],
            )
            tr = D.make_dataset(n_tr, args.mean_func, args.error, rng,
                                dependence=args.dependence, tau=tau)
            te = D.make_dataset(args.n_test, args.mean_func, args.error, rng,
                                dependence=args.dependence, tau=tau)
            cfg = TrainConfig(
                model="rnn_agt", epochs=max(checkpoints), pair_sample_s=s_val,
                pair_batch_b=b_val, hidden_dim=args.hidden,
                gru_layers=args.layers, lr=args.lr, device=args.device,
                eval_at_epochs=checkpoints,
            )
            res = train_model(tr, te, 3, cfg, seeds)
            for ep, m in res.checkpoints.items():
                acc[(s_val, b_val, n_tr, ep)].append(
                    (m["test_cindex"], m["test_amse"])
                )

        runtime[(s_val, b_val, n_tr)].append(
            (time.time() - cell_start) / args.replicates
        )
        done.add((s_val, b_val, n_tr))
        save()

        bits = []
        for ep in checkpoints:
            arr = np.array(acc[(s_val, b_val, n_tr, ep)], dtype=float)
            bits.append(f"E{ep}:{np.nanmean(arr[:, 0]):.3f}/"
                        f"{np.nanmean(arr[:, 1]):.2f}")
        elapsed = time.time() - t0
        eta = elapsed / (k + 1) * (len(grid) - k - 1)
        print(f"[{k+1:2d}/{len(grid)}] s={s_val:3d} b={b_val:4d} n={n_tr:5d}  "
              + "  ".join(bits)
              + f"  ({runtime[(s_val, b_val, n_tr)][0]:.1f}s/fit, "
                f"eta {eta/60:.0f} min)", flush=True)

    # ------------------------------------------------------------------
    results, ses = {}, []
    for key, vals in acc.items():
        arr = np.array(vals, dtype=float)
        results[key] = {
            "cindex": float(np.nanmean(arr[:, 0])),
            "amse": float(np.nanmean(arr[:, 1])),
        }
        if len(arr) > 1:
            ses.append(float(np.nanstd(arr[:, 0], ddof=1) / np.sqrt(len(arr))))
    max_se = max(ses) if ses else 0.0

    body = latex.subsampling_table(results, args.s_grid, args.b_grid,
                                   {n: N_EPOCHS[n] for n in n_trains})
    latex.write_fragment(f"{args.out}.tex",
                         "Table 4: sub-sampling sensitivity", body)
    with open(f"{args.out}.json", "w") as fh:
        json.dump({"results": {"|".join(map(str, k)): v
                               for k, v in results.items()},
                   "raw": {"|".join(map(str, k)): [list(map(float, v))
                                                   for v in vals]
                           for k, vals in acc.items()},
                   "runtime": {"|".join(map(str, k)): v
                               for k, v in runtime.items()},
                   "max_cindex_se": max_se, "args": vars(args)},
                  fh, indent=2)

    print("\n" + body)
    print(f"\nmax Monte Carlo SE on the C-index: {max_se:.4f}")

    # ---- which configuration to adopt -------------------------------
    # Report the cheapest setting statistically indistinguishable from the
    # best, rather than the best, since s buys variance reduction at linear
    # cost and the table exists to find where that stops paying.
    ref_n, ref_ep = n_trains[0], N_EPOCHS[n_trains[0]][-1]
    scores = {(s, b): results[(s, b, ref_n, ref_ep)]["cindex"]
              for s in args.s_grid for b in args.b_grid
              if (s, b, ref_n, ref_ep) in results}
    if scores:
        best_cfg = max(scores, key=scores.get)
        best = scores[best_cfg]
        within = [cfg for cfg, v in scores.items() if v >= best - max_se]
        cheapest = min(within, key=lambda c: (c[0], c[1]))
        print(f"\nAt n={ref_n}, epoch {ref_ep}:")
        print(f"  best        s={best_cfg[0]}, b={best_cfg[1]}  "
              f"C={best:.4f}")
        print(f"  cheapest within 1 SE  s={cheapest[0]}, b={cheapest[1]}  "
              f"C={scores[cheapest]:.4f}")
        print(f"  mean runtime at that setting: "
              f"{np.mean(runtime.get((cheapest[0], cheapest[1], ref_n), [np.nan])):.1f}s/fit")
        print("\nAdopt the cheapest setting within one standard error unless "
              "there is a\nreason to prefer otherwise, and state it in "
              "Section 4.2 and the Table 4 footnote.")


if __name__ == "__main__":
    main()
