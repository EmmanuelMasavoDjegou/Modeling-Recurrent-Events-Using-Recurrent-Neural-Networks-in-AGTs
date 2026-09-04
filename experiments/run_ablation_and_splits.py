#!/usr/bin/env python3
"""
Tables 8 and 9 -- ablation ladder and repeated splits (AE #1, Reviewer 2 #4).

Both tables come from the same B repeated stratified splits, so they are
internally consistent: the ablation increments in Table 8 are paired
differences computed on exactly the splits summarised in Table 9.  Running them
separately would risk reporting a delta that does not equal the difference of
the two means shown elsewhere.

Real data are loaded from CSVs produced by the R preprocessing scripts (see
``--cgd`` and ``--crc``).  Expected columns:

    subject_id, gap_time, delta, <covariate columns...>

one row per gap, ordered within subject.  ``prepare_real_data`` converts that
to the internal subject-list representation.

Usage
-----
    python experiments/run_ablation_and_splits.py \
        --cgd data/cgd.csv --crc data/crc.csv --splits 200
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rnn_agt import latex
from rnn_agt.cox_bridge import (
    COX_LABELS,
    COX_MODELS,
    check_orientation,
    load_cox_predictions,
    merge_into_outcomes,
    score_cox_predictions,
    write_splits,
)
from rnn_agt.evaluation import (
    paired_difference,
    run_cross_validation,
    run_repeated_splits,
    summarise,
)
from rnn_agt.train import TrainConfig

MODELS = ("aft_wrs", "nn_aft", "rnn_agt")
MODEL_LABELS = {"aft_wrs": "AFT-WRS", "nn_aft": "NN-AFT", "rnn_agt": "RNN-AGT"}


def prepare_real_data(
    csv_path: str,
    id_col: str = "id",
    gap_col: str = "gap_time",
    delta_col: str = "delta",
    covariates: List[str] | None = None,
    standardize: bool = True,
    return_ids: bool = False,
):
    """Load a preprocessed gap-time CSV into the subject-list representation.

    Standardizing covariates is on by default.  It does not change AFT-WRS
    (a linear predictor absorbs the scaling) but it materially affects the
    networks, whose initialization assumes inputs of order one.  Leaving raw
    covariates of wildly different scales is a common reason a network
    underperforms a linear model for reasons unrelated to model class.
    """
    df = pd.read_csv(csv_path)
    for col in (id_col, gap_col, delta_col):
        if col not in df.columns:
            raise ValueError(f"{csv_path}: missing required column {col!r}")

    if covariates is None:
        # `event` duplicates `delta` and is kept only for the R models; if it
        # were left in, it would be silently used as a covariate and leak the
        # outcome into the predictor. Bookkeeping columns are excluded too.
        reserved = {id_col, gap_col, delta_col,
                    "event", "status", "time", "stop", "start", "order", "tau",
                    "gtime", "enum"}
        covariates = [c for c in df.columns if c not in reserved]
    if not covariates:
        raise ValueError(f"{csv_path}: no covariate columns found")

    if (df[gap_col] <= 0).any():
        n_bad = int((df[gap_col] <= 0).sum())
        raise ValueError(
            f"{csv_path}: {n_bad} non-positive gap times; the model works on "
            "the log scale, so these must be resolved upstream rather than "
            "silently clipped"
        )

    X = df[covariates].to_numpy(dtype=np.float64)
    if standardize:
        mu, sd = X.mean(axis=0), X.std(axis=0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        X = (X - mu) / sd
    df = df.assign(**{c: X[:, i] for i, c in enumerate(covariates)})

    subjects, ids = [], []
    for sid, grp in df.groupby(id_col, sort=True):
        subjects.append(
            {
                "covariates": grp[covariates].to_numpy(dtype=np.float64)[0],
                "log_gaps": np.log(grp[gap_col].to_numpy(dtype=np.float64)),
                "delta": grp[delta_col].to_numpy(dtype=np.int64),
            }
        )
        ids.append(sid)
    if return_ids:
        return subjects, len(covariates), ids
    return subjects, len(covariates)


def build_configs(args) -> Dict[str, TrainConfig]:
    common = dict(
        epochs=args.epochs, pair_sample_s=args.pair_s,
        pair_batch_b=args.batch, device=args.device,
    )
    return {
        "aft_wrs": TrainConfig(model="aft_wrs", lr=args.lr_linear, **common),
        "nn_aft": TrainConfig(model="nn_aft", lr=args.lr, hidden_dim=args.hidden,
                              gru_layers=args.layers, **common),
        "rnn_agt": TrainConfig(model="rnn_agt", lr=args.lr, hidden_dim=args.hidden,
                               gru_layers=args.layers, **common),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id-col", default="id",
                    help="Subject id column; must match what the R scripts export.")
    ap.add_argument("--cgd", required=True, help="CSV for the CGD data")
    ap.add_argument("--crc", required=True, help="CSV for the CRC readmission data")
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--test-frac", type=float, default=0.30)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--pair-s", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-linear", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/tables89")
    ap.add_argument("--splits-out", default="results/splits",
                    help="Directory for split assignment CSVs handed to R.")
    ap.add_argument("--cox-lp-cgd", default=None,
                    help="Cox linear predictors for CGD, from the R script.")
    ap.add_argument("--cox-lp-crc", default=None,
                    help="Cox linear predictors for CRC, from the R script.")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    configs = build_configs(args)

    datasets = {}
    subject_ids = {}
    for name, path in (("cgd", args.cgd), ("crc", args.crc)):
        subs, p, ids = prepare_real_data(path, return_ids=True)
        datasets[name] = (subs, p)
        subject_ids[name] = ids
        n_rec = sum(len(s["log_gaps"]) for s in subs)
        n_ev = sum(int(s["delta"].sum()) for s in subs)
        print(f"{name}: {len(subs)} subjects, {n_rec} records, {n_ev} events, p={p}")

    split_summary, cv_summary, deltas, ablation = {}, {}, {}, {}
    all_outcomes = {}

    for ds, (subs, p) in datasets.items():
        print(f"\n=== {ds}: {args.splits} repeated splits ===")

        def progress(done, total, _ds=ds):
            if done % 10 == 0 or done == total:
                print(f"  {_ds}: split {done}/{total}", flush=True)

        # Export the assignments so the R Cox script sees identical splits.
        # Written before fitting so the R job can run concurrently.
        splits_path = os.path.join(args.splits_out, f"splits_{ds}.csv")
        splits_df = write_splits(
            subs, subject_ids[ds], splits_path, args.seed,
            n_splits=args.splits, test_frac=args.test_frac,
        )
        print(f"  split assignments -> {splits_path}")

        outcomes = run_repeated_splits(
            subs, p, configs, args.seed, n_splits=args.splits,
            test_frac=args.test_frac, progress=progress,
        )

        cox_path = getattr(args, f"cox_lp_{ds}")
        if cox_path and os.path.exists(cox_path):
            cox_scores = score_cox_predictions(
                load_cox_predictions(cox_path), subs, subject_ids[ds], splits_df
            )
            warning = check_orientation(cox_scores)
            if warning:
                print(f"  WARNING ({ds}): {warning}")
            outcomes = merge_into_outcomes(outcomes, cox_scores)
            print(f"  merged Cox scores from {cox_path}")
        else:
            print(
                f"  no Cox predictors for {ds}; the WLW/PWP-TT/PWP-GT rows of\n"
                f"  Table 9 will be blank. Run the R script on {splits_path}\n"
                f"  and re-run with --cox-lp-{ds}."
            )

        all_outcomes[ds] = outcomes

        split_summary[ds] = summarise(outcomes, "test_cindex")
        ablation[ds] = {}
        for m in MODELS:
            for metric, key in (("test_cindex", "cindex"), ("test_amse", "amse")):
                ablation[ds][f"{m}:{key}"] = summarise(outcomes, metric)[m]

        deltas[ds] = {}
        for contrast, (a, b) in (
            ("nonlinearity", ("nn_aft", "aft_wrs")),
            ("history", ("rnn_agt", "nn_aft")),
        ):
            for metric, key in (("test_cindex", "cindex"), ("test_amse", "amse")):
                deltas[ds][f"{contrast}:{key}"] = paired_difference(
                    outcomes, a, b, metric
                )

        print(f"=== {ds}: {args.folds}-fold CV ===")
        cv_summary[ds] = run_cross_validation(
            subs, p, configs, args.seed, k=args.folds
        )

    payload = {
        "split_summary": split_summary,
        "cv_summary": cv_summary,
        "ablation": ablation,
        "deltas": deltas,
        "args": vars(args),
        "raw": {
            ds: [{"split": o.split_id, "metrics": o.metrics} for o in outs]
            for ds, outs in all_outcomes.items()
        },
    }
    with open(f"{args.out}.json", "w") as fh:
        json.dump(payload, fh, indent=2, default=float)

    t7 = latex.ablation_table(ablation, deltas)
    latex.write_fragment(f"{args.out}_table8_ablation.tex",
                         "Table 8: ablation ladder", t7)

    have_cox = any(
        m in split_summary.get(ds, {}) for ds in datasets for m in COX_MODELS
    )
    table8_models = (COX_MODELS + MODELS) if have_cox else MODELS
    table8_labels = dict(MODEL_LABELS, **COX_LABELS)
    t8 = latex.repeated_splits_table(
        split_summary, cv_summary, table8_models, table8_labels
    )
    latex.write_fragment(f"{args.out}_table9_splits.tex",
                         "Table 9: repeated splits and cross-validation", t8)

    print("\n----- Table 8 -----\n" + t7)
    print("\n----- Table 9 -----\n" + t8)

    print("\n----- Paired increments -----")
    for ds in datasets:
        for contrast in ("nonlinearity", "history"):
            d = deltas[ds][f"{contrast}:cindex"]
            print(
                f"  {ds:4s} delta-{contrast:13s} "
                f"{latex.fmt_diff_ci(d)}  win rate {d['win_rate']:.2f}"
            )
    print(
        "\nRead the win rate alongside the mean: a positive mean increment with "
        "a win rate near 0.5 means the capability is not reliably helping."
    )

    if have_cox:
        print("\n----- RNN-AGT vs. best Cox competitor (paired) -----")
        for ds in datasets:
            best = max(
                COX_MODELS,
                key=lambda m: split_summary[ds].get(m, {}).get("mean", -np.inf),
            )
            d = paired_difference(all_outcomes[ds], "rnn_agt", best, "test_cindex")
            print(
                f"  {ds:4s} vs {COX_LABELS[best]:7s} "
                f"{latex.fmt_diff_ci(d)}  win rate {d['win_rate']:.2f}"
            )
        print(
            "\nThese are the numbers for the Table 9 footnote. Both sides now\n"
            "come from the same splits and the same IPCW estimator, unlike the\n"
            "single-split comparison retained in Table 7."
        )


if __name__ == "__main__":
    main()
