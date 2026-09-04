#!/usr/bin/env python3
"""Run every correctness check and print a pass/fail report.

Run this first, before any experiment. If a check fails, the numbers produced
by the drivers cannot be trusted, and the failure should be resolved rather
than worked around.
"""
from __future__ import annotations
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rnn_agt import data as D
from rnn_agt.diagnostics import (check_cindex_agreement, check_censoring_leak,
                                 check_predictability, check_subsampling_unbiasedness,
                                 summarise_checks)
from rnn_agt.models import RNNAGT
from rnn_agt.seeds import make_seeds

def main() -> None:
    seeds = make_seeds(20260903)
    rng = seeds.data()
    raw = D.generate_subjects(80, D.f_interaction, "normal", rng,
                              D.DEPENDENCE_SPECS["ar1"])
    raw = D.apply_censoring(raw, 50.0, rng)
    subs = D.to_model_subjects(raw)
    max_seq = max(len(s["log_gaps"]) for s in subs)
    pred = np.random.default_rng(0).normal(size=(len(subs), max_seq))

    seeds.seed_torch()
    model = RNNAGT(3, hidden_dim=16, gru_layers=1)

    results = {
        "vectorised C-index matches reference loop": check_cindex_agreement(subs, pred),
        "censoring reaches the outcome (no latent leak)": check_censoring_leak(raw),
        "subsampled Gehan-WRS is unbiased (weighted)":
            check_subsampling_unbiasedness(subs, 3, n_draws=500, s=4),
        "subsampled Gehan-WRS is unbiased (unweighted)":
            check_subsampling_unbiasedness(subs, 3, n_draws=500, s=4, weighted=False),
        "predictor is history-predictable (A3)": check_predictability(model, subs, 3),
    }
    print(summarise_checks(results))

if __name__ == "__main__":
    main()
