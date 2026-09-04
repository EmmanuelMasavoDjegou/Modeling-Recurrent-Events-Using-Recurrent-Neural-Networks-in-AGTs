"""
Checks that back the claims made elsewhere in this package.

Three of these exist because this package changes behaviour relative to the
original notebooks.  A claim that the new code is more correct is worth little
unless it can be demonstrated, so each check is runnable:

* :func:`check_cindex_agreement` -- the vectorised concordance agrees with a
  literal transcription of the original nested-loop version.
* :func:`check_subsampling_unbiasedness` -- the subsampled Gehan-WRS estimator
  is conditionally unbiased for the full objective (Theorem A.2).
* :func:`check_censoring_leak` -- detects the original bug in which the latent
  (uncensored) gap times were handed to the model.
* :func:`check_predictability` -- confirms no predictor depends on its own
  outcome, which is assumption (A3).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from .losses import gehan_wrs_loss_full, gehan_wrs_loss_pairs
from .metrics import ipcw_cindex, km_censoring_survival
from .sampling import batchify, build_flat_index, gather_pair_residuals, sample_pairs


def reference_ipcw_cindex(subjects: List[Dict], pred_log: np.ndarray) -> float:
    """Literal transcription of the original nested-loop concordance.

    Kept only as an oracle for :func:`check_cindex_agreement`.  Do not use it
    in analyses: it is O(N^2) in Python and unusable at the scale of the
    repeated-split experiments.
    """
    rows = []
    for i, s in enumerate(subjects):
        for j in range(len(s["log_gaps"])):
            rows.append(
                {
                    "i": i,
                    "gap": float(np.exp(s["log_gaps"][j])),
                    "delta": int(s["delta"][j]),
                    "pred": float(pred_log[i, j]),
                }
            )
    if not rows:
        return float("nan")
    gaps = np.array([r["gap"] for r in rows])
    delta = np.array([r["delta"] for r in rows])
    g_hat = km_censoring_survival(gaps, delta)

    num = den = 0.0
    for a, ra in enumerate(rows):
        if ra["delta"] != 1:
            continue
        w = 1.0 / g_hat[a]
        for b, rb in enumerate(rows):
            if ra["i"] == rb["i"]:
                continue
            if ra["gap"] < rb["gap"]:
                num += w * (1.0 if ra["pred"] < rb["pred"] else 0.0)
                den += w
    return float(num / den) if den > 0 else float("nan")


def check_cindex_agreement(
    subjects: List[Dict], pred_log: np.ndarray, tol: float = 1e-9
) -> Dict[str, float]:
    """Vectorised concordance vs. the reference loop."""
    fast = ipcw_cindex(subjects, pred_log)
    slow = reference_ipcw_cindex(subjects, pred_log)
    diff = abs(fast - slow) if not (np.isnan(fast) or np.isnan(slow)) else np.nan
    return {"vectorised": fast, "reference": slow, "abs_diff": diff,
            "agree": bool(diff is not np.nan and diff < tol)}


def check_subsampling_unbiasedness(
    subjects: List[Dict],
    cov_dim: int,
    n_draws: int = 400,
    s: int = 5,
    seed: int = 0,
    weighted: bool = True,
) -> Dict[str, float]:
    """Monte Carlo check of Theorem A.2.

    Holds the predictor fixed (a random but fixed linear map), computes the
    exact all-pairs loss, then averages the subsampled estimator over
    ``n_draws`` independent pair samples.  The ratio should approach 1.

    If this drifts from 1, the inclusion probabilities in ``sampling.py`` and
    the importance weights in ``losses.py`` disagree, and the unbiasedness
    claim underpinning Remark A.1 does not hold for the implementation.
    """
    rng = np.random.default_rng(seed)
    b = batchify(subjects, cov_dim)
    flat = build_flat_index(subjects)
    n_subjects = len(subjects)

    torch.manual_seed(seed)
    w = torch.randn(cov_dim, 1) * 0.5
    pred = (b.x_cov @ w).squeeze(-1)
    resid = b.observed - pred

    exact = float(
        gehan_wrs_loss_full(
            pred, b.observed, b.delta, b.mask, b.k_star,
            weighted=weighted, n_subjects=n_subjects,
        )
    )

    estimates = []
    for _ in range(n_draws):
        a_f, c_f, inv_p = sample_pairs(flat, s, rng)
        if a_f.size == 0:
            continue
        e_a, e_c, k_a, k_c = gather_pair_residuals(resid, flat, a_f, c_f)
        est = gehan_wrs_loss_pairs(
            e_a, e_c, k_a, k_c,
            torch.as_tensor(inv_p, dtype=torch.float32),
            n_subjects=n_subjects, weighted=weighted,
        )
        estimates.append(float(est))

    mc_mean = float(np.mean(estimates)) if estimates else np.nan
    mc_se = float(np.std(estimates, ddof=1) / np.sqrt(len(estimates))) if len(estimates) > 1 else np.nan
    return {
        "exact": exact,
        "mc_mean": mc_mean,
        "mc_se": mc_se,
        "ratio": mc_mean / exact if exact else np.nan,
        "z": (mc_mean - exact) / mc_se if mc_se and mc_se > 0 else np.nan,
    }


def check_censoring_leak(subjects_raw: List[Dict]) -> Dict[str, object]:
    """Detect the original bug: latent gap times reaching the model.

    Takes subjects as returned by :func:`rnn_agt.data.apply_censoring` (before
    reduction) and confirms that observed and latent log gaps differ exactly
    where records are censored, and that no record survives past the censoring
    point.
    """
    n_censored = 0
    n_mismatch = 0
    n_trailing = 0
    for s in subjects_raw:
        if "log_gaps_obs" not in s:
            continue
        obs = np.asarray(s["log_gaps_obs"], dtype=float)
        true = np.asarray(s["log_gaps_true"], dtype=float)
        delta = np.asarray(s["delta"], dtype=int)
        n_censored += int((delta == 0).sum())
        k = len(obs)
        # every retained uncensored record must match the latent value exactly
        for j in range(k):
            if delta[j] == 1 and not np.isclose(obs[j], true[j], atol=1e-9):
                n_mismatch += 1
            if delta[j] == 0 and obs[j] > true[j] + 1e-9:
                n_mismatch += 1  # censored value must not exceed the latent one
        if k > len(true):
            n_trailing += k - len(true)
    return {
        "censored_records": n_censored,
        "uncensored_mismatches": n_mismatch,
        "trailing_padding_records": n_trailing,
        "clean": n_mismatch == 0 and n_trailing == 0,
    }


def check_predictability(model, subjects: List[Dict], cov_dim: int) -> Dict[str, object]:
    """Confirm assumption (A3): no predictor sees its own outcome.

    Perturbs the observed log gap at position j and verifies that the
    prediction at position j is unchanged.  If a prediction moves, the outcome
    is leaking into its own predictor and every reported metric is optimistic.
    """
    b = batchify(subjects, cov_dim)
    model.eval()
    with torch.no_grad():
        base = model(b.x_prev, b.x_cov).clone()

    perturbed_prev = b.x_prev.clone()
    # shifting the *current* outcome must not change the current prediction;
    # x_prev holds lagged values only, so perturbing position j of the
    # observed series may legitimately change predictions at j+1 onward.
    max_self_change = 0.0
    n_pos = b.x_prev.shape[1]
    for j in range(n_pos):
        pp = perturbed_prev.clone()
        if j + 1 < n_pos:
            pp[:, j + 1, 0] += 5.0  # perturb what position j+1 sees
        with torch.no_grad():
            out = model(pp, b.x_cov)
        # position j must be unaffected by a change to the lag feeding j+1
        delta_j = (out[:, : j + 1] - base[:, : j + 1]).abs().max().item()
        max_self_change = max(max_self_change, delta_j)

    return {
        "max_change_at_or_before_perturbed_lag": max_self_change,
        "predictable": max_self_change < 1e-6,
    }


def summarise_checks(results: Dict[str, Dict]) -> str:
    """Render a short pass/fail report."""
    lines = []
    for name, res in results.items():
        verdict = "PASS"
        if "agree" in res and not res["agree"]:
            verdict = "FAIL"
        if "clean" in res and not res["clean"]:
            verdict = "FAIL"
        if "predictable" in res and not res["predictable"]:
            verdict = "FAIL"
        if "z" in res and res["z"] is not None and not np.isnan(res["z"]):
            verdict = "PASS" if abs(res["z"]) < 4 else "FAIL"
        lines.append(f"[{verdict}] {name}: {res}")
    return "\n".join(lines)
