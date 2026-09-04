"""
IPCW concordance and AMSE.

The original ``ipcw_cindex`` used a doubly nested ``df.iterrows()`` loop over
every pair of records.  On the CRC data that is roughly 10^6 Python-level
iterations per evaluation; across 200 repeated splits and six methods it would
not finish in reasonable time.  Both metrics are vectorised here.  The
concordance definition is unchanged, so results are comparable to the original
(see :func:`rnn_agt.diagnostics.check_cindex_agreement`, which asserts
agreement against a literal transcription of the original loop).

A note on how training-set metrics are computed, since Reviewer 2 asked.  Both
functions take predictions produced by a single forward pass in evaluation
mode after fitting is complete; neither accumulates anything during
optimization.  The censoring distribution ``G`` is re-estimated on whichever
partition is passed, so a training and a test value are not two draws of the
same quantity: their IPCW weights are on different scales.  That, plus the
small comparable-pair count on a ~90-subject partition, is why a test C-index
can exceed a training one without indicating a bug.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def _flatten_records(
    subjects: List[Dict], pred_log: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flatten to arrays of (subject id, gap, delta, prediction)."""
    subj_ids, gaps, deltas, preds = [], [], [], []
    for i, s in enumerate(subjects):
        l = len(s["log_gaps"])
        if l == 0:
            continue
        lg = np.asarray(s["log_gaps"], dtype=np.float64)
        subj_ids.append(np.full(l, i, dtype=np.int64))
        gaps.append(np.exp(lg))
        deltas.append(np.asarray(s["delta"], dtype=np.int64))
        preds.append(np.asarray(pred_log[i, :l], dtype=np.float64))
    if not subj_ids:
        z = np.array([])
        return z.astype(np.int64), z, z.astype(np.int64), z
    return (
        np.concatenate(subj_ids),
        np.concatenate(gaps),
        np.concatenate(deltas),
        np.concatenate(preds),
    )


def km_censoring_survival(
    gaps: np.ndarray, delta: np.ndarray, eval_at: Optional[np.ndarray] = None
) -> np.ndarray:
    """Kaplan-Meier estimate of the *censoring* survival function G.

    Censoring is the event of interest here, so the indicator is ``1 - delta``.
    Implemented directly rather than via lifelines to remove the dependency
    from the hot path and to make the tie handling explicit: events are
    processed before censorings at the same time, the standard convention.
    """
    if gaps.size == 0:
        return np.array([])
    order = np.argsort(gaps, kind="mergesort")
    g_sorted = gaps[order]
    cens_sorted = (1 - delta)[order].astype(np.float64)

    uniq, start_idx, counts = np.unique(
        g_sorted, return_index=True, return_counts=True
    )
    n = g_sorted.size
    at_risk = n - start_idx                       # subjects at risk at each time
    d_cens = np.add.reduceat(cens_sorted, start_idx)

    with np.errstate(divide="ignore", invalid="ignore"):
        factors = np.where(at_risk > 0, 1.0 - d_cens / at_risk, 1.0)
    surv = np.cumprod(factors)

    if eval_at is None:
        eval_at = gaps
    idx = np.searchsorted(uniq, eval_at, side="right") - 1
    out = np.where(idx >= 0, surv[np.clip(idx, 0, len(surv) - 1)], 1.0)
    return np.clip(out, 1e-6, 1.0)


def ipcw_cindex(
    subjects: List[Dict],
    pred_log: np.ndarray,
    g_hat: Optional[np.ndarray] = None,
) -> float:
    """IPCW concordance index.

    Anchors are uncensored records.  A pair contributes when the anchor's
    observed gap is shorter than the comparison's and the two come from
    different subjects; it is concordant when the anchor's predicted log gap
    is also smaller.  Each pair is weighted by ``1 / G(anchor gap)``.

    Vectorised via a sort-and-scan: for each anchor, the number of records
    with a strictly larger gap, and the number of those with a larger
    prediction, are obtained from cumulative counts rather than an inner loop.
    Same-subject pairs are subtracted off afterwards.
    """
    subj_ids, gaps, delta, preds = _flatten_records(subjects, pred_log)
    if gaps.size == 0:
        return float("nan")
    if g_hat is None:
        g_hat = km_censoring_survival(gaps, delta)

    anchors = np.flatnonzero(delta == 1)
    if anchors.size == 0:
        return float("nan")

    n = gaps.size
    order = np.argsort(gaps, kind="mergesort")
    gaps_s, preds_s, subj_s = gaps[order], preds[order], subj_ids[order]

    # rank of each record's prediction among all records, for the scan
    pred_order = np.argsort(preds_s, kind="mergesort")
    pred_rank = np.empty(n, dtype=np.int64)
    pred_rank[pred_order] = np.arange(n)

    # position of each anchor within the gap-sorted array
    pos_of = np.empty(n, dtype=np.int64)
    pos_of[order] = np.arange(n)

    # Count, for each anchor a, records k with gaps_k > gaps_a (strictly).
    # searchsorted on the sorted gaps gives the first index past the tie block.
    first_gt = np.searchsorted(gaps_s, gaps, side="right")
    n_greater = n - first_gt

    # Among those, count how many have pred > pred_anchor.  Build a Fenwick-free
    # solution: process anchors in decreasing gap order, maintaining a sorted
    # structure of predictions seen so far (those with strictly larger gaps).
    num = 0.0
    den = 0.0
    # Group records by gap value so ties are handled in blocks.
    uniq_gaps, block_start, block_count = np.unique(
        gaps_s, return_index=True, return_counts=True
    )
    n_blocks = uniq_gaps.size

    # sorted list of predictions with strictly greater gap, built from the top
    active = np.empty(n, dtype=np.float64)
    n_active = 0

    same_subj_correction_num = 0.0
    same_subj_correction_den = 0.0

    for b in range(n_blocks - 1, -1, -1):
        lo = block_start[b]
        hi = lo + block_count[b]
        # anchors within this tie block compare against everything in `active`
        for idx in range(lo, hi):
            orig = order[idx]
            if delta[orig] != 1:
                continue
            w = 1.0 / g_hat[orig]
            if n_active > 0:
                # concordant: anchor prediction strictly smaller
                cnt_greater = n_active - np.searchsorted(
                    active[:n_active], preds_s[idx], side="right"
                )
                num += w * cnt_greater
                den += w * n_active
        # then insert this block's predictions into the active set
        block_preds = np.sort(preds_s[lo:hi])
        if n_active == 0:
            active[: hi - lo] = block_preds
            n_active = hi - lo
        else:
            merged = np.empty(n_active + (hi - lo), dtype=np.float64)
            np.concatenate([active[:n_active], block_preds], out=merged)
            merged.sort()
            active[: merged.size] = merged
            n_active = merged.size

    # Remove same-subject pairs, which the scan above included.
    for i, s in enumerate(subjects):
        l = len(s["log_gaps"])
        if l < 2:
            continue
        sel = np.flatnonzero(subj_ids == i)
        gi, pi, di = gaps[sel], preds[sel], delta[sel]
        gi_a = gi[:, None]
        cmp_mask = gi_a < gi[None, :]
        anchor_mask = (di == 1)[:, None]
        both = cmp_mask & anchor_mask
        if not both.any():
            continue
        wi = (1.0 / g_hat[sel])[:, None]
        conc = (pi[:, None] < pi[None, :]) & both
        same_subj_correction_num += float((wi * conc).sum())
        same_subj_correction_den += float((wi * both).sum())

    num -= same_subj_correction_num
    den -= same_subj_correction_den
    return float(num / den) if den > 0 else float("nan")


def amse(
    subjects: List[Dict],
    pred_log: np.ndarray,
    g_hat: Optional[np.ndarray] = None,
) -> float:
    """IPCW-adjusted mean squared error on the log gap-time scale.

    Restricted to uncensored records by ``delta``, so the observed and latent
    log gap times coincide in every contributing term (Section 4.2).
    """
    subj_ids, gaps, delta, preds = _flatten_records(subjects, pred_log)
    if gaps.size == 0:
        return float("nan")
    if g_hat is None:
        g_hat = km_censoring_survival(gaps, delta)
    log_gaps = np.log(gaps)
    contrib = (delta / g_hat) * (log_gaps - preds) ** 2
    denom = float(delta.sum())
    return float(contrib.sum() / denom) if denom > 0 else float("nan")


def evaluate(subjects: List[Dict], pred_log: np.ndarray) -> Dict[str, float]:
    """Both metrics, sharing one estimate of G."""
    subj_ids, gaps, delta, preds = _flatten_records(subjects, pred_log)
    if gaps.size == 0:
        return {"cindex": float("nan"), "amse": float("nan")}
    g_hat = km_censoring_survival(gaps, delta)
    return {
        "cindex": ipcw_cindex(subjects, pred_log, g_hat=g_hat),
        "amse": amse(subjects, pred_log, g_hat=g_hat),
    }
