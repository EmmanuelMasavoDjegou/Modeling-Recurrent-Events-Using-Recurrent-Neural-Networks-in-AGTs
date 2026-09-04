"""
Pair subsampling and batching.

Two changes from the original implementation.

**Vectorised sampling.**  The original ``sample_pairs_from_predictions`` called
``np.delete(all_flat, flat)`` inside a Python loop over every uncensored
record, allocating a fresh N-element array per anchor.  That is O(n' * N) work
and memory churn per epoch, and it dominated runtime on the larger settings.
Here the pool is sampled directly and self-pairs are resampled, which is O(m).

**Explicit inclusion probabilities.**  The original scaled the loss by a scalar
``(N_total - 1) / s``.  That is correct only under exactly uniform sampling
without replacement from a fixed pool.  Carrying per-pair ``inv_prob`` instead
keeps the estimator unbiased under non-uniform or with-replacement sampling,
and makes the unbiasedness claim in Theorem A.2 checkable rather than implicit
(see :func:`rnn_agt.diagnostics.check_subsampling_unbiasedness`).
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Tuple

import numpy as np
import torch


class FlatIndex(NamedTuple):
    """Flattened view of all retained records across subjects."""

    n_total: int
    subj_of_flat: np.ndarray   # (N,) subject index
    pos_of_flat: np.ndarray    # (N,) within-subject event index
    delta_of_flat: np.ndarray  # (N,) censoring indicator
    k_star: np.ndarray         # (n,) K_i* per subject
    prefix: np.ndarray         # (n+1,) offsets into the flat arrays


def build_flat_index(subjects: List[Dict]) -> FlatIndex:
    """Build the flat record index once per dataset."""
    seq_lens = np.array([len(s["log_gaps"]) for s in subjects], dtype=np.int64)
    prefix = np.concatenate([[0], np.cumsum(seq_lens)])
    n_total = int(prefix[-1])

    subj_of_flat = np.repeat(np.arange(len(subjects), dtype=np.int64), seq_lens)
    pos_of_flat = np.concatenate(
        [np.arange(l, dtype=np.int64) for l in seq_lens]
    ) if n_total > 0 else np.array([], dtype=np.int64)
    delta_of_flat = (
        np.concatenate([np.asarray(s["delta"], dtype=np.int64) for s in subjects])
        if n_total > 0
        else np.array([], dtype=np.int64)
    )
    k_star = np.maximum(seq_lens, 1).astype(np.float64)

    return FlatIndex(
        n_total=n_total,
        subj_of_flat=subj_of_flat,
        pos_of_flat=pos_of_flat,
        delta_of_flat=delta_of_flat,
        k_star=k_star,
        prefix=prefix,
    )


def sample_pairs(
    flat: FlatIndex,
    s: int,
    rng: np.random.Generator,
    exclude_same_subject: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample ``s`` comparison records for each uncensored anchor.

    Parameters
    ----------
    flat : FlatIndex
    s : int
        Comparison records drawn per anchor.
    rng : np.random.Generator
        The dedicated sampler stream.
    exclude_same_subject : bool
        If True, comparisons are drawn only from *other* subjects.  The
        manuscript's objective sums over all ordered pairs including
        within-subject ones, so the default is False; the option exists to
        test sensitivity to that choice.

    Returns
    -------
    anchor_flat, compare_flat : (m,) int arrays of flat indices
    inv_prob : (m,) float array of reciprocal inclusion probabilities
    """
    n_total = flat.n_total
    if n_total <= 1:
        empty_i = np.array([], dtype=np.int64)
        return empty_i, empty_i, np.array([], dtype=np.float64)

    anchors = np.flatnonzero(flat.delta_of_flat > 0)
    if anchors.size == 0:
        empty_i = np.array([], dtype=np.int64)
        return empty_i, empty_i, np.array([], dtype=np.float64)

    m = anchors.size * s
    anchor_flat = np.repeat(anchors, s)
    compare_flat = rng.integers(0, n_total, size=m)

    # Resample collisions rather than deleting from the pool per anchor.
    # A handful of passes clears them; the loop is bounded to avoid spinning
    # in the degenerate case where the pool is a single record.
    for _ in range(20):
        if exclude_same_subject:
            bad = flat.subj_of_flat[compare_flat] == flat.subj_of_flat[anchor_flat]
        else:
            bad = compare_flat == anchor_flat
        n_bad = int(bad.sum())
        if n_bad == 0:
            break
        compare_flat[bad] = rng.integers(0, n_total, size=n_bad)

    # Uniform draws from a pool of size (n_total - 1) after excluding self, so
    # each ordered pair has inclusion probability s / (n_total - 1).
    pool_size = float(n_total - 1)
    inv_prob = np.full(m, pool_size / float(s), dtype=np.float64)

    return anchor_flat, compare_flat, inv_prob


class Batched(NamedTuple):
    """Padded tensors for a set of subjects."""

    x_prev: torch.Tensor    # (n, seq, 1) previous log gap
    x_cov: torch.Tensor     # (n, seq, p) covariates
    observed: torch.Tensor  # (n, seq) observed log gap
    delta: torch.Tensor     # (n, seq)
    mask: torch.Tensor      # (n, seq) bool
    seq_lens: torch.Tensor  # (n,)
    k_star: torch.Tensor    # (n,)


def batchify(
    subjects: List[Dict], cov_dim: int, device: torch.device | str = "cpu"
) -> Batched:
    """Pad a list of subjects into dense tensors.

    ``x_prev[:, 0] = 0`` and ``x_prev[:, j] = log_gap[j-1]``: the predictor at
    event j sees the gap observed at j-1 and never its own outcome.  This is
    assumption (A3), enforced here rather than left to the caller.
    """
    n = len(subjects)
    seq_lens = np.array([len(s["log_gaps"]) for s in subjects], dtype=np.int64)
    max_seq = int(seq_lens.max()) if n > 0 and seq_lens.max() > 0 else 1

    x_prev = np.zeros((n, max_seq, 1), dtype=np.float32)
    x_cov = np.zeros((n, max_seq, cov_dim), dtype=np.float32)
    observed = np.zeros((n, max_seq), dtype=np.float32)
    delta = np.zeros((n, max_seq), dtype=np.float32)
    mask = np.zeros((n, max_seq), dtype=bool)

    for i, s in enumerate(subjects):
        l = int(seq_lens[i])
        if l == 0:
            continue
        lg = np.asarray(s["log_gaps"], dtype=np.float32)
        if l > 1:
            x_prev[i, 1:l, 0] = lg[:-1]
        x_cov[i, :l, :] = np.asarray(s["covariates"], dtype=np.float32)[None, :]
        observed[i, :l] = lg
        delta[i, :l] = np.asarray(s["delta"], dtype=np.float32)
        mask[i, :l] = True

    t = lambda a, dt: torch.tensor(a, dtype=dt, device=device)  # noqa: E731
    return Batched(
        x_prev=t(x_prev, torch.float32),
        x_cov=t(x_cov, torch.float32),
        observed=t(observed, torch.float32),
        delta=t(delta, torch.float32),
        mask=torch.tensor(mask, dtype=torch.bool, device=device),
        seq_lens=t(seq_lens, torch.long),
        k_star=t(np.maximum(seq_lens, 1).astype(np.float32), torch.float32),
    )


def gather_pair_residuals(
    resid: torch.Tensor,
    flat: FlatIndex,
    anchor_flat: np.ndarray,
    compare_flat: np.ndarray,
    device: torch.device | str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather residuals and K* values for sampled pairs.

    ``resid`` is the dense (n, seq) residual tensor; the flat indices are
    converted to (row, col) and gathered in one vectorised op.  The original
    code built these with a Python loop that called ``torch.tensor(...)`` per
    pair, which broke the computation graph's efficiency and dominated the
    training step.
    """
    a_rows = torch.as_tensor(flat.subj_of_flat[anchor_flat], device=device)
    a_cols = torch.as_tensor(flat.pos_of_flat[anchor_flat], device=device)
    c_rows = torch.as_tensor(flat.subj_of_flat[compare_flat], device=device)
    c_cols = torch.as_tensor(flat.pos_of_flat[compare_flat], device=device)

    e_anchor = resid[a_rows, a_cols]
    e_compare = resid[c_rows, c_cols]

    k = torch.as_tensor(flat.k_star, dtype=resid.dtype, device=device)
    return e_anchor, e_compare, k[a_rows], k[c_rows]
