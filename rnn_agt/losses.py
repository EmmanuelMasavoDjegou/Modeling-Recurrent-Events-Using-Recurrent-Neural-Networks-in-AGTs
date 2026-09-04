"""
Gehan-type weighted risk-set (WRS) loss.

The objective in the manuscript (Section 3.4) is

    L(w) = (1/n) sum_i sum_j sum_l sum_k
             [ delta_ij / (K_i* K_l*) ] * [ e_lk(w) - e_ij(w) ]^-

with ``[a]^- = max(0, -a)``, ``e_ij = Ytilde_ij - mu_ij(w)`` the residual on the
*observed* log-gap scale, and ``K_i* = max(K_i, 1)``.

**The sign of the hinge was inverted.**  As written in the manuscript, and as
implemented in every notebook, the penalty evaluates to
``max(0, e_ij - e_lk)`` -- anchor minus comparison.  The Gehan objective whose
subgradient reproduces the Gehan estimating function

    U(b) = sum_i sum_j delta_i (Z_i - Z_j) 1{e_i <= e_j}

is ``sum delta_i (e_i - e_j)^-`` with ``a^- = |a| 1(a < 0)``, which expands to
``max(0, e_j - e_i)`` -- comparison minus anchor.  Differentiating confirms it:
with ``e = Y - b'Z``, the active term ``e_j - e_i`` has gradient
``Z_i - Z_j``, giving exactly ``U(b)``.  The two orientations are not
equivalent; they push the predictor in opposite directions.

Empirically the difference is large and grows with censoring.  On the
interaction setting with AR(1) dependence, test C-index under the corrected
sign versus the original, against an oracle (true conditional mean) value:

======================  ==========  ==========  ========
Censoring               Original    Corrected   Oracle
======================  ==========  ==========  ========
25%                     0.882       0.940       0.940
65%                     0.621       0.916       0.963
======================  ==========  ==========  ========

The corrected objective essentially attains oracle discrimination; the
original loses a third of the gap at heavy censoring.  This defect and the
latent-gap leak partially cancelled -- training against uncensored outcomes
compensated for a loss pushing the wrong way -- which is the likeliest reason
neither was noticed.

**The 1/(K_i* K_l*) factor was also missing from the original implementation.**  The
notebook's loss computed an unweighted sum over sampled pairs, rescaled only by
``(N_total - 1) / s`` and divided by the number of uncensored events.  That is a
plain Gehan rank loss, not a weighted risk-set loss.

This is not a cosmetic difference.  The subject-level normalization is the
entire mechanism by which the WRS construction handles induced dependent
censoring: it equalises each subject's contribution regardless of how many
events that subject accrued, which is what removes the over-representation of
subjects with many short gaps.  Without it, a subject contributing six gaps
carries six times the weight of a subject contributing one, and since accruing
many gaps is itself informative about the error process, the estimating
function is biased.  It is also precisely the property Section 3.4.4 relies on
when arguing that the WRS construction survives the move to a nonlinear
predictor, so the implementation and the manuscript's justification have to
agree.

Both forms are provided.  ``weighted=True`` (default) matches the manuscript.
``weighted=False`` reproduces the original behaviour, so the two can be
compared directly and the practical size of the discrepancy measured rather
than assumed.
"""

from __future__ import annotations

from typing import Optional

import torch


def hinge_negative(a: torch.Tensor) -> torch.Tensor:
    """``[a]^- = max(0, -a)``, computed without a branch."""
    return torch.clamp(-a, min=0.0)


def gehan_wrs_loss_full(
    pred: torch.Tensor,
    observed: torch.Tensor,
    delta: torch.Tensor,
    mask: torch.Tensor,
    k_star: torch.Tensor,
    weighted: bool = True,
    n_subjects: Optional[int] = None,
) -> torch.Tensor:
    """Exact (all-pairs) Gehan-WRS loss.

    O(N^2) in the number of retained records, so this is for small samples,
    unit tests, and validating the subsampled estimator.  Training uses
    :func:`gehan_wrs_loss_pairs`.

    Parameters
    ----------
    pred, observed : (batch, seq)
        Predicted and observed log gap times.
    delta : (batch, seq)
        1 if the record is a fully observed event, else 0.
    mask : (batch, seq)
        True at real records, False at padding.
    k_star : (batch,)
        ``K_i* = max(K_i, 1)`` per subject.
    weighted : bool
        Apply the ``1/(K_i* K_l*)`` WRS normalization.
    """
    resid = observed - pred
    flat_resid = resid[mask]                                   # (N,)
    flat_delta = delta[mask]                                   # (N,)
    subj_k = k_star[:, None].expand_as(resid)[mask].to(resid.dtype)

    anchor = flat_delta > 0
    if anchor.sum() == 0:
        return pred.sum() * 0.0

    e_anchor = flat_resid[anchor]                              # (A,)
    k_anchor = subj_k[anchor]                                  # (A,)

    # Penalty = max(0, e_compare - e_anchor). See the module docstring on the
    # sign convention; this is the Jin et al. (2003) orientation, whose
    # gradient reproduces the Gehan estimating function.
    diffs = flat_resid[None, :] - e_anchor[:, None]            # (A, N)
    pen = torch.clamp(diffs, min=0.0)

    if weighted:
        w = 1.0 / (k_anchor[:, None] * subj_k[None, :])
        total = (pen * w).sum()
    else:
        total = pen.sum()

    n = n_subjects if n_subjects is not None else pred.size(0)
    return total / float(max(n, 1))


def gehan_wrs_loss_pairs(
    e_anchor: torch.Tensor,
    e_compare: torch.Tensor,
    k_anchor: torch.Tensor,
    k_compare: torch.Tensor,
    inv_prob: torch.Tensor,
    n_subjects: int,
    weighted: bool = True,
) -> torch.Tensor:
    """Importance-weighted subsampled Gehan-WRS loss.

    Estimates the full objective from a sample of pairs.  With ``inv_prob``
    the reciprocal of each pair's inclusion probability, the estimator is
    conditionally unbiased for the full loss (Theorem A.2), which is what
    licenses the stochastic subgradient argument of Remark A.1.

    Parameters
    ----------
    e_anchor, e_compare : (m,)
        Residuals of the anchor (uncensored) and comparison records.
    k_anchor, k_compare : (m,)
        ``K_i*`` for the anchor's and comparison's subjects.
    inv_prob : (m,)
        Reciprocal inclusion probability of each sampled pair.
    n_subjects : int
        ``n``, the outer normalizing constant.
    weighted : bool
        Apply the WRS normalization.  ``False`` reproduces the original
        unweighted behaviour.
    """
    if e_anchor.numel() == 0:
        return e_anchor.sum() * 0.0

    pen = torch.clamp(e_compare - e_anchor, min=0.0)
    if weighted:
        pen = pen / (k_anchor * k_compare)
    return (pen * inv_prob).sum() / float(max(n_subjects, 1))


class GehanWRSLoss(torch.nn.Module):
    """Module wrapper, for callers that prefer the nn.Module interface."""

    def __init__(self, weighted: bool = True) -> None:
        super().__init__()
        self.weighted = weighted

    def forward(self, *args, **kwargs) -> torch.Tensor:
        kwargs.setdefault("weighted", self.weighted)
        return gehan_wrs_loss_full(*args, **kwargs)
