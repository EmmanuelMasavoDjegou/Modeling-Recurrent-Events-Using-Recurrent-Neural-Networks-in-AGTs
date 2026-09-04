"""
Repeated splits, cross-validation and paired comparisons (Section 5.6).

The original analyses used a single fixed 70/30 split.  With 128 and 403
subjects, the test partitions hold roughly 38 and 121 subjects, and concordance
is computed over comparable pairs among them, so a different draw could
plausibly reorder the methods.  This module replaces that with B repeated
stratified splits and subject-level k-fold CV.

Two design points matter for the comparisons to mean anything.

**All methods share the same splits.**  ``run_repeated_splits`` fits every
model on split b before moving to split b+1, so per-split differences are
*paired*.  A paired interval on the difference is far tighter than the
difference of two marginal intervals, and it is the quantity Section 5.6 asks
about.

**Splits are stratified on observed event count.**  Subjects with several
events carry most of the information about history dependence, and there are
few of them.  An unstratified draw can put nearly all of them on one side,
which adds variance that has nothing to do with the methods being compared.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .seeds import SeedBundle, make_seeds
from .train import TrainConfig, TrainResult, train_model


# --------------------------------------------------------------------------
# Partitioning
# --------------------------------------------------------------------------


def _event_strata(subjects: List[Dict]) -> np.ndarray:
    """Stratum label per subject: number of *observed* events, capped at 3+."""
    counts = np.array(
        [int(np.sum(s["delta"])) for s in subjects], dtype=np.int64
    )
    return np.minimum(counts, 3)


def stratified_split(
    subjects: List[Dict], test_frac: float, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified train/test index split at the subject level."""
    strata = _event_strata(subjects)
    train_idx, test_idx = [], []
    for label in np.unique(strata):
        members = np.flatnonzero(strata == label)
        rng.shuffle(members)
        n_test = int(round(test_frac * members.size))
        # Guarantee both sides are non-empty when the stratum allows it.
        n_test = min(max(n_test, 1 if members.size > 1 else 0), members.size - 1) \
            if members.size > 1 else 0
        test_idx.append(members[:n_test])
        train_idx.append(members[n_test:])
    return (
        np.concatenate(train_idx) if train_idx else np.array([], dtype=np.int64),
        np.concatenate(test_idx) if test_idx else np.array([], dtype=np.int64),
    )


def stratified_folds(
    subjects: List[Dict], k: int, rng: np.random.Generator
) -> List[np.ndarray]:
    """Subject-level stratified k-fold assignment.

    Folds are formed over subjects, never over records, so no subject
    contributes events to both a training and a validation fold.  Splitting at
    the record level would leak a subject's own history across the boundary and
    inflate every method that uses history.
    """
    strata = _event_strata(subjects)
    fold_of = np.empty(len(subjects), dtype=np.int64)
    for label in np.unique(strata):
        members = np.flatnonzero(strata == label)
        rng.shuffle(members)
        fold_of[members] = np.arange(members.size) % k
    return [np.flatnonzero(fold_of == f) for f in range(k)]


def _subset(subjects: List[Dict], idx: Sequence[int]) -> List[Dict]:
    return [subjects[int(i)] for i in idx]


# --------------------------------------------------------------------------
# Repeated-split driver
# --------------------------------------------------------------------------


@dataclass
class SplitOutcome:
    """Per-split metrics for every model, keyed by model name."""

    split_id: int
    metrics: Dict[str, Dict[str, float]]


def run_repeated_splits(
    subjects: List[Dict],
    cov_dim: int,
    configs: Dict[str, TrainConfig],
    master_seed: int,
    n_splits: int = 200,
    test_frac: float = 0.30,
    progress: Optional[Callable[[int, int], None]] = None,
) -> List[SplitOutcome]:
    """Fit every model on each of ``n_splits`` shared stratified splits.

    Parameters
    ----------
    configs : dict
        Maps a display name to a :class:`TrainConfig`.  Every entry is fitted
        on every split.
    master_seed : int
        One number reproduces the whole run.  Each split derives its own
        SeedBundle from ``master_seed + split_id``, so split b is reproducible
        in isolation without re-running the preceding ones.
    """
    outcomes: List[SplitOutcome] = []
    split_rng = make_seeds(master_seed).split()

    for b in range(n_splits):
        tr_idx, te_idx = stratified_split(subjects, test_frac, split_rng)
        train_s, test_s = _subset(subjects, tr_idx), _subset(subjects, te_idx)

        per_model: Dict[str, Dict[str, float]] = {}
        for name, cfg in configs.items():
            seeds_b = make_seeds(master_seed + 1000 * (b + 1))
            res = train_model(train_s, test_s, cov_dim, cfg, seeds_b)
            per_model[name] = dict(res.metrics, n_params=res.n_params)
        outcomes.append(SplitOutcome(split_id=b, metrics=per_model))

        if progress is not None:
            progress(b + 1, n_splits)
    return outcomes


def run_cross_validation(
    subjects: List[Dict],
    cov_dim: int,
    configs: Dict[str, TrainConfig],
    master_seed: int,
    k: int = 5,
) -> Dict[str, Dict[str, float]]:
    """Subject-level k-fold CV; returns mean metrics per model."""
    fold_rng = make_seeds(master_seed).split()
    folds = stratified_folds(subjects, k, fold_rng)

    acc: Dict[str, List[Dict[str, float]]] = {name: [] for name in configs}
    for f, val_idx in enumerate(folds):
        tr_idx = np.concatenate([folds[g] for g in range(k) if g != f])
        train_s, val_s = _subset(subjects, tr_idx), _subset(subjects, val_idx)
        for name, cfg in configs.items():
            seeds_f = make_seeds(master_seed + 7919 * (f + 1))
            res = train_model(train_s, val_s, cov_dim, cfg, seeds_f)
            acc[name].append(res.metrics)

    out: Dict[str, Dict[str, float]] = {}
    for name, rows in acc.items():
        out[name] = {
            key: float(np.nanmean([r.get(key, np.nan) for r in rows]))
            for key in ("test_cindex", "test_amse", "train_cindex", "train_amse")
        }
    return out


# --------------------------------------------------------------------------
# Summaries and paired comparisons
# --------------------------------------------------------------------------


def summarise(
    outcomes: List[SplitOutcome], metric: str = "test_cindex"
) -> Dict[str, Dict[str, float]]:
    """Mean, sd, and a percentile interval per model across splits."""
    names = outcomes[0].metrics.keys() if outcomes else []
    summary: Dict[str, Dict[str, float]] = {}
    for name in names:
        vals = np.array(
            [o.metrics[name].get(metric, np.nan) for o in outcomes], dtype=float
        )
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            summary[name] = {"mean": np.nan, "sd": np.nan, "lo": np.nan, "hi": np.nan}
            continue
        summary[name] = {
            "mean": float(vals.mean()),
            "sd": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
            "lo": float(np.percentile(vals, 2.5)),
            "hi": float(np.percentile(vals, 97.5)),
            "n": int(vals.size),
        }
    return summary


def paired_difference(
    outcomes: List[SplitOutcome],
    model_a: str,
    model_b: str,
    metric: str = "test_cindex",
) -> Dict[str, float]:
    """Paired difference ``a - b`` across shared splits.

    Returns the mean difference, its standard error, a 95% normal interval, a
    percentile interval, and the proportion of splits on which ``a`` wins.
    That last number is worth reporting alongside the mean: a method can carry
    a positive mean difference while losing on 40% of splits, and the two
    convey different things about reliability.
    """
    diffs = []
    for o in outcomes:
        va = o.metrics.get(model_a, {}).get(metric, np.nan)
        vb = o.metrics.get(model_b, {}).get(metric, np.nan)
        if not (np.isnan(va) or np.isnan(vb)):
            diffs.append(va - vb)
    d = np.array(diffs, dtype=float)
    if d.size == 0:
        return {"mean": np.nan, "se": np.nan, "lo": np.nan, "hi": np.nan,
                "win_rate": np.nan, "n": 0}
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(d.size)) if d.size > 1 else 0.0
    return {
        "mean": mean,
        "se": se,
        "lo": mean - 1.96 * se,
        "hi": mean + 1.96 * se,
        "pct_lo": float(np.percentile(d, 2.5)),
        "pct_hi": float(np.percentile(d, 97.5)),
        "win_rate": float((d > 0).mean()),
        "n": int(d.size),
    }
