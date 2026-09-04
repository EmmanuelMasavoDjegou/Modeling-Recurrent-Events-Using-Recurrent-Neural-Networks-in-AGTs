"""
Bridge between the Python AFT models and the R Cox models.

Table 9 compares six methods. Three are fitted here in PyTorch; three
(WLW, PWP-TT, PWP-GT) are fitted in R with ``survival``. For the paired
differences in that table to mean anything, all six must see the same splits
and be scored with the same estimator. This module is what enforces that.

Protocol
--------
1. :func:`write_splits` exports the split assignments Python generated.
2. The R script fits the Cox models on each training partition and exports the
   held-out **linear predictors** -- not concordance indices.
3. :func:`load_cox_predictions` reads them back and
   :func:`score_cox_predictions` computes the IPCW C-index with the same
   :func:`rnn_agt.metrics.ipcw_cindex` used for the neural models.

Why linear predictors rather than a C-index computed in R
---------------------------------------------------------
The previous R scripts reported ``summary(fit)$concordance``: Harrell's C,
computed in sample, unweighted, on the full dataset. RNN-AGT's number is
out-of-sample and IPCW-weighted. Those are different estimands, and comparing
them favours whichever model was scored in sample. Exporting the linear
predictor and scoring it here removes the possibility of that mismatch
recurring, because there is exactly one concordance implementation in the
project.

Sign convention
---------------
A Cox linear predictor is on the log-hazard scale: larger means higher hazard,
so shorter gap times. The AFT models predict log gap time, where larger means
longer. :func:`score_cox_predictions` therefore negates the linear predictor
before scoring. Getting this backwards produces a C-index of roughly ``1 - C``,
which on these datasets lands near 0.35 and looks like a broken model rather
than a sign error, so it is worth checking against
:func:`check_orientation` if a Cox row comes out below 0.5.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .evaluation import SplitOutcome, stratified_split
from .metrics import ipcw_cindex
from .seeds import make_seeds

COX_MODELS = ("WLW", "PWP_TT", "PWP_GT")
COX_LABELS = {"WLW": "WLW", "PWP_TT": "PWP-TT", "PWP_GT": "PWP-GT"}


def write_splits(
    subjects: List[Dict],
    subject_ids: Sequence[int],
    path: str,
    master_seed: int,
    n_splits: int = 200,
    test_frac: float = 0.30,
) -> pd.DataFrame:
    """Generate splits and write them for the R script.

    The generated assignments are returned as well, so the caller can drive the
    Python models over the *same* partitions rather than regenerating them and
    hoping the streams line up.

    Parameters
    ----------
    subjects : list of dict
        Internal representation, used for stratification.
    subject_ids : sequence
        The original dataset ids, in the same order as ``subjects``. These are
        what the R script joins on, so they must be the ids present in the CSV,
        not positional indices.
    """
    if len(subject_ids) != len(subjects):
        raise ValueError("subject_ids must align one-to-one with subjects")

    rng = make_seeds(master_seed).split()
    ids = np.asarray(subject_ids)
    rows = []
    for b in range(n_splits):
        tr_idx, te_idx = stratified_split(subjects, test_frac, rng)
        for i in tr_idx:
            rows.append((b, int(ids[i]), "train"))
        for i in te_idx:
            rows.append((b, int(ids[i]), "test"))

    df = pd.DataFrame(rows, columns=["split_id", "id", "partition"])
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)
    return df


def splits_to_index(
    df: pd.DataFrame, subject_ids: Sequence[int]
) -> List[tuple]:
    """Convert exported assignments back to positional index arrays."""
    pos_of_id = {int(v): i for i, v in enumerate(subject_ids)}
    out = []
    for b, grp in df.groupby("split_id", sort=True):
        tr = np.array(
            [pos_of_id[i] for i in grp.loc[grp.partition == "train", "id"]
             if i in pos_of_id], dtype=np.int64
        )
        te = np.array(
            [pos_of_id[i] for i in grp.loc[grp.partition == "test", "id"]
             if i in pos_of_id], dtype=np.int64
        )
        out.append((int(b), tr, te))
    return out


def load_cox_predictions(path: str) -> pd.DataFrame:
    """Read the linear predictors exported by the R script."""
    df = pd.read_csv(path)
    required = {"split_id", "model", "id", "order", "lp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    return df


def score_cox_predictions(
    cox_df: pd.DataFrame,
    subjects: List[Dict],
    subject_ids: Sequence[int],
    splits_df: pd.DataFrame,
) -> Dict[int, Dict[str, float]]:
    """Score Cox linear predictors with the project's IPCW C-index.

    Returns ``{split_id: {model: cindex}}``.

    Records whose linear predictor is missing -- which happens when a test
    subject's event order was absent from the training partition -- are dropped
    from that split's evaluation and counted in the returned report.
    """
    pos_of_id = {int(v): i for i, v in enumerate(subject_ids)}
    scores: Dict[int, Dict[str, float]] = {}
    n_dropped = 0
    n_total = 0

    for (split_id, model), grp in cox_df.groupby(["split_id", "model"]):
        test_ids = splits_df.loc[
            (splits_df.split_id == split_id) & (splits_df.partition == "test"),
            "id",
        ].unique()

        sub_list, pred_rows = [], []
        for sid in test_ids:
            if sid not in pos_of_id:
                continue
            subj = subjects[pos_of_id[sid]]
            recs = grp.loc[grp.id == sid].sort_values("order")
            lp = recs["lp"].to_numpy(dtype=float)
            k = len(subj["log_gaps"])
            n_total += k
            if len(lp) < k or np.isnan(lp[:k]).any():
                n_dropped += k
                continue
            sub_list.append(subj)
            # Negate: log-hazard scale -> log gap-time scale.
            pred_rows.append(-lp[:k])

        if not sub_list:
            scores.setdefault(int(split_id), {})[str(model)] = float("nan")
            continue

        max_seq = max(len(p) for p in pred_rows)
        pred = np.zeros((len(pred_rows), max_seq), dtype=float)
        for i, p in enumerate(pred_rows):
            pred[i, : len(p)] = p

        scores.setdefault(int(split_id), {})[str(model)] = ipcw_cindex(
            sub_list, pred
        )

    if n_total:
        frac = 100.0 * n_dropped / n_total
        if frac > 1.0:
            print(
                f"  note: {n_dropped}/{n_total} records ({frac:.1f}%) dropped "
                "for missing Cox linear predictors"
            )
    return scores


def check_orientation(
    cox_scores: Dict[int, Dict[str, float]]
) -> Optional[str]:
    """Warn if Cox concordance sits below 0.5 across the board.

    A systematically sub-chance Cox row almost always means the linear
    predictor sign was flipped twice (once here, once in R) rather than that
    the model is genuinely anti-predictive.
    """
    vals = [
        v for per_split in cox_scores.values()
        for v in per_split.values() if not np.isnan(v)
    ]
    if not vals:
        return "no Cox scores available"
    mean = float(np.mean(vals))
    if mean < 0.45:
        return (
            f"mean Cox C-index is {mean:.3f}, below chance. The linear "
            "predictor is probably being negated twice: it should be negated "
            "in Python only, never in the R export."
        )
    return None


def merge_into_outcomes(
    outcomes: List[SplitOutcome], cox_scores: Dict[int, Dict[str, float]]
) -> List[SplitOutcome]:
    """Fold Cox scores into the per-split outcomes from the Python models.

    After this, ``paired_difference(outcomes, "rnn_agt", "PWP_GT")`` is a
    genuine paired comparison: both numbers come from the same split, the same
    test subjects and the same concordance estimator.
    """
    for o in outcomes:
        for model, c in cox_scores.get(o.split_id, {}).items():
            o.metrics[model] = {"test_cindex": c, "test_amse": float("nan")}
    return outcomes
