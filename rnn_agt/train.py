"""
Training loop, shared by all three model classes.

One trainer drives AFT-WRS, NN-AFT and RNN-AGT.  That is deliberate: if each
rung of the ablation had its own optimizer, stopping rule or tie handling,
differences between rungs would confound the capability under test with
incidental fitting differences, and the ablation would not isolate what
Section 5.5 claims it isolates.

Differences from the original loop, beyond the loss and sampler fixes
documented in ``losses.py`` and ``sampling.py``:

* The per-pair Python loop that built ``left_preds``/``right_preds`` lists and
  called ``torch.tensor(...)`` once per pair is replaced by a single gather.
* Pairs are resampled every epoch from a fresh forward pass, as before, but the
  sampler draws from its own RNG stream so that re-running a fit on identical
  data with identical initialization is possible.
* Metrics are computed in ``eval()`` mode after fitting, never accumulated
  during optimization (Section 5.2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.optim as optim

from .losses import gehan_wrs_loss_pairs
from .metrics import evaluate
from .models import build_model, count_parameters
from .sampling import batchify, build_flat_index, gather_pair_residuals, sample_pairs


@dataclass
class TrainConfig:
    """Hyperparameters.

    Defaults follow Section 5.2 of the manuscript.  ``weighted_loss=True``
    applies the WRS normalization; set it False only to reproduce the original
    unweighted behaviour for comparison.
    """

    model: str = "rnn_agt"
    hidden_dim: int = 64
    gru_layers: int = 2
    dropout: float = 0.0
    epochs: int = 10
    pair_sample_s: int = 30
    pair_batch_b: int = 64
    lr: float = 3e-4
    optimizer: str = "rmsprop"
    weighted_loss: bool = True
    exclude_same_subject: bool = False
    device: str = "cpu"
    track_history: bool = False
    extra: Dict = field(default_factory=dict)


@dataclass
class TrainResult:
    model: torch.nn.Module
    metrics: Dict[str, float]
    n_params: int
    epoch_losses: List[float]
    history: Dict[str, List[float]]
    config: TrainConfig


def _make_optimizer(cfg: TrainConfig, params) -> optim.Optimizer:
    if cfg.optimizer == "rmsprop":
        return optim.RMSprop(params, lr=cfg.lr)
    if cfg.optimizer == "adam":
        return optim.Adam(params, lr=cfg.lr)
    if cfg.optimizer == "sgd":
        return optim.SGD(params, lr=cfg.lr, momentum=0.9)
    raise ValueError(f"unknown optimizer: {cfg.optimizer!r}")


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    subjects: List[Dict],
    cov_dim: int,
    device: str = "cpu",
) -> np.ndarray:
    """Dense (n, seq) predictions, in evaluation mode."""
    if not subjects:
        return np.zeros((0, 0), dtype=np.float64)
    was_training = model.training
    model.eval()
    b = batchify(subjects, cov_dim, device=device)
    out = model(b.x_prev, b.x_cov)
    pred = out.detach().cpu().numpy().astype(np.float64)
    pred = np.where(b.mask.cpu().numpy(), pred, 0.0)
    if was_training:
        model.train()
    return pred


def train_model(
    train_subjects: List[Dict],
    test_subjects: Optional[List[Dict]],
    cov_dim: int,
    cfg: TrainConfig,
    seeds,
    verbose: bool = False,
) -> TrainResult:
    """Fit one model and return it with its final metrics.

    Parameters
    ----------
    seeds : rnn_agt.seeds.SeedBundle
        Supplies the sampler stream and the torch initialization seed.
    """
    device = torch.device(cfg.device)

    seeds.seed_torch()
    model = build_model(
        cfg.model,
        cov_dim,
        hidden_dim=cfg.hidden_dim,
        gru_layers=cfg.gru_layers,
        dropout=cfg.dropout,
        **cfg.extra,
    ).to(device)

    optimizer = _make_optimizer(cfg, model.parameters())
    sampler_rng = seeds.sampler()

    flat = build_flat_index(train_subjects)
    n_subjects = len(train_subjects)
    batched = batchify(train_subjects, cov_dim, device=device)

    epoch_losses: List[float] = []
    history: Dict[str, List[float]] = {
        "train_cindex": [], "test_cindex": [], "train_amse": [], "test_amse": []
    }

    for epoch in range(cfg.epochs):
        model.train()

        anchor_flat, compare_flat, inv_prob = sample_pairs(
            flat, cfg.pair_sample_s, sampler_rng,
            exclude_same_subject=cfg.exclude_same_subject,
        )
        m = anchor_flat.size
        if m == 0:
            epoch_losses.append(0.0)
            continue

        perm = sampler_rng.permutation(m)
        anchor_flat, compare_flat = anchor_flat[perm], compare_flat[perm]
        inv_prob = inv_prob[perm]
        inv_prob_t = torch.as_tensor(inv_prob, dtype=torch.float32, device=device)

        batch_losses = []
        for start in range(0, m, cfg.pair_batch_b):
            end = min(start + cfg.pair_batch_b, m)

            optimizer.zero_grad(set_to_none=True)
            # One forward pass over all training subjects per pair-batch.  The
            # sampled pairs reference arbitrary subjects, so restricting the
            # pass to the batch's subjects (as the original did) saves little
            # once the gather is vectorised and complicates index mapping.
            pred = model(batched.x_prev, batched.x_cov)
            resid = batched.observed - pred

            e_a, e_c, k_a, k_c = gather_pair_residuals(
                resid, flat, anchor_flat[start:end], compare_flat[start:end],
                device=device,
            )
            loss = gehan_wrs_loss_pairs(
                e_a, e_c, k_a, k_c, inv_prob_t[start:end],
                n_subjects=n_subjects, weighted=cfg.weighted_loss,
            )
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))

        epoch_losses.append(float(np.mean(batch_losses)) if batch_losses else 0.0)

        if cfg.track_history:
            tr = evaluate(train_subjects, predict(model, train_subjects, cov_dim, cfg.device))
            history["train_cindex"].append(tr["cindex"])
            history["train_amse"].append(tr["amse"])
            if test_subjects:
                te = evaluate(test_subjects, predict(model, test_subjects, cov_dim, cfg.device))
                history["test_cindex"].append(te["cindex"])
                history["test_amse"].append(te["amse"])

        if verbose:
            print(f"  epoch {epoch+1}/{cfg.epochs}  loss={epoch_losses[-1]:.4f}")

    # Final evaluation: one forward pass in eval mode, after fitting.
    model.eval()
    train_metrics = evaluate(train_subjects, predict(model, train_subjects, cov_dim, cfg.device))
    metrics = {
        "train_cindex": train_metrics["cindex"],
        "train_amse": train_metrics["amse"],
    }
    if test_subjects:
        test_metrics = evaluate(test_subjects, predict(model, test_subjects, cov_dim, cfg.device))
        metrics["test_cindex"] = test_metrics["cindex"]
        metrics["test_amse"] = test_metrics["amse"]

    return TrainResult(
        model=model,
        metrics=metrics,
        n_params=count_parameters(model),
        epoch_losses=epoch_losses,
        history=history,
        config=cfg,
    )
