"""
The three model classes forming the ablation ladder (Section 5.5).

======================  ==========  =========  ================================
Model                   Nonlinear   History    Predictor
======================  ==========  =========  ================================
``AFTWRS``              no          no         theta' z_i
``NNAFT``               yes         no         MLP(z_i)
``RNNAGT``              yes         yes        GRU(z_i, prior log-gaps)
======================  ==========  =========  ================================

All three are trained with the *identical* Gehan-WRS objective and the
identical pair sampler, so a difference between adjacent rungs is attributable
to the capability that rung adds and nothing else.  That is the whole point of
the comparison, and it is why ``AFTWRS`` is implemented here as a torch module
rather than fitted by a separate rank-regression routine: an external
implementation would differ in optimizer, convergence criterion and tie
handling, and those differences would contaminate the contrast.

All predictors are *predictable* with respect to the history through event
j-1, which is assumption (A3) of Section 3.4.4.  The GRU input at position j
carries the log-gap observed at position j-1, never at j.  Violating this would
leak the outcome into its own predictor.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class RNNAGT(nn.Module):
    """GRU predictor: nonlinear covariate effects and event-history dependence.

    Parameters
    ----------
    cov_dim : int
        Number of baseline covariates.
    hidden_dim : int
        GRU hidden dimension ``d``.
    gru_layers : int
        Number of stacked recurrent layers ``L``.
    dropout : float
        Inter-layer dropout, active only when ``gru_layers > 1``.  Note that if
        this is non-zero, training and evaluation modes differ, which changes
        how training-set metrics should be interpreted (Section 5.2).  It is 0
        by default precisely so that the manuscript's statement that no
        stochastic regularizer is active remains true.
    """

    def __init__(
        self,
        cov_dim: int,
        hidden_dim: int = 64,
        gru_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.cov_dim = cov_dim
        self.hidden_dim = hidden_dim
        self.gru_layers = gru_layers
        self.input_dim = 1 + cov_dim  # previous log-gap, then covariates

        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.init_fc = nn.Linear(cov_dim, gru_layers * hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def initialize_hidden(self, x_static: torch.Tensor) -> torch.Tensor:
        h0 = torch.tanh(self.init_fc(x_static))
        return h0.view(x_static.size(0), self.gru_layers, self.hidden_dim).permute(
            1, 0, 2
        ).contiguous()

    def forward(self, x_prev: torch.Tensor, x_cov: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x_prev : (batch, seq, 1)
            Previous log gap time; zero at position 0.
        x_cov : (batch, seq, cov_dim)
            Baseline covariates, repeated along the sequence.

        Returns
        -------
        (batch, seq) predicted log gap times.
        """
        x_in = torch.cat([x_prev, x_cov], dim=-1)
        h0 = self.initialize_hidden(x_cov[:, 0, :])
        out, _ = self.gru(x_in, h0)
        return self.output(out).squeeze(-1)

    @property
    def uses_history(self) -> bool:
        return True


class NNAFT(nn.Module):
    """Feed-forward predictor: nonlinear covariate effects, no history.

    This is the middle rung of the ablation.  It receives the baseline
    covariates only: no previous gap times, no recurrent state.  Its prediction
    is therefore constant across the event index j for a given subject, which
    is exactly the capability being withheld.

    ``hidden_dims`` defaults to ``None``, in which case the width is chosen by
    :func:`matched_mlp_width` to bring the parameter count close to a reference
    GRU.  Matching capacity matters: if NN-AFT were much smaller, a gap in its
    favour to RNN-AGT could be read as a capacity effect rather than a history
    effect, and the ablation would not isolate what it claims to.
    """

    def __init__(
        self,
        cov_dim: int,
        hidden_dims: Optional[tuple] = None,
        reference_hidden: int = 64,
        reference_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.cov_dim = cov_dim
        if hidden_dims is None:
            width = matched_mlp_width(cov_dim, reference_hidden, reference_layers)
            hidden_dims = (width, width)
        self.hidden_dims = tuple(hidden_dims)

        layers = []
        prev = cov_dim
        for h in self.hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.Tanh())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_prev: torch.Tensor, x_cov: torch.Tensor) -> torch.Tensor:
        """``x_prev`` is accepted and deliberately ignored.

        The signature matches :class:`RNNAGT` so the trainer can drive either
        without branching.  Ignoring ``x_prev`` here is the ablation.
        """
        out = self.net(x_cov).squeeze(-1)  # (batch, seq)
        return out

    @property
    def uses_history(self) -> bool:
        return False


class AFTWRS(nn.Module):
    """Linear accelerated gap-time predictor: no nonlinearity, no history.

    The bottom rung, and the model of Lyu et al. (2018) that RNN-AGT extends.
    Implemented as a single linear layer with no bias, since the Gehan
    objective is invariant to a location shift of the predictor: adding a
    constant to every prediction leaves all pairwise residual *differences*
    unchanged, so an intercept is not identified.  Including one would make the
    optimizer wander along a flat direction.
    """

    def __init__(self, cov_dim: int) -> None:
        super().__init__()
        self.cov_dim = cov_dim
        self.linear = nn.Linear(cov_dim, 1, bias=False)

    def forward(self, x_prev: torch.Tensor, x_cov: torch.Tensor) -> torch.Tensor:
        return self.linear(x_cov).squeeze(-1)

    @property
    def uses_history(self) -> bool:
        return False

    def coefficients(self) -> torch.Tensor:
        return self.linear.weight.detach().flatten()


# --------------------------------------------------------------------------
# Capacity accounting
# --------------------------------------------------------------------------


def count_parameters(model: nn.Module) -> int:
    """Trainable parameter count, reported in Table 10."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def gru_parameter_count(cov_dim: int, hidden_dim: int, gru_layers: int) -> int:
    """Analytic parameter count for :class:`RNNAGT`, without building it."""
    input_dim = 1 + cov_dim
    total = 0
    for layer in range(gru_layers):
        in_dim = input_dim if layer == 0 else hidden_dim
        # GRU: 3 gates, weight_ih (3d x in), weight_hh (3d x d), 2 bias (3d)
        total += 3 * hidden_dim * in_dim + 3 * hidden_dim * hidden_dim + 6 * hidden_dim
    total += cov_dim * gru_layers * hidden_dim + gru_layers * hidden_dim  # init_fc
    total += hidden_dim + 1  # output layer
    return total


def matched_mlp_width(cov_dim: int, hidden_dim: int, gru_layers: int) -> int:
    """Width of a two-hidden-layer MLP whose size approximates the GRU's.

    Solves ``w^2 + w(cov_dim + 2) + 1 = target`` for a two-layer tanh MLP and
    rounds to the nearest integer, floored at 4.
    """
    target = gru_parameter_count(cov_dim, hidden_dim, gru_layers)
    b = cov_dim + 2.0
    disc = b * b + 4.0 * (target - 1.0)
    w = (-b + max(disc, 0.0) ** 0.5) / 2.0
    return max(int(round(w)), 4)


MODEL_REGISTRY = {
    "aft_wrs": AFTWRS,
    "nn_aft": NNAFT,
    "rnn_agt": RNNAGT,
}

MODEL_LABELS = {
    "aft_wrs": "AFT-WRS",
    "nn_aft": "NN-AFT",
    "rnn_agt": "RNN-AGT",
}


def build_model(kind: str, cov_dim: int, **kwargs) -> nn.Module:
    """Construct a model by name, passing only the kwargs it accepts."""
    if kind == "aft_wrs":
        return AFTWRS(cov_dim)
    if kind == "nn_aft":
        return NNAFT(
            cov_dim,
            hidden_dims=kwargs.get("hidden_dims"),
            reference_hidden=kwargs.get("hidden_dim", 64),
            reference_layers=kwargs.get("gru_layers", 2),
            dropout=kwargs.get("dropout", 0.0),
        )
    if kind == "rnn_agt":
        return RNNAGT(
            cov_dim,
            hidden_dim=kwargs.get("hidden_dim", 64),
            gru_layers=kwargs.get("gru_layers", 2),
            dropout=kwargs.get("dropout", 0.0),
        )
    raise ValueError(f"unknown model kind: {kind!r}")
