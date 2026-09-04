"""
Reproducible seeding.

The manuscript (Section 5.3) states that data generation, network
initialization, pair subsampling and the train/test partitioner each draw from
*separately seeded* generators, so that any one component can be varied while
the others are held fixed.  This module is what makes that claim true.

Do not call ``np.random.*`` or rely on the global NumPy/torch RNG anywhere in
this package.  The original notebooks did (``np.random.seed(42)`` followed by
bare ``np.random.normal(...)`` inside ``generate_gap_times``), which meant the
data-generating stream and the model-initialization stream were entangled: you
could not re-run a fit on identical data without also re-drawing the network
weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass(frozen=True)
class SeedBundle:
    """Four independent streams derived from one master seed.

    Parameters
    ----------
    master : int
        The only number a user needs to record to reproduce a run.

    Notes
    -----
    Streams are derived with ``np.random.SeedSequence.spawn``, which guarantees
    statistical independence between children.  Deriving them as
    ``master + 1``, ``master + 2``, ... would *not*: adjacent seeds can produce
    correlated streams in some generators.
    """

    master: int
    _seq: np.random.SeedSequence = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_seq", np.random.SeedSequence(self.master))

    def _child(self, index: int) -> np.random.SeedSequence:
        return self._seq.spawn(4)[index]

    # -- one generator per component --------------------------------------
    def data(self) -> np.random.Generator:
        """Stream for data generation (covariates, errors, censoring)."""
        return np.random.default_rng(self._child(0))

    def split(self) -> np.random.Generator:
        """Stream for train/test partitioning and CV fold assignment."""
        return np.random.default_rng(self._child(1))

    def sampler(self) -> np.random.Generator:
        """Stream for Gehan pair subsampling."""
        return np.random.default_rng(self._child(2))

    def torch_seed(self) -> int:
        """Seed for network initialization and dropout."""
        return int(np.random.default_rng(self._child(3)).integers(0, 2**31 - 1))

    def seed_torch(self) -> None:
        """Seed torch's global RNG for weight initialization.

        Call immediately before constructing a model.  torch has no ergonomic
        per-module generator for weight init, so the global RNG is unavoidable
        here; isolating it to this one call keeps it controlled.
        """
        torch.manual_seed(self.torch_seed())


def make_seeds(master: int) -> SeedBundle:
    return SeedBundle(master=master)
