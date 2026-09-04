"""
Data generation for the recurrent gap-time simulations.

Two things here differ substantively from the original notebooks and are worth
reading before use.

**1. Censoring now reaches the outcome.**  In the original
``prepare_subjects_for_nn``, ``apply_censoring`` wrote the truncated gaps to
``subj['censored_gaps']`` and the indicators to ``subj['delta']``, but the
function then handed the model ``subj['log_gaps']`` -- the *latent, uncensored*
gap times.  The network was therefore trained against the truth while being
told, via delta, that some of those records were censored.  Padding slots past
the censoring point were also passed through as if they were real records.
Here, ``log_gaps_obs`` is built from the truncated gaps and every sequence is
cut at the censoring point, so the observed and latent scales are distinct.
This is the code-level counterpart of the manuscript's distinction between
``Y_ij = log T_ij`` and ``Ytilde_ij = log G_ij`` (Reviewer 1, comment 4).

**2. Five dependence mechanisms.**  The original supported exchangeable frailty
and AR(1).  Reviewer 2 observed that AR(1) is close to the structure a
recurrent network is built to represent, so three further mechanisms are
provided: nonlinear NAR(1), higher-order AR(2), and a regime-switching
event-dependent process.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

# --------------------------------------------------------------------------
# Mean functions (Section 4.1)
# --------------------------------------------------------------------------


def f_linear(X: np.ndarray) -> np.ndarray:
    """f(z) = z1 - 6 z2 + 4 z3."""
    return X[:, 0] - 6.0 * X[:, 1] + 4.0 * X[:, 2]


def f_interaction(X: np.ndarray) -> np.ndarray:
    """f(z) = 3 z1 + 7 z1 z2 - 5 z3."""
    return 3.0 * X[:, 0] + 7.0 * X[:, 0] * X[:, 1] - 5.0 * X[:, 2]


def f_gam(X: np.ndarray) -> np.ndarray:
    """f(z) = z1 + z2^3 + exp(0.9 z3).

    Note: the original notebook's docstring advertised
    ``x1 + 2*x2**3 + sin(0.9*x3)`` while the body computed
    ``x1 + x2**3 + exp(0.9*x3)``.  The body matched the manuscript, so the
    docstring was the error; it is corrected here.
    """
    return X[:, 0] + X[:, 1] ** 3 + np.exp(0.9 * X[:, 2])


MEAN_FUNCTIONS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "linear": f_linear,
    "interaction": f_interaction,
    "gam": f_gam,
}


# --------------------------------------------------------------------------
# Error distributions
# --------------------------------------------------------------------------


def draw_error(name: str, size: int, rng: np.random.Generator) -> np.ndarray:
    """Standardised error draws, all with location 0 and scale 1."""
    if name == "normal":
        return rng.normal(0.0, 1.0, size)
    if name == "gumbel":
        return rng.gumbel(0.0, 1.0, size)
    if name == "logistic":
        return rng.logistic(0.0, 1.0, size)
    raise ValueError(f"unknown error distribution: {name!r}")


ERROR_DISTRIBUTIONS = ("normal", "gumbel", "logistic")


# --------------------------------------------------------------------------
# Within-subject dependence mechanisms (Section 4.1)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DependenceSpec:
    """Specification of a within-subject dependence mechanism.

    Attributes
    ----------
    name : str
        One of ``frailty``, ``ar1``, ``nar1``, ``ar2``, ``event_dependent``.
    rho : float
        Autoregressive coefficient (``ar1``, ``nar1``) or the base coefficient
        for ``event_dependent``.
    phi : tuple
        AR(2) coefficients ``(phi1, phi2)``.
    rho_short, rho_long : float
        Regime coefficients for ``event_dependent``, applied when the previous
        gap is at or below / above the median.
    label : str
        Human-readable label used in table rows.
    """

    name: str
    rho: float = 0.5
    phi: tuple = (0.4, 0.3)
    rho_short: float = 0.8
    rho_long: float = 0.2
    label: str = ""

    def __post_init__(self) -> None:
        if self.name == "ar2":
            phi1, phi2 = self.phi
            # Stationarity for AR(2): roots of 1 - phi1 u - phi2 u^2 outside
            # the unit circle.  Equivalent triangle conditions:
            ok = (phi1 + phi2 < 1.0) and (phi2 - phi1 < 1.0) and (abs(phi2) < 1.0)
            if not ok:
                raise ValueError(
                    f"AR(2) coefficients {self.phi} are not stationary"
                )
        if not self.label:
            object.__setattr__(self, "label", DEFAULT_LABELS.get(self.name, self.name))


DEFAULT_LABELS = {
    "frailty": "Frailty (exchangeable)",
    "ar1": "AR(1), rho=0.5",
    "nar1": "NAR(1), nonlinear",
    "ar2": "AR(2), phi=(0.4,0.3)",
    "event_dependent": "Event-dependent (switching)",
}

#: The five mechanisms reported in Table 5.
DEPENDENCE_SPECS = {
    "frailty": DependenceSpec("frailty"),
    "ar1": DependenceSpec("ar1", rho=0.5),
    "nar1": DependenceSpec("nar1", rho=0.5),
    "ar2": DependenceSpec("ar2", phi=(0.4, 0.3)),
    "event_dependent": DependenceSpec("event_dependent"),
}


def _ar2_innovation_scale(phi1: float, phi2: float) -> float:
    """Innovation sd giving an AR(2) process unit marginal variance.

    For a stationary AR(2), Var(eta) = sigma^2 / (1 - phi1 r1 - phi2 r2) where
    r1, r2 are the first two autocorrelations.  Solving the Yule-Walker
    equations gives r1 = phi1 / (1 - phi2) and r2 = phi2 + phi1 r1.  Setting
    Var(eta) = 1 and returning sigma keeps all five mechanisms on a common
    error scale, so that differences between them reflect dependence structure
    rather than noise level.
    """
    r1 = phi1 / (1.0 - phi2)
    r2 = phi2 + phi1 * r1
    var_ratio = 1.0 - phi1 * r1 - phi2 * r2
    return float(np.sqrt(max(var_ratio, 1e-8)))


def _draw_dependent_errors(
    spec: DependenceSpec,
    ki: int,
    error_dist: str,
    rng: np.random.Generator,
    median_gap: Optional[float],
    mu_i: float,
) -> np.ndarray:
    """Draw the serially dependent component ``eta`` for one subject."""
    if spec.name == "frailty":
        # No serial component; the exchangeable frailty b_i is added by the
        # caller.
        return draw_error(error_dist, ki, rng)

    eta = np.zeros(ki, dtype=float)
    eta[0] = draw_error(error_dist, 1, rng)[0]

    if spec.name == "ar1":
        rho = spec.rho
        scale = np.sqrt(1.0 - rho**2)
        for j in range(1, ki):
            eta[j] = rho * eta[j - 1] + scale * draw_error(error_dist, 1, rng)[0]
        return eta

    if spec.name == "nar1":
        # Saturating transmission: large past errors propagate far less than a
        # linear filter would predict.  Breaks linearity, holds lag order at 1.
        rho = spec.rho
        scale = np.sqrt(1.0 - rho**2)
        for j in range(1, ki):
            eta[j] = rho * np.tanh(2.0 * eta[j - 1]) + scale * draw_error(
                error_dist, 1, rng
            )[0]
        return eta

    if spec.name == "ar2":
        phi1, phi2 = spec.phi
        sigma = _ar2_innovation_scale(phi1, phi2)
        if ki > 1:
            eta[1] = phi1 * eta[0] + sigma * draw_error(error_dist, 1, rng)[0]
        for j in range(2, ki):
            eta[j] = (
                phi1 * eta[j - 1]
                + phi2 * eta[j - 2]
                + sigma * draw_error(error_dist, 1, rng)[0]
            )
        return eta

    if spec.name == "event_dependent":
        # Dependence strength is a function of the realised history: a subject
        # who has just had a short gap enters a strongly dependent regime.
        # Non-stationary conditional on history, so no fixed autocorrelation
        # function describes it.
        if median_gap is None:
            raise ValueError("event_dependent requires median_gap")
        for j in range(1, ki):
            prev_gap = np.exp(mu_i + eta[j - 1])
            rho_j = spec.rho_short if prev_gap <= median_gap else spec.rho_long
            eta[j] = rho_j * eta[j - 1] + np.sqrt(
                1.0 - rho_j**2
            ) * draw_error(error_dist, 1, rng)[0]
        return eta

    raise ValueError(f"unknown dependence mechanism: {spec.name!r}")


# --------------------------------------------------------------------------
# Covariates
# --------------------------------------------------------------------------


def generate_covariates(
    n: int, rng: np.random.Generator, p: int = 3
) -> np.ndarray:
    """Generate the covariate matrix.

    For ``p == 3`` this reproduces the original design: a Bernoulli treatment
    indicator and two correlated Gaussians.  For ``p > 3`` the extra columns
    are independent standard normals, which supports the high-dimensional
    experiments without changing the first three columns.
    """
    x1 = rng.binomial(1, 0.5, size=n)
    x2 = rng.normal(loc=x1 / 3.0, scale=1.0, size=n)
    x3 = rng.normal(loc=x2 / 3.0, scale=1.0, size=n)
    cols = [x1, x2, x3]
    if p > 3:
        cols.extend(rng.normal(0.0, 1.0, size=n) for _ in range(p - 3))
    elif p < 3:
        cols = cols[:p]
    return np.column_stack(cols).astype(np.float64)


# --------------------------------------------------------------------------
# Subject generation
# --------------------------------------------------------------------------


def generate_subjects(
    n: int,
    mean_func: Callable[[np.ndarray], np.ndarray],
    error_dist: str,
    rng: np.random.Generator,
    dependence: DependenceSpec = DEPENDENCE_SPECS["frailty"],
    sigma_b: float = 0.5,
    poisson_lambda: float = 2.0,
    p: int = 3,
) -> List[Dict]:
    """Generate ``n`` subjects with latent (uncensored) gap times.

    Returns a list of dicts with keys ``covariates``, ``log_gaps_true``,
    ``gaps_true`` and ``frailty``.  Censoring is applied separately by
    :func:`apply_censoring`.
    """
    X = generate_covariates(n, rng, p=p)
    mu = mean_func(X)
    K = rng.poisson(lam=poisson_lambda, size=n)

    median_gap = None
    if dependence.name == "event_dependent":
        # The switching threshold is the median gap of the generating
        # distribution.  Estimating it from a pilot draw keeps the threshold a
        # property of the DGP rather than of the realised sample.
        pilot = mu[:, None] + draw_error(error_dist, (min(n, 2000), 1), rng)[:, :1] \
            if False else mu + draw_error(error_dist, n, rng)
        median_gap = float(np.median(np.exp(pilot)))

    subjects: List[Dict] = []
    for i in range(n):
        ki = int(max(K[i], 1))
        b_i = rng.normal(0.0, sigma_b)
        eta = _draw_dependent_errors(
            dependence, ki, error_dist, rng, median_gap, float(mu[i])
        )
        eps = b_i + eta
        log_gaps = mu[i] + eps
        subjects.append(
            {
                "covariates": X[i].astype(np.float64),
                "log_gaps_true": log_gaps.astype(np.float64),
                "gaps_true": np.exp(log_gaps).astype(np.float64),
                "frailty": float(b_i),
            }
        )
    return subjects


def apply_censoring(
    subjects: List[Dict], tau: float, rng: np.random.Generator
) -> List[Dict]:
    """Apply subject-level administrative censoring and truncate sequences.

    A single monitoring window ``C_i ~ Uniform(0, tau)`` is drawn per subject.
    Gaps are observed in full while their cumulative sum stays within the
    window; the gap straddling the window is observed only partially and is
    marked censored; anything after it never happens and is **dropped**.

    This last point is the fix.  The original code kept those trailing slots as
    zero-length records with ``delta = 0`` while simultaneously handing the
    model the true log gap times for them, so the network saw padding as data.

    Adds keys ``gaps_obs``, ``log_gaps_obs``, ``delta``, ``C``, ``K_obs``.
    """
    for subj in subjects:
        gaps = np.asarray(subj["gaps_true"], dtype=float)
        ki = len(gaps)
        if ki == 0:
            subj.update(
                gaps_obs=np.array([]),
                log_gaps_obs=np.array([]),
                delta=np.array([], dtype=np.int64),
                C=np.nan,
                K_obs=0,
            )
            continue

        c_i = float(rng.uniform(0.0, tau))
        cum = np.cumsum(gaps)
        k_obs = int(np.searchsorted(cum, c_i, side="right"))

        if k_obs >= ki:
            # Window outlasts all generated gaps: everything observed.
            gaps_obs = gaps.copy()
            delta = np.ones(ki, dtype=np.int64)
        else:
            prev = float(cum[k_obs - 1]) if k_obs > 0 else 0.0
            partial = max(c_i - prev, 1e-8)  # strictly positive for the log
            gaps_obs = np.concatenate([gaps[:k_obs], [partial]])
            delta = np.concatenate(
                [np.ones(k_obs, dtype=np.int64), [0]]
            ).astype(np.int64)

        subj["gaps_obs"] = gaps_obs
        subj["log_gaps_obs"] = np.log(gaps_obs)
        subj["delta"] = delta
        subj["C"] = c_i
        subj["K_obs"] = k_obs
    return subjects


def calibrate_tau(
    n: int,
    mean_func: Callable[[np.ndarray], np.ndarray],
    error_dist: str,
    rng: np.random.Generator,
    target_censoring: float,
    dependence: DependenceSpec = DEPENDENCE_SPECS["frailty"],
    sigma_b: float = 0.5,
    poisson_lambda: float = 2.0,
    n_pilot: int = 2000,
    tol: float = 0.005,
    max_iter: int = 40,
) -> float:
    """Find the ``tau`` giving a target proportion of censored records.

    The original code hardcoded ``tau = 3000`` and reported whatever censoring
    fraction resulted.  The manuscript reports results at 25%, 50% and 65%
    incomplete follow-up, so tau has to be solved for, not fixed.  Bisection on
    log-tau, since the censoring fraction is monotone decreasing in tau.
    """
    lo, hi = 1e-3, 1e6
    for _ in range(max_iter):
        mid = float(np.sqrt(lo * hi))
        pilot = generate_subjects(
            n_pilot, mean_func, error_dist, rng, dependence, sigma_b, poisson_lambda
        )
        pilot = apply_censoring(pilot, mid, rng)
        frac = censoring_fraction(pilot)
        if abs(frac - target_censoring) < tol:
            return mid
        if frac > target_censoring:
            lo = mid  # too much censoring -> widen the window
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def censoring_fraction(subjects: List[Dict]) -> float:
    """Proportion of retained records that are censored."""
    total = sum(len(s["delta"]) for s in subjects)
    if total == 0:
        return float("nan")
    observed = sum(int(np.sum(s["delta"])) for s in subjects)
    return (total - observed) / total


def to_model_subjects(subjects: List[Dict], keep_truth: bool = False) -> List[Dict]:
    """Reduce to the fields the models consume.

    Uses ``log_gaps_obs`` -- the observed scale.  ``keep_truth`` retains the
    latent values under ``log_gaps_true`` for diagnostics only; nothing in the
    training path may read that key.
    """
    out = []
    for s in subjects:
        rec = {
            "covariates": np.asarray(s["covariates"], dtype=np.float64),
            "log_gaps": np.asarray(s["log_gaps_obs"], dtype=np.float64),
            "delta": np.asarray(s["delta"], dtype=np.int64),
        }
        if keep_truth and "log_gaps_true" in s:
            rec["log_gaps_true"] = np.asarray(s["log_gaps_true"], dtype=np.float64)
        out.append(rec)
    return out


def make_dataset(
    n: int,
    mean_func_name: str,
    error_dist: str,
    rng: np.random.Generator,
    dependence: str = "frailty",
    target_censoring: Optional[float] = None,
    tau: Optional[float] = None,
    sigma_b: float = 0.5,
    poisson_lambda: float = 2.0,
    p: int = 3,
) -> List[Dict]:
    """End-to-end generation: subjects, censoring, reduction to model fields."""
    mean_func = MEAN_FUNCTIONS[mean_func_name]
    spec = DEPENDENCE_SPECS[dependence]
    if tau is None:
        if target_censoring is None:
            raise ValueError("supply either tau or target_censoring")
        tau = calibrate_tau(
            n, mean_func, error_dist, rng, target_censoring, spec, sigma_b, poisson_lambda
        )
    subjects = generate_subjects(
        n, mean_func, error_dist, rng, spec, sigma_b, poisson_lambda, p=p
    )
    subjects = apply_censoring(subjects, tau, rng)
    return to_model_subjects(subjects)
