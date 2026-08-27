"""Volatility models and their one-day-ahead predictive distributions.

The four competitors, in the order used throughout the project:

1. ``forecast_yesterday_volatility`` -- naive baseline.
2. ``forecast_ewma``                 -- EWMA / RiskMetrics baseline.
3. ``fit_garch_mle``                 -- frequentist GARCH(1,1)-t, plug-in predictive.
4. ``sample_garch_posterior``        -- Bayesian GARCH(1,1)-t, posterior predictive.

The central design constraint
-----------------------------
Models 3 and 4 must differ **only** in whether parameter uncertainty is integrated
over. To make that a property of the code rather than an assertion in the write-up,
both call the same ``garch11_t_loglik``. The frequentist path maximises it; the
Bayesian path adds a log prior and samples it. ``arch`` is imported only in tests, as
an independent cross-check on the MLE, and never produces a headline number.

Two uncertainty sources, kept separate
--------------------------------------
- **Innovation uncertainty** -- the Student-t shock. Present in both models 3 and 4.
- **Parameter uncertainty**  -- uncertainty about (mu, omega, alpha, beta, nu).
  Present in model 4's posterior predictive; absent from model 3's plug-in, which
  conditions on the point estimate as if it were the truth.

The plug-in predictive is a scaled Student-t. The posterior predictive is a mixture of
scaled Student-t distributions over posterior draws and has no closed form, so its
quantiles are obtained numerically.

Model specification
-------------------
    r_t   = mu + eps_t
    eps_t = sqrt(h_t) * z_t,   z_t ~ standardised Student-t(nu), unit variance
    h_t   = omega + alpha * eps_{t-1}^2 + beta * h_{t-1}

with omega > 0, alpha >= 0, beta >= 0, alpha + beta < 1, nu > 4. The nu > 4 constraint
(rather than nu > 2) keeps the kurtosis finite. ``z_t`` is standardised to unit
variance so that ``h_t`` is the conditional variance itself, not a scale parameter that
must be rescaled by nu/(nu-2) downstream.

Scale convention
----------------
Every variance in this module is on the **close-to-close return-variance scale**, so
that a predictive interval built from it can be compared against an observed return.
The Parkinson proxy does not arrive on that scale; ``data.scale_proxy`` converts it
before it reaches ``forecast_yesterday_volatility``. See ``data.PROXY_SCALE_C``.

Status: complete. The frequentist half arrived at Stage 2 -- the shared likelihood, the
MLE, and the plug-in predictive, alongside the interface and both baselines from Stage 1.
The Bayesian half arrived at Stage 3: the priors (frozen at decision D4 *before* any
out-of-sample number existed), ``log_prior`` and ``log_posterior`` as the route-B target
density and as an independent statement of the model, ``sample_garch_posterior`` on
PyMC/NUTS, and the mixture posterior predictive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numba
import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.special import gammaln

# --- The forecaster interface -----------------------------------------------------
#
# Every model in this project -- baseline or GARCH, frequentist or Bayesian -- is
# consumed by the backtest through this one interface. That is deliberate: if the
# baselines took a different path from the models under test, a difference in their
# measured calibration could be an artefact of the plumbing rather than of the
# forecasts, and the project's headline comparison would be unfalsifiable.
#
# A forecast for day t+1 is a variance, a predictive mean, and a predictive
# distribution over tomorrow's *return*. The distribution exposes exactly two
# operations, which are all the evaluation layer needs:
#
#   quantile(q) -> the interval bounds at 90/95/99% and the one-sided 99% VaR
#   cdf(r)      -> the PIT value at the realised return
#
# The PIT is computed once, by the harness, from ``cdf``. No model computes its own,
# so no model can compute it a different way.


class PredictiveDistribution(Protocol):
    """One-day-ahead predictive distribution over the **return**."""

    def quantile(self, q: np.ndarray) -> np.ndarray:
        """Return quantiles at probabilities ``q``."""
        ...

    def cdf(self, r: float) -> float:
        """Return ``F(r)`` -- the PIT value when ``r`` is the realised return."""
        ...


@dataclass(frozen=True)
class NormalPredictive:
    """Gaussian predictive distribution, used by both baselines.

    The governing plan specifies normal intervals for EWMA -- it is the "naive UQ"
    baseline, and the point of a baseline is to be naive in a way the real models are
    not. ``forecast_yesterday_volatility`` is given the same treatment for consistency,
    so that the two baselines differ only in how they estimate variance and not in how
    they turn a variance into an interval.
    """

    mean: float
    variance: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.variance) or self.variance <= 0.0:
            raise ValueError(f"variance must be finite and positive, got {self.variance!r}")

    @property
    def sd(self) -> float:
        return float(np.sqrt(self.variance))

    def quantile(self, q: np.ndarray) -> np.ndarray:
        return np.asarray(stats.norm.ppf(q, loc=self.mean, scale=self.sd), dtype=float)

    def cdf(self, r: float) -> float:
        return float(stats.norm.cdf(r, loc=self.mean, scale=self.sd))


@dataclass(frozen=True)
class StudentTPredictive:
    """Standardised Student-t predictive distribution, used by the GARCH-t models.

    ``variance`` is the conditional variance of the **return**, not a scale parameter.
    A raw Student-t with ``nu`` degrees of freedom has variance ``nu / (nu - 2)``, so
    turning a variance into a scale requires

        scale = sqrt(variance) * sqrt((nu - 2) / nu)

    Dropping that second factor is the error this class exists to make impossible. It
    inflates every interval by a few percent -- about 4% at nu = 6 -- and it does so in
    the direction that flatters the Bayesian model at Stage 3, whose intervals are
    supposed to come out wider for a reason that has nothing to do with a missing
    constant. ``test_dropping_the_standardisation_factor_would_widen_intervals`` proves
    the factor is load-bearing rather than decorative.
    """

    mean: float
    variance: float
    nu: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.variance) or self.variance <= 0.0:
            raise ValueError(f"variance must be finite and positive, got {self.variance!r}")
        if not np.isfinite(self.nu) or self.nu <= 2.0:
            raise ValueError(f"nu must be finite and greater than 2, got {self.nu!r}")

    @property
    def scale(self) -> float:
        """Scale parameter of the underlying raw Student-t."""
        return float(np.sqrt(self.variance) * np.sqrt((self.nu - 2.0) / self.nu))

    def quantile(self, q: np.ndarray) -> np.ndarray:
        return np.asarray(
            stats.t.ppf(q, self.nu, loc=self.mean, scale=self.scale), dtype=float
        )

    def cdf(self, r: float) -> float:
        return float(stats.t.cdf(r, self.nu, loc=self.mean, scale=self.scale))


@dataclass(frozen=True)
class StudentTMixturePredictive:
    """Posterior predictive: an equally weighted mixture of standardised Student-t
    components, one per posterior draw.

    This is the **only** structural difference between models 3 and 4. Both condition on
    the same likelihood and the same data; the plug-in collapses the posterior to a
    point and takes one component, while this integrates over it. Any difference in
    their intervals is parameter uncertainty, full stop -- which is only true if the
    components differ from each other, hence the arrays.

    ``variances`` are conditional variances of the **return**, one per draw, and each
    component is standardised the same way ``StudentTPredictive`` is: the scale is
    smaller than the volatility by ``sqrt((nu - 2) / nu)``. Every draw carries its own
    ``nu``, so the mixture is over three varying quantities, not one.

    Equal weights because posterior draws are already a sample from the posterior;
    weighting them by anything would be counting the same information twice.

    No closed form, so quantiles are solved numerically from the mixture CDF -- a convex
    combination of component CDFs, hence monotone, hence safe to invert by bracketing.
    The bracket is exact rather than heuristic: at ``min_i Q_i(p)`` every component CDF
    is at most ``p`` so the mixture CDF is too, and at ``max_i Q_i(p)`` every component
    CDF is at least ``p``. The root is therefore always enclosed.
    """

    means: np.ndarray
    variances: np.ndarray
    nus: np.ndarray

    def __post_init__(self) -> None:
        means = np.asarray(self.means, dtype=float)
        variances = np.asarray(self.variances, dtype=float)
        nus = np.asarray(self.nus, dtype=float)
        if means.ndim != 1 or means.size == 0:
            raise ValueError(f"means must be a non-empty 1-D array, got {means.shape}")
        if variances.shape != means.shape or nus.shape != means.shape:
            raise ValueError(
                "means, variances and nus must have the same shape, got "
                f"{means.shape}, {variances.shape}, {nus.shape}"
            )
        if not np.all(np.isfinite(means)):
            raise ValueError("every component mean must be finite")
        if not np.all(np.isfinite(variances)) or np.any(variances <= 0.0):
            raise ValueError("every component variance must be finite and positive")
        if not np.all(np.isfinite(nus)) or np.any(nus <= 2.0):
            raise ValueError("every component nu must be finite and greater than 2")
        object.__setattr__(self, "means", means)
        object.__setattr__(self, "variances", variances)
        object.__setattr__(self, "nus", nus)

    @property
    def n_components(self) -> int:
        return int(self.means.size)

    @property
    def scales(self) -> np.ndarray:
        """Scale parameters of the underlying raw Student-t components."""
        return np.sqrt(self.variances) * np.sqrt((self.nus - 2.0) / self.nus)

    @property
    def mean(self) -> float:
        """Predictive mean: the posterior mean of ``mu``.

        Finite for every admissible draw because ``nu > 2`` throughout, and in fact
        ``nu > 4`` under the D4 prior.
        """
        return float(np.mean(self.means))

    def cdf(self, r: float) -> float:
        return float(
            np.mean(stats.t.cdf(float(r), self.nus, loc=self.means, scale=self.scales))
        )

    def quantile(self, q: np.ndarray) -> np.ndarray:
        probs = np.asarray(q, dtype=float)
        flat = np.atleast_1d(probs).ravel()
        if np.any(flat <= 0.0) or np.any(flat >= 1.0):
            raise ValueError("quantile levels must lie strictly in (0, 1)")

        scales = self.scales
        out = np.empty(flat.size, dtype=float)
        for i, p in enumerate(flat):
            component = stats.t.ppf(p, self.nus, loc=self.means, scale=scales)
            lo = float(np.min(component))
            hi = float(np.max(component))
            if hi - lo <= _MIXTURE_DEGENERATE_TOL * max(1.0, abs(hi)):
                # Every component agrees: the mixture *is* that component, and there is
                # no root to bracket. This is the degenerate case a posterior collapsed
                # to a point mass produces, and it must reproduce the plug-in exactly.
                out[i] = 0.5 * (lo + hi)
                continue
            out[i] = optimize.brentq(lambda x, p=p: self.cdf(x) - p, lo, hi)
        # Reshape rather than return the flat vector, so a scalar level gives a scalar
        # back exactly as ``StudentTPredictive.quantile`` does.
        return out.reshape(probs.shape)


@dataclass(frozen=True)
class Forecast:
    """One model's one-day-ahead forecast for one date.

    ``variance`` is on the close-to-close return-variance scale, the same scale as the
    (rescaled) evaluation proxy, so it is directly comparable across all four models.
    """

    variance: float
    mean: float
    distribution: PredictiveDistribution


# --- Parameter handling -----------------------------------------------------------

#: Canonical parameter order for the flat vectors passed to the likelihood, the
#: optimiser, and the sampler. Everything in this module assumes this order.
PARAM_NAMES: tuple[str, ...] = ("mu", "omega", "alpha", "beta", "nu")

#: RiskMetrics daily decay factor.
EWMA_LAMBDA = 0.94

#: Below this relative spread, the components of a posterior predictive mixture are
#: treated as identical and the mixture quantile is read off directly. Reached when a
#: posterior is collapsed to a point mass, which is how the tests establish that this
#: mixture reproduces the plug-in predictive exactly in the no-parameter-uncertainty
#: limit rather than merely resembling it.
_MIXTURE_DEGENERATE_TOL = 1e-12


@dataclass(frozen=True)
class GarchParams:
    """A single GARCH(1,1)-t parameter vector, in natural (unconstrained-free) space."""

    mu: float
    omega: float
    alpha: float
    beta: float
    nu: float

    def to_array(self) -> np.ndarray:
        """Return the parameters in ``PARAM_NAMES`` order."""
        return np.array([self.mu, self.omega, self.alpha, self.beta, self.nu], dtype=float)

    @classmethod
    def from_array(cls, theta: np.ndarray) -> GarchParams:
        """Build from a flat vector in ``PARAM_NAMES`` order."""
        values = np.asarray(theta, dtype=float)
        if values.shape != (len(PARAM_NAMES),):
            raise ValueError(
                f"theta must have shape ({len(PARAM_NAMES)},), got {values.shape}"
            )
        return cls(*(float(v) for v in values))

    def is_valid(self) -> bool:
        """True if the parameters satisfy positivity, stationarity, and nu > 4.

        ``nu`` is NaN for the normal-innovation variant, where the degrees of freedom
        are not a parameter at all; the ``nu > 4`` constraint is vacuous there and is
        skipped rather than failed. Every other constraint applies to both variants.
        """
        if not np.isfinite([self.mu, self.omega, self.alpha, self.beta]).all():
            return False
        if self.omega <= 0.0 or self.alpha < 0.0 or self.beta < 0.0:
            return False
        if self.alpha + self.beta >= 1.0:
            return False
        if np.isnan(self.nu):
            return True  # normal innovations: nu is not a parameter
        return bool(np.isfinite(self.nu) and self.nu > 4.0)


@dataclass(frozen=True)
class FrequentistFit:
    """Result of maximising the GARCH(1,1)-t likelihood.

    Attributes
    ----------
    params:
        The maximum likelihood estimate. A *point*; the plug-in predictive treats it as
        if it were the true parameter, which is precisely the approximation this
        project is testing.
    loglik:
        Maximised log-likelihood.
    converged:
        Whether the optimiser reported success. **Never** discard this. A failed fit
        must propagate to the backtest record and to the report, not be quietly
        replaced by the previous window's estimate.
    message:
        The optimiser's termination message, retained verbatim.
    n_obs:
        Observations used.
    """

    params: GarchParams
    loglik: float
    converged: bool
    message: str
    n_obs: int


@dataclass(frozen=True)
class BayesianFit:
    """Posterior draws from the GARCH(1,1)-t model, with NUTS diagnostics.

    Re-keyed at Stage 3 from the emcee fields the Stage 1 stub carried
    (``acceptance_fraction``, ``autocorr_time``, ``n_effective``), which describe a
    random-walk sampler and have no NUTS analogue. Decision B1-R adopted PyMC/NUTS as
    route A; the diagnostics stored here are the ones that route actually produces and
    that the write-up is obliged to report. Recorded in research_log.md 1.10.

    Mirrors ``FrequentistFit`` deliberately: both carry ``converged`` and a verbatim
    ``message``, so the backtest can apply one rule to a failed fit regardless of which
    model produced it (decision D16, extended to the Bayesian track by D19).

    Attributes
    ----------
    draws:
        Array of shape ``(n_draws, 5)`` in ``PARAM_NAMES`` order, post warm-up and post
        thinning, with chains flattened. On the **raw return scale**: sampling runs on
        percent returns (D14) and the draws are converted back before they are stored,
        so nothing outside the sampler sees the percent convention.
    log_prob:
        Unnormalised log posterior density at each retained draw, shape ``(n_draws,)``.
    r_hat:
        Split-R-hat per parameter, shape ``(5,)``, in ``PARAM_NAMES`` order. The
        convergence criterion, not a decoration.
    ess_bulk:
        Bulk effective sample size per parameter, shape ``(5,)``. Governs how well the
        posterior *centre* is resolved.
    ess_tail:
        Tail effective sample size per parameter, shape ``(5,)``. Governs how well the
        posterior *tails* are resolved, which is what the 99% predictive quantiles are
        built from -- so for this project it is the more relevant of the two.
    n_divergences:
        Number of divergent transitions. A divergence means the sampler failed to
        explore part of the posterior geometry; the draws are then not a sample from
        the target and no amount of them fixes it.
    converged:
        Whether every diagnostic threshold in ``BAYES_CONVERGENCE`` was met. **Never**
        discard this. A failed fit propagates to the backtest record and to the report;
        it is not replaced by the previous window's draws.
    message:
        Which threshold failed, and by how much, retained verbatim for the record.
    n_obs:
        Observations used.
    seed:
        The seed used, so the chain can be reproduced exactly.
    """

    draws: np.ndarray
    log_prob: np.ndarray
    r_hat: np.ndarray
    ess_bulk: np.ndarray
    ess_tail: np.ndarray
    n_divergences: int
    converged: bool
    message: str
    n_obs: int
    seed: int


# --- The shared likelihood --------------------------------------------------------


def garch11_filter(theta: np.ndarray, returns: np.ndarray, h0: float) -> np.ndarray:
    """Run the conditional variance recursion; return ``h`` of the same length.

    ``h[t]`` is the conditional variance of ``returns[t]`` given information through
    ``t-1``. Sequential by construction and therefore the pipeline's hot loop; this is
    the function that is numba-compiled (decision B1).

    ``h0`` seeds ``h[0]`` and must be backcast from the **training** returns only
    (decision D5), never from the full sample.

    Parameter validity is the caller's responsibility: the likelihood functions below
    check ``is_valid`` before calling this, and the backtest only ever passes a fitted
    parameter vector. Handed an inadmissible vector this will happily produce negative
    variances rather than silently clamping them, because a clamp would turn a bug into
    a plausible number.
    """
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"returns must be one-dimensional, got shape {values.shape}")
    if not np.isfinite(h0) or h0 <= 0.0:
        raise ValueError(f"h0 must be finite and positive, got {h0!r}")
    mu, omega, alpha, beta, _nu = (float(v) for v in np.asarray(theta, dtype=float))
    return _filter_kernel(values, mu, omega, alpha, beta, float(h0))


@numba.njit(cache=True)
def _filter_kernel(
    returns: np.ndarray, mu: float, omega: float, alpha: float, beta: float, h0: float
) -> np.ndarray:  # pragma: no cover - compiled; covered through ``garch11_filter``
    """The recursion itself, compiled (decision B1).

    Sequential and therefore the pipeline's hot loop: the backtest runs it 204 times
    for the forecast paths and some hundreds of thousands of times inside the optimiser.
    In pure Python the multi-start MLE over 102 refits takes tens of minutes; compiled
    it takes under a minute.
    """
    n = returns.shape[0]
    h = np.empty(n, dtype=np.float64)
    if n == 0:
        return h
    h[0] = h0
    for t in range(1, n):
        resid = returns[t - 1] - mu
        h[t] = omega + alpha * resid * resid + beta * h[t - 1]
    return h


def garch11_filter_by_draw(
    draws: np.ndarray, returns: np.ndarray, h0: float, positions: np.ndarray
) -> np.ndarray:
    """Conditional variances at ``positions``, one row per posterior draw.

    **Why this exists rather than a loop over ``garch11_filter``.** Every posterior draw
    implies its own filtered variance path, so a Bayesian forecast for a single day needs
    one ``h`` per draw -- 2,000 of them (D18), on every one of 2,134 evaluation days. The
    variances themselves are only wanted at the dates a refit serves, so the recursion is
    run once per draw and read off at ``positions``, and the full path is never
    materialised.

    Identical arithmetic to ``garch11_filter``, which
    ``test_per_draw_filter_matches_the_single_draw_filter`` asserts on the same inputs.
    A second recursion that drifted from the first would put the Bayesian and
    frequentist models on different models while they still shared a likelihood
    function, which is precisely the comparison this project rests on.

    Parameters
    ----------
    draws:
        Posterior draws, shape ``(n_draws, 5)`` in ``PARAM_NAMES`` order.
    returns:
        Returns from the sample start through the last date wanted -- and no further.
        The caller is responsible for that slice; it is the same look-ahead surface
        ``build_garch_paths`` manages for the frequentist track.
    h0:
        Seed for the recursion, backcast from the estimation window alone.
    positions:
        Indices into ``returns``, strictly increasing.

    Returns
    -------
    Array of shape ``(n_draws, len(positions))``.
    """
    theta = np.asarray(draws, dtype=float)
    if theta.ndim != 2 or theta.shape[1] != len(PARAM_NAMES):
        raise ValueError(
            f"draws must have shape (n_draws, {len(PARAM_NAMES)}), got {theta.shape}"
        )
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"returns must be one-dimensional, got shape {values.shape}")
    if not np.isfinite(h0) or h0 <= 0.0:
        raise ValueError(f"h0 must be finite and positive, got {h0!r}")

    index = np.asarray(positions, dtype=np.int64)
    if index.ndim != 1 or index.size == 0:
        raise ValueError(f"positions must be a non-empty 1-D array, got {index.shape}")
    if np.any(np.diff(index) <= 0):
        raise ValueError("positions must be strictly increasing")
    if index[0] < 0 or index[-1] >= values.size:
        raise ValueError("positions must index into returns")

    return _filter_by_draw_kernel(theta, values, float(h0), index)


@numba.njit(cache=True)
def _filter_by_draw_kernel(
    draws: np.ndarray, returns: np.ndarray, h0: float, positions: np.ndarray
) -> np.ndarray:  # pragma: no cover - compiled; covered through ``garch11_filter_by_draw``
    """The per-draw recursion, compiled. See ``_filter_kernel`` for why."""
    n_draws = draws.shape[0]
    n_wanted = positions.shape[0]
    last = positions[n_wanted - 1]
    out = np.empty((n_draws, n_wanted), dtype=np.float64)

    for d in range(n_draws):
        mu = draws[d, 0]
        omega = draws[d, 1]
        alpha = draws[d, 2]
        beta = draws[d, 3]
        h = h0
        j = 0
        if positions[0] == 0:
            out[d, 0] = h
            j = 1
        for t in range(1, last + 1):
            resid = returns[t - 1] - mu
            h = omega + alpha * resid * resid + beta * h
            if j < n_wanted and positions[j] == t:
                out[d, j] = h
                j += 1
    return out


def garch11_t_loglik(theta: np.ndarray, returns: np.ndarray, h0: float) -> float:
    """Log-likelihood of the GARCH(1,1)-t model. **Shared by both models 3 and 4.**

    Returns ``-inf`` for parameters outside the admissible region rather than raising,
    so the same function can be handed to an optimiser and to an MCMC sampler.

    Uses the *standardised* Student-t density (unit variance), so that ``h_t`` from
    ``garch11_filter`` is the conditional variance directly:

        z_t = (r_t - mu) / sqrt(h_t)
        ll_t = lgamma((nu+1)/2) - lgamma(nu/2) - 0.5*log(pi*(nu-2))
               - 0.5*log(h_t)
               - ((nu+1)/2) * log(1 + z_t^2/(nu-2))

    The ``-0.5*log(h_t)`` term is the Jacobian of ``r_t = mu + sqrt(h_t) * z_t``. It is
    the classic silent omission in a hand-written GARCH likelihood: without it the
    optimiser still converges and the estimates still look plausible, but every variance
    is wrong and nothing anywhere fails. It is pinned by
    ``test_t_loglik_matches_an_independent_scipy_implementation``, which computes the
    same quantity a completely different way rather than restating this formula.
    """
    params = GarchParams.from_array(theta)
    if not params.is_valid() or np.isnan(params.nu):
        return -np.inf

    values = np.asarray(returns, dtype=float)
    h = garch11_filter(theta, values, h0)
    if not np.all(np.isfinite(h)) or np.any(h <= 0.0):
        return -np.inf

    nu = params.nu
    z_squared = (values - params.mu) ** 2 / h
    constant = (
        gammaln(0.5 * (nu + 1.0))
        - gammaln(0.5 * nu)
        - 0.5 * np.log(np.pi * (nu - 2.0))
    )
    ll = (
        values.size * constant
        - 0.5 * np.sum(np.log(h))
        - 0.5 * (nu + 1.0) * np.sum(np.log1p(z_squared / (nu - 2.0)))
    )
    return float(ll) if np.isfinite(ll) else -np.inf


def garch11_normal_loglik(theta: np.ndarray, returns: np.ndarray, h0: float) -> float:
    """Log-likelihood of the GARCH(1,1) model with **Gaussian** innovations.

    The Stage 6 ablation. Normal versus t is the cheap, decisive lever on 99% tail
    coverage, and fitting it now costs almost nothing because it shares
    ``garch11_filter`` with the t version.

    Deliberately a separate function rather than a branch inside ``garch11_t_loglik``:
    the project's central claim is that models 3 and 4 share *one* likelihood, and that
    is a stronger claim if the function they share has no innovation-distribution switch
    in it. ``theta[4]`` (``nu``) is ignored -- overwritten with NaN before validation, so
    that the value passed in genuinely cannot affect the result.
    """
    supplied = GarchParams.from_array(theta)
    params = GarchParams(
        mu=supplied.mu,
        omega=supplied.omega,
        alpha=supplied.alpha,
        beta=supplied.beta,
        nu=float("nan"),
    )
    if not params.is_valid():
        return -np.inf

    values = np.asarray(returns, dtype=float)
    h = garch11_filter(theta, values, h0)
    if not np.all(np.isfinite(h)) or np.any(h <= 0.0):
        return -np.inf

    z_squared = (values - params.mu) ** 2 / h
    ll = -0.5 * np.sum(np.log(2.0 * np.pi) + np.log(h) + z_squared)
    return float(ll) if np.isfinite(ll) else -np.inf


#: The two innovation distributions, keyed by the name used throughout the backtest.
LOGLIK_BY_INNOVATION = {"t": garch11_t_loglik, "normal": garch11_normal_loglik}


# --- Priors (decision D4, research_log.md 1.10) ------------------------------------
#
# Frozen on 2026-08-26, before any Bayesian out-of-sample number existed. The ordering
# is the point, not the numbers: this project's headline claim is that integrating over
# parameter uncertainty improves interval coverage, and loose priors widen Bayesian
# intervals for free. Priors chosen after seeing a coverage table would make that claim
# unfalsifiable.
#
# Specified on the **percent return scale** (``FIT_SCALE``), the same convention the MLE
# optimises on:
#
#     mu    ~ Normal(0, 1)
#     omega ~ HalfNormal(1)
#     alpha ~ Beta(2, 10)
#     delta ~ Beta(3, 1),   beta = (1 - alpha) * delta
#     nu    ~ Exponential(mean 10) truncated below at 4
#
# Every support constraint falls out of the parameterisation rather than being imposed
# on top of it: ``omega > 0`` from the half-normal, ``alpha`` and ``delta`` in [0, 1]
# from the betas, ``beta >= 0`` and ``alpha + beta < 1`` from the transform, ``nu > 4``
# from the truncation. No rejection sampling and no hand-written bounds -- which is the
# reason this parameterisation was chosen over sampling ``beta`` directly, since
# stationarity then never has to be rejected by the sampler.
#
# ``delta ~ Beta(3, 1)`` departs from the feasibility probe's ``Beta(10, 2)``, which
# research_log.md 1.6 recorded as a starting point for D4 to confirm. It did not survive
# confirmation: a ``Beta(a, b)`` density vanishes at 1 whenever ``b > 1``, so
# ``Beta(10, 2)`` places **zero** prior density at the stationarity boundary that 1.9
# recorded the data pressing against -- ``alpha + beta`` at or above 0.999 in 16 of the
# 102 MLE refits. The full argument, and the measurement of how little it moves the
# answer, are in research_log.md 1.10.
PRIOR_MU_SD = 1.0
PRIOR_OMEGA_SD = 1.0
PRIOR_ALPHA: tuple[float, float] = (2.0, 10.0)
PRIOR_DELTA: tuple[float, float] = (3.0, 1.0)
PRIOR_NU_MEAN = 10.0
PRIOR_NU_LOWER = 4.0


def log_prior(theta: np.ndarray) -> float:
    """Log prior over (mu, omega, alpha, beta, nu). Returns ``-inf`` outside support.

    The priors themselves are frozen at D4 and stated once, in the block above.

    **Percent scale.** ``theta`` must follow the ``FIT_SCALE`` convention (D14), which is
    where D4 froze the priors and where both the optimiser and the sampler work. Handed a
    raw-return-scale vector this returns a number rather than raising -- it is a density,
    and every admissible vector has one -- so the convention is the caller's
    responsibility. ``sample_garch_posterior`` scales on the way in and back on the way
    out, so nothing outside this module has to know.

    **A density over ``beta``, not over ``delta``.** The prior is specified through
    ``beta = (1 - alpha) * delta``, so the change of variable at fixed ``alpha`` carries
    a Jacobian:

        p(beta | alpha) = p_delta(beta / (1 - alpha)) / (1 - alpha)

    PyMC samples ``delta`` and so evaluates ``p_delta`` directly; the two therefore
    differ by exactly ``log(1 - alpha)``. That is asserted rather than assumed, by
    ``test_the_pymc_graph_and_the_numpy_log_posterior_agree``. Dropping the Jacobian
    would be a real error on the route-B path, where this function *is* the target
    density rather than a cross-check on one.

    Under decision B1-R the sampler is PyMC/NUTS, which takes its priors as distribution
    objects in the model graph. This function is the route-B (emcee) entry point and an
    independent statement of the same priors -- one specification, two implementations,
    each checked against the other.
    """
    params = GarchParams.from_array(theta)
    if not params.is_valid() or np.isnan(params.nu):
        return -np.inf

    # ``is_valid`` has already established alpha + beta < 1 with both non-negative, so
    # 1 - alpha is strictly positive and delta lands in [0, 1).
    delta = params.beta / (1.0 - params.alpha)

    lp = float(stats.norm.logpdf(params.mu, 0.0, PRIOR_MU_SD))
    lp += float(stats.halfnorm.logpdf(params.omega, 0.0, PRIOR_OMEGA_SD))
    lp += float(stats.beta.logpdf(params.alpha, *PRIOR_ALPHA))
    lp += float(stats.beta.logpdf(delta, *PRIOR_DELTA)) - float(np.log1p(-params.alpha))
    lp += float(
        stats.expon.logpdf(params.nu, loc=PRIOR_NU_LOWER, scale=PRIOR_NU_MEAN)
    )
    return lp if np.isfinite(lp) else -np.inf


def log_posterior(theta: np.ndarray, returns: np.ndarray, h0: float) -> float:
    """``log_prior + garch11_t_loglik``. The target density for the sampler.

    Unnormalised, as a target density is. Percent scale throughout, for the reason
    ``log_prior`` gives: ``returns`` and ``h0`` must be on the same scale as ``theta``.

    Short-circuits on an inadmissible ``theta`` so the likelihood -- a recursion over the
    whole estimation window -- is never run at a point the prior has already excluded.
    """
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + garch11_t_loglik(theta, returns, h0)


def backcast_initial_variance(returns: np.ndarray) -> float:
    """Seed value for ``h[0]``, computed from training returns only.

    Must be called on the estimation window alone. Passing the full sample here would
    leak future information into every fit.
    """
    values = np.asarray(returns, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        raise ValueError("need at least two finite returns to backcast h0")
    h0 = float(np.var(finite, ddof=1))
    if not np.isfinite(h0) or h0 <= 0.0:
        raise ValueError(f"backcast variance must be finite and positive, got {h0!r}")
    return h0


def standardised_residuals(
    params: GarchParams, returns: np.ndarray, h0: float
) -> np.ndarray:
    """``(r_t - mu) / sqrt(h_t)`` -- the residual diagnostics' input.

    Under correct specification these are i.i.d. draws from the standardised innovation
    distribution: no autocorrelation in the levels (the mean equation is adequate) and
    none in the squares (the variance equation has absorbed the clustering). Both are
    tested with ``eda.ljung_box_test``, and the QQ plot checks the tail shape against
    the fitted t.
    """
    values = np.asarray(returns, dtype=float)
    h = garch11_filter(params.to_array(), values, h0)
    return (values - params.mu) / np.sqrt(h)


# --- Estimation -------------------------------------------------------------------


#: Internal fitting scale. The likelihood is maximised on ``returns * FIT_SCALE`` and
#: the estimates are converted back before they leave this module, so nothing outside
#: ``fit_garch_mle`` ever sees the percent convention.
#:
#: Daily SPY log returns have a standard deviation near 0.011, which puts ``omega`` at
#: around 1e-6 -- below L-BFGS-B's default convergence tolerances, so the optimiser
#: stops on a parameter it has barely moved. On the percent scale ``omega`` is around
#: 0.02 and the five parameters are within a couple of orders of magnitude of each
#: other. The Stage 3 feasibility probe found the same thing about the sampler's
#: geometry, so using it here keeps the two stages on one convention.
FIT_SCALE = 100.0

#: Fixed multi-start grid. Fixed rather than random so that a refit is reproducible
#: bit-for-bit; ``(alpha, beta)`` pairs that violate stationarity are dropped rather
#: than started from the boundary.
_START_ALPHAS = (0.05, 0.10, 0.20)
_START_BETAS = (0.75, 0.85, 0.90)
_START_NUS = (5.0, 8.0, 15.0)

#: Value the objective returns where the likelihood is ``-inf``. L-BFGS-B approximates
#: its gradient by finite differences and cannot step through an infinite wall, so the
#: barrier has to be finite even though the likelihood itself is not.
_PENALTY = 1e10


def _fit_bounds(innovation: str) -> list[tuple[float, float]]:
    """Box constraints on the percent scale, in ``PARAM_NAMES`` order."""
    bounds = [
        (-5.0, 5.0),  # mu
        (1e-8, 10.0),  # omega
        (1e-8, 0.999),  # alpha
        (1e-8, 0.999),  # beta
    ]
    # nu > 4 keeps the kurtosis finite; the upper bound is where the t is numerically
    # indistinguishable from a normal and the likelihood is flat in nu.
    bounds.append((4.001, 300.0) if innovation == "t" else (4.0, 4.0))
    return bounds


def _start_grid(returns: np.ndarray, innovation: str) -> list[np.ndarray]:
    """Deterministic starting points, on the percent scale."""
    mu0 = float(np.mean(returns))
    var0 = float(np.var(returns, ddof=1))
    nus = _START_NUS if innovation == "t" else (4.0,)

    starts: list[np.ndarray] = []
    for alpha in _START_ALPHAS:
        for beta in _START_BETAS:
            if alpha + beta >= 0.995:
                continue
            omega = var0 * (1.0 - alpha - beta)
            for nu in nus:
                starts.append(np.array([mu0, omega, alpha, beta, nu], dtype=float))
    return starts


def fit_garch_mle(
    returns: np.ndarray,
    *,
    h0: float | None = None,
    innovation: Literal["t", "normal"] = "t",
    start_params: np.ndarray | None = None,
) -> FrequentistFit:
    """Maximise ``garch11_t_loglik`` via ``scipy.optimize``.

    Convergence failures are returned in the ``FrequentistFit``, never swallowed and
    never silently retried into a spurious success.

    Parameters
    ----------
    returns:
        Estimation-window returns, on the raw return scale.
    h0:
        Seed for the variance recursion. Defaults to ``backcast_initial_variance`` of
        ``returns`` -- which is correct precisely because ``returns`` is the estimation
        window and nothing else.
    innovation:
        ``"t"`` for the headline model, ``"normal"`` for the Stage 6 ablation. The
        normal variant returns ``nu = nan``: it has no degrees-of-freedom parameter, and
        recording it as NaN rather than as some sentinel number means anything that
        forgets to branch on it produces a NaN rather than a plausible-looking result.
    start_params:
        Optional single starting point on the **raw** return scale, used instead of the
        multi-start grid. For tests and diagnostics; the backtest never passes it.

    Notes
    -----
    Optimisation runs on ``returns * FIT_SCALE`` (see that constant). Every value in the
    returned ``FrequentistFit`` -- parameters and log-likelihood alike -- is converted
    back to the raw return scale, so the percent convention is invisible from outside.
    """
    if innovation not in LOGLIK_BY_INNOVATION:
        raise ValueError(f"innovation must be 't' or 'normal', got {innovation!r}")

    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 30:
        raise ValueError(
            f"need at least 30 finite returns to fit GARCH(1,1), got {values.size}"
        )

    h0_raw = backcast_initial_variance(values) if h0 is None else float(h0)
    if not np.isfinite(h0_raw) or h0_raw <= 0.0:
        raise ValueError(f"h0 must be finite and positive, got {h0!r}")

    scaled = values * FIT_SCALE
    h0_scaled = h0_raw * FIT_SCALE**2
    loglik_fn = LOGLIK_BY_INNOVATION[innovation]

    def negative_loglik(theta: np.ndarray) -> float:
        ll = loglik_fn(theta, scaled, h0_scaled)
        return _PENALTY if not np.isfinite(ll) else -ll

    if start_params is not None:
        raw_start = np.asarray(start_params, dtype=float)
        starts = [
            np.array(
                [
                    raw_start[0] * FIT_SCALE,
                    raw_start[1] * FIT_SCALE**2,
                    raw_start[2],
                    raw_start[3],
                    raw_start[4] if innovation == "t" else 4.0,
                ],
                dtype=float,
            )
        ]
    else:
        starts = _start_grid(scaled, innovation)

    bounds = _fit_bounds(innovation)
    lower = [lo for lo, _ in bounds]
    upper = [hi for _, hi in bounds]

    results: list[optimize.OptimizeResult] = []
    for start in starts:
        clipped = np.clip(start, lower, upper)
        results.append(
            optimize.minimize(
                negative_loglik, clipped, method="L-BFGS-B", bounds=bounds
            )
        )

    # Pick the best optimum **among the starts that actually converged**, falling back
    # to the best of the rest only when none did -- in which case ``converged`` comes
    # out False and says so.
    #
    # This is the ordinary multi-start rule, not a retry-until-success loop: the grid is
    # fixed, every start is run exactly once, and no failure is re-run in the hope of a
    # better verdict. Ranking purely on the objective would let one abnormally
    # terminated start displace an ordinary success that landed a hair behind it, and
    # the fit would then be reported as failed on the strength of a start that was never
    # the answer. That is what happened at the 2022-09-06 refit before this rule existed.
    def _acceptable(result: optimize.OptimizeResult) -> bool:
        candidate = GarchParams.from_array(result.x)
        if innovation == "normal":
            # nu is not a parameter here; validate it the same way the normal
            # likelihood does, by ignoring whatever the optimiser left in that slot.
            candidate = GarchParams(
                mu=candidate.mu,
                omega=candidate.omega,
                alpha=candidate.alpha,
                beta=candidate.beta,
                nu=float("nan"),
            )
        return bool(result.success) and result.fun < _PENALTY and candidate.is_valid()

    acceptable = [r for r in results if _acceptable(r)]
    best = min(acceptable or results, key=lambda r: float(r.fun))

    scaled_params = GarchParams.from_array(best.x)
    params = GarchParams(
        mu=scaled_params.mu / FIT_SCALE,
        omega=scaled_params.omega / FIT_SCALE**2,
        alpha=scaled_params.alpha,
        beta=scaled_params.beta,
        nu=scaled_params.nu if innovation == "t" else float("nan"),
    )

    # The likelihood of r differs from the likelihood of r * FIT_SCALE by the Jacobian
    # of the change of variable, one factor of FIT_SCALE per observation. NaN when no
    # start reached an admissible point at all, because the penalty value is not a
    # log-likelihood and must not be reported as one.
    hit_barrier = best.fun >= _PENALTY
    loglik = (
        float("nan")
        if hit_barrier
        else -float(best.fun) + values.size * np.log(FIT_SCALE)
    )

    # ``converged`` means the optimiser succeeded *and* landed somewhere admissible.
    # Both halves matter: a "successful" termination on the stationarity barrier is not
    # a fit, and reporting it as one is exactly the failure this field exists to
    # prevent. The optimiser's own message is carried verbatim either way.
    converged = bool(best.success) and params.is_valid() and not hit_barrier
    message = str(best.message)
    if bool(best.success) and not converged:
        message = f"{message} [rejected: optimum is outside the admissible region]"

    return FrequentistFit(
        params=params,
        loglik=loglik,
        converged=converged,
        message=message,
        n_obs=int(values.size),
    )


#: Diagnostic thresholds a Bayesian refit must clear to produce forecasts (decision
#: D19). Fixed before any refit was run, for the reason a threshold exists at all: one
#: loosened after it fires is not a diagnostic.
#:
#: Zero divergences rather than a small tolerance, because a divergence is not noise --
#: it says the sampler failed to explore part of the posterior geometry, so the draws
#: are not a sample from the target and more of them do not fix it. ``ess_tail`` rather
#: than ``ess_bulk`` because the 99% predictive quantiles are built from the tails, and
#: the 99% level is where this project's finding lives.
BAYES_CONVERGENCE: dict[str, float] = {
    "max_r_hat": 1.01,
    "max_divergences": 0,
    "min_ess_tail": 400.0,
}

#: Variables whose diagnostics are checked. ``delta`` is sampled and ``beta`` is the
#: deterministic transform of it, so D19's "every sampled parameter" is covered by the
#: first five plus ``delta``; ``beta`` is carried as well because it is the quantity the
#: forecast path actually uses and the one the write-up reports.
_DIAGNOSTIC_VARS: tuple[str, ...] = PARAM_NAMES + ("delta",)


def _build_pymc_model(
    returns: np.ndarray,
    h0: float,
    *,
    prior_delta: tuple[float, float] = PRIOR_DELTA,
):
    """The GARCH(1,1)-t model graph, on the percent scale. Imports PyMC lazily.

    ``prior_delta`` defaults to the frozen D4 value and exists **only** so the Stage 6
    prior-sensitivity check can re-run the backtest under the two candidates D4
    considered and rejected. Passing anything else re-specifies the model, so it is an
    explicit argument at every call site rather than a module constant that could be
    edited: the frozen priors stay frozen, and a sensitivity run is visibly a different
    run. ``test_the_delta_prior_default_is_the_frozen_one`` pins the default.

    **The same likelihood as ``garch11_t_loglik``, expressed for a different engine.**
    That is the project's central design constraint (see the module docstring), so the
    two must agree to floating-point tolerance on any admissible parameter vector, and
    ``test_the_pymc_graph_and_the_numpy_log_posterior_agree`` checks exactly that. In
    particular this graph observes **every** return with ``h[0] = h0``, matching the
    NumPy likelihood, rather than dropping the first observation as the 1.6 feasibility
    probe did.

    PyMC is imported inside the function rather than at module scope because importing
    it costs several seconds, and ``models`` is imported by every test in the suite
    including the many that never sample.
    """
    import pymc as pm
    import pytensor
    import pytensor.tensor as pt

    values = np.asarray(returns, dtype=float)
    if values.ndim != 1 or values.size < 30:
        raise ValueError(
            f"need at least 30 returns in a one-dimensional array, got shape {values.shape}"
        )
    if not np.isfinite(h0) or h0 <= 0.0:
        raise ValueError(f"h0 must be finite and positive, got {h0!r}")

    with pm.Model() as model:
        mu = pm.Normal("mu", 0.0, PRIOR_MU_SD)
        omega = pm.HalfNormal("omega", PRIOR_OMEGA_SD)
        alpha = pm.Beta("alpha", *PRIOR_ALPHA)
        delta = pm.Beta("delta", *prior_delta)
        # alpha + beta < 1 by construction, so stationarity is never rejected.
        beta = pm.Deterministic("beta", (1.0 - alpha) * delta)
        nu = pm.Truncated(
            "nu", pm.Exponential.dist(scale=PRIOR_NU_MEAN), lower=PRIOR_NU_LOWER
        )

        # h[t] = omega + alpha * (r[t-1] - mu)^2 + beta * h[t-1], seeded at h[0] = h0.
        # The scan carries h[1:]; h[0] is prepended so the observed vector is the whole
        # window, exactly as in the NumPy filter.
        squared_resid = (pt.as_tensor_variable(values[:-1]) - mu) ** 2
        h_rest = pytensor.scan(
            fn=lambda eps2, h_prev, w, a, b: w + a * eps2 + b * h_prev,
            sequences=[squared_resid],
            outputs_info=[pt.as_tensor_variable(float(h0))],
            non_sequences=[omega, alpha, beta],
            strict=True,
            return_updates=False,
        )
        h = pt.concatenate([pt.as_tensor_variable([float(h0)]), h_rest])

        # h is the conditional variance of the return; the Student-t scale is smaller by
        # sqrt((nu-2)/nu). Dropping that factor is trap 3 in the handoff -- it inflates
        # every interval by a few percent, in the direction that flatters this model.
        sigma = pt.sqrt(h) * pt.sqrt((nu - 2.0) / nu)
        pm.StudentT("obs", nu=nu, mu=mu, sigma=sigma, observed=values)

    return model


def sample_garch_posterior(
    returns: np.ndarray,
    *,
    h0: float | None = None,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.9,
    thin: int = 2,
    seed: int = 0,
    cores: int | None = None,
    prior_delta: tuple[float, float] = PRIOR_DELTA,
) -> BayesianFit:
    """Sample the GARCH(1,1)-t posterior with PyMC/NUTS, seeded for reproducibility.

    The defaults are the settings frozen at D17: 4 chains x (1,000 tune + 1,000 draw) at
    ``target_accept = 0.9``, thinned by 2 into the 2,000 draws D18 carries forward. Four
    chains rather than two because split-R-hat is the convergence criterion and is
    unreliable below four; 2,000 retained draws because ``ess_tail`` -- what the 99%
    predictive quantiles are built from -- is the binding resolution constraint.

    Poor mixing is **reported**, never worked around by quietly lengthening the chain
    until the diagnostics look acceptable. A fit that fails ``BAYES_CONVERGENCE`` comes
    back with ``converged=False`` and a message naming what failed; the backtest then
    produces no forecasts for its block (D19), and no failed refit is re-run under a
    different seed in the hope of a better verdict (D16's multi-start rule, restated).

    Parameters
    ----------
    returns:
        Estimation-window returns, on the **raw** return scale. Scaled to percent
        internally (D14) for sampler geometry and converted back before anything is
        returned, so the percent convention stays inside this function.
    h0:
        Seed for the variance recursion, raw scale. Defaults to
        ``backcast_initial_variance(returns)`` -- correct precisely because ``returns``
        is the estimation window and nothing else.
    thin:
        Keep every ``thin``-th draw, deterministically rather than as a random
        subsample. NUTS draws are near-independent, so thinning discards information
        rather than redundancy; the only reason to do it is the per-day cost of the
        mixture quantile solve, which is linear in draw count (D18).
    cores:
        Chains per process pool. ``None`` leaves PyMC's default, which runs them in
        parallel as D17 specifies. **Any caller that samples with more than one core
        must guard its entry point with ``if __name__ == "__main__":``** -- on Windows
        multiprocessing spawns rather than forks, so an unguarded module re-imports
        itself in every worker and forks until the machine gives out, silently
        (problems-and-solutions 37). ``run_all.py`` is guarded; tests pass ``cores=1``.
    prior_delta:
        The ``Beta`` prior on ``delta``, defaulting to the frozen D4 value. Overridden
        only by the Stage 6 prior-sensitivity check, which is a *different run* writing
        to its own directory -- never a way of changing what the headline results were
        computed under. See ``_build_pymc_model``.
    """
    import arviz as az
    import pymc as pm

    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 30:
        raise ValueError(
            f"need at least 30 finite returns to fit GARCH(1,1), got {values.size}"
        )
    if thin < 1:
        raise ValueError(f"thin must be at least 1, got {thin!r}")

    h0_raw = backcast_initial_variance(values) if h0 is None else float(h0)
    scaled = values * FIT_SCALE
    h0_scaled = h0_raw * FIT_SCALE**2

    model = _build_pymc_model(scaled, h0_scaled, prior_delta=prior_delta)
    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            target_accept=target_accept,
            random_seed=seed,
            progressbar=False,
            # Our own verdict is BAYES_CONVERGENCE, applied below; PyMC's would only
            # duplicate it as a warning nobody acts on.
            compute_convergence_checks=False,
        )

    r_hat_all = az.rhat(idata, var_names=list(_DIAGNOSTIC_VARS))
    ess_bulk_all = az.ess(idata, var_names=list(_DIAGNOSTIC_VARS), method="bulk")
    ess_tail_all = az.ess(idata, var_names=list(_DIAGNOSTIC_VARS), method="tail")
    n_divergences = int(idata.sample_stats["diverging"].to_numpy().sum())

    def _by_param(diagnostic) -> np.ndarray:
        return np.array([float(diagnostic[name]) for name in PARAM_NAMES], dtype=float)

    r_hat = _by_param(r_hat_all)
    ess_bulk = _by_param(ess_bulk_all)
    ess_tail = _by_param(ess_tail_all)

    # The verdict spans every diagnostic variable, including sampled ``delta``, which
    # D19 requires and which the five stored slots do not cover.
    failures: list[str] = []
    worst_r_hat = max(float(r_hat_all[name]) for name in _DIAGNOSTIC_VARS)
    worst_ess_tail = min(float(ess_tail_all[name]) for name in _DIAGNOSTIC_VARS)
    for name in _DIAGNOSTIC_VARS:
        value = float(r_hat_all[name])
        if not np.isfinite(value) or value > BAYES_CONVERGENCE["max_r_hat"]:
            failures.append(f"R-hat {value:.4f} on {name}")
        tail = float(ess_tail_all[name])
        if not np.isfinite(tail) or tail < BAYES_CONVERGENCE["min_ess_tail"]:
            failures.append(f"ess_tail {tail:.0f} on {name}")
    if n_divergences > BAYES_CONVERGENCE["max_divergences"]:
        failures.append(f"{n_divergences} divergent transitions")

    converged = not failures
    message = (
        f"max R-hat {worst_r_hat:.4f}, min ess_tail {worst_ess_tail:.0f}, "
        f"{n_divergences} divergences"
        if converged
        else "not converged: " + "; ".join(failures)
    )

    # Chain-major flatten, then keep every ``thin``-th draw. Deterministic given the
    # seed, which is what makes a refit reproducible bit for bit.
    posterior = idata.posterior
    stacked = np.column_stack(
        [posterior[name].to_numpy().reshape(-1) for name in PARAM_NAMES]
    )
    kept = stacked[::thin]

    # ``log_prob`` is the unnormalised target density on the **percent** scale, where
    # the priors are defined -- computed from this module's own ``log_posterior`` rather
    # than read out of the sampler, so it is an independent statement of what was
    # sampled rather than a restatement of it. Comparable across draws within a fit; not
    # across fits, whose windows differ.
    log_prob = np.array(
        [log_posterior(theta, scaled, h0_scaled) for theta in kept], dtype=float
    )

    # Back to the raw return scale. Nothing outside this function sees percent.
    raw = kept.copy()
    raw[:, 0] /= FIT_SCALE
    raw[:, 1] /= FIT_SCALE**2

    return BayesianFit(
        draws=raw,
        log_prob=log_prob,
        r_hat=r_hat,
        ess_bulk=ess_bulk,
        ess_tail=ess_tail,
        n_divergences=n_divergences,
        converged=converged,
        message=message,
        n_obs=int(values.size),
        seed=int(seed),
    )


# --- Predictive distributions -----------------------------------------------------


def plugin_predictive(
    params: GarchParams,
    h_next: float,
) -> PredictiveDistribution:
    """Frequentist plug-in predictive for tomorrow's **return**.

    Conditions on the MLE as if it were the truth, so the predictive carries
    **innovation uncertainty only**. Closed form: a location-scale standardised
    Student-t, or a normal when ``params.nu`` is NaN (the Stage 6 ablation).

    Returns the distribution object rather than a ``(mean, quantiles)`` pair as this
    module's stub once specified. The harness needs ``cdf`` for the PIT as well as
    ``quantile`` for the interval bounds, and taking both from one object is what stops
    the two from being computed under different conventions -- the same reason
    ``NormalPredictive`` exists rather than a pair of free functions. Signature
    deviation recorded in research_log.md.
    """
    if not np.isfinite(h_next) or h_next <= 0.0:
        raise ValueError(f"h_next must be finite and positive, got {h_next!r}")
    if np.isnan(params.nu):
        return NormalPredictive(mean=params.mu, variance=float(h_next))
    return StudentTPredictive(mean=params.mu, variance=float(h_next), nu=params.nu)


def mixture_predictive(
    draws: np.ndarray, h_next_by_draw: np.ndarray
) -> StudentTMixturePredictive:
    """Assemble the posterior predictive from posterior draws and their own variances.

    **One ``h_next`` per draw, and this function will not accept anything else.** Each
    posterior draw implies its own filtered variance path and therefore its own
    ``h_next``; passing a single shared value -- one computed at the posterior mean, say
    -- collapses the mixture to something very close to the plug-in predictive. Nothing
    would fail. The Bayesian and frequentist intervals would simply agree, and that
    agreement would be reported as the project's finding. It is the single most
    dangerous bug available in this codebase, which is why the shape requirement is
    enforced here rather than documented and hoped for, and why
    ``test_posterior_predictive_refuses_one_shared_h_next`` exists.

    Both inputs are on the **raw return scale**, as ``BayesianFit.draws`` is.
    """
    values = np.asarray(draws, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(PARAM_NAMES):
        raise ValueError(
            f"draws must have shape (n_draws, {len(PARAM_NAMES)}), got {values.shape}"
        )

    h_next = np.asarray(h_next_by_draw, dtype=float)
    if h_next.ndim != 1 or h_next.size != values.shape[0]:
        raise ValueError(
            "h_next_by_draw must hold one variance per posterior draw, so shape "
            f"({values.shape[0]},); got {h_next.shape}. Each draw implies its own "
            "filtered variance path -- a single shared h_next would discard most of "
            "the parameter uncertainty and collapse this predictive towards the "
            "frequentist plug-in."
        )

    return StudentTMixturePredictive(
        means=values[:, 0], variances=h_next, nus=values[:, 4]
    )


def posterior_predictive(
    draws: np.ndarray,
    h_next_by_draw: np.ndarray,
    quantile_levels: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Bayesian posterior predictive for tomorrow's **return**.

    A mixture of scaled Student-t distributions over posterior draws, carrying **both**
    innovation and parameter uncertainty. Note that ``h_next`` differs across draws,
    because each parameter vector implies its own filtered variance path -- so the
    caller must pass one ``h_next`` per draw, not a single shared value. See
    ``mixture_predictive``, which enforces that and explains what getting it wrong would
    cost.

    No closed form; quantiles are obtained numerically from the mixture CDF.

    A thin wrapper over ``mixture_predictive``. The backtest uses the distribution
    object directly, because it needs ``cdf`` for the PIT as well as ``quantile`` for
    the interval bounds and taking both from one object is what stops the two from being
    computed under different conventions -- the same reason ``plugin_predictive``
    returns an object. This function is retained as the stated contract of the stage and
    as the form the tests read most naturally.

    Returns
    -------
    mean:
        Predictive mean of the return.
    quantiles:
        Return quantiles at ``quantile_levels``, same shape.
    """
    predictive = mixture_predictive(draws, h_next_by_draw)
    return predictive.mean, predictive.quantile(np.asarray(quantile_levels, dtype=float))


# --- Baselines --------------------------------------------------------------------


def forecast_yesterday_volatility(realised_var: pd.Series) -> pd.Series:
    """Baseline 1: tomorrow's variance forecast is today's realised variance.

    A pure one-step lag. Included as the floor any real model must clear.

    ``realised_var`` must **already be on the close-to-close return-variance scale** --
    that is, ``data.scale_proxy(frame["parkinson_var"])``, not the raw Parkinson column.
    Passing raw Parkinson would understate the variance by roughly a third and produce
    intervals about 23% too narrow, so the resulting VaR breaches would be a units error
    wearing the costume of a calibration failure. The harness does the conversion once,
    centrally, and a test asserts the scaled series is what arrives here.

    The output is indexed so that row ``t`` holds the forecast **for** day ``t``, built
    from the observation at ``t-1``. The first row is therefore NaN.
    """
    if not isinstance(realised_var.index, pd.DatetimeIndex):
        raise TypeError("realised_var must have a DatetimeIndex")
    return realised_var.shift(1).rename("variance")


def forecast_ewma(
    returns: pd.Series,
    *,
    lam: float = EWMA_LAMBDA,
    initial_var: float | None = None,
) -> pd.Series:
    """Baseline 2: EWMA / RiskMetrics recursion.

    ``h_{t+1} = lam * h_t + (1 - lam) * r_t^2``, with ``lam`` fixed at the RiskMetrics
    value rather than estimated, so the baseline stays a genuine baseline. Because the
    recursion is driven by squared **returns**, this baseline is already on the
    close-to-close scale and needs no rescaling -- unlike baseline 1.

    ``initial_var`` seeds ``h`` at the first observation and must be computed from the
    training window only; when omitted it is the sample variance of the ``returns``
    passed in, so the caller is responsible for passing training-window data. The
    harness passes the warm-up block, and a test asserts the seed cannot move when
    out-of-sample returns change.

    As with baseline 1, row ``t`` of the output is the forecast **for** day ``t``, using
    returns through ``t-1`` only. The first row is NaN.
    """
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("returns must have a DatetimeIndex")
    if not 0.0 < lam < 1.0:
        raise ValueError(f"lam must lie strictly in (0, 1), got {lam!r}")

    values = returns.to_numpy(dtype=float)
    n = values.size
    if initial_var is None:
        finite = values[np.isfinite(values)]
        if finite.size < 2:
            raise ValueError("need at least two finite returns to seed the EWMA")
        initial_var = float(np.var(finite, ddof=1))
    if not np.isfinite(initial_var) or initial_var <= 0.0:
        raise ValueError(f"initial_var must be finite and positive, got {initial_var!r}")

    # ``h[t]`` is the forecast FOR day t, so it is written before day t's return is
    # read. Advancing h and then assigning it to the same index position would make the
    # forecast for day t depend on the return of day t -- the single most consequential
    # off-by-one available in this module, and the reason the ordering below is explicit
    # rather than a vectorised ewm() call.
    out = np.full(n, np.nan, dtype=float)
    h = float(initial_var)
    for t in range(n):
        if t > 0:
            r_prev = values[t - 1]
            if np.isfinite(r_prev):
                h = lam * h + (1.0 - lam) * r_prev**2
            out[t] = h
    return pd.Series(out, index=returns.index, name="variance")
