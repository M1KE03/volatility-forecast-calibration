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

Status: the GARCH half is still stubbed. The predictive-distribution interface and both
baselines are implemented (Stage 1). Priors are set at Stage 3 and must be frozen in
research_log.md before any out-of-sample result is computed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from scipy import stats

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
        raise NotImplementedError("GarchParams.to_array")

    @classmethod
    def from_array(cls, theta: np.ndarray) -> GarchParams:
        """Build from a flat vector in ``PARAM_NAMES`` order."""
        raise NotImplementedError("GarchParams.from_array")

    def is_valid(self) -> bool:
        """True if the parameters satisfy positivity, stationarity, and nu > 4."""
        raise NotImplementedError("GarchParams.is_valid")


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
    """Posterior draws from the GARCH(1,1)-t model.

    Attributes
    ----------
    draws:
        Array of shape ``(n_draws, 5)`` in ``PARAM_NAMES`` order, post burn-in and
        post thinning, with walkers flattened.
    log_prob:
        Log posterior density at each retained draw, shape ``(n_draws,)``.
    acceptance_fraction:
        Mean acceptance fraction across walkers. Diagnostic; must be reported.
    autocorr_time:
        Estimated integrated autocorrelation time per parameter, shape ``(5,)``. NaN if
        the chain was too short to estimate it — which is itself a finding and must be
        reported rather than hidden.
    n_effective:
        Crude effective sample size per parameter, shape ``(5,)``.
    seed:
        The seed used, so the chain can be reproduced exactly.
    """

    draws: np.ndarray
    log_prob: np.ndarray
    acceptance_fraction: float
    autocorr_time: np.ndarray
    n_effective: np.ndarray
    seed: int


# --- The shared likelihood --------------------------------------------------------


def garch11_filter(theta: np.ndarray, returns: np.ndarray, h0: float) -> np.ndarray:
    """Run the conditional variance recursion; return ``h`` of the same length.

    ``h[t]`` is the conditional variance of ``returns[t]`` given information through
    ``t-1``. Sequential by construction and therefore the pipeline's hot loop; this is
    the function that is numba-compiled (decision B1).

    ``h0`` seeds ``h[0]`` and must be backcast from the **training** returns only
    (decision D5), never from the full sample.
    """
    raise NotImplementedError("garch11_filter")


def garch11_t_loglik(theta: np.ndarray, returns: np.ndarray, h0: float) -> float:
    """Log-likelihood of the GARCH(1,1)-t model. **Shared by both models 3 and 4.**

    Returns ``-inf`` for parameters outside the admissible region rather than raising,
    so the same function can be handed to an optimiser and to an MCMC sampler.

    Uses the *standardised* Student-t density (unit variance), so that ``h_t`` from
    ``garch11_filter`` is the conditional variance directly.
    """
    raise NotImplementedError("garch11_t_loglik")


def log_prior(theta: np.ndarray) -> float:
    """Log prior over (mu, omega, alpha, beta, nu). Returns ``-inf`` outside support.

    **The priors are not yet chosen.** They must be specified and frozen in
    research_log.md before any out-of-sample result is computed, because loose priors
    widen the Bayesian predictive intervals and could manufacture the project's headline
    finding — that integrating over parameter uncertainty improves interval coverage.
    Choosing them after seeing a coverage table would make that finding unfalsifiable.

    Under decision B1-R the sampler is PyMC/NUTS, which takes its priors as distribution
    objects in the model graph rather than through this function. This entry point is
    retained for the route-B fallback (emcee), where the target density must be assembled
    by hand. Whichever path is used, the priors themselves are the same and are recorded
    once. See ``docs/handoff.md`` for the candidate specification carried over from the
    Stage 3 feasibility probe.
    """
    raise NotImplementedError("log_prior - priors not yet specified; see docs/handoff.md")


def log_posterior(theta: np.ndarray, returns: np.ndarray, h0: float) -> float:
    """``log_prior + garch11_t_loglik``. The target density for the sampler."""
    raise NotImplementedError("log_posterior")


def backcast_initial_variance(returns: np.ndarray) -> float:
    """Seed value for ``h[0]``, computed from training returns only.

    Must be called on the estimation window alone. Passing the full sample here would
    leak future information into every fit.
    """
    raise NotImplementedError("backcast_initial_variance")


# --- Estimation -------------------------------------------------------------------


def fit_garch_mle(
    returns: np.ndarray,
    *,
    h0: float | None = None,
    start_params: np.ndarray | None = None,
) -> FrequentistFit:
    """Maximise ``garch11_t_loglik`` via ``scipy.optimize``.

    Convergence failures are returned in the ``FrequentistFit``, never swallowed and
    never silently retried into a spurious success.
    """
    raise NotImplementedError("fit_garch_mle")


def sample_garch_posterior(
    returns: np.ndarray,
    *,
    h0: float | None = None,
    n_walkers: int = 32,
    n_steps: int = 3000,
    n_burn: int = 1000,
    thin: int = 5,
    seed: int = 0,
) -> BayesianFit:
    """Sample the posterior with ``emcee``, seeded for exact reproducibility.

    Sampler settings are provisional and will be tuned against convergence diagnostics
    on the initial training window before the backtest is run. Poor mixing must be
    reported, not worked around by quietly increasing the chain length until the
    diagnostics look acceptable.
    """
    raise NotImplementedError("sample_garch_posterior")


# --- Predictive distributions -----------------------------------------------------


def plugin_predictive(
    params: GarchParams,
    h_next: float,
    quantile_levels: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Frequentist plug-in predictive for tomorrow's **return**.

    Conditions on the MLE as if it were the truth, so the predictive carries
    **innovation uncertainty only**. Closed form: a location-scale standardised
    Student-t.

    Returns
    -------
    mean:
        Predictive mean of the return (i.e. ``mu``).
    quantiles:
        Return quantiles at ``quantile_levels``, same shape.
    """
    raise NotImplementedError("plugin_predictive")


def posterior_predictive(
    draws: np.ndarray,
    h_next_by_draw: np.ndarray,
    quantile_levels: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Bayesian posterior predictive for tomorrow's **return**.

    A mixture of scaled Student-t distributions over posterior draws, carrying
    **both** innovation and parameter uncertainty. Note that ``h_next`` differs across
    draws, because each parameter vector implies its own filtered variance path — so
    the caller must pass one ``h_next`` per draw, not a single shared value. Getting
    this wrong would discard most of the parameter uncertainty and quietly collapse the
    Bayesian model towards the frequentist one.

    No closed form; quantiles are obtained numerically from the mixture CDF.

    Returns
    -------
    mean:
        Predictive mean of the return.
    quantiles:
        Return quantiles at ``quantile_levels``, same shape.
    """
    raise NotImplementedError("posterior_predictive")


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
