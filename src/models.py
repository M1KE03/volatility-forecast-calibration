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

Status: the frequentist half is complete (Stage 2) -- the shared likelihood, the MLE, and
the plug-in predictive, alongside the interface and both baselines from Stage 1. The
Bayesian half is stubbed: ``log_prior``, ``log_posterior``, ``sample_garch_posterior``
and ``posterior_predictive`` arrive at Stage 3, and the priors must be frozen in
research_log.md before any out-of-sample result is computed.
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
