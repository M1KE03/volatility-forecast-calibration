"""Tests for ``src.models``, in the order the module was built.

Stage 1 covers the predictive interface every model passes through and the two
baselines; Stage 2 the shared GARCH likelihood, the MLE and the plug-in predictive;
Stage 3 the frozen priors, the PyMC/NUTS sampler and the mixture posterior predictive.

Two of these tests are load-bearing beyond their own section. The PyMC graph is checked
against the NumPy log posterior, because "models 3 and 4 share one likelihood" is
otherwise a claim about two implementations that nothing verifies. And the posterior
predictive is checked from both sides -- strictly wider than the plug-in when the
posterior is dispersed, exactly equal to it when the posterior is collapsed -- because a
mixture built on one shared ``h_next`` would agree with the frequentist model and that
agreement would be reported as this project's finding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats

from src import models as M


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2014-01-02", periods=n)


# --- The predictive interface ------------------------------------------------------


def test_normal_predictive_quantiles_match_scipy() -> None:
    dist = M.NormalPredictive(mean=0.0005, variance=1.2e-4)
    probs = np.array([0.005, 0.01, 0.05, 0.5, 0.95, 0.99, 0.995])
    expected = stats.norm.ppf(probs, loc=0.0005, scale=np.sqrt(1.2e-4))
    np.testing.assert_allclose(dist.quantile(probs), expected, rtol=1e-12)


def test_normal_predictive_cdf_inverts_its_own_quantiles() -> None:
    """``cdf`` and ``quantile`` must be consistent: the PIT depends on it.

    The evaluation layer reads interval bounds from one and PIT values from the other.
    If they disagreed, coverage and PIT would tell contradictory stories about the same
    forecast, and there would be no way to see which was wrong.
    """
    dist = M.NormalPredictive(mean=-0.001, variance=4e-4)
    for p in (0.01, 0.1, 0.5, 0.9, 0.99):
        assert dist.cdf(float(dist.quantile(np.array([p]))[0])) == pytest.approx(p)


def test_normal_predictive_rejects_degenerate_variance() -> None:
    """A zero or negative variance must raise, not produce infinite bounds."""
    for bad in (0.0, -1e-6, np.nan, np.inf):
        with pytest.raises(ValueError, match="finite and positive"):
            M.NormalPredictive(mean=0.0, variance=bad)


def test_normal_predictive_sd_is_the_root_of_the_variance() -> None:
    assert M.NormalPredictive(mean=0.0, variance=9e-4).sd == pytest.approx(0.03)


def test_forecast_carries_a_working_distribution() -> None:
    f = M.Forecast(
        variance=1e-4, mean=0.0, distribution=M.NormalPredictive(mean=0.0, variance=1e-4)
    )
    assert f.distribution.cdf(0.0) == pytest.approx(0.5)


# --- Baseline 1: random walk in volatility -----------------------------------------


def test_yesterday_volatility_is_exactly_a_one_step_lag() -> None:
    values = pd.Series([1.0, 2.0, 3.0, 4.0], index=_dates(4))
    out = M.forecast_yesterday_volatility(values)
    assert np.isnan(out.iloc[0])
    np.testing.assert_allclose(out.iloc[1:].to_numpy(), [1.0, 2.0, 3.0])


def test_yesterday_volatility_never_uses_its_own_days_value() -> None:
    """Changing day ``t`` must move the forecast for ``t+1`` and leave ``t`` alone."""
    values = pd.Series([1.0, 2.0, 3.0, 4.0], index=_dates(4))
    baseline = M.forecast_yesterday_volatility(values)

    bumped = values.copy()
    bumped.iloc[2] = 99.0
    moved = M.forecast_yesterday_volatility(bumped)

    assert moved.iloc[2] == baseline.iloc[2]
    assert moved.iloc[3] != baseline.iloc[3]


def test_yesterday_volatility_requires_a_datetime_index() -> None:
    with pytest.raises(TypeError, match="DatetimeIndex"):
        M.forecast_yesterday_volatility(pd.Series([1.0, 2.0]))


# --- Baseline 2: EWMA / RiskMetrics ------------------------------------------------


def test_ewma_matches_a_hand_computed_recursion() -> None:
    returns = pd.Series([0.01, -0.02, 0.015, -0.005, 0.02], index=_dates(5))
    out = M.forecast_ewma(returns, lam=0.94, initial_var=1e-4)

    expected = [np.nan]
    h = 1e-4
    for t in range(1, 5):
        h = 0.94 * h + 0.06 * returns.iloc[t - 1] ** 2
        expected.append(h)

    assert np.isnan(out.iloc[0])
    np.testing.assert_allclose(out.iloc[1:].to_numpy(), expected[1:], rtol=1e-12)


def test_ewma_forecast_for_day_t_ignores_day_t() -> None:
    """The critical off-by-one: ``h[t]`` is written before day ``t``'s return is read."""
    returns = pd.Series([0.01, -0.02, 0.015, -0.005, 0.02], index=_dates(5))
    baseline = M.forecast_ewma(returns, initial_var=1e-4)

    bumped = returns.copy()
    bumped.iloc[2] = 0.5
    moved = M.forecast_ewma(bumped, initial_var=1e-4)

    np.testing.assert_allclose(baseline.iloc[:3].to_numpy(), moved.iloc[:3].to_numpy())
    assert moved.iloc[3] != baseline.iloc[3]


def test_ewma_lambda_is_the_riskmetrics_value() -> None:
    """Fixed, never estimated -- otherwise the baseline stops being a baseline."""
    assert M.EWMA_LAMBDA == 0.94


def test_ewma_rejects_invalid_lambda() -> None:
    returns = pd.Series([0.01, -0.02, 0.015], index=_dates(3))
    for bad in (0.0, 1.0, -0.5, 1.5):
        with pytest.raises(ValueError, match="strictly in"):
            M.forecast_ewma(returns, lam=bad, initial_var=1e-4)


def test_ewma_rejects_degenerate_seed() -> None:
    returns = pd.Series([0.01, -0.02, 0.015], index=_dates(3))
    for bad in (0.0, -1e-8):
        with pytest.raises(ValueError, match="finite and positive"):
            M.forecast_ewma(returns, initial_var=bad)


def test_ewma_seed_defaults_to_the_variance_of_what_it_is_given() -> None:
    """The caller owns the seed window.

    Asserted by equivalence rather than by reading ``out.iloc[1]``, which is already the
    seed advanced one step by ``r[0]`` and is not the seed itself.
    """
    returns = pd.Series([0.01, -0.02, 0.015, -0.005], index=_dates(4))
    expected_seed = float(np.var(returns.to_numpy(), ddof=1))
    pd.testing.assert_series_equal(
        M.forecast_ewma(returns),
        M.forecast_ewma(returns, initial_var=expected_seed),
    )


def test_ewma_tolerates_a_missing_return_without_propagating_nan() -> None:
    """A NaN return must not poison every subsequent forecast.

    The analysis frame has no interior gaps, but a recursion that turns one missing
    observation into an all-NaN tail fails in a way that is easy to miss in aggregate
    statistics, so the behaviour is pinned deliberately.
    """
    returns = pd.Series([0.01, np.nan, 0.015, -0.005], index=_dates(4))
    out = M.forecast_ewma(returns, initial_var=1e-4)
    assert out.iloc[1:].notna().all()


def test_ewma_requires_a_datetime_index() -> None:
    with pytest.raises(TypeError, match="DatetimeIndex"):
        M.forecast_ewma(pd.Series([0.01, -0.02]))


def test_ewma_reacts_to_a_volatility_shock_with_the_right_persistence() -> None:
    """A single large return must raise the forecast and then decay geometrically.

    This is the behavioural signature the Stage 1 acceptance plot checks by eye; pinning
    it numerically means a regression is caught by the suite and not only by looking.
    """
    n = 40
    returns = pd.Series(np.zeros(n), index=_dates(n))
    returns.iloc[10] = 0.10
    out = M.forecast_ewma(returns, lam=0.94, initial_var=1e-6)

    assert out.iloc[11] > out.iloc[10], "forecast must rise the day after the shock"
    decay = out.iloc[13] / out.iloc[12]
    assert decay == pytest.approx(0.94, rel=1e-6), "decay must be lambda once shocks stop"


# ===================================================================================
# Stage 2: the GARCH(1,1) likelihood, the MLE, and the plug-in predictive
# ===================================================================================


def _garch_theta(mu=0.0005, omega=2.0e-6, alpha=0.09, beta=0.88, nu=7.0) -> np.ndarray:
    return np.array([mu, omega, alpha, beta, nu], dtype=float)


def _simulate_garch_t(
    n: int, theta: np.ndarray, *, seed: int, burn: int = 1000
) -> np.ndarray:
    """Simulate from the model this module claims to fit.

    Written out here rather than imported from ``src`` on purpose: a recovery test that
    simulates with the same code it estimates with would pass even if both shared a
    misunderstanding of the model.
    """
    mu, omega, alpha, beta, nu = (float(v) for v in theta)
    rng = np.random.default_rng(seed)
    total = n + burn
    z = rng.standard_t(nu, size=total) * np.sqrt((nu - 2.0) / nu)
    out = np.empty(total)
    h = omega / (1.0 - alpha - beta)
    for t in range(total):
        out[t] = mu + np.sqrt(h) * z[t]
        h = omega + alpha * (out[t] - mu) ** 2 + beta * h
    return out[burn:]


# --- Parameter handling ------------------------------------------------------------


def test_param_array_round_trip_preserves_order() -> None:
    params = M.GarchParams(mu=1.0, omega=2.0, alpha=3.0, beta=4.0, nu=5.0)
    np.testing.assert_allclose(params.to_array(), [1.0, 2.0, 3.0, 4.0, 5.0])
    assert M.GarchParams.from_array(params.to_array()) == params
    assert M.PARAM_NAMES == ("mu", "omega", "alpha", "beta", "nu")


def test_from_array_rejects_the_wrong_length() -> None:
    with pytest.raises(ValueError, match="shape"):
        M.GarchParams.from_array(np.array([1.0, 2.0, 3.0]))


def test_is_valid_accepts_an_admissible_vector() -> None:
    assert M.GarchParams.from_array(_garch_theta()).is_valid()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("omega", 0.0),  # positivity is strict
        ("omega", -1e-8),
        ("alpha", -1e-8),
        ("beta", -1e-8),
        ("beta", 0.95),  # alpha + beta = 1.04 >= 1
        ("nu", 4.0),  # nu > 4 is strict, so the kurtosis stays finite
        ("nu", 2.5),
        ("mu", np.inf),
    ],
)
def test_is_valid_rejects_each_constraint_violation(field: str, value: float) -> None:
    """Each constraint is tested on its own, so one of them cannot go missing unnoticed.

    A single "rejects a bad vector" test would pass while four of the five constraints
    were unimplemented.
    """
    params = M.GarchParams.from_array(_garch_theta())
    broken = M.GarchParams(**{**params.__dict__, field: value})
    assert not broken.is_valid()


def test_is_valid_treats_nan_nu_as_the_normal_variant() -> None:
    """The Gaussian model has no degrees-of-freedom parameter to constrain."""
    assert M.GarchParams.from_array(_garch_theta(nu=float("nan"))).is_valid()


# --- The variance recursion --------------------------------------------------------


def test_filter_matches_a_hand_computed_recursion() -> None:
    theta = _garch_theta(mu=0.001, omega=2e-6, alpha=0.1, beta=0.85)
    returns = np.array([0.01, -0.02, 0.015, -0.005, 0.02])
    h0 = 1e-4

    expected = [h0]
    for t in range(1, 5):
        expected.append(
            2e-6 + 0.1 * (returns[t - 1] - 0.001) ** 2 + 0.85 * expected[t - 1]
        )

    np.testing.assert_allclose(M.garch11_filter(theta, returns, h0), expected, rtol=1e-14)


def test_filter_squares_the_residual_not_the_raw_return() -> None:
    """``(r - mu)**2``, not ``r**2``.

    At mu's realistic size the two differ by a fraction of a percent, so this is a slip
    no plot and no summary statistic would reveal. It is constructed here with a large
    mu so the difference is unmistakable.
    """
    returns = np.array([0.01, -0.02, 0.015, -0.005])
    with_mu = M.garch11_filter(_garch_theta(mu=0.05), returns, 1e-4)
    raw_return_version = M.garch11_filter(_garch_theta(mu=0.0), returns, 1e-4)
    assert not np.allclose(with_mu[1:], raw_return_version[1:])


def test_filter_seeds_the_first_element_with_h0() -> None:
    h = M.garch11_filter(_garch_theta(), np.array([0.01, -0.02, 0.03]), 1.25e-4)
    assert h[0] == 1.25e-4


def test_filter_rejects_a_degenerate_seed() -> None:
    for bad in (0.0, -1e-8, np.nan, np.inf):
        with pytest.raises(ValueError, match="finite and positive"):
            M.garch11_filter(_garch_theta(), np.array([0.01, -0.02]), bad)


# --- The shared likelihood ---------------------------------------------------------


def test_t_loglik_matches_an_independent_scipy_implementation() -> None:
    """**The test that catches a missing ``-0.5*log(h_t)``.**

    ``scipy.stats.t.logpdf`` with the location and scale worked out separately is a
    genuinely independent route to the same number: it shares no line of code with the
    implementation and does not restate its formula. Without the Jacobian term the
    likelihood still optimises and still produces plausible-looking estimates, so no
    other test in this file would fail.
    """
    theta = _garch_theta(mu=0.001, omega=2e-6, alpha=0.1, beta=0.85, nu=7.0)
    returns = np.array([0.01, -0.02, 0.015, -0.005, 0.02])
    h0 = 1e-4

    h = M.garch11_filter(theta, returns, h0)
    scale = np.sqrt(h) * np.sqrt((7.0 - 2.0) / 7.0)
    expected = stats.t.logpdf(returns, 7.0, loc=0.001, scale=scale).sum()

    assert M.garch11_t_loglik(theta, returns, h0) == pytest.approx(expected, rel=1e-12)


def test_dropping_the_jacobian_would_change_the_answer_materially() -> None:
    """Proves the test above is load-bearing rather than decorative.

    Same pattern as the EDA module's negative controls: an equality test is evidence
    only if the quantity it pins would actually move when the thing it guards against
    happened.
    """
    theta = _garch_theta()
    returns = _simulate_garch_t(500, theta, seed=1)
    h0 = 1e-4

    correct = M.garch11_t_loglik(theta, returns, h0)
    h = M.garch11_filter(theta, returns, h0)
    without_jacobian = correct + 0.5 * float(np.sum(np.log(h)))

    assert abs(without_jacobian - correct) > 100.0


def test_normal_loglik_matches_an_independent_scipy_implementation() -> None:
    theta = _garch_theta(mu=0.001, nu=float("nan"))
    returns = np.array([0.01, -0.02, 0.015, -0.005, 0.02])
    h0 = 1e-4

    h = M.garch11_filter(theta, returns, h0)
    expected = stats.norm.logpdf(returns, loc=0.001, scale=np.sqrt(h)).sum()

    assert M.garch11_normal_loglik(theta, returns, h0) == pytest.approx(
        expected, rel=1e-12
    )


def test_normal_loglik_ignores_nu_entirely() -> None:
    """``theta[4]`` must not reach the Gaussian likelihood by any route."""
    returns = np.array([0.01, -0.02, 0.015, -0.005, 0.02])
    values = [
        M.garch11_normal_loglik(_garch_theta(nu=nu), returns, 1e-4)
        for nu in (4.5, 30.0, -7.0, np.nan)
    ]
    assert len(set(values)) == 1


def test_t_loglik_approaches_the_gaussian_as_nu_grows() -> None:
    """The standardisation constant is right only if this limit holds."""
    returns = _simulate_garch_t(400, _garch_theta(), seed=2)
    h0 = 1e-4
    gaussian = M.garch11_normal_loglik(_garch_theta(nu=np.nan), returns, h0)
    student = M.garch11_t_loglik(_garch_theta(nu=1e7), returns, h0)
    assert student == pytest.approx(gaussian, abs=1e-3)


@pytest.mark.parametrize(
    "theta",
    [
        _garch_theta(omega=-1e-6),
        _garch_theta(alpha=-0.01),
        _garch_theta(beta=-0.01),
        _garch_theta(alpha=0.3, beta=0.8),
        _garch_theta(nu=3.0),
        _garch_theta(nu=4.0),
        _garch_theta(nu=np.nan),
    ],
)
def test_t_loglik_returns_minus_inf_outside_the_admissible_region(theta) -> None:
    """Returns, never raises: one function is handed to an optimiser and to a sampler.

    An exception here would abort a walk-forward run, or an MCMC chain, on a proposal
    that is merely rejected in the ordinary course of exploring the space.
    """
    assert M.garch11_t_loglik(theta, np.array([0.01, -0.02, 0.015]), 1e-4) == -np.inf


def test_loglik_is_scale_equivariant() -> None:
    """Fitting on percent returns is a change of variable, not a change of model.

    ``fit_garch_mle`` maximises the likelihood on ``r * FIT_SCALE`` and converts the
    estimates back. That is legitimate only if the two likelihoods differ by exactly the
    Jacobian of the rescaling -- one factor of ``FIT_SCALE`` per observation -- which is
    what this pins.
    """
    returns = _simulate_garch_t(300, _garch_theta(), seed=3)
    h0 = 1e-4
    theta = _garch_theta()
    scaled_theta = _garch_theta(
        mu=theta[0] * M.FIT_SCALE, omega=theta[1] * M.FIT_SCALE**2
    )

    raw = M.garch11_t_loglik(theta, returns, h0)
    scaled = M.garch11_t_loglik(scaled_theta, returns * M.FIT_SCALE, h0 * M.FIT_SCALE**2)
    assert scaled == pytest.approx(raw - returns.size * np.log(M.FIT_SCALE), rel=1e-12)


def test_backcast_is_the_sample_variance_of_what_it_is_given() -> None:
    returns = _simulate_garch_t(500, _garch_theta(), seed=4)
    assert M.backcast_initial_variance(returns) == pytest.approx(
        float(np.var(returns, ddof=1))
    )


def test_backcast_cannot_be_moved_by_data_outside_the_window() -> None:
    """Invariant 4. ``h0`` is a function of the estimation window and nothing else."""
    returns = _simulate_garch_t(500, _garch_theta(), seed=5)
    corrupted = returns.copy()
    corrupted[300:] = 5.0
    assert M.backcast_initial_variance(returns[:300]) == M.backcast_initial_variance(
        corrupted[:300]
    )


def test_backcast_rejects_a_window_it_cannot_estimate_from() -> None:
    with pytest.raises(ValueError, match="at least two"):
        M.backcast_initial_variance(np.array([0.01]))


# --- Estimation --------------------------------------------------------------------


@pytest.mark.slow
def test_mle_recovers_known_parameters_from_simulated_data() -> None:
    """The estimator end to end, against a truth this test knows.

    Tolerances are stated rather than tuned to whatever happened to pass. ``alpha`` and
    ``beta`` are individually the weakly identified pair -- they trade off against each
    other even at n = 5,000 -- while their **sum** is what the data pins down tightly,
    so the tolerance on the sum is tighter than on either part by an order of magnitude.
    Claiming the reverse would be pretending to a precision GARCH does not have.
    """
    truth = _garch_theta(mu=0.0005, omega=2.0e-6, alpha=0.09, beta=0.88, nu=7.0)
    returns = _simulate_garch_t(5000, truth, seed=20260823)

    fit = M.fit_garch_mle(returns)

    assert fit.converged, fit.message
    assert fit.n_obs == 5000
    assert fit.params.alpha == pytest.approx(0.09, abs=0.04)
    assert fit.params.beta == pytest.approx(0.88, abs=0.05)
    assert fit.params.alpha + fit.params.beta == pytest.approx(0.97, abs=0.02)
    assert fit.params.nu == pytest.approx(7.0, rel=0.4)
    assert fit.params.omega == pytest.approx(2.0e-6, rel=0.8)
    assert fit.params.mu == pytest.approx(0.0005, abs=0.0005)


@pytest.mark.slow
def test_mle_agrees_with_the_arch_package() -> None:
    """Independent validation against a mature implementation.

    This is the **only** use of ``arch`` in the project (decision B1) and it never
    produces a headline number: it exists so that "our likelihood is the standard
    GARCH(1,1)-t likelihood" is checked rather than asserted.

    The tolerance is stated and justified rather than pretended exact. ``arch`` seeds
    its recursion with an exponentially weighted backcast while this module uses the
    sample variance of the estimation window (decision D5), so the two optima genuinely
    differ by a small amount that shrinks with sample size. The persistence, which is
    what the forecasts actually depend on, agrees more tightly than either coefficient
    on its own.
    """
    arch = pytest.importorskip("arch")

    returns = _simulate_garch_t(3000, _garch_theta(), seed=7)
    ours = M.fit_garch_mle(returns)
    assert ours.converged, ours.message

    reference = (
        arch.arch_model(returns * 100.0, mean="Constant", vol="GARCH", p=1, q=1, dist="t")
        .fit(disp="off")
        .params
    )
    ref_alpha = float(reference["alpha[1]"])
    ref_beta = float(reference["beta[1]"])

    assert ours.params.alpha == pytest.approx(ref_alpha, abs=0.02)
    assert ours.params.beta == pytest.approx(ref_beta, abs=0.02)
    assert ours.params.alpha + ours.params.beta == pytest.approx(
        ref_alpha + ref_beta, abs=0.01
    )
    assert ours.params.nu == pytest.approx(float(reference["nu"]), rel=0.15)


def test_mle_reports_failure_rather_than_inventing_a_fit() -> None:
    """A series the optimiser cannot fit must be reported as one.

    The honesty contract from ``FrequentistFit``. What must not happen is an exception,
    which would abort a 102-refit run partway, or a ``converged=True`` on a parameter
    vector the optimiser never actually reached. The series here has a variance twenty
    orders of magnitude below a daily return's, so the likelihood surface is flat to
    within machine precision and L-BFGS-B terminates abnormally -- deterministically,
    given the seed.
    """
    hopeless = np.random.default_rng(0).normal(0.0, 1e-12, 200)

    fit = M.fit_garch_mle(hopeless)

    assert not fit.converged
    assert "ABNORMAL" in fit.message or "rejected" in fit.message
    assert isinstance(fit.message, str) and fit.message


def test_mle_refuses_a_window_with_no_variance_at_all() -> None:
    """Structurally unusable input fails loudly instead of returning a verdict.

    A constant series cannot seed the recursion: ``h0`` would be zero and every
    conditional variance after it undefined. That is a different situation from a fit
    that did not converge, and it is not the caller's to interpret -- so it raises,
    following the same principle as ``EstimationWindow.ROLLING`` in the backtest.
    """
    with pytest.raises(ValueError, match="finite and positive"):
        M.fit_garch_mle(np.zeros(200))


def test_mle_is_deterministic() -> None:
    """The multi-start grid is fixed, not random, so a refit reproduces exactly."""
    returns = _simulate_garch_t(800, _garch_theta(), seed=8)
    a = M.fit_garch_mle(returns)
    b = M.fit_garch_mle(returns)
    np.testing.assert_array_equal(a.params.to_array(), b.params.to_array())
    assert a.loglik == b.loglik


def test_mle_prefers_a_converged_start_over_a_better_objective_value() -> None:
    """Multi-start ranks converged optima first, and only then by objective value.

    Ranking purely on the objective lets one abnormally terminated start displace an
    ordinary success that landed a hair behind it, and the whole fit is then reported as
    failed on the strength of a start that was never the answer. That is what happened
    at the 2022-09-06 refit before this rule existed, and it blanked 21 days of the
    headline model's forecasts.
    """
    returns = _simulate_garch_t(900, _garch_theta(), seed=12)
    fit = M.fit_garch_mle(returns)
    assert fit.converged, fit.message
    assert fit.params.is_valid()
    assert np.isfinite(fit.loglik)


def test_mle_rejects_an_unknown_innovation_distribution() -> None:
    with pytest.raises(ValueError, match="'t' or 'normal'"):
        M.fit_garch_mle(
            _simulate_garch_t(100, _garch_theta(), seed=9), innovation="cauchy"
        )


def test_mle_refuses_a_window_too_short_to_identify_the_model() -> None:
    with pytest.raises(ValueError, match="at least 30"):
        M.fit_garch_mle(np.array([0.01, -0.02, 0.015]))


def test_normal_variant_returns_nan_degrees_of_freedom() -> None:
    """NaN rather than a sentinel: anything that forgets to branch produces a NaN.

    A sentinel such as ``nu = 0`` or ``nu = inf`` would flow silently into a predictive
    distribution and come out the other side as a plausible interval.
    """
    fit = M.fit_garch_mle(
        _simulate_garch_t(600, _garch_theta(), seed=10), innovation="normal"
    )
    assert fit.converged, fit.message
    assert np.isnan(fit.params.nu)


def test_standardised_residuals_are_the_residual_over_its_conditional_sd() -> None:
    returns = _simulate_garch_t(400, _garch_theta(), seed=11)
    params = M.GarchParams.from_array(_garch_theta())
    h0 = 1e-4

    resid = M.standardised_residuals(params, returns, h0)
    h = M.garch11_filter(params.to_array(), returns, h0)
    np.testing.assert_allclose(resid, (returns - params.mu) / np.sqrt(h), rtol=1e-14)
    # Well specified: unit scale. A loose bound -- this checks that the simulation and
    # the filter agree about the model, not that the residuals pass a distributional
    # test, which is what the warm-up diagnostics are for.
    assert abs(float(np.std(resid, ddof=1)) - 1.0) < 0.1


# --- The plug-in predictive --------------------------------------------------------


def test_student_t_predictive_has_the_variance_it_was_given() -> None:
    """The standardisation identity, stated as the property it is.

    ``scale**2 * nu/(nu-2) == variance`` is what makes ``h_t`` from the filter the
    conditional variance of the *return* rather than a scale parameter that still needs
    converting.
    """
    dist = M.StudentTPredictive(mean=0.0005, variance=1.2e-4, nu=6.0)
    assert dist.scale**2 * 6.0 / (6.0 - 2.0) == pytest.approx(1.2e-4, rel=1e-12)


def test_student_t_predictive_quantiles_match_scipy() -> None:
    dist = M.StudentTPredictive(mean=0.0005, variance=1.2e-4, nu=6.0)
    probs = np.array([0.005, 0.01, 0.05, 0.5, 0.95, 0.99, 0.995])
    expected = stats.t.ppf(probs, 6.0, loc=0.0005, scale=dist.scale)
    np.testing.assert_allclose(dist.quantile(probs), expected, rtol=1e-12)


def test_student_t_predictive_cdf_inverts_its_own_quantiles() -> None:
    dist = M.StudentTPredictive(mean=-0.001, variance=4e-4, nu=5.5)
    for p in (0.01, 0.1, 0.5, 0.9, 0.99):
        assert dist.cdf(float(dist.quantile(np.array([p]))[0])) == pytest.approx(p)


def test_dropping_the_standardisation_factor_would_widen_intervals() -> None:
    """**Negative control for the ``sqrt((nu-2)/nu)`` factor.**

    Without it the scale is ``sqrt(h)`` instead of ``sqrt(h)*sqrt((nu-2)/nu)``, and
    every interval comes out wider by ``sqrt(nu/(nu-2))`` -- about 9% at nu = 6, and in
    the direction that flatters the Bayesian model at Stage 3, whose intervals are
    supposed to be wider for an entirely different reason. Nothing would fail; the
    project would simply report parameter uncertainty it had never measured.
    """
    variance, nu = 1.2e-4, 6.0
    correct = M.StudentTPredictive(mean=0.0, variance=variance, nu=nu)
    probs = np.array([0.005, 0.995])

    naive = stats.t.ppf(probs, nu, loc=0.0, scale=np.sqrt(variance))
    naive_width = float(naive[1] - naive[0])
    correct_width = float(np.diff(correct.quantile(probs))[0])

    assert naive_width / correct_width == pytest.approx(np.sqrt(nu / (nu - 2.0)), rel=1e-9)
    assert naive_width > correct_width * 1.05


def test_student_t_predictive_rejects_degenerate_inputs() -> None:
    for bad in (0.0, -1e-6, np.nan, np.inf):
        with pytest.raises(ValueError, match="finite and positive"):
            M.StudentTPredictive(mean=0.0, variance=bad, nu=6.0)
    for bad_nu in (2.0, 1.5, -3.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="greater than 2"):
            M.StudentTPredictive(mean=0.0, variance=1e-4, nu=bad_nu)


def test_t_predictive_approaches_the_normal_as_nu_grows() -> None:
    """The mechanism behind the Stage 6 ablation, pinned at the interface."""
    probs = np.array([0.005, 0.01, 0.5, 0.99, 0.995])
    heavy = M.StudentTPredictive(mean=0.0, variance=1e-4, nu=1e7)
    gaussian = M.NormalPredictive(mean=0.0, variance=1e-4)
    np.testing.assert_allclose(heavy.quantile(probs), gaussian.quantile(probs), rtol=1e-4)


def test_plugin_predictive_dispatches_on_the_innovation_distribution() -> None:
    """A NaN ``nu`` means Gaussian innovations and must not reach a Student-t."""
    t_params = M.GarchParams.from_array(_garch_theta(nu=6.0))
    normal_params = M.GarchParams.from_array(_garch_theta(nu=float("nan")))

    assert isinstance(M.plugin_predictive(t_params, 1e-4), M.StudentTPredictive)
    assert isinstance(M.plugin_predictive(normal_params, 1e-4), M.NormalPredictive)


def test_plugin_predictive_carries_the_fitted_mean_and_variance() -> None:
    params = M.GarchParams.from_array(_garch_theta(mu=0.0007, nu=6.0))
    dist = M.plugin_predictive(params, 1.5e-4)
    assert dist.mean == pytest.approx(0.0007)
    assert dist.variance == pytest.approx(1.5e-4)
    assert dist.cdf(0.0007) == pytest.approx(0.5)


def test_plugin_predictive_rejects_a_degenerate_variance() -> None:
    params = M.GarchParams.from_array(_garch_theta())
    for bad in (0.0, -1e-8, np.nan):
        with pytest.raises(ValueError, match="finite and positive"):
            M.plugin_predictive(params, bad)


# ===================================================================================
# Stage 3: the priors, the sampler, and the posterior predictive
# ===================================================================================


def _posterior_draws(
    n: int, *, seed: int = 0, spread: float = 1.0, theta: np.ndarray | None = None
) -> np.ndarray:
    """A stand-in posterior, dispersed around ``theta``.

    Built here rather than sampled, so the predictive tests are about the predictive
    rather than about the sampler, and so ``spread=0`` gives an exactly degenerate
    posterior -- the limit in which the Bayesian model must reproduce the frequentist
    one.
    """
    base = _garch_theta() if theta is None else np.asarray(theta, dtype=float)
    if spread == 0.0:
        return np.tile(base, (n, 1))
    rng = np.random.default_rng(seed)
    scales = np.array([2e-4, 4e-7, 0.02, 0.02, 1.0]) * spread
    draws = base + rng.normal(0.0, scales, size=(n, len(base)))
    draws[:, 1] = np.abs(draws[:, 1])
    draws[:, 4] = np.clip(draws[:, 4], 4.5, None)
    return draws


# --- The priors --------------------------------------------------------------------


def test_priors_are_the_ones_frozen_at_d4() -> None:
    """The frozen decision, restated where a change to it would break a test.

    D4 was resolved in research_log.md 1.10 **before** any Bayesian out-of-sample number
    existed, which is the only thing that makes the project's headline claim falsifiable.
    A later edit to any of these constants is a change to a frozen decision, and it
    should have to argue with a test rather than slip through.
    """
    assert M.PRIOR_MU_SD == 1.0
    assert M.PRIOR_OMEGA_SD == 1.0
    assert M.PRIOR_ALPHA == (2.0, 10.0)
    assert M.PRIOR_DELTA == (3.0, 1.0)
    assert M.PRIOR_NU_MEAN == 10.0
    assert M.PRIOR_NU_LOWER == 4.0


def test_the_delta_prior_default_is_the_frozen_one_everywhere_it_is_settable() -> None:
    """The prior-sensitivity check must not be able to become the default run.

    Stage 6 owes a sensitivity check across the three ``delta`` candidates, and that
    turned out to need a second and third Bayesian backtest rather than a paragraph. The
    alternative prior is therefore threaded through the sampler and the backtest as an
    argument -- which creates a way for a one-line edit to re-specify the model that
    every headline number was computed under, without touching ``PRIOR_DELTA`` and so
    without tripping the test above.

    This pins the other half: wherever the prior is settable, its default is the frozen
    value. A sensitivity run has to say so at the call site.
    """
    import inspect

    from src import backtest as B

    for function in (M._build_pymc_model, M.sample_garch_posterior):
        default = inspect.signature(function).parameters["prior_delta"].default
        assert default == M.PRIOR_DELTA, function.__name__

    for function in (B.build_bayes_paths, B.run_backtest):
        default = inspect.signature(function).parameters["prior_delta"].default
        assert default == M.PRIOR_DELTA, function.__name__


def test_an_alternative_delta_prior_reaches_the_model_graph() -> None:
    """And the other half again: the argument must actually do something.

    A default-valued parameter that is silently ignored would pass the test above and
    make the whole sensitivity check vacuous -- three runs producing one answer, read as
    evidence of robustness. Checked on the graph rather than by sampling, so it costs
    milliseconds: the ``delta`` prior's parameters are read back off the built model.
    """
    pytest.importorskip("pymc")

    theta = np.array([0.05, 0.05, 0.10, 0.85, 7.0])  # percent scale, as D4 specifies
    returns = _simulate_garch_t(200, theta, seed=11)
    h0 = float(np.var(returns, ddof=1))

    def delta_prior_of(model) -> tuple[float, float]:
        # A pm.Beta RV's owner carries (rng, size, alpha, beta); the prior is the last two.
        alpha, beta = model["delta"].owner.inputs[-2:]
        return (float(alpha.eval()), float(beta.eval()))

    frozen = M._build_pymc_model(returns, h0)
    alternative = M._build_pymc_model(returns, h0, prior_delta=(10.0, 2.0))

    assert delta_prior_of(frozen) == M.PRIOR_DELTA
    assert delta_prior_of(alternative) == (10.0, 2.0)


def test_the_delta_prior_does_not_vanish_at_the_stationarity_boundary() -> None:
    """Why D4 departed from the feasibility probe, as a property rather than a note.

    The probe used ``delta ~ Beta(10, 2)``. A ``Beta(a, b)`` density vanishes at 1
    whenever ``b > 1``, so that prior places zero density at ``alpha + beta = 1`` -- the
    boundary research_log.md 1.9 recorded the data pressing against, with ``alpha+beta``
    at or above 0.999 in 16 of the 102 MLE refits. This asserts the adopted prior does
    not, and that the rejected one would have.
    """
    adopted = float(stats.beta.pdf(1.0, *M.PRIOR_DELTA))
    probe = float(stats.beta.pdf(1.0, 10.0, 2.0))

    assert adopted > 0.0
    assert probe == 0.0


def test_the_beta_prior_integrates_to_one_over_its_support() -> None:
    """**Negative control for the change-of-variable Jacobian.**

    The prior is specified over ``delta`` and ``log_prior`` states it over ``beta``, so
    it carries a factor ``1 / (1 - alpha)``. Drop that factor and the conditional
    density of ``beta`` integrates to ``1 - alpha`` instead of 1 -- a 20% error at
    ``alpha = 0.2``, silently, in a function no optimiser would complain about. It would
    matter for real on the route-B (emcee) path, where ``log_prior`` *is* the target.

    Integrating the joint over ``beta`` at fixed everything else must return the joint
    divided by exactly the ``beta`` conditional.
    """
    mu, omega, alpha, nu = 0.0005, 2.0e-6, 0.20, 7.0

    def joint(beta: float) -> float:
        return float(np.exp(M.log_prior(np.array([mu, omega, alpha, beta, nu]))))

    integral, _ = integrate.quad(joint, 0.0, 1.0 - alpha, limit=200)

    reference_beta = 0.6
    delta = reference_beta / (1.0 - alpha)
    conditional = float(stats.beta.pdf(delta, *M.PRIOR_DELTA)) / (1.0 - alpha)
    expected = joint(reference_beta) / conditional

    assert integral == pytest.approx(expected, rel=1e-8)


def test_log_prior_marginals_match_an_independent_construction() -> None:
    """The priors, restated from scipy rather than from the implementation."""
    mu, omega, alpha, beta, nu = 0.0005, 2.0e-6, 0.20, 0.60, 7.0
    delta = beta / (1.0 - alpha)

    expected = (
        stats.norm.logpdf(mu, 0.0, M.PRIOR_MU_SD)
        + stats.halfnorm.logpdf(omega, 0.0, M.PRIOR_OMEGA_SD)
        + stats.beta.logpdf(alpha, *M.PRIOR_ALPHA)
        + stats.beta.logpdf(delta, *M.PRIOR_DELTA)
        - np.log(1.0 - alpha)
        + stats.expon.logpdf(nu, loc=M.PRIOR_NU_LOWER, scale=M.PRIOR_NU_MEAN)
    )

    got = M.log_prior(np.array([mu, omega, alpha, beta, nu]))
    assert got == pytest.approx(float(expected), rel=1e-12)


@pytest.mark.parametrize(
    "theta",
    [
        np.array([0.0, -1e-6, 0.1, 0.8, 7.0]),  # omega <= 0
        np.array([0.0, 1e-6, -0.01, 0.8, 7.0]),  # alpha < 0
        np.array([0.0, 1e-6, 0.3, 0.75, 7.0]),  # alpha + beta >= 1
        np.array([0.0, 1e-6, 0.1, 0.8, 3.9]),  # nu <= 4
        np.array([np.nan, 1e-6, 0.1, 0.8, 7.0]),  # not finite
    ],
)
def test_log_prior_returns_minus_inf_outside_the_support(theta) -> None:
    """Support is enforced by the parameterisation, and stated here as a hard edge.

    Every constraint the model specification names -- positivity, stationarity, finite
    kurtosis -- has to be unreachable, not merely improbable. Returning ``-inf`` rather
    than raising is what lets one function serve an optimiser and a sampler alike.
    """
    assert M.log_prior(theta) == -np.inf


def test_log_posterior_is_the_prior_plus_the_shared_likelihood() -> None:
    """Models 3 and 4 differ only in what is done with one likelihood.

    Not a restatement of the implementation: the point is that the *shared*
    ``garch11_t_loglik`` is the term that appears, so the Bayesian model cannot drift
    onto a likelihood of its own.
    """
    # Percent scale, where D4 froze the priors and where the sampler works.
    theta = np.array([0.05, 0.05, 0.10, 0.85, 7.0])
    returns = _simulate_garch_t(300, theta, seed=3)
    h0 = float(np.var(returns, ddof=1))

    expected = M.log_prior(theta) + M.garch11_t_loglik(theta, returns, h0)
    assert M.log_posterior(theta, returns, h0) == pytest.approx(expected, rel=1e-12)


def test_log_posterior_never_evaluates_the_likelihood_outside_the_support() -> None:
    inadmissible = np.array([0.0, 1e-6, 0.3, 0.75, 7.0])
    returns = _simulate_garch_t(200, _garch_theta(), seed=4)
    assert M.log_posterior(inadmissible, returns, 1e-4) == -np.inf


# --- The PyMC model graph ----------------------------------------------------------


def test_the_pymc_graph_and_the_numpy_log_posterior_agree() -> None:
    """**The load-bearing cross-check of Stage 3.**

    The project's central design constraint is that models 3 and 4 share one likelihood.
    PyMC does not call ``garch11_t_loglik``; it builds its own graph, so that constraint
    is a claim about two implementations rather than about one function -- unless it is
    checked. Two things could break it silently: a variance recursion indexed
    differently from ``garch11_filter``, and a Student-t parameterised by scale where
    the NumPy version uses variance. Either would produce a posterior that samples
    cleanly, converges, and answers a different question.

    The two are compared **without** PyMC's transform Jacobians, since ``log_prior``
    states a density in the natural parameterisation. What remains is the deliberate
    ``log(1 - alpha)``: PyMC evaluates the prior over ``delta``, this module states it
    over ``beta``.
    """
    pt = pytest.importorskip("pytensor.tensor")

    theta = np.array([0.05, 0.05, 0.10, 0.85, 7.0])  # percent scale, as D4 specifies
    returns = _simulate_garch_t(400, theta, seed=11)
    h0 = float(np.var(returns, ddof=1))

    model = M._build_pymc_model(returns, h0)
    named = dict(zip(M.PARAM_NAMES, theta))
    named["delta"] = named["beta"] / (1.0 - named["alpha"])

    point = {}
    for rv in model.free_RVs:
        value_var = model.rvs_to_values[rv]
        transform = model.rvs_to_transforms.get(rv)
        x = np.asarray(named[rv.name], dtype=float)
        if transform is not None:
            x = transform.forward(pt.as_tensor_variable(x), *rv.owner.inputs).eval()
        point[value_var.name] = np.asarray(x, dtype=float)

    pymc_logp = float(model.compile_logp(jacobian=False)(point))
    numpy_logp = M.log_posterior(theta, returns, h0)

    assert pymc_logp - numpy_logp == pytest.approx(float(np.log(1.0 - theta[2])), abs=1e-6)


# --- The sampler -------------------------------------------------------------------


def test_bayes_convergence_thresholds_are_the_ones_frozen_at_d19() -> None:
    """Fixed before any refit ran. A threshold loosened after it fires is not one."""
    assert M.BAYES_CONVERGENCE["max_r_hat"] == 1.01
    assert M.BAYES_CONVERGENCE["max_divergences"] == 0
    assert M.BAYES_CONVERGENCE["min_ess_tail"] == 400.0


@pytest.mark.slow
def test_sampler_recovers_known_parameters_from_simulated_data() -> None:
    """The Bayesian analogue of the MLE recovery test, and on the same data-generating
    process written out independently of ``src``.

    Loose tolerances on purpose: 1,500 observations do not identify ``nu`` sharply, and
    a test that demanded they did would be testing the seed. What it must catch is a
    posterior in the wrong place -- which is what a mis-indexed recursion produces.
    """
    theta = _garch_theta(mu=0.0005, omega=2.0e-6, alpha=0.10, beta=0.85, nu=7.0)
    returns = _simulate_garch_t(1500, theta, seed=7)

    fit = M.sample_garch_posterior(
        returns, draws=500, tune=500, chains=4, thin=1, seed=0, cores=1
    )

    assert fit.converged, fit.message
    posterior_mean = fit.draws.mean(axis=0)
    assert posterior_mean[2] == pytest.approx(0.10, abs=0.06)  # alpha
    assert posterior_mean[3] == pytest.approx(0.85, abs=0.10)  # beta
    assert 4.0 < posterior_mean[4] < 20.0  # nu
    # Raw return scale on the way out, not the percent scale it sampled on.
    assert posterior_mean[1] < 1e-4  # omega
    assert abs(posterior_mean[0]) < 0.01  # mu


@pytest.mark.slow
def test_sampler_is_reproducible_bit_for_bit_under_one_seed() -> None:
    """A refit that cannot be reproduced cannot be audited."""
    returns = _simulate_garch_t(400, _garch_theta(), seed=8)
    kwargs = dict(draws=150, tune=150, chains=2, thin=1, seed=42, cores=1)

    first = M.sample_garch_posterior(returns, **kwargs)
    second = M.sample_garch_posterior(returns, **kwargs)

    np.testing.assert_array_equal(first.draws, second.draws)
    np.testing.assert_array_equal(first.log_prob, second.log_prob)
    assert first.n_divergences == second.n_divergences


@pytest.mark.slow
def test_a_fit_that_fails_its_diagnostics_says_so_rather_than_raising() -> None:
    """D19's failure path, forced rather than waited for.

    Behaviour under a failed fit must not depend on whether the real data happens to
    trigger one. A chain far too short to resolve the tails fails ``min_ess_tail``; the
    fit comes back with ``converged=False`` and a message naming the threshold and the
    parameter, and the backtest then produces no forecasts for that block. Nothing is
    re-run under a different seed in the hope of a better verdict.
    """
    returns = _simulate_garch_t(400, _garch_theta(), seed=9)

    fit = M.sample_garch_posterior(
        returns, draws=60, tune=100, chains=2, thin=1, seed=0, cores=1
    )

    assert not fit.converged
    assert "ess_tail" in fit.message
    assert fit.draws.shape == (120, len(M.PARAM_NAMES))  # the draws survive the verdict


@pytest.mark.slow
def test_thinning_is_deterministic_and_keeps_every_nth_draw() -> None:
    """D18 thins by taking every second draw, not a random subsample."""
    returns = _simulate_garch_t(300, _garch_theta(), seed=10)
    kwargs = dict(draws=100, tune=100, chains=2, seed=1, cores=1)

    full = M.sample_garch_posterior(returns, thin=1, **kwargs)
    thinned = M.sample_garch_posterior(returns, thin=2, **kwargs)

    assert full.draws.shape[0] == 200
    assert thinned.draws.shape[0] == 100
    np.testing.assert_array_equal(thinned.draws, full.draws[::2])


def test_sampler_rejects_a_window_too_short_to_identify_the_model() -> None:
    with pytest.raises(ValueError, match="at least 30"):
        M.sample_garch_posterior(np.zeros(10) + 0.001)


# --- The posterior predictive ------------------------------------------------------


def test_posterior_predictive_refuses_one_shared_h_next() -> None:
    """**The single most dangerous bug in this codebase, made unreachable.**

    Each posterior draw implies its own filtered variance path and therefore its own
    ``h_next``. Passing one shared value -- the variance at the posterior mean, say --
    discards most of the parameter uncertainty and collapses this predictive towards the
    plug-in. Nothing would fail. The Bayesian and frequentist intervals would simply
    agree, and that agreement would be reported as the project's finding.

    So the shape is a contract, not a convention.
    """
    draws = _posterior_draws(200)
    levels = np.array([0.05, 0.95])

    for wrong in (np.array([1.2e-4]), np.full((200, 1), 1.2e-4), np.full(199, 1.2e-4)):
        with pytest.raises(ValueError, match="one variance per posterior draw"):
            M.posterior_predictive(draws, wrong, levels)


def test_a_posterior_collapsed_to_a_point_reproduces_the_plug_in() -> None:
    """The no-parameter-uncertainty limit, where the two models must coincide exactly.

    The other half of the guard above. If the mixture did not reduce to the plug-in when
    the posterior is degenerate, any interval difference measured later would be part
    mechanism and part arithmetic error, and there would be no way to tell which.
    """
    theta = _garch_theta(nu=6.0)
    h_next = 1.2e-4
    levels = np.array([0.005, 0.05, 0.5, 0.95, 0.995])

    draws = _posterior_draws(300, spread=0.0, theta=theta)
    mean, quantiles = M.posterior_predictive(draws, np.full(300, h_next), levels)

    plug_in = M.plugin_predictive(M.GarchParams.from_array(theta), h_next)
    np.testing.assert_allclose(quantiles, plug_in.quantile(levels), rtol=1e-10)
    assert mean == pytest.approx(theta[0], rel=1e-12)


def test_a_dispersed_posterior_is_wider_than_the_plug_in_where_it_matters() -> None:
    """**The project's research question, as an assertion -- and only where it holds.**

    Integrating over parameter uncertainty must widen the predictive relative to
    conditioning on a point estimate. If this ever fails at 95% or 99%, the mixture is
    not carrying parameter uncertainty and every downstream coverage number is measuring
    plumbing rather than statistics.

    **Not at 90%,** and that is not a weakened assertion. A scale mixture holding average
    variance fixed is leptokurtic against the single distribution at that average: more
    peaked in the middle, heavier in the tails, crossing over somewhere between 90% and
    95% for dispersions of this size. Asserting "wider at every level" would be
    asserting something false, and code changed until it passed would be broken code.
    Measured and recorded in research_log.md 1.11 and problems-and-solutions 38; the
    companion test below pins the crossover itself.
    """
    theta = _garch_theta(nu=6.0)
    draws = _posterior_draws(2000, seed=1, spread=1.0, theta=theta)
    rng = np.random.default_rng(2)
    h_next_by_draw = 1.2e-4 * np.exp(rng.normal(0.0, 0.25, size=2000))

    predictive = M.mixture_predictive(draws, h_next_by_draw)
    plug_in = M.plugin_predictive(
        M.GarchParams.from_array(draws.mean(axis=0)), float(h_next_by_draw.mean())
    )

    for level in (0.95, 0.99):
        probs = np.array([0.5 - level / 2.0, 0.5 + level / 2.0])
        mixture_width = float(np.diff(predictive.quantile(probs))[0])
        plug_in_width = float(np.diff(plug_in.quantile(probs))[0])
        assert mixture_width > plug_in_width, f"not wider at the {level:.0%} level"


def test_the_mixture_is_more_peaked_in_the_middle_than_the_plug_in() -> None:
    """The other side of leptokurtosis, pinned so it is not later read as a bug.

    The handoff's Stage 3 sanity check says Bayesian intervals should come out wider and
    to hunt for a bug if they do not. That is right at 99% and wrong at 90%: the same
    mixing that fattens the tails thins the shoulders, because the total variance is
    conserved. Anyone who meets a narrower 90% interval downstream should meet this test
    rather than start debugging.
    """
    theta = _garch_theta(nu=6.0)
    draws = _posterior_draws(2000, seed=1, spread=1.0, theta=theta)
    rng = np.random.default_rng(2)
    h_next_by_draw = 1.2e-4 * np.exp(rng.normal(0.0, 0.25, size=2000))

    predictive = M.mixture_predictive(draws, h_next_by_draw)
    plug_in = M.plugin_predictive(
        M.GarchParams.from_array(draws.mean(axis=0)), float(h_next_by_draw.mean())
    )

    probs = np.array([0.05, 0.95])
    assert float(np.diff(predictive.quantile(probs))[0]) < float(
        np.diff(plug_in.quantile(probs))[0]
    )


def test_mixture_quantiles_match_a_monte_carlo_draw_from_the_same_mixture() -> None:
    """**Independent check on the numerical quantile solve.**

    The mixture CDF is inverted by bracketing between the smallest and largest component
    quantiles. That bracket is provably valid, but "provably" is what every wrong
    implementation also believes, so the answer is checked against sampling from the
    mixture directly -- draw a component, then draw its innovation -- which shares no
    code with the solve.
    """
    theta = _garch_theta(nu=6.0)
    n = 1000
    draws = _posterior_draws(n, seed=1, spread=1.0, theta=theta)
    rng = np.random.default_rng(2)
    h_next_by_draw = 1.2e-4 * np.exp(rng.normal(0.0, 0.25, size=n))
    predictive = M.mixture_predictive(draws, h_next_by_draw)

    mc = np.random.default_rng(99)
    index = mc.integers(0, n, size=500_000)
    nus = draws[index, 4]
    scales = np.sqrt(h_next_by_draw[index]) * np.sqrt((nus - 2.0) / nus)
    sample = draws[index, 0] + scales * mc.standard_t(nus)

    probs = np.array([0.05, 0.5, 0.95, 0.99])
    np.testing.assert_allclose(
        predictive.quantile(probs), np.quantile(sample, probs), rtol=0.0, atol=5e-4
    )


def test_parameter_uncertainty_bites_hardest_in_the_tails() -> None:
    """The mechanism behind the expected Stage 4 result, checked at the source.

    Widening is not uniform across levels: the further into the tail, the more the
    spread of ``h_next`` and ``nu`` across draws matters. That is why this project
    expects its finding at 99% rather than at 90%, and it should be a property of the
    predictive rather than an assertion in the write-up.
    """
    theta = _garch_theta(nu=6.0)
    draws = _posterior_draws(2000, seed=3, spread=1.0, theta=theta)
    rng = np.random.default_rng(4)
    h_next_by_draw = 1.2e-4 * np.exp(rng.normal(0.0, 0.25, size=2000))

    predictive = M.mixture_predictive(draws, h_next_by_draw)
    plug_in = M.plugin_predictive(
        M.GarchParams.from_array(draws.mean(axis=0)), float(h_next_by_draw.mean())
    )

    def ratio(level: float) -> float:
        probs = np.array([0.5 - level / 2.0, 0.5 + level / 2.0])
        return float(
            np.diff(predictive.quantile(probs))[0] / np.diff(plug_in.quantile(probs))[0]
        )

    assert ratio(0.99) > ratio(0.90)


def test_mixture_cdf_is_the_average_of_its_components() -> None:
    """Equal weights, restated from scipy rather than from the implementation."""
    draws = _posterior_draws(50, seed=5)
    h_next_by_draw = np.full(50, 1.2e-4) * np.linspace(0.8, 1.2, 50)
    predictive = M.mixture_predictive(draws, h_next_by_draw)

    r = -0.02
    expected = np.mean(
        [
            M.StudentTPredictive(mean=m, variance=v, nu=nu).cdf(r)
            for m, v, nu in zip(draws[:, 0], h_next_by_draw, draws[:, 4])
        ]
    )
    assert predictive.cdf(r) == pytest.approx(float(expected), rel=1e-12)


def test_mixture_cdf_inverts_its_own_quantiles() -> None:
    """The PIT and the interval bounds come from one object; they must agree."""
    draws = _posterior_draws(300, seed=6)
    h_next_by_draw = 1.2e-4 * np.linspace(0.7, 1.4, 300)
    predictive = M.mixture_predictive(draws, h_next_by_draw)

    for p in (0.005, 0.05, 0.5, 0.95, 0.995):
        x = float(predictive.quantile(np.array([p]))[0])
        assert predictive.cdf(x) == pytest.approx(p, abs=1e-10)


def test_mixture_mean_is_the_posterior_mean_of_mu() -> None:
    draws = _posterior_draws(400, seed=7)
    predictive = M.mixture_predictive(draws, np.full(400, 1.2e-4))
    assert predictive.mean == pytest.approx(float(draws[:, 0].mean()), rel=1e-12)


def test_mixture_rejects_degenerate_components() -> None:
    """One bad draw must not be averaged into a plausible-looking answer."""
    with pytest.raises(ValueError, match="finite and positive"):
        M.StudentTMixturePredictive(
            means=np.zeros(3), variances=np.array([1e-4, 0.0, 1e-4]), nus=np.full(3, 6.0)
        )
    with pytest.raises(ValueError, match="greater than 2"):
        M.StudentTMixturePredictive(
            means=np.zeros(3), variances=np.full(3, 1e-4), nus=np.array([6.0, 1.5, 6.0])
        )
    with pytest.raises(ValueError, match="same shape"):
        M.StudentTMixturePredictive(
            means=np.zeros(3), variances=np.full(2, 1e-4), nus=np.full(3, 6.0)
        )


def test_posterior_predictive_rejects_a_draw_matrix_of_the_wrong_width() -> None:
    with pytest.raises(ValueError, match="shape"):
        M.posterior_predictive(
            np.zeros((10, 4)), np.full(10, 1e-4), np.array([0.05, 0.95])
        )
