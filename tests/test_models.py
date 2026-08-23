"""Stage 1 tests for ``src.models``: the predictive interface and the two baselines.

The GARCH half of the module is still stubbed; its tests arrive with Stages 2 and 3.
What is covered here is the interface every model will pass through, which is worth
getting right before there are four models depending on it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

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
