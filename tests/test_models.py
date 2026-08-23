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
