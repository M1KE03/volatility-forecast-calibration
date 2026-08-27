"""Stage 4 and 5 tests for ``src.evaluation``.

Five categories, kept separate because they fail for different reasons:

- **Definition tests** pin each loss and statistic to the formula the report names,
  including the ones a plausible alternative would quietly change: QLIKE's asymmetry,
  DM's sign convention, the degrees of freedom of each likelihood-ratio test.
- **Power tests** run every diagnostic twice, on data that should trigger it and on
  data that should not. A calibration test that only ever sees a miscalibrated model
  cannot distinguish a working test from one that always rejects -- and "always
  rejects" would hand this project a fabricated headline.
- **Degeneracy tests** cover the cases that arise in the real sample rather than in
  principle. At the 99% level over ~2,100 days a good model produces around twenty
  breaches and often none in consecutive pairs, so ``n11 = 0`` is the *normal* case for
  the independence test, not an edge case; a NaN there would silently blank the money
  test for the best-behaved models.
- **Structural tests** assert the module's central separation: nothing scored against
  observed returns may depend on the volatility proxy. The proxy is scrambled and every
  calibration number must come back bit-identical.
- **Refusal tests** pin D28: a missing forecast is an error, not a row to drop, so that
  no statistic is computed on a sample the table does not name. The regime tables add
  D33's floor: a subsample below thirty days is refused rather than printed beside
  subsamples in the hundreds.

The Stage 5 block at the end covers the regime split. Two of its tests exist to keep
decisions from being tidied away by a later reader: that the regime VaR table runs Kupiec
and deliberately *not* Christoffersen (D32, because a regime subsample's consecutive rows
can be months apart), and that the label the evaluation layer splits on is the lagged one
the data layer built.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src import backtest as B
from src import evaluation as E

SEED = 20260827
LEVELS: tuple[float, ...] = (0.90, 0.95, 0.99)


# --- Fixtures ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def forecasts() -> pd.DataFrame:
    """The committed forecast table. These tests are about the real output."""
    path = Path("data/processed/forecasts.csv")
    if not path.exists():  # pragma: no cover - depends on local state
        pytest.skip(f"{path} not present; run `python run_all.py --stage backtest`")
    return pd.read_csv(path, parse_dates=["date"])


@pytest.fixture(scope="module")
def scored(forecasts: pd.DataFrame) -> pd.DataFrame:
    return E.score_forecasts(forecasts)


def _synthetic_forecasts(n: int = 300, *, missing: int = 0) -> pd.DataFrame:
    """A small two-model forecast table with the same schema as the real one.

    ``missing`` days of the second model are blanked, reproducing the shape a failed
    Bayesian refit leaves behind (D19).
    """
    rng = np.random.default_rng(SEED)
    dates = pd.bdate_range("2017-01-03", periods=n)
    returns = rng.normal(0.0, 0.01, size=n)

    rows = []
    for model, scale in (("model_a", 1.0), ("model_b", 1.2)):
        variance = np.full(n, (0.01 * scale) ** 2)
        sd = np.sqrt(variance)
        row = {
            "date": dates,
            "model": model,
            "variance": variance,
            "mean": 0.0,
            "log_return": returns,
            "proxy_var": np.full(n, 0.01**2),
            "regime": np.where(
                np.arange(n) < n // 2, "calm", np.where(np.arange(n) < 3 * n // 4, "normal", "stressed")
            ),
            "refit_id": np.arange(n) // 21,
            "pit": stats.norm.cdf(returns / sd),
        }
        for level in LEVELS:
            key = E.level_key(level)
            half = stats.norm.ppf(0.5 + level / 2.0) * sd
            row[f"lo_{key}"] = -half
            row[f"hi_{key}"] = half
        row["var_99"] = stats.norm.ppf(0.01) * sd
        rows.append(pd.DataFrame(row))

    frame = pd.concat(rows, ignore_index=True)
    if missing:
        blank = (frame["model"] == "model_b") & (
            frame["date"].isin(dates[-missing:])
        )
        numeric = [c for c in frame.columns if c not in ("date", "model", "regime")]
        frame.loc[blank, numeric] = np.nan
    return frame


# --- Point losses: definitions ----------------------------------------------------


def test_qlike_is_zero_when_the_forecast_equals_the_proxy() -> None:
    values = np.array([1e-4, 4e-4, 9e-4])
    assert E.qlike_loss(values, values) == pytest.approx(np.zeros(3))


def test_qlike_is_positive_away_from_the_proxy() -> None:
    realised = np.full(5, 1e-4)
    assert np.all(E.qlike_loss(realised, realised * 1.5) > 0.0)
    assert np.all(E.qlike_loss(realised, realised * 0.5) > 0.0)


def test_qlike_penalises_under_forecasting_more_than_over_forecasting() -> None:
    """The asymmetry is the reason QLIKE is the headline loss for a risk application.

    Halving and doubling the forecast are the same multiplicative error; QLIKE must
    charge more for the half, because a variance forecast that is too low is what puts
    a book through its limits.
    """
    realised = np.full(3, 1e-4)
    too_low = E.qlike_loss(realised, realised / 2.0)
    too_high = E.qlike_loss(realised, realised * 2.0)
    assert np.all(too_low > too_high)


def test_qlike_differences_are_unchanged_by_the_normalising_term() -> None:
    """The normalised form differs from ``r/f + log f`` by a term in the proxy alone.

    Model *rankings* and every loss *differential* are therefore identical under either
    form, which is what lets the report quote a readable level without changing a
    single comparison.
    """
    rng = np.random.default_rng(SEED)
    realised = np.abs(rng.normal(1e-4, 3e-5, size=200)) + 1e-6
    forecast_a = np.abs(rng.normal(1e-4, 3e-5, size=200)) + 1e-6
    forecast_b = np.abs(rng.normal(1e-4, 3e-5, size=200)) + 1e-6

    normalised = E.qlike_loss(realised, forecast_a) - E.qlike_loss(realised, forecast_b)
    raw_a = realised / forecast_a + np.log(forecast_a)
    raw_b = realised / forecast_b + np.log(forecast_b)
    assert normalised == pytest.approx(raw_a - raw_b)


def test_qlike_refuses_a_non_positive_variance() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        E.qlike_loss(np.array([1e-4, 0.0]), np.array([1e-4, 1e-4]))


def test_mse_is_zero_at_equality_and_positive_otherwise() -> None:
    values = np.array([1e-4, 2e-4])
    assert E.mse_variance_loss(values, values) == pytest.approx(np.zeros(2))
    assert np.all(E.mse_variance_loss(values, values * 2) > 0.0)


def test_both_losses_return_one_value_per_observation() -> None:
    """They are returned per-observation because DM and the bootstrap need the series,
    not its mean."""
    realised = np.full(17, 1e-4)
    forecast = np.full(17, 2e-4)
    assert E.qlike_loss(realised, forecast).shape == (17,)
    assert E.mse_variance_loss(realised, forecast).shape == (17,)


# --- Interval calibration ---------------------------------------------------------


def test_coverage_counts_each_tail_separately() -> None:
    returns = np.array([-3.0, -2.0, 0.0, 2.0, 5.0])
    lower = np.full(5, -2.5)
    upper = np.full(5, 2.5)
    result = E.interval_coverage(returns, lower, upper, 0.90)
    assert result.n == 5
    assert result.n_below == 1
    assert result.n_above == 1
    assert result.inside == 3
    assert result.empirical == pytest.approx(0.6)


def test_interval_bounds_are_inclusive() -> None:
    returns = np.array([-1.0, 1.0])
    result = E.interval_coverage(returns, np.full(2, -1.0), np.full(2, 1.0), 0.95)
    assert result.empirical == 1.0


def test_coverage_of_a_correctly_specified_predictive_is_near_nominal() -> None:
    """The positive case: a model that is right must be scored as right."""
    rng = np.random.default_rng(SEED)
    returns = rng.normal(size=40_000)
    for level in LEVELS:
        half = stats.norm.ppf(0.5 + level / 2.0)
        result = E.interval_coverage(returns, np.full(40_000, -half), np.full(40_000, half), level)
        assert result.empirical == pytest.approx(level, abs=0.01)


def test_coverage_detects_intervals_that_are_too_narrow() -> None:
    """The negative case, and the failure this project exists to detect."""
    rng = np.random.default_rng(SEED)
    returns = rng.normal(size=20_000)
    half = stats.norm.ppf(0.975) * 0.5  # half the width it should be
    result = E.interval_coverage(returns, np.full(20_000, -half), np.full(20_000, half), 0.95)
    assert result.empirical < 0.80


def test_interval_inside_is_the_series_behind_the_coverage_count() -> None:
    rng = np.random.default_rng(SEED)
    returns = rng.normal(size=500)
    lower, upper = np.full(500, -1.5), np.full(500, 1.5)
    inside = E.interval_inside(returns, lower, upper)
    result = E.interval_coverage(returns, lower, upper, 0.90)
    assert inside.mean() == pytest.approx(result.empirical)
    assert set(np.unique(inside)) <= {0.0, 1.0}


def test_inverted_bounds_are_refused() -> None:
    with pytest.raises(ValueError, match="below lower"):
        E.interval_coverage(np.zeros(3), np.ones(3), -np.ones(3), 0.95)


def test_interval_width_is_the_gap_between_the_bounds() -> None:
    assert E.interval_width(np.array([-1.0, -2.0]), np.array([1.0, 3.0])) == pytest.approx(
        np.array([2.0, 5.0])
    )


# --- PIT --------------------------------------------------------------------------


def test_pit_returns_the_stored_cdf_values_it_was_given() -> None:
    """The PIT is computed at forecast time by the model that owns the CDF.

    Recomputing it here would mean a second implementation of every predictive
    distribution -- including the Monte Carlo mixture -- living in the evaluation
    layer, free to drift from the one that actually produced the forecasts.
    """
    stored = np.array([0.1, 0.5, 0.9])
    assert E.pit_values(np.zeros(3), stored) == pytest.approx(stored)


def test_pit_refuses_values_outside_the_unit_interval() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        E.pit_values(np.zeros(3), np.array([0.1, 1.4, 0.9]))


def test_uniformity_test_does_not_fire_on_a_calibrated_predictive() -> None:
    uniform = np.random.default_rng(SEED).random(2000)
    assert E.pit_uniformity_test(uniform).p_value > 0.05


def test_uniformity_test_fires_on_a_predictive_that_is_too_narrow() -> None:
    """Intervals that are too narrow push PIT mass into both tails."""
    rng = np.random.default_rng(SEED)
    returns = rng.normal(size=2000)
    too_narrow = stats.norm.cdf(returns / 0.5)
    assert E.pit_uniformity_test(too_narrow).p_value < 0.01


def test_uniformity_result_carries_its_own_approximation_flag() -> None:
    """The KS p-value assumes a fully specified predictive. These have estimated,
    rolling-refitted parameters, so it is optimistic -- and the flag has to survive
    into any table built from the result, not live in a comment."""
    result = E.pit_uniformity_test(np.random.default_rng(SEED).random(500))
    assert result.p_value_is_approximate is True


# --- VaR backtests ----------------------------------------------------------------


def test_exceedances_flag_the_loss_tail_only() -> None:
    returns = np.array([-0.05, -0.01, 0.0, 0.05])
    var_forecast = np.full(4, -0.02)
    assert E.var_exceedances(returns, var_forecast) == pytest.approx(
        np.array([1.0, 0.0, 0.0, 0.0])
    )


def test_kupiec_matches_an_independent_likelihood_ratio() -> None:
    """Checked against binomial log-densities, where the coefficient cancels in the
    ratio -- an independent route to the same statistic."""
    n, hits, rate = 1000, 20, 0.01
    exceedances = np.zeros(n)
    exceedances[:hits] = 1.0
    expected = -2.0 * (
        stats.binom.logpmf(hits, n, rate) - stats.binom.logpmf(hits, n, hits / n)
    )
    statistic, _ = E.kupiec_pof_test(exceedances, rate)
    assert statistic == pytest.approx(expected)


def test_kupiec_is_silent_when_the_breach_rate_is_nominal() -> None:
    exceedances = np.zeros(1000)
    exceedances[:10] = 1.0
    statistic, p_value = E.kupiec_pof_test(exceedances, 0.01)
    assert statistic == pytest.approx(0.0, abs=1e-9)
    assert p_value > 0.99


def test_kupiec_fires_when_breaches_run_at_three_times_the_nominal_rate() -> None:
    exceedances = np.zeros(1000)
    exceedances[:30] = 1.0
    _, p_value = E.kupiec_pof_test(exceedances, 0.01)
    assert p_value < 0.01


def test_kupiec_is_finite_when_no_breach_occurs_at_all() -> None:
    """Zero breaches is a real outcome for an over-wide interval, and ``0 log 0`` has
    to be handled rather than propagated as a NaN."""
    statistic, p_value = E.kupiec_pof_test(np.zeros(2000), 0.01)
    assert np.isfinite(statistic)
    assert statistic == pytest.approx(-2.0 * 2000 * np.log(0.99))
    assert p_value < 0.01


def test_independence_is_silent_on_an_iid_hit_sequence() -> None:
    rng = np.random.default_rng(SEED)
    exceedances = (rng.random(3000) < 0.01).astype(float)
    _, p_value = E.christoffersen_independence_test(exceedances)
    assert p_value > 0.05


def test_independence_fires_when_breaches_cluster() -> None:
    """The money test. Twenty breaches in one run is the same *count* as twenty spread
    across the sample, and Kupiec cannot tell them apart."""
    clustered = np.zeros(2000)
    clustered[500:520] = 1.0
    _, p_value = E.christoffersen_independence_test(clustered)
    assert p_value < 0.001

    spread = np.zeros(2000)
    spread[::100] = 1.0
    _, spread_p = E.christoffersen_independence_test(spread)
    assert spread_p > 0.05


def test_independence_matches_a_hand_computed_transition_likelihood() -> None:
    """A sequence small enough to count by hand: n00=2, n01=2, n10=2, n11=1."""
    exceedances = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0])
    pi, pi01, pi11 = 3.0 / 7.0, 2.0 / 4.0, 1.0 / 3.0
    log_l_null = 4 * np.log(1 - pi) + 3 * np.log(pi)
    log_l_alt = (
        2 * np.log(1 - pi01) + 2 * np.log(pi01) + 2 * np.log(1 - pi11) + 1 * np.log(pi11)
    )
    statistic, _ = E.christoffersen_independence_test(exceedances)
    assert statistic == pytest.approx(-2.0 * (log_l_null - log_l_alt))


def test_independence_is_finite_when_no_breach_follows_a_breach() -> None:
    """``n11 = 0`` is the *normal* case at the 99% level, not an edge case: about
    twenty breaches over two thousand days rarely produces a consecutive pair. Without
    the ``0 log 0`` convention this test would return NaN for exactly the models that
    are behaving best."""
    exceedances = np.zeros(2000)
    exceedances[::97] = 1.0
    statistic, p_value = E.christoffersen_independence_test(exceedances)
    assert np.isfinite(statistic)
    assert np.isfinite(p_value)


def test_independence_is_zero_when_there_are_no_breaches() -> None:
    statistic, p_value = E.christoffersen_independence_test(np.zeros(500))
    assert statistic == pytest.approx(0.0)
    assert p_value == pytest.approx(1.0)


def test_conditional_coverage_is_its_two_components_on_two_degrees_of_freedom() -> None:
    """Christoffersen's joint test is the sum of the Kupiec and independence
    statistics. The components are computed on samples differing by one observation --
    Kupiec on all n days, independence on the n-1 transitions -- which is his own
    construction and is pinned here so it cannot drift."""
    rng = np.random.default_rng(SEED)
    exceedances = (rng.random(2000) < 0.02).astype(float)

    pof, _ = E.kupiec_pof_test(exceedances, 0.01)
    ind, _ = E.christoffersen_independence_test(exceedances)
    statistic, p_value = E.conditional_coverage_test(exceedances, 0.01)

    assert statistic == pytest.approx(pof + ind)
    assert p_value == pytest.approx(stats.chi2.sf(statistic, df=2))


def test_conditional_coverage_catches_a_failure_kupiec_alone_would_miss() -> None:
    """Correct average coverage, every breach in one fortnight."""
    exceedances = np.zeros(2000)
    exceedances[600:620] = 1.0
    _, kupiec_p = E.kupiec_pof_test(exceedances, 0.01)
    _, cc_p = E.conditional_coverage_test(exceedances, 0.01)
    assert kupiec_p > 0.05
    assert cc_p < 0.01


def test_var_backtests_refuse_a_non_binary_hit_sequence() -> None:
    hits = np.zeros(50)
    hits[3] = 0.5
    with pytest.raises(ValueError, match="only 0 and 1"):
        E.kupiec_pof_test(hits, 0.01)
    with pytest.raises(ValueError, match="only 0 and 1"):
        E.christoffersen_independence_test(hits)


def test_a_nominal_rate_outside_the_unit_interval_is_refused() -> None:
    with pytest.raises(ValueError, match="nominal_rate"):
        E.kupiec_pof_test(np.zeros(50), 1.5)


# --- Diebold-Mariano --------------------------------------------------------------


def test_a_positive_statistic_means_the_first_model_is_the_worse_one() -> None:
    """The one thing the test is not symmetric in. A sign error here would reverse
    every conclusion drawn from it."""
    rng = np.random.default_rng(SEED)
    loss_b = np.abs(rng.normal(size=500))
    loss_a = loss_b + 0.2
    result = E.diebold_mariano(loss_a, loss_b)
    assert result.statistic > 0.0
    assert result.mean_loss_differential > 0.0


def test_swapping_the_arguments_flips_the_sign_and_keeps_the_p_value() -> None:
    rng = np.random.default_rng(SEED)
    loss_a = np.abs(rng.normal(size=500))
    loss_b = np.abs(rng.normal(size=500))
    forward = E.diebold_mariano(loss_a, loss_b)
    reverse = E.diebold_mariano(loss_b, loss_a)
    assert forward.statistic == pytest.approx(-reverse.statistic)
    assert forward.p_value == pytest.approx(reverse.p_value)


def test_dm_does_not_fire_on_two_equally_accurate_forecasters() -> None:
    """The negative case. Two independent draws from the same loss distribution must
    not be separated."""
    rng = np.random.default_rng(SEED)
    loss_a = np.abs(rng.normal(size=1500))
    loss_b = np.abs(rng.normal(size=1500))
    assert E.diebold_mariano(loss_a, loss_b).p_value > 0.05


def test_dm_detects_a_genuine_difference_in_accuracy() -> None:
    rng = np.random.default_rng(SEED)
    loss_a = np.abs(rng.normal(size=1500)) + 0.15
    loss_b = np.abs(rng.normal(size=1500))
    assert E.diebold_mariano(loss_a, loss_b).p_value < 0.01


def test_dm_applies_the_harvey_leybourne_newbold_correction() -> None:
    """At zero lags the statistic is the mean differential over its own standard error,
    shrunk by ``sqrt((n-1)/n)`` and referred to a t distribution rather than a normal.
    All three pieces are checked at once, against the formula written out."""
    rng = np.random.default_rng(SEED)
    differential = rng.normal(0.05, 1.0, size=400)
    loss_a, loss_b = differential, np.zeros(400)

    result = E.diebold_mariano(loss_a, loss_b, lag_truncation=0)
    n = 400
    mean = differential.mean()
    gamma_0 = np.mean((differential - mean) ** 2)
    expected = mean / np.sqrt(gamma_0 / n) * np.sqrt((n - 1) / n)
    assert result.statistic == pytest.approx(expected)
    assert result.p_value == pytest.approx(2 * stats.t.sf(abs(expected), df=n - 1))


def test_the_hac_correction_widens_the_standard_error_on_a_dependent_differential() -> None:
    """A positively autocorrelated loss differential has a larger long-run variance
    than its contemporaneous one, so ignoring the autocorrelation overstates
    significance. That is why a lag truncation is applied at all (D30)."""
    rng = np.random.default_rng(SEED)
    n = 1000
    differential = np.empty(n)
    differential[0] = rng.normal()
    for t in range(1, n):
        differential[t] = 0.8 * differential[t - 1] + rng.normal()
    differential = differential + 0.3

    naive = E.diebold_mariano(differential, np.zeros(n), lag_truncation=0)
    corrected = E.diebold_mariano(differential, np.zeros(n))
    assert abs(corrected.statistic) < abs(naive.statistic)
    assert corrected.lag_truncation > 0


def test_the_default_lag_truncation_follows_the_newey_west_rule() -> None:
    """Fixed before any test was run, so no reported p-value is the product of a lag
    chosen after seeing it."""
    assert E.newey_west_lag(100) == 4
    assert E.newey_west_lag(2092) == 7
    rng = np.random.default_rng(SEED)
    result = E.diebold_mariano(rng.normal(size=2092), rng.normal(size=2092))
    assert result.lag_truncation == E.newey_west_lag(2092)


def test_dm_reports_the_differential_variance_it_was_applied_to() -> None:
    """Models 3 and 4 share a likelihood, so their differential can approach
    degeneracy, where DM is badly sized. The variance is returned so a reader can check
    rather than trust the p-value."""
    rng = np.random.default_rng(SEED)
    loss_b = np.abs(rng.normal(size=300))
    loss_a = loss_b + rng.normal(0.0, 1e-9, size=300)
    result = E.diebold_mariano(loss_a, loss_b)
    assert result.differential_variance < 1e-15
    assert result.n == 300


def test_two_identical_loss_series_are_refused_rather_than_tested() -> None:
    rng = np.random.default_rng(SEED)
    loss = np.abs(rng.normal(size=300))
    with pytest.raises(ValueError, match="zero variance"):
        E.diebold_mariano(loss, loss)


# --- The common sample (D28) ------------------------------------------------------


def test_common_sample_is_the_intersection_not_the_union() -> None:
    frame = _synthetic_forecasts(n=300, missing=42)
    dates = E.common_sample(frame, ("model_a", "model_b"))
    assert len(dates) == 258
    assert len(E.common_sample(frame, ("model_a",))) == 300


def test_common_sample_rejects_a_model_that_is_not_in_the_table() -> None:
    frame = _synthetic_forecasts(n=100)
    with pytest.raises(KeyError, match="not present"):
        E.common_sample(frame, ("model_a", "nonexistent"))


def test_the_real_common_sample_is_the_one_the_handoff_records(
    forecasts: pd.DataFrame,
) -> None:
    """Two Bayesian refits failed their diagnostics, so 42 of the 2,134 evaluation days
    carry no Bayesian forecast (D19). That is a property of the model, and the number
    is pinned so it cannot change unnoticed."""
    assert len(E.common_sample(forecasts, ("yesterday", "ewma", "garch_mle"))) == 2134
    assert len(E.common_sample(forecasts, B.HEADLINE_MODELS)) == 2092


# --- Refusals (D28) ---------------------------------------------------------------


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda x: E.qlike_loss(x, np.full(20, 1e-4)), id="qlike"),
        pytest.param(lambda x: E.mse_variance_loss(x, np.full(20, 1e-4)), id="mse"),
        pytest.param(
            lambda x: E.interval_coverage(x, np.full(20, -1.0), np.full(20, 1.0), 0.95),
            id="coverage",
        ),
        pytest.param(lambda x: E.pit_values(np.zeros(20), np.clip(x, 0, 1)), id="pit"),
        pytest.param(
            lambda x: E.var_exceedances(x, np.full(20, -0.02)), id="exceedances"
        ),
        pytest.param(
            lambda x: E.diebold_mariano(x, np.arange(20, dtype=float)), id="dm"
        ),
    ],
)
def test_a_missing_forecast_is_an_error_rather_than_a_dropped_row(call) -> None:
    """Silently dropping it would let each statistic be computed on a different,
    unstated sample -- and a coverage table whose rows are not comparable is worse than
    one that is missing."""
    values = np.full(20, 1e-4)
    values[5] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        call(values)


def test_series_of_different_lengths_are_refused() -> None:
    with pytest.raises(ValueError, match="same length"):
        E.qlike_loss(np.full(10, 1e-4), np.full(9, 1e-4))


# --- Regime summaries -------------------------------------------------------------


def test_regime_summary_reports_n_alongside_every_statistic() -> None:
    frame = _synthetic_forecasts(n=300)
    frame["loss"] = 1.0
    summary = E.summarise_by_regime(frame, "loss")
    assert {"model", "regime", "n", "loss", "se"} <= set(summary.columns)
    assert (summary["n"] > 0).all()
    assert summary.groupby("model")["n"].sum().eq(300).all()


def test_regime_rows_are_ordered_calm_normal_stressed() -> None:
    """Alphabetical order would put ``stressed`` in the middle of every table."""
    frame = _synthetic_forecasts(n=300)
    frame["loss"] = 1.0
    summary = E.summarise_by_regime(frame, "loss")
    for _, block in summary.groupby("model"):
        assert list(block["regime"]) == list(E.REGIME_ORDER)


def test_regime_summary_of_a_binary_indicator_is_a_rate(scored: pd.DataFrame) -> None:
    """The same function serves losses and coverage: applied to an inside-indicator its
    mean is the empirical coverage of that subsample."""
    summary = E.summarise_by_regime(scored, "inside_95")
    assert ((summary["inside_95"] >= 0.0) & (summary["inside_95"] <= 1.0)).all()
    stressed = summary[(summary["model"] == "garch_mle") & (summary["regime"] == "stressed")]
    assert len(stressed) == 1
    assert stressed["n"].item() > 0


def test_regime_summary_refuses_an_unknown_column() -> None:
    with pytest.raises(KeyError):
        E.summarise_by_regime(_synthetic_forecasts(n=50), "not_a_column")


# --- Structural: calibration never touches the proxy ------------------------------


def test_scrambling_the_proxy_leaves_every_calibration_number_untouched(
    forecasts: pd.DataFrame,
) -> None:
    """The module's central separation, asserted rather than promised.

    Point accuracy is scored against the Parkinson proxy and inherits its D2 scale
    problem. Calibration is scored against observed returns and does not. If a proxy
    value ever leaked into a coverage, PIT or VaR number, the project's headline
    finding would inherit a bias it explicitly claims to be free of -- so the proxy is
    replaced with noise and every calibration table must come back bit-identical.
    """
    rng = np.random.default_rng(SEED)
    scrambled = forecasts.copy()
    scrambled["proxy_var"] = rng.permutation(scrambled["proxy_var"].to_numpy()) * 3.0

    base = E.score_forecasts(forecasts)
    other = E.score_forecasts(scrambled)
    models = B.HEADLINE_MODELS

    for builder in (E.coverage_table, E.var_backtest_table, E.pit_table):
        pd.testing.assert_frame_equal(
            builder(base, models, sample_label="test"),
            builder(other, models, sample_label="test"),
        )
    pd.testing.assert_frame_equal(
        E.decomposition_table(base), E.decomposition_table(other)
    )


def test_scrambling_the_proxy_does_move_the_point_losses(
    forecasts: pd.DataFrame,
) -> None:
    """The negative case for the test above: if nothing moved, it would be asserting
    that the scramble had no effect rather than that calibration is insulated."""
    rng = np.random.default_rng(SEED)
    scrambled = forecasts.copy()
    scrambled["proxy_var"] = rng.permutation(scrambled["proxy_var"].to_numpy()) * 3.0

    base = E.score_forecasts(forecasts)["qlike"].mean()
    other = E.score_forecasts(scrambled)["qlike"].mean()
    assert not np.isclose(base, other)


# --- Result tables ----------------------------------------------------------------


def test_every_table_states_the_sample_it_was_computed_on(scored: pd.DataFrame) -> None:
    """The same statistic on each model's own days and on the four-model common sample
    are two different numbers, both legitimate. A table that does not say which it
    holds cannot be read (D28)."""
    for builder in (E.point_loss_table, E.coverage_table, E.var_backtest_table, E.pit_table):
        table = builder(scored, B.HEADLINE_MODELS, sample_label="own days")
        assert (table["sample"] == "own days").all()
        assert (table["n"] > 0).all()


def test_the_common_sample_restriction_reaches_the_tables(scored: pd.DataFrame) -> None:
    dates = E.common_sample(scored, B.HEADLINE_MODELS)
    table = E.coverage_table(
        scored, B.HEADLINE_MODELS, sample_label="common", dates=dates
    )
    assert (table["n"] == len(dates)).all()

    own = E.coverage_table(scored, B.HEADLINE_MODELS, sample_label="own")
    assert set(own["n"]) == {2134, 2092}


def test_comparison_rows_carry_their_own_pairwise_sample(scored: pd.DataFrame) -> None:
    """A pair of frequentist models is compared on 2,134 days and a pair involving the
    Bayesian model on 2,092. Both are correct; neither may be inferred."""
    table = E.comparison_table(
        scored, (("garch_mle", "ewma"), ("garch_bayes", "garch_mle"))
    )
    assert table.loc[0, "n"] == 2134
    assert table.loc[1, "n"] == 2092
    assert (table["dm_lag_truncation"] >= 0).all()


def test_a_paired_comparison_refuses_to_run_on_misaligned_rows(
    scored: pd.DataFrame,
) -> None:
    """A misalignment would not raise on its own: it would compare model A on one day
    with model B on another and return a plausible number. The pairing is checked."""
    extra = scored[scored["model"] == "garch_mle"].iloc[[0]]
    duplicated = pd.concat([scored, extra], ignore_index=True)
    with pytest.raises(ValueError, match="align date for date"):
        E.comparison_table(duplicated, (("garch_mle", "ewma"),))


def test_the_headline_tables_exclude_the_ablations(scored: pd.DataFrame) -> None:
    """D15: ``garch_mle_normal`` is an ablation, and D29 puts ``garch_bayes_mean`` in
    the same category. Filtering on ``HEADLINE_MODELS`` rather than on whatever happens
    to be in the forecast file is what keeps them out."""
    assert set(B.HEADLINE_MODELS) == {"yesterday", "ewma", "garch_mle", "garch_bayes"}
    table = E.coverage_table(scored, B.HEADLINE_MODELS, sample_label="own")
    assert "garch_mle_normal" not in set(table["model"])
    assert "garch_bayes_mean" not in set(table["model"])


def test_the_decomposition_separates_the_priors_from_parameter_uncertainty(
    scored: pd.DataFrame,
) -> None:
    """The single most important table in the stage (D29).

    ``garch_bayes`` against ``garch_mle`` moves two things at once. The first two
    contrasts move exactly one each, and the third is what the forecast table reports.
    At the 99% level the two causes point in *opposite* directions -- parameter
    uncertainty widens the interval, the priors narrow it -- which is precisely why the
    reported difference must never be attributed to the first alone.
    """
    table = E.decomposition_table(scored)
    overall = table[(table["regime"] == "all") & (table["nominal"] == 0.99)]
    by_contrast = overall.set_index("contrast")["mean_width_ratio"]

    assert by_contrast["parameter uncertainty"] > 1.0
    assert by_contrast["priors"] < 1.0
    assert by_contrast["reported"] < 1.0


def test_the_decomposition_reports_every_regime_with_its_own_n(
    scored: pd.DataFrame,
) -> None:
    table = E.decomposition_table(scored)
    regimes = set(table["regime"])
    assert regimes == {"all", *E.REGIME_ORDER}
    assert (table["n"] > 0).all()
    for contrast, block in table.groupby("contrast"):
        at_99 = block[block["nominal"] == 0.99]
        assert at_99.loc[at_99["regime"] == "all", "n"].item() == sum(
            at_99.loc[at_99["regime"] == r, "n"].item() for r in E.REGIME_ORDER
        )


# --- Regime-conditional tables (Stage 5) ------------------------------------------


def test_the_regime_floor_is_a_stated_constant() -> None:
    """D33. A number this consequential should not be a literal inside a loop."""
    assert E.MIN_REGIME_OBSERVATIONS == 30


def test_a_regime_subsample_below_the_floor_is_refused(scored: pd.DataFrame) -> None:
    """A coverage estimate on twenty days is not a number.

    Printing it beside estimates on eight hundred invites exactly the over-reading the
    regime tables exist to prevent, so it raises rather than appearing with a small
    ``n`` and hoping the reader notices. Never fires on the VIX regimes -- the smallest
    is 341 days -- and guards the further subsetting Stage 6 does.
    """
    thin = scored.copy()
    stressed = thin.index[thin["regime"] == "stressed"]
    thin.loc[stressed[10:], "regime"] = "normal"
    with pytest.raises(ValueError, match="below the 30-day floor"):
        E.regime_coverage_table(thin, ("garch_mle",), sample_label="thin")


def test_every_regime_table_reports_n_and_accounts_for_every_day(
    scored: pd.DataFrame,
) -> None:
    """The regimes partition the sample: no day is scored twice and none is dropped."""
    own_days = int(
        (scored["model"].eq("garch_mle") & scored["exceedance"].notna()).sum()
    )
    var_table = E.regime_var_table(scored, ("garch_mle",), sample_label="own days")
    assert list(var_table["regime"]) == list(E.REGIME_ORDER)
    assert var_table["n"].sum() == own_days

    coverage = E.regime_coverage_table(
        scored, ("garch_mle",), sample_label="own days", levels=(0.95,)
    )
    assert coverage["n"].sum() == own_days


def test_regime_breaches_sum_to_the_full_sample_count(scored: pd.DataFrame) -> None:
    """The regime split re-partitions Stage 4's number rather than recomputing it."""
    full = E.var_backtest_table(scored, ("garch_mle",), sample_label="own days")
    by_regime = E.regime_var_table(scored, ("garch_mle",), sample_label="own days")
    assert by_regime["breaches"].sum() == full["breaches"].iloc[0]


def test_regime_coverage_aggregates_to_the_overall_coverage(
    scored: pd.DataFrame,
) -> None:
    """Weighted by ``n``, the per-regime coverages must return the pooled figure."""
    overall = E.coverage_table(
        scored, ("garch_bayes",), sample_label="own days", levels=(0.95,)
    )
    by_regime = E.regime_coverage_table(
        scored, ("garch_bayes",), sample_label="own days", levels=(0.95,)
    )
    pooled = (by_regime["empirical"] * by_regime["n"]).sum() / by_regime["n"].sum()
    assert pooled == pytest.approx(overall["empirical"].iloc[0])


def test_regime_intervals_bracket_their_point_estimates(scored: pd.DataFrame) -> None:
    for table in (
        E.regime_coverage_table(scored, B.HEADLINE_MODELS, sample_label="own"),
        E.regime_var_table(scored, B.HEADLINE_MODELS, sample_label="own"),
        E.regime_loss_table(scored, B.HEADLINE_MODELS, sample_label="own"),
    ):
        estimate = table["empirical"] if "empirical" in table else (
            table["rate"] if "rate" in table else table["mean"]
        )
        assert (table["ci_lower"] <= estimate).all()
        assert (estimate <= table["ci_upper"]).all()


def test_the_smallest_regime_gets_the_widest_interval(scored: pd.DataFrame) -> None:
    """The point of putting error bars on a crisis subsample.

    Stressed days are 347 of 2,134 and arrive in a handful of long runs, so the
    effective number of independent blocks is smaller still. A difference that fits
    inside that interval is not one this sample can see, and the figure has to show it.
    """
    table = E.regime_var_table(scored, ("garch_mle",), sample_label="own").set_index(
        "regime"
    )
    width = table["ci_upper"] - table["ci_lower"]
    assert width["stressed"] > width["calm"]
    assert width["stressed"] > width["normal"]


def test_covers_nominal_describes_the_interval_rather_than_judging_the_model(
    scored: pd.DataFrame,
) -> None:
    """``covers_nominal`` must be exactly the interval containing the nominal value."""
    coverage = E.regime_coverage_table(scored, B.HEADLINE_MODELS, sample_label="own")
    expected = (coverage["ci_lower"] <= coverage["nominal"]) & (
        coverage["nominal"] <= coverage["ci_upper"]
    )
    assert coverage["covers_nominal"].equals(expected)

    var_table = E.regime_var_table(scored, B.HEADLINE_MODELS, sample_label="own")
    expected = (var_table["ci_lower"] <= var_table["nominal_rate"]) & (
        var_table["nominal_rate"] <= var_table["ci_upper"]
    )
    assert var_table["covers_nominal"].equals(expected)


def test_the_regime_var_table_runs_kupiec_and_not_christoffersen(
    scored: pd.DataFrame,
) -> None:
    """D32, pinned so it cannot be 'completed' by someone tidying up later.

    Kupiec tests a count against a rate and is untroubled by a subsample. Christoffersen
    counts transitions between *consecutive* observations, and consecutive rows of a
    regime subsample can be months apart -- its "yesterday" would be fictitious, and a
    p-value from fictitious transitions is worse than none, because it looks like the
    money test having been run.
    """
    table = E.regime_var_table(scored, B.HEADLINE_MODELS, sample_label="own")
    assert "kupiec_p" in table.columns
    assert not [c for c in table.columns if "independence" in c or "conditional" in c]


def test_regime_tables_carry_the_sample_label(scored: pd.DataFrame) -> None:
    for builder in (E.regime_loss_table, E.regime_coverage_table, E.regime_var_table):
        table = builder(scored, ("garch_mle",), sample_label="own days")
        assert (table["sample"] == "own days").all()


def test_regime_loss_table_rejects_an_unknown_loss(scored: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match="unknown loss"):
        E.regime_loss_table(
            scored, ("garch_mle",), sample_label="own", loss="not_a_loss"
        )


def test_the_regime_label_is_the_lagged_one_the_data_layer_built(
    forecasts: pd.DataFrame,
) -> None:
    """The no-look-ahead statement the regime figure is obliged to make, as a test.

    The regime split conditions on information from day ``t-1``: ``data.assign_vix_regime``
    owns the lag and ``test_data.py`` pins that. What this adds is that the label the
    *evaluation* layer splits on is that same column, carried through the backtest
    unchanged -- so the figure's claim is about the labels actually used, not about a
    function elsewhere that happens to be correct.
    """
    frame_path = Path("data/processed/analysis_frame.csv")
    if not frame_path.exists():  # pragma: no cover - depends on local state
        pytest.skip(f"{frame_path} not present; run `python run_all.py --stage data`")
    frame = pd.read_csv(frame_path, index_col=0, parse_dates=True)

    block = forecasts[forecasts["model"] == "garch_mle"].set_index("date")
    expected = frame.loc[block.index, "regime"]
    assert (block["regime"].to_numpy() == expected.to_numpy()).all()


def test_regime_statistics_ignore_the_volatility_proxy(forecasts: pd.DataFrame) -> None:
    """The Stage 4 separation, re-asserted where the subsampling could have broken it."""
    rng = np.random.default_rng(SEED)
    scrambled = forecasts.copy()
    scrambled["proxy_var"] = rng.permutation(scrambled["proxy_var"].to_numpy()) * 3.0

    base = E.score_forecasts(forecasts)
    other = E.score_forecasts(scrambled)
    for builder in (E.regime_coverage_table, E.regime_var_table):
        pd.testing.assert_frame_equal(
            builder(base, B.HEADLINE_MODELS, sample_label="own"),
            builder(other, B.HEADLINE_MODELS, sample_label="own"),
        )
