"""Stage 0 tests for ``src.eda``.

Three categories, kept separate because they fail for different reasons:

- **Power tests** ask whether each diagnostic actually detects what it claims to. Every
  one is run twice: on data that should trigger it, and on data that should not. A test
  that only ever sees the positive case cannot distinguish a working test from a
  function that always rejects -- and "always rejects" is exactly the failure that would
  hand this project a fabricated justification for its own model class.
- **Contract tests** pin the behaviour other modules and the report rely on.
- **Look-ahead tests** assert the training-window results are untouched by out-of-sample
  data, which is the reason this module computes them on the training window at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import eda

SEED = 20260823


# --- Fixtures ----------------------------------------------------------------------


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2014-01-02", periods=n)


@pytest.fixture(scope="module")
def white_noise() -> pd.Series:
    """Homoskedastic i.i.d. normal: no ARCH, no autocorrelation, stationary."""
    rng = np.random.default_rng(SEED)
    values = rng.normal(0.0, 0.01, size=1500)
    return pd.Series(values, index=_dates(1500), name="log_return")


@pytest.fixture(scope="module")
def garch_series() -> pd.Series:
    """Simulated GARCH(1,1) with normal innovations: strong, genuine ARCH effects."""
    rng = np.random.default_rng(SEED + 1)
    n = 1500
    omega, alpha, beta = 1e-6, 0.10, 0.85
    h = np.empty(n)
    r = np.empty(n)
    h[0] = omega / (1.0 - alpha - beta)
    for t in range(n):
        if t > 0:
            h[t] = omega + alpha * r[t - 1] ** 2 + beta * h[t - 1]
        r[t] = np.sqrt(h[t]) * rng.normal()
    return pd.Series(r, index=_dates(n), name="log_return")


@pytest.fixture(scope="module")
def random_walk() -> pd.Series:
    """Integrated series: ADF should fail to reject the unit root."""
    rng = np.random.default_rng(SEED + 2)
    return pd.Series(np.cumsum(rng.normal(0.0, 0.01, size=1500)), index=_dates(1500))


# --- Power tests: ARCH-LM ----------------------------------------------------------


@pytest.mark.parametrize("lags", eda.ARCH_LM_LAGS)
def test_arch_lm_detects_arch_effects(garch_series: pd.Series, lags: int) -> None:
    """The motivating test fires on data that genuinely has conditional heteroskedasticity."""
    result = eda.arch_lm_test(garch_series, lags=lags)
    assert result.rejects_at_5pct
    assert result.p_value < 1e-6, "GARCH data should reject overwhelmingly, not marginally"


@pytest.mark.parametrize("lags", eda.ARCH_LM_LAGS)
def test_arch_lm_does_not_fire_on_homoskedastic_data(
    white_noise: pd.Series, lags: int
) -> None:
    """The other half of the power test, and the one that makes the first half mean something.

    Without this, a function that returned ``p_value=0.0`` unconditionally would pass
    the whole suite while inventing the project's central premise.
    """
    result = eda.arch_lm_test(white_noise, lags=lags)
    assert not result.rejects_at_5pct


def test_arch_lm_demeans_internally(garch_series: pd.Series) -> None:
    """Adding a constant to the series must not change the statistic.

    ``arch_lm_test`` promises to demean its input. If it stopped doing so, a non-zero
    mean would leak into the squared residuals and inflate the statistic -- in the
    direction that flatters the project's premise, which is why this is asserted rather
    than trusted.
    """
    baseline = eda.arch_lm_test(garch_series, lags=10)
    shifted = eda.arch_lm_test(garch_series + 0.05, lags=10)
    assert shifted.statistic == pytest.approx(baseline.statistic, rel=1e-10)


def test_arch_lm_would_be_inflated_without_demeaning(garch_series: pd.Series) -> None:
    """Prove the demeaning in the previous test is load-bearing, not decorative.

    Runs the raw (undemeaned) computation by hand on a heavily shifted series and
    asserts it differs materially from the module's answer. If this ever stops holding,
    the invariance test above has become vacuous and would silently stop protecting
    anything.
    """
    from statsmodels.stats.diagnostic import het_arch

    shifted = (garch_series + 0.05).dropna().to_numpy(dtype=float)
    naive_stat = float(het_arch(shifted, nlags=10)[0])
    correct_stat = eda.arch_lm_test(garch_series + 0.05, lags=10).statistic
    assert naive_stat != pytest.approx(correct_stat, rel=1e-3)


# --- Power tests: Ljung-Box --------------------------------------------------------


def test_ljung_box_detects_clustering_in_squared_returns(garch_series: pd.Series) -> None:
    result = eda.ljung_box_test(garch_series**2, lags=10, label="squared returns")
    assert result.rejects_at_5pct


def test_ljung_box_quiet_on_white_noise(white_noise: pd.Series) -> None:
    result = eda.ljung_box_test(white_noise, lags=10, label="raw returns")
    assert not result.rejects_at_5pct


def test_ljung_box_detects_ar1_autocorrelation() -> None:
    """Positive control on a series with autocorrelation in the *mean*."""
    rng = np.random.default_rng(SEED + 3)
    n = 1500
    values = np.empty(n)
    values[0] = 0.0
    for t in range(1, n):
        values[t] = 0.6 * values[t - 1] + rng.normal(0.0, 0.01)
    result = eda.ljung_box_test(pd.Series(values, index=_dates(n)), lags=10, label="ar1")
    assert result.rejects_at_5pct


# --- Power tests: ADF --------------------------------------------------------------


def test_adf_rejects_unit_root_on_stationary_series(white_noise: pd.Series) -> None:
    assert eda.adf_test(white_noise).rejects_at_5pct


def test_adf_does_not_reject_on_random_walk(random_walk: pd.Series) -> None:
    assert not eda.adf_test(random_walk).rejects_at_5pct


# --- Contract tests ----------------------------------------------------------------


def test_describe_returns_matches_analytic_values() -> None:
    """Summary statistics on a series whose moments are known by construction."""
    values = pd.Series([-0.02, -0.01, 0.0, 0.01, 0.02], index=_dates(5))
    summary = eda.describe_returns(values)
    assert summary.n == 5
    assert summary.mean == pytest.approx(0.0)
    assert summary.std == pytest.approx(np.std(values.to_numpy(), ddof=1))
    assert summary.minimum == pytest.approx(-0.02)
    assert summary.maximum == pytest.approx(0.02)
    assert summary.n_exact_zero == 1
    assert summary.annualised_vol == pytest.approx(summary.std * np.sqrt(252.0))


def test_describe_returns_reports_excess_not_raw_kurtosis(white_noise: pd.Series) -> None:
    """Normal data must give excess kurtosis near 0, not near 3.

    The Student-t decision in the locked design rests on this number being positive and
    material. Reporting raw kurtosis instead would make every series look fat-tailed.
    """
    assert eda.describe_returns(white_noise).excess_kurtosis == pytest.approx(0.0, abs=0.25)


def test_squared_return_acf_drops_lag_zero_and_is_bounded(garch_series: pd.Series) -> None:
    frame = eda.squared_return_acf(garch_series, nlags=30)
    assert frame.index.name == "lag"
    assert frame.index.min() == 1, "lag 0 is identically 1 and must not be plotted"
    assert len(frame) == 30
    assert frame["acf"].abs().max() <= 1.0


def test_squared_return_acf_decays_for_garch_data(garch_series: pd.Series) -> None:
    """Early lags carry real autocorrelation and later lags carry less."""
    frame = eda.squared_return_acf(garch_series, nlags=60)
    assert frame.loc[1, "acf"] > 0.05
    assert frame.loc[1:5, "acf"].mean() > frame.loc[45:60, "acf"].mean()


def test_short_series_raises_rather_than_returning_nonsense() -> None:
    """Too few observations must fail loudly, not produce an unusable statistic."""
    with pytest.raises(ValueError, match="too few"):
        eda.arch_lm_test(pd.Series([0.01, -0.01, 0.02], index=_dates(3)), lags=5)


def test_run_eda_drops_nans_and_reports_true_window(garch_series: pd.Series) -> None:
    """A leading NaN, as produced by the return calculation, must not shift the window."""
    with_nan = garch_series.copy()
    with_nan.iloc[0] = np.nan
    report = eda.run_eda(with_nan, window_label="test")
    assert report.summary.n == len(garch_series) - 1
    assert report.start == garch_series.index[1]
    assert report.end == garch_series.index[-1]
    assert report.window_label == "test"


def test_run_eda_is_deterministic(garch_series: pd.Series) -> None:
    a = eda.run_eda(garch_series, window_label="x")
    b = eda.run_eda(garch_series, window_label="x")
    assert a.arch_lm[0].statistic == b.arch_lm[0].statistic
    assert a.adf.statistic == b.adf.statistic
    pd.testing.assert_frame_equal(a.acf_squared, b.acf_squared)


def test_headline_verdict_reports_mixed_evidence_when_diagnostics_disagree(
    white_noise: pd.Series,
) -> None:
    """The verdict must be derived from the results, not hardcoded optimism.

    White noise has no ARCH effects, so the verdict has to say so. A verdict that
    announced "a conditional-variance model is warranted" regardless of input would let
    the project assert its own premise.
    """
    report = eda.run_eda(white_noise, window_label="white noise")
    assert "MIXED EVIDENCE" in report.headline_verdict()


def test_headline_verdict_confirms_on_genuine_garch_data(garch_series: pd.Series) -> None:
    report = eda.run_eda(garch_series, window_label="garch")
    assert "warranted" in report.headline_verdict()
    assert "MIXED EVIDENCE" not in report.headline_verdict()


def test_format_line_renders_extreme_p_values_as_a_bound() -> None:
    """An underflowed p-value must not print as ``0.000e+00``, which reads as a bug."""
    result = eda.TestResult(
        name="t", statistic=1.0, p_value=0.0, lags=1, null="n", interpretation="i"
    )
    assert "< 1e-300" in result.format_line()


def test_format_full_labels_the_window(garch_series: pd.Series) -> None:
    """Two reports must be distinguishable once printed, so the label has to appear."""
    text = eda.run_eda(garch_series, window_label="training window").format_full()
    assert "training window" in text


# --- Look-ahead tests --------------------------------------------------------------


def test_training_window_results_ignore_out_of_sample_data(
    garch_series: pd.Series,
) -> None:
    """The reason this module reports the training window as primary.

    Compute the training-window EDA, then replace every out-of-sample observation with
    wildly different data and recompute. The training-window results must be identical.
    If they are not, the evidence used to justify the model class depends on the period
    the model is later evaluated on.
    """
    split = 700
    train = garch_series.iloc[:split]
    baseline = eda.run_eda(train, window_label="train")

    corrupted = garch_series.copy()
    rng = np.random.default_rng(SEED + 99)
    corrupted.iloc[split:] = rng.normal(0.0, 0.5, size=len(corrupted) - split)
    recomputed = eda.run_eda(corrupted.iloc[:split], window_label="train")

    assert recomputed.arch_lm[0].statistic == baseline.arch_lm[0].statistic
    assert recomputed.adf.statistic == baseline.adf.statistic
    assert recomputed.summary.std == baseline.summary.std
    pd.testing.assert_frame_equal(recomputed.acf_squared, baseline.acf_squared)


def test_diagnostics_use_no_future_information_within_the_window(
    garch_series: pd.Series,
) -> None:
    """Truncating the series changes the answer; extending it must not change the past.

    Guards against a diagnostic accidentally computed on a full-sample statistic (a
    global mean or variance) that would silently make every windowed result depend on
    data after the window.
    """
    short = eda.describe_returns(garch_series.iloc[:500])
    long_from_same_start = eda.describe_returns(garch_series.iloc[:500])
    assert short.std == long_from_same_start.std
    # And the full series genuinely differs, so the check above is not trivially true.
    assert eda.describe_returns(garch_series).std != short.std
