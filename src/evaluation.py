"""Forecast evaluation: point losses, interval calibration, VaR backtests, comparisons.

Two families of question, scored against two different targets. Keeping them apart is
essential to the project's argument:

**Point accuracy** — scored against the Parkinson proxy.
    ``qlike_loss``, ``mse_variance_loss``. Subject to open item D2: the proxy measures
    intraday variance while the models forecast close-to-close variance, so it is
    biased low and QLIKE's proxy-robustness assumption is violated. Comparisons are
    within-proxy and must be reported as such.

**Calibration** — scored against observed returns. This is the headline.
    ``interval_coverage``, ``kupiec_pof_test``, ``christoffersen_independence_test``,
    ``conditional_coverage_test``, ``pit_values``. These are **unaffected** by the D2
    proxy problem, because they never touch the proxy.

Interpretation discipline
-------------------------
A lower loss in one sample is not evidence that a model is better. Every headline
comparison must be accompanied by an interval from ``bootstrap`` and by the caveats in
``diebold_mariano``'s docstring. Regime-conditional results are computed on subsamples
selected by the *lagged* VIX label, so the conditioning information was available when
the forecast was made — but the subsamples are small in the stressed regime and the
tests are correspondingly underpowered.

All stubs. See research_log.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# --- Point losses -----------------------------------------------------------------


def qlike_loss(realised_var: np.ndarray, forecast_var: np.ndarray) -> np.ndarray:
    """Per-observation QLIKE loss: ``realised/forecast - log(realised/forecast) - 1``.

    Primary point-loss metric. Asymmetric, penalising under-forecast variance more
    heavily than over-forecast, and robust to *noise* in the volatility proxy — but
    that robustness assumes the proxy is **conditionally unbiased** for the forecast
    target. The Parkinson proxy is not (open item D2). Read that entry before quoting
    a QLIKE number.

    Returns the per-observation loss, not its mean, because the loss series is needed
    for Diebold-Mariano and for the block bootstrap.
    """
    raise NotImplementedError("qlike_loss")


def mse_variance_loss(realised_var: np.ndarray, forecast_var: np.ndarray) -> np.ndarray:
    """Per-observation squared error on the variance scale: ``(realised - forecast)^2``.

    Secondary metric. Dominated by a handful of extreme days, which is exactly why it
    is secondary rather than primary here.
    """
    raise NotImplementedError("mse_variance_loss")


# --- Interval calibration ---------------------------------------------------------


@dataclass(frozen=True)
class CoverageResult:
    """Empirical coverage of one model's intervals at one nominal level.

    Attributes
    ----------
    nominal:
        The nominal level, e.g. 0.95.
    empirical:
        Fraction of observed returns falling inside the interval.
    n:
        Number of observations scored.
    n_below / n_above:
        Breaches of the lower and upper bound separately. A model can hit its nominal
        coverage overall while badly misallocating breaches between the tails, which
        for a volatility model is a substantive failure and not a rounding detail.
    """

    nominal: float
    empirical: float
    n: int
    n_below: int
    n_above: int


def interval_coverage(
    realised_return: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    nominal: float,
) -> CoverageResult:
    """Empirical coverage of a two-sided predictive interval for the **return**.

    Scored against observed returns, never against the volatility proxy.
    """
    raise NotImplementedError("interval_coverage")


def pit_values(
    realised_return: np.ndarray,
    predictive_cdf: np.ndarray,
) -> np.ndarray:
    """Probability integral transform values of the realised returns.

    Under a correctly specified predictive distribution these are i.i.d. Uniform(0,1).
    A finer diagnostic than coverage at three fixed levels: it shows *where* in the
    distribution a model is miscalibrated, which is the difference between "the 95%
    interval is too narrow" and "the left tail is too thin".
    """
    raise NotImplementedError("pit_values")


# --- VaR backtests ----------------------------------------------------------------


def var_exceedances(realised_return: np.ndarray, var_forecast: np.ndarray) -> np.ndarray:
    """Binary exceedance indicator for a one-sided lower-tail VaR.

    1 where the realised return falls below the VaR forecast (decision D5: VaR is the
    loss tail).
    """
    raise NotImplementedError("var_exceedances")


def kupiec_pof_test(exceedances: np.ndarray, nominal_rate: float) -> tuple[float, float]:
    """Kupiec unconditional coverage test. Returns ``(statistic, p_value)``.

    Tests only whether the *number* of exceedances matches the nominal rate. It is
    silent on clustering, which is the failure mode that actually matters in a stress
    episode — hence the two tests below.
    """
    raise NotImplementedError("kupiec_pof_test")


def christoffersen_independence_test(exceedances: np.ndarray) -> tuple[float, float]:
    """Christoffersen independence test. Returns ``(statistic, p_value)``.

    Tests whether exceedances cluster. Clustered breaches mean the model fails exactly
    when it is most costly to fail, which is central to this project's question about
    whether calibration survives high-volatility regimes.
    """
    raise NotImplementedError("christoffersen_independence_test")


def conditional_coverage_test(
    exceedances: np.ndarray, nominal_rate: float
) -> tuple[float, float]:
    """Christoffersen joint test of correct rate and independence."""
    raise NotImplementedError("conditional_coverage_test")


# --- Model comparison -------------------------------------------------------------


@dataclass(frozen=True)
class DieboldMarianoResult:
    """Outcome of a Diebold-Mariano comparison, with its diagnostics attached.

    ``mean_loss_differential`` and ``differential_variance`` are stored so a reader can
    see for themselves whether the test was applied to a near-degenerate differential.
    """

    statistic: float
    p_value: float
    mean_loss_differential: float
    differential_variance: float
    n: int
    lag_truncation: int


def diebold_mariano(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    *,
    lag_truncation: int | None = None,
) -> DieboldMarianoResult:
    """Diebold-Mariano test of equal predictive accuracy, HAC-corrected.

    Caveats that must accompany any reported result — these are not boilerplate:

    1. **Estimated parameters.** DM's standard asymptotics assume the forecasts are not
       functions of estimated parameters. Here they are, and they are re-estimated on a
       rolling schedule. West (1996) and Giacomini-White (2006) address this; the
       nominal p-value should be read as indicative.
    2. **Near-degenerate differential.** The frequentist and Bayesian models share a
       likelihood and differ only by integration over the posterior, so their loss
       differential is small and its variance may approach zero. DM is badly sized in
       that regime. Inspect ``differential_variance`` before quoting ``p_value``.
    3. **Autocorrelation.** The differential is serially correlated; the HAC correction
       is necessary but not sufficient at short samples.
    4. **Multiplicity.** Several pairwise comparisons across several regimes and levels
       are performed. Unadjusted p-values overstate significance.

    The block bootstrap in ``bootstrap`` is the more trustworthy uncertainty statement
    here and should carry the weight of the conclusion.
    """
    raise NotImplementedError("diebold_mariano")


def summarise_by_regime(
    forecasts: pd.DataFrame,
    loss_column: str,
) -> pd.DataFrame:
    """Aggregate a loss or coverage column by the lagged-VIX regime label.

    Reports the subsample size alongside every statistic. The stressed regime is a
    small fraction of the sample, so its estimates are noisy; any regime table that
    omits ``n`` invites over-reading.
    """
    raise NotImplementedError("summarise_by_regime")
