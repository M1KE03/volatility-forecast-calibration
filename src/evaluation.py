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

Missing forecasts are the caller's problem, and deliberately so (D28)
---------------------------------------------------------------------
Two Bayesian refits failed their convergence diagnostics and produced no forecasts for
the 21 days each served, so ``garch_bayes`` and ``garch_bayes_mean`` are NaN on 42 of
the 2,134 evaluation days (D19). Every scoring function here **refuses** a non-finite
input rather than dropping it. Dropping silently would let each statistic be computed
on a different, unstated sample, and a coverage table whose rows are not comparable is
worse than one that is missing. ``common_sample`` is the machinery for making that
choice explicitly, and every table this module feeds carries its ``n``.

What is here beyond the original stub list
------------------------------------------
``common_sample`` is the D28 machinery above. ``pit_uniformity_test`` is the KS test the
governing plan asks for alongside the PIT histograms, with its approximate p-value
flagged in the returned object rather than in a comment. ``interval_inside`` and
``interval_width`` expose the per-observation series behind coverage, which the paired
bootstrap and the D29 decomposition both need.

The **result-table builders** in the last section are the other addition. They turn the
forecast table into the six ``eval_*.csv`` tables the report quotes, and they live here
rather than in ``run_all.py`` so they can be tested -- a table that exists only inside a
script cannot have a test asserting that scrambling the volatility proxy leaves every
calibration number in it untouched, and that assertion is the guarantee this docstring
opens with.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from . import data as D

#: Regime labels, in the order every table and figure reports them.
REGIME_ORDER: tuple[str, ...] = D.REGIME_ORDER


def _check_finite(name: str, values: np.ndarray) -> np.ndarray:
    """Validate one input series: 1-D, non-empty, finite everywhere.

    The refusal to drop NaN is a design decision, not strictness for its own sake: see
    the module docstring.
    """
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(array)):
        n_bad = int((~np.isfinite(array)).sum())
        raise ValueError(
            f"{name} contains {n_bad} non-finite value(s). Evaluation will not choose "
            "a sample on the caller's behalf: select a stated common sample first "
            "(see common_sample) and report the n that results (D28)."
        )
    return array


def _check_aligned(
    pairs: tuple[tuple[str, np.ndarray], ...],
) -> tuple[np.ndarray, ...]:
    """Validate several series that must be aligned observation by observation."""
    arrays = tuple(_check_finite(name, values) for name, values in pairs)
    sizes = {array.size for array in arrays}
    if len(sizes) != 1:
        detail = ", ".join(
            f"{name}={array.size}" for (name, _), array in zip(pairs, arrays)
        )
        raise ValueError(f"series must be the same length, got {detail}")
    return arrays


def common_sample(
    forecasts: pd.DataFrame,
    models: tuple[str, ...],
    *,
    columns: tuple[str, ...] = ("variance", "pit", "var_99"),
) -> pd.DatetimeIndex:
    """Dates on which **every** model in ``models`` has a finite value in every column.

    The intersection, not the union. Any statistic compared across models must be
    computed here, and the resulting ``n`` reported alongside it (D28). Per-model
    marginals may instead use each model's own available days — that is a different,
    equally legitimate sample, and the only rule is that the table says which it is.

    Returns a sorted ``DatetimeIndex``.
    """
    missing = sorted(set(models) - set(forecasts["model"].unique()))
    if missing:
        raise KeyError(f"models not present in the forecast table: {missing}")

    dates: pd.Index | None = None
    for model in models:
        block = forecasts.loc[forecasts["model"] == model]
        finite = block[list(columns)].notna().all(axis=1)
        available = pd.Index(block.loc[finite, "date"].unique())
        dates = available if dates is None else dates.intersection(available)

    assert dates is not None  # models is non-empty: KeyError above guarantees it
    return pd.DatetimeIndex(sorted(dates))


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

    This is the normalised form, which is zero when the forecast equals the proxy and
    positive otherwise. It differs from the raw ``r/f + log f`` form by a term in the
    proxy alone, so it leaves every *difference* between models unchanged while making
    the level readable.
    """
    realised, forecast = _check_aligned(
        (("realised_var", realised_var), ("forecast_var", forecast_var))
    )
    if np.any(realised <= 0.0) or np.any(forecast <= 0.0):
        raise ValueError(
            "QLIKE is defined for strictly positive variances only; the ratio's "
            "logarithm is undefined at zero."
        )
    ratio = realised / forecast
    return ratio - np.log(ratio) - 1.0


def mse_variance_loss(realised_var: np.ndarray, forecast_var: np.ndarray) -> np.ndarray:
    """Per-observation squared error on the variance scale: ``(realised - forecast)^2``.

    Secondary metric. Dominated by a handful of extreme days, which is exactly why it
    is secondary rather than primary here.
    """
    realised, forecast = _check_aligned(
        (("realised_var", realised_var), ("forecast_var", forecast_var))
    )
    return (realised - forecast) ** 2


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

    @property
    def inside(self) -> int:
        """Count of observations inside the interval."""
        return self.n - self.n_below - self.n_above


def interval_coverage(
    realised_return: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    nominal: float,
) -> CoverageResult:
    """Empirical coverage of a two-sided predictive interval for the **return**.

    Scored against observed returns, never against the volatility proxy.

    Bounds are treated as inclusive. At continuous predictive distributions the
    distinction has probability zero; it is fixed here so the count is reproducible
    rather than dependent on floating-point ties.
    """
    if not 0.0 < nominal < 1.0:
        raise ValueError(f"nominal must lie in (0, 1), got {nominal}")
    returns, lo, hi = _check_aligned(
        (("realised_return", realised_return), ("lower", lower), ("upper", upper))
    )
    if np.any(hi < lo):
        raise ValueError("upper bound lies below lower bound on at least one day")

    below = returns < lo
    above = returns > hi
    n = int(returns.size)
    n_below = int(below.sum())
    n_above = int(above.sum())
    return CoverageResult(
        nominal=float(nominal),
        empirical=float((n - n_below - n_above) / n),
        n=n,
        n_below=n_below,
        n_above=n_above,
    )


def interval_inside(
    realised_return: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    """Binary indicator of the realised return falling inside the interval.

    The per-observation series behind ``interval_coverage``, needed by
    ``bootstrap.bootstrap_coverage_difference``, which is the project's headline
    uncertainty statement.
    """
    returns, lo, hi = _check_aligned(
        (("realised_return", realised_return), ("lower", lower), ("upper", upper))
    )
    return ((returns >= lo) & (returns <= hi)).astype(float)


def interval_width(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Per-observation interval width.

    Used by the decomposition table (D29), which separates what the priors do to the
    intervals from what integrating over parameter uncertainty does. Width is the
    quantity in which the two effects were first seen to differ in sign, so it is
    reported alongside coverage rather than in place of it.
    """
    lo, hi = _check_aligned((("lower", lower), ("upper", upper)))
    if np.any(hi < lo):
        raise ValueError("upper bound lies below lower bound on at least one day")
    return hi - lo


def pit_values(
    realised_return: np.ndarray,
    predictive_cdf: np.ndarray,
) -> np.ndarray:
    """Probability integral transform values of the realised returns.

    Under a correctly specified predictive distribution these are i.i.d. Uniform(0,1).
    A finer diagnostic than coverage at three fixed levels: it shows *where* in the
    distribution a model is miscalibrated, which is the difference between "the 95%
    interval is too narrow" and "the left tail is too thin".

    ``predictive_cdf`` is each model's own predictive CDF evaluated at the realised
    return, computed inside the backtest loop at forecast time and stored in the
    ``pit`` column. It is not recomputed here: the CDF is model-specific — a Student-t
    for the plug-in models, a Monte Carlo mixture over posterior draws for the
    Bayesian one — and a second implementation living in the evaluation layer could
    drift from the one that actually produced the forecasts. This function validates
    the stored values and pairs them with the returns they belong to.
    """
    returns, cdf = _check_aligned(
        (("realised_return", realised_return), ("predictive_cdf", predictive_cdf))
    )
    del returns  # required for the alignment and finiteness checks only
    if np.any(cdf < 0.0) or np.any(cdf > 1.0):
        raise ValueError(
            "PIT values must lie in [0, 1]; the stored predictive CDF does not."
        )
    return cdf


@dataclass(frozen=True)
class UniformityResult:
    """A Kolmogorov-Smirnov test of PIT uniformity, with its own caveat attached.

    ``p_value_is_approximate`` is always ``True`` here and is stored rather than
    written in a comment so that it survives into any table built from this object.
    The KS test assumes a fully specified distribution; these predictive distributions
    have estimated parameters, re-estimated on a rolling schedule, which makes the
    nominal p-value optimistic. The histogram is the diagnostic; this is a summary of
    it, not an arbiter.
    """

    statistic: float
    p_value: float
    n: int
    p_value_is_approximate: bool = True


def pit_uniformity_test(pit: np.ndarray) -> UniformityResult:
    """Kolmogorov-Smirnov test of the PIT values against Uniform(0, 1)."""
    values = _check_finite("pit", pit)
    result = stats.kstest(values, "uniform")
    return UniformityResult(
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        n=int(values.size),
    )


# --- VaR backtests ----------------------------------------------------------------


def var_exceedances(realised_return: np.ndarray, var_forecast: np.ndarray) -> np.ndarray:
    """Binary exceedance indicator for a one-sided lower-tail VaR.

    1 where the realised return falls below the VaR forecast (decision D5: VaR is the
    loss tail).
    """
    returns, var_level = _check_aligned(
        (("realised_return", realised_return), ("var_forecast", var_forecast))
    )
    return (returns < var_level).astype(float)


def _check_exceedances(exceedances: np.ndarray) -> np.ndarray:
    """Validate a hit sequence: finite, one-dimensional, and binary."""
    values = _check_finite("exceedances", exceedances)
    if not np.all((values == 0.0) | (values == 1.0)):
        raise ValueError("exceedances must contain only 0 and 1")
    return values


def _xlogy(x: float, y: float) -> float:
    """``x * log(y)`` under the convention ``0 * log(0) = 0``.

    Required rather than cosmetic. At the 99% level over ~2,100 days a well-calibrated
    model produces around 21 breaches, and consecutive breaches are rare enough that
    ``n11 = 0`` is a routine outcome; without this convention the independence test
    returns NaN on exactly the models that are behaving best.
    """
    if x == 0.0:
        return 0.0
    return float(x) * float(np.log(y))


def kupiec_pof_test(exceedances: np.ndarray, nominal_rate: float) -> tuple[float, float]:
    """Kupiec unconditional coverage test. Returns ``(statistic, p_value)``.

    Tests only whether the *number* of exceedances matches the nominal rate. It is
    silent on clustering, which is the failure mode that actually matters in a stress
    episode — hence the two tests below.

    The statistic is the likelihood ratio ``-2 log(L(p) / L(pi_hat))``, chi-squared
    with one degree of freedom under the null.
    """
    if not 0.0 < nominal_rate < 1.0:
        raise ValueError(f"nominal_rate must lie in (0, 1), got {nominal_rate}")
    values = _check_exceedances(exceedances)
    n = values.size
    n_hits = int(values.sum())

    log_l_null = _xlogy(n_hits, nominal_rate) + _xlogy(n - n_hits, 1.0 - nominal_rate)
    pi_hat = n_hits / n
    log_l_alt = _xlogy(n_hits, pi_hat) + _xlogy(n - n_hits, 1.0 - pi_hat)

    statistic = -2.0 * (log_l_null - log_l_alt)
    statistic = max(statistic, 0.0)  # guards floating-point noise at pi_hat == p
    return statistic, float(stats.chi2.sf(statistic, df=1))


def christoffersen_independence_test(exceedances: np.ndarray) -> tuple[float, float]:
    """Christoffersen independence test. Returns ``(statistic, p_value)``.

    Tests whether exceedances cluster. Clustered breaches mean the model fails exactly
    when it is most costly to fail, which is central to this project's question about
    whether calibration survives high-volatility regimes.

    A first-order Markov alternative: the null is that the probability of a breach
    tomorrow does not depend on whether one occurred today. Computed on the ``n - 1``
    transitions, so it is one observation shorter than the Kupiec test above — the
    standard convention, and stated here because the two are summed in
    ``conditional_coverage_test``.
    """
    values = _check_exceedances(exceedances)
    if values.size < 2:
        raise ValueError("the independence test needs at least two observations")

    previous, current = values[:-1], values[1:]
    n00 = float(np.sum((previous == 0) & (current == 0)))
    n01 = float(np.sum((previous == 0) & (current == 1)))
    n10 = float(np.sum((previous == 1) & (current == 0)))
    n11 = float(np.sum((previous == 1) & (current == 1)))

    total = n00 + n01 + n10 + n11
    pi = (n01 + n11) / total
    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0

    log_l_null = _xlogy(n00 + n10, 1.0 - pi) + _xlogy(n01 + n11, pi)
    log_l_alt = (
        _xlogy(n00, 1.0 - pi01)
        + _xlogy(n01, pi01)
        + _xlogy(n10, 1.0 - pi11)
        + _xlogy(n11, pi11)
    )

    statistic = -2.0 * (log_l_null - log_l_alt)
    statistic = max(statistic, 0.0)
    return statistic, float(stats.chi2.sf(statistic, df=1))


def conditional_coverage_test(
    exceedances: np.ndarray, nominal_rate: float
) -> tuple[float, float]:
    """Christoffersen joint test of correct rate and independence.

    The statistic is the sum of the Kupiec and independence statistics and is
    chi-squared with two degrees of freedom. The two components are computed on
    samples differing by one observation — Kupiec on all ``n`` days, independence on
    the ``n - 1`` transitions — which is Christoffersen's own construction and is
    noted so a reader is not left to infer it from the code.
    """
    pof_statistic, _ = kupiec_pof_test(exceedances, nominal_rate)
    ind_statistic, _ = christoffersen_independence_test(exceedances)
    statistic = pof_statistic + ind_statistic
    return statistic, float(stats.chi2.sf(statistic, df=2))


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


def newey_west_lag(n: int) -> int:
    """Automatic Bartlett truncation lag: ``floor(4 (n/100)^(2/9))``.

    Decision D30. The textbook one-step-ahead DM test uses no lags at all, on the
    argument that an optimal one-step forecast has a serially uncorrelated error. These
    forecasts are not optimal and their loss differential is visibly autocorrelated —
    variance regimes persist for weeks — so a HAC correction is applied rather than
    assumed away. The rule is fixed here, before any test was run, so no reported
    p-value is the product of a lag chosen after seeing it.
    """
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def diebold_mariano(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    *,
    lag_truncation: int | None = None,
) -> DieboldMarianoResult:
    """Diebold-Mariano test of equal predictive accuracy, HAC-corrected.

    A **positive** statistic means ``loss_a`` exceeds ``loss_b`` on average — model A
    is the worse forecaster. The convention is stated because the test is symmetric in
    everything except this, and a sign error would reverse every conclusion drawn from
    it. A test pins it.

    Small-sample correction from Harvey, Leybourne and Newbold (1997) applied at
    ``h = 1``, and the statistic referred to a ``t`` distribution with ``n - 1`` degrees
    of freedom rather than the normal.

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
    values_a, values_b = _check_aligned(
        (("loss_a", loss_a), ("loss_b", loss_b))
    )
    differential = values_a - values_b
    n = int(differential.size)
    if n < 3:
        raise ValueError(f"the DM test needs at least three observations, got {n}")

    mean_differential = float(differential.mean())
    centred = differential - mean_differential
    gamma_0 = float(np.mean(centred**2))
    if gamma_0 <= 0.0:
        raise ValueError(
            "the loss differential has zero variance: the two forecast series are "
            "identical on this sample, and no test of equal accuracy is meaningful."
        )

    lag = newey_west_lag(n) if lag_truncation is None else int(lag_truncation)
    if lag < 0:
        raise ValueError(f"lag_truncation must be non-negative, got {lag}")
    if lag >= n:
        raise ValueError(f"lag_truncation {lag} is not shorter than the sample {n}")

    long_run_variance = gamma_0
    for j in range(1, lag + 1):
        gamma_j = float(np.mean(centred[j:] * centred[:-j]))
        weight = 1.0 - j / (lag + 1.0)  # Bartlett
        long_run_variance += 2.0 * weight * gamma_j
    if long_run_variance <= 0.0:
        # A Bartlett-weighted sum is guaranteed non-negative in population but can go
        # negative in a finite sample. Falling back on the contemporaneous variance is
        # the conventional repair; it understates the standard error, so the p-value
        # it produces is conservative in the wrong direction and the fallback is
        # visible in the returned lag.
        long_run_variance = gamma_0
        lag = 0

    statistic = mean_differential / np.sqrt(long_run_variance / n)
    correction = np.sqrt((n - 1.0) / n)  # Harvey-Leybourne-Newbold at h = 1
    statistic = float(statistic * correction)
    p_value = float(2.0 * stats.t.sf(abs(statistic), df=n - 1))

    return DieboldMarianoResult(
        statistic=statistic,
        p_value=p_value,
        mean_loss_differential=mean_differential,
        differential_variance=gamma_0,
        n=n,
        lag_truncation=lag,
    )


def summarise_by_regime(
    forecasts: pd.DataFrame,
    loss_column: str,
) -> pd.DataFrame:
    """Aggregate a loss or coverage column by the lagged-VIX regime label.

    Reports the subsample size alongside every statistic. The stressed regime is a
    small fraction of the sample, so its estimates are noisy; any regime table that
    omits ``n`` invites over-reading.

    Grouped by ``model`` as well when that column is present, which it is for anything
    read out of ``forecasts.csv``. Rows are ordered by ``REGIME_ORDER`` rather than
    alphabetically, so ``stressed`` reads last in every table.

    Works for both families of statistic: applied to a loss column the ``mean`` is the
    mean loss, and applied to a binary indicator — interval coverage, a VaR exceedance
    — it is the empirical rate.
    """
    for column in ("regime", loss_column):
        if column not in forecasts.columns:
            raise KeyError(f"forecast table has no column {column!r}")

    keys = ["model", "regime"] if "model" in forecasts.columns else ["regime"]
    frame = forecasts.loc[forecasts[loss_column].notna(), keys + [loss_column]].copy()
    frame["regime"] = pd.Categorical(
        frame["regime"], categories=list(REGIME_ORDER), ordered=True
    )

    grouped = frame.groupby(keys, observed=False)[loss_column]
    summary = grouped.agg(n="count", mean="mean", std="std").reset_index()
    summary["se"] = summary["std"] / np.sqrt(summary["n"].where(summary["n"] > 0))
    summary = summary.rename(columns={"mean": loss_column})
    return summary.sort_values(keys).reset_index(drop=True)


# --- Result tables ----------------------------------------------------------------
#
# The builders below turn the stored forecast table into the tables the report quotes.
# They live here rather than in ``run_all.py`` so that they are importable and testable:
# a table that only exists inside a script cannot have a test asserting that scrambling
# the volatility proxy leaves every calibration number untouched, and that assertion is
# the structural guarantee this module's opening docstring makes.
#
# Every builder takes a ``sample_label`` and writes it into its output. The label is not
# decoration: the same statistic computed on each model's own days and on the four-model
# common sample are two different numbers, both legitimate, and a table that does not say
# which one it holds cannot be read (D28).

#: The two point losses, by the name each carries in the output tables.
LOSS_FUNCTIONS = {"qlike": qlike_loss, "mse": mse_variance_loss}


def level_key(level: float) -> str:
    """Column suffix for a nominal level: ``0.95`` -> ``"95"``."""
    return str(int(round(level * 100)))


def score_forecasts(
    forecasts: pd.DataFrame,
    *,
    levels: tuple[float, ...] = (0.90, 0.95, 0.99),
    var_level: float = 0.99,
) -> pd.DataFrame:
    """Attach per-observation losses, interval indicators and exceedances.

    Adds ``qlike``, ``mse``, ``inside_<level>``, ``width_<level>`` and ``exceedance``,
    leaving a row NaN wherever the model produced no forecast. Elementwise columns may
    carry NaN; the *statistics* built from them may not, which is why every table
    builder below states the sample it used.
    """
    required = {"model", "date", "variance", "proxy_var", "log_return", "var_99"}
    missing = sorted(required - set(forecasts.columns))
    if missing:
        raise KeyError(f"forecast table is missing columns: {missing}")

    scored = forecasts.copy()
    finite = scored["variance"].notna().to_numpy()

    for name, loss_fn in LOSS_FUNCTIONS.items():
        column = np.full(len(scored), np.nan)
        column[finite] = loss_fn(
            scored.loc[finite, "proxy_var"].to_numpy(),
            scored.loc[finite, "variance"].to_numpy(),
        )
        scored[name] = column

    for level in levels:
        key = level_key(level)
        lo = scored.loc[finite, f"lo_{key}"].to_numpy()
        hi = scored.loc[finite, f"hi_{key}"].to_numpy()
        returns = scored.loc[finite, "log_return"].to_numpy()

        inside = np.full(len(scored), np.nan)
        inside[finite] = interval_inside(returns, lo, hi)
        scored[f"inside_{key}"] = inside

        width = np.full(len(scored), np.nan)
        width[finite] = interval_width(lo, hi)
        scored[f"width_{key}"] = width

    exceedance = np.full(len(scored), np.nan)
    exceedance[finite] = var_exceedances(
        scored.loc[finite, "log_return"].to_numpy(),
        scored.loc[finite, f"var_{level_key(var_level)}"].to_numpy(),
    )
    scored["exceedance"] = exceedance
    return scored


def _block(
    scored: pd.DataFrame, model: str, dates: pd.DatetimeIndex | None
) -> pd.DataFrame:
    """One model's rows, date-ordered, optionally restricted to a stated sample."""
    block = scored.loc[scored["model"] == model]
    if dates is not None:
        block = block.loc[block["date"].isin(dates)]
    return block.sort_values("date")


def _paired_blocks(
    scored: pd.DataFrame, model_a: str, model_b: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two models' rows on their common sample, checked to be aligned date for date.

    Every paired statistic downstream -- the DM test, both bootstrap intervals, the
    width ratios -- is computed on positionally aligned arrays. A misalignment would
    not raise: it would silently compare model A on one day with model B on another and
    return a plausible number. The check costs nothing and removes the failure mode.
    """
    dates = common_sample(scored, (model_a, model_b))
    block_a = _block(scored, model_a, dates).reset_index(drop=True)
    block_b = _block(scored, model_b, dates).reset_index(drop=True)
    if not block_a["date"].equals(block_b["date"]):
        raise ValueError(
            f"{model_a} and {model_b} did not align date for date on their common "
            "sample; the forecast table has duplicate or missing rows."
        )
    return block_a, block_b


def point_loss_table(
    scored: pd.DataFrame,
    models: tuple[str, ...],
    *,
    sample_label: str,
    dates: pd.DatetimeIndex | None = None,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Mean QLIKE and MSE per model, each with a block-bootstrap interval.

    The interval, not the ranking, is the reportable object: a lower mean loss in one
    sample is not evidence that a model forecasts better.
    """
    from . import bootstrap as BS

    rows = []
    for model in models:
        block = _block(scored, model, dates)
        for loss in LOSS_FUNCTIONS:
            values = block.loc[block[loss].notna(), loss].to_numpy()
            ci = BS.bootstrap_mean(values, confidence=confidence)
            rows.append(
                {
                    "sample": sample_label,
                    "model": model,
                    "loss": loss,
                    "n": int(values.size),
                    "mean": ci.point_estimate,
                    "ci_lower": ci.lower,
                    "ci_upper": ci.upper,
                    "confidence": confidence,
                }
            )
    return pd.DataFrame(rows)


def coverage_table(
    scored: pd.DataFrame,
    models: tuple[str, ...],
    *,
    sample_label: str,
    dates: pd.DatetimeIndex | None = None,
    levels: tuple[float, ...] = (0.90, 0.95, 0.99),
) -> pd.DataFrame:
    """Empirical against nominal coverage per model and level, with both tails split.

    ``n_below`` and ``n_above`` are reported separately because a model can hit its
    nominal coverage overall while putting its breaches almost entirely in one tail,
    and for a risk model the lower tail is the one that costs money.
    """
    rows = []
    for model in models:
        block = _block(scored, model, dates)
        finite = block["variance"].notna()
        for level in levels:
            key = level_key(level)
            result = interval_coverage(
                block.loc[finite, "log_return"].to_numpy(),
                block.loc[finite, f"lo_{key}"].to_numpy(),
                block.loc[finite, f"hi_{key}"].to_numpy(),
                level,
            )
            rows.append(
                {
                    "sample": sample_label,
                    "model": model,
                    "nominal": result.nominal,
                    "empirical": result.empirical,
                    "n": result.n,
                    "n_below": result.n_below,
                    "n_above": result.n_above,
                    "mean_width": float(block.loc[finite, f"width_{key}"].mean()),
                }
            )
    return pd.DataFrame(rows)


def var_backtest_table(
    scored: pd.DataFrame,
    models: tuple[str, ...],
    *,
    sample_label: str,
    dates: pd.DatetimeIndex | None = None,
    var_level: float = 0.99,
) -> pd.DataFrame:
    """Kupiec, Christoffersen independence and conditional coverage on the VaR hits.

    Christoffersen's independence test is the one to read first. Correct average
    coverage is compatible with every breach arriving in the same fortnight, and a
    model that fails only during stress episodes fails exactly when its output is
    being used.
    """
    nominal_rate = 1.0 - var_level
    rows = []
    for model in models:
        block = _block(scored, model, dates)
        hits = block.loc[block["exceedance"].notna(), "exceedance"].to_numpy()
        pof_stat, pof_p = kupiec_pof_test(hits, nominal_rate)
        ind_stat, ind_p = christoffersen_independence_test(hits)
        cc_stat, cc_p = conditional_coverage_test(hits, nominal_rate)
        rows.append(
            {
                "sample": sample_label,
                "model": model,
                "n": int(hits.size),
                "breaches": int(hits.sum()),
                "rate": float(hits.mean()),
                "nominal_rate": nominal_rate,
                "kupiec_stat": pof_stat,
                "kupiec_p": pof_p,
                "independence_stat": ind_stat,
                "independence_p": ind_p,
                "conditional_coverage_stat": cc_stat,
                "conditional_coverage_p": cc_p,
            }
        )
    return pd.DataFrame(rows)


def pit_table(
    scored: pd.DataFrame,
    models: tuple[str, ...],
    *,
    sample_label: str,
    dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """KS test of PIT uniformity per model, carrying its own approximation flag."""
    rows = []
    for model in models:
        block = _block(scored, model, dates)
        available = block["pit"].notna()
        values = pit_values(
            block.loc[available, "log_return"].to_numpy(),
            block.loc[available, "pit"].to_numpy(),
        )
        result = pit_uniformity_test(values)
        rows.append(
            {
                "sample": sample_label,
                "model": model,
                "n": result.n,
                "ks_stat": result.statistic,
                "ks_p": result.p_value,
                "p_value_is_approximate": result.p_value_is_approximate,
                "mean_pit": float(values.mean()),
            }
        )
    return pd.DataFrame(rows)


def comparison_table(
    scored: pd.DataFrame,
    pairs: tuple[tuple[str, str], ...],
    *,
    loss: str = "qlike",
    levels: tuple[float, ...] = (0.90, 0.95, 0.99),
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Pairwise comparisons, each on the intersection of the two models' own days.

    One row per pair, carrying the DM test, the block-bootstrap interval on the same
    loss differential, and the bootstrap interval on each coverage difference. The
    sample size is a column rather than a footnote: pairs involving a Bayesian model
    are compared on 2,092 days and pairs that do not on 2,134, and no row should have
    to be cross-referenced to find out which (D28).

    The bootstrap intervals, not the DM p-values, carry the conclusions -- see the
    caveats on ``diebold_mariano``. Both are reported so a reader can see they agree,
    or that they do not.
    """
    from . import bootstrap as BS

    if loss not in LOSS_FUNCTIONS:
        raise KeyError(
            f"unknown loss {loss!r}; expected one of {sorted(LOSS_FUNCTIONS)}"
        )

    rows = []
    for model_a, model_b in pairs:
        block_a, block_b = _paired_blocks(scored, model_a, model_b)

        loss_a = block_a[loss].to_numpy()
        loss_b = block_b[loss].to_numpy()
        dm = diebold_mariano(loss_a, loss_b)
        differential = BS.bootstrap_loss_differential(
            loss_a, loss_b, confidence=confidence
        )

        row = {
            "model_a": model_a,
            "model_b": model_b,
            "loss": loss,
            "n": dm.n,
            "mean_loss_a": float(loss_a.mean()),
            "mean_loss_b": float(loss_b.mean()),
            "mean_differential": dm.mean_loss_differential,
            "differential_variance": dm.differential_variance,
            "dm_statistic": dm.statistic,
            "dm_p": dm.p_value,
            "dm_lag_truncation": dm.lag_truncation,
            "boot_lower": differential.lower,
            "boot_upper": differential.upper,
            "boot_excludes_zero": differential.excludes_zero,
        }
        for level in levels:
            key = level_key(level)
            coverage = BS.bootstrap_coverage_difference(
                block_a[f"inside_{key}"].to_numpy(),
                block_b[f"inside_{key}"].to_numpy(),
                confidence=confidence,
            )
            row[f"coverage_diff_{key}"] = coverage.point_estimate
            row[f"coverage_diff_{key}_lower"] = coverage.lower
            row[f"coverage_diff_{key}_upper"] = coverage.upper
        rows.append(row)
    return pd.DataFrame(rows)


#: The three contrasts the interval comparison must be reported through (D29).
#:
#: ``garch_bayes`` against ``garch_mle`` is what the forecast table holds and what a
#: reader will reach for, but it moves two things at once. The first two rows separate
#: them; the third is the reported difference, kept in the table so that it is visibly
#: the composition of the other two rather than a fourth independent fact.
DECOMPOSITION_CONTRASTS: tuple[tuple[str, str, str], ...] = (
    ("parameter uncertainty", "garch_bayes", "garch_bayes_mean"),
    ("priors", "garch_bayes_mean", "garch_mle"),
    ("reported", "garch_bayes", "garch_mle"),
)


def decomposition_table(
    scored: pd.DataFrame,
    *,
    contrasts: tuple[tuple[str, str, str], ...] = DECOMPOSITION_CONTRASTS,
    levels: tuple[float, ...] = (0.90, 0.95, 0.99),
    by_regime: bool = True,
) -> pd.DataFrame:
    """Split the frequentist-Bayesian interval difference into its two causes.

    **No statement about parameter uncertainty may be sourced from anywhere else.**
    The comparison the project was designed around -- identical likelihood, plug-in
    against posterior predictive, therefore one cause -- holds against a plug-in at the
    *posterior mean*. Against the plug-in at the MLE, which is what ``forecasts.csv``
    holds, the frozen D4 priors also move the point estimate, and at the 99% level they
    move it further and in the opposite direction (research_log.md 1.13). This table is
    the machinery that keeps the two apart.

    Reported as the mean of the per-day width ratios, which is the quantity 1.13 states,
    with the coverage of each side alongside so that a width change is never read as a
    calibration change without checking.
    """
    rows = []
    for name, model_a, model_b in contrasts:
        block_a, block_b = _paired_blocks(scored, model_a, model_b)
        regimes = ["all"] + (list(REGIME_ORDER) if by_regime else [])

        for regime in regimes:
            mask = (
                np.ones(len(block_a), dtype=bool)
                if regime == "all"
                else (block_a["regime"] == regime).to_numpy()
            )
            for level in levels:
                key = level_key(level)
                ratio = (
                    block_a.loc[mask, f"width_{key}"].to_numpy()
                    / block_b.loc[mask, f"width_{key}"].to_numpy()
                )
                rows.append(
                    {
                        "contrast": name,
                        "model_a": model_a,
                        "model_b": model_b,
                        "regime": regime,
                        "nominal": level,
                        "n": int(mask.sum()),
                        "mean_width_ratio": float(ratio.mean()),
                        "coverage_a": float(block_a.loc[mask, f"inside_{key}"].mean()),
                        "coverage_b": float(block_b.loc[mask, f"inside_{key}"].mean()),
                    }
                )
    return pd.DataFrame(rows)
