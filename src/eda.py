"""Exploratory analysis: the evidence that motivates a conditional-variance model.

This module answers one question with tests rather than assertion: *does SPY's daily
return series actually exhibit the conditional heteroskedasticity that GARCH exists to
model?* The answer is the premise of the whole project, so it is established formally
and recorded, not assumed from a plot that looks clustered.

Four diagnostics, each with a distinct job:

``arch_lm_test``
    Engle's LM test for ARCH effects. **The motivating test.** A rejection says squared
    residuals are autocorrelated -- variance is predictable from its own past, which is
    precisely what GARCH parameterises. Without a rejection here the project has no
    subject.
``ljung_box_test``
    Applied to *squared* returns for the same purpose from a different angle, and to
    *raw* returns to show the first moment is close to unpredictable. The contrast is
    the point: little structure in the mean, abundant structure in the variance.
``adf_test``
    Stationarity of the return series. Expected to reject the unit root decisively and
    to be uninteresting -- one honest line in the report.
``squared_return_acf``
    The picture behind the two tests above: slow decay in the ACF of squared returns.

Which window these are computed on
----------------------------------
**Primary results use the training window only** (``TRAIN_START``..``TRAIN_END``).

This is stricter than the governing plan requires, and deliberately so. Model-class
choice is itself a modelling decision. Justifying "we use GARCH" with a test computed
over a sample that includes the 2,134 out-of-sample days would let the evaluation period
argue for the model that is later evaluated on it -- a mild look-ahead, but the exact
species this project is about detecting in others. The training window is what a
forecaster standing on 2016-12-30 actually had.

Full-sample values are also computed, clearly labelled as descriptive context only. They
are reported so a reader can see they agree; they never justify a choice. Nothing in the
locked design (research_log.md 1.1) may be revised in response to either set -- the
decisions were fixed before this module existed, which is what makes them credible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import acf, adfuller

# --- Locked test settings ----------------------------------------------------------
# Fixed here rather than passed in at call sites, so that no reported p-value is the
# product of a lag choice made after seeing a result.

ARCH_LM_LAGS: tuple[int, ...] = (5, 10, 22)
LJUNG_BOX_LAGS: tuple[int, ...] = (5, 10, 22)
ACF_LAGS: int = 60

#: One trading month. Used for the ``22``-lag variants above.
TRADING_MONTH: int = 22


@dataclass(frozen=True)
class TestResult:
    """One hypothesis test, carrying enough context to be reported without a lookup.

    ``null`` and ``interpretation`` are stored rather than left to the caller because a
    bare statistic and p-value invite the reader to guess the direction of the test, and
    for ARCH-LM the intuitive guess is backwards: rejection is the *supportive* outcome
    for this project, not the adverse one.
    """

    name: str
    statistic: float
    p_value: float
    lags: int | None
    null: str
    interpretation: str
    extra: dict[str, float] = field(default_factory=dict)

    @property
    def rejects_at_5pct(self) -> bool:
        return self.p_value < 0.05

    def format_line(self) -> str:
        lag_part = f"  lags={self.lags:<3}" if self.lags is not None else " " * 10
        # p-values below the float64 subnormal floor print as 0.0, which reads as a
        # bug rather than as overwhelming evidence. Render the bound instead.
        p_part = "< 1e-300" if self.p_value < 1e-300 else f"{self.p_value:.3e}"
        verdict = "REJECT " if self.rejects_at_5pct else "no rej."
        return f"{self.name:<34}{lag_part}  stat={self.statistic:>12.3f}  p={p_part:>9}  {verdict}"


@dataclass(frozen=True)
class ReturnSummary:
    """Descriptive statistics for a return series, on the scale returns are stored in."""

    n: int
    mean: float
    std: float
    skew: float
    excess_kurtosis: float
    minimum: float
    maximum: float
    annualised_vol: float
    n_exact_zero: int

    def format_block(self) -> str:
        return "\n".join(
            [
                f"  n                 {self.n:>12,d}",
                f"  mean              {self.mean:>12.6f}",
                f"  std dev           {self.std:>12.6f}",
                f"  annualised vol    {self.annualised_vol:>12.4%}",
                f"  skewness          {self.skew:>12.4f}",
                f"  excess kurtosis   {self.excess_kurtosis:>12.4f}",
                f"  min               {self.minimum:>12.6f}",
                f"  max               {self.maximum:>12.6f}",
                f"  exactly zero      {self.n_exact_zero:>12,d}",
            ]
        )


@dataclass(frozen=True)
class EDAReport:
    """Everything the EDA establishes, for one window.

    ``window_label`` is carried so that a training-window report and a full-sample
    report cannot be confused for one another once printed or pickled.
    """

    window_label: str
    start: pd.Timestamp
    end: pd.Timestamp
    summary: ReturnSummary
    arch_lm: tuple[TestResult, ...]
    ljung_box_squared: tuple[TestResult, ...]
    ljung_box_raw: tuple[TestResult, ...]
    adf: TestResult
    acf_squared: pd.DataFrame

    def format_full(self) -> str:
        parts = [
            f"=== EDA: {self.window_label} "
            f"({self.start.date()} to {self.end.date()}) ===",
            "",
            "Return distribution",
            self.summary.format_block(),
            "",
            "ARCH-LM (Engle) on demeaned returns -- rejection MOTIVATES a GARCH model",
            *(f"  {t.format_line()}" for t in self.arch_lm),
            "",
            "Ljung-Box on SQUARED returns -- variance predictable from its own past",
            *(f"  {t.format_line()}" for t in self.ljung_box_squared),
            "",
            "Ljung-Box on RAW returns -- the mean, by contrast, is near-unpredictable",
            *(f"  {t.format_line()}" for t in self.ljung_box_raw),
            "",
            "Augmented Dickey-Fuller on returns -- expected to be decisive and dull",
            f"  {self.adf.format_line()}",
        ]
        return "\n".join(parts)

    def headline_verdict(self) -> str:
        """The one sentence the EDA exists to license, derived from results not opinion."""
        arch_all_reject = all(t.rejects_at_5pct for t in self.arch_lm)
        lb_sq_all_reject = all(t.rejects_at_5pct for t in self.ljung_box_squared)
        stationary = self.adf.rejects_at_5pct
        if arch_all_reject and lb_sq_all_reject and stationary:
            return (
                "ARCH effects present at every tested lag, squared returns strongly "
                "autocorrelated, returns stationary: a conditional-variance model is "
                "warranted."
            )
        return (
            "MIXED EVIDENCE -- at least one diagnostic did not behave as the locked "
            "design assumed. Do not proceed to modelling without recording why in "
            "research_log.md. "
            f"(ARCH-LM all reject: {arch_all_reject}; "
            f"Ljung-Box squared all reject: {lb_sq_all_reject}; "
            f"ADF rejects unit root: {stationary})"
        )


def _as_clean_array(returns: pd.Series) -> np.ndarray:
    """Drop NaNs and return a float array, refusing input too short to test."""
    clean = pd.Series(returns).dropna().to_numpy(dtype=float)
    if clean.size <= max(ARCH_LM_LAGS) + 1:
        raise ValueError(
            f"series has {clean.size} usable observations, too few for tests at up to "
            f"{max(ARCH_LM_LAGS)} lags"
        )
    return clean


def describe_returns(returns: pd.Series) -> ReturnSummary:
    """Summary statistics. Excess kurtosis is the one to look at.

    Daily equity returns are reliably leptokurtic, which is the direct empirical reason
    the project's likelihood uses a Student-t rather than a normal. A value near zero
    here would contradict the locked design and should stop the project rather than be
    passed over.
    """
    clean = pd.Series(returns).dropna()
    values = clean.to_numpy(dtype=float)
    return ReturnSummary(
        n=int(values.size),
        mean=float(values.mean()),
        std=float(values.std(ddof=1)),
        skew=float(np.asarray(clean.skew(), dtype=float)),
        # pandas returns *excess* (Fisher) kurtosis: 0 for a normal, not 3.
        excess_kurtosis=float(np.asarray(clean.kurtosis(), dtype=float)),
        minimum=float(values.min()),
        maximum=float(values.max()),
        annualised_vol=float(values.std(ddof=1) * np.sqrt(252.0)),
        n_exact_zero=int((values == 0.0).sum()),
    )


def arch_lm_test(returns: pd.Series, lags: int) -> TestResult:
    """Engle's LM test for ARCH effects, on **demeaned** returns.

    The test regresses squared residuals on their own lags. Passing raw rather than
    demeaned returns would fold a non-zero mean into the squares and bias the statistic
    upward -- inflating the very evidence this project relies on, in the direction that
    flatters it. Demeaning is therefore done here and not left to the caller.
    """
    clean = _as_clean_array(returns)
    resid = clean - clean.mean()
    # Indexed rather than unpacked: het_arch appends a results store when ``store=True``,
    # so its tuple arity is not fixed. Taking the first four elements is correct under
    # any of its return shapes; fixed-arity unpacking would break on a flag change.
    out = het_arch(resid, nlags=lags)
    lm_stat, lm_pvalue, f_stat, f_pvalue = (float(np.asarray(v, dtype=float)) for v in out[:4])
    return TestResult(
        name="ARCH-LM (Engle)",
        statistic=float(lm_stat),
        p_value=float(lm_pvalue),
        lags=lags,
        null="no ARCH effects up to the stated lag (conditional variance is constant)",
        interpretation=(
            "Rejection means squared residuals are autocorrelated, i.e. volatility is "
            "predictable from its own history. This is the motivating evidence for a "
            "GARCH-type model."
        ),
        extra={"f_statistic": float(f_stat), "f_p_value": float(f_pvalue)},
    )


def ljung_box_test(series: pd.Series, lags: int, *, label: str) -> TestResult:
    """Ljung-Box portmanteau test for autocorrelation up to ``lags``."""
    clean = _as_clean_array(series)
    table = acorr_ljungbox(clean, lags=[lags], return_df=True)
    return TestResult(
        name=f"Ljung-Box ({label})",
        statistic=float(table["lb_stat"].iloc[0]),
        p_value=float(table["lb_pvalue"].iloc[0]),
        lags=lags,
        null=f"no autocorrelation in {label} up to lag {lags}",
        interpretation=(
            "On squared returns, rejection is evidence of volatility clustering. On raw "
            "returns, failure to reject is the expected and unremarkable outcome."
        ),
    )


def adf_test(returns: pd.Series) -> TestResult:
    """Augmented Dickey-Fuller test, constant and no trend, lag order by AIC.

    A return series is not expected to contain a unit root and this test is not expected
    to be interesting. It is run because omitting it invites the question, and answering
    it in one line is cheaper than defending the omission.
    """
    clean = _as_clean_array(returns)
    # Indexed rather than unpacked for the same reason as het_arch above: adfuller
    # returns five elements when ``autolag=None`` and six with an autolag criterion.
    out = adfuller(clean, regression="c", autolag="AIC")
    stat = float(np.asarray(out[0], dtype=float))
    p_value = float(np.asarray(out[1], dtype=float))
    used_lag = int(out[2])
    crit_values: dict[str, float] = {str(k): float(v) for k, v in dict(out[4]).items()}
    return TestResult(
        name="Augmented Dickey-Fuller",
        statistic=stat,
        p_value=p_value,
        lags=used_lag,
        null="the series contains a unit root (is non-stationary)",
        interpretation=(
            "Rejection means the returns are stationary, as expected. The interesting "
            "structure in this project is in the second moment, not the first."
        ),
        extra={f"crit_{k.replace('%', 'pct')}": float(v) for k, v in crit_values.items()},
    )


def squared_return_acf(returns: pd.Series, nlags: int = ACF_LAGS) -> pd.DataFrame:
    """ACF of squared returns with 95% confidence bounds.

    Returns a frame indexed by lag ``1..nlags`` -- lag 0 is dropped because its value is
    identically 1 and plotting it compresses the axis so that the real decay, which is
    the whole point of the figure, becomes hard to read.
    """
    clean = _as_clean_array(returns)
    squared = (clean - clean.mean()) ** 2
    values, confint = acf(squared, nlags=nlags, alpha=0.05, fft=True)
    frame = pd.DataFrame(
        {
            "acf": values,
            "ci_lower": confint[:, 0] - values,
            "ci_upper": confint[:, 1] - values,
        },
        index=pd.RangeIndex(0, nlags + 1, name="lag"),
    )
    return frame.iloc[1:]


def run_eda(returns: pd.Series, *, window_label: str) -> EDAReport:
    """Compute every diagnostic for one window.

    ``returns`` must be the raw ``log_return`` column; every transform this module needs
    (demeaning, squaring) is applied internally, so no caller can apply one twice.
    """
    clean = pd.Series(returns).dropna()
    if clean.empty:
        raise ValueError(f"no usable returns in window {window_label!r}")

    return EDAReport(
        window_label=window_label,
        start=pd.Timestamp(clean.index.min()),
        end=pd.Timestamp(clean.index.max()),
        summary=describe_returns(clean),
        arch_lm=tuple(arch_lm_test(clean, lags=k) for k in ARCH_LM_LAGS),
        ljung_box_squared=tuple(
            ljung_box_test(clean**2, lags=k, label="squared returns")
            for k in LJUNG_BOX_LAGS
        ),
        ljung_box_raw=tuple(
            ljung_box_test(clean, lags=k, label="raw returns") for k in LJUNG_BOX_LAGS
        ),
        adf=adf_test(clean),
        acf_squared=squared_return_acf(clean),
    )
