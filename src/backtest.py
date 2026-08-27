"""Walk-forward backtest: the loop that produces every out-of-sample forecast.

This is where look-ahead bias would enter if it entered anywhere, so the invariants are
stated explicitly and are enforced by tests in ``tests/test_backtest.py``.

The two-timescale structure
---------------------------
There are two distinct cadences, and conflating them is the most likely subtle bug:

- **Refit cadence (every 21 trading days).** Parameters are re-estimated. Expensive:
  an MLE plus an MCMC run each time.
- **Filter cadence (every day).** Between refits the parameters are held fixed, but the
  conditional variance recursion is still advanced daily with each newly observed
  return. Freezing ``h`` between refits as well would make the forecasts staler than
  the design intends and would not be a GARCH forecast at all.

Neither baseline has an estimated parameter -- EWMA's decay is fixed at the RiskMetrics
value and the random walk has none -- so the refit cadence is a genuine no-op for those
two. It was built before the models it serves on purpose: the governing plan puts the
harness first precisely because it is the component where look-ahead bugs live, and a
harness validated on models with no moving parts is a harness whose failures can only be
its own. Stage 2's two GARCH models are the first to exercise it for real; the Bayesian
model joins at Stage 3 through the same path.

Invariants
----------
1. A forecast for date ``t`` uses returns through ``t-1`` only.
2. Parameters used on date ``t`` come from a fit whose estimation window ends at or
   before ``t-1``.
3. The regime label for date ``t`` comes from the VIX close at ``t-1``.
4. ``h0`` is backcast from the estimation window only.
5. The realised proxy for date ``t`` is an evaluation target and never an input to the
   forecast for date ``t``.
6. No shuffling, no random splitting, no reordering. Ever.

Scale convention
----------------
Every variance in the output frame is on the **close-to-close return-variance scale**,
including the ``proxy_var`` column, which is the Parkinson series multiplied by the
frozen constant ``data.PROXY_SCALE_C``. Intervals are therefore directly comparable to
observed returns, and point losses compare like with like across all four models.

Estimation window
-----------------
Expanding, per the locked design. The governing plan fixes this in its locked-decisions
table, so ``EstimationWindow.ROLLING`` is out of scope rather than merely unselected;
it raises rather than silently producing a result nobody chose.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

from src import data as D
from src import models as M
from src.models import (
    EWMA_LAMBDA,
    Forecast,
    GarchParams,
    NormalPredictive,
    forecast_ewma,
    forecast_yesterday_volatility,
)

#: Locked: parameters are re-estimated every 21 trading days.
REFIT_EVERY = 21

#: Locked: two-sided nominal levels, plus the one-sided VaR level.
TWO_SIDED_LEVELS: tuple[float, ...] = (0.90, 0.95, 0.99)
VAR_LEVEL = 0.99

#: The two parameter-free baselines (Stage 1).
BASELINE_MODELS: tuple[str, ...] = ("yesterday", "ewma")

#: GARCH models fitted by maximum likelihood (Stage 2).
GARCH_MODELS: tuple[str, ...] = ("garch_mle", "garch_mle_normal")

#: The Bayesian GARCH(1,1)-t (Stage 3). Shares ``garch_mle``'s likelihood exactly.
BAYES_MODEL = "garch_bayes"

#: The plug-in predictive **at the posterior mean** -- the same point estimate
#: ``garch_bayes`` integrates over, conditioned on as though it were the truth.
#:
#: **This exists because the comparison the project rests on turned out to have two
#: causes rather than one** (research_log.md 1.13). ``garch_bayes`` against ``garch_mle``
#: differs in two ways at once: the posterior is integrated over rather than maximised,
#: *and* the priors move the point estimate off the likelihood's maximum. Measured at the
#: 99% level on a COVID-period refit, the second effect is nine times the first and points
#: the other way, so the difference between those two models is mostly prior. Inserting
#: this track splits the comparison in two:
#:
#:     garch_bayes / garch_bayes_mean  -- parameter uncertainty, and nothing else
#:     garch_bayes_mean / garch_mle    -- the priors, and nothing else
#:
#: It is an ablation in the sense ``garch_mle_normal`` is, not a fifth competitor, and it
#: is kept out of ``HEADLINE_MODELS`` for the same reason. It costs no sampling: it reuses
#: the posterior each refit already produced.
BAYES_MEAN_MODEL = "garch_bayes_mean"

#: Both models the Bayesian track produces from one set of fits.
BAYES_MODELS: tuple[str, ...] = (BAYES_MODEL, BAYES_MEAN_MODEL)

#: Models whose predictive is a Student-t plug-in at some point estimate. The point
#: estimate differs -- a maximum for the MLE models, a posterior mean for
#: ``garch_bayes_mean`` -- but what is done with it does not, which is the whole reason
#: the comparison above isolates what it claims to.
PLUGIN_MODELS: tuple[str, ...] = GARCH_MODELS + (BAYES_MEAN_MODEL,)

#: Innovation distribution behind each GARCH model.
INNOVATION_BY_MODEL: dict[str, str] = {
    "garch_mle": "t",
    "garch_mle_normal": "normal",
}

#: Every model the backtest produces, in report order.
MODELS: tuple[str, ...] = BASELINE_MODELS + GARCH_MODELS + BAYES_MODELS

#: The two tracks, which exist because of a hundredfold difference in cost: the
#: frequentist track is 306 optimiser fits and takes about a minute, the Bayesian track
#: is 102 NUTS fits and takes about 95. They are refreshed on separate stages of
#: ``run_all.py`` and their forecast tables merged, so that re-running the cheap one does
#: not re-run the expensive one. Splitting them changes no model's output: every model's
#: path is built from the analysis frame and the config alone.
FREQUENTIST_MODELS: tuple[str, ...] = BASELINE_MODELS + GARCH_MODELS
TRACKS: dict[str, tuple[str, ...]] = {
    "frequentist": FREQUENTIST_MODELS,
    "bayes": BAYES_MODELS,
}

#: The models that belong in the headline comparison.
#:
#: ``garch_mle_normal`` is an **ablation**, not a competitor: it exists to show what the
#: Student-t innovation buys at the 99% level (governing plan, Stage 6). It is carried
#: through the full backtest because doing so costs one extra fit per refit and saves
#: re-entering this loop at Stage 6 -- but it must not quietly acquire a row in the
#: four-model tables, so the evaluation layer filters on this tuple rather than on
#: whatever happens to be in the forecast file.
HEADLINE_MODELS: tuple[str, ...] = ("yesterday", "ewma", "garch_mle", BAYES_MODEL)

#: Predictive mean assumed by both baselines.
#:
#: Zero, following the RiskMetrics convention the governing plan names for EWMA. The
#: GARCH models estimate ``mu`` instead; that asymmetry is intentional, because a
#: baseline that quietly acquired a fitted mean would stop being a baseline. The daily
#: mean return is ~0.0005 against a standard deviation of ~0.011, so the effect on
#: coverage is small -- but it is a choice, and it is recorded rather than defaulted.
BASELINE_MEAN = 0.0


class EstimationWindow(Enum):
    """How the estimation window moves as the backtest walks forward.

    EXPANDING
        Window start is fixed at the sample start; the window grows with each refit.
        This is the locked design.
    ROLLING
        Fixed length, both ends moving. Out of scope for this project.
    """

    EXPANDING = "expanding"
    ROLLING = "rolling"


@dataclass(frozen=True)
class BacktestConfig:
    """Everything that determines a backtest run, in one auditable object.

    Persisted alongside the results so any output frame can be traced back to the exact
    configuration that produced it.

    The sampler fields describe NUTS (decision B1-R) and are carried here so that a
    run's provenance is complete in a single object rather than split between this config
    and whatever the Bayesian stage happens to hard-code.

    ``target_accept`` is 0.95 rather than D17's 0.9, raised at D25 before the production
    run after a smoke refit produced four divergent transitions and therefore, under D19,
    no forecasts for the 21 days it served. It is a sampler *effort* parameter, not a
    diagnostic threshold: raising it makes NUTS take smaller steps to meet the unchanged
    D19 criterion, which is the opposite of loosening a test that fired. Applied
    uniformly to every refit and fixed before the run, so it is also not a per-refit
    retry. Measured on the refit that failed: 0 divergences against 4, at 35s against
    33s.
    """

    estimation_window: EstimationWindow = EstimationWindow.EXPANDING
    rolling_window_length: int | None = None
    refit_every: int = REFIT_EVERY
    two_sided_levels: tuple[float, ...] = TWO_SIDED_LEVELS
    var_level: float = VAR_LEVEL
    ewma_lambda: float = EWMA_LAMBDA
    proxy_scale_c: float = D.PROXY_SCALE_C
    mcmc_seed: int = 0
    draws: int = 1000
    tune: int = 1000
    chains: int = 4
    target_accept: float = 0.95
    thin: int = 2

    def __post_init__(self) -> None:
        if self.estimation_window is not EstimationWindow.EXPANDING:
            raise NotImplementedError(
                "the locked design specifies an expanding estimation window; "
                "ROLLING is out of scope for this project"
            )
        if self.refit_every < 1:
            raise ValueError("refit_every must be at least 1")
        if not all(0.0 < lv < 1.0 for lv in self.two_sided_levels):
            raise ValueError("two-sided levels must lie strictly in (0, 1)")


@dataclass(frozen=True)
class RefitRecord:
    """Diagnostics for a single refit, retained so failures are visible in the output.

    Convergence failures, poor mixing, and slow fits are findings about the models, not
    noise to be filtered out. Every one of these records is written to disk.

    The Bayesian fields are NUTS diagnostics (decision B1-R) and are NaN for every row
    that did not come from a sampler.

    ``converged`` and ``message`` are deliberately not named for either estimator. D19
    extends D16's rule to the Bayesian track with the same consequence -- a failed fit
    produces no forecasts for its block -- so the rule is applied once, to this field,
    regardless of whether an optimiser or a sampler produced the verdict. ``mle_loglik``
    keeps its name because it really is specific to the MLE and is NaN on the Bayesian
    rows: a posterior has no maximised log-likelihood.

    One record per (refit date, **fit**) rather than per model, because several models
    can rest on one fit: the two baselines share a record, and so do ``garch_bayes`` and
    ``garch_bayes_mean``, which are two things done with a single posterior. A second
    record would duplicate every field of the first.

    The parameter fields make this table the audit trail for all 102 fits per estimator as
    well as the diagnostics log -- the parameter-stability figure reads it directly, so
    the plotted estimates and the recorded ones cannot drift apart.
    """

    refit_id: int
    refit_date: pd.Timestamp
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    n_obs: int
    model: str = "baseline"
    converged: bool = True
    message: str = "no parameters estimated at this stage"
    mle_loglik: float = float("nan")
    mu: float = float("nan")
    omega: float = float("nan")
    alpha: float = float("nan")
    beta: float = float("nan")
    nu: float = float("nan")
    max_r_hat: float = float("nan")
    min_ess_bulk: float = float("nan")
    min_ess_tail: float = float("nan")
    n_divergences: int = 0
    seconds_elapsed: float = 0.0


def make_refit_dates(
    oos_index: pd.DatetimeIndex,
    *,
    refit_every: int = REFIT_EVERY,
) -> pd.DatetimeIndex:
    """Dates on which parameters are re-estimated.

    The first out-of-sample date is always a refit date. Thereafter every
    ``refit_every``-th trading day. Counting is in trading days from the out-of-sample
    index, not calendar days -- a calendar-month cadence would drift against the trading
    calendar and make the refit count depend on where holidays happened to fall.
    """
    if refit_every < 1:
        raise ValueError("refit_every must be at least 1")
    if len(oos_index) == 0:
        return pd.DatetimeIndex([], name=oos_index.name)
    positions = np.arange(0, len(oos_index), refit_every)
    return pd.DatetimeIndex(oos_index[positions], name=oos_index.name)


def estimation_slice(
    full_index: pd.DatetimeIndex,
    refit_date: pd.Timestamp,
    config: BacktestConfig,
) -> slice:
    """Positional slice of the estimation window for a refit at ``refit_date``.

    The window **must end strictly before** ``refit_date``: the return on the refit
    date has not been observed when the forecast for it is made. This is the single
    most important line in the module and is covered directly by a look-ahead test.
    """
    refit_date = pd.Timestamp(refit_date)
    stop = int(full_index.searchsorted(refit_date, side="left"))
    if stop == 0:
        raise ValueError(
            f"no history strictly before {refit_date.date()}; the estimation window "
            "would be empty"
        )
    if config.estimation_window is EstimationWindow.EXPANDING:
        return slice(0, stop)
    raise NotImplementedError("ROLLING estimation windows are out of scope")


def _quantile_levels(config: BacktestConfig) -> tuple[np.ndarray, list[str]]:
    """Probabilities to evaluate, and the column name each one populates.

    Built from the config rather than hard-coded so that the output schema and the
    locked levels cannot drift apart.
    """
    probs: list[float] = []
    names: list[str] = []
    for level in config.two_sided_levels:
        tail = (1.0 - level) / 2.0
        pct = int(round(level * 100))
        probs.extend([tail, 1.0 - tail])
        names.extend([f"lo_{pct}", f"hi_{pct}"])
    # One-sided VaR: the lower (loss) tail of the return distribution.
    probs.append(1.0 - config.var_level)
    names.append(f"var_{int(round(config.var_level * 100))}")
    return np.asarray(probs, dtype=float), names


def build_baseline_variances(
    frame: pd.DataFrame,
    config: BacktestConfig,
) -> dict[str, pd.Series]:
    """Variance paths for the two baselines, over the whole sample index.

    Both series are indexed so that row ``t`` is the forecast **for** day ``t``, built
    only from observations at ``t-1`` and earlier.

    The EWMA seed is the sample variance of the **warm-up block only**. Seeding it from
    the whole sample would leak the out-of-sample period into every forecast, including
    the earliest ones, and would do so invisibly -- the forecasts would still look
    entirely reasonable.
    """
    scaled_proxy = D.scale_proxy(frame["parkinson_var"], c=config.proxy_scale_c)
    yesterday = forecast_yesterday_volatility(scaled_proxy)

    warmup = frame.loc[pd.Timestamp(D.TRAIN_START) : pd.Timestamp(D.TRAIN_END)]
    warmup_returns = warmup["log_return"].dropna()
    if warmup_returns.size < 2:
        raise ValueError("warm-up block has too few returns to seed the EWMA")
    seed_variance = float(np.var(warmup_returns.to_numpy(dtype=float), ddof=1))

    ewma = forecast_ewma(
        frame["log_return"], lam=config.ewma_lambda, initial_var=seed_variance
    )
    return {"yesterday": yesterday, "ewma": ewma}


#: Columns every model's path frame carries. ``mu`` is the predictive mean; ``nu`` is
#: NaN wherever the predictive distribution is Gaussian, which is how the row builder
#: decides which distribution to construct.
PATH_COLUMNS: tuple[str, ...] = ("variance", "mu", "nu")


def _baseline_paths(
    frame: pd.DataFrame, config: BacktestConfig, oos_index: pd.DatetimeIndex
) -> dict[str, pd.DataFrame]:
    """Wrap the baseline variance series in the common path schema."""
    variances = build_baseline_variances(frame, config)
    paths: dict[str, pd.DataFrame] = {}
    for model in BASELINE_MODELS:
        path = pd.DataFrame(index=oos_index, columns=list(PATH_COLUMNS), dtype=float)
        path["variance"] = variances[model].reindex(oos_index)
        path["mu"] = BASELINE_MEAN
        path["nu"] = np.nan
        paths[model] = path
    return paths


def build_garch_paths(
    frame: pd.DataFrame,
    config: BacktestConfig,
    *,
    model: str,
) -> tuple[pd.DataFrame, list[RefitRecord]]:
    """Variance path and refit diagnostics for one GARCH model.

    The GARCH analogue of ``build_baseline_variances``, and the first thing in this
    project that actually uses the refit cadence.

    **The two cadences, which is the whole point of this function.** At each of the 102
    refit dates the parameters are re-estimated on the expanding window, which ends
    strictly before the refit date. *Between* refits the parameters are held fixed but
    the variance recursion still advances daily: ``garch11_filter`` is run forward with
    those fixed parameters and ``h[t]`` is read off for every date in the block. Because
    ``h[t]`` is defined as the conditional variance of ``returns[t]`` given information
    through ``t-1``, the daily update falls out of the indexing rather than needing a
    second loop -- and a forecast is never more than one day stale.

    Freezing ``h`` between refits as well would leave forecasts up to 21 days out of
    date and would not be a GARCH forecast at all. Pinned by
    ``test_variance_moves_daily_while_parameters_are_held_fixed``.

    ``h0`` is backcast from the estimation window and seeds the recursion at the first
    observation of the sample, exactly as it does inside the likelihood being maximised,
    so the fit and the forecast path share one convention. By the time the recursion
    reaches the evaluation window its influence has decayed through at least 756 factors
    of ``beta``.

    A refit whose optimiser did not converge, or which landed outside the admissible
    region, produces **no forecasts for its block**: the dates stay NaN and the failure
    is written into the ``RefitRecord``. Carrying the previous window's parameters
    forward would hide a failed fit behind plausible numbers, which is the one thing the
    ``converged`` flag exists to prevent.
    """
    if model not in INNOVATION_BY_MODEL:
        raise ValueError(f"{model!r} is not a GARCH model; expected one of {GARCH_MODELS}")
    innovation = INNOVATION_BY_MODEL[model]

    full_index = pd.DatetimeIndex(frame.index)
    returns = frame["log_return"].to_numpy(dtype=float)

    oos = frame.loc[pd.Timestamp(D.OOS_START) : pd.Timestamp(D.OOS_END)]
    oos_index = pd.DatetimeIndex(oos.index)
    refit_dates = make_refit_dates(oos_index, refit_every=config.refit_every)

    path = pd.DataFrame(index=oos_index, columns=list(PATH_COLUMNS), dtype=float)
    records: list[RefitRecord] = []

    for refit_id, refit_date in enumerate(refit_dates):
        window = estimation_slice(full_index, refit_date, config)
        train = returns[window]
        h0 = M.backcast_initial_variance(train)

        started = time.perf_counter()
        fit = M.fit_garch_mle(train, h0=h0, innovation=innovation)
        elapsed = time.perf_counter() - started

        # Dates this fit serves: from its own refit date up to the day before the next.
        block_start = int(oos_index.searchsorted(refit_date, side="left"))
        block_stop = (
            int(oos_index.searchsorted(refit_dates[refit_id + 1], side="left"))
            if refit_id + 1 < len(refit_dates)
            else len(oos_index)
        )
        block = oos_index[block_start:block_stop]

        if fit.converged and len(block) > 0:
            # Filter forward through the last date in the block. h[t] for t in the block
            # depends on returns strictly before t, so this reaches no further than it
            # is allowed to.
            stop = int(full_index.searchsorted(block[-1], side="right"))
            h = M.garch11_filter(fit.params.to_array(), returns[:stop], h0)
            positions = full_index.searchsorted(block.to_numpy(), side="left")
            path.loc[block, "variance"] = h[positions]
            path.loc[block, "mu"] = fit.params.mu
            path.loc[block, "nu"] = fit.params.nu

        records.append(
            RefitRecord(
                refit_id=refit_id,
                refit_date=refit_date,
                train_start=full_index[window.start],
                train_end=full_index[window.stop - 1],
                n_obs=fit.n_obs,
                model=model,
                converged=fit.converged,
                message=fit.message,
                mle_loglik=fit.loglik,
                mu=fit.params.mu,
                omega=fit.params.omega,
                alpha=fit.params.alpha,
                beta=fit.params.beta,
                nu=fit.params.nu,
                seconds_elapsed=elapsed,
            )
        )

    return path, records


def build_bayes_paths(
    frame: pd.DataFrame,
    config: BacktestConfig,
    *,
    cores: int | None = None,
    prior_delta: tuple[float, float] = M.PRIOR_DELTA,
) -> tuple[
    dict[str, pd.DataFrame],
    list[RefitRecord],
    dict[pd.Timestamp, M.PredictiveDistribution],
]:
    """Variance path, refit diagnostics and daily predictives for the Bayesian model.

    The Bayesian analogue of ``build_garch_paths``, and deliberately its mirror image:
    the same expanding window, the same 21-day cadence, the same ``h0`` backcast from the
    estimation window alone, the same daily filter between refits, the same rule that a
    failed fit produces no forecasts for its block. Everything that could differ between
    models 3 and 4 other than **what is done with the posterior** is held identical on
    purpose. That is what licenses the write-up's central claim that any difference in
    their intervals is parameter uncertainty and nothing else.

    Two things genuinely do differ.

    **The variance path is a distribution, not a number.** Each of the 2,000 retained
    draws (D18) implies its own filtered path, so a day's forecast carries 2,000
    variances rather than one, and the predictive is the mixture over them. The single
    ``variance`` written to the path is the posterior mean of ``h`` -- the Bayes estimate
    of tomorrow's conditional variance under squared-error loss, and the quantity
    directly comparable with the frequentist track's plug-in ``h``. It is **not** the
    variance of the predictive distribution, which additionally carries the spread of
    ``mu`` across draws; that quantity belongs to the interval, and the interval is
    reported separately.

    **The predictive cannot be rebuilt from three numbers.** The frequentist path stores
    ``(variance, mu, nu)`` and the harness reconstructs a Student-t from them. A mixture
    over 2,000 draws has no such summary, so the distribution objects are returned
    alongside the path and handed to the harness directly. They still expose exactly
    ``quantile`` and ``cdf``, so the PIT and the interval bounds are computed by the same
    code as every other model's -- which is the invariant that stops a calibration
    difference from being an artefact of the plumbing.

    **Two models, one set of fits.** ``garch_bayes`` integrates over the posterior;
    ``garch_bayes_mean`` conditions on its mean as though it were the truth. They are
    produced together because they must come from the *same* posterior for their
    difference to isolate parameter uncertainty -- computing the second from a separately
    persisted table of posterior means would work until the day the two fell out of step,
    and then it would keep working, quietly. The second costs one extra filter pass per
    refit and no sampling at all.


    **``prior_delta`` is how the Stage 6 sensitivity check runs, and it is not a knob.**
    It defaults to the frozen D4 value, so every headline result is computed under the
    priors that were frozen before any Bayesian out-of-sample number existed. The two
    other candidates D4 considered are re-run through ``run_all.py --stage priors``,
    which writes to its own directory and never touches ``forecasts.csv``. Overriding it
    here produces a *different run*, and the run that produced the report's numbers is
    the one with the default.

    Note that ``garch_bayes_mean``'s variance is ``h`` filtered *at* the posterior mean,
    while ``garch_bayes``'s is the *mean of* ``h`` across draws. Those differ by Jensen's
    inequality, and they should: one is a plug-in, the other a posterior expectation.

    A failed refit blanks both tracks. They rest on the same draws, so a posterior not
    worth integrating is not worth averaging either.

    Returns
    -------
    paths:
        One frame per model in ``BAYES_MODELS``, indexed by out-of-sample date with
        columns ``PATH_COLUMNS``. NaN on the blocks of failed refits.
    records:
        One per refit -- not one per model. Both models come from a single fit, so a
        second record would duplicate every field of the first, exactly as the two
        baselines share one record per refit.
    distributions:
        One mixture predictive per date a converged refit serves, for ``garch_bayes``
        alone. Dates missing from it have no Bayesian forecast and the harness writes NaN.
    """
    full_index = pd.DatetimeIndex(frame.index)
    returns = frame["log_return"].to_numpy(dtype=float)

    oos = frame.loc[pd.Timestamp(D.OOS_START) : pd.Timestamp(D.OOS_END)]
    oos_index = pd.DatetimeIndex(oos.index)
    refit_dates = make_refit_dates(oos_index, refit_every=config.refit_every)

    paths = {
        model: pd.DataFrame(index=oos_index, columns=list(PATH_COLUMNS), dtype=float)
        for model in BAYES_MODELS
    }
    records: list[RefitRecord] = []
    distributions: dict[pd.Timestamp, M.PredictiveDistribution] = {}

    for refit_id, refit_date in enumerate(refit_dates):
        window = estimation_slice(full_index, refit_date, config)
        train = returns[window]
        h0 = M.backcast_initial_variance(train)

        started = time.perf_counter()
        fit = M.sample_garch_posterior(
            train,
            h0=h0,
            draws=config.draws,
            tune=config.tune,
            chains=config.chains,
            target_accept=config.target_accept,
            thin=config.thin,
            # One stream per refit rather than one for the whole backtest, so no two
            # refits share a chain's randomness. Not a retry: the seed is a function of
            # the refit index, fixed before the run, and a failed fit is never re-run
            # under another one (D16, extended to this track by D19).
            seed=config.mcmc_seed + refit_id,
            cores=cores,
            prior_delta=prior_delta,
        )
        elapsed = time.perf_counter() - started

        block_start = int(oos_index.searchsorted(refit_date, side="left"))
        block_stop = (
            int(oos_index.searchsorted(refit_dates[refit_id + 1], side="left"))
            if refit_id + 1 < len(refit_dates)
            else len(oos_index)
        )
        block = oos_index[block_start:block_stop]
        posterior_mean = fit.draws.mean(axis=0)

        if fit.converged and len(block) > 0:
            # h[t] for t in the block depends on returns strictly before t, so slicing
            # the returns at the block's last date reaches no further than allowed --
            # the same bound ``build_garch_paths`` works under.
            stop = int(full_index.searchsorted(block[-1], side="right"))
            positions = full_index.searchsorted(block.to_numpy(), side="left")
            h_by_draw = M.garch11_filter_by_draw(
                fit.draws, returns[:stop], h0, np.asarray(positions, dtype=np.int64)
            )

            paths[BAYES_MODEL].loc[block, "variance"] = h_by_draw.mean(axis=0)
            paths[BAYES_MODEL].loc[block, "mu"] = float(posterior_mean[0])
            paths[BAYES_MODEL].loc[block, "nu"] = float(posterior_mean[4])

            for offset, date in enumerate(block):
                distributions[date] = M.mixture_predictive(
                    fit.draws, h_by_draw[:, offset]
                )

            # The plug-in track: one filter pass at the posterior mean, read off at the
            # same dates. Deliberately ``garch11_filter`` rather than a summary of
            # ``h_by_draw`` -- conditioning on a point estimate means running the
            # recursion at that point, which is what ``build_garch_paths`` does with the
            # MLE and what makes the two plug-ins comparable.
            h_mean = M.garch11_filter(posterior_mean, returns[:stop], h0)[positions]
            paths[BAYES_MEAN_MODEL].loc[block, "variance"] = h_mean
            paths[BAYES_MEAN_MODEL].loc[block, "mu"] = float(posterior_mean[0])
            paths[BAYES_MEAN_MODEL].loc[block, "nu"] = float(posterior_mean[4])
        records.append(
            RefitRecord(
                refit_id=refit_id,
                refit_date=refit_date,
                train_start=full_index[window.start],
                train_end=full_index[window.stop - 1],
                n_obs=fit.n_obs,
                model=BAYES_MODEL,
                converged=fit.converged,
                message=fit.message,
                # A posterior has no maximised log-likelihood; NaN rather than some
                # nearby quantity that would invite a comparison across tracks that is
                # not one.
                mle_loglik=float("nan"),
                mu=float(posterior_mean[0]),
                omega=float(posterior_mean[1]),
                alpha=float(posterior_mean[2]),
                beta=float(posterior_mean[3]),
                nu=float(posterior_mean[4]),
                max_r_hat=float(np.max(fit.r_hat)),
                min_ess_bulk=float(np.min(fit.ess_bulk)),
                min_ess_tail=float(np.min(fit.ess_tail)),
                n_divergences=int(fit.n_divergences),
                seconds_elapsed=elapsed,
            )
        )

    return paths, records, distributions


def _predictive(
    model: str,
    variance: float,
    mu: float,
    nu: float,
    distribution: M.PredictiveDistribution | None = None,
):
    """Build the predictive distribution for one model-day.

    The GARCH models go through ``models.plugin_predictive`` -- conditioning on the MLE
    as though it were the truth, which is exactly the approximation Stage 3 tests. The
    baselines get a Gaussian directly (decision D12): they are the naive-UQ baseline and
    have no fitted innovation distribution to plug in.

    ``garch_bayes`` arrives with its predictive already built, because a mixture over
    2,000 posterior draws cannot be reconstructed from ``(variance, mu, nu)``. Handed
    those three numbers and nothing else it raises rather than falling back on a plug-in.
    That fallback is exactly what ``garch_bayes_mean`` *is*, and the two must never be
    confused: one integrates over the posterior, the other conditions on its mean, and the
    difference between them is the parameter uncertainty this project set out to measure.
    Silently substituting the second for the first would report that quantity as zero.
    """
    if model == BAYES_MODEL:
        if distribution is None:
            raise ValueError(
                f"{model!r} needs its mixture predictive passed in; it cannot be "
                "rebuilt from a variance, a mean and a nu"
            )
        return distribution
    if model in PLUGIN_MODELS:
        params = GarchParams(
            mu=mu, omega=float("nan"), alpha=float("nan"), beta=float("nan"), nu=nu
        )
        return M.plugin_predictive(params, variance)
    return NormalPredictive(mean=mu, variance=variance)


def run_backtest(
    frame: pd.DataFrame,
    config: BacktestConfig | None = None,
    *,
    models: tuple[str, ...] | None = None,
    cores: int | None = None,
    prior_delta: tuple[float, float] = M.PRIOR_DELTA,
) -> tuple[pd.DataFrame, list[RefitRecord]]:
    """Walk forward through the out-of-sample period producing daily forecasts.

    Parameters
    ----------
    frame:
        The analysis frame from ``data.build_analysis_frame``, covering the full sample
        (warm-up block included -- the loop needs history before the first OOS date).
    config:
        Fully specifies the run. Defaults to the locked configuration.
    models:
        Which models to produce. Defaults to all of ``MODELS``. Exists because the
        Bayesian track costs about a hundred minutes against the frequentist track's
        one, so the two are refreshed on separate stages of ``run_all.py`` and their
        forecast tables merged. Restricting this does not change any model's output:
        every model's path is built from the frame and the config alone, never from
        another model's results.
    cores:
        Passed through to the sampler. ``None`` leaves PyMC's default, which runs chains
        in parallel; any caller doing that must guard its entry point (see
        ``models.sample_garch_posterior``). Does not affect results -- chain seeds are
        derived from ``config.mcmc_seed`` regardless of how the chains are scheduled.
    prior_delta:
        The ``Beta`` prior on ``delta``, defaulting to the frozen D4 value. Used only by
        the Stage 6 prior-sensitivity check, which writes to its own directory; see
        ``build_bayes_paths``.

    Returns
    -------
    forecasts:
        **Long format**, one row per (date, model):

        ``date``          index
        ``model``         model name
        ``variance``      one-day-ahead conditional variance, close-to-close scale
        ``mean``          predictive mean of the return
        ``lo_90``..``hi_99``  two-sided interval bounds
        ``var_99``        one-sided VaR, lower (loss) tail
        ``pit``           F(realised return) -- uniform under correct specification
        ``log_return``    realised return, the calibration target
        ``proxy_var``     realised proxy, **scaled**, the point-loss target
        ``regime``        regime label from the lagged VIX
        ``refit_id``      which refit produced this row's parameters

        Long rather than the wide schema this module's stub once described: the
        governing plan asks for tidy output, and every downstream consumer groups by
        model or by regime, which wide format would make awkward.
    records:
        One ``RefitRecord`` per (refit, model) -- the baselines contribute one row per
        refit recording the estimation window, each GARCH model one row per refit
        carrying its estimates, log-likelihood and convergence verdict. Must be
        persisted and inspected, not discarded.
    """
    config = config or BacktestConfig()

    required = {"log_return", "parkinson_var", "regime"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"analysis frame is missing columns: {sorted(missing)}")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("analysis frame index must be sorted ascending")

    oos = frame.loc[pd.Timestamp(D.OOS_START) : pd.Timestamp(D.OOS_END)]
    if oos.empty:
        raise ValueError("no out-of-sample rows in the analysis frame")
    oos_index = pd.DatetimeIndex(oos.index)

    refit_dates = make_refit_dates(oos_index, refit_every=config.refit_every)
    # refit_id for date t is the index of the most recent refit at or before t.
    refit_id_by_date = pd.Series(
        np.searchsorted(refit_dates.to_numpy(), oos_index.to_numpy(), side="right") - 1,
        index=oos_index,
    )

    wanted = MODELS if models is None else tuple(models)
    unknown = set(wanted) - set(MODELS)
    if unknown:
        raise ValueError(f"unknown models: {sorted(unknown)}; expected some of {MODELS}")

    paths: dict[str, pd.DataFrame] = {}
    records: list[RefitRecord] = []
    distributions: dict[str, dict[pd.Timestamp, M.PredictiveDistribution]] = {}

    if any(model in BASELINE_MODELS for model in wanted):
        paths.update(_baseline_paths(frame, config, oos_index))
        # One record per refit for the baselines too: they estimate nothing, but the
        # estimation window is a property of the run rather than of a model, and
        # recording it once per refit keeps the refit table readable next to the GARCH
        # rows.
        records.extend(
            RefitRecord(
                refit_id=refit_id,
                refit_date=refit_date,
                train_start=frame.index[0],
                train_end=frame.index[
                    estimation_slice(
                        pd.DatetimeIndex(frame.index), refit_date, config
                    ).stop
                    - 1
                ],
                n_obs=estimation_slice(
                    pd.DatetimeIndex(frame.index), refit_date, config
                ).stop,
            )
            for refit_id, refit_date in enumerate(refit_dates)
        )

    for model in GARCH_MODELS:
        if model not in wanted:
            continue
        path, garch_records = build_garch_paths(frame, config, model=model)
        paths[model] = path
        records.extend(garch_records)

    if any(model in BAYES_MODELS for model in wanted):
        bayes_paths, bayes_records, bayes_distributions = build_bayes_paths(
            frame, config, cores=cores, prior_delta=prior_delta
        )
        paths.update(bayes_paths)
        records.extend(bayes_records)
        distributions[BAYES_MODEL] = bayes_distributions

    probs, quantile_names = _quantile_levels(config)
    scaled_proxy = D.scale_proxy(frame["parkinson_var"], c=config.proxy_scale_c)

    rows: list[dict[str, object]] = []
    for model in MODELS:
        if model not in wanted:
            continue
        path = paths[model]
        by_date = distributions.get(model, {})
        for date in oos_index:
            variance = float(path.at[date, "variance"])
            mu = float(path.at[date, "mu"])
            nu = float(path.at[date, "nu"])
            realised = float(oos.loc[date, "log_return"])
            if not np.isfinite(variance) or variance <= 0.0 or not np.isfinite(mu):
                # Recorded, not dropped: a missing forecast is a fact about the model
                # and must survive into the evaluation layer rather than vanish.
                row: dict[str, object] = {
                    "date": date,
                    "model": model,
                    "variance": np.nan,
                    "mean": np.nan,
                    **{name: np.nan for name in quantile_names},
                    "pit": np.nan,
                }
            else:
                forecast = Forecast(
                    variance=variance,
                    mean=mu,
                    distribution=_predictive(
                        model, variance, mu, nu, by_date.get(date)
                    ),
                )
                quantiles = forecast.distribution.quantile(probs)
                row = {
                    "date": date,
                    "model": model,
                    "variance": variance,
                    "mean": forecast.mean,
                    **dict(zip(quantile_names, (float(q) for q in quantiles))),
                    "pit": forecast.distribution.cdf(realised),
                }
            row.update(
                {
                    "log_return": realised,
                    "proxy_var": float(scaled_proxy.loc[date]),
                    "regime": oos.loc[date, "regime"],
                    "refit_id": int(refit_id_by_date.loc[date]),
                }
            )
            rows.append(row)

    forecasts = pd.DataFrame(rows).set_index("date").sort_index(kind="stable")
    return forecasts, records


#: Column order used when the merged forecast table is written, so that a rebuilt
#: ``forecasts.csv`` is byte-comparable with the one the previous run produced.
_MODEL_ORDER: dict[str, int] = {model: i for i, model in enumerate(MODELS)}

#: Date columns in the refit table, named so the merge can parse them back into
#: timestamps rather than concatenating strings onto timestamps.
_RECORD_DATE_COLUMNS: tuple[str, ...] = ("refit_date", "train_start", "train_end")

#: Config fields that describe the sampler and therefore only the Bayesian track. The
#: frequentist track carries whatever it was handed in them and never reads them, so
#: they are excluded when the two tracks' configs are compared for compatibility, and
#: the merged config takes them from the track that actually sampled.
SAMPLER_FIELDS: tuple[str, ...] = (
    "mcmc_seed",
    "draws",
    "tune",
    "chains",
    "target_accept",
    "thin",
)


def track_paths(track: str, processed_dir: Path) -> tuple[Path, Path, Path]:
    """The three files one track's partial results live in."""
    if track not in TRACKS:
        raise ValueError(f"unknown track {track!r}; expected one of {sorted(TRACKS)}")
    return (
        processed_dir / f"forecasts_{track}.csv",
        processed_dir / f"refit_records_{track}.csv",
        processed_dir / f"backtest_config_{track}.json",
    )


def save_track(
    track: str,
    forecasts: pd.DataFrame,
    records: list[RefitRecord],
    config: BacktestConfig,
    processed_dir: Path,
) -> list[Path]:
    """Persist one track's results, then rebuild the merged forecast table.

    The partials are the durable artefacts: ``forecasts.csv`` is a view over whichever
    of them exist, rebuilt on every write. That is what lets the frequentist track be
    re-run in a minute without paying the Bayesian track's ninety-five, while leaving
    one complete table for the evaluation layer to read.

    **The configs are compared, not assumed to match.** Merging a Bayesian table
    computed under one proxy scale or refit cadence into a frequentist table computed
    under another would produce a forecast file whose rows answer different questions,
    and nothing downstream could detect it. If the partials disagree the merge refuses
    and names the stage to re-run.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    forecast_path, records_path, config_path = track_paths(track, processed_dir)

    forecasts.to_csv(
        forecast_path, index_label="date", date_format="%Y-%m-%d", lineterminator="\n"
    )
    pd.DataFrame([asdict(r) for r in records]).to_csv(
        records_path, index=False, date_format="%Y-%m-%d", lineterminator="\n"
    )
    _write_config(config, config_path)

    return [forecast_path, records_path, *merge_tracks(processed_dir)]


def merge_tracks(processed_dir: Path) -> list[Path]:
    """Rebuild ``forecasts.csv`` and ``refit_records.csv`` from the tracks on disk.

    Reads the partials back rather than merging in memory, so the merged table is a
    function of what is actually stored -- a stage that failed halfway cannot leave a
    forecast file claiming results it never wrote.
    """
    present: list[str] = []
    frames: list[pd.DataFrame] = []
    record_frames: list[pd.DataFrame] = []
    configs: dict[str, dict] = {}

    for track in TRACKS:
        forecast_path, records_path, config_path = track_paths(track, processed_dir)
        if not forecast_path.exists():
            continue
        present.append(track)
        frames.append(pd.read_csv(forecast_path, index_col=0, parse_dates=True))
        record_frames.append(
            pd.read_csv(records_path, parse_dates=list(_RECORD_DATE_COLUMNS))
        )
        with config_path.open(encoding="utf-8") as fh:
            configs[track] = json.load(fh)

    if not present:
        raise FileNotFoundError(
            f"no track results in {processed_dir}; run `python run_all.py --stage backtest`"
        )

    reference = configs[present[0]]
    for track in present[1:]:
        differing = {
            key
            for key in set(reference) | set(configs[track])
            if reference.get(key) != configs[track].get(key)
        }
        # The sampler settings are the Bayesian track's business alone, and the
        # frequentist track records whatever it was handed. A difference there says
        # nothing about whether the two tables belong together.
        differing -= set(SAMPLER_FIELDS)
        if differing:
            raise ValueError(
                f"tracks {present[0]!r} and {track!r} were run under different "
                f"configurations ({sorted(differing)}); re-run one of them before "
                "their forecasts can be read as one table"
            )

    forecasts = pd.concat(frames)
    forecasts = forecasts.assign(_order=forecasts["model"].map(_MODEL_ORDER))
    forecasts = forecasts.sort_values(["_order"], kind="stable").drop(columns="_order")
    forecasts = forecasts.sort_index(kind="stable")

    records = pd.concat(record_frames, ignore_index=True)

    processed_dir.mkdir(parents=True, exist_ok=True)
    forecast_path = processed_dir / "forecasts.csv"
    records_path = processed_dir / "refit_records.csv"
    forecasts.to_csv(
        forecast_path, index_label="date", date_format="%Y-%m-%d", lineterminator="\n"
    )
    records.to_csv(
        records_path, index=False, date_format="%Y-%m-%d", lineterminator="\n"
    )

    # The merged config is the shared part, with the sampler fields taken from the
    # track that actually sampled. The frequentist track records whatever it was handed
    # in those fields and never reads them, so carrying its values into the merged file
    # would state a `target_accept` no fit ever used -- provenance that is worse than
    # absent, because it looks authoritative.
    merged = dict(reference)
    if "bayes" in configs:
        merged.update({key: configs["bayes"][key] for key in SAMPLER_FIELDS})
    elif "bayes" not in present:
        merged.update({key: None for key in SAMPLER_FIELDS})
    _write_config_dict(merged, processed_dir / "backtest_config.json")

    return [forecast_path, records_path]


def _write_config(config: BacktestConfig, path: Path) -> None:
    payload = asdict(config)
    payload["estimation_window"] = config.estimation_window.value
    payload["two_sided_levels"] = list(config.two_sided_levels)
    _write_config_dict(payload, path)


def _write_config_dict(payload: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")


def save_forecasts(
    forecasts: pd.DataFrame,
    records: list[RefitRecord],
    config: BacktestConfig,
    processed_dir: Path,
) -> list[Path]:
    """Persist one whole run's forecasts, refit diagnostics, and config.

    Notebooks read these artefacts; they never re-run the backtest. Writing the config
    alongside the results keeps every saved number traceable to the run that made it.

    For a run of a single track use ``save_track`` instead, which writes the track's
    partials and rebuilds the merged table from every track on disk. This function
    remains the right one for a run that produced every model at once.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)

    forecast_path = processed_dir / "forecasts.csv"
    forecasts.to_csv(forecast_path, index_label="date", date_format="%Y-%m-%d", lineterminator="\n")

    records_path = processed_dir / "refit_records.csv"
    pd.DataFrame([asdict(r) for r in records]).to_csv(
        records_path, index=False, date_format="%Y-%m-%d", lineterminator="\n"
    )

    config_path = processed_dir / "backtest_config.json"
    payload = asdict(config)
    payload["estimation_window"] = config.estimation_window.value
    payload["two_sided_levels"] = list(config.two_sided_levels)
    with config_path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")

    return [forecast_path, records_path, config_path]
