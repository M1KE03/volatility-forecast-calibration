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

#: Models produced by this module. ``garch_bayes`` is appended at Stage 3.
BASELINE_MODELS: tuple[str, ...] = ("yesterday", "ewma")

#: GARCH models fitted by maximum likelihood (Stage 2).
GARCH_MODELS: tuple[str, ...] = ("garch_mle", "garch_mle_normal")

#: Innovation distribution behind each GARCH model.
INNOVATION_BY_MODEL: dict[str, str] = {
    "garch_mle": "t",
    "garch_mle_normal": "normal",
}

#: Every model the backtest produces, in report order.
MODELS: tuple[str, ...] = BASELINE_MODELS + GARCH_MODELS

#: The models that belong in the headline comparison.
#:
#: ``garch_mle_normal`` is an **ablation**, not a competitor: it exists to show what the
#: Student-t innovation buys at the 99% level (governing plan, Stage 6). It is carried
#: through the full backtest because doing so costs one extra fit per refit and saves
#: re-entering this loop at Stage 6 -- but it must not quietly acquire a row in the
#: four-model tables, so the evaluation layer filters on this tuple rather than on
#: whatever happens to be in the forecast file.
HEADLINE_MODELS: tuple[str, ...] = ("yesterday", "ewma", "garch_mle")

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

    The sampler fields describe NUTS (decision B1-R). They are inert until Stage 3 and
    are carried here so that a run's provenance is complete in a single object rather
    than split between this config and whatever Stage 3 happens to hard-code.
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
    target_accept: float = 0.9

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

    The Bayesian fields are NUTS diagnostics (decision B1-R) and are NaN for the
    baseline-only runs of Stage 1, which estimate nothing.

    One record per (refit date, model). The parameter fields make this table the audit
    trail for all 102 fits per model as well as the diagnostics log -- the
    parameter-stability figure reads it directly, so the plotted estimates and the
    recorded ones cannot drift apart.
    """

    refit_id: int
    refit_date: pd.Timestamp
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    n_obs: int
    model: str = "baseline"
    mle_converged: bool = True
    mle_message: str = "no parameters estimated at this stage"
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
                mle_converged=fit.converged,
                mle_message=fit.message,
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


def _predictive(model: str, variance: float, mu: float, nu: float):
    """Build the predictive distribution for one model-day.

    The GARCH models go through ``models.plugin_predictive`` -- conditioning on the MLE
    as though it were the truth, which is exactly the approximation Stage 3 tests. The
    baselines get a Gaussian directly (decision D12): they are the naive-UQ baseline and
    have no fitted innovation distribution to plug in.
    """
    if model in GARCH_MODELS:
        params = GarchParams(
            mu=mu, omega=float("nan"), alpha=float("nan"), beta=float("nan"), nu=nu
        )
        return M.plugin_predictive(params, variance)
    return NormalPredictive(mean=mu, variance=variance)


def run_backtest(
    frame: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> tuple[pd.DataFrame, list[RefitRecord]]:
    """Walk forward through the out-of-sample period producing daily forecasts.

    Parameters
    ----------
    frame:
        The analysis frame from ``data.build_analysis_frame``, covering the full sample
        (warm-up block included -- the loop needs history before the first OOS date).
    config:
        Fully specifies the run. Defaults to the locked configuration.

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

    paths = _baseline_paths(frame, config, oos_index)
    # One record per refit for the baselines too: they estimate nothing, but the
    # estimation window is a property of the run rather than of a model, and recording
    # it once per refit keeps the refit table readable next to the GARCH rows.
    records = [
        RefitRecord(
            refit_id=refit_id,
            refit_date=refit_date,
            train_start=frame.index[0],
            train_end=frame.index[
                estimation_slice(pd.DatetimeIndex(frame.index), refit_date, config).stop - 1
            ],
            n_obs=estimation_slice(
                pd.DatetimeIndex(frame.index), refit_date, config
            ).stop,
        )
        for refit_id, refit_date in enumerate(refit_dates)
    ]

    for model in GARCH_MODELS:
        path, garch_records = build_garch_paths(frame, config, model=model)
        paths[model] = path
        records.extend(garch_records)

    probs, quantile_names = _quantile_levels(config)
    scaled_proxy = D.scale_proxy(frame["parkinson_var"], c=config.proxy_scale_c)

    rows: list[dict[str, object]] = []
    for model in MODELS:
        path = paths[model]
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
                    distribution=_predictive(model, variance, mu, nu),
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


def save_forecasts(
    forecasts: pd.DataFrame,
    records: list[RefitRecord],
    config: BacktestConfig,
    processed_dir: Path,
) -> list[Path]:
    """Persist forecasts, refit diagnostics, and the config to ``data/processed/``.

    Notebooks read these artefacts; they never re-run the backtest. Writing the config
    alongside the results keeps every saved number traceable to the run that made it.
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
