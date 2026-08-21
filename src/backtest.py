"""Walk-forward backtest: the loop that produces every out-of-sample forecast.

This is where look-ahead bias would enter if it entered anywhere, so the invariants are
stated explicitly and are enforced by tests in ``tests/``.

The two-timescale structure
---------------------------
There are two distinct cadences, and conflating them is the most likely subtle bug:

- **Refit cadence (every 21 trading days).** Parameters are re-estimated. Expensive:
  an MLE plus an MCMC run each time.
- **Filter cadence (every day).** Between refits the parameters are held fixed, but the
  conditional variance recursion is still advanced daily with each newly observed
  return. Freezing ``h`` between refits as well would make the forecasts staler than
  the design intends and would not be a GARCH forecast at all.

Invariants
----------
1. A forecast for date ``t`` uses returns through ``t-1`` only.
2. Parameters used on date ``t`` come from a fit whose estimation window ends at or
   before ``t-1``.
3. The regime label for date ``t`` comes from the VIX close at ``t-1``.
4. ``h0`` is backcast from the estimation window only.
5. No shuffling, no random splitting, no reordering. Ever.

Open item D1 blocks this module: whether the estimation window **expands** or **rolls**
at fixed length is not yet decided. ``EstimationWindow`` exists so the choice is an
explicit parameter rather than an accident of the loop.

All stubs. See research_log.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

#: Locked: parameters are re-estimated every 21 trading days.
REFIT_EVERY = 21

#: Locked: two-sided nominal levels, plus the one-sided VaR level.
TWO_SIDED_LEVELS: tuple[float, ...] = (0.90, 0.95, 0.99)
VAR_LEVEL = 0.99


class EstimationWindow(Enum):
    """How the estimation window moves as the backtest walks forward.

    EXPANDING
        Window start is fixed at the sample start; the window grows with each refit.
    ROLLING
        Window has fixed length; both ends move.

    Open item D1 -- not yet decided. This affects results materially and also affects
    which asymptotic theory applies to the forecast comparison in ``evaluation``.
    """

    EXPANDING = "expanding"
    ROLLING = "rolling"


@dataclass(frozen=True)
class BacktestConfig:
    """Everything that determines a backtest run, in one auditable object.

    Persisted alongside the results so any output frame can be traced back to the exact
    configuration that produced it.
    """

    estimation_window: EstimationWindow
    rolling_window_length: int | None
    refit_every: int = REFIT_EVERY
    two_sided_levels: tuple[float, ...] = TWO_SIDED_LEVELS
    var_level: float = VAR_LEVEL
    mcmc_seed: int = 0
    n_walkers: int = 32
    n_steps: int = 3000
    n_burn: int = 1000
    thin: int = 5


@dataclass(frozen=True)
class RefitRecord:
    """Diagnostics for a single refit, retained so failures are visible in the output.

    Convergence failures, poor mixing, and slow fits are findings about the models, not
    noise to be filtered out. Every one of these records is written to disk.
    """

    refit_date: pd.Timestamp
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    n_obs: int
    mle_converged: bool
    mle_message: str
    mle_loglik: float
    mcmc_acceptance_fraction: float
    mcmc_autocorr_time: np.ndarray
    mcmc_n_effective: np.ndarray
    seconds_elapsed: float


def make_refit_dates(
    oos_index: pd.DatetimeIndex,
    *,
    refit_every: int = REFIT_EVERY,
) -> pd.DatetimeIndex:
    """Dates on which parameters are re-estimated.

    The first out-of-sample date is always a refit date. Thereafter every
    ``refit_every``-th trading day. Counting is in trading days from the out-of-sample
    index, not calendar days.
    """
    raise NotImplementedError("make_refit_dates")


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
    raise NotImplementedError("estimation_slice")


def run_backtest(
    frame: pd.DataFrame,
    config: BacktestConfig,
) -> tuple[pd.DataFrame, list[RefitRecord]]:
    """Walk forward through the out-of-sample period producing daily forecasts.

    Parameters
    ----------
    frame:
        The analysis frame from ``data.build_analysis_frame``, covering the full sample
        (training block included -- the loop needs history before the first OOS date).
    config:
        Fully specifies the run.

    Returns
    -------
    forecasts:
        One row per out-of-sample date, with columns:

        ``log_return``            realised return (the calibration target)
        ``parkinson_var``         realised proxy (the point-loss target)
        ``regime``                regime label from the lagged VIX
        ``{model}_var``           one-day-ahead conditional variance forecast
        ``{model}_mean``          predictive mean of the return
        ``{model}_lo_{level}``    lower bound of the two-sided interval
        ``{model}_hi_{level}``    upper bound of the two-sided interval
        ``{model}_var_{level}``   one-sided VaR (lower tail)
        ``refit_id``              which refit produced this row's parameters

        for ``model`` in {yesterday, ewma, garch_mle, garch_bayes}. Baselines produce a
        variance forecast and, using a Gaussian or t reference, intervals on the same
        footing so that coverage is comparable across all four.
    records:
        One ``RefitRecord`` per refit. Must be persisted and inspected, not discarded.
    """
    raise NotImplementedError("run_backtest")


def save_forecasts(
    forecasts: pd.DataFrame,
    records: list[RefitRecord],
    config: BacktestConfig,
    processed_dir: Path,
) -> None:
    """Persist forecasts, refit diagnostics, and the config to ``data/processed/``.

    Notebooks read these artefacts; they never re-run the backtest. Writing the config
    alongside the results keeps every saved number traceable to the run that made it.
    """
    raise NotImplementedError("save_forecasts")
