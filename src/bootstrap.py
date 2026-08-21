"""Stationary block bootstrap for uncertainty around evaluation differences.

Locked settings: Politis-Romano stationary bootstrap, mean block length 20,
1,000 replications, fixed seed.

Why blocks rather than an i.i.d. bootstrap: loss differentials from volatility
forecasts are strongly serially dependent, and resampling observations independently
would destroy that dependence and produce confidence intervals that are far too narrow.
The stationary bootstrap draws blocks of geometrically distributed length, which
preserves short-range dependence while keeping the resampled series stationary.

Why this carries more weight than the Diebold-Mariano p-value: see the caveats in
``evaluation.diebold_mariano``. The bootstrap makes weaker assumptions about the loss
differential and degrades more gracefully when the differential is small.

All stubs. See research_log.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Locked settings.
MEAN_BLOCK_LENGTH = 20
N_REPLICATIONS = 1000
DEFAULT_SEED = 20240101


@dataclass(frozen=True)
class BootstrapCI:
    """A bootstrap point estimate and percentile interval.

    Attributes
    ----------
    point_estimate:
        The statistic on the original sample, not the bootstrap mean.
    lower / upper:
        Percentile-interval bounds at ``confidence``.
    confidence:
        Nominal confidence level, e.g. 0.95.
    replicate_values:
        All replicate statistics, retained so the distribution can be plotted and
        inspected rather than collapsed to two numbers.
    seed / mean_block_length / n_replications:
        Retained so the interval is exactly reproducible.
    """

    point_estimate: float
    lower: float
    upper: float
    confidence: float
    replicate_values: np.ndarray
    seed: int
    mean_block_length: int
    n_replications: int


def stationary_bootstrap_indices(
    n: int,
    *,
    mean_block_length: int = MEAN_BLOCK_LENGTH,
    n_replications: int = N_REPLICATIONS,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """Generate resampling indices for the Politis-Romano stationary bootstrap.

    Blocks start at a uniformly drawn position and continue with probability
    ``1 - 1/mean_block_length`` at each step, wrapping circularly at the end of the
    series. Block lengths are therefore geometric with the requested mean.

    Returns an integer array of shape ``(n_replications, n)``. Fully determined by
    ``seed`` — the same seed must always give the same indices, and a test asserts it.
    """
    raise NotImplementedError("stationary_bootstrap_indices")


def bootstrap_mean(
    series: np.ndarray,
    *,
    confidence: float = 0.95,
    mean_block_length: int = MEAN_BLOCK_LENGTH,
    n_replications: int = N_REPLICATIONS,
    seed: int = DEFAULT_SEED,
) -> BootstrapCI:
    """Block-bootstrap confidence interval for the mean of a serially dependent series.

    Used for mean QLIKE and mean variance-MSE.
    """
    raise NotImplementedError("bootstrap_mean")


def bootstrap_loss_differential(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    *,
    confidence: float = 0.95,
    mean_block_length: int = MEAN_BLOCK_LENGTH,
    n_replications: int = N_REPLICATIONS,
    seed: int = DEFAULT_SEED,
) -> BootstrapCI:
    """Block-bootstrap interval for the mean loss differential ``loss_a - loss_b``.

    The two loss series **must be resampled with the same indices** within each
    replicate. Resampling them independently would break their pairing and inflate the
    interval, discarding exactly the common variation that makes a paired comparison
    informative. This is the main correctness trap in the module and is covered by a
    test.

    An interval containing zero means the sample cannot distinguish the two models —
    which is a legitimate and reportable finding, not a failed experiment.
    """
    raise NotImplementedError("bootstrap_loss_differential")


def bootstrap_coverage_difference(
    inside_a: np.ndarray,
    inside_b: np.ndarray,
    *,
    confidence: float = 0.95,
    mean_block_length: int = MEAN_BLOCK_LENGTH,
    n_replications: int = N_REPLICATIONS,
    seed: int = DEFAULT_SEED,
) -> BootstrapCI:
    """Block-bootstrap interval for the difference in empirical interval coverage.

    ``inside_a`` and ``inside_b`` are binary indicators of the realised return falling
    inside each model's interval. Paired resampling as above.

    This is the headline uncertainty statement for the project's central claim, since
    the research question is about calibration rather than point accuracy.
    """
    raise NotImplementedError("bootstrap_coverage_difference")
