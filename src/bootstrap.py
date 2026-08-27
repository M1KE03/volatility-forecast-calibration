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

Missing observations are refused, not dropped
---------------------------------------------
Every function here rejects a non-finite input rather than silently discarding it. Two
of the six forecast models have no forecast on 42 of the 2,134 evaluation days -- two
Bayesian refits failed their convergence diagnostics and produced nothing for the 21
days each served (D19) -- so a NaN reaching this module means a caller has not yet said
which sample it is comparing on. Dropping them here would make that choice invisibly,
inside a function whose output is an error bar, and the resulting interval would not
correspond to any stated sample. The common-sample decision belongs to the caller and
must appear in the table it produces (D28).
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

    @property
    def excludes_zero(self) -> bool:
        """Whether the interval lies strictly on one side of zero.

        A convenience for difference statistics, and deliberately *not* named
        ``significant``. An interval containing zero is a finding -- the sample cannot
        separate the two models -- rather than a failed test.
        """
        return (self.lower > 0.0) or (self.upper < 0.0)


def _check_series(name: str, values: np.ndarray) -> np.ndarray:
    """Validate one input series: 1-D, non-empty, finite everywhere."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if array.size < 2:
        raise ValueError(
            f"{name} must have at least two observations, got {array.size}"
        )
    if not np.all(np.isfinite(array)):
        n_bad = int((~np.isfinite(array)).sum())
        raise ValueError(
            f"{name} contains {n_bad} non-finite value(s). The bootstrap will not "
            "choose a sample on the caller's behalf: align the series on a stated "
            "common sample first, and report the n that results (D28)."
        )
    return array


def _check_pair(
    a: np.ndarray, b: np.ndarray, *, names: tuple[str, str]
) -> tuple[np.ndarray, np.ndarray]:
    """Validate a paired input, which must be aligned observation by observation."""
    array_a = _check_series(names[0], a)
    array_b = _check_series(names[1], b)
    if array_a.size != array_b.size:
        raise ValueError(
            f"paired series must be the same length, got {array_a.size} and "
            f"{array_b.size}. They must already be aligned on the same dates."
        )
    return array_a, array_b


def _percentile_interval(
    replicates: np.ndarray, confidence: float
) -> tuple[float, float]:
    """Two-sided percentile interval at ``confidence``."""
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence}")
    tail = 100.0 * (1.0 - confidence) / 2.0
    lower, upper = np.percentile(replicates, [tail, 100.0 - tail])
    return float(lower), float(upper)


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
    ``seed`` -- the same seed must always give the same indices, and a test asserts it.
    """
    if n < 2:
        raise ValueError(f"n must be at least 2, got {n}")
    if mean_block_length < 1:
        raise ValueError(
            f"mean_block_length must be at least 1, got {mean_block_length}"
        )
    if n_replications < 1:
        raise ValueError(f"n_replications must be at least 1, got {n_replications}")

    rng = np.random.default_rng(seed)
    p_new_block = 1.0 / float(mean_block_length)

    # Both random streams are drawn up front so the loop below is a pure recursion over
    # columns: where a new block starts take a fresh uniform position, otherwise step
    # one observation forward, wrapping circularly at the end of the series.
    starts = rng.integers(0, n, size=(n_replications, n))
    new_block = rng.random((n_replications, n)) < p_new_block

    indices = np.empty((n_replications, n), dtype=np.int64)
    indices[:, 0] = starts[:, 0]
    for t in range(1, n):
        continued = (indices[:, t - 1] + 1) % n
        indices[:, t] = np.where(new_block[:, t], starts[:, t], continued)
    return indices


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
    values = _check_series("series", series)
    indices = stationary_bootstrap_indices(
        values.size,
        mean_block_length=mean_block_length,
        n_replications=n_replications,
        seed=seed,
    )
    replicates = values[indices].mean(axis=1)
    lower, upper = _percentile_interval(replicates, confidence)
    return BootstrapCI(
        point_estimate=float(values.mean()),
        lower=lower,
        upper=upper,
        confidence=confidence,
        replicate_values=replicates,
        seed=seed,
        mean_block_length=mean_block_length,
        n_replications=n_replications,
    )


def _paired_difference_ci(
    a: np.ndarray,
    b: np.ndarray,
    *,
    confidence: float,
    mean_block_length: int,
    n_replications: int,
    seed: int,
) -> BootstrapCI:
    """Bootstrap the mean of ``a - b`` under a single shared index draw.

    Differencing before resampling is what enforces the pairing: one index array is
    drawn and applied to the differential itself, so no replicate can ever combine
    model A on one set of dates with model B on another. See the two public wrappers.
    """
    indices = stationary_bootstrap_indices(
        a.size,
        mean_block_length=mean_block_length,
        n_replications=n_replications,
        seed=seed,
    )
    differential = a - b
    replicates = differential[indices].mean(axis=1)
    lower, upper = _percentile_interval(replicates, confidence)
    return BootstrapCI(
        point_estimate=float(differential.mean()),
        lower=lower,
        upper=upper,
        confidence=confidence,
        replicate_values=replicates,
        seed=seed,
        mean_block_length=mean_block_length,
        n_replications=n_replications,
    )


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

    An interval containing zero means the sample cannot distinguish the two models --
    which is a legitimate and reportable finding, not a failed experiment.
    """
    values_a, values_b = _check_pair(loss_a, loss_b, names=("loss_a", "loss_b"))
    return _paired_difference_ci(
        values_a,
        values_b,
        confidence=confidence,
        mean_block_length=mean_block_length,
        n_replications=n_replications,
        seed=seed,
    )


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
    values_a, values_b = _check_pair(
        inside_a, inside_b, names=("inside_a", "inside_b")
    )
    for name, values in (("inside_a", values_a), ("inside_b", values_b)):
        if not np.all((values == 0.0) | (values == 1.0)):
            raise ValueError(
                f"{name} must be a binary indicator of the realised return falling "
                "inside the interval; it contains values other than 0 and 1."
            )
    return _paired_difference_ci(
        values_a,
        values_b,
        confidence=confidence,
        mean_block_length=mean_block_length,
        n_replications=n_replications,
        seed=seed,
    )
