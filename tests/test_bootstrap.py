"""Stage 4 tests for ``src.bootstrap``.

Four categories:

- **Mechanism tests** ask whether the resampling scheme is the one the locked design
  names -- geometric block lengths with the requested mean, circular wrapping, and
  indices that are a pure function of the seed.
- **Power tests** ask whether the block bootstrap actually does the job an i.i.d.
  bootstrap could not: on a serially dependent series it must produce a *wider*
  interval, because that dependence is precisely what an i.i.d. resample destroys. The
  negative case is the i.i.d. comparison itself.
- **Pairing tests** cover the module's main correctness trap. Resampling two loss
  series independently would inflate every paired interval, and the failure is silent:
  the numbers stay plausible. The test is built so that a paired bootstrap must return
  a zero-width interval and an unpaired one cannot.
- **Refusal tests** pin the decision that a NaN is an error rather than a row to drop
  (D28). Dropping it would compute an error bar for a sample nobody stated.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import bootstrap as B

SEED = 20260827


# --- Fixtures ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def ar1_series() -> np.ndarray:
    """Strongly autocorrelated series: an i.i.d. bootstrap understates its variance."""
    rng = np.random.default_rng(SEED)
    n = 1000
    phi = 0.9
    values = np.empty(n)
    values[0] = rng.normal()
    for t in range(1, n):
        values[t] = phi * values[t - 1] + rng.normal()
    return values


@pytest.fixture(scope="module")
def iid_series() -> np.ndarray:
    """Serially independent series: blocks buy nothing, and should cost little."""
    return np.random.default_rng(SEED + 1).normal(size=1000)


# --- Mechanism -------------------------------------------------------------------


def test_indices_have_the_requested_shape_and_stay_in_range() -> None:
    indices = B.stationary_bootstrap_indices(200, n_replications=50, seed=SEED)
    assert indices.shape == (50, 200)
    assert indices.min() >= 0
    assert indices.max() <= 199
    assert np.issubdtype(indices.dtype, np.integer)


def test_the_same_seed_always_gives_the_same_indices() -> None:
    first = B.stationary_bootstrap_indices(120, n_replications=20, seed=SEED)
    second = B.stationary_bootstrap_indices(120, n_replications=20, seed=SEED)
    assert np.array_equal(first, second)


def test_a_different_seed_gives_different_indices() -> None:
    first = B.stationary_bootstrap_indices(120, n_replications=20, seed=SEED)
    second = B.stationary_bootstrap_indices(120, n_replications=20, seed=SEED + 1)
    assert not np.array_equal(first, second)


def test_mean_block_length_matches_the_requested_geometric_mean() -> None:
    """Blocks are counted by their breaks: a step that is not ``previous + 1 mod n``.

    Slightly biased downward -- the final block of each replicate is truncated at the
    end of the series, and a fresh draw occasionally lands on the next index by chance
    -- so the tolerance is loose enough to admit that and tight enough to reject a
    scheme that is not geometric with mean 20.
    """
    n, requested = 5000, 20
    indices = B.stationary_bootstrap_indices(
        n, mean_block_length=requested, n_replications=100, seed=SEED
    )
    breaks = np.sum(indices[:, 1:] != (indices[:, :-1] + 1) % n) + indices.shape[0]
    measured = indices.size / breaks
    assert 0.9 * requested < measured < 1.1 * requested


def test_a_block_length_of_one_is_an_iid_bootstrap() -> None:
    """At ``mean_block_length = 1`` every position is drawn afresh.

    The degenerate end of the scheme, checked because it is what the power test below
    compares against: if this were not an i.i.d. resample, that comparison would be
    measuring something else.
    """
    n = 400
    indices = B.stationary_bootstrap_indices(
        n, mean_block_length=1, n_replications=50, seed=SEED
    )
    consecutive = np.mean(indices[:, 1:] == (indices[:, :-1] + 1) % n)
    assert consecutive < 5.0 / n  # only chance coincidences, at rate 1/n


def test_blocks_wrap_circularly_rather_than_stopping_at_the_end() -> None:
    """With a block length far longer than the series, every replicate is one block.

    Circular wrapping is what keeps every observation equally likely to be drawn; a
    scheme that truncated at the end would under-sample the start of the sample.
    """
    n = 50
    indices = B.stationary_bootstrap_indices(
        n, mean_block_length=10**6, n_replications=20, seed=SEED
    )
    steps_are_consecutive = indices[:, 1:] == (indices[:, :-1] + 1) % n
    assert steps_are_consecutive.all()
    wrapped = np.any((indices[:, :-1] == n - 1) & (indices[:, 1:] == 0))
    assert wrapped


def test_locked_settings_are_the_ones_the_plan_names() -> None:
    """Mean block length 20, 1,000 replications, a fixed seed. Locked at 1.1."""
    assert B.MEAN_BLOCK_LENGTH == 20
    assert B.N_REPLICATIONS == 1000
    assert isinstance(B.DEFAULT_SEED, int)


# --- Power -----------------------------------------------------------------------


def test_bootstrap_mean_point_estimate_is_the_sample_mean(ar1_series) -> None:
    """The point estimate is the statistic on the original sample, not its bootstrap
    mean -- the latter would carry the resampling's own bias into the report."""
    ci = B.bootstrap_mean(ar1_series, seed=SEED)
    assert ci.point_estimate == pytest.approx(ar1_series.mean())


def test_bootstrap_mean_interval_brackets_its_point_estimate(ar1_series) -> None:
    ci = B.bootstrap_mean(ar1_series, seed=SEED)
    assert ci.lower < ci.point_estimate < ci.upper


def test_blocks_widen_the_interval_on_a_dependent_series(ar1_series) -> None:
    """The reason the design specifies a block bootstrap at all.

    On an AR(1) with phi = 0.9 the i.i.d. bootstrap ignores the dependence and reports
    an interval far too narrow. If this ever fails, the blocks are not doing their job
    and every uncertainty statement in the report is understated.
    """
    blocked = B.bootstrap_mean(ar1_series, seed=SEED)
    iid = B.bootstrap_mean(ar1_series, mean_block_length=1, seed=SEED)
    assert (blocked.upper - blocked.lower) > 2.0 * (iid.upper - iid.lower)


def test_blocks_cost_little_on_an_independent_series(iid_series) -> None:
    """The negative case: where there is no dependence, blocks must not manufacture
    uncertainty that is not there."""
    blocked = B.bootstrap_mean(iid_series, seed=SEED)
    iid = B.bootstrap_mean(iid_series, mean_block_length=1, seed=SEED)
    ratio = (blocked.upper - blocked.lower) / (iid.upper - iid.lower)
    assert 0.8 < ratio < 1.3


def test_a_wider_confidence_level_gives_a_wider_interval(ar1_series) -> None:
    narrow = B.bootstrap_mean(ar1_series, confidence=0.80, seed=SEED)
    wide = B.bootstrap_mean(ar1_series, confidence=0.99, seed=SEED)
    assert (wide.upper - wide.lower) > (narrow.upper - narrow.lower)


def test_replicate_values_are_retained_for_inspection(ar1_series) -> None:
    ci = B.bootstrap_mean(ar1_series, n_replications=250, seed=SEED)
    assert ci.replicate_values.shape == (250,)
    assert ci.n_replications == 250
    assert ci.mean_block_length == B.MEAN_BLOCK_LENGTH
    assert ci.seed == SEED


# --- Pairing: the module's main correctness trap ---------------------------------


def test_loss_differential_resamples_both_series_with_the_same_indices() -> None:
    """The trap, constructed so a failure cannot hide.

    Two loss series differing by exactly a constant have a constant differential, so a
    *paired* bootstrap must return that constant with zero width in every replicate.
    An implementation that drew separate indices for each series would resample two
    unrelated stretches of a highly variable loss and return a wide interval instead.
    """
    rng = np.random.default_rng(SEED)
    loss_a = np.abs(rng.normal(size=500)) + 1.0
    loss_b = loss_a - 0.5

    ci = B.bootstrap_loss_differential(loss_a, loss_b, seed=SEED)
    assert ci.point_estimate == pytest.approx(0.5)
    assert ci.lower == pytest.approx(0.5)
    assert ci.upper == pytest.approx(0.5)


def test_coverage_difference_of_identical_indicators_is_exactly_zero() -> None:
    """Same construction for the headline calibration statistic."""
    rng = np.random.default_rng(SEED)
    inside = (rng.random(600) < 0.95).astype(float)

    ci = B.bootstrap_coverage_difference(inside, inside, seed=SEED)
    assert ci.point_estimate == 0.0
    assert ci.lower == 0.0
    assert ci.upper == 0.0
    assert not ci.excludes_zero


def test_loss_differential_reports_the_sign_of_a_minus_b() -> None:
    rng = np.random.default_rng(SEED)
    loss_a = np.abs(rng.normal(size=400))
    loss_b = loss_a + 0.25  # b is uniformly the worse forecaster

    ci = B.bootstrap_loss_differential(loss_a, loss_b, seed=SEED)
    assert ci.point_estimate < 0.0
    assert ci.excludes_zero


def test_an_interval_containing_zero_is_reported_as_such() -> None:
    """``excludes_zero`` is a description of the interval, not a verdict on the models.

    A sample that cannot separate two forecasters is a finding the report states.
    """
    rng = np.random.default_rng(SEED)
    loss_a = rng.normal(size=800)
    loss_b = rng.normal(size=800)

    ci = B.bootstrap_loss_differential(loss_a, loss_b, seed=SEED)
    assert ci.lower < 0.0 < ci.upper
    assert not ci.excludes_zero


def test_coverage_difference_detects_a_real_gap() -> None:
    rng = np.random.default_rng(SEED)
    inside_a = (rng.random(1000) < 0.95).astype(float)
    inside_b = (rng.random(1000) < 0.80).astype(float)

    ci = B.bootstrap_coverage_difference(inside_a, inside_b, seed=SEED)
    assert ci.point_estimate > 0.0
    assert ci.excludes_zero


# --- Refusals --------------------------------------------------------------------


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_input_is_refused_rather_than_dropped(bad: float) -> None:
    """D28: dropping a NaN here would compute an error bar for an unstated sample."""
    values = np.ones(50)
    values[7] = bad
    with pytest.raises(ValueError, match="non-finite"):
        B.bootstrap_mean(values)


def test_a_paired_call_refuses_a_non_finite_value_in_either_series() -> None:
    good = np.ones(50)
    bad = good.copy()
    bad[3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        B.bootstrap_loss_differential(good, bad)
    with pytest.raises(ValueError, match="non-finite"):
        B.bootstrap_loss_differential(bad, good)


def test_mismatched_lengths_are_refused() -> None:
    with pytest.raises(ValueError, match="same length"):
        B.bootstrap_loss_differential(np.ones(50), np.ones(49))


def test_coverage_difference_refuses_a_non_binary_indicator() -> None:
    """It takes an *indicator*, not a coverage rate; the two are easy to confuse."""
    inside = np.ones(50)
    inside[4] = 0.5
    with pytest.raises(ValueError, match="binary"):
        B.bootstrap_coverage_difference(inside, np.ones(50))


def test_a_confidence_level_outside_the_unit_interval_is_refused() -> None:
    with pytest.raises(ValueError, match="confidence"):
        B.bootstrap_mean(np.ones(50) + np.arange(50), confidence=1.5)


def test_a_series_shorter_than_two_observations_is_refused() -> None:
    with pytest.raises(ValueError, match="at least two"):
        B.bootstrap_mean(np.array([1.0]))


def test_invalid_resampling_settings_are_refused() -> None:
    with pytest.raises(ValueError, match="mean_block_length"):
        B.stationary_bootstrap_indices(100, mean_block_length=0)
    with pytest.raises(ValueError, match="n_replications"):
        B.stationary_bootstrap_indices(100, n_replications=0)
    with pytest.raises(ValueError, match="n must be"):
        B.stationary_bootstrap_indices(1)
