"""Stage 1 tests for ``src.backtest`` -- the look-ahead audit.

The governing plan's cut list names this audit as something never to drop, and the
reason is worth restating: a calibration result computed on a leaking backtest is worse
than no result at all, because it looks fine. Every other number this project will
produce is downstream of the loop these tests cover.

Two categories:

- **Look-ahead tests** ask whether anything sees the future. The master test is
  ``test_master_look_ahead_corrupting_the_future_leaves_the_past_identical``; the others
  cover the specific mechanisms by which leakage would arrive.
- **Contract tests** pin the schema, the refit cadence, and the invariants the
  evaluation layer will rely on.

A note on the master test's precise claim
-----------------------------------------
Corrupting the frame from date ``t`` onward must leave every *forecast* for dates up to
and including ``t`` unchanged, because the forecast for ``t`` is built from data through
``t-1``. It must **not** leave the *evaluation* columns at ``t`` unchanged: ``log_return``
and the ``pit`` derived from it are functions of day ``t`` itself. Asserting the stronger
claim would be asserting something false and would have to be weakened later, which is
how a look-ahead test quietly stops testing anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import backtest as B
from src import data as D

FORECAST_COLUMNS = [
    "variance",
    "mean",
    "lo_90",
    "hi_90",
    "lo_95",
    "hi_95",
    "lo_99",
    "hi_99",
    "var_99",
]


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    """The real analysis frame. These tests are about the real pipeline, not a mock."""
    path = "data/processed/analysis_frame.csv"
    try:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    except FileNotFoundError:  # pragma: no cover - depends on local state
        pytest.skip(f"{path} not present; run `python run_all.py --stage data`")


@pytest.fixture(scope="module")
def baseline_run(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[B.RefitRecord]]:
    return B.run_backtest(frame)


# --- Look-ahead tests --------------------------------------------------------------


@pytest.mark.parametrize("corrupt_from", ["2018-02-05", "2020-03-16", "2022-06-13"])
def test_master_look_ahead_corrupting_the_future_leaves_the_past_identical(
    frame: pd.DataFrame, corrupt_from: str
) -> None:
    """The master test. If this passes, the pipeline has no forward leakage.

    Chosen dates are the worst cases available: the Volmageddon spike, the largest COVID
    drawdown day, and a 2022 selloff. If any forecast reached forward, these are the
    days on which the corruption would be most visible.
    """
    baseline, _ = B.run_backtest(frame)

    cut = pd.Timestamp(corrupt_from)
    corrupted = frame.copy()
    mask = corrupted.index >= cut
    assert mask.sum() > 0, "corruption window is empty; the test would be vacuous"
    rng = np.random.default_rng(20260823)
    corrupted.loc[mask, "log_return"] = rng.normal(0.0, 0.5, size=int(mask.sum()))
    corrupted.loc[mask, "parkinson_var"] = rng.uniform(0.01, 0.10, size=int(mask.sum()))

    recomputed, _ = B.run_backtest(corrupted)

    # Forecasts up to and including the cut date use data through cut-1 only.
    past = baseline.index <= cut
    for model in B.BASELINE_MODELS:
        a = baseline[(baseline.model == model) & past][FORECAST_COLUMNS]
        b = recomputed[(recomputed.model == model) & past][FORECAST_COLUMNS]
        pd.testing.assert_frame_equal(a, b, check_exact=True)

    # And strictly before the cut, the evaluation columns are untouched too.
    strictly_past = baseline.index < cut
    for model in B.BASELINE_MODELS:
        a = baseline[(baseline.model == model) & strictly_past]
        b = recomputed[(recomputed.model == model) & strictly_past]
        pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_corruption_actually_changes_the_future(frame: pd.DataFrame) -> None:
    """Proves the master test is not passing vacuously.

    If the corruption had no effect anywhere, the assertions above would hold for a
    backtest that ignored its input entirely.
    """
    baseline, _ = B.run_backtest(frame)
    cut = pd.Timestamp("2020-03-16")
    corrupted = frame.copy()
    mask = corrupted.index >= cut
    corrupted.loc[mask, "log_return"] = 0.5
    corrupted.loc[mask, "parkinson_var"] = 0.05
    recomputed, _ = B.run_backtest(corrupted)

    future = baseline.index > cut
    for model in B.BASELINE_MODELS:
        a = baseline[(baseline.model == model) & future]["variance"]
        b = recomputed[(recomputed.model == model) & future]["variance"]
        assert not np.allclose(a.to_numpy(), b.to_numpy()), (
            f"corrupting the future did not move {model}'s later forecasts"
        )


def test_random_walk_forecast_never_uses_its_own_days_proxy(
    frame: pd.DataFrame, baseline_run
) -> None:
    """Invariant 5: the day-``t`` proxy is an evaluation target, never an input.

    Asserted directly on the numbers: the RW forecast for day ``t`` must equal the
    scaled proxy at ``t-1``, and must differ from the scaled proxy at ``t``.
    """
    forecasts, _ = baseline_run
    rw = forecasts[forecasts.model == "yesterday"]
    scaled = D.scale_proxy(frame["parkinson_var"])

    expected_lagged = scaled.shift(1).reindex(rw.index)
    np.testing.assert_allclose(rw["variance"].to_numpy(), expected_lagged.to_numpy())

    same_day = scaled.reindex(rw.index)
    differs = ~np.isclose(rw["variance"].to_numpy(), same_day.to_numpy())
    assert differs.mean() > 0.99, "RW forecast looks like the same day's proxy"


def test_ewma_seed_cannot_be_moved_by_out_of_sample_returns(frame: pd.DataFrame) -> None:
    """The EWMA seed comes from the warm-up block only.

    Seeding from the full sample would leak the evaluation period into every forecast
    including the earliest, and would do so invisibly -- the forecasts would still look
    perfectly reasonable.
    """
    config = B.BacktestConfig()
    baseline = B.build_baseline_variances(frame, config)["ewma"]

    corrupted = frame.copy()
    oos_mask = corrupted.index >= pd.Timestamp(D.OOS_START)
    corrupted.loc[oos_mask, "log_return"] = 1.0
    recomputed = B.build_baseline_variances(corrupted, config)["ewma"]

    first_oos = pd.Timestamp(D.OOS_START)
    np.testing.assert_allclose(
        baseline.loc[:first_oos].to_numpy(),
        recomputed.loc[:first_oos].to_numpy(),
        equal_nan=True,
    )


def test_ewma_forecast_for_day_t_ignores_the_return_on_day_t(frame: pd.DataFrame) -> None:
    """Guards the off-by-one that would make the EWMA forecast peek one day ahead."""
    config = B.BacktestConfig()
    baseline = B.build_baseline_variances(frame, config)["ewma"]

    target = pd.Timestamp("2020-03-16")
    corrupted = frame.copy()
    corrupted.loc[target, "log_return"] = 0.5
    recomputed = B.build_baseline_variances(corrupted, config)["ewma"]

    assert baseline.loc[target] == recomputed.loc[target]
    later = corrupted.index[corrupted.index > target][0]
    assert baseline.loc[later] != recomputed.loc[later], (
        "changing a return did not affect the next day's EWMA forecast"
    )


def test_estimation_slice_never_includes_the_refit_date(frame: pd.DataFrame) -> None:
    """Invariant 2, asserted at the boundary where it would fail."""
    config = B.BacktestConfig()
    index = pd.DatetimeIndex(frame.index)
    oos_index = pd.DatetimeIndex(
        frame.loc[pd.Timestamp(D.OOS_START) : pd.Timestamp(D.OOS_END)].index
    )
    for refit_date in B.make_refit_dates(oos_index):
        sl = B.estimation_slice(index, refit_date, config)
        window = index[sl]
        assert window.max() < refit_date, (
            f"estimation window for {refit_date.date()} reaches to {window.max().date()}"
        )


def test_estimation_slice_expands(frame: pd.DataFrame) -> None:
    """The locked design is an expanding window: start fixed, end moving."""
    config = B.BacktestConfig()
    index = pd.DatetimeIndex(frame.index)
    oos_index = pd.DatetimeIndex(
        frame.loc[pd.Timestamp(D.OOS_START) : pd.Timestamp(D.OOS_END)].index
    )
    refit_dates = B.make_refit_dates(oos_index)
    lengths = [B.estimation_slice(index, d, config).stop for d in refit_dates]
    starts = [B.estimation_slice(index, d, config).start for d in refit_dates]
    assert set(starts) == {0}, "expanding window must keep a fixed start"
    assert lengths == sorted(lengths) and lengths[0] < lengths[-1]


def test_no_reordering_of_dates(baseline_run) -> None:
    """Invariant 6. A shuffle anywhere would invalidate every serial-dependence claim."""
    forecasts, _ = baseline_run
    for model in B.BASELINE_MODELS:
        idx = forecasts[forecasts.model == model].index
        assert idx.is_monotonic_increasing
        assert idx.is_unique


# --- Contract tests ----------------------------------------------------------------


def test_forecast_table_is_complete_over_the_evaluation_window(
    frame: pd.DataFrame, baseline_run
) -> None:
    """Stage 1's headline acceptance criterion."""
    forecasts, _ = baseline_run
    oos = frame.loc[pd.Timestamp(D.OOS_START) : pd.Timestamp(D.OOS_END)]

    for model in B.BASELINE_MODELS:
        sub = forecasts[forecasts.model == model]
        assert len(sub) == len(oos) == 2134
        assert sub.index.equals(pd.DatetimeIndex(oos.index))
        for column in FORECAST_COLUMNS + ["pit"]:
            assert sub[column].notna().all(), f"{model} has gaps in {column}"


def test_pit_lies_in_the_unit_interval(baseline_run) -> None:
    forecasts, _ = baseline_run
    pit = forecasts["pit"].dropna()
    assert pit.between(0.0, 1.0).all()


def test_interval_bounds_are_correctly_ordered(baseline_run) -> None:
    """Wider nominal level implies a wider interval, and VaR sits inside the 99% band.

    ``var_99`` is the 1% quantile while ``lo_99`` is the 0.5% quantile of a two-sided
    99% interval, so the one-sided VaR must be the less extreme of the two. Getting this
    backwards is an easy and silent error that would misstate every VaR breach count.
    """
    forecasts, _ = baseline_run
    f = forecasts.dropna(subset=FORECAST_COLUMNS)
    assert (f.lo_99 < f.lo_95).all()
    assert (f.lo_95 < f.lo_90).all()
    assert (f.lo_90 < f.hi_90).all()
    assert (f.hi_90 < f.hi_95).all()
    assert (f.hi_95 < f.hi_99).all()
    assert (f.lo_99 < f.var_99).all()
    assert (f.var_99 < f.lo_95).all()


def test_make_refit_dates_cadence(frame: pd.DataFrame) -> None:
    oos_index = pd.DatetimeIndex(
        frame.loc[pd.Timestamp(D.OOS_START) : pd.Timestamp(D.OOS_END)].index
    )
    refits = B.make_refit_dates(oos_index, refit_every=21)

    assert refits[0] == oos_index[0], "the first OOS date must be a refit date"
    positions = oos_index.searchsorted(refits)
    assert np.all(np.diff(positions) == 21), "refits must be 21 trading days apart"
    assert len(refits) == int(np.ceil(len(oos_index) / 21)) == 102


def test_make_refit_dates_handles_empty_index() -> None:
    assert len(B.make_refit_dates(pd.DatetimeIndex([]))) == 0


def test_refit_id_is_constant_between_refits(baseline_run) -> None:
    """Non-decreasing, and each id spans exactly one refit period."""
    forecasts, records = baseline_run
    ids = forecasts[forecasts.model == "ewma"]["refit_id"]
    assert ids.is_monotonic_increasing
    assert ids.iloc[0] == 0
    assert ids.max() == len(records) - 1
    counts = ids.value_counts()
    assert (counts.drop(ids.max()) == 21).all(), "refit blocks must be 21 days"


def test_refit_records_never_reach_the_refit_date(baseline_run) -> None:
    _, records = baseline_run
    for record in records:
        assert record.train_end < record.refit_date
        assert record.n_obs > 0


def test_run_is_deterministic(frame: pd.DataFrame) -> None:
    a, _ = B.run_backtest(frame)
    b, _ = B.run_backtest(frame)
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_rolling_window_is_rejected_rather_than_silently_supported() -> None:
    """Out-of-scope options must fail loudly, not produce a result nobody chose."""
    with pytest.raises(NotImplementedError, match="expanding"):
        B.BacktestConfig(estimation_window=B.EstimationWindow.ROLLING)


def test_missing_columns_are_rejected(frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="missing columns"):
        B.run_backtest(frame.drop(columns=["parkinson_var"]))


def test_variances_are_on_the_close_to_close_scale(frame: pd.DataFrame, baseline_run) -> None:
    """The RW baseline must carry the scale constant, not raw Parkinson.

    Without this the RW intervals would be ~23% too narrow and its VaR breaches would be
    a units error rather than a calibration finding.
    """
    forecasts, _ = baseline_run
    rw = forecasts[forecasts.model == "yesterday"]["variance"]
    raw_lagged = frame["parkinson_var"].shift(1).reindex(rw.index)
    ratio = (rw / raw_lagged).dropna()
    np.testing.assert_allclose(ratio.to_numpy(), D.PROXY_SCALE_C, rtol=1e-12)


def test_proxy_column_is_scaled(frame: pd.DataFrame, baseline_run) -> None:
    forecasts, _ = baseline_run
    sub = forecasts[forecasts.model == "ewma"]
    raw = frame["parkinson_var"].reindex(sub.index)
    np.testing.assert_allclose(
        sub["proxy_var"].to_numpy(), (raw * D.PROXY_SCALE_C).to_numpy()
    )
