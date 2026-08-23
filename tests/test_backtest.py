"""Tests for ``src.backtest`` -- the look-ahead audit.

The governing plan's cut list names this audit as something never to drop, and the
reason is worth restating: a calibration result computed on a leaking backtest is worse
than no result at all, because it looks fine. Every other number this project will
produce is downstream of the loop these tests cover.

Three categories:

- **Look-ahead tests** ask whether anything sees the future. The master test is
  ``test_master_look_ahead_corrupting_the_future_leaves_the_past_identical``; the others
  cover the specific mechanisms by which leakage would arrive.
- **Contract tests** pin the schema, the refit cadence, and the invariants the
  evaluation layer will rely on.
- **Two-cadence tests** (Stage 2) check that parameters are held between refits while
  the variance recursion keeps advancing daily.

Running cost, and why the marker exists
---------------------------------------
Since Stage 2 a full run refits GARCH 204 times and takes about a minute, so the tests
that need one are marked ``slow`` and ``pytest -m "not slow"`` is the fast inner loop.
Plain ``pytest`` still runs them: the audit is on the never-cut list, and a default test
run must not be the thing that skips it.

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

from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from src import backtest as B
from src import data as D
from src import models as M

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
def full_run(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[B.RefitRecord]]:
    """One uncorrupted run of the whole backtest, shared by every test that needs it.

    Module-scoped because it now costs about a minute: 102 GARCH refits per variant.
    Recomputing it per test would put several minutes on the suite for no extra
    coverage, since it is the same deterministic run each time --
    ``test_run_is_deterministic`` is what establishes that.
    """
    return B.run_backtest(frame)


# --- Look-ahead tests --------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("corrupt_from", ["2018-02-05", "2020-03-16", "2022-06-13"])
def test_master_look_ahead_corrupting_the_future_leaves_the_past_identical(
    frame: pd.DataFrame, full_run, corrupt_from: str
) -> None:
    """The master test. If this passes, the pipeline has no forward leakage.

    Chosen dates are the worst cases available: the Volmageddon spike, the largest COVID
    drawdown day, and a 2022 selloff. If any forecast reached forward, these are the
    days on which the corruption would be most visible.

    Since Stage 2 this covers the GARCH models too, which is the first time it has had
    anything to say: both baselines are parameter-free, so until now the test could not
    have caught a refit that estimated on data it should not have seen. Every route by
    which the future could reach a forecast -- the estimation window, the backcast seed,
    the daily filter -- is now inside its scope.
    """
    baseline, _ = full_run

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
    for model in B.MODELS:
        a = baseline[(baseline.model == model) & past][FORECAST_COLUMNS]
        b = recomputed[(recomputed.model == model) & past][FORECAST_COLUMNS]
        pd.testing.assert_frame_equal(a, b, check_exact=True)

    # And strictly before the cut, the evaluation columns are untouched too.
    strictly_past = baseline.index < cut
    for model in B.MODELS:
        a = baseline[(baseline.model == model) & strictly_past]
        b = recomputed[(recomputed.model == model) & strictly_past]
        pd.testing.assert_frame_equal(a, b, check_exact=True)


@pytest.mark.slow
def test_corruption_actually_changes_the_future(frame: pd.DataFrame, full_run) -> None:
    """Proves the master test is not passing vacuously.

    If the corruption had no effect anywhere, the assertions above would hold for a
    backtest that ignored its input entirely.
    """
    baseline, _ = full_run
    cut = pd.Timestamp("2020-03-16")
    corrupted = frame.copy()
    mask = corrupted.index >= cut
    corrupted.loc[mask, "log_return"] = 0.5
    corrupted.loc[mask, "parkinson_var"] = 0.05
    recomputed, _ = B.run_backtest(corrupted)

    future = baseline.index > cut
    for model in B.MODELS:
        a = baseline[(baseline.model == model) & future]["variance"]
        b = recomputed[(recomputed.model == model) & future]["variance"]
        assert not np.allclose(a.to_numpy(), b.to_numpy()), (
            f"corrupting the future did not move {model}'s later forecasts"
        )


def test_random_walk_forecast_never_uses_its_own_days_proxy(
    frame: pd.DataFrame, full_run
) -> None:
    """Invariant 5: the day-``t`` proxy is an evaluation target, never an input.

    Asserted directly on the numbers: the RW forecast for day ``t`` must equal the
    scaled proxy at ``t-1``, and must differ from the scaled proxy at ``t``.
    """
    forecasts, _ = full_run
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


def test_no_reordering_of_dates(full_run) -> None:
    """Invariant 6. A shuffle anywhere would invalidate every serial-dependence claim."""
    forecasts, _ = full_run
    for model in B.MODELS:
        idx = forecasts[forecasts.model == model].index
        assert idx.is_monotonic_increasing
        assert idx.is_unique


# --- Contract tests ----------------------------------------------------------------


def test_forecast_table_is_complete_over_the_evaluation_window(
    frame: pd.DataFrame, full_run
) -> None:
    """Stage 1's headline acceptance criterion."""
    forecasts, _ = full_run
    oos = frame.loc[pd.Timestamp(D.OOS_START) : pd.Timestamp(D.OOS_END)]

    for model in B.MODELS:
        sub = forecasts[forecasts.model == model]
        assert len(sub) == len(oos) == 2134
        assert sub.index.equals(pd.DatetimeIndex(oos.index))
        for column in FORECAST_COLUMNS + ["pit"]:
            assert sub[column].notna().all(), f"{model} has gaps in {column}"


def test_pit_lies_in_the_unit_interval(full_run) -> None:
    forecasts, _ = full_run
    pit = forecasts["pit"].dropna()
    assert pit.between(0.0, 1.0).all()


def test_interval_bounds_are_correctly_ordered(full_run) -> None:
    """Wider nominal level implies a wider interval, and VaR sits inside the 99% band.

    ``var_99`` is the 1% quantile while ``lo_99`` is the 0.5% quantile of a two-sided
    99% interval, so the one-sided VaR must be the less extreme of the two. Getting this
    backwards is an easy and silent error that would misstate every VaR breach count.
    """
    forecasts, _ = full_run
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


def test_refit_id_is_constant_between_refits(full_run) -> None:
    """Non-decreasing, and each id spans exactly one refit period.

    ``records`` now holds one row per (refit, model), so the number of refits is the
    number of distinct ``refit_id`` values rather than the length of the list.
    """
    forecasts, records = full_run
    n_refits = len({record.refit_id for record in records})
    ids = forecasts[forecasts.model == "ewma"]["refit_id"]
    assert ids.is_monotonic_increasing
    assert ids.iloc[0] == 0
    assert ids.max() == n_refits - 1
    counts = ids.value_counts()
    assert (counts.drop(ids.max()) == 21).all(), "refit blocks must be 21 days"


def test_refit_records_never_reach_the_refit_date(full_run) -> None:
    _, records = full_run
    for record in records:
        assert record.train_end < record.refit_date
        assert record.n_obs > 0


@pytest.mark.slow
def test_run_is_deterministic(frame: pd.DataFrame, full_run) -> None:
    """Bit-for-bit, including the 204 GARCH fits.

    The multi-start grid is fixed rather than random precisely so that this holds: a
    published number that moves between runs cannot be checked by anyone.
    """
    a, _ = full_run
    b, _ = B.run_backtest(frame)
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_rolling_window_is_rejected_rather_than_silently_supported() -> None:
    """Out-of-scope options must fail loudly, not produce a result nobody chose."""
    with pytest.raises(NotImplementedError, match="expanding"):
        B.BacktestConfig(estimation_window=B.EstimationWindow.ROLLING)


def test_missing_columns_are_rejected(frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="missing columns"):
        B.run_backtest(frame.drop(columns=["parkinson_var"]))


def test_variances_are_on_the_close_to_close_scale(frame: pd.DataFrame, full_run) -> None:
    """The RW baseline must carry the scale constant, not raw Parkinson.

    Without this the RW intervals would be ~23% too narrow and its VaR breaches would be
    a units error rather than a calibration finding.
    """
    forecasts, _ = full_run
    rw = forecasts[forecasts.model == "yesterday"]["variance"]
    raw_lagged = frame["parkinson_var"].shift(1).reindex(rw.index)
    ratio = (rw / raw_lagged).dropna()
    np.testing.assert_allclose(ratio.to_numpy(), D.PROXY_SCALE_C, rtol=1e-12)


def test_proxy_column_is_scaled(frame: pd.DataFrame, full_run) -> None:
    forecasts, _ = full_run
    sub = forecasts[forecasts.model == "ewma"]
    raw = frame["parkinson_var"].reindex(sub.index)
    np.testing.assert_allclose(
        sub["proxy_var"].to_numpy(), (raw * D.PROXY_SCALE_C).to_numpy()
    )


# ===================================================================================
# Stage 2: the GARCH models in the loop
# ===================================================================================

GARCH_PARAM_COLUMNS = ["mu", "omega", "alpha", "beta"]


def _records_frame(records: list[B.RefitRecord]) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in records])


@pytest.mark.slow
def test_variance_moves_daily_while_parameters_are_held_fixed(full_run) -> None:
    """**The two-cadence test.** Parameters refit monthly; the variance filters daily.

    This is the bug the module docstring warns about, and it is invisible in aggregate:
    a GARCH that froze ``h`` between refits would still produce a full forecast table,
    still track volatility roughly, and still score plausibly. It would simply be using
    a variance up to 21 days stale, which is not a one-day-ahead GARCH forecast at all.

    Both halves have to be asserted. That ``refit_id`` is constant across a block says
    the parameters were held; that ``variance`` changes every single day within that
    same block says the recursion kept running anyway.
    """
    forecasts, _ = full_run
    for model in B.GARCH_MODELS:
        sub = forecasts[forecasts.model == model]
        blocks = sub.groupby("refit_id")
        assert blocks.ngroups == 102

        for refit_id, block in blocks:
            if len(block) < 2:
                continue
            assert block["refit_id"].nunique() == 1
            assert block["mean"].nunique() == 1, (
                f"{model} block {refit_id}: the predictive mean moved between refits"
            )
            assert block["variance"].nunique() == len(block), (
                f"{model} block {refit_id}: variance is not updating daily"
            )


@pytest.mark.slow
def test_parameters_are_constant_within_a_refit_block(full_run) -> None:
    """The complement of the test above, read off the refit records.

    One estimate per block and no more: a fit leaking into a neighbouring block would
    mean the forecasts for those days used parameters from a window that had seen them.
    """
    _, records = full_run
    frame = _records_frame(records)
    for model in B.GARCH_MODELS:
        fits = frame[frame["model"] == model]
        assert len(fits) == 102
        assert fits["refit_id"].is_unique


@pytest.mark.slow
def test_every_garch_refit_estimates_on_data_before_its_refit_date(full_run) -> None:
    """Invariant 2, now that it has something to constrain.

    Until Stage 2 both models were parameter-free, so this could not have failed. It is
    the single line separating a walk-forward backtest from an in-sample fit.
    """
    _, records = full_run
    frame = _records_frame(records)
    fits = frame[frame["model"].isin(B.GARCH_MODELS)]
    assert not fits.empty
    assert (fits["train_end"] < fits["refit_date"]).all()
    assert (fits["n_obs"] > 0).all()
    # Expanding window: every fit starts at the same place and sees more each time.
    assert fits.groupby("model")["train_start"].nunique().eq(1).all()
    for _, group in fits.groupby("model"):
        assert group.sort_values("refit_id")["n_obs"].is_monotonic_increasing


@pytest.mark.slow
def test_every_refit_converged(full_run) -> None:
    """Reported, not assumed.

    A non-converged refit is a legitimate outcome that produces no forecasts for its
    block, so it must be visible rather than hidden behind a gap in a table. This test
    asserts the current state of the real run: all 102 refits converge for both
    variants. If that ever changes, the failure message says which one, and the right
    response is to report it -- not to relax the assertion.
    """
    _, records = full_run
    frame = _records_frame(records)
    fits = frame[frame["model"].isin(B.GARCH_MODELS)]
    failed = fits[~fits["mle_converged"]]
    assert failed.empty, (
        "refits did not converge:\n"
        + failed[["model", "refit_date", "mle_message"]].to_string()
    )
    assert np.isfinite(fits["mle_loglik"]).all()


@pytest.mark.slow
def test_a_failed_refit_produces_no_forecasts_rather_than_stale_ones(
    frame: pd.DataFrame, monkeypatch
) -> None:
    """The honesty contract, end to end.

    Carrying the previous window's parameters forward when a fit fails would hide the
    failure behind entirely plausible numbers -- which is the one outcome the
    ``converged`` flag exists to prevent. The alternative, a 21-day hole in the forecast
    table, is loud and is what the evaluation layer will see.

    One refit is forced to fail here rather than hunting for real data that makes the
    optimiser stumble, because the behaviour under failure should not depend on which
    day happens to trigger it.
    """
    config = B.BacktestConfig()
    oos_index = pd.DatetimeIndex(
        frame.loc[pd.Timestamp(D.OOS_START) : pd.Timestamp(D.OOS_END)].index
    )
    refit_dates = B.make_refit_dates(oos_index, refit_every=config.refit_every)
    doomed = refit_dates[50]
    doomed_n_obs = B.estimation_slice(
        pd.DatetimeIndex(frame.index), doomed, config
    ).stop

    real_fit = M.fit_garch_mle

    def failing_fit(returns, **kwargs):
        fit = real_fit(returns, **kwargs)
        # Fail exactly the refit whose estimation window ends just before ``doomed``.
        # The window length identifies it uniquely, because the window expands.
        if len(returns) == doomed_n_obs:
            return M.FrequentistFit(
                params=fit.params,
                loglik=float("nan"),
                converged=False,
                message="ABNORMAL: forced failure",
                n_obs=fit.n_obs,
            )
        return fit

    monkeypatch.setattr(M, "fit_garch_mle", failing_fit)
    path, records = B.build_garch_paths(frame, config, model="garch_mle")

    failed = [r for r in records if not r.mle_converged]
    assert len(failed) == 1, "the forced failure did not land on exactly one refit"
    assert failed[0].refit_date == doomed

    block = path.loc[doomed : refit_dates[51]].iloc[:-1]
    assert len(block) == config.refit_every
    assert block["variance"].isna().all(), "a failed refit still produced forecasts"

    # And the neighbouring blocks are unaffected: the failure is contained, not
    # contagious. A fit that had been quietly carried forward would show up here as a
    # block that is complete when it should be empty.
    assert path.drop(index=block.index)["variance"].notna().all()


@pytest.mark.slow
def test_garch_variances_are_on_the_close_to_close_return_scale(
    frame: pd.DataFrame, full_run
) -> None:
    """Decision D10, checked where it would be easiest to break.

    The GARCH models are driven by squared returns and so arrive on the right scale
    already -- which means the way to get this wrong is to apply the Parkinson constant
    ``c`` to them as well. That would inflate every GARCH variance by 52% and turn the
    calibration comparison into a units comparison.

    Asserted against the mean squared return, which is the close-to-close variance the
    forecasts are supposed to be tracking. A band rather than a point: these are
    forecasts, not an identity.
    """
    forecasts, _ = full_run
    mean_squared_return = float(
        (frame.loc[pd.Timestamp(D.OOS_START) : pd.Timestamp(D.OOS_END), "log_return"] ** 2)
        .mean()
    )
    for model in B.GARCH_MODELS:
        ratio = forecasts[forecasts.model == model]["variance"].mean() / mean_squared_return
        assert 0.5 < ratio < 2.0, f"{model} mean variance is {ratio:.2f}x mean r^2"
        # Specifically not off by the proxy scale constant in either direction.
        assert abs(ratio - D.PROXY_SCALE_C) > 0.2
        assert abs(ratio - 1.0 / D.PROXY_SCALE_C) > 0.2


@pytest.mark.slow
def test_student_t_intervals_are_wider_in_the_tail_than_gaussian_ones(full_run) -> None:
    """The Stage 6 ablation's mechanism, pinned now that both variants exist.

    At the 99% level the innovation distribution is most of what sets the interval
    width, so the t model's VaR must sit further into the loss tail than the normal
    model's on essentially every day. If it did not, the Stage 6 comparison would be
    measuring something other than the innovation assumption.

    Deliberately checked at 99% and not at 90%: near the centre the two distributions
    are close and the fitted variances differ enough to reverse the ordering, which is
    itself the reason the ablation is run at the tail.
    """
    forecasts, _ = full_run
    t_model = forecasts[forecasts.model == "garch_mle"]
    normal = forecasts[forecasts.model == "garch_mle_normal"]
    assert t_model.index.equals(normal.index)

    more_extreme = (t_model["var_99"].to_numpy() < normal["var_99"].to_numpy()).mean()
    assert more_extreme > 0.95, (
        f"GARCH-t 99% VaR is more extreme on only {more_extreme:.1%} of days"
    )

    t_width = (t_model["hi_99"] - t_model["lo_99"]).mean()
    normal_width = (normal["hi_99"] - normal["lo_99"]).mean()
    assert normal_width < t_width


@pytest.mark.slow
def test_garch_predictive_mean_is_the_fitted_mu_not_zero(full_run) -> None:
    """The GARCH models estimate ``mu``; the baselines do not (decisions D11, D12).

    The asymmetry is intentional and worth pinning, because a GARCH that quietly
    inherited the baselines' zero mean would shift every interval by the same small
    amount and never announce it.
    """
    forecasts, records = full_run
    fits = _records_frame(records)

    for model in B.GARCH_MODELS:
        sub = forecasts[forecasts.model == model]
        assert (sub["mean"] != 0.0).all()
        assert set(np.round(sub["mean"].unique(), 12)) <= set(
            np.round(fits[fits["model"] == model]["mu"].unique(), 12)
        )

    for model in B.BASELINE_MODELS:
        assert (forecasts[forecasts.model == model]["mean"] == 0.0).all()


def test_build_garch_paths_rejects_a_model_it_does_not_fit(frame: pd.DataFrame) -> None:
    """Out-of-scope arguments fail loudly rather than returning something plausible."""
    with pytest.raises(ValueError, match="not a GARCH model"):
        B.build_garch_paths(frame, B.BacktestConfig(), model="ewma")


def test_the_model_registry_is_internally_consistent() -> None:
    """The tuples the evaluation layer will filter on must agree with each other."""
    assert B.MODELS == B.BASELINE_MODELS + B.GARCH_MODELS
    assert set(B.HEADLINE_MODELS) < set(B.MODELS)
    assert "garch_mle_normal" not in B.HEADLINE_MODELS, (
        "the ablation must not appear in the headline comparison"
    )
    assert set(B.INNOVATION_BY_MODEL) == set(B.GARCH_MODELS)
