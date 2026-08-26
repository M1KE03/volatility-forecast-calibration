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

import json
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


#: The models the audit runs with the real estimators in place.
#:
#: The Bayesian track is audited too, but separately and against a recording stub -- see
#: the section at the bottom of this file for why that is the stronger arrangement rather
#: than a concession, and ``test_bayesian_look_ahead_with_the_real_sampler`` for the
#: confirmation run that uses NUTS itself. Both halves are under the same corruption
#: dates and the same claim.
AUDITED_MODELS: tuple[str, ...] = B.FREQUENTIST_MODELS


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
    return B.run_backtest(frame, models=AUDITED_MODELS)


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

    recomputed, _ = B.run_backtest(corrupted, models=AUDITED_MODELS)

    # Forecasts up to and including the cut date use data through cut-1 only.
    past = baseline.index <= cut
    for model in AUDITED_MODELS:
        a = baseline[(baseline.model == model) & past][FORECAST_COLUMNS]
        b = recomputed[(recomputed.model == model) & past][FORECAST_COLUMNS]
        pd.testing.assert_frame_equal(a, b, check_exact=True)

    # And strictly before the cut, the evaluation columns are untouched too.
    strictly_past = baseline.index < cut
    for model in AUDITED_MODELS:
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
    recomputed, _ = B.run_backtest(corrupted, models=AUDITED_MODELS)

    future = baseline.index > cut
    for model in AUDITED_MODELS:
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
    for model in AUDITED_MODELS:
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

    for model in AUDITED_MODELS:
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
    b, _ = B.run_backtest(frame, models=AUDITED_MODELS)
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_rolling_window_is_rejected_rather_than_silently_supported() -> None:
    """Out-of-scope options must fail loudly, not produce a result nobody chose."""
    with pytest.raises(NotImplementedError, match="expanding"):
        B.BacktestConfig(estimation_window=B.EstimationWindow.ROLLING)


def test_missing_columns_are_rejected(frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="missing columns"):
        B.run_backtest(frame.drop(columns=["parkinson_var"]), models=AUDITED_MODELS)


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
    failed = fits[~fits["converged"]]
    assert failed.empty, (
        "refits did not converge:\n"
        + failed[["model", "refit_date", "message"]].to_string()
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

    failed = [r for r in records if not r.converged]
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
    assert B.MODELS == B.BASELINE_MODELS + B.GARCH_MODELS + B.BAYES_MODELS
    assert B.MODELS == B.FREQUENTIST_MODELS + B.BAYES_MODELS
    assert set(B.HEADLINE_MODELS) < set(B.MODELS)
    assert B.BAYES_MODEL in B.HEADLINE_MODELS, (
        "the model the project is named after belongs in the headline comparison"
    )
    assert set().union(*B.TRACKS.values()) == set(B.MODELS), (
        "every model must belong to a track, or a stage will silently never build it"
    )
    assert B.BAYES_MEAN_MODEL not in B.HEADLINE_MODELS, (
        "the posterior-mean plug-in splits a comparison; it does not enter one"
    )
    assert set(B.PLUGIN_MODELS) < set(B.MODELS)
    assert B.BAYES_MODEL not in B.PLUGIN_MODELS, (
        "the mixture predictive must never be treated as a plug-in"
    )
    assert "garch_mle_normal" not in B.HEADLINE_MODELS, (
        "the ablation must not appear in the headline comparison"
    )
    assert set(B.INNOVATION_BY_MODEL) == set(B.GARCH_MODELS)


# --- The two-track split and its merge ---------------------------------------------
#
# Plumbing, but plumbing that decides what every downstream table is computed from. A
# merge that silently joined two runs made under different configurations would produce
# a forecast file whose rows answered different questions, and nothing further down could
# detect it.


def _fake_track_output(
    model: str, n: int = 5
) -> tuple[pd.DataFrame, list[B.RefitRecord]]:
    """A minimal forecast table and refit record for one model."""
    dates = pd.bdate_range("2017-01-03", periods=n)
    forecasts = pd.DataFrame(
        {
            "model": model,
            "variance": np.linspace(1e-4, 2e-4, n),
            "mean": 0.0,
            "lo_90": -0.01,
            "hi_90": 0.01,
            "pit": np.linspace(0.1, 0.9, n),
        },
        index=pd.Index(dates, name="date"),
    )
    records = [
        B.RefitRecord(
            refit_id=0,
            refit_date=dates[0],
            train_start=dates[0],
            train_end=dates[-1],
            n_obs=n,
            model=model,
        )
    ]
    return forecasts, records


def test_merged_table_is_the_union_of_the_tracks(tmp_path) -> None:
    config = B.BacktestConfig()
    freq, freq_records = _fake_track_output("garch_mle")
    bayes, bayes_records = _fake_track_output("garch_bayes")

    B.save_track("frequentist", freq, freq_records, config, tmp_path)
    B.save_track("bayes", bayes, bayes_records, config, tmp_path)

    merged = pd.read_csv(tmp_path / "forecasts.csv", index_col=0, parse_dates=True)
    assert set(merged["model"]) == {"garch_mle", "garch_bayes"}
    assert len(merged) == len(freq) + len(bayes)
    # Report order within a date, so a rebuilt file is comparable with the last one.
    first_day = merged.loc[merged.index[0]]
    assert list(first_day["model"]) == ["garch_mle", "garch_bayes"]

    records = pd.read_csv(tmp_path / "refit_records.csv")
    assert set(records["model"]) == {"garch_mle", "garch_bayes"}


def test_a_track_written_alone_still_produces_a_merged_table(tmp_path) -> None:
    """The frequentist stage must leave a usable forecasts.csv before the Bayesian
    stage has ever run -- otherwise Stage 3 becomes a prerequisite for Stage 2's own
    output."""
    config = B.BacktestConfig()
    freq, freq_records = _fake_track_output("garch_mle")
    B.save_track("frequentist", freq, freq_records, config, tmp_path)

    merged = pd.read_csv(tmp_path / "forecasts.csv", index_col=0, parse_dates=True)
    assert set(merged["model"]) == {"garch_mle"}


def test_merging_refuses_tracks_run_under_different_configurations(tmp_path) -> None:
    """**The check that stops two half-runs from being read as one.**

    Two tracks made under different proxy scales produce rows that are not comparable,
    and the difference is invisible in the merged file. The merge has to be the thing
    that notices, because nothing downstream can.
    """
    freq, freq_records = _fake_track_output("garch_mle")
    bayes, bayes_records = _fake_track_output("garch_bayes")

    B.save_track("frequentist", freq, freq_records, B.BacktestConfig(), tmp_path)
    with pytest.raises(ValueError, match="different configurations"):
        B.save_track(
            "bayes",
            bayes,
            bayes_records,
            B.BacktestConfig(proxy_scale_c=1.0),
            tmp_path,
        )


def test_a_sampler_setting_does_not_make_two_tracks_incompatible(tmp_path) -> None:
    """The negative control for the check above.

    The frequentist track records whatever sampler settings it was handed and never
    reads them. If a difference there blocked the merge, raising ``target_accept`` would
    force a pointless re-run of a track it cannot affect -- and a check that fires on
    irrelevant differences gets disabled.
    """
    freq, freq_records = _fake_track_output("garch_mle")
    bayes, bayes_records = _fake_track_output("garch_bayes")

    B.save_track(
        "frequentist", freq, freq_records, B.BacktestConfig(target_accept=0.9), tmp_path
    )
    B.save_track(
        "bayes", bayes, bayes_records, B.BacktestConfig(target_accept=0.95), tmp_path
    )

    with (tmp_path / "backtest_config.json").open(encoding="utf-8") as fh:
        merged = json.load(fh)

    # And the merged config reports the settings that were actually sampled under, not
    # the inert copy the frequentist track carried.
    assert merged["target_accept"] == 0.95


def test_run_backtest_rejects_a_model_it_does_not_have(frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="unknown models"):
        B.run_backtest(frame, models=("garch_bayes", "stochastic_vol"))


# --- The Bayesian track's look-ahead audit -----------------------------------------
#
# Run against a **recording stub** in place of NUTS, and the substitution is what makes
# the audit stronger here rather than weaker.
#
# D20 authorised running the real sampler at a reduced draw count, reasoning that draws
# control Monte Carlo precision rather than which data reaches a fit. The reasoning
# holds; the premise does not. Measured at Stage 3, a refit costs 9-17s *before it draws
# anything* -- the floor is compiling the gradient of the scan recursion -- so cutting
# draws by 96% cuts wall clock by about 70%, and six backtest runs x 102 refits puts a
# default ``pytest`` near two hours. An audit that slow gets skipped in practice, which
# is the cut the governing plan's never-cut list exists to prevent. Recorded at
# research_log.md 1.11-1.12 and problems-and-solutions 39.
#
# What the stub changes: nothing the audit is about. Every look-ahead surface --
# ``estimation_slice``, the ``h0`` backcast, the ``returns[:stop]`` slice, the block
# assignment, the per-draw filter, the mixture, the PIT -- runs in the real code path at
# all 102 refit dates. And *which data reached each fit* stops being an inference from
# output equality and becomes an assertion on the array the sampler was actually handed,
# which the real sampler cannot give.
#
# What it does not cover: the sampler's own internals. Those receive nothing but the two
# arrays recorded here, and ``test_bayesian_look_ahead_with_the_real_sampler`` covers
# them end to end behind its own marker.


def _stub_posterior_draws(returns: np.ndarray, n_draws: int = 64) -> np.ndarray:
    """A deterministic stand-in for a posterior, computed from the estimation window.

    Two properties are load-bearing. The draws must **depend on the data**, or corrupting
    the past would not move the forecasts and the audit would pass on a pipeline that
    ignored its input entirely (the failure #24 records). And they must **differ from
    each other**, or the mixture would collapse to a single component and the per-draw
    filter would never be exercised.

    Every draw is admissible by construction: ``alpha + beta`` spans [0.94, 0.96] and
    ``nu`` spans [5, 9].
    """
    values = np.asarray(returns, dtype=float)
    grid = np.linspace(-1.0, 1.0, n_draws)
    alpha = 0.10 + 0.02 * grid
    beta = 0.85 - 0.03 * grid
    variance = float(values.var(ddof=1))
    return np.column_stack(
        [
            np.full(n_draws, float(values.mean())),
            variance * (1.0 - alpha - beta),
            alpha,
            beta,
            7.0 + 2.0 * grid,
        ]
    )


def _install_recording_stub(monkeypatch) -> list[tuple[np.ndarray, float]]:
    """Replace the sampler with the stub; return the list its calls are recorded into."""
    calls: list[tuple[np.ndarray, float]] = []

    def stub(returns, *, h0=None, **kwargs):
        values = np.asarray(returns, dtype=float)
        h0_used = M.backcast_initial_variance(values) if h0 is None else float(h0)
        calls.append((values.copy(), h0_used))
        draws = _stub_posterior_draws(values)
        return M.BayesianFit(
            draws=draws,
            log_prob=np.zeros(draws.shape[0]),
            r_hat=np.ones(len(M.PARAM_NAMES)),
            ess_bulk=np.full(len(M.PARAM_NAMES), 4000.0),
            ess_tail=np.full(len(M.PARAM_NAMES), 4000.0),
            n_divergences=0,
            converged=True,
            message="stub sampler",
            n_obs=int(values.size),
            seed=0,
        )

    monkeypatch.setattr(M, "sample_garch_posterior", stub)
    return calls


@pytest.mark.slow
@pytest.mark.parametrize("corrupt_from", ["2018-02-05", "2020-03-16", "2022-06-13"])
def test_bayesian_look_ahead_corrupting_the_future_leaves_the_past_identical(
    frame: pd.DataFrame, corrupt_from: str, monkeypatch
) -> None:
    """The master test, extended to the model the project is named after.

    Same corruption dates as the frequentist audit, and the same claim: a forecast for a
    date at or before the cut cannot move when every observation from the cut onward is
    replaced with noise.
    """
    cut = pd.Timestamp(corrupt_from)

    _install_recording_stub(monkeypatch)
    baseline, _ = B.run_backtest(frame, models=B.BAYES_MODELS)

    corrupted = frame.copy()
    mask = corrupted.index >= cut
    assert mask.sum() > 0, "corruption window is empty; the test would be vacuous"
    rng = np.random.default_rng(20260826)
    corrupted.loc[mask, "log_return"] = rng.normal(0.0, 0.5, size=int(mask.sum()))
    corrupted.loc[mask, "parkinson_var"] = rng.uniform(0.01, 0.10, size=int(mask.sum()))

    _install_recording_stub(monkeypatch)
    recomputed, _ = B.run_backtest(corrupted, models=B.BAYES_MODELS)

    past = baseline.index <= cut
    pd.testing.assert_frame_equal(
        baseline[past][FORECAST_COLUMNS], recomputed[past][FORECAST_COLUMNS],
        check_exact=True,
    )
    strictly_past = baseline.index < cut
    pd.testing.assert_frame_equal(
        baseline[strictly_past], recomputed[strictly_past], check_exact=True
    )


@pytest.mark.slow
def test_bayesian_corruption_actually_changes_the_future(
    frame: pd.DataFrame, monkeypatch
) -> None:
    """The negative control. Without it the test above would pass on a pipeline whose
    Bayesian forecasts ignored the data entirely -- which is exactly what a stub could
    quietly become if its draws stopped depending on the estimation window."""
    cut = pd.Timestamp("2020-03-16")

    _install_recording_stub(monkeypatch)
    baseline, _ = B.run_backtest(frame, models=B.BAYES_MODELS)

    corrupted = frame.copy()
    mask = corrupted.index >= cut
    corrupted.loc[mask, "log_return"] = 0.5
    corrupted.loc[mask, "parkinson_var"] = 0.05

    _install_recording_stub(monkeypatch)
    recomputed, _ = B.run_backtest(corrupted, models=B.BAYES_MODELS)

    future = baseline.index > cut
    assert not np.allclose(
        baseline[future]["variance"].to_numpy(), recomputed[future]["variance"].to_numpy()
    ), "corrupting the future did not move the Bayesian model's later forecasts"


@pytest.mark.slow
def test_every_bayesian_fit_saw_only_data_from_before_its_refit_date(
    frame: pd.DataFrame, monkeypatch
) -> None:
    """**What the stub buys that the real sampler could not.**

    The audit above infers that no future data reached a fit from the fact that no
    forecast moved. This asserts it directly, on the array each fit was handed: every
    estimation window is a prefix of the sample, ends strictly before its own refit date,
    and grows by exactly the refit cadence. A leak would have to survive being read off
    the sampler's own input.
    """
    calls = _install_recording_stub(monkeypatch)
    config = B.BacktestConfig()
    B.run_backtest(frame, config, models=B.BAYES_MODELS)

    full_index = pd.DatetimeIndex(frame.index)
    returns = frame["log_return"].to_numpy(dtype=float)
    oos_index = pd.DatetimeIndex(
        frame.loc[pd.Timestamp(D.OOS_START) : pd.Timestamp(D.OOS_END)].index
    )
    refit_dates = B.make_refit_dates(oos_index, refit_every=config.refit_every)

    assert len(calls) == len(refit_dates) == 102

    for refit_date, (window, h0) in zip(refit_dates, calls):
        # The window is a prefix of the sample: same values, same order, no gaps.
        np.testing.assert_array_equal(window, returns[: window.size])
        # And it stops before the date it is used to forecast.
        last_used = full_index[window.size - 1]
        assert last_used < refit_date, (
            f"the fit for {refit_date.date()} was handed data through {last_used.date()}"
        )
        # h0 is backcast from that window alone, not from the sample.
        assert h0 == pytest.approx(M.backcast_initial_variance(window), rel=1e-12)

    sizes = [window.size for window, _ in calls]
    assert sizes == sorted(sizes) and len(set(sizes)) == len(sizes)
    assert set(np.diff(sizes)) == {config.refit_every}


@pytest.mark.bayes_audit
def test_bayesian_look_ahead_with_the_real_sampler(frame: pd.DataFrame) -> None:
    """The same audit with NUTS actually sampling. **Not run by default; about 40
    minutes.**

    The stubbed audit above covers every surface by which the future could reach a
    forecast, and covers the sampler's inputs better than this test can. What only this
    one covers is the sampler itself: that a seeded PyMC run over an estimation window is
    a function of that window and nothing else, end to end, in the real code path.

    Reduced draws, per D20 -- the part of that decision that survives measurement. The
    assertions are exact equalities under a fixed seed, and a fixed seed is as exact at
    50 draws as at 2,000. One corruption date rather than three, the largest COVID
    drawdown day, because the marginal date buys less here than the stubbed audit already
    gives.

    Run it deliberately before the write-up and record the result in research_log.md:

        pytest -m bayes_audit
    """
    config = B.BacktestConfig(draws=25, tune=50, chains=2)
    cut = pd.Timestamp("2020-03-16")

    baseline, _ = B.run_backtest(frame, config, models=B.BAYES_MODELS, cores=1)

    corrupted = frame.copy()
    mask = corrupted.index >= cut
    rng = np.random.default_rng(20260826)
    corrupted.loc[mask, "log_return"] = rng.normal(0.0, 0.5, size=int(mask.sum()))
    corrupted.loc[mask, "parkinson_var"] = rng.uniform(0.01, 0.10, size=int(mask.sum()))
    recomputed, _ = B.run_backtest(corrupted, config, models=B.BAYES_MODELS, cores=1)

    past = baseline.index <= cut
    pd.testing.assert_frame_equal(
        baseline[past][FORECAST_COLUMNS], recomputed[past][FORECAST_COLUMNS],
        check_exact=True,
    )


# --- The posterior-mean plug-in, and the decomposition it exists for ----------------


def test_the_posterior_mean_track_is_an_ablation_not_a_competitor() -> None:
    """``garch_bayes_mean`` splits a comparison; it does not enter one.

    It is the plug-in at the same point estimate ``garch_bayes`` integrates over, so
    putting it in the headline table would show the reader two rows that differ by a
    quantity the report is trying to *measure* rather than by a modelling choice anyone
    would make.
    """
    assert B.BAYES_MEAN_MODEL in B.MODELS
    assert B.BAYES_MEAN_MODEL in B.BAYES_MODELS
    assert B.BAYES_MEAN_MODEL not in B.HEADLINE_MODELS
    # It is a plug-in: same treatment of a point estimate as the MLE models get.
    assert B.BAYES_MEAN_MODEL in B.PLUGIN_MODELS
    assert B.BAYES_MODEL not in B.PLUGIN_MODELS


def test_the_posterior_mean_track_gets_a_plug_in_and_the_mixture_track_cannot() -> None:
    """**The substitution that would report parameter uncertainty as zero.**

    ``garch_bayes`` must never fall back on a plug-in built from its own summary
    statistics. If it did, it would equal ``garch_bayes_mean`` by construction, their
    difference would be exactly zero, and the project would conclude that integrating
    over the posterior changes nothing -- from a bug, not from the data.
    """
    plug_in = B._predictive(B.BAYES_MEAN_MODEL, 1.2e-4, 0.0005, 6.0)
    assert isinstance(plug_in, M.StudentTPredictive)

    with pytest.raises(ValueError, match="cannot be rebuilt"):
        B._predictive(B.BAYES_MODEL, 1.2e-4, 0.0005, 6.0)


@pytest.mark.slow
def test_the_two_bayesian_tracks_come_from_the_same_fits(
    frame: pd.DataFrame, monkeypatch
) -> None:
    """One posterior, two things done with it -- so they stand or fall together.

    Same refit dates, same failed blocks, same point estimate. If the two tracks could
    disagree about which days have a forecast, their difference would be taken over an
    unstated and shifting sample.
    """
    _install_recording_stub(monkeypatch)
    forecasts, records = B.run_backtest(frame, models=B.BAYES_MODELS)

    bayes = forecasts[forecasts.model == B.BAYES_MODEL]
    mean = forecasts[forecasts.model == B.BAYES_MEAN_MODEL]

    assert bayes.index.equals(mean.index)
    np.testing.assert_array_equal(
        bayes["variance"].isna().to_numpy(), mean["variance"].isna().to_numpy()
    )
    # The predictive mean is the same point estimate in both.
    np.testing.assert_allclose(
        bayes["mean"].to_numpy(), mean["mean"].to_numpy(), rtol=1e-12
    )
    # One record per refit, not one per model: both rest on a single fit.
    assert len(records) == 102
    assert set(r.model for r in records) == {B.BAYES_MODEL}


@pytest.mark.slow
def test_integrating_the_posterior_widens_the_tail_against_its_own_plug_in(
    frame: pd.DataFrame, monkeypatch
) -> None:
    """**The project's research question, isolated at last.**

    ``garch_bayes`` against ``garch_mle`` is confounded: the priors move the point
    estimate as well, and by more (research_log.md 1.13). Against ``garch_bayes_mean``
    the point estimate is held fixed and the *only* remaining difference is whether the
    posterior is integrated over. At the 99% level that must widen the interval.

    Run against the stub, whose draws are dispersed by construction, so this tests the
    plumbing that isolates the effect rather than the size of the effect in the data --
    which is a result, not an invariant, and belongs in the evaluation layer.
    """
    _install_recording_stub(monkeypatch)
    forecasts, _ = B.run_backtest(frame, models=B.BAYES_MODELS)

    bayes = forecasts[forecasts.model == B.BAYES_MODEL]
    mean = forecasts[forecasts.model == B.BAYES_MEAN_MODEL]
    finite = bayes["variance"].notna().to_numpy()
    assert finite.sum() > 2000, "the comparison must not rest on a handful of days"

    width_99 = (bayes["hi_99"] - bayes["lo_99"]).to_numpy()[finite]
    plug_in_99 = (mean["hi_99"] - mean["lo_99"]).to_numpy()[finite]
    assert np.all(width_99 > plug_in_99), (
        "integrating over the posterior must widen the 99% interval on every day"
    )

    # And the crossover recorded at research_log.md 1.11 shows up here too: the same
    # mixing that fattens the tails thins the shoulders.
    width_90 = (bayes["hi_90"] - bayes["lo_90"]).to_numpy()[finite]
    plug_in_90 = (mean["hi_90"] - mean["lo_90"]).to_numpy()[finite]
    assert np.mean(width_90 / plug_in_90) < np.mean(width_99 / plug_in_99)
