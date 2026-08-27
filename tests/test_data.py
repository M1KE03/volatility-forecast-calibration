"""Stage 0 tests for ``src.data``.

Two categories, kept visually separate because they fail for different reasons:

- **Transformation tests** ask whether the arithmetic is right. They compare against
  closed-form answers, never against a previous run of this code.
- **Look-ahead tests** ask whether anything sees the future. The pattern is: build the
  frame, corrupt every input row after some date ``t``, rebuild, and assert that
  nothing at or before ``t`` moved. That pattern is the template for every look-ahead
  test in this project and is what protects the conclusions rather than the arithmetic.

No test here touches the network or the committed snapshot in ``data/raw/``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import data as D

# --- Synthetic fixtures ------------------------------------------------------------
# Deliberately hand-made rather than sampled from the real snapshot, so that every
# expected value below is derivable by hand.


def _bdays(n: int, start: str = "2020-01-01") -> pd.DatetimeIndex:
    idx = pd.bdate_range(start=start, periods=n, name="date")
    return pd.DatetimeIndex(idx)


def make_spy(n: int = 40, start: str = "2020-01-01") -> pd.DataFrame:
    """A SPY-shaped frame with a strictly positive range and a *varying* return path.

    The variation matters. An earlier version of this fixture used a constant-growth
    path, which made every daily return identical -- and a full-sample statistic
    (a demeaned return, say) is then numerically invisible, so the look-ahead test
    below silently could not fail. Deterministic, seeded, but genuinely uneven.
    """
    idx = _bdays(n, start)
    rng = np.random.default_rng(20240102)
    steps = rng.normal(0.0004, 0.011, size=n)
    steps[0] = 0.0
    close = 100.0 * np.exp(np.cumsum(steps))
    spread = 0.004 + 0.010 * rng.random(n)
    return pd.DataFrame(
        {
            "open": close * (1.0 - 0.3 * spread),
            "high": close * (1.0 + spread),
            "low": close * (1.0 - spread),
            "close": close,
            "adj_close": close,
            "volume": np.full(n, 1_000_000.0),
        },
        index=idx,
    )


def make_vix(n: int = 40, start: str = "2020-01-01") -> pd.DataFrame:
    """A VIX-shaped frame whose close sweeps across both regime cut-points."""
    idx = _bdays(n, start)
    close = np.linspace(10.0, 35.0, n)
    return pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "adj_close": close,
            "volume": np.zeros(n),
        },
        index=idx,
    )


def write_cache(raw_dir: Path, ticker: str, frame: pd.DataFrame) -> Path:
    path = raw_dir / D.cache_filename(ticker)
    frame.to_csv(path, index_label="date", date_format="%Y-%m-%d", lineterminator="\n")
    return path


# ===================================================================================
# Transformation correctness
# ===================================================================================


def test_log_returns_exact_on_geometric_series() -> None:
    """Plan 2.3.1 -- a constant-growth path has a constant, analytically known return."""
    g = 1.0173
    n = 25
    prices = pd.Series(50.0 * g ** np.arange(n), index=_bdays(n))

    got = D.compute_log_returns(prices)

    assert math.isnan(got.iloc[0]), "the first return is undefined by construction"
    np.testing.assert_allclose(got.iloc[1:].to_numpy(), np.log(g), rtol=0, atol=1e-12)


def test_parkinson_on_ratio_e_is_one_over_four_ln_two() -> None:
    """Plan 2.3.2 -- with log(H/L) = 1 the estimator collapses to its constant."""
    high = pd.Series([math.e], index=_bdays(1))
    low = pd.Series([1.0], index=_bdays(1))

    got = D.parkinson_variance(high, low).iloc[0]

    assert got == pytest.approx(1.0 / (4.0 * math.log(2.0)), rel=0, abs=1e-15)


@pytest.mark.parametrize("scale", [0.25, 1.0, 7.0, 1234.5])
def test_parkinson_invariant_to_price_scaling(scale: float) -> None:
    """Plan 2.3.3 -- H/L is unchanged by a within-day multiplicative adjustment.

    This is what makes the adjusted-versus-raw price question moot for the proxy
    (decision D5), so it is asserted rather than assumed.
    """
    spy = make_spy()
    base = D.parkinson_variance(spy["high"], spy["low"])
    scaled = D.parkinson_variance(spy["high"] * scale, spy["low"] * scale)

    pd.testing.assert_series_equal(base, scaled)


def test_parkinson_is_nan_not_zero_on_degenerate_range() -> None:
    """A high <= low day is undefined, not riskless. Zeroing it would be a silent lie."""
    idx = _bdays(3)
    high = pd.Series([10.0, 10.0, 11.0], index=idx)
    low = pd.Series([9.0, 10.0, 12.0], index=idx)

    got = D.parkinson_variance(high, low)

    assert got.iloc[0] > 0.0
    assert math.isnan(got.iloc[1]), "high == low must be NaN"
    assert math.isnan(got.iloc[2]), "high < low must be NaN"


@pytest.mark.parametrize(
    ("vix_value", "expected"),
    [
        (14.999, D.REGIME_CALM),
        (15.0, D.REGIME_NORMAL),
        (20.0, D.REGIME_NORMAL),
        (25.0, D.REGIME_NORMAL),
        (25.001, D.REGIME_STRESSED),
    ],
)
def test_regime_boundaries_are_closed_on_normal(vix_value: float, expected: str) -> None:
    """Plan 2.3.4 -- the locked spec is ``< 15`` calm and ``> 25`` stressed.

    Both cut-points are therefore inside ``normal``. Getting either boundary wrong
    would move days between regimes in exactly the table the project reports.
    """
    series = pd.Series([vix_value, vix_value], index=_bdays(2))

    got = D.assign_vix_regime(series)

    assert got.iloc[1] == expected


def test_regime_at_t_is_the_bin_of_the_input_at_t_minus_one() -> None:
    """Plan 2.3.5 -- the label on date t must come from the VIX close on t-1."""
    vix = make_vix()

    regimes = D.assign_vix_regime(vix["close"])

    assert pd.isna(regimes.iloc[0]), "no lagged value exists for the first row"
    for t in range(1, len(vix)):
        prev = vix["close"].iloc[t - 1]
        expected = (
            D.REGIME_CALM
            if prev < D.VIX_CALM_MAX
            else D.REGIME_STRESSED
            if prev > D.VIX_STRESSED_MIN
            else D.REGIME_NORMAL
        )
        assert regimes.iloc[t] == expected


def test_regime_is_an_ordered_categorical() -> None:
    """Downstream regime tables sort on this; an unordered category sorts alphabetically."""
    got = D.assign_vix_regime(make_vix()["close"])

    assert isinstance(got.dtype, pd.CategoricalDtype)
    assert got.dtype.ordered
    assert list(got.cat.categories) == list(D.REGIME_ORDER)


# ===================================================================================
# Look-ahead
# ===================================================================================


@pytest.mark.parametrize("t_pos", [5, 12, 20, 33])
def test_no_future_information_reaches_an_earlier_row(t_pos: int) -> None:
    """Plan 2.3.6 -- the corrupt-the-future test, the template for the whole project.

    Every input row strictly after ``t`` is replaced with NaN and the frame rebuilt.
    Nothing at or before ``t`` may move.

    Note the deviation from the plan's wording, which said "from ``t`` onward".
    Row ``t``'s own ``log_return`` and ``parkinson_var`` are by definition functions of
    row ``t``'s inputs, so NaN-ing row ``t`` itself would change them for a reason that
    is not look-ahead. Corrupting strictly after ``t`` is the test that isolates
    forward leakage. The complementary direction -- that the *regime* on ``t`` does not
    use the VIX close on ``t`` -- is asserted separately below.
    """
    spy, vix = make_spy(), make_vix()
    window = {"sample_start": "2020-01-01", "sample_end": "2020-12-31"}

    full, _ = D.build_analysis_frame(spy, vix, **window)
    t = full.index[t_pos]

    spy_c, vix_c = spy.copy(), vix.copy()
    spy_c.loc[spy_c.index > t, :] = np.nan
    vix_c.loc[vix_c.index > t, :] = np.nan
    corrupted, _ = D.build_analysis_frame(spy_c, vix_c, **window)

    pd.testing.assert_frame_equal(full.loc[:t], corrupted.loc[:t])


def test_regime_on_t_ignores_the_vix_close_on_t() -> None:
    """The other half of the look-ahead check for the regime label specifically.

    Perturbing the VIX close on date ``t`` must not change the regime assigned to
    ``t``; it must change the regime assigned to ``t+1``. A missing lag passes the
    first assertion's negation and fails here.
    """
    spy, vix = make_spy(), make_vix()
    window = {"sample_start": "2020-01-01", "sample_end": "2020-12-31"}
    base, _ = D.build_analysis_frame(spy, vix, **window)

    t_pos = 15
    t = vix.index[t_pos]
    bumped = vix.copy()
    bumped.loc[t, "close"] = 99.0  # unambiguously stressed

    got, _ = D.build_analysis_frame(spy, bumped, **window)

    assert got.loc[t, "regime"] == base.loc[t, "regime"]
    assert got.iloc[t_pos + 1]["regime"] == D.REGIME_STRESSED


def test_split_is_chronological_ordered_and_lossless() -> None:
    """Plan 2.3.7 -- train ends strictly before OOS begins, with no reordering."""
    spy, vix = make_spy(n=60), make_vix(n=60)
    frame, _ = D.build_analysis_frame(
        spy, vix, sample_start="2020-01-01", sample_end="2020-12-31"
    )
    boundary = 30

    train, oos = D.split_train_oos(
        frame,
        train_start=str(frame.index[0].date()),
        train_end=str(frame.index[boundary - 1].date()),
        oos_start=str(frame.index[boundary].date()),
        oos_end=str(frame.index[-1].date()),
    )

    assert train.index.max() < oos.index.min()
    assert train.index.is_monotonic_increasing and oos.index.is_monotonic_increasing
    union = train.index.append(oos.index)
    assert union.equals(frame.index), "no row lost, duplicated, or reordered"


def test_split_rejects_an_overlapping_boundary() -> None:
    """An overlapping split would leak training data into the OOS period."""
    frame, _ = D.build_analysis_frame(make_spy(), make_vix())

    with pytest.raises(ValueError):
        D.split_train_oos(
            frame,
            train_start="2020-01-01",
            train_end="2020-02-01",
            oos_start="2020-01-15",
            oos_end="2020-03-01",
        )


# ===================================================================================
# Integrity and data quality
# ===================================================================================


def test_verify_manifest_detects_an_altered_byte(tmp_path: Path) -> None:
    """Plan 2.3.8 -- the hash is the whole point of committing the snapshot."""
    write_cache(tmp_path, D.SPY_TICKER, make_spy())
    D.write_manifest(tmp_path)
    assert D.verify_manifest(tmp_path) == {"SPY.csv": True}

    path = tmp_path / "SPY.csv"
    raw = bytearray(path.read_bytes())
    digit_pos = next(i for i, b in enumerate(raw) if raw[i : i + 1].isdigit() and i > 20)
    raw[digit_pos] = ord("9") if raw[digit_pos] != ord("9") else ord("8")
    path.write_bytes(bytes(raw))

    assert D.verify_manifest(tmp_path) == {"SPY.csv": False}


def test_verify_manifest_flags_an_untracked_or_missing_file(tmp_path: Path) -> None:
    """A file the manifest does not describe is not a verified file."""
    write_cache(tmp_path, D.SPY_TICKER, make_spy())
    D.write_manifest(tmp_path)
    write_cache(tmp_path, D.VIX_TICKER, make_vix())

    status = D.verify_manifest(tmp_path)

    assert status == {"SPY.csv": True, "VIX.csv": False}

    (tmp_path / "SPY.csv").unlink()
    assert D.verify_manifest(tmp_path)["SPY.csv"] is False


def test_load_raw_raises_on_hash_mismatch_rather_than_returning_data(
    tmp_path: Path,
) -> None:
    """Plan 2.3.9 -- warn-and-continue would publish numbers from unknown data."""
    write_cache(tmp_path, D.SPY_TICKER, make_spy())
    D.write_manifest(tmp_path)

    manifest_path = tmp_path / D.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["SPY.csv"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="does not match its manifest"):
        D.load_raw(D.SPY_TICKER, tmp_path, verify=True)

    # ...and the escape hatch is explicit, never the default.
    assert len(D.load_raw(D.SPY_TICKER, tmp_path, verify=False)) == 40


def test_load_raw_roundtrips_the_canonical_columns(tmp_path: Path) -> None:
    """Nothing downstream may depend on the provider's column capitalisation."""
    original = make_spy()
    write_cache(tmp_path, D.SPY_TICKER, original)
    D.write_manifest(tmp_path)

    got = D.load_raw(D.SPY_TICKER, tmp_path)

    assert list(got.columns) == list(D.RAW_COLUMNS)
    assert isinstance(got.index, pd.DatetimeIndex)
    np.testing.assert_allclose(got["adj_close"], original["adj_close"])


def test_download_raw_refuses_to_overwrite_the_cache(tmp_path: Path) -> None:
    """Plan 2.3.10 -- a routine run must not silently replace the published snapshot.

    The guard fires before any network call, which is also why this test is offline.
    """
    write_cache(tmp_path, D.SPY_TICKER, make_spy())

    with pytest.raises(FileExistsError):
        D.download_raw(D.SPY_TICKER, "2014-01-01", "2025-06-30", tmp_path, refresh=False)


def test_calendar_mismatches_are_dropped_and_reported() -> None:
    """Plan 2.3.11 -- disagreeing calendars are dropped, never filled (decision D5).

    A forward fill here would attach a *later* VIX close to an earlier date, which is
    look-ahead dressed up as data cleaning.
    """
    spy, vix = make_spy(), make_vix()
    spy_only = spy.index[7]
    vix_only = vix.index[13]
    spy = spy.drop(index=vix_only)
    vix = vix.drop(index=spy_only)

    frame, report = D.build_analysis_frame(
        spy, vix, sample_start="2020-01-01", sample_end="2020-12-31"
    )

    assert spy_only not in frame.index
    assert vix_only not in frame.index
    assert set(report.calendar_mismatch_dates) == {spy_only, vix_only}
    assert report.missing_dates_spy == [vix_only]
    assert not frame.isna().all(axis=1).any(), "dropped, not filled with NaN rows"
    assert report.n_rows == len(frame)


def test_quality_report_records_degenerate_ranges_and_zero_returns() -> None:
    """Findings are returned, not logged and discarded."""
    spy, vix = make_spy(), make_vix()
    bad_range = spy.index[4]
    stale = spy.index[9]
    spy.loc[bad_range, "low"] = spy.loc[bad_range, "high"]
    spy.loc[stale, "adj_close"] = spy["adj_close"].iloc[8]

    frame, report = D.build_analysis_frame(
        spy, vix, sample_start="2020-01-01", sample_end="2020-12-31"
    )

    assert report.nonpositive_range_dates == [bad_range]
    assert math.isnan(frame.loc[bad_range, "parkinson_var"])
    assert stale in report.zero_return_dates
    assert "none" not in D.DataQualityReport(
        n_rows=1,
        missing_dates_spy=[bad_range],
        calendar_mismatch_dates=[bad_range],
        nonpositive_range_dates=[bad_range],
        zero_return_dates=[bad_range],
    ).format_full().split("rows in analysis frame")[1]


def test_analysis_frame_is_trimmed_to_the_sample_window_but_lags_use_the_buffer() -> None:
    """Decision D8 -- the download buffer feeds the lag and nothing else.

    Without it the first in-sample row would carry a NaN return and a NaN regime,
    quietly shortening the locked training window.
    """
    spy, vix = make_spy(n=60), make_vix(n=60)
    sample_start = str(spy.index[10].date())
    sample_end = str(spy.index[-1].date())

    frame, _ = D.build_analysis_frame(
        spy, vix, sample_start=sample_start, sample_end=sample_end
    )

    assert frame.index[0] == spy.index[10], "trimmed to the sample window"
    first = frame.iloc[0]
    assert not math.isnan(first["log_return"])
    assert not pd.isna(first["regime"])
    assert first["vix_close_lagged"] == pytest.approx(vix["close"].iloc[9])


# ===================================================================================
# The committed snapshot itself
# ===================================================================================

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

pytestmark_snapshot = pytest.mark.skipif(
    not (RAW_DIR / D.MANIFEST_NAME).exists(),
    reason="no committed snapshot present; run `python run_all.py --stage data`",
)


@pytestmark_snapshot
def test_committed_snapshot_matches_its_manifest() -> None:
    """The snapshot in the repository is the one the manifest describes."""
    status = D.verify_manifest(RAW_DIR)

    assert status, "manifest is empty"
    assert all(status.values()), f"hash mismatch: {status}"


@pytestmark_snapshot
def test_snapshot_bytes_are_lf_only() -> None:
    """The snapshot's line endings must survive a checkout on any platform.

    ``core.autocrlf=true`` plus no ``.gitattributes`` rewrites LF to CRLF on checkout,
    which changes the bytes, breaks every SHA-256, and makes the pipeline refuse to run
    from a fresh clone. ``.gitattributes`` marks these files ``-text``; this asserts the
    result. The failure mode is invisible in the authoring worktree, which is precisely
    why it is checked here rather than trusted.
    """
    for csv_path in sorted(RAW_DIR.glob("*.csv")):
        assert b"\r\n" not in csv_path.read_bytes(), (
            f"{csv_path.name} contains CRLF; its manifest hash cannot match. "
            "Check .gitattributes."
        )


@pytestmark_snapshot
def test_real_frame_covers_the_locked_windows() -> None:
    """The locked sample, training, and OOS windows are all actually populated."""
    spy = D.load_raw(D.SPY_TICKER, RAW_DIR)
    vix = D.load_raw(D.VIX_TICKER, RAW_DIR)

    frame, report = D.build_analysis_frame(spy, vix)
    train, oos = D.split_train_oos(frame)

    assert frame.index.min() >= pd.Timestamp(D.SAMPLE_START)
    assert frame.index.max() <= pd.Timestamp(D.SAMPLE_END)
    assert frame.index.is_monotonic_increasing and frame.index.is_unique
    assert train.index.max() < oos.index.min()
    assert len(train) > 700 and len(oos) > 2000
    assert report.n_rows == len(frame)
    # The first in-sample row must be usable: decision D8's whole purpose.
    assert frame["log_return"].notna().all()
    assert frame["regime"].notna().all()


# --- The proxy scale constant c ----------------------------------------------------

@pytest.fixture(scope="module")
def analysis_frame() -> pd.DataFrame:
    """The committed analysis frame. ``c`` is a property of the real data, not a mock."""
    path = Path("data/processed/analysis_frame.csv")
    if not path.exists():  # pragma: no cover - depends on local state
        pytest.skip(f"{path} not present; run `python run_all.py --stage data`")
    return pd.read_csv(path, index_col=0, parse_dates=True)


#
# The master convention puts every variance in the project on the close-to-close
# return-variance scale. ``c`` is the constant that gets the Parkinson proxy there. It
# is estimated on the warm-up sample and frozen, so the tests that matter are the ones
# proving it cannot move in response to out-of-sample data.


def test_proxy_scale_cannot_see_out_of_sample_data(analysis_frame: pd.DataFrame) -> None:
    """The look-ahead test for ``c``.

    ``c`` multiplies every RW forecast and the evaluation proxy on every out-of-sample
    day. If it could be moved by out-of-sample observations, the entire point-loss table
    and the RW baseline's intervals would depend on the period they are evaluated over,
    and nothing would fail to announce it.
    """
    baseline = D.compute_proxy_scale(analysis_frame)

    corrupted = analysis_frame.copy()
    oos_mask = corrupted.index >= pd.Timestamp(D.OOS_START)
    assert oos_mask.sum() > 0
    corrupted.loc[oos_mask, "log_return"] = 0.5
    corrupted.loc[oos_mask, "parkinson_var"] = 0.25

    assert D.compute_proxy_scale(corrupted) == baseline


def test_proxy_scale_does_move_with_warm_up_data(analysis_frame: pd.DataFrame) -> None:
    """Proves the previous test is not vacuous."""
    baseline = D.compute_proxy_scale(analysis_frame)
    corrupted = analysis_frame.copy()
    warm_mask = corrupted.index <= pd.Timestamp(D.TRAIN_END)
    corrupted.loc[warm_mask, "log_return"] = corrupted.loc[warm_mask, "log_return"] * 2.0
    assert D.compute_proxy_scale(corrupted) != baseline


def test_frozen_proxy_scale_matches_a_fresh_computation(
    analysis_frame: pd.DataFrame,
) -> None:
    """The hard-coded constant must agree with the committed snapshot.

    ``PROXY_SCALE_C`` is hard-coded so that published numbers cannot drift silently if
    the frame is ever rebuilt differently. That protection only works if something
    checks the two against each other, which is this test.
    """
    computed = D.compute_proxy_scale(analysis_frame)
    assert abs(computed - D.PROXY_SCALE_C) < D.PROXY_SCALE_TOL


def test_proxy_scale_is_greater_than_one(analysis_frame: pd.DataFrame) -> None:
    """Parkinson misses overnight moves, so it must understate close-to-close variance.

    A value at or below 1 would mean the intraday range explained the whole daily move,
    which for SPY would indicate the proxy or the returns were misaligned.
    """
    assert 1.0 < D.compute_proxy_scale(analysis_frame) < 3.0


def test_proxy_scale_refuses_a_window_reaching_into_the_evaluation_period(
    analysis_frame: pd.DataFrame,
) -> None:
    """Widening the window must raise rather than quietly returning a leaked constant."""
    with pytest.raises(ValueError, match="warm-up data only"):
        D.compute_proxy_scale(analysis_frame, train_end="2020-12-31")


def test_scale_proxy_is_a_pure_multiplication(analysis_frame: pd.DataFrame) -> None:
    raw = analysis_frame["parkinson_var"]
    scaled = D.scale_proxy(raw, c=2.0)
    np.testing.assert_allclose(scaled.to_numpy(), raw.to_numpy() * 2.0)


def test_scale_proxy_rejects_a_degenerate_constant(analysis_frame: pd.DataFrame) -> None:
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and positive"):
            D.scale_proxy(analysis_frame["parkinson_var"], c=bad)


# --- The trailing-volatility regime sensitivity (Stage 5) --------------------------


def test_trailing_vol_regime_uses_only_information_through_yesterday() -> None:
    """The same lag discipline as ``assign_vix_regime``, and for the same reason.

    The label attached to date t must be computable by someone standing at the close of
    t-1. Corrupting the proxy from a cut date onward must leave every label strictly
    before that date untouched.
    """
    rng = np.random.default_rng(20260827)
    index = pd.bdate_range("2014-01-02", periods=600)
    proxy = pd.Series(np.abs(rng.normal(1e-4, 3e-5, size=600)), index=index)

    thresholds = D.trailing_vol_thresholds(proxy, train_end="2015-06-30")
    clean = D.assign_trailing_vol_regime(proxy, thresholds=thresholds)

    cut = index[400]
    corrupted_proxy = proxy.copy()
    corrupted_proxy.loc[cut:] *= 50.0
    corrupted = D.assign_trailing_vol_regime(corrupted_proxy, thresholds=thresholds)

    before = index < cut
    assert (clean[before].astype(str) == corrupted[before].astype(str)).all()


def test_corrupting_the_proxy_does_move_later_trailing_vol_labels() -> None:
    """The vacuity check beside the test above: the corruption has to bite somewhere."""
    rng = np.random.default_rng(20260827)
    index = pd.bdate_range("2014-01-02", periods=600)
    proxy = pd.Series(np.abs(rng.normal(1e-4, 3e-5, size=600)), index=index)
    thresholds = D.trailing_vol_thresholds(proxy, train_end="2015-06-30")

    clean = D.assign_trailing_vol_regime(proxy, thresholds=thresholds)
    corrupted_proxy = proxy.copy()
    corrupted_proxy.iloc[400:] *= 50.0
    corrupted = D.assign_trailing_vol_regime(corrupted_proxy, thresholds=thresholds)

    after = index > index[420]
    assert (clean[after].astype(str) != corrupted[after].astype(str)).any()


def test_trailing_vol_thresholds_cannot_see_the_evaluation_period() -> None:
    """The reason the cut-points are a separate function from the labeller.

    Terciles taken over the full sample would be a function of the evaluation period, so
    every label would depend on days that had not happened yet -- and the crisis regime
    would be defined using the crisis. The locked VIX thresholds have this property by
    construction at 15 and 25; this alternative has to earn it.
    """
    frame_path = Path("data/processed/analysis_frame.csv")
    if not frame_path.exists():  # pragma: no cover - depends on local state
        pytest.skip(f"{frame_path} not present; run `python run_all.py --stage data`")
    frame = pd.read_csv(frame_path, index_col=0, parse_dates=True)

    baseline = D.trailing_vol_thresholds(frame["parkinson_var"])
    corrupted = frame["parkinson_var"].copy()
    corrupted.loc["2017-01-03":] *= 100.0
    assert D.trailing_vol_thresholds(corrupted) == baseline


def test_trailing_vol_thresholds_do_move_with_warm_up_data() -> None:
    """And the vacuity check: warm-up data must matter, or the test above is empty."""
    frame_path = Path("data/processed/analysis_frame.csv")
    if not frame_path.exists():  # pragma: no cover - depends on local state
        pytest.skip(f"{frame_path} not present; run `python run_all.py --stage data`")
    frame = pd.read_csv(frame_path, index_col=0, parse_dates=True)

    baseline = D.trailing_vol_thresholds(frame["parkinson_var"])
    bumped = frame["parkinson_var"].copy()
    bumped.loc[:D.TRAIN_END] *= 3.0
    assert D.trailing_vol_thresholds(bumped) != baseline


def test_trailing_vol_labels_are_the_ordered_regime_categories() -> None:
    """They reuse REGIME_ORDER because they play the same role -- but they are a
    different partition, and no table may compare a row across the two definitions."""
    rng = np.random.default_rng(20260827)
    index = pd.bdate_range("2014-01-02", periods=400)
    proxy = pd.Series(np.abs(rng.normal(1e-4, 3e-5, size=400)), index=index)
    labels = D.assign_trailing_vol_regime(proxy, train_end="2015-06-30")
    assert list(labels.cat.categories) == list(D.REGIME_ORDER)
    assert labels.iloc[: D.TRAILING_VOL_WINDOW].isna().all()


def test_unordered_trailing_vol_thresholds_are_refused() -> None:
    rng = np.random.default_rng(20260827)
    index = pd.bdate_range("2014-01-02", periods=200)
    proxy = pd.Series(np.abs(rng.normal(1e-4, 3e-5, size=200)), index=index)
    with pytest.raises(ValueError, match="ordered"):
        D.assign_trailing_vol_regime(proxy, thresholds=(5.0, 1.0))
