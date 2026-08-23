"""Data acquisition, integrity, and construction of the analysis frame.

Responsibilities, in pipeline order:

1. Download raw daily OHLC + adjusted close for SPY and the VIX close, and cache them
   as CSV under ``data/raw/`` with a SHA-256 manifest (decision B3: Yahoo Finance
   revises history, so a download script alone is not reproducible).
2. Verify a cached snapshot against the manifest before use.
3. Derive the three distinct quantities the project depends on, which must never be
   conflated:
     - ``log_return``       -- observed daily log return from **adjusted** close.
                               This is the calibration target for intervals and VaR.
     - ``parkinson_var``    -- a high/low range estimator of **intraday** variance.
                               This is the target for point-forecast loss only.
     - ``vix_close_lagged`` -- the previous day's VIX close, used for regime labels.
4. Assign regime labels from the *lagged* VIX close.

Look-ahead discipline
---------------------
Every function here is either a pointwise transform or an explicitly lagged one. No
function may use a rolling, expanding, or full-sample statistic computed over data that
includes observations after the row it is assigned to. Scaling constants that need
estimation (see open item D2) are estimated in ``backtest``/``evaluation`` from the
training window only, never here.

The download buffer (decision D8)
---------------------------------
Raw data is downloaded from ``DOWNLOAD_START``, one month before the locked
``SAMPLE_START``. The lagged quantities (``log_return`` needs the previous adjusted
close, ``regime`` needs the previous VIX close) are otherwise undefined on the first
in-sample row, which would silently shorten the locked training window. The buffer
feeds the lag only: ``build_analysis_frame`` trims its output to
``[SAMPLE_START, SAMPLE_END]``, so no buffer row ever reaches the backtest.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# --- Locked constants -------------------------------------------------------------
# Mirrored from the locked design decisions; see README.md and research_log.md 1.1.

SPY_TICKER = "SPY"
VIX_TICKER = "^VIX"

SAMPLE_START = "2014-01-01"
SAMPLE_END = "2025-06-30"

# One month of lead-in, used only to define the lag on the first in-sample row.
# Never enters the analysis frame. See decision D8.
DOWNLOAD_START = "2013-12-01"

TRAIN_START = "2014-01-02"
TRAIN_END = "2016-12-30"

OOS_START = "2017-01-03"
OOS_END = "2025-06-30"

# Regime cut-points on the LAGGED VIX close.
VIX_CALM_MAX = 15.0
VIX_STRESSED_MIN = 25.0

REGIME_CALM = "calm"
REGIME_NORMAL = "normal"
REGIME_STRESSED = "stressed"

REGIME_ORDER: tuple[str, str, str] = (REGIME_CALM, REGIME_NORMAL, REGIME_STRESSED)

MANIFEST_NAME = "manifest.json"

# Canonical column names. Nothing downstream may depend on the provider's spelling.
RAW_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "adj_close", "volume")

PARKINSON_FACTOR = 1.0 / (4.0 * np.log(2.0))


@dataclass(frozen=True)
class DataQualityReport:
    """Everything that went wrong or looked odd while building the analysis frame.

    This is returned rather than logged-and-discarded so that data problems are
    surfaced in the report instead of silently absorbed. Never suppress these.

    Attributes
    ----------
    n_rows:
        Rows in the final analysis frame.
    missing_dates_spy:
        Dates within the sample window on which ^VIX traded but SPY has no bar.
    calendar_mismatch_dates:
        Dates within the sample window on which the SPY and VIX calendars disagreed,
        in either direction. These rows are dropped, never forward-filled from a later
        observation (decision D5).
    nonpositive_range_dates:
        Dates where high <= low, which makes the Parkinson estimator undefined.
    zero_return_dates:
        Dates with an exactly zero log return, usually a stale-price artefact.
    """

    n_rows: int
    missing_dates_spy: list[pd.Timestamp]
    calendar_mismatch_dates: list[pd.Timestamp]
    nonpositive_range_dates: list[pd.Timestamp]
    zero_return_dates: list[pd.Timestamp]

    def format_full(self) -> str:
        """Render every finding with every affected date listed and none elided.

        Truncating this to a count would defeat the point of returning it: the Stage 0
        acceptance criterion is that any dropped date is visible.
        """

        def block(title: str, dates: list[pd.Timestamp]) -> str:
            if not dates:
                return f"  {title}: none"
            listed = "\n".join(f"      {pd.Timestamp(d).date()}" for d in dates)
            return f"  {title}: {len(dates)}\n{listed}"

        return "\n".join(
            [
                "DataQualityReport",
                f"  rows in analysis frame: {self.n_rows}",
                block("SPY bar missing on a VIX trading day", self.missing_dates_spy),
                block("SPY/VIX calendar mismatch (row dropped)", self.calendar_mismatch_dates),
                block("high <= low, Parkinson undefined (NaN)", self.nonpositive_range_dates),
                block("exactly zero log return", self.zero_return_dates),
            ]
        )


# --- Raw data acquisition and integrity -------------------------------------------


def cache_filename(ticker: str) -> str:
    """Filesystem-safe CSV name for a ticker: ``^VIX`` -> ``VIX.csv``."""
    stem = "".join(ch for ch in ticker if ch.isalnum())
    if not stem:
        raise ValueError(f"ticker {ticker!r} has no filesystem-safe representation")
    return f"{stem}.csv"


def _normalise_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a yfinance frame to the canonical index and column names.

    yfinance returns MultiIndex columns in some configurations and a flat index in
    others, and the spelling of ``Adj Close`` has moved between versions. Both are
    absorbed here so nothing downstream depends on either.
    """
    if isinstance(df.columns, pd.MultiIndex):
        price_names = {"Open", "High", "Low", "Close", "Adj Close"}
        keep = [
            i
            for i in range(df.columns.nlevels)
            if set(df.columns.get_level_values(i)) & price_names
        ]
        level = keep[0] if keep else 0
        df = df.copy()
        df.columns = df.columns.get_level_values(level)

    rename = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Adj_Close": "adj_close",
        "adjclose": "adj_close",
        "Volume": "volume",
    }
    df = df.rename(columns=rename)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    if "adj_close" not in df.columns and "close" in df.columns:
        # ^VIX is an index: there is no corporate action to adjust for, so the
        # adjusted close is the close. Made explicit rather than silently assumed.
        df["adj_close"] = df["close"]

    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"download is missing required columns: {missing}")

    df = df[list(RAW_COLUMNS)].copy()
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = pd.DatetimeIndex(idx).normalize()
    df.index.name = "date"
    return df.sort_index()


def download_raw(
    ticker: str,
    start: str,
    end: str,
    raw_dir: Path,
    *,
    refresh: bool = False,
) -> Path:
    """Download one ticker's daily bars to ``raw_dir`` and return the CSV path.

    Downloads with ``auto_adjust=False`` so that raw OHLC and the adjusted close are
    both retained and the adjustment is auditable (decision D5).

    Refuses to overwrite an existing cached file unless ``refresh`` is True, so that a
    routine pipeline run can never silently replace the committed snapshot that
    published results were computed from.

    ``end`` is inclusive here, unlike yfinance's own exclusive end date.

    Raises
    ------
    FileExistsError
        If the cache file exists and ``refresh`` is False.
    """
    raw_dir = Path(raw_dir)
    path = raw_dir / cache_filename(ticker)
    if path.exists() and not refresh:
        raise FileExistsError(
            f"{path} already exists; pass refresh=True (or --refresh) to overwrite "
            "the committed snapshot"
        )

    import yfinance as yf  # lazy: the offline tests must not need the network

    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    df = yf.download(
        ticker,
        start=start,
        end=end_exclusive,
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        raise RuntimeError(f"yfinance returned no rows for {ticker} over {start}..{end}")

    df = _normalise_raw(df)
    raw_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index_label="date", date_format="%Y-%m-%d", lineterminator="\n")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(65536)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(raw_dir: Path) -> Path:
    """Write a SHA-256 manifest of every CSV in ``raw_dir``; return the manifest path.

    ``downloaded_utc`` is the file's modification time, i.e. when the snapshot was
    written. It is metadata for the reader, not an input to any check: the hash is
    what ``verify_manifest`` compares.
    """
    raw_dir = Path(raw_dir)
    entries: dict[str, dict[str, object]] = {}
    for csv_path in sorted(raw_dir.glob("*.csv")):
        df = pd.read_csv(csv_path, index_col="date", parse_dates=["date"])
        mtime = datetime.fromtimestamp(csv_path.stat().st_mtime, tz=timezone.utc)
        entries[csv_path.name] = {
            "sha256": _sha256(csv_path),
            "n_rows": int(len(df)),
            "first_date": str(pd.Timestamp(df.index.min()).date()),
            "last_date": str(pd.Timestamp(df.index.max()).date()),
            "downloaded_utc": mtime.isoformat(timespec="seconds"),
        }
    manifest_path = raw_dir / MANIFEST_NAME
    manifest_path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def verify_manifest(raw_dir: Path) -> dict[str, bool]:
    """Check each cached CSV against the manifest. Maps filename -> hash matches.

    Callers must treat any False as fatal rather than proceeding on altered data. A
    CSV on disk but absent from the manifest is False, as is a manifest entry whose
    file has gone missing: both mean the snapshot on disk is not the one described.
    """
    raw_dir = Path(raw_dir)
    manifest_path = raw_dir / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"no manifest at {manifest_path}; run write_manifest first"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    names = set(manifest) | {p.name for p in raw_dir.glob("*.csv")}
    result: dict[str, bool] = {}
    for name in sorted(names):
        path = raw_dir / name
        recorded = manifest.get(name)
        result[name] = bool(
            recorded is not None
            and path.exists()
            and _sha256(path) == recorded.get("sha256")
        )
    return result


def load_raw(ticker: str, raw_dir: Path, *, verify: bool = True) -> pd.DataFrame:
    """Load a cached raw CSV as a DatetimeIndex-ed frame, optionally verifying its hash.

    Returns
    -------
    A frame indexed by date with columns ``open``, ``high``, ``low``, ``close``,
    ``adj_close``, ``volume``. Column names are normalised to lowercase here so that
    nothing downstream depends on the provider's capitalisation.

    Raises
    ------
    ValueError
        If ``verify`` is True and the file's hash does not match the manifest. The
        data is not returned in that case: a mismatch means any result computed from
        it would not be the published one.
    """
    raw_dir = Path(raw_dir)
    name = cache_filename(ticker)
    path = raw_dir / name
    if not path.exists():
        raise FileNotFoundError(f"no cached snapshot at {path}")

    if verify:
        status = verify_manifest(raw_dir)
        if not status.get(name, False):
            raise ValueError(
                f"{path} does not match its manifest entry. The cached snapshot has "
                "been altered or replaced; refusing to return it."
            )

    df = pd.read_csv(path, index_col="date", parse_dates=["date"])
    df.index = pd.DatetimeIndex(df.index).normalize()
    df.index.name = "date"
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df[list(RAW_COLUMNS)].sort_index()


# --- Derived quantities -----------------------------------------------------------


def compute_log_returns(adj_close: pd.Series) -> pd.Series:
    """Daily log returns from adjusted closes: r_t = log(P_t) - log(P_{t-1}).

    The first observation is NaN by construction. This is the **observed return** --
    the calibration target for predictive intervals and VaR, and not the same object
    as the Parkinson proxy.
    """
    prices = pd.to_numeric(adj_close, errors="coerce").astype(float)
    prices = prices.where(prices > 0.0)
    return np.log(prices).diff().rename("log_return")


def parkinson_variance(high: pd.Series, low: pd.Series) -> pd.Series:
    """Parkinson (1980) daily variance estimator: log(H/L)^2 / (4 * log 2).

    This estimates the variance of the **intraday continuous** price path. It excludes
    the overnight gap and is therefore biased low relative to the close-to-close
    variance the GARCH models forecast. See open item D2 in research_log.md before
    using this in a loss function.

    Invariant to a within-day multiplicative price adjustment, since the ratio H/L is
    unchanged by it.

    NaN where ``high <= low`` or either input is missing. Those dates are recorded in
    the DataQualityReport and are never zeroed: a zero variance would assert the day
    was riskless, which is not what a degenerate range means.
    """
    h = pd.to_numeric(high, errors="coerce").astype(float)
    lo = pd.to_numeric(low, errors="coerce").astype(float)
    valid = (h > 0.0) & (lo > 0.0) & (h > lo)
    ratio = (h / lo).where(valid)
    return (PARKINSON_FACTOR * np.log(ratio) ** 2).rename("parkinson_var")


def assign_vix_regime(
    vix_close: pd.Series,
    *,
    calm_max: float = VIX_CALM_MAX,
    stressed_min: float = VIX_STRESSED_MIN,
) -> pd.Series:
    """Label each date's regime from the **previous** day's VIX close.

    The regime attached to forecast date t uses the VIX close at t-1, so the label is
    in the information set at the time the forecast for t is made. Passing an
    already-lagged series would double-lag it; this function does the lagging itself.

    Bins: ``< calm_max`` calm, ``[calm_max, stressed_min]`` normal (inclusive at both
    ends), ``> stressed_min`` stressed.

    Returns an ordered categorical over {calm, normal, stressed}; the first
    observation is NaN because no lagged value exists.
    """
    lagged = pd.to_numeric(vix_close, errors="coerce").astype(float).shift(1)
    labels = np.where(
        lagged < calm_max,
        REGIME_CALM,
        np.where(lagged > stressed_min, REGIME_STRESSED, REGIME_NORMAL),
    )
    labels = np.where(lagged.isna().to_numpy(), None, labels)
    return pd.Series(
        pd.Categorical(labels, categories=list(REGIME_ORDER), ordered=True),
        index=vix_close.index,
        name="regime",
    )


def build_analysis_frame(
    spy_raw: pd.DataFrame,
    vix_raw: pd.DataFrame,
    *,
    sample_start: str = SAMPLE_START,
    sample_end: str = SAMPLE_END,
) -> tuple[pd.DataFrame, DataQualityReport]:
    """Join SPY and VIX and derive every column the rest of the pipeline consumes.

    Returns
    -------
    frame:
        Indexed by trading date, with columns:
          ``log_return``       observed daily log return (calibration target)
          ``parkinson_var``    intraday variance proxy (point-loss target only)
          ``vix_close_lagged`` previous day's VIX close
          ``regime``           categorical regime label from the lagged VIX
    report:
        Data quality findings. Callers must surface these, not swallow them.

    Dates where the SPY and VIX calendars disagree are dropped and recorded in the
    report; VIX is never forward-filled from a later observation (decision D5).

    Derived columns are computed on the full joined history and only then trimmed to
    ``[sample_start, sample_end]``, so the lag on the first in-sample row comes from
    real data rather than being left NaN (decision D8). ``sample_start`` and
    ``sample_end`` are keyword-only additions to the original scaffold signature.
    """
    spy = spy_raw.sort_index()
    vix = vix_raw.sort_index()

    common = spy.index.intersection(vix.index)
    mismatch = spy.index.symmetric_difference(vix.index)
    missing_spy = vix.index.difference(spy.index)

    spy_c = spy.loc[common]
    vix_close = pd.to_numeric(vix.loc[common, "close"], errors="coerce").astype(float)

    frame = pd.DataFrame(
        {
            "log_return": compute_log_returns(spy_c["adj_close"]),
            "parkinson_var": parkinson_variance(spy_c["high"], spy_c["low"]),
            "vix_close_lagged": vix_close.shift(1),
            "regime": assign_vix_regime(vix_close),
        },
        index=common,
    )
    frame.index.name = "date"

    lo, hi = pd.Timestamp(sample_start), pd.Timestamp(sample_end)
    frame = frame.loc[(frame.index >= lo) & (frame.index <= hi)]

    def in_window(idx: pd.Index) -> list[pd.Timestamp]:
        return [pd.Timestamp(d) for d in idx if lo <= pd.Timestamp(d) <= hi]

    report = DataQualityReport(
        n_rows=int(len(frame)),
        missing_dates_spy=in_window(missing_spy),
        calendar_mismatch_dates=in_window(mismatch),
        nonpositive_range_dates=list(frame.index[frame["parkinson_var"].isna()]),
        zero_return_dates=list(frame.index[frame["log_return"] == 0.0]),
    )
    return frame, report


def split_train_oos(
    frame: pd.DataFrame,
    *,
    train_start: str = TRAIN_START,
    train_end: str = TRAIN_END,
    oos_start: str = OOS_START,
    oos_end: str = OOS_END,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split chronologically into the initial training block and the OOS block.

    Purely date-based and strictly ordered. There is no random splitting anywhere in
    this project, and this function must never acquire a shuffle option.
    """
    if pd.Timestamp(train_end) >= pd.Timestamp(oos_start):
        raise ValueError("train_end must fall strictly before oos_start")
    idx = frame.index
    train_mask = (idx >= pd.Timestamp(train_start)) & (idx <= pd.Timestamp(train_end))
    oos_mask = (idx >= pd.Timestamp(oos_start)) & (idx <= pd.Timestamp(oos_end))
    return frame.loc[train_mask], frame.loc[oos_mask]
