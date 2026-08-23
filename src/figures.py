"""Figure rendering.

Figures are produced here and written by ``run_all.py``, never by a notebook. The report
must be reproducible from the command line alone, which it would not be if any figure
existed only as the side effect of someone having executed a cell.

House style is set once in ``apply_house_style`` so that every figure in the report
shares axes, fonts and colours without each function restating them.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # no interactive backend; figures are written, not shown
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# --- House style -------------------------------------------------------------------

#: Stress-period shading used across figures. Chosen from the locked evaluation window's
#: known episodes, not from anything estimated, so no figure setting depends on a result.
STRESS_PERIODS: tuple[tuple[str, str, str], ...] = (
    ("2018-02-01", "2018-02-28", "Volmageddon"),
    ("2020-02-20", "2020-04-30", "COVID crash"),
    ("2022-01-01", "2022-10-31", "2022 bear market"),
)

COLOURS = {
    "returns": "#31456b",
    "accent": "#b5482e",
    "muted": "#8a8f98",
    "train": "#e8ecf3",
    "grid": "#d8dce3",
}


def apply_house_style() -> None:
    """Set rcParams once. Idempotent, so notebooks may call it freely."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "semibold",
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": COLOURS["grid"],
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
            "legend.frameon": False,
            "legend.fontsize": 8,
        }
    )


def shade_stress_periods(ax: Axes, *, label: bool = True) -> None:
    """Shade the known stress episodes on a date axis.

    Periods lying entirely outside the axis's current x-limits are skipped. Without
    that guard, drawing a 2022 span on a chart of spring 2020 silently stretches the
    axis out to 2022 and squashes the window the figure exists to show.
    """
    lo, hi = ax.get_xlim()
    for start, end, name in STRESS_PERIODS:
        if mdates.date2num(pd.Timestamp(end)) < lo or mdates.date2num(pd.Timestamp(start)) > hi:
            continue
        # Matplotlib accepts datetimes on a date axis but types axvspan as float, so the
        # conversion is made explicit rather than relying on the runtime coercion.
        ax.axvspan(
            mdates.date2num(pd.Timestamp(start)),
            mdates.date2num(pd.Timestamp(end)),
            color=COLOURS["accent"],
            alpha=0.10,
            linewidth=0,
        )
        if label:
            ax.annotate(
                name,
                xy=(pd.Timestamp(start), 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(2, -10),
                textcoords="offset points",
                fontsize=7,
                color=COLOURS["accent"],
            )


def _finish(fig: Figure, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# --- Stage 0 figures ---------------------------------------------------------------


def plot_returns_with_regimes(
    frame: pd.DataFrame, out_path: Path, *, train_end: str
) -> Path:
    """Daily log returns over the full sample: the volatility-clustering picture.

    The training/out-of-sample boundary is drawn because every result in this project is
    conditional on it, and a reader should never have to take its location on trust.
    """
    apply_house_style()
    fig, ax = plt.subplots(figsize=(9.5, 3.4))

    returns = frame["log_return"].dropna()
    ax.plot(returns.index, returns.to_numpy(), linewidth=0.5, color=COLOURS["returns"])

    boundary = mdates.date2num(pd.Timestamp(train_end))
    ax.axvspan(
        mdates.date2num(returns.index.min()),
        boundary,
        color=COLOURS["train"],
        zorder=0,
        linewidth=0,
    )
    ax.axvline(boundary, color=COLOURS["muted"], linewidth=1.0, linestyle="--")
    ax.annotate(
        "training window",
        xy=(returns.index.min(), 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(4, -10),
        textcoords="offset points",
        fontsize=7,
        color=COLOURS["muted"],
    )
    shade_stress_periods(ax)

    ax.set_title(
        "SPY daily log returns: quiet and turbulent periods cluster, "
        "which is what GARCH parameterises"
    )
    ax.set_ylabel("log return")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.margins(x=0.01)
    return _finish(fig, out_path)


def plot_squared_return_acf(
    acf_frame: pd.DataFrame, out_path: Path, *, window_label: str
) -> Path:
    """ACF of squared returns with 95% bands: the evidence behind the ARCH-LM test."""
    apply_house_style()
    fig, ax = plt.subplots(figsize=(7.0, 3.2))

    lags = acf_frame.index.to_numpy()
    values = acf_frame["acf"].to_numpy()
    ax.vlines(lags, 0.0, values, color=COLOURS["returns"], linewidth=1.6)
    ax.scatter(lags, values, s=9, color=COLOURS["returns"], zorder=3)
    ax.fill_between(
        lags,
        acf_frame["ci_lower"].to_numpy(),
        acf_frame["ci_upper"].to_numpy(),
        color=COLOURS["muted"],
        alpha=0.20,
        linewidth=0,
        label="95% band under the no-autocorrelation null",
    )
    ax.axhline(0.0, color="black", linewidth=0.8)

    ax.set_title(f"ACF of squared returns -- {window_label}")
    ax.set_xlabel("lag (trading days)")
    ax.set_ylabel("autocorrelation")
    ax.legend(loc="upper right")
    ax.margins(x=0.01)
    return _finish(fig, out_path)


def plot_return_distribution(returns: pd.Series, out_path: Path) -> Path:
    """Return histogram against a fitted normal: the fat tails that motivate Student-t.

    The normal is fitted by matching mean and standard deviation, which is the
    comparison that makes the tail discrepancy legible -- a fitted t would hide exactly
    the thing the figure exists to show.
    """
    apply_house_style()
    from scipy import stats

    clean = returns.dropna().to_numpy(dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.3))

    ax = axes[0]
    # Outline rather than filled bars: on a log y-axis a filled histogram extends every
    # bar down to the axis floor, so a bin holding one observation looks as solid as one
    # holding two hundred -- which hides precisely the sparse tail this panel is for.
    ax.hist(
        clean,
        bins=80,
        density=True,
        histtype="step",
        color=COLOURS["returns"],
        linewidth=1.0,
    )
    grid = np.linspace(clean.min(), clean.max(), 400)
    ax.plot(
        grid,
        stats.norm.pdf(grid, clean.mean(), clean.std(ddof=1)),
        color=COLOURS["accent"],
        linewidth=1.4,
        label="normal, matched mean and sd",
    )
    ax.set_yscale("log")
    ax.set_title("Return density (log scale reveals the tails)")
    ax.set_xlabel("log return")
    ax.legend(loc="lower center")

    ax = axes[1]
    stats.probplot(clean, dist="norm", plot=ax)
    ax.get_lines()[0].set(
        marker="o", markersize=2.0, color=COLOURS["returns"], alpha=0.6, linestyle="none"
    )
    ax.get_lines()[1].set(color=COLOURS["accent"], linewidth=1.2)
    ax.set_title("Normal Q-Q: departure in both tails")
    ax.set_ylabel("sample quantiles")

    fig.tight_layout()
    return _finish(fig, out_path)


def plot_vix_with_regimes(frame: pd.DataFrame, out_path: Path) -> Path:
    """Lagged VIX with the locked regime thresholds drawn on.

    The thresholds are horizontal lines because they are fixed ex ante at 15 and 25.
    Anything estimated from the sample would have to be drawn as a time-varying band,
    and the visual difference is the point: these cannot have been tuned to a result.
    """
    apply_house_style()
    fig, ax = plt.subplots(figsize=(9.5, 3.0))

    vix = frame["vix_close_lagged"].dropna()
    ax.plot(vix.index, vix.to_numpy(), linewidth=0.7, color=COLOURS["returns"])
    for level, name in ((15.0, "calm / normal"), (25.0, "normal / stressed")):
        ax.axhline(level, color=COLOURS["accent"], linewidth=1.0, linestyle="--")
        # Anchored left, where the early-sample VIX is calmest, and given an opaque
        # background: the series revisits both thresholds repeatedly, so a bare label
        # anywhere on this axis eventually sits on top of the data it describes.
        ax.annotate(
            f"{name} = {level:g}",
            xy=(0.0, level),
            xycoords=("axes fraction", "data"),
            xytext=(4, 3),
            textcoords="offset points",
            ha="left",
            fontsize=7,
            color=COLOURS["accent"],
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.0},
        )

    ax.set_title("Lagged VIX close and the ex-ante regime thresholds")
    ax.set_ylabel("VIX (close, t-1)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.margins(x=0.01)
    return _finish(fig, out_path)


# --- Stage 1 figures ---------------------------------------------------------------

#: Display labels for the model keys used in the forecast frame.
MODEL_LABELS = {
    "yesterday": "RW-in-vol (yesterday's scaled Parkinson)",
    "ewma": "EWMA (RiskMetrics, lambda = 0.94)",
    "garch_mle": "GARCH(1,1)-t, plug-in",
    "garch_bayes": "GARCH(1,1)-t, posterior predictive",
}

MODEL_COLOURS = {
    "yesterday": "#8a8f98",
    "ewma": "#b5482e",
    "garch_mle": "#31456b",
    "garch_bayes": "#2e7d6b",
}


def plot_forecasts_vs_realised(
    forecasts: pd.DataFrame,
    frame: pd.DataFrame,
    out_path: Path,
    *,
    start: str,
    end: str,
) -> Path:
    """Forecast volatility against the realised proxy over a window.

    The Stage 1 acceptance check. An EWMA forecast is a weighted average of past squared
    returns, so it *must* lag a volatility spike on the way in and overshoot on the way
    out -- it has no mechanism to do anything else. If the plotted line instead tracked
    the spike contemporaneously, the recursion would be reading the current day's return
    and the whole backtest would be leaking one day. That failure is far easier to see
    here than in any summary statistic, which is why the plot is an acceptance criterion
    and not a decoration.

    Everything is drawn as annualised volatility rather than variance: variance on a
    crisis window is unreadable, since a 5x volatility move is a 25x variance move.
    """
    apply_house_style()

    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    window = forecasts.loc[lo:hi]
    if window.empty:
        raise ValueError(f"no forecasts between {start} and {end}")

    def annualise(variance: pd.Series) -> pd.Series:
        return np.sqrt(variance.astype(float) * 252.0)

    fig, (ax, ax_r) = plt.subplots(
        2, 1, figsize=(10.0, 5.6), sharex=True, height_ratios=[3, 1]
    )

    realised = window[window.model == window.model.iloc[0]]["proxy_var"]
    ax.fill_between(
        realised.index,
        0.0,
        annualise(realised).to_numpy(),
        color=COLOURS["muted"],
        alpha=0.30,
        linewidth=0,
        label="realised (scaled Parkinson, annualised)",
    )

    for model in window.model.unique():
        sub = window[window.model == model]
        ax.plot(
            sub.index,
            annualise(sub["variance"]).to_numpy(),
            linewidth=1.5,
            color=MODEL_COLOURS.get(str(model), COLOURS["returns"]),
            label=MODEL_LABELS.get(str(model), str(model)),
        )

    # Pin the limits to the requested window before shading, then restore them after,
    # so no annotation can widen the view past what was asked for.
    ax.set_xlim(mdates.date2num(lo), mdates.date2num(hi))
    shade_stress_periods(ax, label=False)
    ax.set_xlim(mdates.date2num(lo), mdates.date2num(hi))

    ax.set_title(
        "One-day-ahead forecasts against realised volatility: "
        "EWMA lags into the spike and overshoots after"
    )
    ax.set_ylabel("annualised volatility")
    ax.legend(loc="upper left")

    returns = frame.loc[lo:hi, "log_return"]
    ax_r.plot(returns.index, returns.to_numpy(), linewidth=0.7, color=COLOURS["returns"])
    ax_r.axhline(0.0, color="black", linewidth=0.6)
    ax_r.set_ylabel("log return")
    ax_r.set_xlim(mdates.date2num(lo), mdates.date2num(hi))
    ax_r.xaxis.set_major_locator(mdates.MonthLocator())
    ax_r.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    for tick in ax_r.get_xticklabels():
        tick.set_fontsize(7)

    fig.tight_layout()
    return _finish(fig, out_path)
