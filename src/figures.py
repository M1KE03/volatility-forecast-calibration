"""Figure rendering.

Figures are produced here and written by ``run_all.py``, never by a notebook. The report
must be reproducible from the command line alone, which it would not be if any figure
existed only as the side effect of someone having executed a cell.

House style is set once in ``apply_house_style`` so that every figure in the report
shares axes, fonts and colours without each function restating them.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # no interactive backend; figures are written, not shown
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# --- House style -------------------------------------------------------------------
#
# The register is a sell-side research note or a journal exhibit: a left-aligned title
# block, a rule under it, horizontal gridlines only, no box around the plot, and a source
# line at the foot. The aim is that a figure lifted out of this repository and dropped
# into a PDF looks like it belongs there without further work.
#
# Three rules the whole file follows:
#
# 1. **Titles state the finding, not the contents.** "EWMA lags into the spike" rather
#    than "Forecasts and realised volatility". A reader who looks only at the exhibits
#    should still come away with the argument.
# 2. **Direct labelling beats a legend** wherever the series can be named at its own
#    right-hand end, because it removes the colour-matching step entirely.
# 3. **The source line is not decoration.** Every figure names the table it was drawn
#    from, so a number in a slide can be traced back to a file without asking anyone.

#: Stress-period shading used across figures. Chosen from the locked evaluation window's
#: known episodes, not from anything estimated, so no figure setting depends on a result.
STRESS_PERIODS: tuple[tuple[str, str, str], ...] = (
    ("2018-02-01", "2018-02-28", "Volmageddon"),
    ("2020-02-20", "2020-04-30", "COVID crash"),
    ("2022-01-01", "2022-10-31", "2022 bear market"),
)

#: Preferred faces, most-wanted first. Segoe UI ships with Windows and is the closest
#: thing to a neutral research-note sans that is guaranteed present here; the rest are
#: fallbacks so a figure rendered on another machine degrades rather than breaks.
FONT_STACK: tuple[str, ...] = (
    "Segoe UI",
    "Helvetica Neue",
    "Helvetica",
    "Arial",
    "DejaVu Sans",
)

COLOURS = {
    # Text and furniture.
    "ink": "#11161D",        # titles, axis labels, the darkest thing on the page
    "muted": "#6B7480",      # subtitles, source lines, de-emphasised series
    "rule": "#B9C1CC",       # the rule under the title block
    "grid": "#E7EBF0",       # horizontal gridlines, behind everything
    "panel": "#FFFFFF",
    # Data.
    "returns": "#1F3864",    # primary navy
    "accent": "#B23A2E",     # a single warm accent, used sparingly
    "train": "#EEF1F6",      # warm-up shading
    "positive": "#2E6E5B",
}

#: Per-model colours. Baselines are deliberately desaturated so the two GARCH models
#: carry the eye: in almost every figure the baselines are context and the comparison
#: that matters is between the navy and the teal.
MODEL_COLOURS = {
    "yesterday": "#9AA3AF",
    "ewma": "#B23A2E",
    "garch_mle": "#1F3864",
    "garch_mle_normal": "#7C93BF",
    "garch_bayes": "#2E6E5B",
    "garch_bayes_mean": "#84B3A2",
}

MODEL_LABELS = {
    "yesterday": "RW-in-vol (yesterday's scaled Parkinson)",
    "ewma": "EWMA (RiskMetrics, lambda = 0.94)",
    "garch_mle": "GARCH(1,1)-t, plug-in",
    "garch_mle_normal": "GARCH(1,1)-normal, plug-in (ablation)",
    "garch_bayes": "GARCH(1,1)-t, posterior predictive",
    "garch_bayes_mean": "GARCH(1,1)-t, plug-in at the posterior mean (ablation)",
}

#: Short forms, for axis ticks and panel titles where the full label will not fit.
MODEL_SHORT = {
    "yesterday": "RW-in-vol",
    "ewma": "EWMA",
    "garch_mle": "GARCH-t (MLE)",
    "garch_mle_normal": "GARCH-normal",
    "garch_bayes": "GARCH-t (Bayes)",
    "garch_bayes_mean": "GARCH-t (post. mean)",
}

#: Regime palette, ordered calm -> stressed so the ramp reads as increasing severity.
REGIME_COLOURS = {
    "calm": "#8FA9CE",
    "normal": "#1F3864",
    "stressed": "#B23A2E",
}


def apply_house_style() -> None:
    """Set rcParams once. Idempotent, so notebooks may call it freely.

    Everything here is furniture: type, gridlines, tick geometry, the absence of a box
    around the plot. Nothing that carries meaning is set here -- colours that encode a
    model or a regime live in the dictionaries above, so a reader can find out what navy
    means without reading a style function.
    """
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.28,
            "figure.facecolor": COLOURS["panel"],
            "axes.facecolor": COLOURS["panel"],
            "font.family": "sans-serif",
            "font.sans-serif": list(FONT_STACK),
            "font.size": 9,
            "text.color": COLOURS["ink"],
            # Titles are drawn by `title_block`, not by matplotlib, so the built-in
            # title is styled only for the panel headings of small-multiple figures.
            "axes.titlesize": 9.5,
            "axes.titleweight": "semibold",
            "axes.titlecolor": COLOURS["ink"],
            "axes.titlelocation": "left",
            "axes.titlepad": 6.0,
            "axes.labelsize": 8.5,
            "axes.labelcolor": COLOURS["muted"],
            "axes.edgecolor": COLOURS["rule"],
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.axisbelow": True,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": COLOURS["grid"],
            "grid.linewidth": 0.8,
            "grid.alpha": 1.0,
            "xtick.color": COLOURS["muted"],
            "ytick.color": COLOURS["muted"],
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.0,
            "ytick.major.size": 0.0,
            "xtick.major.width": 0.8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "legend.handlelength": 1.6,
            "legend.borderaxespad": 0.0,
            "lines.solid_capstyle": "round",
        }
    )


def title_block(
    fig: Figure,
    title: str,
    subtitle: str | None = None,
    *,
    x: float = 0.0,
    y: float = 1.0,
) -> None:
    """Left-aligned title, optional subtitle, and a rule beneath them.

    Placed in figure coordinates rather than on an axis, so it sits flush left over a
    whole grid of panels instead of being centred over one of them. The title states the
    finding; the subtitle carries the qualification that keeps the finding honest, which
    is usually the sample size or the thing the figure cannot show.
    """
    fig.text(
        x, y, title,
        ha="left", va="bottom",
        fontsize=11.5, fontweight="semibold", color=COLOURS["ink"],
    )
    if subtitle:
        fig.text(
            x, y - 0.052, subtitle,
            ha="left", va="bottom",
            fontsize=8.6, color=COLOURS["muted"],
        )


def source_note(
    fig: Figure, text: str, *, x: float = 0.0, y: float = -0.02, width: int = 118
) -> None:
    """A small grey source line at the foot, naming the table the figure was drawn from.

    Every figure carries one. A chart in a slide deck outlives the conversation that
    produced it, and the only defence is that it says where its numbers came from.

    **Wrapped, and that is not cosmetic.** These notes are saved with
    ``bbox_inches="tight"``, which grows the canvas to fit whatever is on it -- so a note
    left as one long line silently stretches the figure to the width of the sentence and
    squeezes the plot into a corner of it. Wrapping keeps the figure the size it was
    designed at.
    """
    fig.text(
        x, y, "\n".join(textwrap.wrap(text, width=width)),
        ha="left", va="top",
        fontsize=7.4, color=COLOURS["muted"], linespacing=1.5,
    )


def label_at_end(
    ax: Axes,
    x,
    y: float,
    text: str,
    colour: str,
    *,
    dx: int = 6,
    fontsize: float = 8.0,
) -> None:
    """Name a series at its own right-hand end, in its own colour.

    Preferred over a legend wherever the lines end far enough apart to be labelled. It
    removes the colour-matching step a legend forces on the reader, and it survives
    being printed in greyscale.
    """
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(dx, 0),
        textcoords="offset points",
        ha="left", va="center",
        fontsize=fontsize, fontweight="semibold", color=colour,
        annotation_clip=False,
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
            color=COLOURS["muted"],
            alpha=0.10,
            linewidth=0,
            zorder=0,
        )
        if label:
            ax.annotate(
                name,
                xy=(pd.Timestamp(start), 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(3, -9),
                textcoords="offset points",
                fontsize=7.2,
                color=COLOURS["muted"],
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


# --- Stage 2 figures ---------------------------------------------------------------


def plot_garch_residual_diagnostics(
    residuals: np.ndarray,
    nu: float,
    out_path: Path,
    *,
    acf_levels: pd.DataFrame,
    acf_squares: pd.DataFrame,
) -> Path:
    """Standardised residuals from the warm-up GARCH fit: QQ plot and two ACFs.

    Under correct specification these residuals are i.i.d. draws from the standardised
    Student-t: the QQ plot sits on the diagonal, and neither the levels nor the squares
    show autocorrelation. The two ACFs test different claims and both are needed --
    structure left in the levels would indict the constant-mean assumption, structure
    left in the squares would mean the variance equation has not absorbed the
    clustering that motivated fitting a GARCH at all.

    The QQ reference is the **fitted** t, not a normal. Comparing against a normal here
    would only restate Stage 0's finding that returns are fat-tailed; the question at
    this stage is whether the t that was actually fitted is the right t.
    """
    from scipy import stats

    apply_house_style()
    fig, (ax_qq, ax_lev, ax_sq) = plt.subplots(1, 3, figsize=(10.5, 3.3))

    clean = np.asarray(residuals, dtype=float)
    clean = clean[np.isfinite(clean)]
    n = clean.size
    # Standardised t: unit variance, so the reference quantiles carry the same
    # sqrt((nu-2)/nu) factor the predictive distribution does.
    probs = (np.arange(1, n + 1) - 0.5) / n
    theoretical = stats.t.ppf(probs, nu) * np.sqrt((nu - 2.0) / nu)
    ax_qq.scatter(theoretical, np.sort(clean), s=5, color=COLOURS["returns"], alpha=0.6)
    span = [float(theoretical.min()), float(theoretical.max())]
    ax_qq.plot(span, span, color=COLOURS["accent"], linewidth=1.0)
    ax_qq.set_title(f"QQ vs fitted t (nu = {nu:.2f})")
    ax_qq.set_xlabel("theoretical quantile")
    ax_qq.set_ylabel("standardised residual")

    for ax, acf_frame, title in (
        (ax_lev, acf_levels, "ACF of standardised residuals"),
        (ax_sq, acf_squares, "ACF of squared standardised residuals"),
    ):
        lags = acf_frame.index.to_numpy()
        values = acf_frame["acf"].to_numpy()
        ax.vlines(lags, 0.0, values, color=COLOURS["returns"], linewidth=1.4)
        ax.scatter(lags, values, s=7, color=COLOURS["returns"], zorder=3)
        ax.fill_between(
            lags,
            acf_frame["ci_lower"].to_numpy(),
            acf_frame["ci_upper"].to_numpy(),
            color=COLOURS["muted"],
            alpha=0.20,
            linewidth=0,
        )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("lag (trading days)")
        ax.margins(x=0.01)

    fig.tight_layout()
    return _finish(fig, out_path)


def plot_parameter_stability(records: pd.DataFrame, out_path: Path) -> Path:
    """GARCH estimates across the 102 refits.

    Nearly free once the refits exist, and it answers a question the coverage tables
    cannot: whether the model the backtest is scoring is one model or a slowly drifting
    sequence of them. Persistence ``alpha + beta`` is drawn on its own panel with the
    stationarity boundary marked, because it is the quantity that would announce a
    fitting problem by walking into 1.

    Reads the persisted ``RefitRecord`` table, so the plotted estimates are the same
    numbers the audit trail holds rather than a second, separately computed set.
    """
    apply_house_style()
    # Rows with an estimated alpha, i.e. the models that actually fit something. Keyed
    # on the data rather than on a list of model names, so a model added later appears
    # here without this function needing to be told about it.
    fitted = records[records["alpha"].notna()].copy()
    fitted["refit_date"] = pd.to_datetime(fitted["refit_date"])

    panels = (
        ("alpha", "alpha"),
        ("beta", "beta"),
        ("persistence", "alpha + beta"),
        ("nu", "nu (Student-t d.o.f.)"),
    )
    fig, axes = plt.subplots(4, 1, figsize=(7.4, 8.0), sharex=True)

    for ax, (column, label) in zip(axes, panels):
        for model, group in fitted.groupby("model", sort=False):
            if column == "persistence":
                values = group["alpha"].to_numpy() + group["beta"].to_numpy()
            else:
                values = group[column].to_numpy()
            if not np.any(np.isfinite(values)):
                continue  # nu is undefined for the normal-innovation variant
            ax.plot(
                group["refit_date"].to_numpy(),
                values,
                linewidth=1.2,
                marker="o",
                markersize=2.5,
                color=MODEL_COLOURS.get(model, COLOURS["muted"]),
                label=MODEL_LABELS.get(model, model),
            )
        if column == "persistence":
            ax.axhline(1.0, color=COLOURS["accent"], linewidth=0.9, linestyle="--")
            ax.annotate(
                "stationarity boundary",
                xy=(0.01, 1.0),
                xycoords=("axes fraction", "data"),
                xytext=(0, 3),
                textcoords="offset points",
                fontsize=7,
                color=COLOURS["accent"],
            )
        ax.set_ylabel(label)
        # Stress labels go on the bottom panel, not the top one: the top panel carries
        # the legend, and top-anchored annotations there print straight through it.
        shade_stress_periods(ax, label=(column == "nu"))

    axes[0].set_title("GARCH(1,1) estimates across the 102 refits (expanding window)")
    axes[0].legend(loc="upper left")
    axes[-1].set_xlabel("refit date")
    fig.tight_layout()
    return _finish(fig, out_path)


# --- Stage 4 figures ---------------------------------------------------------------


def plot_pit_histograms(
    scored: pd.DataFrame,
    out_path: Path,
    *,
    models: tuple[str, ...],
    bins: int = 20,
) -> Path:
    """PIT histograms, one panel per model, against the uniform density.

    Finer than coverage at three fixed levels: coverage says an interval is too narrow,
    the histogram says *where*. A U shape is a predictive that is too narrow overall; a
    single tall left-hand bar is a left tail that is too thin, which is the failure that
    matters for a risk model and the one a symmetric coverage number can hide.

    The 45-degree reference is the uniform density, not a fitted curve. The KS test in
    the companion table is a summary of this picture, and its p-value is approximate
    because the predictive distributions have estimated parameters.
    """
    apply_house_style()
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.6), sharex=True, sharey=True)

    for ax, model in zip(axes.flat, models):
        values = scored.loc[
            (scored["model"] == model) & scored["pit"].notna(), "pit"
        ].to_numpy()
        ax.hist(
            values,
            bins=bins,
            range=(0.0, 1.0),
            color=MODEL_COLOURS.get(model, COLOURS["returns"]),
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.axhline(
            len(values) / bins,
            color=COLOURS["accent"],
            linewidth=1.0,
            linestyle="--",
            label="uniform",
        )
        short = MODEL_LABELS.get(model, model).split(" (")[0]
        ax.set_title(f"{short}   n = {len(values):,}", fontsize=9)
        ax.set_xlim(0.0, 1.0)

    for ax in axes[-1]:
        ax.set_xlabel("PIT value")
    for ax in axes[:, 0]:
        ax.set_ylabel("days")
    axes.flat[0].legend(loc="upper center")

    fig.suptitle(
        "Probability integral transform of the realised return, by model",
        fontsize=10,
        fontweight="semibold",
    )
    fig.tight_layout()
    return _finish(fig, out_path)


def plot_coverage_vs_nominal(
    coverage: pd.DataFrame,
    out_path: Path,
    *,
    models: tuple[str, ...],
) -> Path:
    """Empirical against nominal coverage at each level, one marker per model.

    The 45-degree line is perfect calibration. Distance below it is an interval that is
    too narrow -- the direction that matters, because it is the one that understates
    risk. Sample size is in the caption rather than the axes: every point on this figure
    is computed on the same stated sample.
    """
    apply_house_style()
    fig, ax = plt.subplots(figsize=(5.6, 5.4))

    # Both axes share one range and an equal aspect, so the reference line is a true
    # 45 degrees and vertical distance below it can be read directly off the chart. The
    # line is dashed and black rather than grey: one of the models is drawn in grey, and
    # a reference that could be mistaken for a series is worse than none.
    floor = min(0.90, float(coverage["empirical"].min())) - 0.015
    ax.plot(
        [floor, 1.0],
        [floor, 1.0],
        color="black",
        linewidth=0.9,
        linestyle="--",
        alpha=0.6,
        zorder=1,
    )
    midpoint = floor + 0.62 * (1.0 - floor)
    ax.annotate(
        "perfect calibration",
        xy=(midpoint, midpoint),
        xytext=(0, 5),
        textcoords="offset points",
        fontsize=7.5,
        color="black",
        alpha=0.6,
        rotation=45,
        rotation_mode="anchor",
        ha="center",
        va="bottom",
    )
    ax.set_xlim(floor, 1.0)
    ax.set_ylim(floor, 1.0)

    for model in models:
        block = coverage[coverage["model"] == model].sort_values("nominal")
        ax.plot(
            block["nominal"],
            block["empirical"],
            marker="o",
            markersize=5,
            linewidth=1.2,
            color=MODEL_COLOURS.get(model, COLOURS["returns"]),
            label=MODEL_LABELS.get(model, model),
            zorder=2,
        )

    ax.set_xticks([0.90, 0.95, 0.99])
    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("empirical coverage")
    ax.set_title("Does a 95% interval contain 95% of returns?")
    ax.legend(loc="upper left", fontsize=7.5)
    fig.tight_layout()
    return _finish(fig, out_path)


def plot_var_hit_sequence(
    scored: pd.DataFrame,
    out_path: Path,
    *,
    models: tuple[str, ...],
    var_level: float = 0.99,
) -> Path:
    """Timeline of 99% VaR breaches per model, with the stress episodes shaded.

    The project's headline figure, and the one that makes Christoffersen's point
    visually. A model with the right *number* of breaches can still have put them all
    inside one fortnight; that is visible here and invisible in a coverage table.

    One row per model, a tick per breach, and the expected count printed against the
    realised one so the two failure modes -- too many breaches, and breaches in the
    wrong place -- can be read off the same picture.
    """
    apply_house_style()
    fig, ax = plt.subplots(figsize=(9.0, 0.72 * len(models) + 1.9))

    expected_rate = 1.0 - var_level
    for row, model in enumerate(reversed(models)):
        block = scored.loc[
            (scored["model"] == model) & scored["exceedance"].notna()
        ].sort_values("date")
        breaches = block.loc[block["exceedance"] == 1.0, "date"]
        colour = MODEL_COLOURS.get(model, COLOURS["returns"])

        ax.hlines(row, block["date"].min(), block["date"].max(), color=COLOURS["grid"], linewidth=1.0)
        ax.vlines(breaches, row - 0.28, row + 0.28, color=colour, linewidth=1.1)
        ax.annotate(
            f"{len(breaches)} breaches / {expected_rate * len(block):.0f} expected",
            xy=(1.0, row),
            xycoords=("axes fraction", "data"),
            xytext=(6, -3),
            textcoords="offset points",
            fontsize=7.5,
            color=colour,
        )

    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(
        [MODEL_LABELS.get(m, m).split(" (")[0] for m in reversed(models)], fontsize=8
    )
    ax.set_ylim(-0.6, len(models) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("")
    ax.set_title(
        f"{var_level:.0%} VaR breaches: how many, and whether they arrive together"
    )
    shade_stress_periods(ax)
    fig.tight_layout()
    return _finish(fig, out_path)


def plot_interval_decomposition(
    decomposition: pd.DataFrame,
    out_path: Path,
    *,
    nominal: float = 0.99,
) -> Path:
    """The two causes of the frequentist-Bayesian interval difference, side by side.

    The comparison the project was designed around -- one likelihood, plug-in against
    posterior predictive -- turned out to move two things at once against the MLE
    plug-in that ``forecasts.csv`` holds (research_log.md 1.13). This figure is the
    reason that correction is legible rather than a paragraph: at the 99% level the two
    causes point in opposite directions, and the reported difference is the smaller,
    prior-dominated net of them.
    """
    apply_house_style()
    fig, ax = plt.subplots(figsize=(7.0, 3.8))

    block = decomposition[decomposition["nominal"] == nominal]
    contrasts = list(dict.fromkeys(block["contrast"]))
    regimes = ["all", *(r for r in block["regime"].unique() if r != "all")]
    width = 0.8 / len(contrasts)
    palette = (COLOURS["returns"], COLOURS["accent"], COLOURS["muted"])

    for offset, (contrast, colour) in enumerate(zip(contrasts, palette)):
        rows = block[block["contrast"] == contrast].set_index("regime")
        heights = [rows.loc[r, "mean_width_ratio"] - 1.0 for r in regimes]
        ax.bar(
            np.arange(len(regimes)) + offset * width,
            heights,
            width=width,
            bottom=1.0,
            color=colour,
            label=contrast,
        )

    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(regimes)) + width)
    ax.set_xticklabels([f"{r}\n(n = {block[block['regime'] == r]['n'].iloc[0]:,})" for r in regimes])
    ax.set_ylabel("mean interval width ratio")
    ax.set_title(
        f"What actually moves the {nominal:.0%} interval: the priors, or parameter uncertainty"
    )
    ax.legend(loc="lower left", ncol=3)
    fig.tight_layout()
    return _finish(fig, out_path)


# --- Stage 5 figures ---------------------------------------------------------------



def plot_regime_coverage(
    regime_coverage: pd.DataFrame,
    out_path: Path,
    *,
    models: tuple[str, ...],
) -> Path:
    """Coverage against nominal by regime, one panel per model, with bootstrap CIs.

    The figure the research question asks for: does 99% still mean 99% when VIX > 25?

    Error bars are stationary-block-bootstrap intervals computed *within* each regime,
    and their width is part of the message rather than an apology for it. The stressed
    regime is a few hundred days made of a small number of long runs, so its intervals
    are wide and a difference that fits inside one is not a difference this sample can
    see.

    A marker below the diagonal is an interval that is too narrow -- the direction that
    understates risk, and the only direction that matters for the use these forecasts
    are put to.
    """
    apply_house_style()
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 7.2), sharex=True, sharey=True)

    levels = sorted(regime_coverage["nominal"].unique())
    offsets = np.linspace(-0.006, 0.006, len(REGIME_COLOURS))

    for ax, model in zip(axes.flat, models):
        block = regime_coverage[regime_coverage["model"] == model]
        ax.plot(
            [0.86, 1.005], [0.86, 1.005], color="black", linewidth=0.9,
            linestyle="--", alpha=0.6, zorder=1,
        )
        for offset, (regime, colour) in zip(offsets, REGIME_COLOURS.items()):
            rows = block[block["regime"] == regime].sort_values("nominal")
            if rows.empty:
                continue
            x = rows["nominal"].to_numpy() + offset
            y = rows["empirical"].to_numpy()
            ax.errorbar(
                x,
                y,
                yerr=[y - rows["ci_lower"].to_numpy(), rows["ci_upper"].to_numpy() - y],
                fmt="o",
                markersize=4.5,
                linewidth=1.1,
                capsize=2.5,
                color=colour,
                label=f"{regime} (n = {rows['n'].iloc[0]:,})",
                zorder=2,
            )
        short = MODEL_LABELS.get(model, model).split(" (")[0]
        ax.set_title(short, fontsize=9)
        ax.set_xticks(levels)
        ax.set_xlim(0.875, 1.005)
        ax.legend(loc="upper left", fontsize=7)

    for ax in axes[-1]:
        ax.set_xlabel("nominal coverage")
    for ax in axes[:, 0]:
        ax.set_ylabel("empirical coverage")

    fig.suptitle(
        "Does a 99% interval still contain 99% of returns when VIX > 25?",
        fontsize=10,
        fontweight="semibold",
    )
    fig.tight_layout()
    return _finish(fig, out_path)


def plot_regime_var_rate(
    regime_var: pd.DataFrame,
    out_path: Path,
    *,
    models: tuple[str, ...],
    var_level: float = 0.99,
) -> Path:
    """99% VaR breach rate by regime, with bootstrap intervals and the nominal line.

    The companion to the hit-sequence timeline, and the figure that answers the regime
    question for the tail specifically. A bar whose interval straddles the dashed line
    is a rate this sample cannot distinguish from nominal; one clear of it is a model
    breaching more often than it promised, in that regime.

    Read it against the two-sided coverage panel rather than instead of it. They can
    disagree, and where they do the model is putting the right *number* of breaches in
    the wrong *tail* -- which a two-sided number cannot show and a risk manager cares
    about more than either.
    """
    apply_house_style()
    fig, ax = plt.subplots(figsize=(9.0, 4.6))

    nominal_rate = 1.0 - var_level
    regimes = [r for r in REGIME_COLOURS if r in set(regime_var["regime"])]
    width = 0.74 / len(regimes)
    positions = np.arange(len(models))

    for offset, regime in enumerate(regimes):
        rows = regime_var[regime_var["regime"] == regime].set_index("model")
        rows = rows.reindex(list(models))
        x = positions + offset * width
        rate = rows["rate"].to_numpy()
        ax.bar(x, rate, width=width * 0.92, color=REGIME_COLOURS[regime], label=regime)
        ax.errorbar(
            x,
            rate,
            yerr=[rate - rows["ci_lower"].to_numpy(), rows["ci_upper"].to_numpy() - rate],
            fmt="none",
            ecolor=COLOURS["ink"],
            elinewidth=0.9,
            capsize=2.0,
            capthick=0.9,
            alpha=0.55,
        )

    ax.axhline(nominal_rate, color=COLOURS["ink"], linewidth=1.0, linestyle=(0, (4, 3)))
    ax.annotate(
        f"nominal {nominal_rate:.0%}",
        xy=(1.0, nominal_rate),
        xycoords=("axes fraction", "data"),
        xytext=(6, 0),
        textcoords="offset points",
        va="center", fontsize=7.6, color=COLOURS["ink"],
        annotation_clip=False,
    )

    ax.set_xticks(positions + width)
    ax.set_xticklabels([MODEL_SHORT.get(m, m) for m in models], fontsize=8.5)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylim(0, None)
    ax.tick_params(axis="x", length=0)
    ax.legend(loc="upper left", ncol=3, fontsize=8, columnspacing=1.4)

    title_block(
        fig,
        "The GARCH models breach most often in the middle of the volatility distribution",
        f"{var_level:.0%} VaR breach rate by lagged-VIX regime. Bars are point estimates; "
        "whiskers are 95% stationary-block-bootstrap intervals computed within each regime.",
    )
    source_note(
        fig,
        "Source: eval_regime_var.csv. Evaluation window 2017-01-03 to 2025-06-30; "
        "n = 757 / 1,030 / 347 days (calm / normal / stressed). Overlapping whiskers are "
        "not a test — the differences between regimes are in "
        "eval_regime_differences.csv, where only normal against calm separates.",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return _finish(fig, out_path)


def plot_tail_allocation(
    tail_asymmetry: pd.DataFrame,
    out_path: Path,
    *,
    models: tuple[str, ...],
) -> Path:
    """Where each model's interval breaches land, against where they should.

    The project's sharpest diagnostic, and the one a coverage number cannot show. Each
    pair of bars is one nominal level: exceptions below the lower bound on the left,
    above the upper bound on the right, with the dashed line at the count each tail
    should hold if the predictive had the right shape.

    A model that is merely too narrow overshoots both bars equally. A model whose
    *shape* is wrong overshoots one and undershoots the other -- and for these GARCH
    models it is always the loss tail that overshoots, at every level, because a
    symmetric Student-t innovation cannot represent a return series whose standardised
    residuals are skewed -0.79.
    """
    apply_house_style()
    fig, axes = plt.subplots(
        1, len(models), figsize=(2.55 * len(models), 4.3), sharey=True
    )
    axes = np.atleast_1d(axes)

    for ax, model in zip(axes, models):
        block = tail_asymmetry[tail_asymmetry["model"] == model].sort_values("nominal")
        positions = np.arange(len(block))
        colour = MODEL_COLOURS.get(model, COLOURS["returns"])

        ax.bar(positions - 0.19, block["n_below"], width=0.34, color=colour)
        ax.bar(positions + 0.19, block["n_above"], width=0.34, color=colour, alpha=0.35)
        for x, expected in zip(positions, block["expected_per_tail"]):
            ax.hlines(
                expected, x - 0.42, x + 0.42,
                color=COLOURS["ink"], linewidth=1.1, linestyle=(0, (3, 2)), zorder=3,
            )

        # The p-value belongs on the panel, not in a caption: it is the whole reason
        # the reader should believe the eye rather than dismiss the gap as noise.
        worst = block.loc[block["symmetry_p"].idxmin()]
        ax.annotate(
            f"symmetry p = {worst.symmetry_p:.0e}" if worst.symmetry_p < 0.01
            else f"symmetry p = {worst.symmetry_p:.2f}",
            xy=(0.5, 0.97), xycoords="axes fraction",
            ha="center", va="top", fontsize=7.4,
            color=COLOURS["accent"] if worst.symmetry_p < 0.01 else COLOURS["muted"],
        )

        ax.set_xticks(positions)
        ax.set_xticklabels([f"{lv:.0%}" for lv in block["nominal"]])
        ax.set_title(MODEL_SHORT.get(model, model))
        ax.tick_params(axis="x", length=0)

    axes[0].set_ylabel("interval exceptions")
    axes[0].annotate(
        "loss tail",
        xy=(-0.19, tail_asymmetry.iloc[0]["n_below"]),
        xytext=(0, 6), textcoords="offset points",
        ha="center", fontsize=7.4, fontweight="semibold",
        color=MODEL_COLOURS.get(models[0], COLOURS["returns"]),
    )

    title_block(
        fig,
        "The intervals are the wrong shape, not just the wrong width",
        "Interval exceptions by tail. Solid bar is the loss tail, faded bar the upper "
        "tail, dashed rule the count each tail should hold. A model that is merely too "
        "narrow overshoots both equally.",
    )
    source_note(
        fig,
        "Source: eval_tail_asymmetry.csv. Evaluation window 2017-01-03 to 2025-06-30. "
        "Symmetry p is an exact binomial test of the split against 50/50, at the level "
        "where it is smallest. Both GARCH models reject at every level; the "
        "standardised residuals are skewed −0.79, which a symmetric Student-t "
        "innovation cannot represent.",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return _finish(fig, out_path)
