"""Single entry point for the volatility forecast calibration pipeline.

Usage
-----
    python run_all.py --help
    python run_all.py --stage data
    python run_all.py --all

``data``, ``eda`` and ``backtest`` are implemented (Stages 0-2). ``evaluate`` and
``figures`` are still stubs and raise ``NotImplementedError``: their interfaces exist so
they can be reviewed before any implementation is written.

Stages run in the order listed and each depends on its predecessor's artefacts in
``data/processed/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"

STAGE_ORDER: tuple[str, ...] = ("data", "eda", "backtest", "evaluate", "figures")

STAGE_HELP: dict[str, str] = {
    "data": "Download or load cached SPY and ^VIX bars, verify hashes, build the "
            "analysis frame (returns, Parkinson proxy, lagged-VIX regimes).",
    "eda": "Test for the ARCH effects that motivate a conditional-variance model "
           "(Engle LM, Ljung-Box, ADF) on the training window, and render the "
           "Stage 0 figures.",
    "backtest": "Walk forward through the out-of-sample period, refitting every 21 "
                "trading days and filtering daily in between, producing forecasts and "
                "predictive intervals for every model built so far.",
    "evaluate": "Compute QLIKE and variance MSE, interval coverage, VaR backtests, "
                "Diebold-Mariano comparisons, and block-bootstrap intervals.",
    "figures": "Render the figures used in report/report.md.",
}


def stage_data(args: argparse.Namespace) -> None:
    """Build the analysis frame from cached or freshly downloaded raw data.

    Downloads only when a snapshot is absent, or when ``--refresh`` is passed. The
    committed snapshot is never replaced by a routine run (decision B3).
    """
    from src import data as D

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    downloaded = False
    for ticker in (D.SPY_TICKER, D.VIX_TICKER):
        path = RAW_DIR / D.cache_filename(ticker)
        if path.exists() and not args.refresh:
            print(f"cache hit : {path.name} (no network access)")
            continue
        print(f"downloading {ticker} {D.DOWNLOAD_START}..{D.SAMPLE_END} -> {path.name}")
        D.download_raw(
            ticker, D.DOWNLOAD_START, D.SAMPLE_END, RAW_DIR, refresh=args.refresh
        )
        downloaded = True

    manifest_path = RAW_DIR / D.MANIFEST_NAME
    if downloaded or not manifest_path.exists():
        D.write_manifest(RAW_DIR)
        print(f"manifest written: {manifest_path.name}")

    status = D.verify_manifest(RAW_DIR)
    for name, ok in status.items():
        print(f"sha256 {'OK  ' if ok else 'FAIL'} {name}")
    if not all(status.values()):
        raise SystemExit("manifest verification failed; refusing to build on altered data")

    spy = D.load_raw(D.SPY_TICKER, RAW_DIR, verify=True)
    vix = D.load_raw(D.VIX_TICKER, RAW_DIR, verify=True)
    frame, report = D.build_analysis_frame(spy, vix)

    # Printed in full, never summarised away: any dropped date must be visible.
    print(report.format_full())

    train, oos = D.split_train_oos(frame)
    print(
        f"analysis frame : {frame.index.min().date()} .. {frame.index.max().date()} "
        f"({len(frame)} rows)"
    )
    print(f"  train : {train.index.min().date()} .. {train.index.max().date()} "
          f"({len(train)} rows)")
    print(f"  oos   : {oos.index.min().date()} .. {oos.index.max().date()} "
          f"({len(oos)} rows)")
    print("  regime counts (lagged VIX):")
    for label, count in frame["regime"].value_counts().sort_index().items():
        print(f"    {label:<9} {count}")

    out = PROCESSED_DIR / "analysis_frame.csv"
    frame.to_csv(out, index_label="date", date_format="%Y-%m-%d", lineterminator="\n")
    print(f"wrote {out}")


def stage_eda(args: argparse.Namespace) -> None:
    """Establish, by test, that the data warrants a conditional-variance model.

    Primary results are computed on the **training window only**. Full-sample values are
    printed afterwards as descriptive context and are explicitly not a justification for
    anything -- see the module docstring of ``src.eda``.
    """
    import pandas as pd

    from src import eda, figures
    from src.data import TRAIN_END, TRAIN_START

    frame_path = PROCESSED_DIR / "analysis_frame.csv"
    if not frame_path.exists():
        raise SystemExit(
            f"{frame_path} not found. Run `python run_all.py --stage data` first."
        )
    frame = pd.read_csv(frame_path, index_col=0, parse_dates=True)

    train_returns = frame.loc[TRAIN_START:TRAIN_END, "log_return"]
    train_report = eda.run_eda(train_returns, window_label="training window (PRIMARY)")
    print(train_report.format_full())
    print()
    print("VERDICT:", train_report.headline_verdict())

    full_report = eda.run_eda(
        frame["log_return"], window_label="full sample (DESCRIPTIVE CONTEXT ONLY)"
    )
    print()
    print(full_report.format_full())
    print()
    print(
        "The full-sample block above is reported so a reader can see it agrees with"
        " the training window. It justifies nothing: every design decision was locked"
        " before these numbers existed (research_log.md 1.1)."
    )

    written = [
        figures.plot_returns_with_regimes(
            frame, FIGURES_DIR / "01_returns_clustering.png", train_end=TRAIN_END
        ),
        figures.plot_squared_return_acf(
            train_report.acf_squared,
            FIGURES_DIR / "02_squared_return_acf.png",
            window_label="training window",
        ),
        figures.plot_return_distribution(
            train_returns, FIGURES_DIR / "03_return_distribution.png"
        ),
        figures.plot_vix_with_regimes(frame, FIGURES_DIR / "04_vix_regimes.png"),
    ]
    print()
    for path in written:
        print(f"wrote {path}")


def stage_backtest(args: argparse.Namespace) -> None:
    """Run the walk-forward backtest and persist forecasts and refit diagnostics.

    Stage 1 produced the two baselines; Stage 2 added the frequentist GARCH(1,1)-t and
    its normal-innovation ablation, which join the same loop without the loop changing.
    The Bayesian model joins at Stage 3 the same way.

    Takes about a minute: 204 maximum-likelihood fits, 102 per GARCH variant.
    """
    from dataclasses import asdict

    import pandas as pd

    from src import backtest as B
    from src import eda, figures, models
    from src.data import TRAIN_END, TRAIN_START

    frame_path = PROCESSED_DIR / "analysis_frame.csv"
    if not frame_path.exists():
        raise SystemExit(
            f"{frame_path} not found. Run `python run_all.py --stage data` first."
        )
    frame = pd.read_csv(frame_path, index_col=0, parse_dates=True)

    config = B.BacktestConfig(mcmc_seed=args.seed)
    print(f"proxy scale c = {config.proxy_scale_c:.6f}  (frozen, warm-up only)")
    forecasts, records = B.run_backtest(frame, config)
    record_frame = pd.DataFrame([asdict(r) for r in records])

    n_refits = record_frame["refit_id"].nunique()
    print(f"models    : {', '.join(B.MODELS)}")
    print(f"refits    : {n_refits} at a {config.refit_every}-day cadence")
    for model in B.MODELS:
        sub = forecasts[forecasts.model == model]
        ann = (sub["variance"].mean() * 252) ** 0.5
        print(
            f"  {model:<18} {len(sub):,} rows, "
            f"{sub['variance'].notna().sum():,} finite, "
            f"mean annualised vol {ann:.2%}"
        )

    # Convergence is reported whether or not it is good news. A refit that failed
    # produces no forecasts for its block, so a silent failure would show up only as a
    # gap in a table nobody reads.
    print()
    for model in B.GARCH_MODELS:
        fits = record_frame[record_frame["model"] == model]
        failed = fits[~fits["mle_converged"]]
        print(
            f"  {model:<18} {len(fits) - len(failed)}/{len(fits)} refits converged, "
            f"{fits['seconds_elapsed'].sum():.1f}s total"
        )
        for _, bad in failed.iterrows():
            print(
                f"    !! {bad['refit_date'].date()} did NOT converge: "
                f"{bad['mle_message']}  ({config.refit_every} days have no forecast)"
            )

    written = B.save_forecasts(forecasts, records, config, PROCESSED_DIR)
    for path in written:
        print(f"wrote {path}")

    # --- Warm-up residual diagnostics -------------------------------------------
    #
    # Computed on the first fit, whose estimation window is the warm-up block, so
    # nothing here has seen an out-of-sample observation.
    warmup_returns = frame.loc[TRAIN_START:TRAIN_END, "log_return"].dropna()
    warmup_values = warmup_returns.to_numpy(dtype=float)
    h0 = models.backcast_initial_variance(warmup_values)
    warmup_fit = models.fit_garch_mle(warmup_values, h0=h0)
    resid = models.standardised_residuals(warmup_fit.params, warmup_values, h0)
    resid_series = pd.Series(resid, index=warmup_returns.index)

    print()
    print(f"warm-up GARCH(1,1)-t fit ({warmup_fit.n_obs} obs, "
          f"{warmup_returns.index.min().date()}..{warmup_returns.index.max().date()})")
    p = warmup_fit.params
    print(f"  mu={p.mu:.6f}  omega={p.omega:.4e}  alpha={p.alpha:.4f}  "
          f"beta={p.beta:.4f}  alpha+beta={p.alpha + p.beta:.4f}  nu={p.nu:.2f}")
    print(f"  loglik={warmup_fit.loglik:.2f}  converged={warmup_fit.converged}")
    print("  standardised-residual diagnostics (no rejection is the good outcome):")
    for lags in (5, 10, 22):
        levels = eda.ljung_box_test(resid_series, lags, label="std. residuals")
        squares = eda.ljung_box_test(resid_series**2, lags, label="squared std. residuals")
        print(f"    lag {lags:>2}: levels p = {levels.p_value:.4f}, "
              f"squares p = {squares.p_value:.4f}")

    acf_levels = eda.series_acf(resid_series)
    acf_squares = eda.squared_return_acf(resid_series)

    written_figures = [
        figures.plot_forecasts_vs_realised(
            forecasts,
            frame,
            FIGURES_DIR / "05_baseline_forecasts_covid.png",
            start="2019-11-01",
            end="2020-06-30",
        ),
        figures.plot_garch_residual_diagnostics(
            resid,
            warmup_fit.params.nu,
            FIGURES_DIR / "06_garch_residual_diagnostics.png",
            acf_levels=acf_levels,
            acf_squares=acf_squares,
        ),
        figures.plot_parameter_stability(
            record_frame, FIGURES_DIR / "07_garch_parameter_stability.png"
        ),
    ]
    print()
    for path in written_figures:
        print(f"wrote {path}")


def stage_evaluate(args: argparse.Namespace) -> None:
    """Score the forecasts and persist the evaluation tables."""
    raise NotImplementedError("stage 'evaluate' not implemented")


def stage_figures(args: argparse.Namespace) -> None:
    """Render figures from the persisted evaluation tables."""
    raise NotImplementedError("stage 'figures' not implemented")


STAGES = {
    "data": stage_data,
    "eda": stage_eda,
    "backtest": stage_backtest,
    "evaluate": stage_evaluate,
    "figures": stage_figures,
}


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser."""
    epilog = "stages:\n" + "\n".join(
        f"  {name:<10} {STAGE_HELP[name]}" for name in STAGE_ORDER
    )
    parser = argparse.ArgumentParser(
        prog="run_all.py",
        description=__doc__.split("Usage")[0].strip(),
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--stage",
        choices=STAGE_ORDER,
        help="Run a single stage.",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Run every stage in order.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-download raw data, overwriting the committed snapshot. Off by "
             "default so a routine run can never silently replace the data that "
             "published results were computed from.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the MCMC sampler. The bootstrap seed is fixed separately in "
             "src/bootstrap.py so evaluation intervals stay reproducible "
             "independently of the sampler. (default: %(default)s)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to the requested stage(s)."""
    args = build_parser().parse_args(argv)
    to_run = STAGE_ORDER if args.all else (args.stage,)
    for name in to_run:
        print(f"=== stage: {name} ===", flush=True)
        STAGES[name](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
