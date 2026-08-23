"""Single entry point for the volatility forecast calibration pipeline.

Usage
-----
    python run_all.py --help
    python run_all.py --stage data
    python run_all.py --all

``data`` is implemented (Stage 0). ``backtest``, ``evaluate`` and ``figures`` are still
stubs and raise ``NotImplementedError``: their interfaces exist so they can be reviewed
before any implementation is written.

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
                "trading days, producing daily forecasts and predictive intervals "
                "for all four models.",
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

    Stage 1 produces the two baselines. The GARCH models join the same loop at Stages 2
    and 3 without the loop itself changing.
    """
    import pandas as pd

    from src import backtest as B
    from src import figures

    frame_path = PROCESSED_DIR / "analysis_frame.csv"
    if not frame_path.exists():
        raise SystemExit(
            f"{frame_path} not found. Run `python run_all.py --stage data` first."
        )
    frame = pd.read_csv(frame_path, index_col=0, parse_dates=True)

    config = B.BacktestConfig(mcmc_seed=args.seed)
    print(f"proxy scale c = {config.proxy_scale_c:.6f}  (frozen, warm-up only)")
    forecasts, records = B.run_backtest(frame, config)

    print(f"models    : {', '.join(B.BASELINE_MODELS)}")
    print(f"refits    : {len(records)} at a {config.refit_every}-day cadence")
    for model in B.BASELINE_MODELS:
        sub = forecasts[forecasts.model == model]
        ann = (sub["variance"].mean() * 252) ** 0.5
        print(
            f"  {model:<10} {len(sub):,} rows, "
            f"{sub['variance'].notna().sum():,} finite, "
            f"mean annualised vol {ann:.2%}"
        )

    written = B.save_forecasts(forecasts, records, config, PROCESSED_DIR)
    for path in written:
        print(f"wrote {path}")

    fig_path = figures.plot_forecasts_vs_realised(
        forecasts,
        frame,
        FIGURES_DIR / "05_baseline_forecasts_covid.png",
        start="2019-11-01",
        end="2020-06-30",
    )
    print(f"wrote {fig_path}")


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
