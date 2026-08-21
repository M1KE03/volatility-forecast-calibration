"""Single entry point for the volatility forecast calibration pipeline.

Usage
-----
    python run_all.py --help
    python run_all.py --stage data
    python run_all.py --all

Every stage is currently an unimplemented stub and raises ``NotImplementedError``.
That is the intended state at this point in the project: the interfaces exist so they
can be reviewed before any implementation is written.

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

STAGE_ORDER: tuple[str, ...] = ("data", "backtest", "evaluate", "figures")

STAGE_HELP: dict[str, str] = {
    "data": "Download or load cached SPY and ^VIX bars, verify hashes, build the "
            "analysis frame (returns, Parkinson proxy, lagged-VIX regimes).",
    "backtest": "Walk forward through the out-of-sample period, refitting every 21 "
                "trading days, producing daily forecasts and predictive intervals "
                "for all four models.",
    "evaluate": "Compute QLIKE and variance MSE, interval coverage, VaR backtests, "
                "Diebold-Mariano comparisons, and block-bootstrap intervals.",
    "figures": "Render the figures used in report/report.md.",
}


def stage_data(args: argparse.Namespace) -> None:
    """Build the analysis frame from cached or freshly downloaded raw data."""
    raise NotImplementedError("stage 'data' not implemented")


def stage_backtest(args: argparse.Namespace) -> None:
    """Run the walk-forward backtest and persist forecasts and refit diagnostics."""
    raise NotImplementedError("stage 'backtest' not implemented")


def stage_evaluate(args: argparse.Namespace) -> None:
    """Score the forecasts and persist the evaluation tables."""
    raise NotImplementedError("stage 'evaluate' not implemented")


def stage_figures(args: argparse.Namespace) -> None:
    """Render figures from the persisted evaluation tables."""
    raise NotImplementedError("stage 'figures' not implemented")


STAGES = {
    "data": stage_data,
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
