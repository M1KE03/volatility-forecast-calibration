"""Single entry point for the volatility forecast calibration pipeline.

Usage
-----
    python run_all.py --help
    python run_all.py --stage data
    python run_all.py --all

Every stage is implemented (Stages 0-4). ``evaluate`` and ``figures`` read the stored
forecast table and never refit anything, so no number they print can move without
``forecasts.csv`` having moved first.

Stages run in the order listed and each depends on its predecessor's artefacts in
``data/processed/``.

``backtest`` and ``bayes`` are two halves of one walk-forward run, split because the
frequentist half takes a minute and the Bayesian half takes about ninety-five. Each
writes its own partial tables and rebuilds the merged ``forecasts.csv`` from whichever
partials are on disk, so re-running the cheap half never re-runs the expensive one, and
the evaluation layer still reads one table. ``--all`` runs both, and therefore takes
about an hour and a half.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"

STAGE_ORDER: tuple[str, ...] = ("data", "eda", "backtest", "bayes", "evaluate", "figures")

STAGE_HELP: dict[str, str] = {
    "data": "Download or load cached SPY and ^VIX bars, verify hashes, build the "
            "analysis frame (returns, Parkinson proxy, lagged-VIX regimes).",
    "eda": "Test for the ARCH effects that motivate a conditional-variance model "
           "(Engle LM, Ljung-Box, ADF) on the training window, and render the "
           "Stage 0 figures.",
    "backtest": "Walk forward through the out-of-sample period, refitting every 21 "
                "trading days and filtering daily in between, producing forecasts and "
                "predictive intervals for the baselines and the maximum-likelihood "
                "GARCH models. About a minute.",
    "bayes": "The same walk-forward run for the Bayesian GARCH(1,1)-t: 102 NUTS fits, "
             "each carrying its whole posterior into the predictive rather than its "
             "maximum. About 95 minutes; results merge into the same forecast table.",
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
    The Bayesian model joined at Stage 3 the same way, and runs from ``--stage bayes``
    for cost reasons alone -- see this module's docstring.

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
    forecasts, records = B.run_backtest(frame, config, models=B.FREQUENTIST_MODELS)
    record_frame = pd.DataFrame([asdict(r) for r in records])

    n_refits = record_frame["refit_id"].nunique()
    print(f"models    : {', '.join(B.FREQUENTIST_MODELS)}")
    print(f"refits    : {n_refits} at a {config.refit_every}-day cadence")
    for model in B.FREQUENTIST_MODELS:
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
        failed = fits[~fits["converged"]]
        print(
            f"  {model:<18} {len(fits) - len(failed)}/{len(fits)} refits converged, "
            f"{fits['seconds_elapsed'].sum():.1f}s total"
        )
        for _, bad in failed.iterrows():
            print(
                f"    !! {bad['refit_date'].date()} did NOT converge: "
                f"{bad['message']}  ({config.refit_every} days have no forecast)"
            )

    written = B.save_track("frequentist", forecasts, records, config, PROCESSED_DIR)
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


def stage_bayes(args: argparse.Namespace) -> None:
    """Run the Bayesian half of the walk-forward backtest and merge its forecasts.

    **About 95 minutes**, and that is the honest cost of the thing the project is named
    after: 102 NUTS fits at 4 chains x (1,000 tune + 1,000 draw), on windows growing from
    756 observations to 2,877. It is a separate stage from ``backtest`` for that reason
    alone -- both halves are the same loop, run over the same refit dates, under the same
    config.

    Two models come out of those fits. ``garch_bayes`` integrates over each posterior;
    ``garch_bayes_mean`` conditions on its mean. The second costs no sampling and exists
    so that the comparison this project rests on has one cause at a time: ``garch_bayes``
    against ``garch_bayes_mean`` is parameter uncertainty, and ``garch_bayes_mean``
    against ``garch_mle`` is the priors (research_log.md 1.13).

    A refit that fails its convergence diagnostics produces no forecasts for the 21 days
    it serves (decision D19), and those days stay NaN in the forecast table. That is
    reported here rather than filtered out: a Bayesian model that cannot be sampled on
    some windows is a finding about the model, and the evaluation layer has to see the
    gap to say so.
    """
    from dataclasses import asdict

    import pandas as pd

    from src import backtest as B

    frame_path = PROCESSED_DIR / "analysis_frame.csv"
    if not frame_path.exists():
        raise SystemExit(
            f"{frame_path} not found. Run `python run_all.py --stage data` first."
        )
    frame = pd.read_csv(frame_path, index_col=0, parse_dates=True)

    config = B.BacktestConfig(mcmc_seed=args.seed)
    print(
        f"sampler   : {config.chains} chains x ({config.tune} tune + {config.draws} "
        f"draw), target_accept {config.target_accept}, thinned by {config.thin}"
    )
    print(f"seed      : {config.mcmc_seed} (+ refit index, so no two refits share a chain)")
    print("this stage takes about 95 minutes", flush=True)

    forecasts, records = B.run_backtest(frame, config, models=B.BAYES_MODELS)
    record_frame = pd.DataFrame([asdict(r) for r in records])

    print()
    for model in B.BAYES_MODELS:
        sub = forecasts[forecasts.model == model]
        ann = (sub["variance"].mean() * 252) ** 0.5
        print(
            f"  {model:<18} {len(sub):,} rows, "
            f"{sub['variance'].notna().sum():,} finite, "
            f"mean annualised vol {ann:.2%}"
        )

    failed = record_frame[~record_frame["converged"]]
    print(
        f"  refits    : {len(record_frame) - len(failed)}/{len(record_frame)} converged, "
        f"{record_frame['seconds_elapsed'].sum() / 60:.1f} min sampling"
    )
    print(
        f"  worst R-hat {record_frame['max_r_hat'].max():.4f}, "
        f"min ess_bulk {record_frame['min_ess_bulk'].min():.0f}, "
        f"min ess_tail {record_frame['min_ess_tail'].min():.0f}, "
        f"{int(record_frame['n_divergences'].sum())} divergences in total"
    )
    for _, bad in failed.iterrows():
        print(
            f"    !! {bad['refit_date'].date()} did NOT converge: {bad['message']}  "
            f"({config.refit_every} days have no forecast)"
        )

    # The decomposition the run exists to make possible, reported where it is produced
    # rather than left for the evaluation layer to discover.
    bayes = forecasts[forecasts.model == B.BAYES_MODEL]
    mean = forecasts[forecasts.model == B.BAYES_MEAN_MODEL]
    both = bayes["variance"].notna().to_numpy() & mean["variance"].notna().to_numpy()
    if both.any():
        print()
        print("  parameter uncertainty alone (garch_bayes / garch_bayes_mean widths):")
        for level, lo, hi in (("90%", "lo_90", "hi_90"), ("95%", "lo_95", "hi_95"),
                              ("99%", "lo_99", "hi_99")):
            ratio = (
                (bayes[hi] - bayes[lo]).to_numpy()[both]
                / (mean[hi] - mean[lo]).to_numpy()[both]
            )
            print(f"    {level}: mean ratio {ratio.mean():.4f}")

    written = B.save_track("bayes", forecasts, records, config, PROCESSED_DIR)
    print()
    for path in written:
        print(f"wrote {path}")


def stage_evaluate(args: argparse.Namespace) -> None:
    """Score the forecasts and persist the evaluation tables.

    Reads ``forecasts.csv`` and nothing else: no model is refitted here, and no number
    in this stage can move without the forecast table having moved first.

    Two samples, both reported (D28). Per-model marginals use each model's own
    available days; every cross-model comparison uses the intersection, which for a
    pair involving the Bayesian model is 2,092 of the 2,134 evaluation days -- two
    refits failed their convergence diagnostics and produced no forecasts for the 21
    days each served (D19). Both are correct, they are different numbers, and every
    table carries the label and the ``n`` saying which it holds.

    The ablations stay out of the headline tables and are scored separately:
    ``garch_mle_normal`` under D15, ``garch_bayes_mean`` under D29. The second is not
    a fifth competitor but the control that splits the frequentist-Bayesian interval
    difference into the part the priors cause and the part parameter uncertainty
    causes -- see ``evaluation.decomposition_table``.
    """
    import itertools

    import pandas as pd

    from src import backtest as B
    from src import bootstrap as BS
    from src import evaluation as E

    forecast_path = PROCESSED_DIR / "forecasts.csv"
    if not forecast_path.exists():
        raise SystemExit(
            f"{forecast_path} not found. Run `python run_all.py --stage backtest` "
            "and `--stage bayes` first."
        )
    forecasts = pd.read_csv(forecast_path, parse_dates=["date"])
    scored = E.score_forecasts(forecasts)

    headline = B.HEADLINE_MODELS
    scored_models = tuple(dict.fromkeys((*headline, *B.BAYES_MODELS, *B.GARCH_MODELS)))
    common = E.common_sample(scored, headline)

    print(f"models    : {', '.join(scored_models)}")
    print(f"headline  : {', '.join(headline)}  (ablations scored, reported separately)")
    print(
        f"samples   : own days per model; common sample across the headline four is "
        f"{len(common):,} of {scored['date'].nunique():,} evaluation days"
    )
    print(
        f"bootstrap : stationary block, mean length {BS.MEAN_BLOCK_LENGTH}, "
        f"{BS.N_REPLICATIONS:,} replications, seed {BS.DEFAULT_SEED}"
    )
    print()

    # Pairwise comparisons: every headline pair, plus the two contrasts that split the
    # frequentist-Bayesian difference into its two causes.
    pairs = tuple(itertools.combinations(headline, 2)) + (
        (B.BAYES_MODEL, B.BAYES_MEAN_MODEL),
        (B.BAYES_MEAN_MODEL, "garch_mle"),
    )

    tables = {
        "eval_point_losses": pd.concat(
            [
                E.point_loss_table(scored, scored_models, sample_label="own days"),
                E.point_loss_table(
                    scored, headline, sample_label="common sample", dates=common
                ),
            ],
            ignore_index=True,
        ),
        "eval_coverage": pd.concat(
            [
                E.coverage_table(scored, scored_models, sample_label="own days"),
                E.coverage_table(
                    scored, headline, sample_label="common sample", dates=common
                ),
            ],
            ignore_index=True,
        ),
        "eval_var_backtests": pd.concat(
            [
                E.var_backtest_table(scored, scored_models, sample_label="own days"),
                E.var_backtest_table(
                    scored, headline, sample_label="common sample", dates=common
                ),
            ],
            ignore_index=True,
        ),
        "eval_pit": pd.concat(
            [
                E.pit_table(scored, scored_models, sample_label="own days"),
                E.pit_table(
                    scored, headline, sample_label="common sample", dates=common
                ),
            ],
            ignore_index=True,
        ),
        "eval_comparisons": E.comparison_table(scored, pairs),
        "eval_decomposition": E.decomposition_table(scored),
    }

    coverage = tables["eval_coverage"]
    own = coverage[coverage["sample"] == "own days"]
    print("interval coverage, empirical against nominal (own days)")
    for model in headline:
        block = own[own["model"] == model].sort_values("nominal")
        cells = "  ".join(
            f"{row.nominal:.0%}: {row.empirical:.4f}" for row in block.itertuples()
        )
        print(f"  {model:<18} n={block['n'].iloc[0]:,}  {cells}")

    var_table = tables["eval_var_backtests"]
    var_own = var_table[var_table["sample"] == "own days"]
    print()
    print("99% VaR backtests (own days). Christoffersen independence is the money test.")
    for model in headline:
        row = var_own[var_own["model"] == model].iloc[0]
        print(
            f"  {model:<18} {row.breaches:>3} breaches / "
            f"{row.nominal_rate * row.n:.0f} expected   "
            f"Kupiec p={row.kupiec_p:.4f}  independence p={row.independence_p:.4f}  "
            f"conditional p={row.conditional_coverage_p:.4f}"
        )

    losses = tables["eval_point_losses"]
    qlike = losses[(losses["loss"] == "qlike") & (losses["sample"] == "common sample")]
    print()
    print(f"mean QLIKE on the common sample (n = {len(common):,}), 95% block-bootstrap CI")
    for row in qlike.sort_values("mean").itertuples():
        print(f"  {row.model:<18} {row.mean:.4f}  [{row.ci_lower:.4f}, {row.ci_upper:.4f}]")

    comparisons = tables["eval_comparisons"]
    print()
    print("pairwise QLIKE comparisons -- the bootstrap interval carries the conclusion,")
    print("not the DM p-value (estimated parameters, multiplicity: see the docstring).")
    for row in comparisons.itertuples():
        verdict = "separates" if row.boot_excludes_zero else "cannot separate"
        print(
            f"  {row.model_a:<18} vs {row.model_b:<18} n={row.n:,}  "
            f"d={row.mean_differential:+.5f}  DM p={row.dm_p:.4f}  "
            f"boot [{row.boot_lower:+.5f}, {row.boot_upper:+.5f}]  {verdict}"
        )

    decomposition = tables["eval_decomposition"]
    print()
    print("interval width, decomposed (D29). Never quote the reported row as parameter")
    print("uncertainty: at 99% the two causes point in opposite directions.")
    for level in (0.90, 0.95, 0.99):
        block = decomposition[
            (decomposition["nominal"] == level) & (decomposition["regime"] == "all")
        ].set_index("contrast")
        cells = "  ".join(
            f"{name}: {block.loc[name, 'mean_width_ratio']:.4f}" for name in block.index
        )
        print(f"  {level:.0%}  {cells}")

    print()
    for name, table in tables.items():
        path = PROCESSED_DIR / f"{name}.csv"
        table.to_csv(path, index=False)
        print(f"wrote {path}  ({len(table)} rows)")


def stage_figures(args: argparse.Namespace) -> None:
    """Render figures from the persisted evaluation tables.

    Every figure is a function of the tables ``--stage evaluate`` wrote, so a figure
    and the number it draws can never disagree.
    """
    import pandas as pd

    from src import backtest as B
    from src import evaluation as E
    from src import figures

    forecast_path = PROCESSED_DIR / "forecasts.csv"
    coverage_path = PROCESSED_DIR / "eval_coverage.csv"
    if not coverage_path.exists():
        raise SystemExit(
            f"{coverage_path} not found. Run `python run_all.py --stage evaluate` first."
        )
    scored = E.score_forecasts(pd.read_csv(forecast_path, parse_dates=["date"]))
    coverage = pd.read_csv(coverage_path)
    decomposition = pd.read_csv(PROCESSED_DIR / "eval_decomposition.csv")

    headline = B.HEADLINE_MODELS
    own = coverage[coverage["sample"] == "own days"]

    written = [
        figures.plot_pit_histograms(
            scored, FIGURES_DIR / "08_pit_histograms.png", models=headline
        ),
        figures.plot_coverage_vs_nominal(
            own, FIGURES_DIR / "09_coverage_vs_nominal.png", models=headline
        ),
        figures.plot_var_hit_sequence(
            scored, FIGURES_DIR / "10_var_hit_sequence.png", models=headline
        ),
        figures.plot_interval_decomposition(
            decomposition, FIGURES_DIR / "11_interval_decomposition.png"
        ),
    ]
    print()
    for path in written:
        print(f"wrote {path}")


STAGES = {
    "data": stage_data,
    "eda": stage_eda,
    "backtest": stage_backtest,
    "bayes": stage_bayes,
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
        help="Base seed for the MCMC sampler; each refit samples under this seed "
             "plus its own index, so no two refits share a chain's randomness. The "
             "bootstrap seed is fixed separately in src/bootstrap.py so evaluation "
             "intervals stay reproducible independently of the sampler. "
             "(default: %(default)s)",
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
