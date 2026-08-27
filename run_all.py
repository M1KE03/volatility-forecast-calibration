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
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"

STAGE_ORDER: tuple[str, ...] = (
    "data", "eda", "backtest", "bayes", "evaluate", "robustness", "figures", "priors",
)

#: What ``--all`` runs, and it is not every stage. ``priors`` is the Stage 6
#: prior-sensitivity check: two more full Bayesian backtests, about 190 minutes, which
#: produce no headline number and answer a robustness question. Including it would turn
#: the one-command reproduction a stranger runs first into a five-hour job. It is
#: selectable with ``--stage priors`` and is documented as opt-in.
ALL_STAGES: tuple[str, ...] = tuple(s for s in STAGE_ORDER if s != "priors")

#: The two ``delta`` priors D4 considered and rejected, re-run by ``--stage priors``.
#: The frozen ``Beta(3, 1)`` is not among them: it is what every other stage already
#: ran under, and re-running it would only reproduce ``forecasts.csv``.
PRIOR_DELTA_CANDIDATES: tuple[tuple[float, float], ...] = ((10.0, 2.0), (1.0, 1.0))

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
    "robustness": "Stage 6 checks: the QLIKE ranking on the raw unscaled proxy (D10), "
                  "a 63-day refit cadence on the frequentist track, and the prior "
                  "sensitivity summary if `--stage priors` has been run. About 30 "
                  "seconds; re-runs nothing expensive.",
    "figures": "Render the figures used in report/report.md.",
    "priors": "Prior sensitivity (D4's debt): re-run the Bayesian backtest under each "
              "of the two rejected `delta` priors, writing to "
              "data/processed/prior_sensitivity/ and never touching forecasts.csv. "
              "About 190 minutes. Opt-in: not part of --all.",
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
    from src import data as D
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
        # Stage 5. Split on the *lagged* VIX label, so the conditioning information was
        # available when the forecast was made -- the regime definition carries no
        # look-ahead by construction, and the report says so where the figure appears.
        "eval_regime_losses": E.regime_loss_table(
            scored, scored_models, sample_label="own days"
        ),
        "eval_regime_coverage": E.regime_coverage_table(
            scored, scored_models, sample_label="own days"
        ),
        "eval_regime_var": E.regime_var_table(
            scored, scored_models, sample_label="own days"
        ),
        # Where the breaches land, and whether the split is symmetric. A coverage number
        # cannot show this, and for these models it is the sharpest thing in the stage.
        "eval_tail_asymmetry": E.tail_asymmetry_table(
            scored, scored_models, sample_label="own days"
        ),
        # Differences *between* regimes. "Rejects here and not there" is a claim about
        # two regimes and needs this table, not two rows of the one above (#46).
        "eval_regime_differences": E.regime_difference_table(
            scored, scored_models, sample_label="own days"
        ),
    }

    # The regime sensitivity the plan asks for: terciles of trailing 21-day Parkinson
    # volatility in place of the VIX bands, with the cut-points estimated on the warm-up
    # window alone so the labels carry no look-ahead either. Third on the cut list, and
    # cheap now that the tables are functions -- it is the same analysis under a
    # different partition, not a second analysis.
    frame = pd.read_csv(
        PROCESSED_DIR / "analysis_frame.csv", index_col=0, parse_dates=True
    )
    thresholds = D.trailing_vol_thresholds(frame["parkinson_var"])
    trailing = scored.assign(
        regime=scored["date"]
        .map(D.assign_trailing_vol_regime(frame["parkinson_var"]))
        .astype(str)
    )
    label = "trailing-vol terciles"
    tables["eval_regime_coverage_trailing"] = E.regime_coverage_table(
        trailing, scored_models, sample_label=label
    )
    tables["eval_regime_var_trailing"] = E.regime_var_table(
        trailing, scored_models, sample_label=label
    )

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

    regime_var = tables["eval_regime_var"]
    print()
    print("99% VaR breach rate by regime, with 95% block-bootstrap intervals.")
    print("Christoffersen is deliberately absent here: consecutive rows of a regime")
    print("subsample can be months apart, so its transitions would be fictitious (D32).")
    for model in headline:
        cells = []
        for row in regime_var[regime_var["model"] == model].itertuples():
            flag = " " if row.covers_nominal else "*"
            cells.append(
                f"{row.regime}: {row.rate:.4f} [{row.ci_lower:.4f}, {row.ci_upper:.4f}]{flag}"
            )
        print(f"  {model:<18} " + "  ".join(cells))
    print("  (* = the interval excludes the nominal 1% rate)")

    asymmetry = tables["eval_tail_asymmetry"]
    print()
    print("Where the interval breaches land. Under a symmetric predictive the two tails")
    print("should be equally populated whatever the model gets wrong about scale.")
    for model in headline:
        block = asymmetry[asymmetry["model"] == model]
        skew = block["residual_skew"].iloc[0]
        cells = "  ".join(
            f"{row.nominal:.0%}: {row.n_below}/{row.n_above}"
            f"{'*' if row.symmetry_p < 0.01 else ' '}"
            for row in block.itertuples()
        )
        print(f"  {model:<18} below/above  {cells}   residual skew {skew:+.3f}")
    print("  (* = the split rejects symmetry at 1%)")

    differences = tables["eval_regime_differences"]
    print()
    print("Differences BETWEEN regimes in the 99% breach rate. A regime that rejects")
    print("against nominal where another does not is not thereby different from it.")
    for model in headline:
        for row in differences[differences["model"] == model].itertuples():
            verdict = "separates" if row.separates else "cannot separate"
            print(
                f"  {model:<18} {row.regime_a:>8} - {row.regime_b:<8} "
                f"{row.difference:+.4f}  [{row.ci_lower:+.4f}, {row.ci_upper:+.4f}]  "
                f"{verdict}"
            )

    regime_coverage = tables["eval_regime_coverage"]
    at_99 = regime_coverage[regime_coverage["nominal"] == 0.99]
    print()
    print("99% two-sided coverage by regime, and where the breaches land.")
    print("A model can hold its total coverage while putting every breach in one tail,")
    print("which is a substantive failure for a risk model and not a rounding detail.")
    for model in headline:
        for row in at_99[at_99["model"] == model].itertuples():
            print(
                f"  {model:<18} {row.regime:<9} n={row.n:>5,}  "
                f"coverage {row.empirical:.4f}  "
                f"breaches {row.n_below:>3} below / {row.n_above:>3} above  "
                f"({0.005 * row.n:.1f} expected each side)"
            )

    print()
    print(
        "Sensitivity: the same split on terciles of trailing 21-day Parkinson vol, "
        f"cut at {thresholds[0]:.3e} and {thresholds[1]:.3e}"
    )
    print("(estimated on the warm-up window alone, so the labels carry no look-ahead).")
    print("These buckets are not the VIX regimes and no row compares across the two.")
    for model in headline:
        cells = []
        for row in tables["eval_regime_var_trailing"].itertuples():
            if row.model != model:
                continue
            flag = " " if row.covers_nominal else "*"
            cells.append(f"{row.regime}: {row.rate:.4f} (n={row.n:,}){flag}")
        print(f"  {model:<18} " + "  ".join(cells))

    print()
    for name, table in tables.items():
        path = PROCESSED_DIR / f"{name}.csv"
        table.to_csv(path, index=False)
        print(f"wrote {path}  ({len(table)} rows)")


#: Refit cadence for the Stage 6 sensitivity check, against the locked 21 days.
CADENCE_SENSITIVITY = 63


def stage_robustness(args: argparse.Namespace) -> None:
    """Stage 6: the checks that ask whether the conclusions hinge on a choice.

    Three of them, in ascending order of what they would cost to be wrong about.

    **The raw-proxy QLIKE ranking**, owed by D10. Every headline point loss is scored
    against the Parkinson series multiplied by the frozen constant ``c = 1.517318``. If
    the ranking of models moved when ``c`` was removed, the ranking would be a fact
    about the constant rather than about the models. It is a re-scoring, not a re-run.

    **The 63-day refit cadence**, second on the governing plan's cut list. Frequentist
    track only, and that is a stated choice rather than an oversight: the question is
    whether conclusions depend on how often parameters are re-estimated, and the
    frequentist track answers it in twenty seconds where the Bayesian track would cost
    another ninety-five minutes. The report says so.

    **Prior sensitivity**, owed by D4, if ``--stage priors`` has been run. That stage is
    opt-in and takes about 190 minutes; this one reports what it produced and says
    plainly when it has produced nothing yet, rather than silently omitting the check.

    Like ``priors``, the cadence re-run writes to its own directory and never calls
    ``save_track``, so it cannot reach ``forecasts.csv``.
    """
    from dataclasses import asdict

    import pandas as pd

    from src import backtest as B
    from src import data as D
    from src import evaluation as E

    forecast_path = PROCESSED_DIR / "forecasts.csv"
    frame_path = PROCESSED_DIR / "analysis_frame.csv"
    for path in (forecast_path, frame_path):
        if not path.exists():
            raise SystemExit(f"{path} not found. Run the earlier stages first.")

    forecasts = pd.read_csv(forecast_path, parse_dates=["date"])
    frame = pd.read_csv(frame_path, index_col=0, parse_dates=True)
    scored = E.score_forecasts(forecasts)
    headline = B.HEADLINE_MODELS
    common = E.common_sample(scored, headline)
    tables: dict[str, pd.DataFrame] = {}

    # --- 1. Does the QLIKE ranking survive removing c? (D10) ----------------------
    raw = E.score_forecasts(forecasts, proxy_var=frame["parkinson_var"])
    scaled_losses = E.point_loss_table(
        scored, headline, sample_label="scaled proxy (headline)", dates=common
    )
    raw_losses = E.point_loss_table(
        raw, headline, sample_label="raw Parkinson", dates=common
    )
    tables["eval_raw_proxy_losses"] = pd.concat(
        [scaled_losses, raw_losses], ignore_index=True
    )

    print(f"1. QLIKE ranking, scaled against raw proxy (n = {len(common):,})")
    orderings = {}
    for label, table in (("scaled", scaled_losses), ("raw", raw_losses)):
        qlike = table[table["loss"] == "qlike"].sort_values("mean")
        orderings[label] = list(qlike["model"])
        cells = "  ".join(
            f"{row.model} {row.mean:.4f}" for row in qlike.itertuples()
        )
        print(f"   {label:<7} {cells}")
    verdict = (
        "unchanged" if orderings["scaled"] == orderings["raw"] else "CHANGED"
    )
    print(f"   verdict: the ranking is {verdict} when c is removed (D10).")

    # --- 2. Does anything hinge on the 21-day refit cadence? ----------------------
    print()
    print(f"2. Refit cadence {B.REFIT_EVERY} -> {CADENCE_SENSITIVITY} days, "
          "frequentist track only (a stated choice; see the docstring)", flush=True)
    cadence_dir = PROCESSED_DIR / f"cadence_{CADENCE_SENSITIVITY}"
    cadence_dir.mkdir(parents=True, exist_ok=True)
    cadence_config = B.BacktestConfig(refit_every=CADENCE_SENSITIVITY)
    cadence_forecasts, cadence_records = B.run_backtest(
        frame, cadence_config, models=B.FREQUENTIST_MODELS
    )
    cadence_forecasts.to_csv(
        cadence_dir / "forecasts.csv",
        index_label="date",
        date_format="%Y-%m-%d",
        lineterminator="\n",
    )
    pd.DataFrame([asdict(r) for r in cadence_records]).to_csv(
        cadence_dir / "refit_records.csv",
        index=False,
        date_format="%Y-%m-%d",
        lineterminator="\n",
    )

    cadence_scored = E.score_forecasts(cadence_forecasts.reset_index())
    rows = []
    for model in B.FREQUENTIST_MODELS:
        base = E.var_backtest_table(scored, (model,), sample_label="21 days")
        alt = E.var_backtest_table(cadence_scored, (model,), sample_label="63 days")
        base_q = E.point_loss_table(scored, (model,), sample_label="21 days")
        alt_q = E.point_loss_table(cadence_scored, (model,), sample_label="63 days")
        rows.append(
            {
                "model": model,
                "qlike_21": base_q.loc[base_q["loss"] == "qlike", "mean"].item(),
                "qlike_63": alt_q.loc[alt_q["loss"] == "qlike", "mean"].item(),
                "breaches_21": base["breaches"].item(),
                "breaches_63": alt["breaches"].item(),
                "kupiec_p_21": base["kupiec_p"].item(),
                "kupiec_p_63": alt["kupiec_p"].item(),
            }
        )
    tables["eval_cadence_comparison"] = pd.DataFrame(rows)

    # The baselines estimate nothing, so the refit cadence is a genuine no-op for them.
    # Anything else would mean the cadence is reaching a model it has no business
    # touching, which is a harness bug rather than a robustness finding.
    #
    # Compared against a *fresh in-memory* 21-day run rather than against
    # ``forecasts.csv``, so the assertion can be exact equality. The stored table has
    # been through a CSV round trip and differs from the in-memory value in the last
    # bit, which is not a difference worth a tolerance -- and a tolerance here would be
    # the loosest link in a check whose whole job is to be strict. The baselines cost
    # seconds, so buying the exactness is free.
    baseline_21, _ = B.run_backtest(frame, B.BacktestConfig(), models=B.BASELINE_MODELS)
    for model in B.BASELINE_MODELS:
        a = baseline_21.loc[baseline_21["model"] == model, "variance"].to_numpy()
        b = cadence_forecasts.loc[
            cadence_forecasts["model"] == model, "variance"
        ].to_numpy()
        if not (a == b).all():
            raise SystemExit(
                f"cadence changed {model!r}, which estimates no parameters -- this is a "
                "harness bug, not a robustness result"
            )
    print(f"   baselines identical at both cadences, as they must be "
          f"({', '.join(B.BASELINE_MODELS)})")
    for row in tables["eval_cadence_comparison"].itertuples():
        if row.model in B.BASELINE_MODELS:
            continue
        print(
            f"   {row.model:<18} QLIKE {row.qlike_21:.4f} -> {row.qlike_63:.4f}   "
            f"99% VaR breaches {row.breaches_21} -> {row.breaches_63}   "
            f"Kupiec p {row.kupiec_p_21:.4f} -> {row.kupiec_p_63:.4f}"
        )

    # --- 3. Prior sensitivity, if the runs have happened (D4) ---------------------
    print()
    print("3. Prior sensitivity on delta (D4)")
    prior_dir = PROCESSED_DIR / "prior_sensitivity"
    prior_files = sorted(prior_dir.glob("forecasts_delta_*.csv")) if prior_dir.exists() else []
    if not prior_files:
        print("   not yet run. `python run_all.py --stage priors` (~190 min).")
        print("   D4's obligation is outstanding until it is, and if it is cut it must")
        print("   be cut explicitly into the limitations section rather than by omission.")
    else:
        # Every prior run is compared on the intersection of all of them with the
        # headline sample. The three runs lose different days to failed refits -- 42
        # under the frozen prior, none under Beta(10,2), 21 under Beta(1,1) -- so a
        # breach count on each run's own days would not be comparable across priors,
        # which is the only comparison this table exists to support (D28).
        runs = {"delta_3_1 (frozen, D4)": scored}
        for path in prior_files:
            runs[path.stem.replace("forecasts_", "")] = E.score_forecasts(
                pd.read_csv(path, parse_dates=["date"])
            )
        prior_common = common
        for run in runs.values():
            prior_common = prior_common.intersection(
                E.common_sample(run, B.BAYES_MODELS)
            )

        def series(run: pd.DataFrame, model: str, column: str):
            block = run[
                (run["model"] == model) & run["date"].isin(prior_common)
            ].sort_values("date")
            return block[column].to_numpy()

        rows = []
        for label, run in runs.items():
            row = {"prior": label, "n": len(prior_common)}
            for level in (0.90, 0.95, 0.99):
                key = E.level_key(level)
                row[f"param_uncertainty_{key}"] = float(
                    (
                        series(run, B.BAYES_MODEL, f"width_{key}")
                        / series(run, B.BAYES_MEAN_MODEL, f"width_{key}")
                    ).mean()
                )
                # Always against the *same* frequentist plug-in: the question is what
                # each prior does to the point estimate, and garch_mle is the fixed
                # reference all three are measured from.
                row[f"priors_{key}"] = float(
                    (
                        series(run, B.BAYES_MEAN_MODEL, f"width_{key}")
                        / series(scored, "garch_mle", f"width_{key}")
                    ).mean()
                )
            row["coverage_99"] = float(series(run, B.BAYES_MODEL, "inside_99").mean())
            row["breaches_99"] = int(series(run, B.BAYES_MODEL, "exceedance").sum())
            rows.append(row)
        tables["eval_prior_sensitivity"] = pd.DataFrame(rows)

        print(f"   compared on {len(prior_common):,} days common to all three runs (D28)")
        print("   parameter uncertainty is garch_bayes / garch_bayes_mean within each run;")
        print("   the priors are that run's garch_bayes_mean / the one garch_mle plug-in.")
        for row in tables["eval_prior_sensitivity"].itertuples():
            print(
                f"   {row.prior:<22} param-unc "
                f"{row.param_uncertainty_90:.4f}/{row.param_uncertainty_95:.4f}/"
                f"{row.param_uncertainty_99:.4f}   priors "
                f"{row.priors_90:.4f}/{row.priors_95:.4f}/{row.priors_99:.4f}   "
                f"99% coverage {row.coverage_99:.4f}, {row.breaches_99} breaches"
            )

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
        figures.plot_regime_coverage(
            pd.read_csv(PROCESSED_DIR / "eval_regime_coverage.csv"),
            FIGURES_DIR / "12_regime_coverage.png",
            models=headline,
        ),
        figures.plot_regime_var_rate(
            pd.read_csv(PROCESSED_DIR / "eval_regime_var.csv"),
            FIGURES_DIR / "13_regime_var_rate.png",
            models=headline,
        ),
        figures.plot_tail_allocation(
            pd.read_csv(PROCESSED_DIR / "eval_tail_asymmetry.csv"),
            FIGURES_DIR / "14_tail_allocation.png",
            models=headline,
        ),
    ]
    print()
    for path in written:
        print(f"wrote {path}")


def stage_priors(args: argparse.Namespace) -> None:
    """Re-run the Bayesian backtest under each rejected ``delta`` prior (D4's debt).

    **About 190 minutes: two more full Bayesian backtests.** That is the honest cost of
    the obligation D4 incurred when it froze ``delta ~ Beta(3, 1)`` -- the log promised
    prior sensitivity across all three candidates on evaluation-window forecasts, and
    the check turned out to be a second and third backtest rather than a paragraph.

    It is deliberately **not** part of ``--all``. It answers a robustness question and
    produces no headline number; making an hour-and-a-half pipeline into a five-hour one
    so that a Stage 6 check rides along would be a bad trade, and ``--all`` is the
    command a stranger runs first.

    **Nothing here can touch the headline results, and that is enforced structurally
    rather than by care.** These runs write to ``data/processed/prior_sensitivity/``,
    under their own file names, and never call ``save_track`` or ``merge_tracks`` -- the
    machinery that rebuilds ``forecasts.csv`` iterates over ``backtest.TRACKS``, which
    these are not in. The frozen priors stay frozen: ``models.PRIOR_DELTA`` is not
    edited, the alternative is passed as an argument, and a test pins the default so a
    sensitivity run cannot become the default run by a one-line change.

    Everything except the prior is held identical to the production run -- same config,
    same refit dates, same ``mcmc_seed``, so the same chain randomness. The prior is the
    only thing that moves, which is what makes the comparison a sensitivity check rather
    than two unrelated runs.
    """
    import json
    from dataclasses import asdict

    import pandas as pd

    from src import backtest as B
    from src import models as M

    frame_path = PROCESSED_DIR / "analysis_frame.csv"
    if not frame_path.exists():
        raise SystemExit(
            f"{frame_path} not found. Run `python run_all.py --stage data` first."
        )
    frame = pd.read_csv(frame_path, index_col=0, parse_dates=True)

    out_dir = PROCESSED_DIR / "prior_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    config = B.BacktestConfig(mcmc_seed=args.seed)

    print(f"frozen prior (D4)   : delta ~ Beta{M.PRIOR_DELTA}  -- not re-run here")
    print(f"candidates          : {', '.join(f'Beta{c}' for c in PRIOR_DELTA_CANDIDATES)}")
    print(f"output              : {out_dir}{os.sep}  (forecasts.csv is never touched)")
    print(f"seed                : {config.mcmc_seed}, identical to the production run")
    print("this stage takes about 190 minutes", flush=True)

    for candidate in PRIOR_DELTA_CANDIDATES:
        label = f"delta_{candidate[0]:g}_{candidate[1]:g}".replace(".", "p")
        print()
        print(f"--- delta ~ Beta{candidate} ---", flush=True)

        forecasts, records = B.run_backtest(
            frame,
            config,
            models=B.BAYES_MODELS,
            prior_delta=candidate,
        )
        record_frame = pd.DataFrame([asdict(r) for r in records])

        fits = record_frame[record_frame["model"] == B.BAYES_MODEL]
        failed = fits[~fits["converged"]]
        print(
            f"  {len(fits) - len(failed)}/{len(fits)} refits converged, "
            f"{fits['seconds_elapsed'].sum() / 60:.1f} min sampling"
        )
        for _, bad in failed.iterrows():
            print(f"    !! {bad['refit_date'].date()} did NOT converge: {bad['message']}")

        sub = forecasts[forecasts.model == B.BAYES_MODEL]
        print(
            f"  {sub['variance'].notna().sum():,} of {len(sub):,} days forecast; "
            f"mean annualised vol {(sub['variance'].mean() * 252) ** 0.5:.2%}"
        )

        forecast_path = out_dir / f"forecasts_{label}.csv"
        records_path = out_dir / f"refit_records_{label}.csv"
        config_path = out_dir / f"config_{label}.json"

        forecasts.to_csv(
            forecast_path, index_label="date", date_format="%Y-%m-%d", lineterminator="\n"
        )
        record_frame.to_csv(
            records_path, index=False, date_format="%Y-%m-%d", lineterminator="\n"
        )
        payload = asdict(config)
        payload["estimation_window"] = config.estimation_window.value
        payload["prior_delta"] = list(candidate)
        payload["frozen_prior_delta"] = list(M.PRIOR_DELTA)
        with config_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")

        for path in (forecast_path, records_path, config_path):
            print(f"  wrote {path}")


STAGES = {
    "data": stage_data,
    "eda": stage_eda,
    "backtest": stage_backtest,
    "bayes": stage_bayes,
    "evaluate": stage_evaluate,
    "robustness": stage_robustness,
    "figures": stage_figures,
    "priors": stage_priors,
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
        help="Run the pipeline in order. Excludes `priors`, which is a 190-minute "
             "robustness check rather than part of producing the results; run it "
             "explicitly with `--stage priors`.",
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
    to_run = ALL_STAGES if args.all else (args.stage,)
    for name in to_run:
        print(f"=== stage: {name} ===", flush=True)
        STAGES[name](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
