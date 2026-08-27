# Handoff: how to continue

**State as of 2026-08-27. Stages 4 and 5 complete.** All four forecasters exist, plus
two ablations. The evaluation layer scores all of it — point losses, PIT, coverage, VaR
backtests, comparisons, the interval decomposition, and the regime split with bootstrap
intervals throughout — writing eleven tables and six figures from `forecasts.csv` alone
and refitting nothing. All four never-cut items are discharged.

**Stage 6 is next, and part of it is already running.** The two prior-sensitivity
backtests owed by D4 were launched from `--stage priors` and take about 190 minutes; see
§5 for what to do when they land.

**Read §4 before you quote a single number.** Three of the headline results are not what
the governing plan anticipated, and one of them reverses the question the project is
named after.

This document is written for whoever picks the project up next — including a future
session of the same work. It assumes no memory of how anything got here.

Read in this order:

1. This file — where things stand and what to do next.
2. `docs/project1-implementation-plan.md` — the **governing plan**. It is the contract.
3. `research_log.md` §1 — the decision register. Locked decisions are not revisable.
4. `docs/problems-and-solutions.md` — the traps already hit, so they are not hit twice.

---

## 1. Get running in five minutes

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python run_all.py --stage data       # cached; no network unless --refresh
python run_all.py --stage eda        # diagnostics + 4 figures
python run_all.py --stage backtest   # baselines + GARCH + 3 figures (~1 min)
python run_all.py --stage bayes      # the Bayesian track (~95 min)
python run_all.py --stage evaluate   # every table, from forecasts.csv (~30s)
python run_all.py --stage figures    # figures 08-13, from those tables (~10s)
python run_all.py --stage priors     # D4's prior sensitivity, ~190 min, opt-in
pytest -q                            # expect 342 passed, 1 skipped (~13 min)
pytest -m "not slow" -q              # inner loop (~2 min)
pytest --bayes-audit -m bayes_audit   # the audit with NUTS itself (~40 min); run at §1.15
```

**The backtest is two stages, and that is a cost decision, not a design one.** The
frequentist track is 306 optimiser fits and takes a minute; the Bayesian track is 102
NUTS fits and takes about ninety-five. Each writes its own partial tables
(`forecasts_<track>.csv`) and rebuilds the merged `forecasts.csv` from whichever partials
are on disk, so re-running the cheap one never re-runs the expensive one and the
evaluation layer still reads one table. The merge compares the two configs and refuses to
join runs made under different ones (D24).

**`evaluate` and `figures` are cheap and re-runnable at will.** Neither refits anything.
`evaluate` reads `forecasts.csv` and writes eleven `eval_*.csv` tables; `figures` reads
only those tables, so a figure and the number it draws cannot disagree.

**`priors` is opt-in and is not part of `--all`.** It is two more full Bayesian backtests
under the `delta` priors D4 rejected — about 190 minutes, no headline number — so
including it would turn the one-command reproduction into a five-hour job. It writes to
`data/processed/prior_sensitivity/` and cannot touch `forecasts.csv`: the merge that
rebuilds that file iterates over `backtest.TRACKS`, which these runs are not in.

**One prerequisite `pip` does not supply: a C/C++ compiler on `PATH`.** PyMC's PyTensor
backend needs one. This machine uses MinGW-w64 GCC 16.2.0 (UCRT, x86_64) at
`C:\Users\micha\toolchains\mingw64`, already on the user `PATH`. Verify:

```bash
g++ --version                                       # expect 16.2.0
python -c "import pytensor; print(pytensor.config.cxx)"
```

If that second command prints an empty string on a new machine, install a compiler before
touching the Bayesian track. Provenance and the verified SHA-256 are in `research_log.md`
§1.6. Nothing in Stages 4-7 needs it.

---

## 2. What exists, precisely

| Module | State |
|---|---|
| `src/data.py` | **Complete.** Download, SHA-256 manifest, returns, Parkinson proxy, VIX regimes, `PROXY_SCALE_C`, and the trailing-volatility tercile labels used for the Stage 5 sensitivity. |
| `src/eda.py` | **Complete.** ARCH-LM, Ljung-Box, ADF, ACF (`series_acf` and `squared_return_acf`). |
| `src/figures.py` | **Partial.** House style + 13 figures (Stages 0-5). More added per stage. |
| `src/backtest.py` | **Complete.** The loop, both cadences, both tracks, the model registry, the partial/merge machinery, long-format output. |
| `src/models.py` | **Complete.** Interface, both baselines, the shared likelihood, MLE, plug-in predictive, the frozen priors, `log_prior`/`log_posterior`, the PyMC/NUTS sampler, the mixture posterior predictive. |
| `src/evaluation.py` | **Complete.** QLIKE/MSE, coverage, PIT + KS, Kupiec, Christoffersen independence and conditional coverage, DM with HLN, `common_sample`, the six result-table builders, and the three regime tables with bootstrap intervals. |
| `src/bootstrap.py` | **Complete.** Politis-Romano indices, mean CI, paired loss-differential and coverage-difference CIs. |

Notebooks are the presentation layer and define no analysis logic: `01_eda.ipynb` and
`02_results.ipynb` read from `src/` and from the persisted tables. They are committed
**unexecuted**, with no stored outputs — `run_all.py` is what reproduces the results, and
a notebook carrying its own outputs would be a second, unversioned source of numbers.

Artefacts in `data/processed/`: `analysis_frame.csv` (2,890 rows), `forecasts.csv`
(12,804 rows = 2,134 dates × 6 models), `refit_records.csv` (408 rows), the per-track
partials, `backtest_config.json`, and the Stage 4 tables — `eval_point_losses.csv`,
`eval_coverage.csv`, `eval_var_backtests.csv`, `eval_pit.csv`, `eval_comparisons.csv`,
`eval_decomposition.csv`, `eval_regime_losses.csv`, `eval_regime_coverage.csv`,
`eval_regime_var.csv`, and the two `_trailing` sensitivity tables.

`data/processed/prior_sensitivity/` holds the Stage 6 prior runs, one forecast table,
refit-record table and config per candidate prior. Nothing else reads that directory.

---

## 3. Rules that are not yours to change

Violating any of these invalidates the project rather than merely altering it.

**Locked design** (`research_log.md` §1.1). Asset, sample window, warm-up block, 21-day
refit cadence, expanding window, interval levels, QLIKE primary, DM test, stationary
bootstrap, VIX thresholds at 15/25. Fixed before any code was written. That is what makes
them credible; changing one after seeing a result is the look-ahead the project exists to
avoid.

**The scale convention (D10).** Every variance forecast *and* the evaluation proxy are on
the close-to-close return-variance scale. `c = 1.517318`, frozen, warm-up only. Any new
model must produce variance on this scale. Do not re-derive `c`; do not apply it twice.

**The priors (D4) are frozen and were frozen before any Bayesian out-of-sample number
existed.** They are stated once, in the block above `log_prior` in `models.py`, and
`test_priors_are_the_ones_frozen_at_d4` breaks if any of them moves. Prior sensitivity is
a Stage 6 robustness question, not a modelling knob.

**The Stage 4 decisions (§1.16), and D29 above all.** No statement about parameter
uncertainty may be sourced from `garch_bayes` against `garch_mle`. That comparison moves
two things at once, and `eval_decomposition.csv` is the only place the two are separated.
D28 (every table names its sample and carries its `n`), D30 (the DM lag rule) and D31
(Giacomini-White stays out of scope) sit alongside it.

**The never-cut list**, from the governing plan §5: the look-ahead audit, the
Christoffersen test, bootstrap CIs on regime coverage, and the limitations section. If the
budget runs out, cut from §5's ordered list — not from these. **All four are now
discharged**: the look-ahead audit, the Christoffersen test, bootstrap CIs on every
per-regime coverage estimate, and the limitations section, which is Stage 7's to write and
already has its content in §4 below.

**Append-only log.** `research_log.md` is never edited to match a later result. New
decisions get new numbered subsections; §1.12 supersedes parts of §1.10 exactly that way.

---

## 4. What the results actually say — **read before quoting anything**

*The full record is `research_log.md` §1.16-1.17 and the Stage 4 and 5 changelog entries.
What follows is what the write-up needs and what a reader will otherwise get wrong.*

**The evaluation layer reproduces §1.13 exactly** from the stored table rather than from
the run that produced it — 0.9975 / 0.9989 / 1.0032 for parameter uncertainty at
90 / 95 / 99, and 0.9970 / 0.9926 / 0.9811 for the priors. If those six numbers ever move
without `forecasts.csv` moving, something in `evaluation.py` has broken.

**Result 1 — separation is GARCH versus no GARCH, not estimator versus estimator.** The
plan predicted the four models would be hard to separate, with that difficulty as the
setup for the calibration act. Half of that holds. Mean QLIKE on the 2,092-day common
sample: `garch_bayes` 0.4577, `garch_mle` 0.4599, `ewma` 0.5239, `yesterday` 0.7809, with
both GARCH models clear of both baselines by bootstrap intervals nowhere near zero. The
pair that is hard to separate is the one the research question turns on: `garch_mle`
against `garch_bayes` is 0.0022, half a percent of the loss level, clearing zero by a hair
at `[+0.00019, +0.00418]` — and `garch_bayes_mean` against `garch_mle` does not clear it
at all.

**Result 2 — the 99% VaR fails on the level of tail risk, not its timing.** Over the full
sample all four models fail Kupiec — 87, 54, 37 and 37 breaches against 21 expected; the
least strongly rejected, `garch_mle`, still at p = 0.002. **Christoffersen's independence
test rejects for none of them** (p = 0.67 and 0.69 for the GARCH models, 0.76 for
`yesterday`, 0.058 for `ewma`). The plan called independence the money test on the
argument that correct average coverage can hide clustered breaches; here the average
coverage is wrong and the clustering is absent, which is the mirror image. **A
non-rejection on 37 events is a failure to reject on a small sample, not a demonstration
that breaches are well timed** — say so, and do not recycle the plan's sentence as though
it had been borne out.

**Result 3 — parameter uncertainty changes no coverage number at all.** `garch_bayes` and
`garch_bayes_mean` have identical coverage at every level, in every regime, and identical
99% breach counts, despite the posterior predictive being 0.32% wider at 99%. Nothing in
2,092 days of returns lands in the gap. At n ≥ 750, integrating over parameter uncertainty
is not what determines whether a risk model's intervals are calibrated. The governing
plan's risk register called this outcome; it has arrived, and it is a finding rather than
a null result.

**Result 4 — and this is the one that reverses the question — calibration survives the
crisis and fails in the middle.** 99% VaR breach rate by lagged-VIX regime, with 95%
bootstrap intervals:

| model | calm (n=757) | normal (n=1,030) | stressed (n=347) |
|---|---|---|---|
| `garch_mle` | 1.19% [0.53, 1.85] | **2.33% [1.65, 3.11]** | 1.15% [0.29, 2.02] |
| `garch_bayes` | 1.19% [0.53, 1.98] | **2.31% [1.51, 3.12]** | 1.47% [0.29, 2.93] |
| `ewma` | 1.59% [0.79, 2.38] | 2.91% [2.04, 3.88] | 3.46% [1.44, 6.05] |

Both GARCH models are indistinguishable from nominal in calm *and* in stress and clearly
too high in the middle band. The baselines degrade monotonically with volatility, which is
what one would have predicted for all four. The project asks whether 99% still means 99%
when VIX > 25; for the GARCH models the answer is yes, and the failure is in the regime
nobody would have examined.

**The mechanism is tail misallocation, and a two-sided number cannot show it.** At the 99%
two-sided level in the normal regime, `garch_mle` has **14 breaches below the interval and
none above**, against 5.2 expected in each tail. Total coverage there is 0.9864 against a
nominal 0.99 — a near miss — while every single breach is a loss. This is why
`interval_coverage` splits the tails, and it is the most reportable thing in Stage 5.

**The sensitivity does not overturn Result 4 and sharpens it.** Under terciles of trailing
21-day Parkinson volatility with cut points from the warm-up window alone (D35), the GARCH
models are closest to nominal in the *top* tercile (1.35%, Kupiec p = 0.29) and worst in
the bottom one (2.06%, p = 0.012). The two definitions disagree about which non-stressed
bucket is weakest and agree on what matters: **these models are not worse in stress, they
are worse outside it.** The two partitions are not comparable row by row — the tercile
"stressed" bucket is 1,040 days and a far weaker notion of stress than VIX above 25.

**Two cautions the report must carry.** The stressed intervals are wide: [0.29%, 2.02%]
admits rates from a third of nominal to double it, so "survives stress" means "this sample
cannot show it failing". And *why* the middle band is the weak spot is not established —
the plausible story, that 15–25 is where regime transitions happen and a GARCH forecast
lags a change in level by construction, is a conjecture this design cannot test.

**Four things about how to read the tables.**

1. **Every table names its sample and carries its `n` (D28).** Per-model rows use each
   model's own days, so a four-model table mixes n = 2,134 and n = 2,092; every pairwise
   comparison uses the *pairwise* intersection; headline tables are additionally
   recomputed on the four-model common sample under the label `common sample`. Nothing is
   inferable — read the column.
2. **A missing forecast is an error, not a dropped row.** Every function in
   `evaluation.py` and `bootstrap.py` raises on a non-finite input. The fix is to select a
   sample with `evaluation.common_sample` and report its `n`, never to filter the NaN away
   at the call site. Regime subsamples below 30 days raise too (D33).
3. **The ablations stay out of the headline tables**, filtered on
   `backtest.HEADLINE_MODELS` (D15 for `garch_mle_normal`, D29 for `garch_bayes_mean`).
   They are scored — they appear in the `own days` rows — but they are not competitors.
4. **Christoffersen is absent from the regime tables on purpose (D32)**, because
   consecutive rows of a regime subsample can be months apart. Do not add it.

---

## 5. Stage 6 — robustness (1.5–2h). **Start here.**

**The expensive part is already running, or has already finished.** `--stage priors`
launched two full Bayesian backtests under the `delta` priors D4 considered and rejected,
`Beta(10, 2)` and `Beta(1, 1)`, at the production config and the same `mcmc_seed`, so the
prior is the only thing that differs. They write `forecasts_delta_*.csv`,
`refit_records_delta_*.csv` and `config_delta_*.json` into
`data/processed/prior_sensitivity/`.

**When they land, the analysis is cheap and the plumbing already exists.** Score each with
`evaluation.score_forecasts`, and run `coverage_table`, `var_backtest_table` and
`decomposition_table` against them exactly as the headline tables are built. The question
to answer is narrow and should be stated narrowly: **how much of the interval difference
attributed to "the priors" at §1.13 is specific to `Beta(3, 1)`?** A smoke check on the
warm-up window before the runs began put mean `alpha+beta` at 0.9392 under the frozen
prior, 0.9322 under `Beta(10, 2)` and 0.9386 under `Beta(1, 1)`, with the 99th percentile
of persistence at 0.9969, 0.9845 and 0.9998 — so the priors do separate, most visibly at
the stationarity boundary, and the sensitivity is real rather than a formality.

The rest of Stage 6:

- **Raw-Parkinson QLIKE ranking**, owed by D10 — show the ranking does not hinge on `c`.
  `score_forecasts` takes the scaled proxy from the forecast table, so this needs the
  unscaled series from the analysis frame and a second pass; it is a table, not a re-run.
- **GARCH-normal against GARCH-t at 99%.** Already measured at Stage 4: `garch_mle_normal`
  fails the PIT at 2.6e-4 and takes 53 breaches against 37 for the t. This is a write-up
  task, not a computation.
- **63-day refit cadence**, second on the cut list. A frequentist-only re-run is a minute
  (`BacktestConfig(refit_every=63)`); the Bayesian equivalent is another 95, so decide
  explicitly whether the check is frequentist-only and say which in the report.

**One trap specific to this stage.** `prior_delta` is threaded through
`sample_garch_posterior`, `build_bayes_paths` and `run_backtest` as an argument that
defaults to the frozen D4 value. Two tests guard it — one that every default is
`PRIOR_DELTA`, one that the argument actually reaches the model graph. Do not "simplify"
it into a module constant: that would put a one-line edit between the frozen priors and
every headline number.

---

## 6. Stage 7, in brief

**Write-up (3–4h).** Two pages. Never claim a model is better on a lower loss in
one period; conclusions rest on bootstrap intervals, and an interval containing zero is a
legitimate reportable finding. Keep the seven quantities distinct throughout: observed
returns, the proxy, conditional variance forecasts, predictive intervals, VaR forecasts,
parameter uncertainty, innovation uncertainty. The four §4 results are the spine of the
paper, and three of them contradict what the plan expected — say so plainly. A
pre-registered protocol that produced a surprise is the most credible thing this project
has, and Result 4 in particular inverts the question in the title: calibration survives
the crisis and fails in the quiet.

The limitations section is on the never-cut list and its content already exists: one
asset, one horizon, roughly two stress episodes; a proxy that is biased for the forecast
target; wide intervals on every crisis subsample; several tests that did not reject on
samples too small to have fired; and no explanation for *why* the middle regime is the
weak one.

---

## 7. Obligations already incurred

These were promised in the log and must be honoured, not rediscovered:

| Owed | Where | Stage |
|---|---|---|
| QLIKE ranking on **raw** Parkinson, to show it does not hinge on `c` | D10 | 6 |
| Report says proxy is *approximately unbiased on average* — **not** that proxy-robustness is restored | D10 | 7 |
| Constant-mean simplification named as a limitation (raw-return Ljung-Box rejects out of sample) | §1.7 | 7 |
| ~~`garch_mle_normal` kept out of the headline four-model tables~~ **done** at Stage 4: every builder filters on `HEADLINE_MODELS`, and a test asserts it | D15 | — |
| Near-boundary persistence (`alpha+beta` > 0.999 in 16 of 102 MLE refits) noted rather than discovered late | §1.9 | 7 |
| Report states the `delta` prior was chosen on a structural criterion, with full-sample posterior summaries consulted for magnitude | §1.10 | 7 |
| Prior sensitivity across all three `delta` candidates, on evaluation-window forecasts — **runs launched**, `--stage priors`; the analysis is what remains | §1.10 | 6 |
| Report states `target_accept` was raised to 0.95 after a smoke refit diverged | §1.12 | 7 |
| ~~`pytest --bayes-audit -m bayes_audit` run once, result recorded~~ **done**, §1.15. Owed again only if the sampler, the model graph or `build_bayes_paths` changes | §1.12 | — |
| ~~Missing Bayesian days reported as a property of the model, every comparison stating its common sample~~ **done** at D28: enforced by refusal, and every table carries `sample` and `n` | D19 | — |
| ~~Interval differences decomposed via `garch_bayes_mean`~~ **done** at D29: `eval_decomposition.csv`. The obligation now is *usage* — no report sentence may attribute the reported difference to parameter uncertainty | §1.13, D27 | 5, 7 |
| ~~The README's "any interval difference is parameter uncertainty, full stop" corrected~~ **done** before Stage 4 | §1.13 | — |
| Christoffersen's non-rejection reported as a non-rejection, with the power caveat, not as a pass | §1.16 | 7 |
| DM p-values reported as unadjusted across eight comparisons | §1.16 | 7 |
| ~~Bootstrap CIs on every per-regime coverage estimate (never-cut)~~ **done** at Stage 5: `eval_regime_coverage.csv`, and figure 12 draws them | plan §5 | — |
| The two regime definitions reported as different partitions, with no row compared across them | D35 | 7 |
| "Survives stress" stated as *this sample cannot show it failing*, given a [0.29%, 2.02%] interval | §1.17 | 7 |
| The transition explanation for the middle-regime failure presented as a conjecture the design cannot test | §1.17 | 7 |

---

## 8. Traps specific to this codebase

Ordered by how much damage they do while looking fine.

1. **Shared `h_next` across posterior draws.** Silently destroys the research question by
   collapsing the Bayesian predictive onto the plug-in. `mixture_predictive` now refuses
   any `h_next` that is not one per draw, and two tests pin both limits — strictly wider
   when the posterior is dispersed, exactly equal when it is collapsed to a point.
2. **Missing `-0.5*ln(h_t)`** in the log-likelihood. Optimises fine; every variance wrong.
3. **Dropping `sqrt((nu-2)/nu)`** in the plug-in quantile. Inflates intervals a few
   percent, in the direction that flatters the Bayesian model.
4. **Freezing `h` between refits** as well as the parameters. Not a GARCH forecast.
5. **Applying `c` twice**, or to a model already on the return scale. EWMA and GARCH are
   driven by squared returns and need no scaling; only the Parkinson-based RW does.
6. **Weakening a look-ahead test until it passes.** Three times now the correct claim was
   subtler than the obvious one — problems-and-solutions #3, #7 and #38. If a test fails,
   first establish whether the *test* is wrong before touching the code, then keep
   whichever version is actually true. #38 is the one to read: the false claim came from
   this very document, and the repair it invited would have manufactured the project's
   finding.
7. **Positive-only tests.** Every diagnostic needs a case where it must *not* fire.
8. **Trimming the multi-start grid.** Looks like free speed; with one start, 93 of the 102
   refits land on a worse local optimum (#30).
9. **Reading a fit's convergence verdict off the best objective value.** Different
   questions; entangling them cost 21 days of forecasts once already (#29).
10. **Assuming you know which knob controls a cost.** D20 traded away Monte Carlo
    precision to make the audit affordable, and the audit's cost turned out to be graph
    compilation, which draws do not touch (#39). Measure before deciding.
11. **Gating an expensive test with a marker deselection.** A command line's `-m` replaces
    an `addopts` one rather than adding to it, so the fast inner loop selected the
    forty-minute test (#40). The gate is now an explicit `--bayes-audit` flag.
12. **Quoting `garch_bayes` against `garch_mle` as parameter uncertainty.** The project
    said this in its plan, its README and an earlier version of this document, and it is
    false against the MLE plug-in that `forecasts.csv` holds (#41). Every such statement
    must come from `eval_decomposition.csv`'s first contrast.
13. **Reading a non-rejection as a pass.** Christoffersen's independence test does not
    fire for any model at 99% — on hit sequences of 37 breaches. That is a failure to
    reject, on a small sample, and reporting it as evidence of well-timed breaches would
    be the same error as reporting a wide bootstrap interval as agreement. The same trap
    is waiting in the stressed regime, where the GARCH models "pass" Kupiec on four and
    five breaches.
14. **Comparing a row across the two regime definitions.** The VIX bands and the trailing-
    volatility terciles are different partitions of the same days — the tercile "stressed"
    bucket is 1,040 days against the VIX one's 347. They agree on the conclusion and
    disagree on which non-stressed bucket is weakest; presenting either as confirming the
    other row by row would be false.
15. **Estimating a regime threshold on the full sample.** Terciles over all 2,890 days
    would define the crisis regime using the crisis, and nothing in the output would show
    it. `trailing_vol_thresholds` reads the warm-up window only and is pinned by a
    corrupt-the-future test on both sides (D35).
16. **Turning `prior_delta` into a module constant.** It is an argument defaulting to the
    frozen D4 value precisely so that a sensitivity run has to declare itself at the call
    site. Collapsing it into a constant puts a one-line edit between the frozen priors and
    every headline number.

---

## 9. Budget

The governing plan budgets ~15-20h total. Stages 0-5 account for roughly 16-17h of that;
Stages 6-7 are estimated at 5-6h, of which the prior-sensitivity compute is unattended. **The plan is over its own budget**, which is what the
cut list exists for. Cut in its stated order — the SV stretch goal is already out, then
the refit-cadence sensitivity, then the trailing-vol regime sensitivity, then MSE as a
secondary loss. Never the four never-cut items in §3.

The prior-sensitivity check owed by D4 is a Bayesian backtest per candidate prior,
roughly 95 minutes each. It was **bought rather than cut**: both runs are launched, so what
remains of that obligation is analysis rather than compute.

---

## 10. First three commands for the next session

```bash
pytest -q                                             # confirm 342 pass, 1 skips, before touching anything
python -c "import pandas as pd; print(pd.read_csv('data/processed/eval_regime_var.csv').to_string())"
sed -n '1,60p' docs/project1-implementation-plan.md   # re-read the contract
```

Then start at §5 of this document.
