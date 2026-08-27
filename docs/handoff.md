# Handoff: the finished project, and how to work on it

**Complete as of 2026-08-28. Every stage of the governing plan is done, every obligation in
the decision register is discharged, and all four never-cut items are delivered.**

Four forecasters plus two ablations, each with a forecast table over the 2,134-day
evaluation window. The Bayesian GARCH(1,1)-t is fitted by NUTS at each of 102 refit dates,
carrying its whole posterior into the predictive. The evaluation layer scores all of it —
point losses, PIT, coverage, VaR backtests, pairwise comparisons, the interval
decomposition, the regime split, the tail-allocation diagnostic and the robustness checks —
writing sixteen tables and fourteen figures from the stored forecast table alone, refitting
nothing. `report/report.md`
is written. 351 tests pass, one skipped by design.

**The reproduction claim is tested, not asserted.** A fresh `git clone` run through the
pipeline reproduces every forecast table **byte for byte**, the NUTS-sampled Bayesian track
included, in a separate process and with identical convergence diagnostics. Only wall-clock
timings differ.

**Read §4 before you quote a single number.** The strongest result is one the governing
plan did not anticipate at all, three of the others contradict what it did anticipate, and
one claim this project published had to be weakened after it was tested properly. The
numbers are all defensible; the sentences a reader will expect to attach to them are not.

This document assumes no memory of how anything got here. It is written for whoever picks
the project up next, including a future session of this work, and for the case where that
person's job is to *change* something rather than to continue it — which is what §3 and §8
are for.

Read in this order:

1. This file — what exists, what is true, and what must not be broken.
2. `docs/project1-implementation-plan.md` — the **governing plan**. It is the contract, and
   it is left as written even where the results contradicted it.
3. `research_log.md` §1 — the decision register. Locked decisions are not revisable.
4. `docs/problems-and-solutions.md` — 45 entries on what went wrong and why, including the
   three occasions the project's own claims turned out to be false.

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
python run_all.py --stage robustness # Stage 6 checks: raw proxy, cadence, priors (~30s)
python run_all.py --stage figures    # figures 08-13, from those tables (~10s)
python run_all.py --stage priors     # D4's prior sensitivity, ~190 min, opt-in
pytest -q                            # expect 351 passed, 1 skipped (~18 min)
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
`evaluate` reads `forecasts.csv` and writes sixteen `eval_*.csv` tables; `figures` reads
only those tables, so a figure and the number it draws cannot disagree.

**`robustness` is in `--all`; `priors` is not (D37).** `robustness` re-scores what
exists and re-runs only the cheap track, so it costs thirty seconds. It also asserts on
every run that the two parameter-free baselines are bit-identical at both cadences, and
stops if they are not: a refit cadence reaching a model that estimates nothing would be a
harness bug, not a robustness finding.

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
| `src/figures.py` | **Complete for the report.** House style + 14 figures, all written by `run_all.py` and none by a notebook. |
| `src/backtest.py` | **Complete.** The loop, both cadences, both tracks, the model registry, the partial/merge machinery, long-format output. |
| `src/models.py` | **Complete.** Interface, both baselines, the shared likelihood, MLE, plug-in predictive, the frozen priors, `log_prior`/`log_posterior`, the PyMC/NUTS sampler, the mixture posterior predictive. |
| `src/evaluation.py` | **Complete.** QLIKE/MSE, coverage, PIT + KS, Kupiec, Christoffersen independence and conditional coverage, DM with HLN, `common_sample`, the six result-table builders, and the three regime tables with bootstrap intervals. |
| `src/bootstrap.py` | **Complete.** Politis-Romano indices, mean CI, paired loss-differential and coverage-difference CIs. |

Notebooks are the presentation layer and define no analysis logic: `01_eda.ipynb`,
`02_results.ipynb` and `03_robustness.ipynb` read from `src/` and from the persisted
tables. They are committed
**unexecuted**, with no stored outputs — `run_all.py` is what reproduces the results, and
a notebook carrying its own outputs would be a second, unversioned source of numbers.

Artefacts in `data/processed/`: `analysis_frame.csv` (2,890 rows), `forecasts.csv`
(12,804 rows = 2,134 dates × 6 models), `refit_records.csv` (408 rows), the per-track
partials, `backtest_config.json`, and the Stage 4 tables — `eval_point_losses.csv`,
`eval_coverage.csv`, `eval_var_backtests.csv`, `eval_pit.csv`, `eval_comparisons.csv`,
`eval_decomposition.csv`, `eval_regime_losses.csv`, `eval_regime_coverage.csv`,
`eval_regime_var.csv`, the two `_trailing` sensitivity tables, and the Stage 6 pair
`eval_raw_proxy_losses.csv` and `eval_cadence_comparison.csv`.

`data/processed/cadence_63/` holds the 63-day frequentist re-run.

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

**Result 4 — no evidence that tail calibration degrades in high volatility, and the middle
band is the weak spot. State this carefully; the obvious phrasing overstates it.** 99% VaR
breach rate by lagged-VIX regime, with 95% bootstrap intervals:

| model | calm (n=757) | normal (n=1,030) | stressed (n=347) |
|---|---|---|---|
| `garch_mle` | 1.19% [0.53, 1.85] | **2.33% [1.65, 3.11]** | 1.15% [0.29, 2.02] |
| `garch_bayes` | 1.19% [0.53, 1.98] | **2.31% [1.51, 3.12]** | 1.47% [0.29, 2.93] |
| `ewma` | 1.59% [0.79, 2.38] | 2.91% [2.04, 3.88] | 3.46% [1.44, 6.05] |

Both GARCH models are indistinguishable from nominal in calm *and* in stress and clearly
too high in the middle band. The baselines degrade monotonically with volatility, which is
what one would have predicted for all four.

**But a regime that rejects against nominal where another does not is not thereby different
from it** — the subsamples are 757, 1,030 and 347 days, so rejection against a fixed rate is
partly a question of power. `eval_regime_differences.csv` (D38) tests the comparison the
claim actually needs:

| difference in 99% breach rate | GARCH-t (MLE) | supported? |
|---|---|---|
| normal − calm | +1.14pp [+0.16, +2.13] | **yes** |
| normal − stressed | +1.18pp [−0.17, +2.34] | no |
| calm − stressed | +0.04pp [−1.22, +1.14] | no |

Under the trailing-volatility definition nothing separates at all. **The supported claim is
the negative one**: no evidence that calibration degrades in stress, which is a failure to
find an effect rather than a demonstration that there is none. An earlier version of this
document, the report and the README all said "calibration survives the crisis and fails in
the quiet". That overstated it, and problems-and-solutions #46 records how it happened.

**Result 5 — the intervals are the wrong *shape*, and this is the strongest thing here.**
The governing plan did not anticipate it, because no forecaster in the locked lineup can
express skew: all four are symmetric about a constant mean.

Under a symmetric predictive the two tails should be equally populated whatever the model
gets wrong about scale. For both GARCH models they are not, at every level — `garch_mle`
runs 156/81, 77/25 and 20/2 below/above at 90/95/99%, against 106.7, 53.4 and 10.7 expected
in each tail, rejecting symmetry at p = 1.3e-6, 2.5e-7 and 1.2e-4. The standardised
residuals have skew **−0.79 at p ≈ 1e-40**.

**And the PIT mean is 0.4998**, so the misallocation cancels in aggregate and the KS test
passes at p = 0.115. A model can satisfy the usual distributional check while its loss tail
is systematically too thin. `eval_tail_asymmetry.csv` and figure 14 carry it; D39 records
it. The naive baseline is *not* asymmetric — it is simply too narrow, and wrong shape versus
wrong scale is the distinction the whole table exists to draw.

**The sensitivity does not overturn Result 4 and sharpens it.** Under terciles of trailing
21-day Parkinson volatility with cut points from the warm-up window alone (D35), the GARCH
models are closest to nominal in the *top* tercile (1.35%, Kupiec p = 0.29) and worst in
the bottom one (2.06%, p = 0.012). The two definitions disagree about which non-stressed
bucket is weakest and agree on the negative claim: **no evidence that these models degrade
in high volatility.** Under the tercile definition no pairwise regime difference is
significant at all, so the agreement is between two failures to find an effect. The two partitions are not comparable row by row — the tercile
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

## 5. What is left

**Nothing in the plan.** Every stage is complete, every register obligation is discharged,
the reproduction is verified end to end, and the suite is green at 351.

Three things are the researcher's own, and none is blocked on anything:

- **The report's prose and length.** `report/report.md` runs to roughly 2,750 words and six
  tables against the plan's two-page budget — about double. The overrun is the twelve
  disclosure obligations plus the tables; cutting to length means moving them to an
  appendix rather than dropping them. The researcher owns the wording. Recorded here so the
  deviation from the contract is on the record rather than discovered later.
- **A commit.** The working tree carries the final documentation pass.
- **Anything beyond the plan.** `docs/project1-implementation-plan.md` §5 lists what was
  cut and in what order; the SV stretch goal went first and never returned. New ideas
  belong in a `future-work.md` rather than in this repository's scope, which is what kept
  the project finishable.

---

## 6. What the write-up is obliged to do

*`report/report.md` is written and already does all of this. Kept here because any edit to
it has to keep doing it.*

**Two pages.** Never claim a model is better on a lower loss in
one period; conclusions rest on bootstrap intervals, and an interval containing zero is a
legitimate reportable finding. Keep the seven quantities distinct throughout: observed
returns, the proxy, conditional variance forecasts, predictive intervals, VaR forecasts,
parameter uncertainty, innovation uncertainty. The four §4 results are the spine of the
paper, and three of them contradict what the plan expected — say so plainly. A
pre-registered protocol that produced a surprise is the most credible thing this project
has. Result 5 is the strongest and was not anticipated at all; Result 4 answers the
question in the title in the negative — no evidence that calibration degrades under stress
— which is a failure to find an effect and must not be written as a positive finding.

The limitations section is on the never-cut list and its content already exists: one
asset, one horizon, roughly two stress episodes; a proxy that is biased for the forecast
target; wide intervals on every crisis subsample; several tests that did not reject on
samples too small to have fired; and no explanation for *why* the middle regime is the
weak one.

---

## 7. Obligations already incurred

These were promised in the log and must be honoured, not rediscovered:

*Everything marked Stage 7 is discharged in `report/report.md` as written; the row stays
so that an edit cannot silently drop it.*

| Owed | Where | Stage |
|---|---|---|
| ~~QLIKE ranking on **raw** Parkinson, to show it does not hinge on `c`~~ **done**: the ranking is unchanged | D10 | — |
| Report says proxy is *approximately unbiased on average* — **not** that proxy-robustness is restored | D10 | 7 |
| Constant-mean simplification named as a limitation (raw-return Ljung-Box rejects out of sample) | §1.7 | 7 |
| ~~`garch_mle_normal` kept out of the headline four-model tables~~ **done** at Stage 4: every builder filters on `HEADLINE_MODELS`, and a test asserts it | D15 | — |
| Near-boundary persistence (`alpha+beta` > 0.999 in 16 of 102 MLE refits) noted rather than discovered late | §1.9 | 7 |
| Report states the `delta` prior was chosen on a structural criterion, with full-sample posterior summaries consulted for magnitude | §1.10 | 7 |
| ~~Prior sensitivity across all three `delta` candidates~~ **done**: parameter uncertainty identical under all three, no calibration verdict moves | §1.10 | — |
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

The governing plan budgeted ~15-20h. The work came in at roughly 21-22h of attended time,
over by about a quarter, plus about five hours of unattended sampling — the Bayesian
backtest, two prior-sensitivity backtests, and the clean-clone verification.

The overrun is concentrated in Stages 3-5 and it is traceable: the decomposition track
(`garch_bayes_mean`) did not exist in the original estimate and was added because the
comparison the project rests on turned out to have two causes; the regime analysis grew a
sensitivity the plan had listed as cuttable; and the evaluation layer acquired a common-
sample discipline the estimate had not anticipated. None of it was scope creep in the usual
sense — each addition was forced by something the previous stage had found. **The plan is over its own budget**, which is what the
cut list exists for. Cut in its stated order — the SV stretch goal is already out, then
the refit-cadence sensitivity, then the trailing-vol regime sensitivity, then MSE as a
secondary loss. Never the four never-cut items in §3.

The prior-sensitivity check owed by D4 is a Bayesian backtest per candidate prior,
roughly 95 minutes each. It was **bought rather than cut**: both runs are launched, so what
remains of that obligation is analysis rather than compute.

---

## 10. First commands for the next session

Nothing here is a to-do list; it is how to confirm the repository is in the state this
document describes before changing anything.

```bash
pytest -q                                             # 351 pass, 1 skips (~18 min)
pytest -m "not slow" -q                               # the inner loop (~1 min)
python run_all.py --stage evaluate                    # rebuild every table from forecasts.csv
python run_all.py --stage figures                     # rebuild all 14 figures from those tables
```

`evaluate` and `figures` are deterministic and take under a minute between them: if either
produces a table or figure that differs from what is committed, something upstream has
changed and §4's numbers are no longer the ones in the files.

Then read §4 before quoting anything, and §3 and §8 before changing anything.
