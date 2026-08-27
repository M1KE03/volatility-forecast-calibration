# Handoff: how to continue

**State as of 2026-08-27. Stage 4 complete.** All four forecasters exist, plus two
ablations, each with a forecast table over the 2,134-day evaluation window. The
evaluation layer scores all of it and writes six tables and four figures from
`forecasts.csv` alone, refitting nothing. `evaluation.py` and `bootstrap.py` are complete.
**Stage 5, the regime-conditional analysis, is next**, and most of its machinery already
exists.

**Read §4 before you quote a single number.** The headline results are not the ones the
governing plan anticipated — in two different places, and in opposite directions.

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
python run_all.py --stage evaluate   # every table, from forecasts.csv (~15s)
python run_all.py --stage figures    # figures 08-11, from those tables (~5s)
pytest -q                            # expect 321 passed, 1 skipped (~13 min)
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
`evaluate` reads `forecasts.csv` and writes six `eval_*.csv` tables; `figures` reads only
those tables, so a figure and the number it draws cannot disagree.

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
| `src/data.py` | **Complete.** Download, SHA-256 manifest, returns, Parkinson proxy, VIX regimes, `PROXY_SCALE_C`. |
| `src/eda.py` | **Complete.** ARCH-LM, Ljung-Box, ADF, ACF (`series_acf` and `squared_return_acf`). |
| `src/figures.py` | **Partial.** House style + 11 figures (Stages 0-4). More added per stage. |
| `src/backtest.py` | **Complete.** The loop, both cadences, both tracks, the model registry, the partial/merge machinery, long-format output. |
| `src/models.py` | **Complete.** Interface, both baselines, the shared likelihood, MLE, plug-in predictive, the frozen priors, `log_prior`/`log_posterior`, the PyMC/NUTS sampler, the mixture posterior predictive. |
| `src/evaluation.py` | **Complete.** QLIKE/MSE, coverage, PIT + KS, Kupiec, Christoffersen independence and conditional coverage, DM with HLN, regime summaries, `common_sample`, and the six result-table builders. |
| `src/bootstrap.py` | **Complete.** Politis-Romano indices, mean CI, paired loss-differential and coverage-difference CIs. |

Notebooks are the presentation layer and define no analysis logic: `01_eda.ipynb` and
`02_results.ipynb` read from `src/` and from the persisted tables. They are committed
**unexecuted**, with no stored outputs — `run_all.py` is what reproduces the results, and
a notebook carrying its own outputs would be a second, unversioned source of numbers.

Artefacts in `data/processed/`: `analysis_frame.csv` (2,890 rows), `forecasts.csv`
(12,804 rows = 2,134 dates × 6 models), `refit_records.csv` (408 rows), the per-track
partials, `backtest_config.json`, and the Stage 4 tables — `eval_point_losses.csv`,
`eval_coverage.csv`, `eval_var_backtests.csv`, `eval_pit.csv`, `eval_comparisons.csv`,
`eval_decomposition.csv`.

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
budget runs out, cut from §5's ordered list — not from these. Two of the four are now
done; bootstrap CIs on regime coverage are Stage 5's, and the machinery is built.

**Append-only log.** `research_log.md` is never edited to match a later result. New
decisions get new numbered subsections; §1.12 supersedes parts of §1.10 exactly that way.

---

## 4. Stage 4 — the evaluation layer — **done**, and two results are not what was expected

*The full record is `research_log.md` §1.16 and the Stage 4 changelog entry. What follows
is what Stage 5 and the write-up need to know.*

**The layer reproduces §1.13 exactly** from the stored table rather than from the run that
produced it — 0.9975 / 0.9989 / 1.0032 for parameter uncertainty at 90 / 95 / 99, and
0.9970 / 0.9926 / 0.9811 for the priors. Treat that as the evaluation code's own
regression check: if those six numbers ever move without `forecasts.csv` moving, something
in `evaluation.py` has broken.

**Result 1 — the models separate on point accuracy, and the pair that matters does not.**
The plan predicted the four would be hard to separate, with that difficulty as the setup
for the calibration act. On the 2,092-day common sample, mean QLIKE is `garch_bayes`
0.4577, `garch_mle` 0.4599, `ewma` 0.5239, `yesterday` 0.7809, and both GARCH models beat
both baselines with bootstrap intervals nowhere near zero. The difficulty is *inside* the
GARCH pair: `garch_mle` against `garch_bayes` is 0.0022, half a percent of the loss level,
whose bootstrap interval `[+0.00019, +0.00418]` clears zero by a hair — and
`garch_bayes_mean` against `garch_mle` does not clear it at all. **The report should say
the separation is between GARCH and no GARCH, not between estimators.**

**Result 2 — the 99% VaR failure is in the level, not the timing, and the plan expected
the reverse.** All four models fail Kupiec: 87 breaches for `yesterday`, 54 for `ewma`, 37
for each GARCH model, against 21 expected; the least strongly rejected of the four,
`garch_mle`, still at p = 0.002. **Christoffersen independence fires for none of
them** — p = 0.67 and 0.69 for the GARCH models, 0.76 for `yesterday`, and 0.058 for
`ewma`, the only one close. Breaches are too many, but they are not clustered.

The plan called Christoffersen the money test on the argument that correct *average*
coverage can hide clustered failures. Here the average coverage is wrong and the
clustering is absent — the mirror image. That is a finding, not a null result, and it is
the most interesting thing in the stage: a test that does not fire is evidence only if it
had the power to. **Say which of the two failure modes each model exhibits, and do not
recycle the plan's sentence about Christoffersen as though it had been borne out.**

**Result 3 — parameter uncertainty changes no coverage number at all.** `garch_bayes` and
`garch_bayes_mean` have identical coverage at every level and identical 99% breach counts,
to the last digit, despite different interval widths. Parameter uncertainty moves the 99%
width by 0.32%; nothing in 2,092 days of returns lands in the gap. That is the cleanest
statement of the project's central measurement and it belongs in the write-up: **at
n ≥ 750, integrating over parameter uncertainty is not what determines whether a risk
model's intervals are calibrated.** The governing plan's risk register called this
outcome and it has arrived.

**Coverage and PIT, for reference.** `garch_mle` runs 0.8889 / 0.9522 / 0.9897 against
nominal 0.90 / 0.95 / 0.99, `garch_bayes` 0.8886 / 0.9517 / 0.9890 — under-covering at 90
and 99, indistinguishable from each other everywhere. Both GARCH-t models pass the KS test
on the PIT (p = 0.115 and 0.175); `ewma` fails at 3e-14, `yesterday` at 1e-7, and
`garch_mle_normal` at 2.6e-4, which is Stage 6's innovation ablation arriving early and
pointing the way it was expected to.

**Four things about how to read the tables.**

1. **Every table names its sample and carries its `n` (D28).** Per-model rows use each
   model's own days, so a four-model table mixes n = 2,134 and n = 2,092; every pairwise
   comparison uses the *pairwise* intersection; and the headline tables are additionally
   recomputed on the four-model common sample under the label `common sample`. Nothing is
   inferable — read the column.
2. **A missing forecast is an error, not a dropped row.** Every function in
   `evaluation.py` and `bootstrap.py` raises on a non-finite input. If you get that error,
   the fix is to select a sample with `evaluation.common_sample` and report its `n`, never
   to filter the NaN away at the call site.
3. **The ablations stay out of the headline tables**, filtered on
   `backtest.HEADLINE_MODELS` (D15 for `garch_mle_normal`, D29 for `garch_bayes_mean`).
   They are scored — they are in the `own days` rows of every table — but they are not
   competitors.
4. **Every p-value in `eval_comparisons.csv` is unadjusted** across eight pairwise
   comparisons, and DM's asymptotics assume forecasts are not functions of estimated
   parameters, which here they are. The bootstrap intervals carry the conclusions.

---

## 5. Stage 5 — regime-conditional analysis (2–2.5h). **Start here.**

Split every Stage 4 statistic on the lagged-VIX label. Most of the machinery exists:
`evaluation.summarise_by_regime` works on any loss or binary indicator column and returns
`n` beside every statistic, `score_forecasts` already attaches `inside_<level>` and
`exceedance` columns, and `bootstrap_coverage_difference` is built and tested.

What Stage 5 owes:

- **Per-regime QLIKE ranking, coverage at each level, and the 99% breach rate**, per
  model, with `n` on every row.
- **Stationary-block-bootstrap CIs on every per-regime coverage estimate.** This is on the
  never-cut list. Their width is itself on-theme: the stressed regime is 341 of the 2,092
  days the Bayesian model forecasts, and an error bar that swallows the effect is the
  honest answer.
- **The likely headline figure:** coverage against nominal by regime, one panel per model
  — the plot that shows whether 99% means 99% when VIX > 25.
- **An explicit no-look-ahead statement for the regime definition.** The label is built
  from the *lagged* VIX close, so the conditioning information was available when the
  forecast was made; say so where the figure is introduced.
- Sensitivity on trailing-realised-vol terciles, relegated to the robustness notebook and
  third on the cut list.

Two cautions specific to this stage. First, `eval_decomposition.csv` is **already**
regime-split, and its stressed-regime rows are the ones the write-up will want — the
priors narrow the 99% interval by 3.2% on stressed days while parameter uncertainty widens
it by 0.27%, flat across regimes. Do not recompute that by hand. Second, the per-regime
subsamples are small and every test on them is underpowered; a Christoffersen test that
does not fire on 341 days is close to uninformative, and the report should say so rather
than report it as a pass.

---

## 6. Stages 6-7, in brief

**Stage 6 — robustness (1.5–2h).** GARCH-normal vs GARCH-t at 99% (already visible in the
Stage 4 tables: it fails the PIT at 2.6e-4 and takes 53 breaches against 37 — write it up
properly rather than re-deriving it). Refit cadence at 63 days. **QLIKE ranking on raw,
unscaled Parkinson** — owed by D10. **Prior sensitivity across `Beta(10,2)`, `Beta(3,1)`
and `Beta(1,1)` on `delta`** — owed by D4, and cheap to state though not to compute: it is
a second and third Bayesian backtest, so budget for it or cut it explicitly rather than by
omission.

**Stage 7 — write-up (3–4h).** Two pages. Never claim a model is better on a lower loss in
one period; conclusions rest on bootstrap intervals, and an interval containing zero is a
legitimate reportable finding. Keep the seven quantities distinct throughout: observed
returns, the proxy, conditional variance forecasts, predictive intervals, VaR forecasts,
parameter uncertainty, innovation uncertainty. The three §4 results above are the spine of
the paper, and two of them contradict what the plan expected — say so plainly; a
pre-registered protocol that produced a surprise is the most credible thing this project
has.

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
| Prior sensitivity across all three `delta` candidates, on evaluation-window forecasts | §1.10 | 6 |
| Report states `target_accept` was raised to 0.95 after a smoke refit diverged | §1.12 | 7 |
| ~~`pytest --bayes-audit -m bayes_audit` run once, result recorded~~ **done**, §1.15. Owed again only if the sampler, the model graph or `build_bayes_paths` changes | §1.12 | — |
| ~~Missing Bayesian days reported as a property of the model, every comparison stating its common sample~~ **done** at D28: enforced by refusal, and every table carries `sample` and `n` | D19 | — |
| ~~Interval differences decomposed via `garch_bayes_mean`~~ **done** at D29: `eval_decomposition.csv`. The obligation now is *usage* — no report sentence may attribute the reported difference to parameter uncertainty | §1.13, D27 | 5, 7 |
| ~~The README's "any interval difference is parameter uncertainty, full stop" corrected~~ **done** before Stage 4 | §1.13 | — |
| Christoffersen's non-rejection reported as a non-rejection, with the power caveat, not as a pass | §1.16 | 5, 7 |
| DM p-values reported as unadjusted across eight comparisons | §1.16 | 7 |
| Bootstrap CIs on every per-regime coverage estimate (never-cut) | plan §5 | 5 |

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
    be the same error as reporting a wide bootstrap interval as agreement.

---

## 9. Budget

The governing plan budgets ~15-20h total. Stages 0-4 account for roughly 14-15h of that;
Stages 5-7 are estimated at 7-9h. **The plan is over its own budget**, which is what the
cut list exists for. Cut in its stated order — the SV stretch goal is already out, then
the refit-cadence sensitivity, then the trailing-vol regime sensitivity, then MSE as a
secondary loss. Never the four never-cut items in §3.

Note that the prior-sensitivity check owed by D4 is a *Bayesian backtest per candidate
prior*, roughly 95 minutes each. If it goes, it goes explicitly and into the limitations
section, not by omission.

---

## 10. First three commands for the next session

```bash
pytest -q                                             # confirm 321 pass, 1 skips, before touching anything
python -c "import pandas as pd; print(pd.read_csv('data/processed/eval_var_backtests.csv').to_string())"
sed -n '1,60p' docs/project1-implementation-plan.md   # re-read the contract
```

Then start at §5 of this document.
