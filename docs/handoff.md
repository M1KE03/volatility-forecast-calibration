# Handoff: how to continue

**State as of 2026-08-26. Stage 3 complete.** All four forecasters exist, plus the
GARCH-normal ablation, each with a forecast table over the 2,134-day evaluation window.
The Bayesian GARCH(1,1)-t is fitted by NUTS at each of the 102 refit dates and carries its
whole posterior into the predictive. 100 of the 102 refits converged; the two that did not
leave 42 evaluation days without a Bayesian forecast, which is a fact about the model
rather than a gap to be filled. 227 tests pass, plus one skipped by default (§1 explains which). `evaluation.py` and `bootstrap.py` are
still entirely stubs, and they are the next thing.

**Read §4 before you read a single number out of `forecasts.csv`.** The Bayesian
intervals came out narrower than the frequentist ones, and the reason is not the one the
project has been expecting.

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
pytest -q                            # expect 227 passed, 1 skipped (~13 min)
pytest -m "not slow" -q              # inner loop (~2 min)
pytest --bayes-audit -m bayes_audit   # the audit with NUTS itself (~40 min), owed once
```

**The backtest is two stages, and that is a cost decision, not a design one.** The
frequentist track is 306 optimiser fits and takes a minute; the Bayesian track is 102
NUTS fits and takes about ninety-five. Each writes its own partial tables
(`forecasts_<track>.csv`) and rebuilds the merged `forecasts.csv` from whichever partials
are on disk, so re-running the cheap one never re-runs the expensive one and the
evaluation layer still reads one table. The merge compares the two configs and refuses to
join runs made under different ones (D24).

**One prerequisite `pip` does not supply: a C/C++ compiler on `PATH`.** PyMC's PyTensor
backend needs one. This machine uses MinGW-w64 GCC 16.2.0 (UCRT, x86_64) at
`C:\Users\micha\toolchains\mingw64`, already on the user `PATH`. Verify:

```bash
g++ --version                                       # expect 16.2.0
python -c "import pytensor; print(pytensor.config.cxx)"
```

If that second command prints an empty string on a new machine, install a compiler before
touching the Bayesian track. Provenance and the verified SHA-256 are in `research_log.md`
§1.6.

---

## 2. What exists, precisely

| Module | State |
|---|---|
| `src/data.py` | **Complete.** Download, SHA-256 manifest, returns, Parkinson proxy, VIX regimes, `PROXY_SCALE_C`. |
| `src/eda.py` | **Complete.** ARCH-LM, Ljung-Box, ADF, ACF (`series_acf` and `squared_return_acf`). |
| `src/figures.py` | **Partial.** House style + 7 figures (Stages 0-2). More added per stage. |
| `src/backtest.py` | **Complete.** The loop, both cadences, both tracks, the model registry, the partial/merge machinery, long-format output. |
| `src/models.py` | **Complete.** Interface, both baselines, the shared likelihood, MLE, plug-in predictive, the frozen priors, `log_prior`/`log_posterior`, the PyMC/NUTS sampler, the mixture posterior predictive. |
| `src/evaluation.py` | **All stubs** (10 functions). **Next.** |
| `src/bootstrap.py` | **All stubs** (4 functions). **Next.** |

Artefacts in `data/processed/`: `analysis_frame.csv` (2,890 rows), `forecasts.csv`
(10,670 rows = 2,134 dates × 5 models), `refit_records.csv`
(408 = 102 refits × 4 model tracks, carrying parameter estimates and
convergence verdicts for both estimators), the per-track partials, and
`backtest_config.json`.

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

**The never-cut list**, from the governing plan §5: the look-ahead audit, the
Christoffersen test, bootstrap CIs on regime coverage, and the limitations section. If the
budget runs out, cut from §5's ordered list — not from these.

**Append-only log.** `research_log.md` is never edited to match a later result. New
decisions get new numbered subsections; §1.12 supersedes parts of §1.10 exactly that way.

---

## 4. Stage 3 — Bayesian GARCH(1,1)-t — **done**

*The full record is `research_log.md` §1.10-1.13 and the Stage 3 changelog entry. What
follows is only what Stage 4 needs to know.*

**The run.** 102 refits at 4 chains x (1,000 tune + 1,000 draw), `target_accept` 0.95,
82.5 minutes of sampling. **100 of 102 converged**; worst R-hat 1.0049, minimum
`ess_bulk` 1,475, minimum `ess_tail` 972, four divergent transitions in total. The two
failures (2025-02-10 and 2025-03-12) leave 42 of the 2,134 evaluation days without a
Bayesian forecast, NaN in the table under D19 and not re-run.

**The headline result is not the expected one, and §1.13 is not optional reading.**
Bayesian intervals came out *narrower* than the frequentist plug-in at every level --
0.9945 at 90%, 0.9915 at 95%, 0.9842 at 99% -- and most in stress: 0.9709 at 99% on
stressed days against 0.9962 on calm ones. That is the opposite of what this project set
out to find, and the first instinct is that the mixture is not carrying parameter
uncertainty. It is. Decomposed on a COVID-period refit at the 99% level:

| | ratio |
|---|---|
| posterior predictive over plug-in **at the posterior mean** -- parameter uncertainty | **1.0046** |
| plug-in at the posterior mean over plug-in **at the MLE** -- the priors | **0.9550** |

Parameter uncertainty widens, as designed and as tested. It is simply small at these
sample sizes, and it is outweighed four to five times over by the priors moving the point
estimate.

Five things carry into Stage 4.

1. **The two models genuinely share one likelihood, and this is now verified rather
   than asserted.** `test_the_pymc_graph_and_the_numpy_log_posterior_agree` evaluates the
   PyMC graph and `garch11_t_loglik` at the same parameter vector and requires they agree
   to floating point — up to `log(1 - alpha)`, the deliberate difference between a prior
   stated over `delta` (PyMC samples it) and one stated over `beta` (`log_prior` states
   it, with the change-of-variable Jacobian route B would need). What that buys is
   narrower than the sentence this project has been repeating: the two models share a
   likelihood, so they differ in the *estimator*, not in the model. It does **not** follow
   that their interval difference is parameter uncertainty -- see point 3.

2. **The Bayesian predictive is not wider at every level, and the previous version of
   this document was wrong to say it should be.** It said intervals should be ≥
   frequentist at every level and to hunt for the shared-`h_next` bug otherwise. A scale
   mixture holding average variance fixed is leptokurtic against the single distribution
   at that average — more peaked in the middle, heavier in the tails — so the ratio
   crosses 1 somewhere between 90% and 95%. Measured on a dispersed posterior: 0.995 at
   90%, 1.002 at 95%, 1.017 at 99%, 1.032 at 99.8%. Verified against four million draws
   from the same mixture before the claim rather than the code was changed. See
   problems-and-solutions #38, and the two tests that pin either side of the crossover.

3. **The one-cause comparison is not currently one cause, and this is the single most
   important thing on this page.** The claim in the README, in the plan's interview
   ammunition, and until now in this document -- *models 3 and 4 share a likelihood, so
   any interval difference is parameter uncertainty, full stop* -- is true of the
   posterior predictive against a plug-in **at the posterior mean**. It is false of the
   posterior predictive against a plug-in **at the MLE**, which is what `forecasts.csv`
   holds, because those are two different point estimates and the priors move one of them.

   Two priors move it, both frozen at D4 long before any of this was visible, and both
   pushing the same way. `beta = (1 - alpha) * delta` with `delta ~ Beta(3, 1)` keeps
   `alpha + beta < 1` by construction and so pulls persistence off the boundary the MLE
   runs into -- mean 0.9786 against 0.9893, and the MLE reaches 0.99998 in 16 of the 102
   refits while the posterior never passes 0.991. Lower persistence means less carry-over
   after a shock, which is why the gap is widest in the stressed regime. And the `nu`
   prior has mean 14, leaning near-normal, so posterior `nu` sits above the MLE (5.51
   against 5.20) and thins the Bayesian tails exactly at 99%.

   **So do not attribute the frequentist-Bayesian interval difference to parameter
   uncertainty.** Decompose it. The cheap way is a fourth GARCH track -- plug-in at the
   posterior mean -- which needs no sampling at all, since the posterior means for all 102
   refits are already in `refit_records.csv` and the track is `garch11_filter` plus
   `plugin_predictive`. It would be an ablation in the sense `garch_mle_normal` is, kept
   out of the headline table. That is a recommendation, not a decision; §1.13 records it
   as open.

4. **A failed refit means missing days, and the evaluation layer must handle them.**
   Under D19 a Bayesian refit that fails its diagnostics produces no forecasts for the 21
   days it serves; those rows are NaN in `forecasts.csv`, not absent. Here that is 42 days: two refits, 2025-02-10 and 2025-03-12.
   Coverage and loss must be computed on the dates a model actually forecast, and every
   pairwise comparison — the DM tests especially — must state the common sample it used.

5. **`target_accept` is 0.95, not D17's 0.9** (D25), raised before the production run
   after a smoke refit produced four divergences and therefore no forecasts. The report
   must say so. It is a sampler effort parameter rather than a diagnostic threshold, and
   the distinction is argued in §1.12; do not treat it as licence to tune other frozen
   settings after seeing a result.

---

## 5. Stage 4 — evaluation layer (3-4h). **The intellectual core. Start here.**

QLIKE and MSE against the scaled proxy; DM tests with the Harvey-Leybourne-Newbold
correction; PIT histograms; coverage tables at 90/95/99 and 99% VaR; Kupiec POF and
**Christoffersen** independence and conditional coverage on the 99% VaR hit sequences.

Everything reads `data/processed/forecasts.csv` and nothing re-runs the backtest.

**Before any of it, settle §4 point 3.** The interval comparison the whole write-up turns
on is currently confounded between the priors and parameter uncertainty, and every
coverage table Stage 4 produces will inherit that confound. Deciding what to do about it
after the tables exist is how a result gets chosen rather than found.

Expect the four models to be **hard to separate on point accuracy**. That is the setup for
the calibration act, not a failure. Christoffersen is the money test: correct *average*
coverage can hide breaches that cluster in crises, and clustering is the failure mode that
actually ruins risk models.

Four caveats belong in the report body, not a footnote:

- DM's asymptotics assume forecasts are not functions of estimated parameters. Here they
  are.
- Models 3 and 4 share a likelihood, so their loss differential may be near-degenerate.
  Return the differential variance so a reader can check rather than trust the p-value.
- `garch_mle_normal` is an ablation and must stay out of the headline tables. Filter on
  `backtest.HEADLINE_MODELS`, not on whatever happens to be in the forecast file (D15).
- KS-uniformity p-values on the PIT are approximate because parameters are estimated.
  Knowing that caveat is worth more than the test.

---

## 6. Stages 5-7, in brief

**Stage 5 — regime analysis (2-2.5h).** Split on the lagged-VIX label. **Report `n`
alongside every statistic** — stressed days are 375 of 2,890, and a regime table without
sample sizes invites over-reading. Bootstrap CIs on every per-regime coverage estimate are
mandatory; their width is itself on-theme. Likely headline figure: coverage vs nominal by
regime, one panel per model.

**Stage 6 — robustness (1.5-2h).** GARCH-normal vs GARCH-t at 99% (expect normal to fail
hard). Refit cadence at 63 days. **QLIKE ranking on raw, unscaled Parkinson** — owed by
D10. **Prior sensitivity across `Beta(10,2)`, `Beta(3,1)` and `Beta(1,1)` on `delta`** —
owed by D4, and now cheap to state though not to compute: it is a second and third
Bayesian backtest, so budget for it or cut it explicitly rather than by omission.

**Stage 7 — write-up (3-4h).** Two pages. Never claim a model is better on a lower loss in
one period; conclusions rest on bootstrap intervals, and an interval containing zero is a
legitimate reportable finding. Keep the seven quantities distinct throughout: observed
returns, the proxy, conditional variance forecasts, predictive intervals, VaR forecasts,
parameter uncertainty, innovation uncertainty.

---

## 7. Obligations already incurred

These were promised in the log and must be honoured, not rediscovered:

| Owed | Where | Stage |
|---|---|---|
| QLIKE ranking on **raw** Parkinson, to show it does not hinge on `c` | D10 | 6 |
| Report says proxy is *approximately unbiased on average* — **not** that proxy-robustness is restored | D10 | 7 |
| Constant-mean simplification named as a limitation (raw-return Ljung-Box rejects out of sample) | §1.7 | 7 |
| `garch_mle_normal` kept out of the headline four-model tables (filter on `HEADLINE_MODELS`) | D15 | 4, 5 |
| Near-boundary persistence (`alpha+beta` > 0.999 in 16 of 102 MLE refits) noted rather than discovered late | §1.9 | 7 |
| Report states the `delta` prior was chosen on a structural criterion, with full-sample posterior summaries consulted for magnitude | §1.10 | 7 |
| Prior sensitivity across all three `delta` candidates, on evaluation-window forecasts | §1.10 | 6 |
| Report states `target_accept` was raised to 0.95 after a smoke refit diverged | §1.12 | 7 |
| `pytest --bayes-audit -m bayes_audit` run once, with its result recorded in the log | §1.12 | before 7 |
| Missing Bayesian days reported as a property of the model, and every comparison stating its common sample | D19 | 4 |
| Frequentist-Bayesian interval differences **decomposed**, never attributed to parameter uncertainty on their own | §1.13 | 4, 7 |
| The README's "any interval difference is parameter uncertainty, full stop" corrected wherever it appears | §1.13 | 4 |

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

---

## 9. Budget

The governing plan budgets ~15-20h total. Stages 0-3 account for roughly 11-12h of that;
Stages 4-7 are estimated at 10-13h. **The plan is over its own budget**, which is what the
cut list exists for. Cut in its stated order — the SV stretch goal is already out, then
the refit-cadence sensitivity, then the trailing-vol regime sensitivity, then MSE as a
secondary loss. Never the four never-cut items in §3.

Note that the prior-sensitivity check owed by D4 is now a *Bayesian backtest per
candidate prior*, roughly 95 minutes each. If it goes, it goes explicitly and into the
limitations section, not by omission.

---

## 10. First three commands for the next session

```bash
pytest -q                                             # confirm 227 pass, 1 skips, before touching anything
python -c "import pandas as pd; f=pd.read_csv('data/processed/forecasts.csv'); print(f.groupby('model')['variance'].agg(['count','mean']))"
sed -n '1,60p' docs/project1-implementation-plan.md   # re-read the contract
```

Then start at §5 of this document.
