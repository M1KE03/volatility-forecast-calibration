# Handoff: how to continue

**State as of 2026-08-23.** Stage 2 complete. Both baselines and the frequentist
GARCH(1,1)-t forecast over the full evaluation window, plus the GARCH-normal ablation.
The Bayesian model does not exist. 188 tests pass.

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
pytest -q                            # expect 188 passed (~10 min)
pytest -m "not slow" -q              # inner loop, one backtest run (~1 min)
```

`--stage backtest` refits GARCH 204 times, which is where its minute goes. The `slow`
marker covers the 15 tests that each need their *own* backtest run -- the look-ahead
audit's corrupted re-runs, and the determinism check. The contract tests share one
module-scoped run, so `-m "not slow"` still pays for that one and takes about a minute.
Plain `pytest` runs everything: the audit is on the never-cut list, and a default test
run must not be the thing that skips it.

**One prerequisite `pip` does not supply: a C/C++ compiler on `PATH`.** PyMC's PyTensor
backend needs one. This machine uses MinGW-w64 GCC 16.2.0 (UCRT, x86_64) at
`C:\Users\micha\toolchains\mingw64`, already on the user `PATH`. Verify:

```bash
g++ --version                                       # expect 16.2.0
python -c "import pytensor; print(pytensor.config.cxx)"
```

If that second command prints an empty string on a new machine, install a compiler before
attempting Stage 3. Provenance and the verified SHA-256 are in `research_log.md` §1.6.

---

## 2. What exists, precisely

| Module | State |
|---|---|
| `src/data.py` | **Complete.** Download, SHA-256 manifest, returns, Parkinson proxy, VIX regimes, `PROXY_SCALE_C`. |
| `src/eda.py` | **Complete.** ARCH-LM, Ljung-Box, ADF, ACF (`series_acf` and `squared_return_acf`). |
| `src/figures.py` | **Partial.** House style + 7 figures (Stages 0-2). More added per stage. |
| `src/backtest.py` | **Complete for the frequentist models.** The loop, both cadences, model registry, long-format output, persistence. |
| `src/models.py` | **Frequentist half complete.** Interface, both baselines, the shared likelihood, MLE, plug-in predictive. **`log_prior`, `log_posterior`, `sample_garch_posterior` and `posterior_predictive` are stubs.** |
| `src/evaluation.py` | **All stubs** (10 functions). |
| `src/bootstrap.py` | **All stubs** (4 functions). |

Artefacts in `data/processed/`: `analysis_frame.csv` (2,890 rows), `forecasts.csv`
(8,536 rows = 2,134 dates × 4 models), `refit_records.csv` (306 = 102 refits × 3 model
tracks, carrying the parameter estimates and convergence verdicts),
`backtest_config.json`.

Tests, as collected: `test_models.py` 68, `test_data.py` 39, `test_backtest.py` 32,
`test_eda.py` 26, `test_smoke.py` 23 -- 188 in total.

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

**Priors must be frozen before any Bayesian number exists** (see §5). Loose priors widen
the Bayesian predictive intervals and could manufacture the headline finding.

**The never-cut list**, from the governing plan §5: the look-ahead audit, the
Christoffersen test, bootstrap CIs on regime coverage, and the limitations section. If the
budget runs out, cut from §5's ordered list — not from these.

**Append-only log.** `research_log.md` is never edited to match a later result. New
decisions get new numbered subsections.

---

## 4. Stage 2 — frequentist GARCH(1,1)-t — **done**

*Kept short. The full record is `research_log.md` §1.9 and its Stage 2 changelog entry;
what follows is only what Stage 3 needs to know.*

`models.py` now holds the shared likelihood (`garch11_filter`, `garch11_t_loglik`), the
MLE (`fit_garch_mle`), and the plug-in predictive (`StudentTPredictive`,
`plugin_predictive`). `backtest.build_garch_paths` runs the two cadences: refit at each
of the 102 refit dates, filter daily in between. All 102 refits converge for both the t
and the normal variants; the look-ahead audit passes with them in the loop.

Four things carry into Stage 3:

1. **The likelihood is validated.** Warm-up MLE gives alpha 0.2150, beta 0.7353, nu 5.86
   against the PyMC probe's 0.219 / 0.713 / 6.4 on the same window. Two implementations
   sharing no code. If your Stage 3 posterior sits somewhere else entirely, the sampler
   or the model graph is wrong, not the data.
2. **Returns are fitted on the percent scale internally** (D14), converted back before
   they leave `fit_garch_mle`. The probe did the same for sampler geometry. Keep the
   convention, and remember to convert variances back before they enter the forecast
   table.
3. **The frequentist interval widths are now on record.** They are what the Bayesian
   intervals get compared against, and the sanity check in §5 depends on them: Bayesian
   must come out **wider**.
4. **Persistence is near the boundary.** `alpha + beta` exceeds 0.999 in 16 of the 102
   refits, peaking at 0.99998. The probe's `beta = (1 - alpha) * delta` construction
   enforces stationarity by construction, so the posterior will press against the same
   edge. A prior that pushes back hard on it is making a modelling choice, and it should
   be a visible one.

**The next thing to do is §5.**

---

## 5. Stage 3 — Bayesian GARCH(1,1)-t

*Governing plan estimate: 3.5-4.5h. The one stage with real risk.*

### Freeze the priors first

Nothing here may produce an out-of-sample number until the priors are written into
`research_log.md`. This is not bureaucracy: the project's headline claim is that
integrating over parameter uncertainty improves interval coverage. Priors chosen after
seeing a coverage table would make that claim unfalsifiable.

### The de-risking is already done

A feasibility probe fitted this model to the real warm-up window before Stage 3 was
scheduled. Results: **R-hat 1.00 on every parameter, zero divergences, ESS 485-809, ~10s
sampling** for 2 chains × (500 tune + 500 draw). Estimates `mu` 0.071, `omega` 0.059,
`alpha` 0.219, `beta` 0.713, `nu` 6.4 — with `alpha + beta = 0.93` and `nu = 6.4` being
the persistence and tail thickness one expects from daily SPY, which sanity-checks the
likelihood as much as the sampler.

The working specification, as a starting point for the frozen priors:

```python
with pm.Model() as model:
    mu    = pm.Normal("mu", 0.0, 1.0)
    omega = pm.HalfNormal("omega", 1.0)
    alpha = pm.Beta("alpha", 2.0, 10.0)
    delta = pm.Beta("delta", 10.0, 2.0)
    beta  = pm.Deterministic("beta", (1.0 - alpha) * delta)   # alpha+beta<1 by construction
    nu    = pm.Truncated("nu", pm.Exponential.dist(1 / 10.0), lower=4.0)

    h = garch_h(omega, alpha, beta, r, mu, h0)     # pytensor.scan recursion
    sigma = pt.sqrt(h) * pt.sqrt((nu - 2.0) / nu)
    pm.StudentT("obs", nu=nu, mu=mu, sigma=sigma, observed=r[1:])
```

Two details from the probe worth keeping:

- **Returns were scaled to percent** (`r * 100`). It improves sampler geometry
  materially. If you keep this, remember to convert variances back before they enter the
  forecast table, which is on the raw return scale.
- `beta = (1 - alpha) * delta` with `delta ~ Beta(10, 2)` enforces `alpha + beta < 1`
  **by construction**, so stationarity never has to be rejected by the sampler.

### The single most dangerous bug in the codebase

Each posterior draw implies its **own** filtered variance path and therefore its own
`h_next`. `posterior_predictive` must take `h_next_by_draw` of shape `(n_draws,)`.

Passing one shared `h_next` — for instance one computed at the posterior mean — collapses
the posterior predictive to something very close to the plug-in predictive. It would
**destroy the project's research question while producing entirely plausible-looking
output.** Nothing would fail; the Bayesian and frequentist intervals would simply agree,
and that agreement would be reported as a finding.

Guard it with a test: a posterior with dispersed draws must give a predictive strictly
**wider** than the plug-in at the same nominal level; and a posterior artificially
collapsed to a point mass must reproduce the plug-in to numerical tolerance.

### Compute

The variance recursion is deterministic given a draw, so one monthly posterior fit yields
exact daily one-step-ahead predictive distributions for the whole month. Refilter per-draw
from scratch only at a refit, then advance one step per day: that keeps cost at
`O(n_draws)` per day rather than `O(n_draws × T)`.

At ~10s per fit and 102 refits, expect roughly 20 minutes end to end — the graph compiles
once, not per refit. Thin to a fixed number of draws (~1,000-2,000) before carrying them
through the backtest.

### If PyMC fights you

Route B is `emcee` over a hand-written NumPy likelihood — already pinned, and the
governing plan states it "costs you nothing scientifically". `log_prior` and
`log_posterior` in `models.py` exist as that entry point. Switch rather than debug past
about half a day.

### Sanity check before proceeding

Bayesian intervals should be **≥ frequentist widths**, most visibly early in the sample
and at the 99% level. If they are not, hunt for a bug — most likely the shared-`h_next`
one above — before computing anything downstream.

---

## 6. Stages 4-7, in brief

**Stage 4 — evaluation layer (3-4h). The intellectual core.** QLIKE and MSE against the
scaled proxy; DM tests with the Harvey-Leybourne-Newbold correction; PIT histograms;
coverage tables at 90/95/99 and 99% VaR; Kupiec POF and **Christoffersen** independence
and conditional coverage on the 99% VaR hit sequences.

Expect the four models to be **hard to separate on point accuracy**. That is the setup for
the calibration act, not a failure. Christoffersen is the money test: correct *average*
coverage can hide breaches that cluster in crises, and clustering is the failure mode that
actually ruins risk models.

Two caveats belong in the report body, not a footnote: DM's asymptotics assume forecasts
are not functions of estimated parameters (here they are), and models 3 and 4 share a
likelihood so their loss differential may be near-degenerate — return the differential
variance so a reader can check rather than trust the p-value.

**Stage 5 — regime analysis (2-2.5h).** Split on the lagged-VIX label. **Report `n`
alongside every statistic** — stressed days are 375 of 2,890, and a regime table without
sample sizes invites over-reading. Bootstrap CIs on every per-regime coverage estimate are
mandatory; their width is itself on-theme. Likely headline figure: coverage vs nominal by
regime, one panel per model.

**Stage 6 — robustness (1.5-2h).** GARCH-normal vs GARCH-t at 99% (expect normal to fail
hard). Refit cadence at 63 days. **QLIKE ranking on raw, unscaled Parkinson** — this one
is owed by decision D10 and shows the ranking does not hinge on `c`.

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
| Priors frozen in the log **before** any Bayesian out-of-sample number | §5 above | 3 |
| ~~Normal-innovation GARCH variant fitted~~ **done at Stage 2** — full forecast table, key `garch_mle_normal` | plan §3, D15 | used at 6 |
| `garch_mle_normal` kept out of the headline four-model tables (filter on `HEADLINE_MODELS`) | D15 | 4, 5 |
| Near-boundary persistence (`alpha+beta` > 0.999 in 16 of 102 refits) noted rather than discovered late | §1.9 | 7 |
| `BayesianFit`'s emcee fields re-keyed to NUTS before Stage 3 fills any of them | §1.9 | 3 |

---

## 8. Traps specific to this codebase

Ordered by how much damage they do while looking fine.

1. **Shared `h_next` across posterior draws** (§5). Silently destroys the research
   question.
2. **Missing `-0.5*ln(h_t)`** in the log-likelihood. Optimises fine; every variance wrong.
3. **Dropping `sqrt((nu-2)/nu)`** in the plug-in quantile. Inflates intervals a few
   percent, in the direction that flatters the Bayesian model.
4. **Freezing `h` between refits** as well as the parameters. Not a GARCH forecast.
   Guarded since Stage 2 by `test_variance_moves_daily_while_parameters_are_held_fixed`,
   which asserts both halves: `refit_id` constant across a block, `variance` taking a
   distinct value on every day of that same block.
5. **Applying `c` twice**, or to a model already on the return scale. EWMA and GARCH are
   driven by squared returns and need no scaling; only the Parkinson-based RW does.
6. **Weakening a look-ahead test until it passes.** Twice already the correct claim was
   subtler than the obvious one — see `problems-and-solutions.md` #3 and #7. If a
   look-ahead test fails, first establish whether the *test* is wrong before touching the
   code, then keep whichever version is actually true.
7. **Positive-only tests.** Every diagnostic needs a case where it must *not* fire.
   Otherwise a function that always rejects passes the suite while fabricating a result.
8. **Trimming the multi-start grid.** It looks like free speed — 51s against 2s — and all
   grid sizes report 102/102 converged. With one start, 93 of the 102 refits land on a
   *worse* local optimum. Measured, and recorded at `problems-and-solutions.md` #30.
9. **Reading a fit's convergence verdict off the best objective value.** They are
   different questions; entangling them cost 21 days of forecasts once already (#29).

---

## 9. Budget

The governing plan budgets ~15-20h total. Stages 0-2 account for roughly 7-8h of that;
Stages 3-7 are estimated at 13-17h. **The plan is already over its own budget**, which is
what the cut list in §5 of the plan exists for. Cut in its stated order — the SV stretch
goal is already out, then the refit-cadence sensitivity, then the trailing-vol regime
sensitivity, then MSE as a secondary loss. Never the four never-cut items in §3 above.

The checkpoint rule from the plan still stands: **if Stage 3 is not producing forecasts
after about half a day, switch to route B immediately** rather than continuing to debug
PyMC.

---

## 10. First three commands for the next session

```bash
pytest -q                                  # confirm 188 pass before touching anything
python run_all.py --stage backtest         # confirm artefacts regenerate
sed -n '1,60p' docs/project1-implementation-plan.md   # re-read the contract
```

Then start at §4 of this document.
