# Handoff: how to continue

**State as of 2026-08-23.** Stage 1 complete. Two baselines forecast; no GARCH model
exists. 127 tests pass.

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
python run_all.py --stage backtest   # baselines + acceptance figure
pytest -q                            # expect 127 passed
```

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
| `src/eda.py` | **Complete.** ARCH-LM, Ljung-Box, ADF, ACF of squared returns. |
| `src/figures.py` | **Partial.** House style + 5 figures (Stages 0-1). More added per stage. |
| `src/backtest.py` | **Complete for baselines.** The loop, refit schedule, long-format output, persistence. |
| `src/models.py` | **Half.** Interface + both baselines done. **Every GARCH function is a stub.** |
| `src/evaluation.py` | **All stubs** (10 functions). |
| `src/bootstrap.py` | **All stubs** (4 functions). |

Artefacts in `data/processed/`: `analysis_frame.csv` (2,890 rows), `forecasts.csv`
(4,268 rows = 2,134 dates × 2 models), `refit_records.csv` (102), `backtest_config.json`.

Tests: `test_data.py` 29, `test_eda.py` 22, `test_backtest.py` 20, `test_models.py` 17,
`test_smoke.py` 5.

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

## 4. Stage 2 — frequentist GARCH(1,1)-t

*Governing plan estimate: 2-2.5h. This is the next thing to do.*

### What to build

In `src/models.py`, working bottom-up so each piece is testable before the next depends
on it:

1. `GarchParams.to_array` / `from_array` / `is_valid` — trivial, but `is_valid` encodes
   the constraints (`omega > 0`, `alpha, beta >= 0`, `alpha + beta < 1`, `nu > 4`).
2. `garch11_filter` — the recursion. Note it is the **residual** squared,
   `(r[t-1] - mu)**2`, not the raw return squared.
3. `garch11_t_loglik` — the standardised Student-t log-likelihood. **The `-0.5*ln(h_t)`
   Jacobian term is the classic silent omission**: without it the likelihood still
   optimises and still looks plausible, but every variance estimate is wrong.
4. `backcast_initial_variance` — sample variance of the estimation window. Training data
   only.
5. `fit_garch_mle` — `scipy.optimize.minimize`, L-BFGS-B with bounds plus a stationarity
   check, multi-start from a small fixed grid. **Return `converged` and the optimiser
   message verbatim; never silently retry a failure into a success.**
6. `plugin_predictive` — closed form. The scale factor `sqrt((nu-2)/nu)` converts a
   standard t quantile to the standardised (unit-variance) t. **Dropping it inflates
   every interval by a few percent, biasing the project's central comparison in the
   Bayesian model's favour.**

Then extend `backtest.py`: add `"garch_mle"` to the model list and give the loop a real
refit at each of the 102 refit dates, with daily filtering in between.

### The two-cadence rule

This is where the loop stops being trivial:

- **Refit (every 21 days):** re-estimate parameters on the expanding window.
- **Filter (every day):** parameters held fixed, but the variance recursion still advances
  daily with each newly observed return.

Freezing `h` between refits as well is a plausible-looking mistake that makes forecasts up
to 21 days stale and is not a GARCH forecast at all. A test should assert `refit_id` is
constant across a block while `variance` still changes day to day.

### Also at this stage

- Fit a **normal-innovation variant** too. It costs almost nothing now and is the Stage 6
  ablation — normal vs t is the cheap, decisive lever on 99% tail coverage.
- Residual diagnostics on the warm-up fit: Ljung-Box on standardised residuals and their
  squares, QQ plot against the fitted t.
- Parameter-stability plot across the 102 refits (nearly free, good diagnostic).

### Tests that matter

- **Recovery:** simulate 5,000 observations from known parameters with a fixed seed;
  assert the MLE recovers them within a stated tolerance.
- **Cross-check against `arch`:** fit the same series with `arch`'s GARCH(1,1)-t and
  assert agreement to tight tolerance. This is the *only* use of `arch` in the project and
  it must never produce a headline number.
- `garch11_filter` matches a hand-computed 5-step recursion exactly.
- As `nu -> large`, the t log-likelihood approaches the Gaussian one.
- Returns `-inf` outside the constraint set — never raises, because the same function is
  handed to both an optimiser and a sampler.

### Done when

GARCH forecasts complete over all 2,134 days, residual diagnostics recorded, the
look-ahead audit **still passing** with the new model in the loop.

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
| Normal-innovation GARCH variant fitted | plan §3 | 2, used at 6 |

---

## 8. Traps specific to this codebase

Ordered by how much damage they do while looking fine.

1. **Shared `h_next` across posterior draws** (§5). Silently destroys the research
   question.
2. **Missing `-0.5*ln(h_t)`** in the log-likelihood. Optimises fine; every variance wrong.
3. **Dropping `sqrt((nu-2)/nu)`** in the plug-in quantile. Inflates intervals a few
   percent, in the direction that flatters the Bayesian model.
4. **Freezing `h` between refits** as well as the parameters. Not a GARCH forecast.
5. **Applying `c` twice**, or to a model already on the return scale. EWMA and GARCH are
   driven by squared returns and need no scaling; only the Parkinson-based RW does.
6. **Weakening a look-ahead test until it passes.** Twice already the correct claim was
   subtler than the obvious one — see `problems-and-solutions.md` #3 and #7. If a
   look-ahead test fails, first establish whether the *test* is wrong before touching the
   code, then keep whichever version is actually true.
7. **Positive-only tests.** Every diagnostic needs a case where it must *not* fire.
   Otherwise a function that always rejects passes the suite while fabricating a result.

---

## 9. Budget

The governing plan budgets ~15-20h total. Stages 0 and 1 account for roughly 4-5h of that;
Stages 2-7 are estimated at 15-19.5h. **The plan is already over its own budget**, which is
what the cut list in §5 of the plan exists for. Cut in its stated order — the SV stretch
goal is already out, then the refit-cadence sensitivity, then the trailing-vol regime
sensitivity, then MSE as a secondary loss. Never the four never-cut items in §3 above.

The checkpoint rule from the plan still stands: **if Stage 3 is not producing forecasts
after about half a day, switch to route B immediately** rather than continuing to debug
PyMC.

---

## 10. First three commands for the next session

```bash
pytest -q                                  # confirm 127 pass before touching anything
python run_all.py --stage backtest         # confirm artefacts regenerate
sed -n '1,60p' docs/project1-implementation-plan.md   # re-read the contract
```

Then start at §4 of this document.
