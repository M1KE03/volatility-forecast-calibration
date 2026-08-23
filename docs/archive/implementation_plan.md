# Implementation plan — exactly what remains to be done

Companion to `research_log.md`. The log records **what was decided and why**; this file
records **what must be built**, function by function, with the formula, the failure
mode, and the acceptance criterion for each.

Scope note: this document does not introduce any new model, feature, metric, or output
beyond the locked design. Where something is genuinely undecided it is marked
**DECISION REQUIRED** and left undecided rather than resolved silently.

**Status as of 2026-08-22:** Stage 2 complete. `src/data.py` is implemented and the
analysis frame is built (2890 rows; 756 train, 2134 out-of-sample). 55 tests pass.
`models.py`, `backtest.py`, `evaluation.py` and `bootstrap.py` are still stubs. No model,
no forecast, no result exists yet.

Next: **Stage 3**, whose frequentist half and baselines are unblocked. Its Bayesian half
needs **D4** (priors) and **D6** (draws retained).

---

## 0. Decisions that must be made before the stages that need them

Nothing below can be implemented correctly around an unmade decision. Each is stated
with the options and the consequence of getting it wrong.

| ID | Question | Blocks | Recommendation |
|---|---|---|---|
| **D1** | Estimation window: expanding or fixed-length rolling? | Stage 4 | Expanding primary, 1,000-day rolling as robustness |
| **D2** | How to handle the Parkinson intraday vs close-to-close scale mismatch under QLIKE | Stage 5 | Keep Parkinson primary; add train-window-only scaling constant; add squared-return proxy as robustness |
| **D3** | Add Giacomini–White alongside Diebold–Mariano? | Stage 5 | Propose yes; **not approved**, will not be added without your say-so |
| **D4** | Priors on (mu, omega, alpha, beta, nu) | Stage 3 | Must be written into `research_log.md` and frozen **before any out-of-sample number is computed** |

Two further decisions surfaced while specifying the stages below:

| ID | Question | Blocks | Recommendation |
|---|---|---|---|
| **D6** | How many posterior draws to carry through the backtest (see §3.6) | Stage 3/4 | Thin to a fixed 2,000 draws; it is a compute/accuracy tradeoff, not a modelling choice |
| **D7** | Predictive intervals for the two baselines: Gaussian or Student-t reference? | Stage 4 | Gaussian. The baselines are meant to be naive, and a fitted t would quietly make them competitors rather than baselines |

---

## Stage 2 — `src/data.py` — **DONE (2026-08-22)**

**Objective:** turn Yahoo Finance downloads into one verified analysis frame, and prove
by test that nothing in it looks forward.

**Depends on:** nothing.

Outcome, with the deviations from this specification recorded in `research_log.md` 1.4
(decision D8, the `build_analysis_frame` signature change, the redefinition of
`missing_dates_spy`, and the corrected wording of look-ahead test 6): calendars agree on
every sample date, no rows dropped, Parkinson defined everywhere, six exactly-zero
returns retained and listed. Regimes: calm 1198, normal 1317, stressed 375.

### 2.1 Functions

| Function | What it must do |
|---|---|
| `download_raw` | Fetch daily bars for one ticker over the sample window with `auto_adjust=False`. Write CSV to `data/raw/{ticker}.csv`. Raise `FileExistsError` if the file exists and `refresh=False`. Normalise columns to `open, high, low, close, adj_close, volume`. |
| `write_manifest` | SHA-256 every CSV in `data/raw/`; write `data/raw/manifest.json` as `{filename: {sha256, n_rows, first_date, last_date, downloaded_utc}}`. |
| `verify_manifest` | Re-hash each CSV, compare to manifest, return `{filename: bool}`. |
| `load_raw` | Read a cached CSV to a `DatetimeIndex` frame. If `verify=True` and the hash mismatches, **raise** — do not warn and continue. |
| `compute_log_returns` | `r_t = log(adj_close_t) - log(adj_close_{t-1})`. First value NaN. |
| `parkinson_variance` | `sigma2_t = (1 / (4 * ln 2)) * (ln(H_t / L_t))^2`. NaN where `H <= L`, recorded in the quality report — never silently zeroed. |
| `assign_vix_regime` | Lag the VIX close by one row, **then** bin: `< 15` calm, `15..25` normal (inclusive both ends), `> 25` stressed. Ordered categorical. First row NaN. |
| `build_analysis_frame` | Inner-join SPY and VIX on date; derive all columns; return `(frame, DataQualityReport)`. |
| `split_train_oos` | Pure date-based chronological split. |

### 2.2 The one thing most likely to go wrong

`assign_vix_regime` does its own lagging. If a caller passes an already-lagged series
the regime is double-lagged, which silently misattributes every stressed day and would
corrupt the headline regime result without any error surfacing. The function's contract
is "give me the raw VIX close series"; a test must cover this.

### 2.3 Tests to write (`tests/test_data.py`)

**Transformation correctness**
1. `compute_log_returns` on a known geometric series returns the exact analytic answer.
2. `parkinson_variance` on `H/L = e` returns `1 / (4 ln 2)`.
3. `parkinson_variance` is invariant to multiplying `H` and `L` by any positive constant
   (this is what makes the adjusted-vs-raw question moot; see D5).
4. `assign_vix_regime` puts boundary values 15.0 and 25.0 in `normal`, per the locked
   spec (`< 15` and `> 25` are the strict cases).
5. `assign_vix_regime` output on row `t` equals the bin of the input at row `t-1`.

**Look-ahead**
6. Build the frame; record row `t`. Replace every input row from `t` onward with NaN and
   rebuild. Assert row `t`'s `log_return`, `parkinson_var`, and `regime` are unchanged.
   Repeat at several `t`. This is the strongest available check and is the template for
   every look-ahead test in the project.
7. `split_train_oos` — assert `train.index.max() < oos.index.min()` and that the union
   is a subset of the input with no reordering.

**Integrity / data quality**
8. `verify_manifest` returns `False` after a byte is altered in a cached CSV.
9. `load_raw` raises on hash mismatch rather than returning data.
10. `download_raw` raises `FileExistsError` when the cache exists and `refresh=False`.
11. Calendar mismatches between SPY and ^VIX appear in `DataQualityReport` and the
    affected rows are absent from the frame — assert they were **dropped**, not filled.

### 2.4 Acceptance criteria

- `python run_all.py --stage data` downloads once, writes the manifest, builds the frame,
  prints the `DataQualityReport` in full, and saves the frame to `data/processed/`.
- Re-running without `--refresh` uses the cache and does not hit the network.
- `data/raw/SPY.csv`, `data/raw/VIX.csv`, `data/raw/manifest.json` committed to git.
- The quality report is printed, not swallowed. Any dropped date is visible.
- All Stage 2 tests pass.

---

## Stage 3 — `src/models.py`

**Objective:** one likelihood, two estimators, two predictive distributions, two
baselines.

**Depends on:** Stage 2. **Blocked by D4** (priors) for the Bayesian half only — the
frequentist half and the baselines can be built first.

### 3.1 The shared likelihood — the core of the project

Model:

```
r_t   = mu + eps_t
eps_t = sqrt(h_t) * z_t,     z_t ~ standardised Student-t(nu), Var(z_t) = 1
h_t   = omega + alpha * eps_{t-1}^2 + beta * h_{t-1}
```

Standardised t density (unit variance, so `h_t` *is* the conditional variance and needs
no `nu/(nu-2)` rescaling anywhere downstream):

```
f(z) = G((nu+1)/2) / [ G(nu/2) * sqrt(pi * (nu-2)) ] * (1 + z^2/(nu-2))^(-(nu+1)/2)
```

Per-observation log-likelihood, with `z_t = (r_t - mu)/sqrt(h_t)`:

```
ll_t = lnG((nu+1)/2) - lnG(nu/2) - 0.5*ln(pi*(nu-2))
       - 0.5*ln(h_t) - ((nu+1)/2) * ln(1 + z_t^2/(nu-2))
```

The `- 0.5*ln(h_t)` term is the Jacobian of `r -> z`. Omitting it is the classic silent
error: the likelihood still optimises and still looks plausible, but every variance
estimate is wrong.

Constraints: `omega > 0`, `alpha >= 0`, `beta >= 0`, `alpha + beta < 1`, `nu > 4`.
Return `-inf` outside, never raise — the same function is handed to both an optimiser
and a sampler.

### 3.2 Functions

| Function | Notes |
|---|---|
| `garch11_filter` | numba `@njit`. `h[0] = h0`; `h[t] = omega + alpha*(r[t-1]-mu)^2 + beta*h[t-1]`. Note it is the **residual** squared, not the raw return squared. |
| `garch11_t_loglik` | numba `@njit`. Formula above. |
| `backcast_initial_variance` | Sample variance of the estimation window's returns. Training data only. |
| `log_prior` | **D4 — not yet specified.** |
| `log_posterior` | `log_prior + garch11_t_loglik`. |
| `fit_garch_mle` | `scipy.optimize.minimize` on the negative log-likelihood, L-BFGS-B with box bounds plus a stationarity check, multi-start from a small fixed grid. Return `converged` and the optimiser message verbatim. |
| `sample_garch_posterior` | `emcee.EnsembleSampler`, seeded. Return acceptance fraction, autocorrelation time, effective sample size. |
| `plugin_predictive` | Closed form. |
| `posterior_predictive` | Mixture. See §3.5. |
| `forecast_yesterday_volatility` | One-step lag of the realised variance series. |
| `forecast_ewma` | `h_{t+1} = 0.94*h_t + 0.06*r_t^2`, seeded from the training window. Lambda fixed, never estimated. |

### 3.3 Frequentist plug-in predictive

Conditions on the MLE as if true. Carries **innovation uncertainty only**. For quantile
level `q`:

```
r_q = mu_hat + sqrt(h_next) * sqrt((nu-2)/nu) * t_ppf(q, nu)
```

The `sqrt((nu-2)/nu)` factor converts a standard t quantile to the standardised
(unit-variance) t. Dropping it inflates every interval by a few percent and would bias
the project's central comparison in the Bayesian model's favour.

### 3.4 Bayesian posterior predictive

Carries **both** innovation and parameter uncertainty. No closed form. Mixture CDF:

```
F(r) = (1/J) * sum_j  T_nu_j( (r - mu_j) / (sqrt(h_next_j) * sqrt((nu_j-2)/nu_j)) )
```

Quantiles by root-finding (Brent) on `F(r) - q`, bracketed by the min and max of the
per-draw quantiles.

### 3.5 The critical detail: `h_next` varies by draw

Each posterior draw `theta_j` implies its **own** filtered variance path and therefore
its own `h_next_j`. Passing a single shared `h_next` — for instance one computed at the
posterior mean — collapses the posterior predictive to something very close to the
plug-in predictive and would destroy the project's research question while producing
entirely plausible-looking output. This is the single most dangerous bug available in
this codebase.

`posterior_predictive` therefore takes `h_next_by_draw` of shape `(n_draws,)`, and a
test must assert that a posterior with dispersed draws yields a predictive strictly
wider than the plug-in at the same nominal level.

### 3.6 Compute note — DECISION REQUIRED (D6)

Default sampler settings give `32 walkers * (3000-1000)/5 thin = 12,800` retained draws.
Carrying 12,800 variance paths through ~2,140 daily forecast steps is wasteful. Proposal:
thin to a fixed **2,000** draws for the predictive stage. Pure compute/accuracy tradeoff,
no modelling content, but it should be an explicit recorded choice rather than an
accident.

### 3.7 Tests to write (`tests/test_models.py`)

**Likelihood**
1. `garch11_filter` matches a hand-computed 5-step recursion exactly.
2. `garch11_t_loglik` matches a direct `scipy.stats.t` computation over the same series.
3. As `nu -> large`, the standardised t log-likelihood approaches the Gaussian one.
4. Returns `-inf` for `alpha + beta >= 1`, `omega <= 0`, `nu <= 4`.

**Estimation**
5. **Recovery:** simulate 5,000 observations from known parameters with a fixed seed;
   assert the MLE recovers them within a stated tolerance.
6. **Cross-check against `arch`:** fit the same series with `arch`'s GARCH(1,1)-t and
   assert agreement to tight tolerance. This is the sole use of `arch` in the project
   and it never produces a headline number.
7. A deliberately non-converging fit returns `converged=False` — it must not raise, and
   must not be silently retried into a spurious success.

**Bayesian**
8. Same seed gives a bit-identical chain.
9. Posterior mode is close to the MLE under a diffuse prior — a basic sanity check that
   the likelihood is genuinely shared between the two paths.
10. **The wideness test:** with dispersed posterior draws, the posterior-predictive 95%
    interval is strictly wider than the plug-in 95% interval. This is the guard against
    the §3.5 bug.
11. With a posterior artificially collapsed to a point mass, the posterior predictive
    equals the plug-in predictive to numerical tolerance.

**Baselines**
12. `forecast_ewma` matches a hand-computed 5-step recursion.
13. `forecast_yesterday_volatility` is exactly a one-step lag — assert no row uses its
    own day's value.

### 3.8 Acceptance criteria

- Frequentist and Bayesian paths demonstrably call the same `garch11_t_loglik`.
- Recovery and `arch` cross-check both pass.
- Convergence diagnostics returned and non-suppressible.
- Priors frozen in `research_log.md` before any out-of-sample run.

---

## Stage 4 — `src/backtest.py`

**Objective:** the walk-forward loop. This is where look-ahead bias would enter if it
enters anywhere.

**Depends on:** Stages 2–3. **Blocked by D1.**

### 4.1 The two cadences

- **Refit every 21 trading days.** Re-estimate parameters: one MLE plus one MCMC run.
- **Filter every day.** Between refits, parameters are held fixed but the variance
  recursion is still advanced daily with each newly observed return.

Freezing `h` between refits as well is a plausible-looking mistake that would make the
forecasts up to 21 days stale and would not be a GARCH forecast at all.

### 4.2 Loop specification

```
for each out-of-sample date t:
    if t is a refit date:
        window = estimation_slice(...)        # MUST end strictly before t
        h0     = backcast_initial_variance(window returns)
        mle    = fit_garch_mle(window)
        post   = sample_garch_posterior(window, seed=config.mcmc_seed + refit_id)
        h_mle    = filter(mle.params, history up to t-1)[-1]
        h_by_draw = [filter(theta_j, history up to t-1)[-1] for j in draws]
    else:
        h_mle     = advance one step with mle.params and r_{t-1}
        h_by_draw = advance one step per draw with theta_j and r_{t-1}

    produce forecasts for date t from h_mle / h_by_draw
    observe r_t, record it
```

The per-draw state is refiltered from scratch only at a refit and advanced one step per
day thereafter. That keeps the cost at `O(n_draws)` per day rather than
`O(n_draws * T)`, and is the difference between a backtest that runs in minutes and one
that does not finish.

### 4.3 Invariants, each enforced by a test

1. A forecast for date `t` uses returns through `t-1` only.
2. Parameters used on `t` come from a fit whose window ends at or before `t-1`.
3. The regime label for `t` comes from the VIX close at `t-1`.
4. `h0` is backcast from the estimation window only.
5. No shuffling, no random splitting, no reordering, ever.

### 4.4 Output schema

One row per out-of-sample date:

```
log_return, parkinson_var, regime, refit_id
{model}_var, {model}_mean,
{model}_lo_90, {model}_hi_90, {model}_lo_95, {model}_hi_95,
{model}_lo_99, {model}_hi_99, {model}_var99
```

for `model` in `{yesterday, ewma, garch_mle, garch_bayes}`. Written to
`data/processed/` alongside the `RefitRecord` list and the serialised `BacktestConfig`,
so every saved number is traceable to the run that produced it.

### 4.5 Tests to write (`tests/test_backtest.py`)

1. **The master look-ahead test.** Run the backtest. Then corrupt every observation from
   date `t` onward and re-run. Assert every forecast for dates `< t` is bit-identical.
   Run at several `t`. If this passes, the pipeline has no forward leakage.
2. `estimation_slice` never includes the refit date itself — assert at boundaries.
3. `make_refit_dates` — first OOS date is a refit date; consecutive refits are exactly
   21 trading days apart; the count matches `ceil(n_oos / 21)`.
4. Between refits, `refit_id` is constant while `{model}_var` still changes day to day
   (proves daily filtering is actually happening).
5. A failed MLE propagates to the `RefitRecord` with `mle_converged=False` and the run
   continues, with the failure visible in the output rather than absorbed.
6. Same seed, same config, bit-identical forecast frame.

### 4.6 Acceptance criteria

- Full backtest runs end to end. Expected wall time on this machine: a few minutes,
  based on the measured 5.5 us/likelihood-eval.
- All refit diagnostics persisted, including any failures.
- Master look-ahead test passes.
- **Do not proceed to Stage 5 until it does.** A calibration result computed on a
  leaking backtest is worse than no result, because it looks fine.

---

## Stage 5 — `src/evaluation.py`

**Objective:** score the forecasts. **Blocked by D2 and D3.**

### 5.1 Point losses (scored against the Parkinson proxy)

```
QLIKE_t = rv_t/h_t - ln(rv_t/h_t) - 1        # >= 0, minimised at h_t = rv_t
MSE_t   = (rv_t - h_t)^2
```

Note the common alternative `QLIKE = ln(h_t) + rv_t/h_t` differs by `ln(rv_t) - 1`,
which does not involve the forecast — so Diebold–Mariano differentials are identical
under either form. The version above is used because it is non-negative and zero at a
perfect forecast, which makes the tables easier to read honestly.

**D2 applies to every number in this subsection.** Report both scaled and unscaled
QLIKE, with the scaling constant estimated on the initial training window only.

### 5.2 Calibration (scored against observed returns — the headline, unaffected by D2)

| Function | Formula / notes |
|---|---|
| `interval_coverage` | Fraction of `r_t` inside `[lo, hi]`, plus separate lower/upper breach counts. A model can hit nominal coverage overall while badly misallocating breaches between tails; that is a substantive failure, not a rounding detail. |
| `pit_values` | `F_t(r_t)` under each model's predictive. Uniform(0,1) and i.i.d. under correct specification. |
| `var_exceedances` | `1` where `r_t < VaR_t` (lower/loss tail). |
| `kupiec_pof_test` | `LR_uc = -2 ln[ ((1-p)^(n-x) p^x) / ((1-pi_hat)^(n-x) pi_hat^x) ]`, `pi_hat = x/n`, chi-sq(1). |
| `christoffersen_independence_test` | Markov transition counts `n_ij`, chi-sq(1). |
| `conditional_coverage_test` | `LR_cc = LR_uc + LR_ind`, chi-sq(2). |

Kupiec alone is not enough: it counts breaches but is blind to clustering, and clustered
breaches are exactly the failure mode that matters in a stress episode — which is the
second half of the research question.

### 5.3 Diebold–Mariano

`d_t = L_a,t - L_b,t`; `DM = mean(d) / sqrt(HAC_var(d)/n)`; Newey–West bandwidth
`floor(4*(n/100)^(2/9))` unless overridden.

Four caveats must accompany every reported DM result, and belong in the report text
rather than a footnote:

1. **Estimated parameters.** DM's standard asymptotics assume forecasts are not
   functions of estimated parameters. Here they are, and are re-estimated on a rolling
   schedule. Read nominal p-values as indicative.
2. **Near-degenerate differential.** Models 3 and 4 share a likelihood and differ only
   by integration over the posterior, so `var(d)` may approach zero and DM is badly
   sized there. `DieboldMarianoResult.differential_variance` is returned precisely so a
   reader can check this rather than take the p-value on trust.
3. **Autocorrelation.** HAC correction is necessary but not sufficient at short samples.
4. **Multiplicity.** Multiple pairwise comparisons across regimes and levels;
   unadjusted p-values overstate significance.

The block bootstrap is the more trustworthy uncertainty statement and should carry the
weight of the conclusions.

### 5.4 Regime conditioning

`summarise_by_regime` splits on the lagged-VIX label and **must report `n` alongside
every statistic**. The stressed regime is a small fraction of the sample; a regime table
without sample sizes invites over-reading, which for this project would be the most
likely route to a wrong conclusion.

### 5.5 Tests to write (`tests/test_evaluation.py`)

1. QLIKE is zero when `h == rv`, positive otherwise.
2. QLIKE is asymmetric: under-forecasting is penalised more than over-forecasting by the
   same ratio.
3. Coverage of intervals drawn from a known distribution matches nominal within
   simulation error.
4. PIT values from a correctly specified model pass a uniformity test; from a
   deliberately too-narrow model, they do not.
5. Kupiec reproduces a published worked example.
6. Christoffersen independence rejects on a deliberately clustered exceedance sequence
   and does not reject on an i.i.d. one at the same rate.
7. DM on two identical loss series returns a statistic of exactly 0.
8. DM matches a hand-computed value on a small fixed series.
9. `summarise_by_regime` partitions with no row lost or double-counted.

---

## Stage 6 — `src/bootstrap.py`

**Objective:** uncertainty intervals around every headline difference.

**Depends on:** Stage 5. Not blocked.

### 6.1 Specification

Politis–Romano stationary bootstrap. Blocks start at a uniform position and continue
with probability `1 - 1/20` per step, wrapping circularly; block lengths are geometric
with mean 20. 1,000 replications, fixed seed.

### 6.2 The correctness trap

`bootstrap_loss_differential` and `bootstrap_coverage_difference` must resample **both**
series with the **same indices** within each replicate. Independent resampling breaks
the pairing and inflates the interval, discarding exactly the common variation that
makes a paired comparison informative — and it fails in the safe-looking direction, so
it would never announce itself.

### 6.3 Tests to write (`tests/test_bootstrap.py`)

1. Same seed gives identical indices; different seeds do not.
2. Index array has shape `(1000, n)` with every entry in `[0, n)`.
3. Empirical mean block length converges to 20 over many replicates.
4. On i.i.d. data, the bootstrap CI for the mean approximately matches the analytic CI.
5. On strongly autocorrelated data, the block-bootstrap CI is **wider** than an i.i.d.
   bootstrap CI. This is the whole justification for using it.
6. **Paired resampling:** with `loss_a == loss_b`, the differential CI is exactly
   `[0, 0]`. Independent resampling would produce a non-degenerate interval, so this
   test catches the §6.2 bug directly.

---

## Stage 7 — Figures and notebooks

**Depends on:** Stages 2–6 complete and validated.

Notebooks are created only now, each as a thin presentation layer over cached artefacts
in `data/processed/`. They must contain no reusable logic and must not re-run the
backtest.

| Notebook | Content |
|---|---|
| `01_eda.ipynb` | Return series, the Parkinson proxy, VIX with regime bands, the training/OOS boundary, data quality findings. |
| `02_results.ipynb` | Coverage tables, PIT histograms, VaR exceedance timelines, QLIKE tables, DM results, bootstrap intervals. |
| `03_robustness.ipynb` | Prior sensitivity (D4), rolling vs expanding window (D1), the alternative squared-return proxy (D2), sampler settings. |

Figures written to `figures/` by `run_all.py --stage figures`, not by the notebooks, so
that the report's figures are reproducible from the CLI alone.

`jupyterlab` gets added to `requirements.txt` at this stage, pinned.

---

## Stage 8 — `report/report.md`

**Depends on:** everything above.

Written last, against real numbers only. The banner at the top of the file forbids
placeholder figures, and it should stay there until the numbers are real.

Specific obligations for the write-up:

- Never claim a model is better on the basis of a lower loss in one period. Conclusions
  rest on bootstrap intervals; an interval containing zero is a legitimate, reportable
  finding, not a failed experiment.
- Keep the seven quantities distinct throughout: observed returns, Parkinson proxy,
  conditional variance forecasts, predictive intervals, VaR forecasts, parameter
  uncertainty, innovation uncertainty.
- State the D2 proxy caveat in the results section, not only in the methods section, and
  note explicitly that it does not touch the calibration results.
- Report every convergence failure, every dropped date, and every underpowered regime
  subsample.

---

## Test inventory

Two categories, tracked separately because they fail for different reasons.

**Transformation tests** — is the arithmetic right? Log returns, Parkinson, regime
binning, GARCH filter, t log-likelihood, EWMA recursion, QLIKE, Kupiec, Christoffersen,
DM, bootstrap block lengths.

**Look-ahead tests** — does anything see the future? The corrupt-the-future pattern
applied to the analysis frame (Stage 2), the estimation-window slice (Stage 4), the full
backtest (Stage 4), and regime assignment (Stage 2). Plus: no random splitting anywhere,
and `h0`/EWMA seeds drawn from training data only.

The second category is the one that protects the project's conclusions. The first only
protects its arithmetic.

---

## Sequencing summary

| Stage | Module | Blocked by | Can start |
|---|---|---|---|
| 2 | `data.py` | — | **done** |
| 3 | `models.py` | D4, D6 (Bayesian half only) | **now** |
| 4 | `backtest.py` | D1, Stage 3 | after 3 |
| 5 | `evaluation.py` | D2, D3, Stage 4 | after 4 |
| 6 | `bootstrap.py` | Stage 5 | after 5 |
| 7 | figures, notebooks | Stages 2–6 | after 6 |
| 8 | report | all | last |

Stage 2 is done. Stage 3's frequentist half and baselines can proceed now; its
Bayesian half needs D4 and D6.
