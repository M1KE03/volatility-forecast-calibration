# Research log

Two sections: a **decision register** (what was decided, why, and what is still open)
and a **changelog** (what changed in each stage). Append-only. Nothing here should be
edited to match a later result.

This file is the *chronological* record. `docs/problems-and-solutions.md` is the
*problem-oriented* view of the same material: every difficulty encountered, why it
mattered, and the fix. The two are kept in sync; where they disagree, this file is
authoritative because it is append-only and dated.

---

## 1. Decision register

### 1.1 Locked by the researcher before any code was written

These are inputs to the project, not choices made during implementation. They are
reproduced in `README.md` and are not to be changed without an explicit entry here.

Asset SPY; daily frequency; yfinance; sample 2014-01-01 to 2025-06-30; initial training
2014-01-02 to 2016-12-30; out-of-sample 2017-01-03 to 2025-06-30; log returns from
adjusted close; Parkinson high/low variance proxy for point forecasts; observed returns
as the calibration target; 1-day horizon; refit every 21 trading days; two-sided 90/95/99%
intervals plus one-sided 99% VaR; QLIKE primary and variance-MSE secondary;
Diebold-Mariano as the primary comparison test; stationary block bootstrap with mean
block length 20, 1,000 replications, fixed seed; VIX regimes on the lagged close at 15
and 25.

Out of scope: LSTM, Transformer, stochastic volatility, macro variables, multiple assets,
dashboards.

### 1.2 Decided at Stage 0 (2026-08-21)

**B1 — Bayesian backend: numba-compiled shared likelihood + emcee.**
The GARCH variance recursion is sequential and cannot be vectorised, so the likelihood is
an O(T) loop. The locked design implies roughly 102 refits (about 2,140 out-of-sample days
divided by 21). At, say, 32 walkers x 3,000 draws, that is order 1e5 likelihood
evaluations per refit and order 1e10 inner-loop iterations across the backtest —
infeasible in pure Python. Options requiring a system C/C++ compiler (Stan/cmdstanpy,
PyMC's fast backend) were rejected because no compiler is present on the target machine
and requiring one damages reproducibility. numba ships its own LLVM in its wheel; emcee
is pure Python and seedable. Rejected alternatives kept in reserve: NumPyro/JAX scan, or
a hand-rolled adaptive Metropolis sampler if emcee mixes poorly on this posterior.

Consequence: `src/models.py` holds **one** `garch11_t_loglik` used by both the
frequentist MLE (via `scipy.optimize`) and the Bayesian posterior. `arch` is a
**validation-only** dependency — a unit test asserts our MLE optimum matches an `arch`
fit to tight tolerance. `arch` never produces a headline number.

**B2 — Scaffold in place.**
The working directory is `volatility forecast calibration/`, not
`project-1-volatility-uq/`. Scaffolded in place rather than nesting a subdirectory; the
Git repository name is what will matter downstream. Reversible by renaming the folder.

**B3 — `data/raw/` is committed, not gitignored.**
Yahoo Finance revises history, so a download script alone does not make results
reproducible. SPY and ^VIX daily for 2014-2025 is under 1 MB. The raw CSVs will be
committed with a SHA-256 manifest; the loader verifies the hash and refuses to overwrite
a cached file without an explicit `--refresh`.

**D5 — Conventions adopted** (each applied identically to all four models, so none
biases the comparison):

- *Mean equation:* constant mu, estimated. Forcing mu = 0 would displace every predictive
  interval by the same small amount, and intervals are evaluated against observed returns.
- *Initial conditional variance h0:* backcast from the sample variance of the **training**
  returns only.
- *VIX regime alignment:* the regime for forecast date t uses the VIX close at t-1. If the
  ^VIX and SPY calendars disagree on a date, that date is dropped and logged. VIX is never
  forward-filled from a later observation.
- *Price adjustment:* log(High/Low) is invariant to a within-day multiplicative adjustment
  factor, so Parkinson is unaffected by the adjusted-versus-raw question. Returns use
  adjusted closes. Data is downloaded with `auto_adjust=False` and both columns retained
  so this is auditable rather than assumed.
- *VaR sign:* one-sided 99% VaR is the lower (loss) tail of the return distribution.

**Notebooks deferred.** Empty `.ipynb` files are noise and merge badly. Each notebook is
created at the stage where it acquires content: `01_eda` after `data.py`, `02_results`
after `evaluation.py`, `03_robustness` last.

### 1.3 Open items — must be resolved before the stage named

**D1 — Expanding vs rolling estimation window. Blocks `backtest.py`.**
The locked decisions fix the initial training period and the refit cadence but not
whether the estimation window expands or rolls at fixed length. This materially changes
results and changes which asymptotic theory applies to the forecast comparison.
Recommendation: expanding window as primary (about 750 observations is thin for GARCH-t,
and watching parameter uncertainty shrink over the sample is directly relevant to the
research question), with a fixed 1,000-day rolling window as a robustness check.
**Not yet decided.**

**D2 — Parkinson proxy scale mismatch. Blocks `evaluation.py`.**
Parkinson estimates *intraday* variance; the models forecast *close-to-close* variance,
which includes the overnight gap. Parkinson is therefore systematically biased low.
QLIKE's proxy-robustness property requires a *conditionally unbiased* proxy, so a
multiplicative bias violates its assumptions. In practice a roughly constant
multiplicative bias mostly shifts the QLIKE level and is less damaging to *differences*
between models scored on the same proxy — which is what the DM test uses — but this is
not harmless and must not be papered over. Proposed handling: (1) keep Parkinson as
primary exactly as locked, but estimate a scaling constant on the **initial training
window only** and report both scaled and unscaled QLIKE; (2) add squared close-to-close
returns as a secondary, noisier but conditionally unbiased proxy for robustness;
(3) state the bias explicitly in the report and note that QLIKE comparisons are
within-proxy. Affects the point-forecast loss **only**; interval calibration and VaR are
scored against observed returns and are unaffected. **Not yet decided.**

**D3 — DM test against a nearly degenerate loss differential. Blocks `evaluation.py`.**
Models 3 and 4 share a likelihood and differ only by integration over the posterior.
Their QLIKE differential will be small and strongly autocorrelated, and DM is poorly
sized as the differential approaches degeneracy. DM will be implemented as locked. A
secondary Giacomini-White conditional predictive ability test — valid under estimation
uncertainty with a rolling scheme, which is exactly this setting — is **proposed but not
approved** and will not be added without explicit approval.

**D4 — Priors. Blocks `models.py`.**
Priors are unspecified and drive the headline finding, since the research question is
precisely whether parameter uncertainty improves calibration; loose priors would widen
the Bayesian intervals and could manufacture the result. Priors and support constraints
(omega > 0; alpha, beta >= 0; alpha + beta < 1; nu > 4 so kurtosis is finite) must be
written into this log and frozen **before any out-of-sample result is computed**. Prior
sensitivity belongs in `03_robustness`. **Not yet decided.**

**D6 — Posterior draws carried through the backtest. Blocks `models.py`/`backtest.py`.**
Default sampler settings retain 12,800 draws; carrying that many variance paths across
~2,140 daily steps is wasteful. Proposal: thin to a fixed 2,000 draws for the predictive
stage. Compute/accuracy tradeoff, no modelling content. **Not yet decided.**

**D7 — Reference distribution for baseline intervals. Blocks `backtest.py`.**
The baselines yield a variance forecast but no innovation distribution, yet they need
intervals to be comparable on coverage. Proposal: Gaussian reference. Fitting a
Student-t would promote them from baselines to competitors. **Not yet decided.**

### 1.4 Decided at Stage 2 (2026-08-22)

**D8 — One month of download buffer before `SAMPLE_START`.**
The locked sample begins 2014-01-01 and the training window at 2014-01-02, which is the
first trading day of the sample. Both derived quantities that the pipeline depends on
are lagged: `log_return` on 2014-01-02 needs the adjusted close on 2013-12-31, and the
regime label for 2014-01-02 needs the VIX close on 2013-12-31. Downloading only from
2014-01-01 makes both NaN, which would silently shorten the locked training window by a
day and leave the first row unusable.

Resolution: raw data is downloaded from `DOWNLOAD_START = 2013-12-01`, and
`build_analysis_frame` computes every derived column on the full joined history and
*then* trims to `[SAMPLE_START, SAMPLE_END]`. The buffer feeds the lag and nothing else;
no buffer row reaches the backtest, and the locked windows are unchanged. The committed
raw CSVs therefore start at 2013-12-02 (the first trading day at or after
`DOWNLOAD_START`) while the analysis frame starts at 2014-01-02 as locked. Verified by
`test_analysis_frame_is_trimmed_to_the_sample_window_but_lags_use_the_buffer`.

**Signature change (flagged, per the Stage 1 note).**
`build_analysis_frame` gains keyword-only `sample_start` and `sample_end` parameters,
defaulting to the locked constants. Needed to implement D8 and to let tests build frames
over synthetic date ranges. No positional argument changed.

**Interpretation of `DataQualityReport.missing_dates_spy`.**
The Stage 1 docstring said "SPY dates present in the requested range but absent from the
download", which is not computable without a trading-day calendar; a naive business-day
comparison would report every market holiday as a data problem. Redefined as *dates on
which ^VIX traded but SPY has no bar*, which is a genuine gap rather than a holiday.
`calendar_mismatch_dates` covers disagreements in both directions and remains the
authoritative list of dropped rows.

**Deviation in the look-ahead test's construction.**
`docs/implementation_plan.md` §2.3.6 says to corrupt every input row "from `t` onward"
and assert row `t` is unchanged. That cannot hold as written: row `t`'s `log_return` and
`parkinson_var` are by definition functions of row `t`'s own inputs, so NaN-ing row `t`
changes them for a reason that is not look-ahead. Implemented as: corrupt every input
row **strictly after** `t`, assert every row at or before `t` is bit-identical. The
complementary direction — that the regime on `t` ignores the VIX close on `t` — is
asserted by a separate test, so the lag is still covered from both sides.

---

### 1.5 Plan switch (2026-08-23)

**The governing plan is now `docs/project1-implementation-plan.md`.**
`docs/implementation_plan.md` was retired at the switch and subsequently deleted from
the working tree. It survives only in git history (`git log -- docs/implementation_plan.md`,
last present at commit d2689b9). Nothing in this repository should be implemented from
it; the notes below record what it said so that the history of the decision is legible
without recovering the file.

The switch changes no locked decision. §1.1 of this register and §1 of the governing
plan agree line for line — asset, sample window, warm-up, refit cadence, interval levels,
losses, DM, bootstrap, VIX thresholds, the four forecasters, and the exclusions. Nothing
built so far is invalidated: `src/data.py` and its 56 tests implement the data half of
the governing plan's Stage 0 exactly as specified.

**Stage numbering is re-keyed and the old numbering is retired.** The archived plan
numbered stages by module (its Stage 2 = `data.py`); the governing plan numbers them by
workflow (its Stage 2 = frequentist GARCH). Every "Stage N" reference in this repository
from this entry forward means the governing plan's N:

| Stage | Governing plan | State |
|---|---|---|
| 0 | Scaffold, data, EDA | data done; **EDA outstanding** |
| 1 | Backtest harness + RW-in-vol and EWMA baselines | not started |
| 2 | Frequentist GARCH(1,1)-t | not started |
| 3 | Bayesian GARCH(1,1)-t | not started |
| 4 | Evaluation layer | not started |
| 5 | Regime-conditional analysis | not started |
| 6 | Robustness and ablations | not started |
| 7 | Write-up and polish | not started |

**D1, D2, D3, D6 and D7 are withdrawn, not resolved.** They were raised by the archived
plan; the governing plan does not carry them. Where it is silent, its own default reading
applies as each stage is reached, and the reading actually used is recorded here at that
point rather than pre-negotiated. D3 (Giacomini–White) was never authorised scope and is
dropped outright.

**B1 stands.** The governing plan §3 prefers PyMC (route A) with emcee as route B. B1
already chose the numba likelihood + emcee path, for a machine-specific reason that has
not changed: no system C/C++ compiler is present, which is what route A needs. The
governing plan states route B "costs you nothing scientifically" and sets an end-of-day-4
deadline to fall back to it; that fallback is simply taken up front. `arch` remains a
validation-only dependency.

**What the switch adds to the outstanding work**, all of it absent from the archived plan:

- Stage 0 EDA: ARCH-LM (the motivating test), Ljung–Box on squared returns, ACF of r²,
  ADF on returns.
- Stage 2: a normal-innovation GARCH variant, fitted as the Stage 6 ablation.
- Stage 6: GARCH-normal vs GARCH-t coverage at 99%; a 63-day refit-cadence spot check;
  the squared-return proxy sensitivity.
- Stage 7 and the project as a whole: a ~15–20h budget, a week layout, an ordered cut
  list, and a never-cut list (look-ahead audit, Christoffersen, bootstrap CIs on regime
  coverage, limitations section).

**What the switch drops:** the archived plan's function-by-function specifications,
formulas, named failure modes, and ~50-test inventory. These are not carried forward as a
document. Where a formula there was correct it will be re-derived at the stage that needs
it; the archived file remains readable if a detail is worth recovering.

---

### 1.6 B1 revised — route A adopted (2026-08-23)

**B1-R — Bayesian backend is now PyMC + `pytensor.scan` (governing plan route A).**
This supersedes B1, which chose numba + emcee. B1 is left standing above as the record
of what was believed on 2026-08-21; it is no longer what the project does.

B1 rested on one factual premise: *no system C/C++ compiler is present on the target
machine, and requiring one damages reproducibility*. The researcher directed that the
compiler be installed. It was, so the premise no longer holds:

- **MinGW-w64 GCC 16.2.0**, UCRT runtime, x86_64, POSIX threads, SEH — the UCRT build is
  the one that matches CPython 3.12's own runtime.
- Source: WinLibs release `16.2.0posix-14.0.0-ucrt-r1`, the `.zip` asset, SHA-256
  `c1f52294597c0b73786b2a78eb5d176d89226d2f21875eab75e783a8b1cefcc4`, **verified against
  the publisher's published hash before extraction**.
- Installed to `C:\Users\micha\toolchains\mingw64`, which is on the user PATH. Portable and
  needs no administrator rights; uninstalling is deleting the directory and removing the
  PATH entry. MSVC Build Tools was rejected: it requires elevation, which this session
  does not have.

**A second fact undercut B1 independently of the compiler.** PyTensor 3.3.0's default
linker is `NumbaLinker`, not the C linker — PyTensor now compiles through numba, which
B1 had already accepted as needing no system compiler. Route A was therefore probably
open the whole time. B1's reasoning was sound on the evidence it had; the evidence was
incomplete. Recorded because the failure mode — rejecting an option on a premise never
tested — is worth not repeating.

**Route A was verified before adoption, not assumed.** A throwaway probe fitted
GARCH(1,1)-t to the actual training window (756 returns, percent scale), with
`alpha + beta < 1` enforced by construction via `beta = (1 - alpha) * delta`,
`delta ~ Beta(10, 2)`, and `nu` truncated below at 4:

| Quantity | Result |
|---|---|
| R-hat, all parameters | 1.00 |
| Divergences | 0 |
| ESS (bulk) | 485-809 |
| Sampling time | ~10 s for 2 chains x (500 tune + 500 draw) |
| mu, omega, alpha, beta, nu | 0.071, 0.059, 0.219, 0.713, 6.4 |

`alpha + beta = 0.93` and `nu = 6.4` are the persistence and tail thickness one expects
from daily SPY, which is a sanity check on the likelihood as much as on the sampler.
The ~30 s wall clock included ~20 s of one-time graph compilation that is **not** paid
per refit, so the 102-refit backtest is on the order of 20 minutes, not the 50 the naive
extrapolation suggested. This is a probe, not project code: `models.py` must still be
built and tested properly, and the priors above are a starting point that D4 has to
confirm.

**Consequences.**

- Route B (emcee) is demoted to the fallback the governing plan always intended it to be.
  `emcee` stays pinned; if a refit fails to sample it is the escape hatch.
- `numba` is downgraded 0.67.0 -> 0.66.0, `llvmlite` 0.49.0 -> 0.48.0, and `numpy`
  2.5.2 -> 2.4.6, all forced by `pytensor 3.3.0`'s ceilings. `llvmlite` is now pinned
  explicitly so it cannot drift away from `numba`.
- **The full suite was re-run after the downgrade: 56 pass.** The data layer is unaffected
  by the numpy change, which was the thing worth checking rather than assuming.
- Reproducibility now has a component that `pip install -r requirements.txt` does not
  supply. The compiler, its version, its source and its hash are therefore recorded here
  and in `requirements.txt`, and the README states the prerequisite.

---

### 1.7 Decided at Stage 0 EDA (2026-08-23)

**D9 — EDA diagnostics are computed on the training window as the primary result.**
The governing plan asks for ARCH-LM, Ljung-Box, ACF and ADF without saying over which
window. Choosing the full sample would have been the obvious reading and is what most
write-ups do.

Training window chosen instead. Model-class choice is a modelling decision: justifying
"we use GARCH" with a statistic computed over the 2,134 out-of-sample days would let the
evaluation period argue for the model that is later evaluated on it. It is a mild
look-ahead, but it is the exact species this project exists to detect in others, and the
project would have no standing to report a coverage failure elsewhere while committing
this one in its own motivation section.

Full-sample values are computed and printed too, labelled "DESCRIPTIVE CONTEXT ONLY".
They exist so a reader can confirm the two agree. They justify nothing, and nothing in
1.1 may be revised in response to either set. Enforced by
`test_training_window_results_ignore_out_of_sample_data`.

---

### 1.8 Decided at Stage 1 (2026-08-23)

**D10 -- Master scale convention: everything on the close-to-close return-variance
scale.** Directed by the researcher, and it supersedes the archived plan's withdrawn D2.

Every variance forecast in the project and the evaluation proxy live on one scale. The
Parkinson proxy is multiplied by a constant c = mean(r^2) / mean(sigma^2_P) estimated on
the warm-up sample (2014-01-02..2016-12-30, 756 observations) and frozen thereafter.

    c = 1.517318

The RW baseline forecasts yesterday's **scaled** Parkinson and derives its intervals from
that variance. EWMA and both GARCH models are already driven by squared returns and are
unchanged.

*Why it was needed.* Parkinson is built from the intraday high/low range and sees no part
of the overnight move, which for SPY is a large share of daily variance. Raw Parkinson
understates close-to-close variance by roughly a third, and the error pulls in two
directions at once:

| | RW (Parkinson-based) | EWMA / GARCH (r^2-based) |
|---|---|---|
| QLIKE against the raw proxy | units match -- unfairly advantaged | penalised by the 1.52 factor |
| Intervals against observed returns | ~23% too narrow -- over-breaches | correctly scaled |

Left alone, the RW baseline would have won the point-forecast table and failed the
calibration table, both for a units reason with nothing to do with forecasting or
uncertainty quantification -- contaminating precisely the comparison this project exists
to make. Separately, QLIKE's proxy-robustness (Patton 2011) is a statement about an
unbiased proxy; a systematically low proxy forfeits it.

*What the constant does not buy, recorded before any result depends on it.* Scaling
removes the systematic level error, which is first-order. It does **not** make the proxy
conditionally unbiased. The overnight share of variance moves around: within the warm-up
alone the ratio is 1.39 in 2014, 1.57 in 2015, 1.56 in 2016, and across quarters it
ranges from 1.18 to 1.94. Residual conditional bias remains and plausibly co-moves with
regime, since gaps dominate in stress.

Consequences, all of which are obligations on later stages:

- The report says the proxy is **approximately unbiased on average**. It must not claim
  proxy-robustness is restored.
- Stage 6 re-runs the QLIKE ranking on **raw** Parkinson, to show the ranking does not
  hinge on c.
- c is hard-coded as data.PROXY_SCALE_C rather than recomputed at import. A constant
  recomputed on the fly would move silently if the frame were ever rebuilt differently,
  and every published number would move with it without anything failing. A test
  recomputes it from the committed snapshot and asserts agreement.
- test_proxy_scale_cannot_see_out_of_sample_data asserts out-of-sample observations
  cannot move c.

**D11 -- Baselines use a zero predictive mean.** RiskMetrics convention, which the
governing plan names for EWMA. The GARCH models estimate mu instead. The asymmetry is
intentional: a baseline that quietly acquired a fitted mean would stop being a baseline.
The daily mean return is ~0.0005 against a standard deviation of ~0.011, so the effect on
coverage is small -- but it is a choice, not a default.

**D12 -- Both baselines get Gaussian predictive distributions.** The governing plan
specifies normal intervals for EWMA as the "naive UQ" baseline and is silent on the RW.
Both are given the same treatment so the two baselines differ only in how they estimate
variance, never in how they turn a variance into an interval.

**D13 -- Forecast output is long, not wide.** The governing plan asks for a tidy frame;
the archived plan's stub docstring specified a wide schema. Long wins: every downstream
consumer groups by model or by regime.

### 1.9 Decided at Stage 2 (2026-08-23)

**D14 -- The MLE is maximised on percent returns and converted back.**
`fit_garch_mle` multiplies its input by `FIT_SCALE = 100`, optimises, and converts the
estimates to the raw return scale before returning them. Nothing outside that function
sees the percent convention: `GarchParams`, `FrequentistFit.loglik`, the forecast table
and every figure are on the raw log-return scale throughout.

Daily SPY log returns have a standard deviation near 0.011, which puts `omega` at around
5e-6 -- below L-BFGS-B's default convergence tolerances, so the optimiser stops on a
parameter it has barely moved. On the percent scale `omega` is around 0.05 and the five
parameters sit within two orders of magnitude of each other. The Stage 3 feasibility
probe (1.6) found the same thing about the sampler's geometry and scaled to percent for
the same reason, so this keeps the two stages on one convention.

The rescaling is a change of variable, not a change of model: the two log-likelihoods
differ by exactly one factor of `FIT_SCALE` per observation, and
`test_loglik_is_scale_equivariant` asserts that identity rather than trusting it.

**D15 -- The normal-innovation variant runs through the full backtest as
`garch_mle_normal`.**
The governing plan owes a GARCH-normal versus GARCH-t comparison at 99% coverage at
Stage 6. Fitting it now costs one extra optimisation per refit -- 5 seconds across the
whole backtest against 51 for the t variant -- and saves re-entering the walk-forward
loop later, so it produces a complete forecast table over all 2,134 evaluation days.

It is an **ablation, not a competitor**. `backtest.HEADLINE_MODELS` excludes it, and the
evaluation layer must filter on that tuple rather than on whatever models happen to be
present in `forecasts.csv`. The lineup locked in 1.1 is four forecasters and this is not
a fifth.

`nu` is `NaN` for this variant rather than a sentinel value. Anything downstream that
forgets to branch on the innovation distribution therefore produces a NaN, which is
visible, instead of a plausible number computed under the wrong distribution.

**D16 -- A refit that does not converge produces no forecasts for its block.**
The 21 days served by a failed fit stay NaN in the forecast table and the failure is
written into its `RefitRecord` with the optimiser's message verbatim.

The alternative -- carrying the previous window's parameters forward -- was rejected. It
would hide a failed fit behind numbers that look entirely ordinary, and it is exactly
what `FrequentistFit.converged` exists to prevent. A gap in the forecast table is loud,
survives into the evaluation layer, and is a fact about the model rather than about the
reporting. `run_all.py --stage backtest` prints a `!!` line for any such refit.

As implemented, all 102 refits converge for both variants, so no gap exists in the
current run. `test_every_refit_converged` asserts that state and
`test_a_failed_refit_produces_no_forecasts_rather_than_stale_ones` forces a failure to
check the handling, because behaviour under failure should not depend on whether the
real data happens to trigger one.

**Multi-start ranks converged optima first.** `fit_garch_mle` runs a fixed 18-point
grid and takes the best objective value **among the starts that converged**, falling
back to the best of the rest only if none did -- in which case `converged` is False and
says so. Ranking purely on the objective value is what the first implementation did, and
it cost the 2022-09-06 refit: one abnormally terminated start had an objective a hair
below an ordinary success, displaced it, and blanked 21 days of the headline model's
forecasts. The grid is fixed and every start runs exactly once; no failure is re-run in
the hope of a better verdict.

**The size of that grid is load-bearing, and was measured rather than assumed.** The
18-point grid costs 51 seconds across the backtest against 2 seconds for a single start,
which is the kind of gap that invites trimming. Refitting all 102 windows under reduced
grids says otherwise: with one start per refit, 93 of the 102 land on a *different*
optimum, alpha moving by up to 0.049 and nu by up to 1.6; with six starts, 40 of 102 move
and alpha by up to 0.011. Since the reduced grids are subsets, every one of those
differences is a worse optimum. The likelihood has multiple local maxima on this data and
a small grid finds the wrong one about a third of the time, silently. The full grid
stays.

**Signature deviation: `plugin_predictive` returns a distribution object.**
The Stage 1 stub documented `plugin_predictive(params, h_next, quantile_levels) ->
(mean, quantiles)`. It now returns a `PredictiveDistribution` -- a `StudentTPredictive`,
or a `NormalPredictive` when `nu` is NaN. The harness needs `cdf` for the PIT as well as
`quantile` for the interval bounds, and taking both from one object is what stops them
being computed under different conventions. Same reasoning as `NormalPredictive`
existing at Stage 1 rather than a pair of free functions. Flagged here rather than
changed silently, following the precedent at 1.4.

**Schema change: `RefitRecord` gains `model` and the five parameter fields.**
One record per (refit, model): the baselines contribute one row per refit recording the
estimation window, each GARCH model one row per refit carrying `mu, omega, alpha, beta,
nu`, its log-likelihood and its convergence verdict. `refit_records.csv` is therefore
both the diagnostics log and the parameter audit trail, and
`figures.plot_parameter_stability` reads it directly so the plotted estimates and the
recorded ones cannot drift apart. A separate `garch_params.csv` was considered and
dropped: two artefacts carrying the same numbers is one more thing that can disagree.

Consequence for tests: the number of refits is now the number of distinct `refit_id`
values, not `len(records)`.

**Observed, not decided: persistence sits close to the stationarity boundary.**
`alpha + beta` for the t variant runs from 0.950 at the warm-up fit to 0.99998 at its
highest, and is at or above 0.999 in 16 of the 102 refits. Every fit is admissible and
converged, so nothing here is invalid, but the constraint is very nearly binding over
the later sample -- near-IGARCH behaviour, which is what a long daily equity sample
containing 2020 and 2022 tends to produce. Recorded now because it bears on Stage 3:
the probe's `beta = (1 - alpha) * delta` construction enforces `alpha + beta < 1` by
construction, so the posterior will press against the same boundary, and a prior that
pushes back hard on it would be making a modelling choice that should be visible rather
than incidental.
---

## 2. Changelog

### Stage 0 — Inspection and risk review (2026-08-21)

No files created. Findings:

- Working directory empty; no Git repository in it or any parent.
- Python 3.12.10, pip 25.0.1, no packages installed beyond pip, no virtual environment.
- No `cl.exe` and no `gcc` on PATH — no system C/C++ toolchain. This drove decision B1.

Raised blocking items B1–B3 and open items D1–D5. B1–B3 and D5 resolved as recorded
above; D1–D4 remain open.

### Stage 1 — Repository scaffold (2026-08-21)

Created a complete but entirely unimplemented skeleton. **No research logic, no numbers.**

- `git init -b main`, first commit.
- Added `.gitignore` (note: `data/raw/` deliberately tracked, per B3).
- Added `requirements.txt`. Direct dependencies pinned; the full set was verified to
  resolve together via a pip resolver dry-run on Python 3.12 / Windows (44 packages, no
  conflicts). Confirmed numba 0.67.0 requires `numpy<2.6`, so numpy 2.5.2 is within
  range; this was the binding compatibility constraint.
- Added `README.md` documenting scope, locked decisions, the quantities that must not be
  conflated, and the Parkinson scale caveat.
- Added `run_all.py` — argparse skeleton dispatching to four stages, each raising
  `NotImplementedError`.
- Added `src/` package: `data.py`, `models.py`, `backtest.py`, `evaluation.py`,
  `bootstrap.py`. Every module carries a docstring and typed, documented function
  signatures whose bodies raise `NotImplementedError`. The intent is that the interfaces
  and the data flow between modules are reviewable *before* any implementation exists.
- Added `tests/test_smoke.py` — asserts the package imports and the directory layout is
  present. It does **not** test behaviour, because there is none.
- Added `report/report.md` with section headings, explicitly marked unpopulated.
- Directory placeholders for `data/raw`, `data/processed`, `figures`, `notebooks`.

Known consequence: the Stage 1 signatures will not be perfect and some will be revised in
later stages. Every signature change will be called out explicitly rather than made
quietly.

### Stage 1a — Implementation plan (2026-08-22)

Added `docs/implementation_plan.md`: a function-by-function specification of Stages 2-8,
each with its formula, its dominant failure mode, its tests, and its acceptance
criteria. No code, no design change.

Note: this adds a `docs/` directory, which is not in the locked target repository
structure. Flagged rather than assumed; the file can be moved to the repository root or
under `report/` on request.

Two decisions surfaced while writing the specification and are recorded as open:

**D6 — Number of posterior draws carried through the backtest. Blocks Stages 3-4.**
Default sampler settings retain 32 x (3000-1000)/5 = 12,800 draws. Carrying that many
variance paths across ~2,140 daily forecast steps is wasteful. Proposal: thin to a fixed
2,000 draws for the predictive stage. A compute/accuracy tradeoff with no modelling
content, but it should be recorded rather than left to accident. **Not yet decided.**

**D7 — Reference distribution for the baselines' predictive intervals. Blocks Stage 4.**
The two baselines produce a variance forecast but no innovation distribution, yet they
need intervals to be comparable on coverage. Proposal: a Gaussian reference. Fitting a
Student-t to the baselines would quietly promote them from baselines to competitors and
would muddy what the GARCH comparison is actually demonstrating. **Not yet decided.**

Also recorded in the plan, as implementation hazards rather than decisions:
- The `-0.5*ln(h_t)` Jacobian term in the t log-likelihood. Omitting it still optimises
  and still looks plausible, but every variance estimate is wrong.
- The `sqrt((nu-2)/nu)` standardisation factor in predictive quantiles. Dropping it
  inflates intervals and would bias the central comparison toward the Bayesian model.
- `h_next` must vary across posterior draws. A single shared value collapses the
  posterior predictive onto the plug-in and destroys the research question while
  producing entirely plausible output. Guarded by a dedicated test.
- Paired resampling in the bootstrap. Independent resampling inflates intervals and
  fails in the safe-looking direction.

### Stage 2 — `src/data.py` (2026-08-22)

First stage with real logic. Implemented the whole of `src/data.py`, the `data` stage of
`run_all.py`, and `tests/test_data.py`. **Still no model, no forecast, no result.**

Downloaded and committed the raw snapshot:

| File | Rows | Range | Purpose |
|---|---|---|---|
| `data/raw/SPY.csv` | 2911 | 2013-12-02 .. 2025-06-30 | prices, high/low |
| `data/raw/VIX.csv` | 2911 | 2013-12-02 .. 2025-06-30 | regime labels |
| `data/raw/manifest.json` | — | — | SHA-256 + row counts + ranges |

Analysis frame: **2890 rows**, 2014-01-02 .. 2025-06-30. Train **756 rows**
(2014-01-02 .. 2016-12-30); out-of-sample **2134 rows** (2017-01-03 .. 2025-06-30). The
OOS count implies `ceil(2134 / 21) = 102` refits, matching the estimate in decision B1.

Regime counts on the lagged VIX close: calm 1198, normal 1317, stressed 375. The
stressed regime is 13% of the sample. That is enough for a coverage statement at the
90% and 95% levels and thin for the 99% level and for the 99% VaR, where the expected
exceedance count in the stressed subsample is single digits. Recorded now so the
regime tables are read with the right expectations rather than discovering it at
Stage 5; `summarise_by_regime` must report `n` for exactly this reason.

Data quality findings, printed in full by `run_all.py --stage data`:

- SPY and ^VIX calendars agree on **every** date in the sample. No rows dropped.
- No date has `high <= low`; the Parkinson proxy is defined everywhere.
- Six dates have an exactly zero log return: 2016-04-22, 2017-01-10, 2018-05-08,
  2018-10-08, 2022-08-11, 2023-07-21. All are ordinary flat closes on liquid days
  rather than stale-price artefacts, and all are retained. Noted because a zero return
  sits at the centre of every predictive interval and will never register as a breach.

Decisions recorded above as D8, plus a flagged signature change to
`build_analysis_frame` and a documented redefinition of `missing_dates_spy`.

**Tests: 55 pass** (20 structural, 35 new). `tests/test_smoke.py::test_stages_are_stubs`
replaced by a parametrised `test_unimplemented_stages_still_raise` over the three
remaining stub stages, so the scaffold's story stays accurate as stages land.

The look-ahead tests were **mutation-checked** rather than merely observed to pass:

1. Removing the lag from `assign_vix_regime` — caught by two tests.
2. Demeaning `log_return` by a full-sample mean — *initially not caught*. The synthetic
   SPY fixture used a constant-growth price path, so every daily return was identical
   and a full-sample statistic was numerically invisible. The fixture now uses a seeded,
   genuinely uneven return path, and the same mutation is caught at all four values of
   `t`. Recorded because a look-ahead test that cannot fail is worse than no test: it
   reports safety it never checked. The same check must be run against the Stage 4
   master look-ahead test before its result is trusted.

Acceptance criteria for Stage 2, all met: one download then cache-only reruns with no
network access; manifest written and verified before the frame is built; the quality
report printed in full with no date elided; frame saved to
`data/processed/analysis_frame.csv`; raw CSVs and manifest committed.

**Reproducibility bug found and fixed at Stage 2: line-ending conversion silently broke
the committed snapshot.**

This machine has `core.autocrlf=true` and the repository had no `.gitattributes`. Git
therefore rewrites LF to CRLF on **checkout**. The raw CSVs are written with LF and are
hashed byte-for-byte by `manifest.json`, so a fresh clone would receive different bytes,
every SHA-256 would mismatch, and `load_raw` would raise — correctly, but the effect is
that the pipeline could not be run from a clone at all. The entire point of decision B3
was defeated by a Git default.

The failure is invisible in the authoring worktree, because those files were produced by
a write rather than by a checkout. It was confirmed by actually cloning the repository:
both CSVs arrived with 2,912 CRLF line endings and both hashes failed.

Fix: added `.gitattributes` marking `data/raw/*.csv` and `data/raw/*.json` as `-text`,
which freezes their bytes on every platform. Re-verified by cloning again: 0 CRLF, both
hashes match. Deliberately scoped to `data/raw/` only — normalising the whole repository's
line endings is a separate change and would rewrite every source file.

Guarded by `test_snapshot_bytes_are_lf_only`, alongside
`test_committed_snapshot_matches_its_manifest` which would fail outright in an affected
clone. **Tests: 56 pass.**

General lesson, applicable to the remaining stages: an integrity check that has never
been observed to fail is not evidence that it works. Both the look-ahead tests and the
hash verification were checked here by deliberately breaking the thing they guard.


---

### Stage 1b — Plan switch (2026-08-23)

`docs/project1-implementation-plan.md` replaces `docs/implementation_plan.md` as the
governing plan. The old file was retired, and was then deleted from the working tree
rather than kept as an archived copy; it remains recoverable from git history. Rationale,
the full stage re-keying, and the withdrawal of D1/D2/D3/D6/D7 are recorded in §1.5
above.

No code changed. `src/data.py` is untouched and its tests still pass — the locked
decisions are identical across both documents, so there was nothing in the data layer for
the switch to invalidate. README and the Stage-2-means-`data.py` comments in `run_all.py`,
`src/data.py` and `tests/test_data.py` were re-keyed to the governing plan's numbering,
under which the data layer is part of **Stage 0**.

The claim "Stage 2 complete" is therefore now false under the current numbering and has
been removed everywhere. The true state is: Stage 0 partially complete — data layer
built and verified, EDA not started.


---

### Stage 1c — C++ toolchain installed, route A adopted (2026-08-23)

At the researcher's direction, a C/C++ compiler was installed so the governing plan's
preferred Bayesian route (PyMC) could be used. Full reasoning, provenance and the
verification probe are in §1.6 above (decision B1-R).

Changed: `requirements.txt` (compiler prerequisite documented; `pymc`, `pytensor`,
`arviz` and an explicit `llvmlite` pin added; `numba`, `numpy` and `llvmlite` re-pinned
downward to pytensor's ceilings; `emcee` re-labelled as the route B fallback) and
`README.md` (prerequisite plus install instructions).

No source file changed. Tests: 56 pass, before and after the dependency downgrade.

Still true, and worth stating plainly because the environment work can look like
progress: no model, no forecast and no result exists. Stage 0's EDA is next.


---

### Stage 0 EDA — `src/eda.py`, `src/figures.py` (2026-08-23)

The premise of the project, established by test rather than assumed. **Stage 0 is now
complete** under the governing plan's numbering.

New: `src/eda.py` (Engle ARCH-LM, Ljung-Box, ADF, ACF of squared returns, return
summary), `src/figures.py` (house style plus the four Stage 0 figures),
`run_all.py --stage eda`, `notebooks/01_eda.ipynb`, `tests/test_eda.py`.
`statsmodels==0.14.6` pinned as a direct dependency — it was present only as an `arch`
dependency and is now imported directly. **Tests: 82 pass** (56 + 26 new).

**Training-window results, which are the ones that license the design:**

| Diagnostic | Result |
|---|---|
| ARCH-LM, lags 5 / 10 / 22 | p = 1.5e-23 / 1.8e-21 / 1.0e-17 — rejects at every lag |
| Ljung-Box, squared returns, 5 / 10 / 22 | p = 3.4e-44 / 1.1e-51 / 1.8e-47 — rejects |
| Ljung-Box, raw returns, 5 / 10 / 22 | p = 0.45 / 0.62 / 0.26 — **no rejection** |
| ADF on returns | stat -27.39, p < 1e-300 — stationary |
| Excess kurtosis | 2.46 |
| Annualised vol | 13.36% |

The contrast between rows 2 and 3 is the finding, not row 2 alone: abundant structure in
the second moment, none detectable in the first. That is precisely the regime in which a
conditional-variance model earns its place. Excess kurtosis of 2.46 is the independent
empirical warrant for the Student-t innovation in 1.1.

**One full-sample discrepancy, recorded because it is a genuine limitation.**
Over the full sample, Ljung-Box on **raw** returns *rejects* (p = 7.1e-14 at 5 lags),
where on the training window it does not. Short-horizon return autocorrelation rises in
crises and the out-of-sample period contains several. Full-sample excess kurtosis is
14.71 against 2.46 in training, driven by the same episodes.

This does not bias the project's central comparison: the constant mean of decision D5 is
applied identically to all four forecasters, so it cannot favour one over another. It
does mean the constant-mean assumption is a real simplification over the evaluation
period. **It belongs in the report's limitations section**, and is written down now so
that it is reported as a known property rather than discovered late and quietly dropped.

**Test design note.** Every diagnostic is tested twice: on data that should trigger it
(simulated GARCH, AR(1), random walk) and on data that should not (i.i.d. normal).
Testing only the positive case cannot distinguish a working test from a function that
always rejects — and "always rejects" would have handed this project a fabricated
justification for its own model class. `test_arch_lm_would_be_inflated_without_demeaning`
exists for the same reason: it proves the demeaning invariance test is load-bearing
rather than vacuous.

Still true: no model, no forecast, no result. Stage 1 (backtest harness plus the
RW-in-vol and EWMA baselines) is next, and the governing plan is explicit that the
harness is built before any interesting model, because look-ahead bugs live there.


---

### Stage 1 -- backtest harness and baselines (2026-08-23)

The walk-forward loop and the two baselines. **The look-ahead audit passes**, which is
the precondition the governing plan sets before any further stage may proceed.

New: src/backtest.py implemented (was stubs), NormalPredictive / Forecast /
PredictiveDistribution and both baselines in src/models.py, data.compute_proxy_scale /
data.scale_proxy / data.PROXY_SCALE_C, figures.plot_forecasts_vs_realised,
run_all.py --stage backtest, tests/test_backtest.py (22 tests), tests/test_models.py
(17 tests), and 8 proxy-scale tests in tests/test_data.py. **Tests: 127 pass.**

**Output.** 2,134 rows per model over 2017-01-03..2025-06-30, 102 refits at the 21-day
cadence, no gaps in any forecast or PIT column. Mean annualised volatility 17.71% (RW)
and 18.86% (EWMA) against a full-sample realised 17.61%.

**Stub debt cleared.** The stubs predated the plan switch and B1-R. BacktestConfig
carried emcee fields (n_walkers, n_steps, n_burn, thin); these are now NUTS fields
(draws, tune, chains, target_accept), inert until Stage 3. RefitRecord likewise moved
from acceptance-fraction/autocorrelation to R-hat, ESS and divergence counts. The "open
item D1 blocks this module" note is gone: the governing plan fixes an expanding window in
its locked-decisions table, so EstimationWindow.ROLLING now raises rather than silently
producing a result nobody chose.

**The refit cadence is a no-op at this stage** and that is deliberate. Neither baseline
estimates anything -- EWMA's decay is fixed at 0.94 and the random walk has no parameters
-- so the scaffolding is exercised for the first time by GARCH at Stage 2. Building it
against models with no moving parts means any failure it shows is its own.

**The look-ahead audit.** The master test corrupts every observation from date t onward,
re-runs, and asserts the forecasts are bit-identical before t. Run at three dates chosen
as the worst available cases: Volmageddon (2018-02-05), the largest COVID drawdown day
(2020-03-16), and a 2022 selloff (2022-06-13).

One point of precision, because getting it wrong is how a look-ahead test quietly stops
testing anything. Corrupting from t leaves the *forecast* columns unchanged for dates up
to **and including** t, since the forecast for t is built from data through t-1. It does
**not** leave the *evaluation* columns at t unchanged: log_return and the pit derived
from it are functions of day t itself. The test asserts the first over <= t and the
second over < t. This is the same distinction recorded at 1.4 for the Stage 0 frame test.

test_corruption_actually_changes_the_future accompanies it, because all of the above
would also hold for a backtest that ignored its input entirely.

**The acceptance plot** (figures/05_baseline_forecasts_covid.png, Nov 2019 - Jun 2020) is
the check that no summary statistic would have caught. An EWMA forecast is a weighted
average of *past* squared returns, so it must lag a spike on the way in and overshoot on
the way out. Observed: realised volatility peaks at ~1.03 annualised on 16 March 2020
while EWMA is at ~0.70 that day, reaching its own peak of 0.81 about a week later; on the
way out realised falls to ~0.2 by early May while EWMA is still at 0.6 in late April and
~0.30 in June against realised ~0.15. Both series are flat at ~0.10 through mid-February
and move only after the large returns arrive. Had the recursion been reading the current
day's return, the EWMA line would sit on top of the realised peak instead of trailing it.

Next: Stage 2, frequentist GARCH(1,1)-t, which joins this loop without the loop changing.

### Stage 2 -- frequentist GARCH(1,1)-t (2026-08-23)

The first model in this project that estimates anything, and therefore the first real
exercise of the refit machinery built at Stage 1. **The look-ahead audit still passes
with both GARCH models in the loop**, which is the condition the governing plan sets for
proceeding.

New: the GARCH half of src/models.py implemented (was stubs) -- GarchParams methods,
garch11_filter, garch11_t_loglik, garch11_normal_loglik, backcast_initial_variance,
standardised_residuals, fit_garch_mle, StudentTPredictive, plugin_predictive;
backtest.build_garch_paths and the model registry; RefitRecord extended;
eda.series_acf factored out of eda.squared_return_acf;
figures.plot_garch_residual_diagnostics and figures.plot_parameter_stability;
run_all.py --stage backtest extended; pytest.ini added to register the `slow` marker.
**Tests: 188 pass** (test_models.py 68, up from 17; test_backtest.py 32, up from 22;
unchanged elsewhere).

Still stubbed and belonging to Stage 3: log_prior, log_posterior,
sample_garch_posterior, posterior_predictive, and all of evaluation.py and bootstrap.py.

**Output.** Four forecast tracks, 2,134 rows each, no gaps in any forecast or PIT
column. 102 refits per GARCH variant, all converged. Mean annualised volatility: 17.71%
(RW), 18.86% (EWMA), 19.37% (GARCH-t), 18.54% (GARCH-normal). Fitting cost 51s for the t
variant and 5s for the normal one, so a full backtest run is about a minute.

**The warm-up fit sanity-checks the likelihood against an independent estimate.**
On the 756-observation warm-up window: mu 0.000704, omega 4.81e-6, alpha 0.2150, beta
0.7353, alpha+beta 0.9503, nu 5.86. The Stage 3 PyMC probe (1.6) fitted the same window
under priors and got alpha 0.219, beta 0.713, nu 6.4. Two implementations that share no
code -- a hand-written NumPy likelihood maximised by L-BFGS-B, and a pytensor.scan graph
sampled by NUTS -- agreeing to that tolerance is stronger evidence that the likelihood is
right than any single number either produced.

**Residual diagnostics on the warm-up fit.** Ljung-Box on standardised residuals gives
p = 0.50 / 0.57 / 0.29 at 5 / 10 / 22 lags; on their squares, p = 0.62 / 0.88 / 0.98.
The second row is the one that matters: the same test on raw squared returns rejected at
p = 3.4e-44 (1.7), so the variance equation has absorbed the clustering it was fitted to
absorb. The first row says the constant mean is adequate in sample, which is consistent
with the raw-return Ljung-Box at 1.7 and does not disturb the limitation recorded there
about the evaluation period.

The QQ plot (figures/06) shows the empirical residuals with *thinner* tails than the
fitted t at both ends. Not a defect -- the MLE trades tail fit against the centre across
the whole distribution -- but it is the direction that matters for this project, since a
predictive whose tails are fatter than the data warrants will over-cover at 99%. Whether
it does is a Stage 4 question and is left to Stage 4.

**Parameter stability** (figures/07). alpha jumps at COVID and decays afterwards; beta
rises steadily as the expanding window lengthens; nu falls from 5.9 to about 4.65 by
late 2017 and climbs back above 5.8 by 2024. Persistence is discussed at 1.9.

**The two-cadence rule is now load-bearing.** Parameters are re-estimated at each of the
102 refit dates on the expanding window; between refits they are held fixed while
garch11_filter keeps advancing the variance recursion daily. Because h[t] is defined as
the conditional variance of returns[t] given information through t-1, the daily update
falls out of the indexing rather than needing a second loop.
test_variance_moves_daily_while_parameters_are_held_fixed asserts both halves -- refit_id
and the predictive mean constant across a block, variance taking 21 distinct values
within that same block. Freezing h between refits as well would have produced a
complete, plausible forecast table built on variances up to 21 days stale.

**Compiled inner loop.** The variance recursion is numba-compiled (B1 always specified
this). The multi-start MLE evaluates it on the order of 1e6 times across the backtest;
in pure Python that is tens of minutes, compiled it is under a minute.

**Test-suite cost and the `slow` marker.** A full run now takes about a minute, and the
look-ahead audit needs several of them. Two changes, neither of which reduces coverage:
the uncorrupted run became a module-scoped fixture instead of being recomputed in each
test that needs it, and the 15 tests that each require their *own* run -- the audit's
three corrupted re-runs, the vacuity check, the determinism test -- are marked `slow`.
`pytest` takes 10 minutes and runs everything; `pytest -m "not slow"` takes 1 minute,
which is the shared fixture's own cost rather than seconds. The marker is a convenience
for development and never what a default run applies: the audit is on the never-cut
list.

**What the audit now covers that it could not before.** Until this stage both models
were parameter-free, so the master test could not have caught a refit that estimated on
data it should not have seen -- there was no estimation to catch. Every route by which
the future could reach a forecast is now inside its scope: the estimation window
(estimation_slice ends strictly before the refit date), the backcast seed (h0 from the
estimation window alone), and the daily filter.

Next: Stage 3, Bayesian GARCH(1,1)-t. D4 -- the priors -- is still open and must be
frozen in this log before any Bayesian out-of-sample number is computed.
