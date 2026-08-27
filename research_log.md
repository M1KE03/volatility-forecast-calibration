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
### 1.10 Decided at Stage 3 (2026-08-26)

**D4 resolved -- the priors, frozen before any Bayesian out-of-sample number exists.**
On the percent return scale (D14), with `beta` a deterministic transform rather than a
sampled parameter:

| Parameter | Prior | Reasoning |
|---|---|---|
| `mu` | `Normal(0, 1)` | Daily SPY returns have a mean near 0.05 and a standard deviation near 1.1 on this scale. A unit-scale prior is weakly informative: wide enough not to bind, narrow enough to exclude absurdities. |
| `omega` | `HalfNormal(1)` | Positivity by construction. The MLE puts `omega` near 0.05, so a unit scale is diffuse by a factor of ~20 and does not bind. |
| `alpha` | `Beta(2, 10)` | Support is exactly `[0, 1]`, prior mean 0.167 against an MLE range of 0.181-0.241 across the 102 refits. Mildly informative and centred where the data lives. |
| `delta` | `Beta(3, 1)` | `beta = (1 - alpha) * delta`, so `alpha + beta < 1` holds **by construction** and stationarity never has to be rejected by the sampler. See the deviation note below. |
| `nu` | `Exponential(1/10)` truncated below at 4 | Prior mean ~14, i.e. leaning toward near-normal tails, so fat tails have to be earned from the data. The truncation at 4 keeps the kurtosis finite (the model-specification constraint stated in `models.py`), not merely the variance. |

Support constraints follow from the parameterisation rather than being imposed on top of
it: `omega > 0` from `HalfNormal`, `alpha` and `delta` in `[0, 1]` from `Beta`,
`beta >= 0` and `alpha + beta < 1` from the transform, `nu > 4` from the truncation. No
rejection sampling and no hand-written bounds, which is the reason this parameterisation
was chosen over sampling `beta` directly.

**Deviation from the specification recorded at 1.6: `delta ~ Beta(3, 1)`, not
`Beta(10, 2)`.** The feasibility probe used `Beta(10, 2)` and 1.6 called it a starting
point that D4 had to confirm. It does not survive confirmation, for a structural reason
fixed before the comparison was run: a `Beta(a, b)` density vanishes at 1 whenever
`b > 1`, so `Beta(10, 2)` places **zero** prior density at `delta = 1` -- exactly the
near-IGARCH boundary that 1.9 recorded the data pressing against, with `alpha + beta`
at or above 0.999 in 16 of the 102 MLE refits. A prior that vanishes where the
likelihood concentrates is making a modelling choice, and 1.9 asked for that choice to be
visible rather than incidental. `Beta(3, 1)` has density 3 at the boundary and is
increasing toward it; `Beta(1, 1)` has density 1 there.

The magnitude was then measured rather than assumed, at 4 chains x (1,000 tune + 1,000
draw):

| Window | `delta` prior | posterior `alpha+beta` | 90% interval | P(>0.99) | divergences |
|---|---|---|---|---|---|
| warm-up, n=756 (MLE 0.95034) | `Beta(10,2)` | 0.93091 | [0.8777, 0.9746] | 0.006 | 0 |
| | `Beta(3,1)` | 0.93962 | [0.8800, 0.9872] | 0.038 | 0 |
| | `Beta(1,1)` | 0.93586 | [0.8748, 0.9858] | 0.032 | 0 |
| full sample, n=2,890 (MLE 0.99536) | `Beta(10,2)` | 0.98356 | [0.9688, 0.9956] | 0.234 | 0 |
| | `Beta(3,1)` | 0.98876 | [0.9739, 0.9990] | 0.500 | 0 |
| | `Beta(1,1)` | 0.98855 | [0.9735, 0.9990] | 0.497 | 1 |

Three things this shows. The prior does bind, but only on the long window -- at n=756 the
three answers differ by 0.009, and at n=2,890 `Beta(10,2)` sits 0.005 below the other
two, placing the MLE at roughly its 95th percentile. `Beta(3,1)` and `Beta(1,1)` agree to
within Monte Carlo noise, which is what identifies that gap as the prior binding rather
than as chance. And `Beta(1,1)` produced a divergence where `Beta(3,1)` produced none, so
the flatter prior buys nothing and costs sampler behaviour.

**The stakes are small and are recorded as small.** A 0.005 difference in `alpha + beta`
moves a one-day-ahead variance forecast by about 0.5%, hence interval widths by about
0.25%. This choice will not decide the headline, and this entry should not later be read
as implying it did. The choice was made on the structural argument; the measurement
establishes magnitude, not the winner.

**Disclosure: the full-sample rows above were computed on data that includes the
evaluation period.** The rule in 1.3 is that priors are frozen before any out-of-sample
*number* is computed, and no forecast, interval, coverage, loss or VaR quantity was
produced -- these are posterior parameter summaries. The information they carry, that
persistence runs to the stationarity boundary on long windows, was already on this log's
record at 1.9 from the frequentist refits before Stage 3 began, and the selection
criterion was structural and fixed in advance. It is recorded anyway rather than passed
over, because a project whose subject is look-ahead does not get to decide for itself
which of its own peeks were harmless. **The report must state that the prior on `delta`
was selected on a structural criterion, with full-sample posterior summaries consulted
for magnitude.** Prior sensitivity across all three candidates belongs in
`03_robustness`, where the comparison can be redone on evaluation-window forecasts.

**D17 -- Sampler settings: 4 chains x (1,000 tune + 1,000 draw), `target_accept = 0.9`,
chains run in parallel.**
Four chains rather than two because split-R-hat is the convergence criterion and PyMC
itself warns that fewer than four makes it unreliable. Two thousand retained draws per
refit rather than the probe's 1,000 because `ess_tail` -- not `ess_bulk` -- is what the
99% predictive quantiles are built from, and the 99% level is where this project's
finding lives. Measured: `ess_tail` at or above 1,319 on the full window under the
adopted prior, no divergences, R-hat at most 1.002.

Cost was measured, not extrapolated: 30.4 s per fit at n=756 and 83.6 s at n=2,890,
near-linear in window length, so **95-100 minutes for a full Bayesian backtest** across
the 102 expanding refits. The alternative of 2 chains x (500 + 500) was measured at 17.0 s
on the warm-up window, about 40 minutes end to end, and was rejected: a factor of two in
wall clock does not justify a weaker convergence diagnostic on the stage the project is
named after. Note that 1.6's "on the order of 20 minutes" was an extrapolation from a
warm-up-sized window at the smaller setting and understated the real cost by roughly five
times; windows grow to 3.8x the warm-up length by the end of the backtest.

**D18 -- 2,000 of the 4,000 posterior draws are carried into the predictive stage.**
The withdrawn D6 proposed thinning to a fixed 2,000 from a much larger emcee chain. Under
NUTS the arithmetic is different: 4 chains x 1,000 post-warm-up draws is 4,000, of which
the predictive stage carries every second draw -- deterministically, not as a random
subsample. NUTS draws are near-independent (`ess_bulk` runs 1,300-2,600 out of 4,000), so
thinning discards real information rather than redundancy; the only reason to do it is the
per-day cost of the mixture quantile solve, which is linear in draw count. 2,000 is the
compromise, and it is a compute decision with no modelling content. If the daily
predictive proves cheaper than expected, carrying all 4,000 is strictly better and needs
no re-freezing.

**D19 -- A Bayesian refit that fails its diagnostics produces no forecasts for its
block.** D16 extended to the Bayesian track, with the same reasoning and the same
consequence: the 21 days served by a failed fit stay NaN and the failure is written into
its record verbatim. Convergence means **all** of: R-hat at most 1.01 on every sampled
parameter, zero divergent transitions, and `ess_tail` at least 400 on every sampled
parameter.

Zero divergences rather than a small tolerance, because a divergence is not noise -- it
says the sampler failed to explore part of the posterior geometry, so the draws are not a
sample from the target and more of them do not fix it. The thresholds are fixed now,
before any refit has been run, for the reason handoff 8.6 gives: a diagnostic threshold
loosened after it fires is not a diagnostic. **No failed refit is re-run with a different
seed in the hope of a better verdict**, which is D16's multi-start rule in another
costume. All three candidate priors cleared these thresholds on both the warm-up and the
full window, so the criterion is not expected to bind -- but behaviour under failure must
not depend on whether the real data happens to trigger one, and a forced-failure test
alongside `test_a_failed_refit_produces_no_forecasts_rather_than_stale_ones` will cover
it.

**D20 -- The look-ahead audit runs the Bayesian track at a reduced draw count.**
At the adopted settings a full Bayesian backtest is ~100 minutes, and the audit runs the
backtest once clean plus once per corruption date, which would put a full `pytest` run
above five hours. Problems-and-solutions 35 rejects buying audit speed by truncating the
sample or dropping a model, and that rejection stands. This is a different trade, and the
distinction is the whole justification: **draw count controls Monte Carlo precision, not
which data reaches a fit.** Every surface the audit exists to check -- the estimation
slice ending strictly before the refit date, `h0` backcast from the estimation window
alone, the daily filter -- is exercised over all 102 refit dates, in the same code path,
at any draw count. The audit's assertions are exact equalities under a fixed seed, and a
fixed seed is as exact at 100 draws as at 2,000.

What this does **not** cover is a bug that appears only at production draw counts. Nothing
in the forecast path is plausibly draw-count dependent in that way, but the claim is
recorded so it can be checked rather than assumed, and the headline artefacts are
generated at full settings regardless.

---

### 1.11 Decided during Stage 3 implementation (2026-08-26)

Everything here was settled after D4 and D17-D20 were frozen and before any Bayesian
forecast reached the evaluation layer. Two entries record a measurement that contradicts
an earlier decision's premise; per the append-only rule they supersede rather than edit
it.

**D21 -- The PyMC graph observes every return, not all but the first.** The 1.6
feasibility probe wrote `observed=r[1:]`, dropping the first observation because the
scan produces `h[1:]`. `garch11_t_loglik` does not: it seeds `h[0] = h0` and observes the
whole window. The project's central design constraint is that models 3 and 4 share one
likelihood, so the graph follows the NumPy function rather than the probe, and
`test_the_pymc_graph_and_the_numpy_log_posterior_agree` checks the two agree at a fixed
parameter vector -- to within `log(1 - alpha)`, which is the deliberate difference
between a prior stated over `delta` (PyMC samples it) and one stated over `beta`
(`log_prior` states it, with the change-of-variable Jacobian that the emcee fallback
would need).

The check found nothing wrong, which is the point of running it: "same likelihood" was
until now a claim about two implementations that nothing verified.

**D22 -- The `variance` column of a Bayesian forecast row is the posterior mean of
`h`.** Not the variance of the predictive distribution, which additionally carries the
spread of `mu` across draws. The column feeds QLIKE and MSE against the proxy, where the
quantity wanted is an estimate of tomorrow's conditional variance, and the posterior mean
is that estimate under squared-error loss. It is also the quantity directly comparable
with the frequentist track's plug-in `h`. The predictive's own spread is what the
interval columns report, and reporting it twice under two names would invite exactly the
double-count this project exists to be careful about.

**D23 -- Each refit samples under `mcmc_seed + refit_id`.** One RNG stream per refit
rather than one per backtest, so no two refits share a chain's randomness. Fixed before
the run and a function of the refit index alone, so it is not a retry: D16's rule that no
failed fit is re-run under a different seed (extended to this track by D19) is untouched.

**D24 -- The backtest runs as two tracks, `frequentist` and `bayes`, whose forecast
tables are merged.** The frequentist track is 306 optimiser fits and takes about a
minute; the Bayesian track is 102 NUTS fits and takes about 95. Splitting them means
re-running the cheap one does not re-run the expensive one. Each track writes
`forecasts_<track>.csv`, `refit_records_<track>.csv` and `backtest_config_<track>.json`;
`forecasts.csv` is rebuilt from whichever partials exist, so the evaluation layer still
reads one table and cannot read a table that was never written.

The merge **compares the configs** rather than assuming they match, ignoring only the
sampler fields, which are the Bayesian track's business alone. Merging tables computed
under different proxy scales or refit cadences would produce a forecast file whose rows
answer different questions and nothing downstream could detect it.

`RefitRecord.mle_converged` and `.mle_message` are renamed to `.converged` and
`.message` for the same reason: D19 applies D16's rule to both tracks with the same
consequence, so the rule reads one field regardless of which estimator produced the
verdict. `mle_loglik` keeps its name and is NaN on Bayesian rows -- a posterior has no
maximised log-likelihood, and giving that slot a nearby quantity would invite a
comparison across tracks that is not one.

**The Bayesian predictive is not wider at every level, and the handoff's sanity check is
wrong to expect it.** Measured on a dispersed posterior against the plug-in at its mean:
width ratios of 0.995 at 90%, 1.002 at 95%, 1.017 at 99% and 1.032 at 99.8%. A scale
mixture holding average variance fixed is leptokurtic against the single distribution at
that average -- more peaked in the middle, heavier in the tails, because total variance
is conserved. The crossover sits between 90% and 95%.

This matters beyond bookkeeping: the handoff instructs the reader who meets a narrower
interval to hunt for the shared-`h_next` bug, and the natural repair for a bug that is
not there would manufacture the widening this project set out to measure. The mixture
quantiles were verified against four million draws from the same mixture taken by a
different route, agreeing to four significant figures, before the claim rather than the
code was changed. Recorded at problems-and-solutions 38 and pinned by two tests, one on
each side of the crossover.

**D20's premise is contradicted by measurement; the audit's Bayesian coverage is
reopened.** D20 authorised running the Bayesian track in the look-ahead audit at a
reduced draw count, on the reasoning that draw count controls Monte Carlo precision
rather than which data reaches a fit. That reasoning stands. What does not is the
assumption that draws are what the audit pays for. Measured: a refit at the frozen
settings costs 29s at n=756 and 83s at n=2,890; at 2 chains x (50 tune + 25 draw) it
costs 9s and 17s. Cutting the sampling work by 96% cuts the wall clock by about 70%,
because the floor is compiling the gradient of the `scan` recursion, and the returns sit
in that graph as a constant, so codegen scales with the window. Six backtest runs x 102
refits x ~12s puts a default `pytest` near two hours.

A `pytensor.shared` variable to keep the data out of the graph was tried and is worse --
20-29s per refit, the static shape being unknown defeats the optimiser -- and reverted.

`AUDITED_MODELS` in `tests/test_backtest.py` currently holds the frequentist track alone
and carries this open item in its docstring, where a reader meets it. **This is not a
resolution**, and the audit is on the governing plan's never-cut list: leaving the model
the project is named after outside its own look-ahead audit is not an available outcome.
The options, none yet chosen:

1. **Accept the cost.** A default `pytest` of about two hours. Honest, and unusable as an
   inner loop; the risk is that the audit gets skipped in practice, which is the cut it
   was protected from.
2. **Audit the Bayesian track with the sampler stubbed.** Replace `sample_garch_posterior`
   with a deterministic draw generator built from the training window, and have the stub
   record the exact array it was handed. Every look-ahead surface the audit exists to
   check -- the estimation slice, the `h0` backcast, the returns slice, the block
   assignment, the daily filter, the mixture, the PIT -- stays inside the audit at all
   102 refit dates, and *which data reached each fit* becomes an assertion on recorded
   inputs rather than an inference from output equality, which is stronger than what the
   real sampler gives. What is not covered is the sampler's own internals, which take
   nothing but the two arrays they are handed.
3. **A hybrid**: option 2 in the default suite, plus one real-sampler Bayesian audit over
   all 102 refits behind its own marker, run deliberately before the write-up and its
   result recorded here.

---

### 1.12 Decided after the Stage 3 smoke run (2026-08-26)

Both decisions here respond to measurements taken on a 13-refit smoke run over
2017-01-03 to 2018-01-05, before any Bayesian forecast reached the evaluation layer.
They resolve the open item 1.11 left and supersede two parts of D17 and D20.

**D25 -- `target_accept` raised from 0.9 to 0.95, for every refit, before the production
run.**

One of the 13 smoke refits (2017-02-02, n=777) produced four divergent transitions and
so, under D19, no forecasts for the 21 days it served. At that rate roughly 8 of 102
refits would fail and about 170 of the 2,134 evaluation days would carry no Bayesian
forecast. The cost of that is not mainly the missing days. It is that they would not be
missing at random: a posterior whose geometry defeats the sampler is plausibly a window
where persistence runs hard against the stationarity boundary, and 1.9 recorded the data
doing exactly that. The Bayesian model would then be evaluated on a subsample selected by
a mechanism correlated with the thing being measured, and every pairwise comparison --
the DM tests especially -- would be run on dates chosen partly by the sampler.

Measured on the refit that failed, at the frozen 4 chains x (1,000 + 1,000):

| `target_accept` | divergences | converged | R-hat | min `ess_tail` | seconds |
|---|---|---|---|---|---|
| 0.90 | 4 | no | -- | -- | 33 |
| 0.95 | 0 | yes | 1.0044 | 1,582 | 35 |
| 0.99 | 0 | yes | 1.0017 | 1,395 | 51 |

0.95 removes them at no cost; 0.99 costs half again as much wall clock for nothing.

**This is a change to a frozen decision made after seeing it fail, so the reason it is
not the thing D19 forbids has to be stated rather than assumed.** `target_accept` is not
a diagnostic threshold. It is how hard the sampler works: raising it shortens the leapfrog
step so NUTS meets the *unchanged* D19 criterion, which is the opposite of loosening a
test that fired. And it is applied uniformly to all 102 refits and fixed before the
production run, so it is not the per-refit reseeding D16 and D19 rule out -- no fit is
re-run under different settings because it failed. Had the change been to accept a small
number of divergences, or to re-run only the refits that failed, it would have been that
thing and would not have been made.

Recorded here rather than folded into D17 because the log is append-only, and because a
reader is entitled to know that the setting the production run used was chosen after a
smoke run at a different one. **The report must say so.**

**D26 -- The Bayesian look-ahead audit runs against a recording stub on every test run,
with a real-sampler confirmation behind its own marker.**

This resolves the item 1.11 left open, and supersedes D20's arrangement while keeping the
half of D20 that survives measurement.

On every `pytest`, the Bayesian track is audited with `sample_garch_posterior` replaced
by a deterministic stand-in that derives its draws from the estimation window and
**records the exact array and `h0` it was handed**. Every surface by which the future
could reach a forecast stays in the real code path at all 102 refit dates: the estimation
slice, the backcast, the `returns[:stop]` bound, the block assignment, the per-draw
filter, the mixture, the PIT. And the central question -- did any fit see data from on or
after its own refit date -- stops being an inference from output equality and becomes an
assertion on the sampler's own input, which the real sampler cannot provide. The three
corruption dates and the negative control carry over unchanged; the whole thing runs in
84 seconds.

What the stub does not cover is the sampler's internals, which receive nothing but those
two recorded arrays. `pytest -m bayes_audit` covers them end to end with NUTS actually
sampling, at D20's reduced draw count -- the part of D20 that measurement leaves intact,
since the assertions are exact equalities under a fixed seed and a fixed seed is as exact
at 50 draws as at 2,000. It takes about 40 minutes, is deselected by default, and is to
be **run deliberately before the write-up with its result recorded here**.

The default deselection is the only one the never-cut rule tolerates, and only because it
removes no coverage from a default run: the marked test adds a confirmation, it does not
carry the audit. The alternative -- the real sampler in the default suite -- was rejected
on the measurement in 1.11: near two hours for a `pytest` run, which cuts the audit by
attrition rather than by decision.

---

### 1.13 Stage 3 production run, and what it does to the headline comparison (2026-08-26)

**The run.** 102 refits, 4 chains x (1,000 tune + 1,000 draw) at `target_accept` 0.95,
82.5 minutes of sampling. **100 of 102 converged.** Worst R-hat 1.0049, minimum
`ess_bulk` 1,475, minimum `ess_tail` 972, four divergent transitions in total. The two
failures are 2025-02-10 (three divergences) and 2025-03-12 (one), so 42 of the 2,134
evaluation days carry no Bayesian forecast and are NaN in `forecasts.csv` under D19.
Neither was re-run. Mean annualised volatility 18.94%, against 19.37% for `garch_mle`,
18.86% for `ewma` and 17.71% for `yesterday`.

**The sanity check fails in the direction nobody expected, and it is not a bug.** Over
the full evaluation window the Bayesian intervals are *narrower* than the frequentist
plug-in at every level -- mean width ratios 0.9945 at 90%, 0.9915 at 95%, 0.9842 at 99%,
0.9869 on the 99% VaR -- and the gap is widest exactly where the project's finding was
expected to live: by regime at the 99% level, 0.9962 calm, 0.9796 normal, **0.9709
stressed**.

The handoff said to hunt for the shared-`h_next` bug if this happened. It was hunted, by
decomposition rather than by inspection. Three predictives were built for one day in the
COVID stress (refit 2020-04-06, n=1,575):

| | 90% | 95% | 99% |
|---|---|---|---|
| **C / B** -- posterior predictive over plug-in **at the posterior mean** | 0.9968 | 0.9991 | **1.0046** |
| **B / A** -- plug-in at the posterior mean over plug-in **at the MLE** | 0.9725 | 0.9677 | **0.9550** |
| **C / A** -- what the forecast table reports | 0.9695 | 0.9669 | 0.9594 |

`C / B` is the mixture doing exactly what it was built to do: wider at 99%, marginally
narrower at 90%, the leptokurtosis crossover recorded at 1.11 and pinned by tests. The
per-draw `h_next` has a spread of 12.4% of its mean, so parameter uncertainty is present
and is being integrated over. It is simply **small at this sample size** -- 0.46% on the
99% width at n = 1,575 -- which is the outcome the governing plan's risk register named in
advance: "at n≈750+, parameter uncertainty moves intervals less than
innovation-distribution choice."

`B / A` is four to five times larger and points the other way. It is the **priors moving
the point estimate**, and the mechanism is visible in the refit table:

| | mean `alpha+beta` | max | mean `nu` |
|---|---|---|---|
| `garch_mle` | 0.98926 | 0.99998 (16 refits above 0.999) | 5.198 |
| `garch_bayes` | 0.97859 | 0.99086 | 5.508 |

Both mechanisms push the same way. `beta = (1 - alpha) * delta` with `delta ~ Beta(3, 1)`
makes `alpha + beta < 1` hold by construction and so pulls persistence off the boundary
the MLE runs into, which lowers the one-step variance forecast after a shock -- hence the
gap being largest in the stressed regime. And the `nu` prior, with mean 14, leans toward
near-normal tails and lifts posterior `nu` above the MLE, thinning the Bayesian tails
exactly at 99%. Both are consequences of priors frozen at D4 before any of this was
visible, which is what makes them reportable rather than embarrassing.

**The consequence, and it is not cosmetic. This project's stated one-cause comparison is
not currently a one-cause comparison.** The claim -- in the README, in the plan's
"interview ammunition", and in this log -- is that models 3 and 4 share a likelihood so
that *any interval difference is parameter uncertainty, full stop*. That is true of `C`
against `B`. It is **false** of `C` against `A`, which is what `forecasts.csv` reports,
because `A` sits at the maximum of the likelihood and `B` sits at the mean of a
posterior that the priors have moved. The measured difference between the two models is
mostly prior, not parameter uncertainty, and reporting it as the latter would be a
straightforward misattribution -- the more dangerous for being in the direction of a
tidy story.

Nothing here is a defect in the code, and no result is withdrawn. What changes is what
Stage 4 is allowed to say about the numbers it computes. **The report must not attribute
the frequentist-Bayesian interval difference to parameter uncertainty without the
decomposition above**, and the honest headline is now a comparison of three things rather
than two.

**Recommended, but not decided here**, since it adds a model track and that is a scope
decision: carry a fourth GARCH track, plug-in at the posterior mean, through the same
backtest. It costs almost nothing -- the posterior means for all 102 refits are already
in `refit_records.csv`, and the track is `garch11_filter` plus `plugin_predictive`, no
sampling -- and it turns the muddled two-way comparison into the clean three-way one the
table above shows on a single day. It would be an ablation in the sense `garch_mle_normal`
is, not a fifth competitor, and would stay out of the headline table.

---

### 1.14 The posterior-mean plug-in track (2026-08-26)

**D27 -- `garch_bayes_mean` joins the backtest as an ablation.** 1.13 recorded it as
recommended and left it open; it is now built, on the reasoning 1.13 gives. The
comparison this project rests on had two causes, and this track separates them:

    garch_bayes / garch_bayes_mean   -- parameter uncertainty, and nothing else
    garch_bayes_mean / garch_mle     -- the priors, and nothing else

It is the plug-in predictive at the posterior mean: the same point estimate
`garch_bayes` integrates over, conditioned on as though it were the truth, exactly as
`garch_mle` conditions on the maximum likelihood estimate. **An ablation in the sense
`garch_mle_normal` is**, kept out of `HEADLINE_MODELS` for the same reason -- it differs
from `garch_bayes` by a quantity the report is trying to measure rather than by a
modelling choice anyone would make.

**Built inside `build_bayes_paths` rather than from the persisted posterior means.** The
means for all 102 refits sit in `refit_records.csv` and the track could have been derived
from them without re-running anything, which is why 1.13 called it nearly free. It was
not done that way. The two tracks have to come from the *same* posterior for their
difference to isolate parameter uncertainty, and a version that read a separate file
would keep working after the two fell out of step -- silently, and in a way no test
downstream could see. The cost of doing it properly is one extra `garch11_filter` pass
per refit and a re-run of the stage.

Its `variance` is `h` filtered **at** the posterior mean, while `garch_bayes`'s is the
**mean of** `h` across draws. Those differ by Jensen's inequality and are meant to: one
is a plug-in, the other a posterior expectation. A failed refit blanks both tracks, since
a posterior not worth integrating is not worth averaging either.

**The re-run doubles as the reproducibility check the project owed.** Determinism under a
fixed seed is tested at 150 draws on simulated data; it had never been demonstrated on
the production configuration. Re-running the stage with unchanged seeds and settings
recomputes all 102 posteriors, so `garch_bayes`'s rows must come back bit-identical to
the first run's. It does, with one exception that is worth stating precisely. Every refit
record field is identical -- `mu`, `omega`, `alpha`, `beta`, `nu`, `max_r_hat`,
`min_ess_tail` -- so all 102 posteriors were reproduced exactly, and in the forecast table
`variance`, every interval bound, `var_99` and `pit` are identical bit for bit. The
`mean` column moved in 0.98% of rows, by at most 1e-16 on a quantity of order 7e-4.

The cause is in this session's own refactor rather than in the sampler. Run 1 computed
`fit.draws[:, 0].mean()` -- a strided 1-D reduction, which NumPy does pairwise. Run 2
computes `fit.draws.mean(axis=0)` once and indexes it, which accumulates sequentially
over 2,000 rows. Sequential summation of n terms carries error of order n*eps against
pairwise's log(n)*eps, and 2,000 * 2.2e-16 is exactly the size of the discrepancy
observed. The two-line demonstration is in the session record.

The current form is kept rather than reverted, because both tracks now read the *same*
posterior-mean vector: reverting would make `garch_bayes` and `garch_bayes_mean` disagree
about `mu` at the same 1e-16, which is worse for a comparison whose entire point is that
the point estimate is held fixed.

**What this does not do.** It does not rescue the sentence 1.13 retired. `garch_bayes`
against `garch_mle` remains the confounded comparison; this track means the confound can
now be measured rather than merely disclosed. The report still owes the decomposition,
and the headline table still compares four models of which this is not one.

**The decomposition, over the full evaluation window.** Mean width ratios
across the 2,092 days with a Bayesian forecast:

| level | C/B -- parameter uncertainty | B/A -- the priors | C/A -- what the table reports |
|---|---|---|---|
| 90% | 0.9975 | 0.9970 | 0.9945 |
| 95% | 0.9989 | 0.9926 | 0.9915 |
| 99% | **1.0032** | **0.9811** | 0.9842 |

and by regime at the 99% level:

| regime | n | C/B | B/A | C/A |
|---|---|---|---|---|
| calm | 756 | 1.0036 | 0.9927 | 0.9962 |
| normal | 995 | 1.0031 | 0.9766 | 0.9796 |
| stressed | 341 | 1.0027 | 0.9683 | 0.9709 |

Three things follow, and Stage 4 should not have to rediscover any of them.

Parameter uncertainty behaves exactly as the theory says and as the tests demand: it
widens the 99% interval, narrows the shoulders, and the crossover between 95% and 99% is
the leptokurtosis recorded at 1.11. Its magnitude is 0.32% at the 99% level. That is the
project's headline effect, measured, and it is small -- which is a finding, not a
disappointment, and the governing plan predicted it in the risk register.

**The regime dependence of the reported difference is entirely the priors.** C/B is flat
across regimes to within four parts in ten thousand -- 1.0036, 1.0031, 1.0027 -- while B/A
runs from 0.9927 in calm markets to 0.9683 in stressed ones. The story "the Bayesian
model's intervals behave differently in a crisis" is true of the reported comparison and
false of parameter uncertainty. It is the `delta` prior pulling persistence off the
boundary, and it bites hardest exactly when the MLE is closest to that boundary.

The mean forecast variance ranks `garch_mle` 19.30% annualised, `garch_bayes_mean` 18.94%,
`garch_bayes` 18.94%. The gap between the first two is the prior; the gap between the last
two is Jensen's inequality on `h` and is three parts in ten thousand.

---

### 1.15 The real-sampler look-ahead audit, run (2026-08-26)

D26 owed one run of `pytest --bayes-audit -m bayes_audit` with its result recorded here.
**It passed**, in 39 minutes 41 seconds, against the code as committed -- so with both
Bayesian tracks in scope, since it was run after D27 added the second.

What it establishes beyond the stubbed audit that runs on every `pytest`: that a seeded
NUTS run over an estimation window is a function of that window and nothing else, end to
end. The clean run and the run with every observation from 2020-03-16 onward replaced by
noise produce bit-identical forecast columns for every date up to and including the cut,
across all 102 refits, with the real sampler in the loop.

It is not owed again unless the sampler, the model graph, or `build_bayes_paths` changes.

### 1.16 Decided at Stage 4, before any table was built (2026-08-27)

Four decisions, all taken and written down before `evaluation.py` produced a number. The
order matters: three of the four change what the headline tables say, and deciding them
after seeing the tables would be choosing a result rather than finding one.

**D28 — the common sample, and the refusal to pick one silently.**
Two Bayesian refits failed their diagnostics, so `garch_bayes` and `garch_bayes_mean`
have no forecast on 42 of the 2,134 evaluation days (D19). Every statistic therefore has
to say which days it was computed on. The rule:

- **Per-model marginals** — mean loss, coverage, breach counts, PIT — use each model's
  own available days. The four-model tables therefore mix n = 2,134 and n = 2,092 rows,
  and every row carries its `n` and a `sample` label.
- **Every cross-model comparison** — DM, bootstrap differentials, coverage differences —
  uses the *pairwise* intersection, so a pair of frequentist models is compared on 2,134
  days and any pair involving a Bayesian model on 2,092. The `n` is a column.
- The headline tables are additionally recomputed on the four-model common sample
  (2,092 days), written alongside under the label `common sample`, so a reader can see
  whether the 42 days move anything. They do not, at the resolution reported.

Enforced rather than documented: every scoring function in `evaluation.py` and every
function in `bootstrap.py` **raises** on a non-finite input instead of dropping it. A
dropped NaN would compute a statistic on a sample nobody stated, inside a function whose
output is an error bar. `evaluation.common_sample` is the only way to select a sample and
it returns dates, so the choice is always visible at the call site.

**D29 — the interval comparison is reported through the decomposition, never raw.**
Settled before the coverage tables existed, because every one of them would otherwise
inherit the confound 1.13 exposed. `garch_bayes` against `garch_mle` moves two things at
once — the posterior is integrated over rather than maximised, *and* the frozen D4 priors
move the point estimate — so:

- The headline tables hold the four `HEADLINE_MODELS` only. The frequentist-Bayesian row
  in them is labelled *the reported difference* and is never described as parameter
  uncertainty.
- `garch_bayes_mean` is an ablation in the sense `garch_mle_normal` is (D15), kept out of
  the headline tables and given its own table, `eval_decomposition.csv`, built by
  `evaluation.decomposition_table`. It carries all three contrasts at all three levels,
  overall and by regime, with each side's coverage next to each width ratio.
- Every interpretive statement about parameter uncertainty is sourced from
  `garch_bayes / garch_bayes_mean`. The report may quote the reported difference as such,
  and must not decompose it by assertion.

The stage's own run reproduces 1.13's numbers exactly from the stored table — 0.9975,
0.9989, 1.0032 for parameter uncertainty at 90/95/99 and 0.9970, 0.9926, 0.9811 for the
priors — which is a check on the evaluation layer as much as a restatement.

**D30 — Diebold-Mariano lag truncation: Newey-West automatic, `floor(4 (n/100)^(2/9))`.**
The textbook one-step-ahead DM test uses no lags, on the argument that an optimal
one-step forecast has a serially uncorrelated error. These forecasts are not optimal and
the QLIKE differential is visibly autocorrelated — variance regimes persist for weeks —
so a HAC correction is applied rather than assumed away. At n = 2,092 the rule gives 7
lags. Fixed in `evaluation.newey_west_lag` before any test was run, and the lag actually
used is returned on every result so a reader never has to ask. The Harvey-Leybourne-
Newbold small-sample correction is applied at h = 1 and the statistic referred to a `t`
distribution on n-1 degrees of freedom. Where a finite sample makes the Bartlett-weighted
sum non-positive, the fallback is the contemporaneous variance and the returned lag drops
to 0, so the fallback is visible rather than silent.

**D31 — Giacomini-White stays out.**
Raised as archived-plan D3 and withdrawn with it at 1.5 as never-authorised scope. It is
the theoretically better-suited test here — it is valid under estimation uncertainty with
a rolling scheme, which is exactly this setting — and Stage 4 is where it would be
tempting to add it back, having just written the DM caveats down. It is not added. The
block bootstrap carries the conclusions instead, which is what the locked design says,
and the DM caveat about estimated parameters is reported rather than engineered away.

**One thing Stage 4 does *not* decide.** The KS test on the PIT is reported with its
p-value flagged approximate — the predictive distributions have estimated, rolling-
refitted parameters — and `UniformityResult` carries `p_value_is_approximate` as a field
rather than a comment, so the caveat survives into any table built from it. The histogram
is the diagnostic; the test is a summary of it.

### 1.17 Decided at Stage 5, before the regime tables were built (2026-08-27)

Four decisions, all taken before a regime statistic existed. Two of them are refusals,
which is the harder kind to make after seeing a table that would have been nicer with the
extra number in it.

**D32 — Kupiec per regime; Christoffersen deliberately not.**
Kupiec tests a count against a rate. A regime subsample is still a set of days and their
order does not enter the statistic, so it transfers without argument. Christoffersen's
independence test does not: it counts transitions between *consecutive* observations, and
consecutive rows of a regime subsample can be months apart. Its "yesterday" would be
fictitious, and a p-value computed from fictitious transitions is worse than no p-value,
because it looks like the money test having been run. The full-sample version stays in
`eval_var_backtests.csv`, which is where that question is answered.
`test_the_regime_var_table_runs_kupiec_and_not_christoffersen` pins the absence, so it
cannot be filled in later by someone tidying up.

**D33 — a subsample below 30 days is refused, not reported.**
`evaluation.MIN_REGIME_OBSERVATIONS = 30`. A coverage estimate on twenty days is not a
number, and printing it beside estimates on eight hundred invites exactly the
over-reading the regime tables exist to prevent. It never fires on the VIX regimes — the
smallest is 341 days — and exists for the tercile sensitivity and anything else that
subsets further.

**D34 — the bootstrap resamples *within* the regime, conditioning on the labels.**
The statistic is coverage **given** stressed, so the stressed days are the sample rather
than a draw, and the interval conditions on them. The alternative — resample the whole
series and recompute the regime statistic in each replicate — treats regime membership as
random too, answers a different question, and returns a wider interval for that reason
rather than this one. One consequence belongs in the report rather than in a code
comment: regimes are persistent, so a regime subsample is a modest number of long runs
and its effective number of independent blocks is far below its `n`. That is *why* the
stressed intervals are wide.

**D35 — the tercile thresholds are estimated on the warm-up window alone.**
The plan's regime sensitivity is terciles of trailing 21-day Parkinson volatility. Cut
points taken over the full sample would be a function of the evaluation period, so every
label would depend on days that had not happened yet — and the crisis regime would be
defined using the crisis. The locked VIX thresholds have this property by construction at
15 and 25; the alternative has to earn it, so `trailing_vol_thresholds` is a separate
function that reads `:TRAIN_END` and is pinned by a corrupt-the-future test on both sides.
The consequence is that the out-of-sample buckets are not equal thirds — the warm-up was
calm, so 1,040 of the 2,134 evaluation days land in the top bucket. That is the honest
cost of the discipline and is reported rather than corrected.

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

### Stage 3 -- Bayesian GARCH(1,1)-t (2026-08-26)

The fourth forecaster exists. `models.py` is complete; `backtest.py` runs two tracks;
`forecasts.csv` carries all five model keys over the 2,134-day evaluation window.

**Priors first, code second.** D4 was resolved and written into 1.10 before a line of the
sampler was written, and the constants live in one block above `log_prior` with a test
that breaks if any of them moves. The one departure from the 1.6 feasibility probe --
`delta ~ Beta(3, 1)` rather than `Beta(10, 2)` -- was made on the structural argument that
a `Beta(a, b)` density vanishes at 1 whenever `b > 1`, and the magnitude was measured
afterwards rather than used to choose.

**One likelihood, now verified rather than asserted.** The project's central design
constraint is that models 3 and 4 share `garch11_t_loglik`. PyMC does not call it -- it
builds its own graph -- so until this stage the constraint was a claim about two
implementations that nothing checked. `test_the_pymc_graph_and_the_numpy_log_posterior_agree`
evaluates both at the same parameter vector and requires agreement to floating point,
up to the deliberate `log(1 - alpha)` between a prior stated over `delta` and one stated
over `beta`. The graph also observes every return with `h[0] = h0`, matching the NumPy
likelihood rather than the probe's `r[1:]` (D21).

**The dangerous bug is unreachable rather than avoided.** `mixture_predictive` refuses any
`h_next` that is not one value per draw, and two tests pin both limits: a dispersed
posterior is strictly wider than the plug-in at 95% and 99%, and a posterior collapsed to
a point mass reproduces it exactly -- to 0.0, not to a tolerance.

**The sanity check inherited from the handoff was wrong, and following it would have
manufactured the finding.** It said Bayesian intervals should be at least as wide at every
level. A scale mixture holding average variance fixed is leptokurtic against the single
distribution at that average, so the ratio crosses 1 between 90% and 95%. The quantile
solve was verified against four million draws from the same mixture before the claim
rather than the code was changed. Problems-and-solutions 38.

**Cost, and a decision reopened because its premise was wrong.** A refit costs 29s at
n=756 and 83s at n=2,890, so the production run is about 95 minutes -- and the same refit
at 2 chains x (50 tune + 25 draw) still costs 9s, because the floor is compiling the
gradient of the scan recursion, not sampling. D20 had traded Monte Carlo precision to make
the look-ahead audit affordable on the assumption that draws were what it paid for. D26
replaces that: the Bayesian track is audited on every run against a recording stub, which
covers every look-ahead surface at all 102 refit dates and additionally asserts what data
each fit was handed, while `pytest --bayes-audit` runs the real sampler end to end.
Problems-and-solutions 39 and 40.

**`target_accept` 0.9 -> 0.95 (D25)**, before the production run, after a smoke refit
produced four divergences and therefore no forecasts for the 21 days it served. Measured
on that refit: 0 divergences at 0.95, at 35s against 33s. It is sampler effort rather than
a diagnostic threshold, and the report is obliged to say it was changed.

**The run.** 100 of 102 refits converged; worst R-hat 1.0049, minimum `ess_tail` 972, four
divergences in total. The two failures leave 42 evaluation days without a Bayesian
forecast, recorded as NaN and not re-run.

**The result, and it is not the expected one.** Bayesian intervals come out *narrower*
than the frequentist plug-in at every level and most in stress -- 0.9709 at the 99% level
on stressed days. Decomposed at 1.13: the mixture is carrying parameter uncertainty
correctly (+0.46% on the 99% width), and is outweighed four to five times over by the
priors moving the point estimate, which pull persistence off the boundary the MLE runs
into (mean `alpha+beta` 0.9786 against 0.9893) and lift `nu` above it (5.51 against 5.20).
**The consequence is that the frequentist-Bayesian interval difference is not attributable
to parameter uncertainty**, which is what this project has been saying it would be, and
1.13 records what Stage 4 must therefore do differently.

Next: Stage 4, the evaluation layer. Everything it needs is in `forecasts.csv`; it should
re-run no part of the backtest.

### Stage 4 -- evaluation layer (2026-08-27)

`evaluation.py` and `bootstrap.py` are complete, `run_all.py --stage evaluate` and
`--stage figures` are implemented, and six tables, four figures and `02_results.ipynb`
come out of the stored forecast table. Nothing in this stage refits anything: every number is a function of
`forecasts.csv`, so none of it can move unless the backtest moves first. 321 tests pass
and one is skipped by default, in 12m40s: 92 new tests in `tests/test_evaluation.py` and
`tests/test_bootstrap.py`, against two removed from `test_smoke.py` -- the parametrised
`test_unimplemented_stages_still_raise` emptied when `evaluate` and `figures` landed, and
a test parametrised over nothing collects nothing and asserts nothing.

**Decisions before tables.** D28 (the common sample), D29 (the interval comparison
reported only through the decomposition), D30 (the DM lag rule) and D31 (Giacomini-White
stays out) are at 1.16, all written before the first table was built. D29 is the one the
handoff insisted on: the coverage tables would otherwise have inherited 1.13's confound
and the choice of what to do about it would have been made with the answers visible.

**The evaluation layer reproduces 1.13 exactly**, from the stored table rather than from
the run that produced it: parameter uncertainty 0.9975 / 0.9989 / 1.0032 at 90 / 95 / 99,
the priors 0.9970 / 0.9926 / 0.9811. That agreement is a check on the new code, not a
restatement of an old result.

**Point accuracy: the models are separable, and the expected finding does not hold.** The
governing plan predicted the four would be hard to separate, with that difficulty as the
setup for the calibration act. On mean QLIKE over the 2,092-day common sample the ordering
is `garch_bayes` 0.4577, `garch_mle` 0.4599, `ewma` 0.5239, `yesterday` 0.7809, and both
GARCH models separate from both baselines with bootstrap intervals nowhere near zero. What
*is* hard to separate is the pair the research question turns on: `garch_mle` against
`garch_bayes` differs by 0.0022, half a percent of the loss level, on a differential whose
variance is 0.00115. Its bootstrap interval excludes zero by a hair, `[+0.00019, +0.00418]`
-- and `garch_bayes_mean` against `garch_mle` does not exclude it at all,
`[-0.00394, +0.00034]`. The report should say the difference is between GARCH and no
GARCH, not between estimators.

**Calibration: the two GARCH-t models pass the PIT test and the baselines fail it
decisively.** KS against uniform: `garch_bayes` p = 0.175, `garch_mle` p = 0.115, against
`ewma` 3e-14 and `yesterday` 1e-7. `garch_mle_normal` fails at 2.6e-4, which is Stage 6's
innovation ablation arriving early and pointing the way it was expected to. Every one of
these p-values is approximate because the parameters are estimated, and the flag rides in
the table rather than in a footnote.

**Interval coverage is below nominal at 90% and 99% for every model.** `garch_mle` runs
0.8889 / 0.9522 / 0.9897 against 0.90 / 0.95 / 0.99 and `garch_bayes` 0.8886 / 0.9517 /
0.9890. The two are indistinguishable at every level -- which is the honest answer to the
research question as posed, and it is the answer 1.13 predicted once the width differences
turned out to be tenths of a percent.

**The 99% VaR result, and it is not the shape the plan anticipated.** All four models fail
Kupiec: 87 breaches for `yesterday`, 54 for `ewma`, 37 for each GARCH model against 21
expected, and the least strongly rejected of the four, `garch_mle`, still at p = 0.002. **Christoffersen independence fires for none of
them** -- p = 0.67 and 0.69 for the GARCH models, 0.76 for `yesterday`, 0.058 for `ewma`,
which is the only one close. So the models fail on the *level* of tail risk and not on its
*timing*: the breaches are too many but they are not clustered. The plan called
Christoffersen the money test on the argument that correct average coverage can hide
clustered failures; here the average coverage is wrong and the clustering is absent, which
is the mirror image and no less reportable. Conditional coverage rejects for all four,
driven entirely by its Kupiec component.

**`garch_bayes` and `garch_bayes_mean` have identical 99% breach counts and identical
coverage at every level**, to the last digit, despite different interval widths. Parameter
uncertainty moves the 99% width by 0.32%; nothing in 2,092 days of returns lands in the
gap. That is the cleanest statement of this project's central measurement, and it belongs
in the report: at n = 750 and above, integrating over parameter uncertainty is not what
determines whether a risk model's intervals are calibrated.

**The proxy separation is asserted, not promised.** `test_scrambling_the_proxy_leaves_
every_calibration_number_untouched` replaces `proxy_var` with a scaled permutation and
requires every coverage, VaR, PIT and decomposition table to come back bit-identical, with
a negative case requiring the point losses to move. Calibration is scored against observed
returns and inherits none of the D2 proxy problem; that is now a test rather than a
paragraph.

**Two limits worth carrying forward.** The QLIKE numbers are on the scaled proxy and the
raw-Parkinson ranking owed by D10 is still Stage 6's. And every p-value in
`eval_comparisons.csv` is unadjusted across eight pairwise comparisons; the multiplicity
caveat is in the docstring and belongs in the report body.

Next: Stage 5, regime-conditional analysis. `summarise_by_regime` and
`bootstrap_coverage_difference` are built and tested, the decomposition table is already
regime-split, and the never-cut requirement is bootstrap CIs on every per-regime coverage
estimate with `n` beside it -- stressed days are 341 of the 2,092 the Bayesian model
forecasts.

### Stage 5 -- regime-conditional analysis (2026-08-27)

Three regime tables, two figures, and the tercile sensitivity the plan lists third on its
cut list. The last never-cut item is discharged: every per-regime coverage estimate now
carries a stationary-block-bootstrap interval. Decisions D32-D35 are at 1.17, all taken
before a regime statistic existed.

**The project's headline question has a surprising answer.** It asked whether interval
calibration survives high-volatility regimes. For the GARCH models it does -- and they
fail where nobody was looking. 99% VaR breach rates, with 95% bootstrap intervals:

| model | calm (n=757) | normal (n=1,030) | stressed (n=347) |
|---|---|---|---|
| `garch_mle` | 1.19% [0.53, 1.85] | **2.33% [1.65, 3.11]** | 1.15% [0.29, 2.02] |
| `garch_bayes` | 1.19% [0.53, 1.98] | **2.31% [1.51, 3.12]** | 1.47% [0.29, 2.93] |
| `ewma` | 1.59% [0.79, 2.38] | 2.91% [2.04, 3.88] | 3.46% [1.44, 6.05] |
| `yesterday` | 3.70% | 4.08% | 4.90% |

Both GARCH models are indistinguishable from nominal in calm *and* in stress, and clearly
too high in the middle band. The baselines degrade monotonically with volatility, which
is the pattern one would have predicted for all four. Kupiec: p = 0.61 / 0.0003 / 0.78 for
`garch_mle`.

**The mechanism is tail misallocation, and it is invisible in a two-sided number.** At the
99% two-sided level in the normal regime, `garch_mle` has **14 breaches below the interval
and none above**, against 5.2 expected in each tail; `garch_bayes` has 15 and 1. Total
coverage there is 0.9864 against a nominal 0.99 -- a near miss -- while every breach is a
loss. That is why `interval_coverage` counts the tails separately, and it is the single
most reportable thing in the stage.

**Two-sided coverage, for the record.** Both GARCH models over-cover in calm at every
level (0.925 / 0.967 / 0.996 against 0.90 / 0.95 / 0.99) and under-cover at 90% everywhere
else (0.874 normal, 0.856 stressed). At 95% and 99% their intervals contain the nominal
level in the normal and stressed regimes. The frequentist and Bayesian models are
indistinguishable from each other in every regime at every level, which is 1.16's Result 3
holding under conditioning.

**QLIKE by regime.** The GARCH models' mean loss is *lowest* in the stressed regime (0.42
against 0.48 normal) while EWMA's is highest there (0.63) -- so the GARCH advantage is
widest exactly where it matters. The level is not comparable across regimes, only the
ordering within one, and the report must say so or a rising column will read as
deterioration.

**The sensitivity does not overturn it, and sharpens it.** Under terciles of trailing
21-day Parkinson volatility, with cut points from the warm-up window alone (D35), the
GARCH models are closest to nominal in the *top* tercile (1.35%, Kupiec p = 0.29, n=1,040)
and worst in the bottom one (2.06%, p = 0.012). The two definitions disagree about which
non-stressed bucket is the weak one -- the VIX split says the middle band, the tercile
split says the calm bucket -- and agree on the thing that matters: **these models are not
worse in stress, they are worse outside it.** The buckets are not the same partition and
no row compares across the two; the tercile "stressed" bucket is 1,040 days and a much
weaker notion of stress than VIX above 25.

**What this does not establish.** Why the middle is the weak spot. The plausible story --
the 15-25 band is where regime *transitions* happen and a GARCH forecast lags a change in
level by construction -- is a conjecture this design cannot test, and the report should
present it as one. And the stressed intervals are wide: [0.29%, 2.02%] admits rates from a
third of nominal to double it, so "survives stress" means "this sample cannot show it
failing", not "is known to hold". Regimes are persistent, so those 347 days are a handful
of long runs and the effective sample is smaller than the count suggests (D34).

**Also this stage.** `data.assign_trailing_vol_regime` and `trailing_vol_thresholds`, with
corrupt-the-future tests on both sides of the look-ahead claim and vacuity checks beside
them. 342 tests pass and one is skipped by default, in 13m03s -- 21 new since Stage 4: 13
for the regime tables, 6 for the trailing-volatility labels, and 2 guarding the
`prior_delta` plumbing that the Stage 6 runs need.

Next: Stage 6. The prior-sensitivity runs owed by D4 are under way as two full Bayesian
backtests under `Beta(10, 2)` and `Beta(1, 1)`, launched from `--stage priors`, writing to
`data/processed/prior_sensitivity/` and touching nothing else. What remains after them is
the raw-Parkinson QLIKE ranking owed by D10, the 63-day cadence spot check, and writing up
the GARCH-normal ablation that Stage 4 already measured.
