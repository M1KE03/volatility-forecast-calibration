# Problems faced, and what was done about them

Companion to `research_log.md` and `README.md`.

- `research_log.md` is the **chronological** record: what was decided, when, and why.
- `README.md` is the **outward** view: what the project is and how to run it.
- This file is the **problem-oriented** view: every difficulty encountered so far, why it
  mattered, and the fix that is now in the repository.

Scope: everything up to the end of **Stage 5** (regime analysis complete, 2026-08-27).
The entries are grouped by the stage that produced them. The summary table below covers
entries 1-36, through Stage 2; entries 37 onward — the Bayesian model and the evaluation
layer — are listed in full further down without a summary row.

A note on what is included. Several entries are mistakes made during this work rather
than external obstacles — a plan followed too long, a premise never tested, a test that
asserted the wrong thing. They are recorded in the same register as everything else. A
project whose entire subject is whether stated uncertainty can be trusted has no business
keeping a tidier account of itself than it demands of its models.

---

## Summary

| # | Problem | Fix | Enforced by |
|---|---|---|---|
| **Look-ahead and leakage** | | | |
| 1 | Lagged quantities undefined on the first in-sample row | One-month download buffer (D8) | `test_analysis_frame_is_trimmed_...` |
| 2 | `assign_vix_regime` could be double-lagged silently | Function owns its own lag; contract tested | `test_data.py` regime tests |
| 3 | Specified look-ahead test could not hold as written | Corrupt *strictly after* `t` | `test_data.py` test 6 |
| 4 | EDA on full sample would justify the model with evaluation data | Training window primary (D9) | `test_training_window_results_ignore_oos` |
| 5 | EWMA off-by-one would read the current day's return | Explicit loop, `h[t]` written before `r[t]` read | `test_ewma_forecast_for_day_t_ignores_day_t` |
| 6 | EWMA seed from the full sample would leak invisibly | Seed from warm-up only | `test_ewma_seed_cannot_be_moved_by_oos` |
| 7 | Master audit's claim was subtly wrong if stated too strongly | Forecast cols `≤ t`, evaluation cols `< t` | `test_master_look_ahead_...` |
| **Measurement and scale** | | | |
| 8 | Parkinson understates close-to-close variance by 1.52× | Frozen constant `c` (D10) | `test_variances_are_on_the_close_to_close_scale` |
| 9 | `c` buys unconditional, not conditional, unbiasedness | Honest wording + Stage 6 raw-proxy check | Documented obligation |
| 10 | Constant mean is a real simplification out of sample | Recorded as a limitation | Log §1.7 |
| **Environment and reproducibility** | | | |
| 11 | PyMC rejected on a premise that was never tested | Compiler installed; B1 revised (B1-R) | Verified by probe |
| 12 | MSVC needs elevation this session does not have | Portable MinGW-w64, no admin | `g++ --version` |
| 13 | PyMC downgraded three verified pins | Re-pinned; suite re-run | 56 tests, before and after |
| 14 | `statsmodels` was an invisible transitive dependency | Pinned explicitly | `requirements.txt` |
| 15 | The compiler is a repro component `pip` cannot supply | Version, source and SHA-256 documented | README + log §1.6 |
| 16 | CRLF endings broke raw-data hashes on a fresh clone | `.gitattributes` marks them `-text` | `test_snapshot_bytes_are_lf_only` |
| **Process and planning** | | | |
| 17 | Two plans with colliding stage numbering | Switched to one; re-keyed everywhere | Log §1.5 |
| 18 | Stubs configured for a sampler no longer used | Replaced emcee fields with NUTS fields | `BacktestConfig` |
| 19 | A stub blocked on a withdrawn decision | Expanding window locked; ROLLING raises | `test_rolling_window_is_rejected` |
| 20 | Stub schema contradicted the governing plan | Long format (D13) | `test_forecast_table_is_complete` |
| 21 | Deleting the archived plan left dangling references | Repointed at git history | — |
| **Test integrity** | | | |
| 22 | Positive-only tests cannot detect a always-firing diagnostic | Every diagnostic tested twice | `test_eda.py` negative controls |
| 23 | An invariance test can quietly become vacuous | Prove the invariance is load-bearing | `test_arch_lm_would_be_inflated_...` |
| 24 | The audit would pass on a backtest ignoring its input | Assert corruption *does* change the future | `test_corruption_actually_changes_...` |
| 25 | A test of mine asserted the wrong quantity | Assert by equivalence instead | `test_ewma_seed_defaults_...` |
| **Presentation** | | | |
| 26 | Log-scale histogram hid the sparse tails | Step outline, not filled bars | Visual check |
| 27 | Threshold labels collided with the data | Left anchor + opaque bbox | Visual check |
| 28 | Stress shading stretched an axis by two years | Clip spans to the window | Visual check |

---

## Look-ahead and leakage

This project's headline claim is that it can detect miscalibrated uncertainty. It has
standing to make that claim only if its own pipeline is clean, so leakage hazards get
more space here than anything else.

### 1. Lagged quantities were undefined on the first in-sample row

**Problem.** The locked sample starts 2014-01-01 and the training window at 2014-01-02,
the first trading day. Two derived columns are lagged: `log_return` on 2014-01-02 needs
the adjusted close on 2013-12-31, and the regime label needs the VIX close on the same
day. Downloading only from 2014-01-01 makes both NaN.

**Why it mattered.** Not a crash — a silent one-day shortening of a *locked* window. The
first row would have been unusable and the training sample would no longer have been the
one the design specified.

**Solution.** Decision D8: download from `DOWNLOAD_START = 2013-12-01`, compute every
derived column on the full joined history, and *then* trim to `[SAMPLE_START,
SAMPLE_END]`. The buffer feeds the lag and nothing else; no buffer row reaches the
backtest. Committed raw CSVs start 2013-12-02; the analysis frame starts 2014-01-02 as
locked.

### 2. The regime function could be double-lagged without any error

**Problem.** `assign_vix_regime` performs its own lag internally. A caller who passed an
already-lagged series would get a two-day lag.

**Why it mattered.** No exception, no warning. Every stressed day would be misattributed,
corrupting the headline regime result while looking entirely normal.

**Solution.** The contract is fixed and documented — the function takes the *raw* VIX
close — and a test asserts that the regime on row `t` equals the bin of the input at row
`t-1`, from both directions.

### 3. The specified look-ahead test could not hold as written

**Problem.** The original plan said: corrupt every input row "from `t` onward", assert row
`t` is unchanged. That is false by construction. Row `t`'s `log_return` and
`parkinson_var` are functions of row `t`'s own inputs, so NaN-ing row `t` changes them
for a reason that is not look-ahead.

**Why it mattered.** A test that cannot pass gets weakened until it does, and the weakened
version often tests nothing.

**Solution.** Implemented as: corrupt every input row **strictly after** `t`, assert every
row at or before `t` is bit-identical. The complementary direction — that the regime on
`t` ignores the VIX close on `t` — is a separate test, so the lag is covered from both
sides. This same distinction recurs at #7.

### 4. Full-sample EDA would have let the evaluation period justify the model

**Problem.** The governing plan asks for ARCH-LM, Ljung-Box, ACF and ADF without saying
over which window. The obvious reading — and what most write-ups do — is the full sample.

**Why it mattered.** Model-class choice is a modelling decision. Justifying "we use GARCH"
with a statistic computed over the 2,134 out-of-sample days lets the evaluation period
argue for the model later evaluated on it. It is mild as leakage goes, but it is the
exact species this project exists to detect in others. Committing it in the motivation
section would leave the project with no standing to report a coverage failure elsewhere.

**Solution.** Decision D9: training window is primary. Full-sample values are computed and
printed, labelled `DESCRIPTIVE CONTEXT ONLY`, so a reader can confirm the two agree —
they justify nothing. Enforced by a test that corrupts out-of-sample data and asserts the
training-window statistics are unchanged.

### 5. The EWMA recursion had a one-day leak available in it

**Problem.** `h[t]` is the forecast **for** day `t`. Advancing `h` and then assigning it
to the same index position makes the forecast for day `t` depend on the return of day
`t`.

**Why it mattered.** The single most consequential off-by-one in the module, and it would
have produced forecasts that looked excellent — a "forecast" that has seen the day it
forecasts tracks realised volatility beautifully.

**Solution.** An explicit loop with the ordering written out, rather than a vectorised
`ewm()` call whose convention is easy to misread. Two tests: one asserts changing the
return on day `t` leaves the forecast for `t` untouched and moves the forecast for `t+1`;
the acceptance plot (#below) shows the same thing visually.

### 6. The EWMA seed could have leaked the whole evaluation period

**Problem.** `forecast_ewma` defaults its seed to the variance of whatever series it is
handed. Handed the full sample, it seeds from data including 2017-2025.

**Why it mattered.** Invisible. Every forecast including the earliest would carry
information from the evaluation period, and the forecasts would still look perfectly
reasonable.

**Solution.** The harness computes the seed from the warm-up block only and passes it
explicitly. A test corrupts every out-of-sample return and asserts the early forecasts do
not move.

### 7. The master audit's claim had to be stated precisely or not at all

**Problem.** The natural phrasing — "corrupting from `t` leaves everything before `t`
identical" — is either too weak or too strong depending on which columns are meant.

**Why it mattered.** Corrupting from `t` must leave the **forecast** columns unchanged up
to *and including* `t`, because the forecast for `t` is built from data through `t-1`. It
must **not** leave the **evaluation** columns at `t` unchanged: `log_return` and the `pit`
derived from it are functions of day `t` itself. Asserting the stronger claim would be
asserting something false, which would have to be weakened later — the failure mode of
#3, repeating.

**Solution.** The test asserts forecast columns over `≤ t` and all columns over `< t`, at
three dates chosen as worst cases: Volmageddon (2018-02-05), the largest COVID drawdown
day (2020-03-16), and a 2022 selloff (2022-06-13).

---

## Measurement and scale

### 8. The Parkinson proxy is on the wrong scale, and the error cut both ways

**Problem.** The Parkinson estimator is built from the intraday high/low range and sees no
part of the overnight move. For SPY the overnight session carries a large share of daily
variance. Measured on the warm-up sample, raw Parkinson understates close-to-close
variance by a factor of **1.517** (10.85% vs 13.35% annualised).

**Why it mattered.** This is the subtlest problem encountered so far, because the error
pulls in two directions at once:

| | RW baseline (Parkinson-based) | EWMA / GARCH (r²-based) |
|---|---|---|
| QLIKE against the raw proxy | units match — unfairly **advantaged** | penalised by the 1.52 factor |
| Intervals against observed returns | ~23% too narrow — **over-breaches** | correctly scaled |

Left alone, the RW baseline would have won the point-forecast table and failed the
calibration table, both for a units reason with nothing to do with forecasting or
uncertainty quantification — contaminating precisely the comparison the project exists to
make. Separately, QLIKE's proxy-robustness (Patton 2011) is a statement about an
*unbiased* proxy; a systematically low proxy forfeits the property that makes QLIKE safe
to rank with.

**Solution.** Decision D10, a master convention: every variance forecast and the
evaluation proxy live on the close-to-close return-variance scale. One constant, estimated
on the warm-up sample only and frozen:

```
c = mean(r²) / mean(σ²_P) = 1.517318      2014-01-02 .. 2016-12-30, n = 756
```

The RW baseline forecasts yesterday's *scaled* Parkinson and derives its intervals from
that variance. EWMA and both GARCH models are already driven by squared returns and are
unchanged.

`c` is hard-coded rather than recomputed at import: a constant recomputed on the fly would
move silently if the frame were ever rebuilt differently, and every published number would
move with it without anything failing. A test recomputes it from the committed snapshot
and asserts agreement; another asserts out-of-sample data cannot move it.

### 9. The constant fixes the level, not the conditional bias

**Problem.** The rationale for `c` invokes conditional unbiasedness, which is what Patton's
result actually requires. A frozen scalar delivers the *unconditional* kind.

**Why it mattered.** Within the warm-up alone the ratio is 1.39 (2014), 1.57 (2015), 1.56
(2016); across quarters it ranges from **1.18 to 1.94**. The overnight share of variance
moves around and plausibly co-moves with regime, since gaps dominate in stress. Residual
conditional bias therefore remains, and it is exactly the kind of thing a reader of a
calibration paper will ask about.

**Solution.** Not a code change — a wording and scope obligation, recorded before any
result depends on it. The report says the proxy is *approximately unbiased on average*
and must not claim proxy-robustness is restored. Stage 6 re-runs the QLIKE ranking on raw
Parkinson to show the ranking does not hinge on `c`.

### 10. The constant-mean assumption is weaker out of sample than in

**Problem.** On the training window, Ljung-Box on raw returns does not reject at any lag
(p = 0.45 / 0.62 / 0.26). On the full sample it rejects decisively (p = 7.1e-14 at 5
lags). Full-sample excess kurtosis is 14.71 against 2.46 in training.

**Why it mattered.** Short-horizon return autocorrelation rises in crises and the
evaluation period contains several. The design assumes a constant mean.

**Solution.** No design change: the constant mean is applied identically to all four
forecasters, so it cannot favour one over another and cannot bias the comparison. But it
is a real simplification over the evaluation period, so it is written into the log now and
carried to the report's limitations section — reported as a known property rather than
discovered late and quietly dropped.

---

## Environment and reproducibility

### 11. A capable option was rejected on a premise nobody tested

**Problem.** Decision B1 chose a hand-written numba likelihood with `emcee`, rejecting
PyMC because no system C/C++ compiler was present. When the compiler was installed, two
things emerged: the probe fit worked immediately (R-hat 1.00, zero divergences, ~10s per
fit), **and** PyTensor 3.3's default linker turned out to be `NumbaLinker` — it compiles
through numba, which B1 had already accepted as needing no system compiler.

**Why it mattered.** Route A was probably open the entire time. B1's reasoning was sound
on the evidence it had; the evidence was never checked. The general failure — rejecting an
option on an untested premise and then building around the rejection — is worth more than
this one instance of it.

**Solution.** Decision B1-R. Route A adopted, verified by probe on the real training window
before adoption rather than assumed. `emcee` demoted to the fallback the governing plan
always intended, and left pinned.

### 12. The standard toolchain needed rights this session did not have

**Problem.** MSVC Build Tools is the canonical Windows compiler and requires elevation.
The session had no admin rights, so a `winget install` would have stalled on a UAC prompt
that could not be answered.

**Solution.** Portable MinGW-w64 GCC 16.2.0 (UCRT, x86_64) from WinLibs, unpacked to a
user directory and added to the user `PATH`. No elevation, and uninstalling is deleting a
folder. UCRT specifically, because it matches CPython 3.12's own runtime.

### 13. Installing PyMC silently downgraded three verified pins

**Problem.** `pytensor 3.3.0` sets ceilings that forced `numba` 0.67.0 → 0.66.0,
`llvmlite` 0.49.0 → 0.48.0, and `numpy` 2.5.2 → 2.4.6.

**Why it mattered.** `numba` is the compiled inner loop the original design rested on, and
`numpy` touches every computation in the data layer. A dependency resolver moving three
pins is not a detail to notice later.

**Solution.** Re-pinned to the installed versions with the reason recorded inline;
`llvmlite` pinned explicitly so it cannot drift away from `numba`. **The full suite was
re-run before and after: 56 pass either way** — which was the thing worth checking rather
than assuming.

### 14. A directly-used library was never a declared dependency

**Problem.** `statsmodels` supplies ARCH-LM, Ljung-Box and ADF. It was present only as a
transitive dependency of `arch`, and `src/eda.py` imports it directly.

**Why it mattered.** A future `arch` release dropping or changing that dependency would
break the EDA with no pin to explain why.

**Solution.** `statsmodels==0.14.6` pinned as a direct dependency.

### 15. Reproducibility now has a component `pip` cannot deliver

**Problem.** `pip install -r requirements.txt` no longer reproduces the environment,
because PyTensor needs a compiler that pip does not supply.

**Solution.** Treated as a first-class part of the deliverable rather than a footnote: the
compiler's version, source, download URL and **verified SHA-256** are recorded in
`requirements.txt`, the README, and the log. The hash was checked against the publisher's
published value before extraction — the same standard the project already applies to its
price data.

### 16. Git line-ending normalisation broke the data integrity check

**Problem.** `data/raw/*.csv` is committed with a SHA-256 manifest so results cannot drift
when Yahoo revises history. Git's default line-ending handling rewrote those files on
checkout, so every hash mismatched on a fresh clone and the loader refused to run — which
defeated the entire point of committing the snapshot.

**Why it mattered.** Invisible in the authoring worktree, because those files were
produced by a write rather than a checkout. It was found only by actually cloning the
repository.

**Solution.** `.gitattributes` marks `data/raw/*.csv` and `*.json` as `-text`, freezing
their bytes on every platform. Verified by cloning again: 0 CRLF, both hashes match.
Deliberately scoped to `data/raw/` only. The general lesson, which now applies to the
look-ahead tests as well: *an integrity check that has never been observed to fail is not
evidence that it works.*

---

## Process and planning

### 17. Two plans were in play, with stage numbers that meant opposite things

**Problem.** The repository carried both `docs/implementation_plan.md` (numbered by
module: its Stage 2 = `data.py`) and `docs/project1-implementation-plan.md` (numbered by
workflow: its Stage 2 = frequentist GARCH). Work had proceeded against the first.

**Why it mattered.** More than a naming clash. The README announced "Stage 2 complete",
which under the incoming numbering asserts a *frequentist GARCH model exists* — when in
fact only the data layer did. Anyone reading the repo would have been actively misled.
Reassuringly, the two documents' locked decisions were identical line for line, so no
completed work was invalidated.

**Solution.** One governing plan (`project1`), the other retired. Every `Stage N`
reference in the repository re-keyed — README, `run_all.py`, `src/data.py`,
`tests/test_data.py`, `tests/test_smoke.py`. Five open decisions carried only by the
retired plan (D1, D2, D3, D6, D7) withdrawn rather than resolved; D3 (Giacomini-White)
dropped outright as never-authorised scope.

### 18. The stubs were configured for a sampler no longer being used

**Problem.** `BacktestConfig` carried `n_walkers`, `n_steps`, `n_burn`, `thin` — emcee
parameters — after B1-R switched the project to PyMC. `RefitRecord` likewise carried
acceptance fraction and autocorrelation time.

**Why it mattered.** Dead configuration that reads as live. A later reader would
reasonably assume the fields were used.

**Solution.** Replaced with NUTS fields (`draws`, `tune`, `chains`, `target_accept`) and
NUTS diagnostics (R-hat, ESS bulk/tail, divergences). Inert until Stage 3, but correct.

### 19. A stub was blocked on a decision that no longer existed

**Problem.** `backtest.py` carried "Open item D1 blocks this module" — expanding versus
rolling estimation window. D1 was withdrawn at the plan switch, and the governing plan
fixes an expanding window in its locked-decisions table.

**Solution.** Note removed, expanding window locked. `EstimationWindow.ROLLING` now
**raises** rather than silently producing a result nobody chose — out-of-scope options
should fail loudly, not quietly work.

### 20. The stub's output schema contradicted the governing plan

**Problem.** The stub docstring specified a wide schema (`{model}_var`, `{model}_lo_90`,
…); the governing plan asks for a tidy/long frame.

**Solution.** Long (decision D13). Every downstream consumer groups by model or by regime,
which wide format makes awkward.

### 21. Deleting the retired plan left dangling references

**Problem.** The archived copy was deleted from the working tree, leaving four references
to `docs/archive/` in the README and log pointing at a path that no longer exists.

**Solution.** Repointed at git history (`git log -- docs/implementation_plan.md`, last
present at commit `d2689b9`), so the trail is still followable.

---

## Test integrity

A recurring theme, and arguably the most transferable material here: **a test that only
ever sees the case it is supposed to detect cannot distinguish a working test from a
broken one.**

### 22. Positive-only diagnostics would have fabricated the project's premise

**Problem.** The EDA exists to establish that ARCH effects are present. Testing
`arch_lm_test` only on data that has ARCH effects would pass identically for a function
that returned `p_value = 0.0` unconditionally.

**Why it mattered.** That function would have handed the project a fabricated
justification for its own model class — and it would have been reported as evidence.

**Solution.** Every diagnostic is tested twice: on data that should trigger it (simulated
GARCH, AR(1), random walk) and on data that should not (i.i.d. normal). `headline_verdict`
is likewise tested on white noise, where it must report mixed evidence rather than
optimism.

### 23. An invariance test can quietly become vacuous

**Problem.** `arch_lm_test` promises to demean internally, tested by asserting the
statistic is unchanged when a constant is added. But if the function stopped demeaning and
the effect happened to be negligible, that test would still pass.

**Solution.** A companion test computes the undemeaned statistic by hand and asserts it
differs materially — proving the invariance test is load-bearing rather than decorative.

### 24. The audit would pass on a backtest that ignored its input entirely

**Problem.** "Corrupting the future leaves the past identical" is trivially satisfied by a
function that returns the same thing regardless of input.

**Solution.** `test_corruption_actually_changes_the_future` asserts the corruption *does*
move later forecasts. Same pattern as #23, applied to the most important test in the repo.

### 25. One of my own tests asserted the wrong quantity

**Problem.** `test_ewma_seed_defaults_to_the_variance_of_what_it_is_given` compared
`out.iloc[1]` to the seed. But `out.iloc[1]` is the seed *already advanced one step* by
`r[0]`, not the seed. The test failed, and the failure was in the test.

**Why it is here.** The tempting fix is to loosen the tolerance until it passes. That
would have left a test that no longer checked the default seed at all.

**Solution.** Asserted by equivalence instead — `forecast_ewma(returns)` must equal
`forecast_ewma(returns, initial_var=var(returns, ddof=1))` — which tests the actual
contract and cannot be satisfied by coincidence.

---

## Presentation

Minor next to the above, but each one hid something the figure existed to show.

### 26. A log-scale histogram made one observation look like two hundred

**Problem.** Filled histogram bars extend to the axis floor. On a log y-axis this makes a
bin holding a single observation look as solid as a bin holding hundreds — hiding
precisely the sparse tail that motivates the Student-t innovation.

**Solution.** `histtype="step"`. The tail observations now read as individual spikes
standing above the matched normal curve.

### 27. Threshold labels sat on top of the data they described

**Problem.** The VIX regime thresholds at 15 and 25 are revisited constantly, so a label
placed anywhere on that axis eventually collides with the series.

**Solution.** Anchored left, where the early-sample VIX is calmest, with an opaque
background box.

### 28. Stress shading stretched an axis by two years

**Problem.** The COVID-window forecast plot drew all three stress spans including the 2022
bear market, which pushed the x-axis out to 2022 and compressed the six-month window the
figure exists to show into a sliver, with month labels overprinting into illegibility.

**Solution.** `shade_stress_periods` skips spans entirely outside the current limits, and
the plot pins its limits before and after shading so no annotation can widen the view.

---

## The check that caught what no statistic would have

Worth separating out, because it is the reason Stage 1's acceptance criteria included
looking at a picture.

An EWMA forecast is a weighted average of *past* squared returns. It therefore **must**
lag a volatility spike on the way in and overshoot on the way out — it has no mechanism to
do anything else. Plotting forecasts against realised volatility through March 2020:

- Realised peaks at ~1.03 annualised on 16 March 2020; EWMA is at ~0.70 that day and does
  not reach its own peak of 0.81 until roughly a week later.
- On the way out, realised falls to ~0.2 by early May while EWMA is still at 0.6 in late
  April and ~0.30 in June against realised ~0.15.
- Both series sit flat at ~0.10 through mid-February and move only *after* the large
  returns arrive.

Had the recursion been reading the current day's return, the EWMA line would sit on top of
the realised peak instead of trailing it. No summary statistic in the evaluation layer
would have flagged that — the forecasts would simply have scored suspiciously well.

`figures/05_baseline_forecasts_covid.png`.

---

# Stage 2 (frequentist GARCH)

Added 2026-08-23, when `models.py` acquired a model that estimates something. The
numbering continues the table above.

| # | Problem | Fix | Enforced by |
|---|---|---|---|
| **Estimation** | | | |
| 29 | Multi-start ranked on objective value alone blanked 21 days of forecasts | Rank converged optima first | `test_mle_prefers_a_converged_start...` |
| 30 | A small multi-start grid finds a worse optimum a third of the time | Keep the 18-point grid; measured, not assumed | Log §1.9 |
| 31 | `omega ~ 5e-6` sits below the optimiser's convergence tolerance | Fit on percent returns, convert back (D14) | `test_loglik_is_scale_equivariant` |
| 32 | `nu` has no meaning in the normal variant but still occupies a slot | Overwritten with NaN before validation | `test_normal_loglik_ignores_nu_entirely` |
| **Test design** | | | |
| 33 | A "degenerate data" test conflated two different failures | Split: unusable input raises, unfittable input reports | Two tests, one each |
| 34 | Forcing a failed refit with `refit_every=1` meant 2,134 fits | Force the failure at the locked cadence instead | Runtime: 18 min to 1 |
| 35 | The suite went from 8 seconds to 10 minutes | `slow` marker + module-scoped run fixture | `pytest -m "not slow"`, 1 min |
| **Presentation** | | | |
| 36 | Stress labels printed through the legend | Labels on the bottom panel, legend on the top | Visual check |

---

## Estimation

### 29. Ranking multi-starts on objective value alone lost 21 days of forecasts

**Problem.** `fit_garch_mle` runs 18 fixed starting points and returns the best. The
first implementation took the best by objective value and reported that start's
convergence verdict. At the 2022-09-06 refit, one start terminated abnormally
(`ABNORMAL:` — L-BFGS-B's line-search failure) at an objective a hair below an ordinary
success, so it won the ranking and the whole refit was reported as failed.

**Why it mattered.** Under decision D16 a failed refit produces no forecasts for its
block, so this blanked 21 days of the headline model — 1% of the evaluation window — on
the strength of a starting point that was never the answer. It was visible only because
`run_all.py` prints the convergence count; the forecast table would otherwise have had a
hole in it that nothing announced until the evaluation layer met a NaN.

The general shape is worth more than the instance: a selection rule and a reporting rule
were entangled. The optimum and the verdict about it were being read off the same object
without asking whether that object was admissible first.

**Solution.** Rank the acceptable starts — converged, admissible, not on the barrier —
and take the best of those; fall back to the best of the rest only if none qualify, in
which case `converged` is False and says so. This is the ordinary multi-start rule and
not a retry loop: the grid is fixed, every start runs exactly once, and no failure is
re-run in the hope of a better verdict. All 102 refits now converge for both variants.

### 30. The obvious optimisation would have changed a third of the estimates

**Problem.** The 18-point grid costs 51 seconds across the backtest; a single start
costs 2. That is a tempting saving, and "the optimiser converges either way" is the
argument that would justify it — all four grid sizes tried reported 102/102 converged.

**Why it mattered.** Convergence says the optimiser stopped at a local optimum, not that
it stopped at the right one. Refitting all 102 windows under reduced grids: with one
start, **93 of 102** land on a different optimum (alpha moving by up to 0.049, nu by up
to 1.6); with six starts, 40 of 102 move. Since the reduced grids are subsets of the
full one, every difference is a *worse* optimum. The likelihood has multiple local maxima
on this data.

Nothing would have failed. The parameter-stability figure would have looked much the
same, the forecasts would have been plausible, and roughly a third of the project's
published estimates would have been wrong.

**Solution.** Keep the grid. The measurement is recorded in the log (§1.9) so the next
person to notice the 51 seconds finds the answer rather than repeating the experiment.

### 31. The natural scale for the data is the wrong scale for the optimiser

**Problem.** Daily SPY log returns have a standard deviation near 0.011, so `omega`
lands around 5e-6 — below L-BFGS-B's default convergence tolerances. The optimiser
terminates on a parameter it has barely moved.

**Why it mattered.** Not a crash and not obviously wrong: it returns estimates, reports
success, and produces forecasts. It simply stops early on the one parameter that sets
the unconditional variance level.

**Solution.** Decision D14: fit on `returns * 100`, convert the estimates back. Contained
entirely inside `fit_garch_mle`, so no other module sees the percent convention, and it
matches what the Stage 3 probe already found about the sampler's geometry. The rescaling
is a change of variable, so the two log-likelihoods must differ by exactly one factor of
100 per observation — asserted rather than assumed.

### 32. `nu` had no meaning in the normal variant but still occupied a slot

**Problem.** Both innovation distributions share `garch11_filter` and the five-element
parameter vector, but the Gaussian model has no degrees-of-freedom parameter. Whatever
sits in `theta[4]` is meaningless — and `is_valid` was still checking `nu > 4` against it.

**Why it mattered.** Two ways to go wrong in opposite directions. A value that passes
validation makes the normal variant's admissibility depend on a number that means
nothing; a value that fails it makes every normal fit report as inadmissible. The first
implementation hit the second: with `nu` bounded at exactly 4.0, `nu > 4` was false and
no normal fit could ever be accepted.

**Solution.** `garch11_normal_loglik` overwrites `nu` with NaN before validating, so the
value passed in genuinely cannot affect the result, and `is_valid` treats NaN as "this
variant has no such parameter". Fits return `nu = NaN` rather than a sentinel: anything
downstream that forgets to branch on the innovation distribution produces a NaN, which is
visible, rather than a plausible number computed under the wrong distribution.

---

## Test design

### 33. One test was asking two different questions

**Problem.** `test_mle_reports_failure_rather_than_inventing_a_fit` fed the optimiser
`np.zeros(200)` and expected `converged=False`. It got a `ValueError` from
`backcast_initial_variance` instead, because a constant series has zero variance and
cannot seed the recursion at all.

**Why it mattered.** The tempting fix is to wrap the backcast so the function returns a
failed fit instead of raising. That would be wrong: "the optimiser did not converge" and
"this input cannot be fitted by any procedure" are different facts, and collapsing them
loses the distinction the `converged` flag exists to carry.

**Solution.** Two tests. Structurally unusable input raises — the same principle as
`EstimationWindow.ROLLING` raising rather than silently doing something nobody chose.
Well-formed input the optimiser cannot handle (a series with variance 1e-24, so the
likelihood surface is flat to machine precision) returns `converged=False` with the
optimiser's message intact.

### 34. The forced-failure test asked for 2,134 refits

**Problem.** The test that forces one refit to fail used `BacktestConfig(refit_every=1)`
to make the target date easy to identify. At the locked cadence that is 102 fits; at a
cadence of 1 it is 2,134, so a test intended to check error handling ran for eighteen
minutes.

**Why it mattered.** More than slowness. A test that runs the loop at a cadence the
project never uses is testing a configuration nobody will ship, and the block structure
it exercises — one date per block — is precisely the structure where the two-cadence bug
cannot appear.

**Solution.** Force the failure at the locked cadence by identifying the target refit
through its estimation-window length, which is unique because the window expands. The
test now also asserts the whole 21-day block is empty and that neighbouring blocks are
untouched, which is what "contained, not contagious" actually means.

### 35. The suite went from eight seconds to ten minutes

**Problem.** A full backtest now costs about a minute. The look-ahead audit runs it once
uncorrupted and once per corruption date, and the corrupted runs are *slower* than clean
ones because random noise makes the optimiser work harder.

**Why it mattered.** A suite nobody runs is a suite that does not protect anything, and
the audit at the centre of it is on the governing plan's never-cut list. The wrong fix
is to shrink the audit — run it on a truncated sample, or drop GARCH from it — which
buys speed by removing the coverage that motivated the test.

**Solution.** Two changes, neither of which reduces coverage. The uncorrupted run became
a module-scoped fixture instead of being recomputed in each test that needed it. The 15
tests that each need their *own* run — the audit's three corrupted re-runs, the vacuity
check beside them, the determinism test — are marked `slow`.

`pytest` takes 10 minutes; `pytest -m "not slow"` takes 1. Worth being precise about why
the second number is a minute and not seconds: the contract tests share that fixture, so
the inner loop still pays for one full backtest. Marking them slow as well would buy the
seconds back by skipping the checks that the forecast table is complete and its intervals
correctly ordered, which is not a trade worth making. Plain `pytest` runs the marked
tests too — a default test run must not be the thing that skips the audit.

---

## Presentation

### 36. The stress labels printed through the legend

**Problem.** `plot_parameter_stability` puts a legend in its top panel and calls
`shade_stress_periods`, which anchors its labels to the top of whichever axes it is
given. On the top panel the two occupy the same space.

**Solution.** Labels on the bottom panel, where the `nu` series sits well below the top
of the axes; legend on the top panel. The third variation on #27 — annotations anchored
to a fixed corner will eventually collide with whatever else lives there, and the fix is
always to move one of them somewhere with known headroom.

---

## Environment

### 37. PyMC's parallel chains hung the machine instead of erroring

**Problem.** Sampling with `cores > 1` from a script with no `if __name__ ==
"__main__":` guard. On Windows, multiprocessing uses spawn rather than fork, so each
worker re-imports the module that started it -- which starts sampling again, which
spawns more workers. The script does not fail; it forks until something gives out. The
first run was killed at a ten-minute timeout having produced no output at all, and the
symptom -- silence -- looks exactly like a slow model.

**Why it mattered.** The visible cost was one wasted timeout. The invisible one was the
conclusion nearly drawn from it: that the full-window fit was intractably slow, which
would have argued for cutting chains or draws, i.e. weakening the convergence diagnostics
on the strength of a bug. Python does print a `RuntimeError` about safe importing of the
main module, but it prints it from the *child* process, so it lands in a log nobody is
tailing while the parent keeps going.

**Solution.** Guard the entry point. `run_all.py` already has one at its `if __name__ ==
"__main__":` line, so the production path was never at risk -- but any throwaway probe,
any notebook cell shelling out, and any test that samples with `cores > 1` is. The real
timings, once guarded: 30.4 s at n=756 and 83.6 s at n=2,890, against 56 s and an unknown
for the same fits run sequentially.

**Generalisation.** A process that produces no output is not evidence of a slow process.
Before concluding that something is expensive, confirm it has started.

---

## Inference

### 38. The sanity check I was handed would have sent me hunting a bug that was not there

**Problem.** The Stage 3 handoff states the check plainly: "Bayesian intervals should be
**≥ frequentist widths**, most visibly early in the sample and at the 99% level. If they
are not, hunt for a bug." Written as a test over 90%, 95% and 99%, that failed
immediately at 90% -- the mixture predictive came out *narrower* than the plug-in,
0.03491 against 0.03508.

**Why it mattered.** The instruction is to hunt for a bug, and the handoff names the bug
to hunt: a single shared `h_next` across draws. There was no such bug -- the same test
passed at 95% and 99%, and the guard tests for that specific failure were already
passing. Following the instruction would have meant changing correct code until an
incorrect assertion passed, and the natural "fix" -- inflating the mixture's spread --
would have manufactured exactly the widening the project set out to measure.

**Solution.** Establish whether the test is wrong before touching the code
(generalisation of #3 and #7, now applied to a claim inherited from a document rather
than one of my own). The mixture quantiles were checked against four million draws taken
from the same mixture by a different route -- sample a component, then sample its
innovation -- and agreed to four significant figures. The solve was right, so the claim
was wrong.

And it is wrong for a reason, not by accident. A scale mixture holding *average* variance
fixed is leptokurtic against the single distribution at that average: more peaked in the
middle, heavier in the tails, because the total variance is conserved and has to come
from somewhere. Measured on the same dispersed posterior:

| Two-sided level | mixture / plug-in width |
|---|---|
| 90% | 0.995 |
| 95% | 1.002 |
| 99% | 1.017 |
| 99.8% | 1.032 |

The crossover sits between 90% and 95%. The handoff's "most visibly at the 99% level" was
pointing at this without saying so; what it got wrong was the "≥" at every level.

**Generalisation.** An inherited sanity check is a hypothesis, not an oracle. This one
came from the project's own earlier self and was still only approximately true -- and the
approximation failed in the direction that would have caused the most damage, since the
repair it invites is indistinguishable from the finding it protects. Two tests now pin
both sides: wider at 95% and 99%, narrower at 90%.

---

### 39. Cutting the draw count did not cut the audit's cost

**Problem.** Decision D20 authorised running the Bayesian track in the look-ahead audit
at a reduced draw count, reasoning that draw count controls Monte Carlo precision rather
than which data reaches a fit, and that the audit's assertions are exact equalities under
a fixed seed at any number of draws. The reasoning holds. The premise -- that draws are
what the audit is paying for -- does not.

Measured, at a 756-observation window: a refit at the frozen settings (4 chains x
(1,000 tune + 1,000 draw)) takes 29s; the same refit at 2 chains x (50 tune + 25 draw)
takes 9s. Cutting the work by 96% cut the wall clock by 69%. At 2,890 observations the
same comparison is 83s against 17s, and the reduced-setting cost *rises with window
length* even though the sampling work is identical.

**Why it mattered.** The floor is compilation, not sampling: PyTensor must build and
compile the gradient of the `scan` recursion for NUTS, the returns are a constant baked
into that graph, so codegen scales with the window. Six backtest runs (one clean, three
corruption dates, the corruption-effect check, the determinism check) x 102 refits x
~12s puts a default `pytest` near two hours. A two-hour default test run is one nobody
runs, and an audit nobody runs has been cut -- by attrition rather than by decision,
which is worse, because the cut list at least gets argued.

Passing the data through a `pytensor.shared` variable to keep it out of the graph was
tried and is worse, not better: 20-29s per refit at the reduced settings, because the
static shape is then unknown and the optimiser gives up. Reverted.

**Solution.** D26: on every `pytest` the Bayesian track is audited with the sampler
replaced by a deterministic stand-in that derives its draws from the estimation window
and **records the exact array and `h0` it was handed**. Every look-ahead surface stays in
the real code path at all 102 refit dates, and the central question -- did any fit see
data from on or after its own refit date -- becomes an assertion on the sampler's input
rather than an inference from output equality, which the real sampler cannot give. Eighty-
four seconds. `pytest -m bayes_audit` then covers the sampler's own internals end to end
at D20's reduced draw count, deselected by default and run deliberately before the
write-up.

That default deselection is the only one the never-cut rule tolerates, and only because
it removes no coverage from a default run.

**Generalisation.** "This knob controls the cost" deserves a measurement before it
becomes a decision. D20 was written on an entirely reasonable model of where the time
goes, and that model was wrong by a factor that changed what the decision was worth.

---

### 40. The gate on the forty-minute test was disarmed by the fast inner loop

**Problem.** D26 put the real-sampler audit behind a `bayes_audit` marker and deselected
it by default with `addopts = -m "not bayes_audit"` in `pytest.ini`. Then `pytest -m "not
slow"` -- the documented fast inner loop, run dozens of times a day -- started the
forty-minute test. It ran for ten minutes before being killed.

**Why it mattered.** A command line's `-m` *replaces* the one in `addopts` rather than
combining with it. So the deselection held for `pytest` and evaporated for every
selection anyone would actually type. Worse, it evaporated silently: the fast suite
simply stopped being fast, and the obvious diagnosis for a suddenly slow test run is that
something in the new code is slow, not that a gate has quietly opened.

**Solution.** An explicit `--bayes-audit` flag in `conftest.py`, which skips the test on
collection unless passed. No `-m` can touch it. The marker stays, so
`pytest --bayes-audit -m bayes_audit` still selects the test alone.

**Generalisation.** A gate that a plausible everyday command disarms is not a gate. When
something must be opt-in, make the opt-in a switch of its own rather than an option that
composes -- and check the gate under the command lines people actually type, not only
under the bare one.

---

### 41. The one-cause comparison had two causes

**Problem.** The project's central claim, in the README and the plan alike: models 3 and
4 share one likelihood, so *any* difference in their intervals is parameter uncertainty,
full stop. The Stage 3 production run made the Bayesian intervals about 1.6% narrower at
99%, and 2.9% narrower on stressed days. Read through that claim, this says parameter
uncertainty *shrinks* intervals in a crisis -- a striking result, and a false one.

**Why it mattered.** The claim is true of the posterior predictive against a plug-in **at
the posterior mean**. It is false of the posterior predictive against a plug-in at the
**MLE**, which is what the forecast table holds, because the two point estimates are not
the same estimate: the MLE maximises the likelihood, and the posterior mean sits wherever
the priors moved it. Decomposed on one COVID-period refit, at the 99% level: parameter
uncertainty widens by 0.5%, and the priors narrow by 4.5%. The reported difference is
nine parts prior to one part parameter uncertainty, and every word this project had
written about it said the opposite.

Nothing in the code was wrong. The claim was a sentence written before there were two
point estimates to compare, and it went on sounding right after that stopped being true.

**Solution.** Recorded at research_log.md 1.13, with the decomposition, and Stage 4 is
now obliged to run it rather than attribute the difference. The cheap fix for the
comparison itself is a fourth GARCH track -- plug-in at the posterior mean -- which needs
no sampling, since the posterior means for all 102 refits are already in
`refit_records.csv`.

**Generalisation.** "The two models differ in exactly one respect" is a claim about a
*comparison*, not about the code, and it decays silently when either side gains a moving
part. The likelihood really is shared; the sentence stopped being true anyway. Check what
a headline claim asserts against what the tables actually contain, each time the tables
change.

---

### 42. The money test did not fire, and the plan's sentence was ready anyway

**Problem.** The governing plan names Christoffersen's independence test the money test,
on the argument that correct *average* coverage can hide breaches that cluster in crises.
The Stage 4 tables came back the other way round. All four models fail Kupiec at the 99%
VaR -- 87, 54, 37 and 37 breaches against 21 expected -- and independence fires for none
of them: p = 0.67 and 0.69 for the GARCH models, 0.76 for `yesterday`, 0.058 for `ewma`.
The breaches are too many and they are not clustered.

**Why it mattered.** Two ways to get this wrong, and the first was already half-written.
The plan's sentence about clustered failure is quotable, and a write-up assembled from
the plan would carry it into a results section where the tables say the opposite. The
second is subtler: reading "independence does not reject" as "breaches are well timed".
It is a failure to reject on hit sequences of 37 events, which is a small sample for a
first-order Markov test; the honest reading is that the test had little power here, not
that the models passed something. Both errors flatter the models, which is the direction
that should always draw suspicion.

**Solution.** The result is written up as what it is -- failure in the *level* of tail
risk, not its *timing* -- in research_log.md's Stage 4 entry, and the non-rejection is
recorded with its power caveat as an obligation in `docs/handoff.md` 7 so it cannot be
quietly upgraded to a pass at the write-up. The test itself is unchanged and stays on the
never-cut list: a diagnostic that does not fire is still worth running, and knowing which
of two failure modes a model has is exactly what it bought.

**Generalisation.** A pre-registered expectation is a hypothesis, not a template for the
results section. When the tables contradict the plan, the finding is the contradiction --
and the sentence in the plan is the thing that has to change, in the write-up, not in the
plan itself, which is the contract and stays as written.

---

### 43. The robustness check had a look-ahead in it by default

**Problem.** The governing plan's regime sensitivity is "terciles of trailing 21-day
Parkinson vol". Written the obvious way -- `trailing.quantile([1/3, 2/3])` on the series
you have -- the cut points are computed over all 2,890 days, which means every day's
regime label is a function of days that had not happened yet. The 2020 and 2022 episodes
would help decide what counts as a stressed day in 2017.

**Why it mattered.** It is the same class of error the whole project exists to detect, and
this one would have been invisible: the labels look sensible, the buckets come out at
roughly a third each, every downstream table runs, and the resulting figure would have
been a *more* flattering version of the real one. The locked VIX thresholds have no such
problem -- 15 and 25 were fixed before any code was written, which is exactly why the plan
locked numbers rather than quantiles -- so the alternative definition had to earn a
property the primary one gets for free.

The plan's own text says the thresholds are "fixed on the warm-up sample". Easy to read
past; the parenthesis is doing real work.

**Solution.** `trailing_vol_thresholds` is a separate function from
`assign_trailing_vol_regime`, reads `:TRAIN_END` and nothing after it, and is pinned on
both sides: multiplying every evaluation-period observation by 100 leaves the cut points
bit-identical, and tripling the warm-up moves them. The labeller takes them as an argument
so a caller can pin them explicitly, never so they can be re-estimated later. Decision
D35.

The visible cost is that the out-of-sample buckets are not equal thirds. The warm-up was
calm, so 1,040 of the 2,134 evaluation days land in the top bucket and only 367 in the
middle -- and the tercile "stressed" bucket is therefore a much weaker notion of stress
than VIX above 25. That is reported rather than corrected, because every available
correction reintroduces the look-ahead.

**Generalisation.** A threshold estimated from data is a parameter, and it inherits every
question asked of a parameter -- including which sample it was allowed to see. Sensitivity
checks are where this slips through, because the check is by definition not the headline
result and gets a fraction of the scrutiny. Give the alternative the same discipline as
the primary, or the robustness check is the least trustworthy number in the report.

---

## Still open

Carried forward deliberately, not overlooked:

- **Residual conditional bias in the scaled proxy** (#9). Wording obligation on the
  report; robustness check owed at Stage 6.
- **Constant mean over the evaluation period** (#10). Limitations section.
- ~~**Priors for the Bayesian GARCH.**~~ **Frozen at Stage 3** in `research_log.md`
  1.10, before any Bayesian out-of-sample number existed. `delta ~ Beta(3, 1)` departs
  from the 1.6 probe's `Beta(10, 2)`, which placed zero density at the stationarity
  boundary the data presses against. What remains owed is the report sentence disclosing
  that full-sample posterior summaries were consulted for magnitude, and the prior
  sensitivity check in `03_robustness`.
- ~~**`BayesianFit` still carries emcee fields.**~~ **Re-keyed at Stage 3** to
  `r_hat`, `ess_bulk`, `ess_tail`, `n_divergences`, plus `converged`/`message`/`n_obs`
  mirroring `FrequentistFit` so D19 can apply D16's rule to both tracks. Nothing outside
  `models.py` referenced the old fields.
- **Persistence near the stationarity boundary.** `alpha + beta` reaches 0.99998 and
  exceeds 0.999 in 16 of the 102 refits (log §1.9). Admissible throughout, but the
  Bayesian model enforces the same constraint by construction, so the posterior will
  press against the same edge. Worth a sentence in the report rather than a discovery
  at Stage 7.
- ~~**Fitted tails slightly fatter than the residuals warrant.**~~ **Answered at Stage 4,
  and the answer is the opposite of the worry.** The warm-up QQ plot put the empirical
  standardised residuals inside the fitted t at both ends, which suggested over-coverage
  at 99%. Out of sample there is *under*-coverage: `garch_mle` covers 0.9897 against a
  nominal 0.99 two-sided, and on the one-sided 99% VaR both GARCH models take 37 breaches
  against 21 expected, rejecting Kupiec at p = 0.002. What the warm-up window suggested
  about the tails did not survive the evaluation window, which is its own small lesson
  about diagnosing a model on the sample it was fitted to.
