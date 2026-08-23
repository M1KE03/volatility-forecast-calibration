# Problems faced, and what was done about them

Companion to `research_log.md` and `README.md`.

- `research_log.md` is the **chronological** record: what was decided, when, and why.
- `README.md` is the **outward** view: what the project is and how to run it.
- This file is the **problem-oriented** view: every difficulty encountered so far, why it
  mattered, and the fix that is now in the repository.

Scope: everything up to the end of **Stage 1** (backtest harness and baselines complete,
2026-08-23). No GARCH model exists yet, so nothing here concerns model estimation.

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

## Still open

Carried forward deliberately, not overlooked:

- **Residual conditional bias in the scaled proxy** (#9). Wording obligation on the
  report; robustness check owed at Stage 6.
- **Constant mean over the evaluation period** (#10). Limitations section.
- **Priors for the Bayesian GARCH.** Must be frozen in `research_log.md` *before* any
  out-of-sample number is computed. Stage 3.
- **The refit machinery has never been exercised.** Both baselines are parameter-free, so
  the 21-day cadence is a no-op so far. Stage 2 is the first real test of it.
