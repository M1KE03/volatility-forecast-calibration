# Research log

Two sections: a **decision register** (what was decided, why, and what is still open)
and a **changelog** (what changed in each stage). Append-only. Nothing here should be
edited to match a later result.

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
