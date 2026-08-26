# Trusting the Error Bars: Calibration of Frequentist vs Bayesian Volatility Forecasts

**Status: Stage 3 complete. All four forecasters exist — both baselines, the frequentist
GARCH(1,1)-t and the Bayesian GARCH(1,1)-t — plus the GARCH-normal ablation, each with a
forecast table over the 2,134-day evaluation window. The Bayesian model is fitted by NUTS
at each of the 102 refit dates, 100 of which converged. The look-ahead audit covers both
estimators. `evaluation.py` and `bootstrap.py` are still stubs and are next.**

Stage numbers follow [`docs/project1-implementation-plan.md`](docs/project1-implementation-plan.md),
the governing plan.

## Research question

For one-day-ahead S&P 500 volatility, are the predictive intervals of a frequentist
GARCH(1,1)-t model and its Bayesian counterpart calibrated at their nominal levels out
of sample, and does that calibration survive high-volatility regimes?

The methodological core is a comparison of a **frequentist plug-in predictive
distribution** against a **Bayesian posterior-predictive distribution** built from the
*same* GARCH likelihood. The two differ only in whether parameter uncertainty is
integrated over. The question is whether doing so improves predictive-interval
calibration.

## Models compared

| # | Model | Key | Role | State |
|---|---|---|---|---|
| 1 | Yesterday's volatility | `yesterday` | Naive baseline | built |
| 2 | EWMA / RiskMetrics | `ewma` | Industry baseline | built |
| 3 | GARCH(1,1)-t, frequentist MLE | `garch_mle` | Plug-in predictive | built |
| 4 | GARCH(1,1)-t, Bayesian | `garch_bayes` | Posterior predictive | built |

Models 3 and 4 share one log-likelihood implementation (`src/models.py`), so that "the
same underlying likelihood" is a property of the code rather than a claim — and since
Stage 3 a property that is checked, because PyMC builds its own graph rather than calling
that implementation, and a test requires the two to agree at a fixed parameter vector.

**What that does and does not license.** The two models differ in the *estimator*, not in
the model. It does **not** follow that the difference between their intervals is
parameter uncertainty, and an earlier version of this README said it did. The frequentist
predictive sits at the maximum of the likelihood; the Bayesian one integrates a posterior
that the priors have moved off that maximum. Measured on a COVID-period refit at the 99%
level, parameter uncertainty widens the interval by 0.5% and the priors narrow it by 4.5%
— nine parts prior to one part parameter uncertainty, pointing opposite ways. The
mechanism, the numbers and what the evaluation layer must do about it are in
`research_log.md` §1.13; the trap is problems-and-solutions #41.

A fifth track, `garch_mle_normal`, is carried through the same backtest. It is the
**Stage 6 ablation** — normal versus Student-t innovations, the cheap and decisive lever
on 99% tail coverage — and not a competitor: `backtest.HEADLINE_MODELS` excludes it and
the evaluation layer filters on that tuple. It is fitted now because doing so costs five
seconds per full run and saves re-entering the walk-forward loop later.

## Locked design decisions

| Item | Value |
|---|---|
| Asset | SPY |
| Frequency | Daily |
| Data source | Yahoo Finance via `yfinance` |
| Main sample | 2014-01-01 to 2025-06-30 |
| Initial training period | 2014-01-02 to 2016-12-30 |
| Out-of-sample period | 2017-01-03 to 2025-06-30 |
| Returns | Daily log returns from adjusted close |
| Point-forecast proxy | Parkinson variance from daily high/low |
| Calibration target | Observed daily returns |
| Forecast horizon | 1 trading day |
| Refit cadence | Every 21 trading days |
| Intervals evaluated | Two-sided 90%, 95%, 99%; one-sided 99% VaR |
| Primary point loss | QLIKE |
| Secondary point loss | MSE on the variance scale |
| Primary comparison test | Diebold-Mariano, with caveats |
| Uncertainty quantification | Stationary block bootstrap, mean block length 20, 1,000 replications, fixed seed |
| Regimes | Lagged VIX close: calm < 15, normal 15-25, stressed > 25 |

Explicitly **out of scope** in the core version: LSTM, Transformer, stochastic
volatility, macro variables, multiple assets, dashboards.

The plan of work is [`docs/project1-implementation-plan.md`](docs/project1-implementation-plan.md):
eight stages, a ~15-20h budget, and an ordered cut list. The full decision register,
including decisions made during scaffolding and the items still open, is in
[`research_log.md`](research_log.md); §1.5 there records the 2026-08-23 switch to this
plan and the retirement of the earlier one, which was deleted at the switch and
survives only in git history (see `git log -- docs/implementation_plan.md`).

For a problem-oriented view of the same history — every difficulty encountered, why it
mattered, and the fix now in the repository — see
[`docs/problems-and-solutions.md`](docs/problems-and-solutions.md). It covers the
look-ahead hazards, the proxy scale mismatch, the environment and toolchain work, and a
set of test-integrity lessons that generalise beyond this project.

**Picking this up cold?** [`docs/handoff.md`](docs/handoff.md) is the entry point: what
exists, what is next, which rules are not revisable, and the specific traps that do damage
while looking fine.

## Quantities that must not be conflated

This project involves several distinct objects that are easy to blur together. They
are kept distinct in code and in the report:

- **Observed returns** -- realised daily log returns. The target for interval
  calibration and VaR.
- **Parkinson volatility proxy** -- a high/low range estimator of *intraday* variance.
  The target for point-forecast loss only. It is **not** the same quantity the models
  forecast (see the caveat below).
- **Conditional variance forecast** -- the model's one-day-ahead variance.
- **Predictive interval** -- a two-sided interval for the *return*, not for the variance.
- **VaR forecast** -- a one-sided lower quantile of the return distribution.
- **Parameter uncertainty** -- uncertainty about GARCH coefficients. Present in the
  Bayesian posterior predictive; absent from the frequentist plug-in. **Not** the same
  thing as the difference between the two models' intervals, which also contains the
  priors' effect on the point estimate (see above).
- **Innovation uncertainty** -- uncertainty from the Student-t shock. Present in both.

### Known caveat: Parkinson is not on the forecast target's scale

Parkinson variance estimates the variance of the *intraday continuous* price path. The
GARCH models forecast the variance of the *close-to-close* return, which also contains
the overnight gap. Parkinson is therefore biased low relative to the forecast target,
and QLIKE's proxy-robustness property assumes a conditionally unbiased proxy. This
affects the **point-forecast loss only**; the interval-calibration and VaR results are
evaluated against observed returns and are unaffected. The governing plan handles this
by reporting the squared-return proxy as a robustness check at Stage 6, and the caveat
belongs in the results section of the report, not only the methods section.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Requires Python 3.12 **and a C/C++ compiler on `PATH`** — PyMC's PyTensor backend needs
one. `pip install -r requirements.txt` does not supply it, so it is the one part of the
environment that is not reproducible from that file alone.

This machine uses **MinGW-w64 GCC 16.2.0** (UCRT, x86_64) from the
[WinLibs](https://winlibs.com/) release `16.2.0posix-14.0.0-ucrt-r1`, unpacked to
`C:\Users\micha\toolchains\mingw64` and added to the user `PATH`. It is portable: no
administrator rights are needed, and uninstalling means deleting that directory. Verify
with:

```bash
g++ --version                 # expect 16.2.0
python -c "import pytensor; print(pytensor.config.cxx)"
```

MSVC Build Tools works equally well if you already have it. Provenance, the verified
SHA-256, and why this replaced the earlier compiler-free design are in `research_log.md`
§1.6 (decision B1-R).

## Usage

```bash
python run_all.py --help            # list pipeline stages
python run_all.py --stage data      # build the analysis frame (implemented)
python run_all.py --stage eda       # ARCH-LM, Ljung-Box, ADF + Stage 0 figures (implemented)
python run_all.py --stage backtest  # walk-forward loop, baselines + MLE GARCH (~1 min)
python run_all.py --stage bayes     # the Bayesian track, 102 NUTS fits (~95 min)
python run_all.py --all             # run the full pipeline
```

**The backtest is two stages, for cost rather than design.** `--stage backtest` refits
GARCH at each of the 102 refit dates for both innovation distributions -- 306
maximum-likelihood fits, about a minute. `--stage bayes` runs the same walk-forward loop
over the same refit dates with NUTS, which takes about ninety-five. Each writes its own
partial tables and rebuilds the merged `forecasts.csv` from whichever partials are on
disk, so re-running the cheap track never re-runs the expensive one and the evaluation
layer still reads one table. The merge compares the two configurations and refuses to
join runs made under different ones.

A Bayesian refit that fails its convergence diagnostics produces no forecasts for the 21
days it serves, and those days stay NaN rather than being filled from the previous
window. Two of the 102 did, so `garch_bayes` has 2,092 of the 2,134 rows.

`--stage data` uses the committed snapshot in `data/raw/` and makes no network call;
pass `--refresh` to re-download, which deliberately replaces that snapshot. It prints the
full data quality report and writes `data/processed/analysis_frame.csv`.

The `evaluate` and `figures` stages still raise `NotImplementedError`.

## The scale convention

Every variance forecast in this project and the evaluation proxy are on the
**close-to-close return-variance scale**. This is what makes the four models comparable
to each other and their intervals comparable to observed returns.

The Parkinson estimator is built from the intraday high/low range and so sees no part of
the overnight move, which for SPY is a large share of daily variance. It understates
close-to-close variance by roughly a third. The correction is a single constant estimated
on the warm-up sample only and frozen:

```
c = mean(r^2) / mean(sigma^2_P) = 1.517318      (2014-01-02 .. 2016-12-30, n = 756)
```

Without it the error would pull in two directions at once: the RW baseline would be
unfairly advantaged on QLIKE (its units match the raw proxy) while its intervals came out
~23% too narrow, so it would win the point-forecast table and fail the calibration table
for reasons having nothing to do with forecasting. Separately, QLIKE's proxy-robustness
(Patton 2011) presumes an unbiased proxy.

**What `c` does not do.** It removes the systematic level error but does *not* make the
proxy conditionally unbiased -- within the warm-up alone the quarterly ratio ranges from
1.18 to 1.94, and the overnight share plausibly co-moves with regime. The proxy is
approximately unbiased *on average*; proxy-robustness is approached, not restored. Stage
6 re-runs the QLIKE ranking on raw Parkinson to show the ranking does not hinge on `c`.

## The look-ahead audit

The single test the whole project's credibility rests on. It runs the backtest, corrupts
every observation from a date `t` onward, re-runs, and asserts the forecasts are
bit-identical before `t` -- at Volmageddon, the largest COVID drawdown day, and a 2022
selloff.

A precise distinction is built into it. Corrupting from `t` must leave the *forecast*
columns unchanged up to and including `t`, because the forecast for `t` is built from
data through `t-1`. It must **not** leave the *evaluation* columns at `t` unchanged --
`log_return` and its `pit` are functions of day `t` itself. Asserting the stronger claim
would be asserting something false, and would have to be weakened later, which is how a
look-ahead test quietly stops testing anything.

`test_corruption_actually_changes_the_future` sits beside it, because everything above
would also hold for a backtest that ignored its input entirely.

Since Stage 2 the audit covers the GARCH models, which is the first time it has had
anything to say. Both baselines are parameter-free, so until there was a model that
estimated something the audit could not have caught a refit trained on data it should
not have seen. Every route by which the future could reach a forecast is now inside its
scope: the estimation window, the backcast seed, and the daily filter.

Since Stage 3 the Bayesian track is audited too, under the same corruption dates and the
same claim, but against a **recording stub** in place of NUTS. That is not a concession.
A Bayesian refit costs 9-17 seconds before it draws anything, because the floor is
compiling the gradient of the variance recursion, so a real-sampler audit would put a
default `pytest` near two hours -- and an audit that slow gets skipped, which is the cut
the never-cut list exists to prevent. Every look-ahead surface still runs in the real code
path at all 102 refit dates, and the stub additionally records the exact array each fit
was handed, so "no fit saw data from on or after its own refit date" becomes an assertion
on the sampler's input rather than an inference from its output. The real sampler is
audited end to end behind an opt-in flag.

```bash
pytest -q                              # everything, including both audits (~18 min)
pytest -m "not slow" -q                # inner loop (~2 min)
pytest --bayes-audit -m bayes_audit    # the audit with NUTS itself (~40 min)
```

The `slow` marker covers the 15 tests that each need their *own* backtest run: the
audit's three corrupted re-runs, the vacuity check beside them, and the determinism
test. Everything else shares one module-scoped run, which is why the inner loop still
costs a minute rather than seconds. Plain `pytest` runs the marked tests too — the audit
is on the governing plan's never-cut list, and a default test run must not be the thing
that skips it.

### What the EDA establishes

Diagnostics are computed on the **training window only** (2014-01-02 to 2016-12-30).
Justifying the model class with a statistic computed over the out-of-sample period would
let the evaluation window argue for the model later evaluated on it -- a mild look-ahead,
but the exact species this project exists to detect. Full-sample values are printed as
labelled descriptive context and justify nothing.

| Diagnostic | Training-window result |
|---|---|
| ARCH-LM (Engle), lags 5 / 10 / 22 | p = 1.5e-23 / 1.8e-21 / 1.0e-17 — rejects at every lag |
| Ljung-Box, **squared** returns | p = 3.4e-44 / 1.1e-51 / 1.8e-47 — rejects |
| Ljung-Box, **raw** returns | p = 0.45 / 0.62 / 0.26 — no rejection |
| ADF on returns | -27.39, p < 1e-300 — stationary |
| Excess kurtosis | 2.46 |

The contrast between the two Ljung-Box rows is the point, not the first row alone:
abundant structure in the second moment, none detectable in the first. That is the regime
in which a conditional-variance model earns its place. The excess kurtosis is the
independent empirical warrant for the Student-t innovation.

A known limitation, recorded up front: over the **full** sample the raw-return Ljung-Box
does reject (p = 7.1e-14), as short-horizon autocorrelation rises in crises. The constant
mean is applied identically to all four forecasters so it cannot bias the comparison, but
it is a real simplification over the evaluation period and is carried into the report's
limitations section.

### What the GARCH fit establishes

Two cadences run in the backtest and conflating them is the subtle bug the harness was
built to prevent. Parameters are re-estimated every 21 trading days on the expanding
window; *between* refits they are held fixed while the variance recursion still advances
daily with each newly observed return. Freezing the variance between refits as well
would produce a complete, plausible forecast table built on variances up to 21 days
stale — which is not a one-day-ahead GARCH forecast at all.

On the warm-up window (756 observations, training only):

| Quantity | Value |
|---|---|
| mu, omega | 0.000704, 4.81e-6 |
| alpha, beta, alpha + beta | 0.2150, 0.7353, 0.9503 |
| nu (Student-t d.o.f.) | 5.86 |
| Ljung-Box, standardised **residuals**, lags 5 / 10 / 22 | p = 0.50 / 0.57 / 0.29 — no rejection |
| Ljung-Box, **squared** standardised residuals | p = 0.62 / 0.88 / 0.98 — no rejection |
| Refits converged | 102 / 102 for both innovation distributions |

The squared-residual row is the one that matters. The same test on raw squared returns
rejects at p = 3.4e-44; after the GARCH filter there is nothing left to reject, so the
variance equation has absorbed the clustering it was fitted to absorb. The residual row
says the constant mean is adequate in sample, which is consistent with the raw-return
Ljung-Box above and does not disturb the limitation recorded there about the evaluation
period.

An independent check on the likelihood: the Stage 3 feasibility probe fitted this same
window with PyMC under priors and obtained alpha 0.219, beta 0.713, nu 6.4. A
hand-written NumPy likelihood maximised by L-BFGS-B and a `pytensor.scan` graph sampled
by NUTS share no code, so their agreement is better evidence than either number alone.

`figures/07_garch_parameter_stability.png` tracks the estimates across all 102 refits.
Persistence rises through the sample and exceeds 0.999 in 16 of them, peaking at
0.99998 — every fit admissible and converged, but very nearly at the stationarity
boundary. That is ordinary for a long daily equity sample containing 2020 and 2022, and
it is recorded because the Bayesian model at Stage 3 enforces the same constraint by
construction and will press against the same edge.

### The analysis frame

2,890 trading days, 2014-01-02 to 2025-06-30: 756 training rows (2014-01-02 to
2016-12-30) and 2,134 out-of-sample rows (2017-01-03 to 2025-06-30), implying 102 refits
at the locked 21-day cadence. Regime counts on the lagged VIX close are calm 1,198,
normal 1,317, stressed 375 — the stressed regime is 13% of the sample, which is thin for
the 99% level and for the 99% VaR and is why every regime table reports `n`.

Raw data is downloaded from 2013-12-01, one month before the sample start, so that the
lagged quantities are defined on the first in-sample row. The buffer feeds the lag only
and never reaches the analysis frame (decision D8 in `research_log.md`).

## Repository layout

```
.
├── README.md
├── requirements.txt
├── conftest.py            # the --bayes-audit opt-in gate
├── pytest.ini             # registers the `slow` and `bayes_audit` markers
├── run_all.py             # single entry point
├── research_log.md        # decision register + changelog
├── docs/
│   ├── handoff.md                        # START HERE: state, next actions, traps
│   ├── project1-implementation-plan.md   # governing plan: stages, budget, cut list
│   └── problems-and-solutions.md         # every difficulty hit so far, and its fix
├── data/
│   ├── raw/               # committed price snapshots + SHA-256 manifest
│   └── processed/         # derived, gitignored, regenerable
├── src/
│   ├── data.py            # download, caching, returns, proxy, regimes
│   ├── eda.py             # ARCH-LM, Ljung-Box, ADF, ACF
│   ├── models.py          # shared likelihood, MLE, priors, NUTS, predictives
│   ├── backtest.py        # walk-forward loop, refit schedule, two tracks, merge
│   ├── figures.py         # house style, all figures
│   ├── evaluation.py      # losses, coverage tests, DM        (stubs)
│   └── bootstrap.py       # stationary block bootstrap        (stubs)
├── notebooks/             # thin presentation layer only
├── tests/
├── figures/
└── report/report.md
```

## Reproducibility

- Dependencies are pinned in `requirements.txt`.
- Raw price data is committed as CSV with a SHA-256 manifest, because Yahoo Finance
  revises history and a download script alone is not reproducible.
- All stochastic components take an explicit seed.
- `run_all.py` is the only supported entry point. Notebooks read cached artefacts and
  contain no reusable logic.
